"""Audit completed transcripts with Gemini 3.7 Flash and selectively reprocess them."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from downloader import DEFAULT_FEED_URL, Episode, fetch_episodes
from pipeline import PodcastPipeline

load_dotenv()

logger = logging.getLogger("prosody_audit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).parent
AUDIT_STATE_PATH = ROOT / "prosody_audit.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.75
def _fingerprint(audio_path: Path, transcript_path: Path) -> str:
    """Identify the exact local audio/transcript pair audited based on contents and sizes."""
    md_hash = hashlib.sha256(transcript_path.read_bytes()).hexdigest()
    audio_size = audio_path.stat().st_size if audio_path.exists() else 0
    return f"{audio_size}:{md_hash}"


def _save_state(state: dict[str, Any]) -> None:
    """Persist audit progress atomically after each episode."""
    tmp_path = AUDIT_STATE_PATH.with_suffix(".tmp.json")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(AUDIT_STATE_PATH)


def _load_state() -> dict[str, Any]:
    if not AUDIT_STATE_PATH.exists():
        return {"version": 1, "episodes": {}}
    try:
        state = json.loads(AUDIT_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(state, dict) and isinstance(state.get("episodes"), dict):
            return state
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable audit state: %s", exc)
    return {"version": 1, "episodes": {}}


def _eligible_episodes(episodes: list[Episode], pipeline: PodcastPipeline) -> list[tuple[Episode, Path, Path]]:
    eligible = []
    for episode in episodes:
        md_path = pipeline.processed_dir / episode.md_filename
        audio_path = pipeline.audio_dir / episode.audio_filename
        record = pipeline.manifest.get(episode.guid, {})
        if (
            record.get("status") in {"SUCCESS", "PARTIAL", "FAILED"}
            and md_path.exists()
            and md_path.stat().st_size > 200
            and audio_path.exists()
            and audio_path.stat().st_size > 1024
        ):
            eligible.append((episode, audio_path, md_path))
    return eligible


def audit_and_reprocess(
    feed_url: str = DEFAULT_FEED_URL,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit eligible completed episodes, then reprocess only high-confidence failures."""
    episodes = fetch_episodes(feed_url)
    pipeline = PodcastPipeline(model_name="gemini-3.7-flash")
    # Clients are initialized across Vertex AI and all configured API keys
    eligible = _eligible_episodes(episodes, pipeline)
    if limit > 0:
        eligible = eligible[:limit]

    logger.info("Found %d completed episodes with both audio and transcript artifacts.", len(eligible))
    state = _load_state()
    audit_results: list[dict[str, Any]] = []

    for episode, audio_path, md_path in eligible:
        fingerprint = _fingerprint(audio_path, md_path)
        previous = state["episodes"].get(episode.guid, {})
        if previous.get("fingerprint") == fingerprint and previous.get("audit_status") in {"AUDITED", "REPROCESSED"}:
            logger.info("Audit already complete: %s", md_path.name)
            audit_results.append(previous)
            continue

        logger.info("Auditing with Gemini 3.7 Flash: %s", md_path.name)
        result: dict[str, Any] = {
            "guid": episode.guid,
            "index": episode.index,
            "date_iso": episode.date_iso,
            "title": episode.title,
            "audio_file": audio_path.name,
            "transcript_file": md_path.name,
            "fingerprint": fingerprint,
            "audit_status": "ERROR",
            "audited_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            audit = pipeline.transcriber.audit_transcript(
                audio_path=audio_path,
                existing_transcript=md_path.read_text(encoding="utf-8"),
            )
            model_needs_reprocess = bool(audit.get("needs_reprocess", False))
            confidence = float(audit.get("confidence", 0.0))
            selected = model_needs_reprocess and confidence >= confidence_threshold
            result.update(
                {
                    "audit_status": "AUDITED",
                    "model_used": audit.get("model_used"),
                    "needs_reprocess": model_needs_reprocess,
                    "confidence": confidence,
                    "selected_for_reprocess": selected,
                    "score": audit.get("score"),
                    "issues": audit.get("issues", []),
                    "reason": audit.get("reason", ""),
                    "recommendation": audit.get("recommendation", "keep"),
                }
            )
            logger.info(
                "Audit decision for %s: %s (confidence %.2f, model %s)",
                episode.title,
                "REPROCESS" if selected else "KEEP",
                confidence,
                result["model_used"] or "unknown",
            )
        except Exception as exc:
            result["error"] = str(exc)
            logger.error("Audit failed for %s: %s", episode.title, exc)

        state["episodes"][episode.guid] = result
        _save_state(state)
        audit_results.append(result)

    selected = [r for r in audit_results if r.get("audit_status") == "AUDITED" and r.get("selected_for_reprocess")]
    reprocessed: list[dict[str, Any]] = []
    by_guid = {episode.guid: episode for episode, _, _ in eligible}
    for result in selected:
        episode = by_guid[result["guid"]]
        ep_audio = pipeline.audio_dir / episode.audio_filename
        ep_md = pipeline.processed_dir / episode.md_filename
        logger.info("Reprocessing transcript with Gemini 3.7 Flash: %s", episode.title)
        try:
            record = pipeline.process_episode(episode, reprocess_transcript=True)
            result["reprocess_status"] = record.status
            result["reprocessed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            result["reprocess_model"] = pipeline.transcriber.last_model_used
            if record.status in {"SUCCESS", "PARTIAL"} and ep_audio.exists() and ep_md.exists():
                result["fingerprint"] = _fingerprint(ep_audio, ep_md)
                result["audit_status"] = "REPROCESSED"
                result["selected_for_reprocess"] = False
                result["needs_reprocess"] = False
            logger.info("Reprocessed %s: %s", episode.title, record.status)
        except Exception as exc:
            result["reprocess_status"] = "ERROR"
            result["reprocess_error"] = str(exc)
            logger.error("Reprocessing failed for %s: %s", episode.title, exc)
        state["episodes"][episode.guid] = result
        _save_state(state)
        reprocessed.append(result)

    return audit_results, reprocessed


def print_summary(audits: list[dict[str, Any]], reprocessed: list[dict[str, Any]]) -> None:
    audited = [r for r in audits if r.get("audit_status") == "AUDITED"]
    errors = [r for r in audits if r.get("audit_status") != "AUDITED"]
    selected = [r for r in audited if r.get("selected_for_reprocess")]
    print("\n" + "=" * 110)
    print("GEMINI 3.7 FLASH PROSODY AUDIT SUMMARY")
    print("=" * 110)
    print(f"Audited: {len(audited)} | Audit errors: {len(errors)} | Selected for reprocessing: {len(selected)} | Reprocessed: {len(reprocessed)}")
    print("-" * 110)
    for result in audits:
        decision = "REPROCESSED" if result.get("reprocess_status") else ("REPROCESS" if result.get("selected_for_reprocess") else "KEEP")
        print(
            f"{result.get('date_iso', 'UNKNOWN'):10} | {decision:11} | "
            f"{result.get('model_used', 'n/a'):24} | {result.get('title', '')}"
        )
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and selectively reprocess existing prosody transcripts")
    parser.add_argument("--feed", default=DEFAULT_FEED_URL)
    parser.add_argument("--limit", type=int, default=0, help="Limit eligible episodes (0 = all)")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Minimum Gemini confidence required to reprocess (default: 0.75)",
    )
    args = parser.parse_args()

    audits, reprocessed = audit_and_reprocess(
        feed_url=args.feed,
        confidence_threshold=args.confidence_threshold,
        limit=args.limit,
    )
    print_summary(audits, reprocessed)


if __name__ == "__main__":
    main()
