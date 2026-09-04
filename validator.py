"""Automated QA, validation, and guardrails pipeline for large-scale podcast processing."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import static_ffmpeg

load_dotenv()
logger = logging.getLogger("validator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

static_ffmpeg.add_paths()

DEFAULT_AUDIO_DIR = Path(__file__).parent / "RawAudio"
DEFAULT_PROCESSED_DIR = Path(__file__).parent / "ProcessedMD"
DEFAULT_TRIMMED_DIR = Path(__file__).parent / "TrimmedAudio"
DEFAULT_LOCAL_DRIVE = Path(
    os.getenv(
        "GOOGLE_DRIVE_LOCAL_PATH",
        "/Users/stephenfinney/Library/CloudStorage/GoogleDrive-ssfinney92@gmail.com/My Drive/Christ Chapel Podcasts",
    )
)
MANIFEST_PATH = Path(__file__).parent / "manifest.json"
REPORT_PATH = Path(__file__).parent / "qa_report.json"


@dataclass
class CheckResult:
    check_name: str
    passed: bool
    severity: str  # "ERROR", "WARNING", "INFO"
    message: str


@dataclass
class EpisodeQAReport:
    filename: str
    date_iso: str
    title: str
    status: str  # "PASS", "WARN", "FAIL"
    raw_audio_size_mb: float = 0.0
    raw_duration_s: float = 0.0
    trimmed_audio_size_mb: float = 0.0
    trimmed_duration_s: float = 0.0
    sermon_ratio: float = 0.0
    word_count: int = 0
    prosody_stress_count: int = 0
    prosody_tag_count: int = 0
    checks: List[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def get_audio_info(audio_path: Path) -> Tuple[float, int]:
    """Return (duration_seconds, bitrate_bps) via ffprobe."""
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        return 0.0, 0
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return 0.0, 0
    try:
        cmd = [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(res.stdout).get("format", {})
        dur = float(data.get("duration", 0.0))
        bitrate = int(data.get("bit_rate", 0))
        return dur, bitrate
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", audio_path, exc)
        return 0.0, 0

def _timestamp_seconds(value: str) -> float:
    try:
        parts = [float(part) for part in value.split(":")]
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0.0



def detect_repetition_loops(text: str, max_consecutive: int = 8) -> Optional[Tuple[str, int]]:
    """Return the first repeated phrase and count.

    Sentence-frequency detection catches long model loops whose cycle exceeds
    the token n-gram window; the token pass still catches short consecutive
    loops while allowing ordinary liturgical repetition to remain a warning.
    """
    sentences = [
        " ".join(re.findall(r"\b\w+\b", sentence.lower()))
        for sentence in re.split(r"(?:[.!?]\s+|\n+)", text)
    ]
    sentence_counts = Counter(sentence for sentence in sentences if len(sentence.split()) >= 4)
    if sentence_counts:
        sentence, repeats = sentence_counts.most_common(1)[0]
        if repeats >= max_consecutive:
            return sentence, repeats

    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        words = re.findall(r"\b\w+\b", paragraph.lower())
        if len(words) < 15:
            continue
        for size in (3, 4, 5, 6):
            for index in range(len(words) - size * max_consecutive + 1):
                pattern = words[index : index + size]
                repeats = 1
                cursor = index + size
                while cursor + size <= len(words) and words[cursor : cursor + size] == pattern:
                    repeats += 1
                    cursor += size
                if repeats >= max_consecutive:
                    return " ".join(pattern), repeats
    return None


class PodcastValidator:
    """Evaluates audio, transcripts, trimmed sermons, and Drive sync status against QA guardrails."""

    def __init__(
        self,
        audio_dir: Path = DEFAULT_AUDIO_DIR,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        trimmed_dir: Path = DEFAULT_TRIMMED_DIR,
        local_drive_path: Path = DEFAULT_LOCAL_DRIVE,
    ):
        self.audio_dir = audio_dir
        self.processed_dir = processed_dir
        self.trimmed_dir = trimmed_dir
        self.local_drive = local_drive_path
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        self.manifest_by_md = {
            record.get("md_file"): record
            for record in manifest.values()
            if record.get("md_file")
        }

    def validate_episode(self, md_path: Path) -> EpisodeQAReport:
        """Run all guardrail checks for a given episode."""
        stem = md_path.stem
        raw_audio_path = self.audio_dir / f"{stem}.mp3"
        trimmed_audio_path = self.trimmed_dir / f"{stem} - Preaching.mp3"

        checks: List[CheckResult] = []

        # Read Markdown
        transcript_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
        title_match = re.search(r'title:\s*"([^"]+)"', transcript_text)
        date_match = re.search(r'date:\s*"([^"]+)"', transcript_text)
        title = title_match.group(1) if title_match else stem
        date_iso = date_match.group(1) if date_match else "UNKNOWN"

        # 1. Raw Audio Checks
        raw_dur, raw_bitrate = get_audio_info(raw_audio_path)
        raw_size_mb = raw_audio_path.stat().st_size / (1024 * 1024) if raw_audio_path.exists() else 0.0
        sermon_ratio = 0.0

        is_ci = os.getenv("CI", "false").lower() in ["true", "1", "yes"]

        if not raw_audio_path.exists():
            severity = "INFO" if is_ci else "ERROR"
            msg = "Raw audio MP3 not present (CI/remote environment)." if is_ci else "Raw audio MP3 is missing."
            checks.append(CheckResult("RAW_AUDIO_EXISTS", is_ci, severity, msg))
        elif raw_size_mb < 2.0:
            checks.append(CheckResult("RAW_AUDIO_SIZE", False, "ERROR", f"Raw audio is suspiciously small: {raw_size_mb:.2f} MB."))
        else:
            checks.append(CheckResult("RAW_AUDIO_SIZE", True, "INFO", f"Raw audio size: {raw_size_mb:.2f} MB."))

        if raw_dur > 0 and raw_dur < 600:  # < 10 mins
            checks.append(CheckResult("RAW_AUDIO_DURATION", False, "WARNING", f"Service duration is unusually short: {raw_dur/60:.1f}m."))
        elif raw_dur >= 600:
            checks.append(CheckResult("RAW_AUDIO_DURATION", True, "INFO", f"Service duration: {raw_dur/60:.1f}m."))

        # 2. Trimmed Preaching Audio Checks
        trimmed_dur, trimmed_bitrate = get_audio_info(trimmed_audio_path)
        trimmed_size_mb = trimmed_audio_path.stat().st_size / (1024 * 1024) if trimmed_audio_path.exists() else 0.0

        if not trimmed_audio_path.exists():
            severity = "INFO" if is_ci else "ERROR"
            msg = "Trimmed audio MP3 not present (CI/remote environment)." if is_ci else "Trimmed preaching audio is missing."
            checks.append(CheckResult("TRIMMED_AUDIO_EXISTS", is_ci, severity, msg))
        else:
            checks.append(CheckResult("TRIMMED_AUDIO_EXISTS", True, "INFO", f"Trimmed audio exists ({trimmed_size_mb:.2f} MB)."))
            # Duration sanity (sermon should normally be 15m - 80m)
            if trimmed_dur < 600:  # < 10 mins
                checks.append(CheckResult("PREACHING_DURATION", False, "WARNING", f"Sermon length is very short: {trimmed_dur/60:.1f}m."))
            elif trimmed_dur > 5400:  # > 90 mins
                checks.append(CheckResult("PREACHING_DURATION", False, "WARNING", f"Sermon length is very long: {trimmed_dur/60:.1f}m."))
            else:
                checks.append(CheckResult("PREACHING_DURATION", True, "INFO", f"Sermon length: {trimmed_dur/60:.1f}m."))

            # Sermon ratio (sermon should be 20% to 90% of the full service).
            if raw_dur > 0:
                sermon_ratio = trimmed_dur / raw_dur
                if sermon_ratio < 0.20 or sermon_ratio > 0.90:
                    checks.append(CheckResult("SERMON_RATIO", False, "ERROR", f"Sermon takes {sermon_ratio*100:.1f}% of total service."))
                else:
                    checks.append(CheckResult("SERMON_RATIO", True, "INFO", f"Sermon ratio: {sermon_ratio*100:.1f}% of service."))
            else:
                sermon_ratio = 0.0
                checks.append(CheckResult("SERMON_RATIO", False, "WARNING", "Sermon ratio unavailable because raw audio duration could not be measured."))

            manifest_record = self.manifest_by_md.get(md_path.name)
            if manifest_record:
                declared_start = _timestamp_seconds(manifest_record.get("preaching_start"))
                declared_end = _timestamp_seconds(manifest_record.get("preaching_end"))
                if declared_start < 60:
                    checks.append(CheckResult("PREACHING_START", False, "ERROR", f"Preaching starts implausibly early at {manifest_record.get('preaching_start')}."))
                else:
                    checks.append(CheckResult("PREACHING_START", True, "INFO", f"Preaching starts at {manifest_record.get('preaching_start')}."))
                declared_duration = declared_end - declared_start
                if declared_duration > 0 and abs(declared_duration - trimmed_dur) > 5:
                    checks.append(CheckResult("MANIFEST_AUDIO_DURATION", False, "ERROR", f"Manifest boundary duration ({declared_duration:.1f}s) differs from trimmed audio ({trimmed_dur:.1f}s)."))
                elif declared_duration > 0:
                    checks.append(CheckResult("MANIFEST_AUDIO_DURATION", True, "INFO", "Manifest boundaries match trimmed audio duration."))

        # 3. Transcript Quality & Prosody Checks
        words = re.findall(r"\b\w+\b", transcript_text)
        word_count = len(words)

        if not md_path.exists() or len(transcript_text) < 500:
            checks.append(CheckResult("TRANSCRIPT_EXISTS", False, "ERROR", "Transcript Markdown is missing or empty."))
        else:
            checks.append(CheckResult("TRANSCRIPT_EXISTS", True, "INFO", f"Transcript size: {len(transcript_text)} chars ({word_count} words)."))

        if word_count > 50000:
            checks.append(CheckResult("TRANSCRIPT_WORD_COUNT", False, "ERROR", f"Transcript is implausibly long: {word_count} words."))
        elif word_count > 30000:
            checks.append(CheckResult("TRANSCRIPT_WORD_COUNT", False, "WARNING", f"Transcript is unusually long: {word_count} words."))
        else:
            checks.append(CheckResult("TRANSCRIPT_WORD_COUNT", True, "INFO", f"Transcript word count is plausible: {word_count}."))

        # YAML Frontmatter
        if "---" in transcript_text and "title:" in transcript_text and "author:" in transcript_text:
            checks.append(CheckResult("FRONTMATTER_VALID", True, "INFO", "Valid YAML frontmatter present."))
        else:
            checks.append(CheckResult("FRONTMATTER_VALID", False, "WARNING", "Missing standard YAML frontmatter fields."))

        # Vocal stress count (bold words: **word**)
        bold_stresses = re.findall(r"\*\*[^*]+\*\*", transcript_text)
        prosody_stress_count = len(bold_stresses)
        if prosody_stress_count < 10:
            checks.append(CheckResult("PROSODY_STRESS", False, "WARNING", f"Low vocal stress bolding: only {prosody_stress_count} bold tags found."))
        else:
            checks.append(CheckResult("PROSODY_STRESS", True, "INFO", f"Found {prosody_stress_count} vocal stress emphasis tags."))

        # Pitch & Cadence tags ([pause: ...], [rising pitch], [whispering], etc.)
        inline_tags = re.findall(r"\[(pause:[^\]]+|rising pitch[^\]]*|pitch drop[^\]]*|whispering|laughing|crying|applause|cheering|music)\]", transcript_text, re.IGNORECASE)
        prosody_tag_count = len(inline_tags)
        if prosody_tag_count < 5:
            checks.append(CheckResult("PROSODY_TAGS", False, "WARNING", f"Low prosody inflection tags: {prosody_tag_count} found."))
        else:
            checks.append(CheckResult("PROSODY_TAGS", True, "INFO", f"Found {prosody_tag_count} inline prosody/cadence tags."))

        # Speaker headings with timestamps
        speaker_headings = re.findall(r"###\s+.*?\s+\[(\d{1,2}:\d{2}(?::\d{2})?)\]", transcript_text)
        if not speaker_headings:
            checks.append(CheckResult("SPEAKER_TIMESTAMPS", False, "WARNING", "No speaker timestamp headings found."))
        else:
            checks.append(CheckResult("SPEAKER_TIMESTAMPS", True, "INFO", f"Found {len(speaker_headings)} speaker timestamp headings."))

        # Repetition loop detection. Small refrain-like loops are warnings;
        # runaway model loops are unusable artifacts and must fail QA.
        rep_loop = detect_repetition_loops(transcript_text)
        if rep_loop:
            phrase, repeat_count = rep_loop
            severity = "ERROR" if repeat_count >= 50 else "WARNING"
            checks.append(CheckResult("NO_REPETITION_LOOPS", False, severity, f"Hallucination loop: Consecutive phrase '{phrase}' repeated {repeat_count} times."))
        else:
            checks.append(CheckResult("NO_REPETITION_LOOPS", True, "INFO", "No repetition loops detected."))

        # 4. Google Drive Sync Checks
        drive_md = self.local_drive / "Transcripts" / md_path.name
        drive_mp3 = self.local_drive / "TrimmedAudio" / trimmed_audio_path.name
        if self.local_drive.exists():
            if drive_md.exists() and drive_md.stat().st_size == md_path.stat().st_size:
                checks.append(CheckResult("DRIVE_TRANSCRIPT_SYNC", True, "INFO", "Transcript synced to Google Drive with matching byte count."))
            else:
                checks.append(CheckResult("DRIVE_TRANSCRIPT_SYNC", False, "WARNING", "Transcript missing from Drive or byte count differs."))

            if drive_mp3.exists() and trimmed_audio_path.exists() and drive_mp3.stat().st_size == trimmed_audio_path.stat().st_size:
                checks.append(CheckResult("DRIVE_AUDIO_SYNC", True, "INFO", "Trimmed audio synced to Google Drive with matching byte count."))
            else:
                checks.append(CheckResult("DRIVE_AUDIO_SYNC", False, "WARNING", "Trimmed audio missing from Drive or byte count differs."))

        # Determine overall episode status
        has_error = any(not c.passed and c.severity == "ERROR" for c in checks)
        has_warn = any(not c.passed and c.severity == "WARNING" for c in checks)

        if has_error:
            overall_status = "FAIL"
        elif has_warn:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        return EpisodeQAReport(
            filename=md_path.name,
            date_iso=date_iso,
            title=title,
            status=overall_status,
            raw_audio_size_mb=round(raw_size_mb, 2),
            raw_duration_s=round(raw_dur, 1),
            trimmed_audio_size_mb=round(trimmed_size_mb, 2),
            trimmed_duration_s=round(trimmed_dur, 1),
            sermon_ratio=round(sermon_ratio, 2),
            word_count=word_count,
            prosody_stress_count=prosody_stress_count,
            prosody_tag_count=prosody_tag_count,
            checks=checks,
        )

    def audit_all(self, output_json: bool = True) -> List[EpisodeQAReport]:
        """Audit all processed episodes and print a comprehensive summary table."""
        md_files = sorted(self.processed_dir.glob("*.md"))
        reports: List[EpisodeQAReport] = []

        for f in md_files:
            rep = self.validate_episode(f)
            reports.append(rep)

        self.print_audit_table(reports)

        if output_json:
            with open(REPORT_PATH, "w", encoding="utf-8") as out:
                json.dump([r.to_dict() for r in reports], out, indent=2)
            logger.info(f"QA audit report written to: {REPORT_PATH}")

        return reports

    def print_audit_table(self, reports: List[EpisodeQAReport]):
        """Print clean terminal audit table with pass/warn/fail indicators."""
        print("\n" + "=" * 115)
        print(" " * 42 + "CI / QA GUARDRAILS REPORT")
        print("=" * 115)
        header = f"{'Status':<6} | {'Date':<10} | {'Title':<35} | {'Sermon Dur':<11} | {'Words':<7} | {'Prosody':<9} | {'Flags / Notes'}"
        print(header)
        print("-" * 115)

        pass_count = 0
        warn_count = 0
        fail_count = 0

        for r in reports:
            short_title = (r.title[:32] + "...") if len(r.title) > 35 else r.title
            sermon_dur = f"{r.trimmed_duration_s/60:.1f}m ({int(r.sermon_ratio*100)}%)" if r.trimmed_duration_s > 0 else "N/A"
            prosody_info = f"{r.prosody_stress_count}b / {r.prosody_tag_count}t"

            issues = [c.message for c in r.checks if not c.passed]
            notes = issues[0] if issues else "All checks passed"
            if len(notes) > 28:
                notes = notes[:25] + "..."

            if r.status == "PASS":
                pass_count += 1
            elif r.status == "WARN":
                warn_count += 1
            else:
                fail_count += 1

            row = f"{r.status:<6} | {r.date_iso:<10} | {short_title:<35} | {sermon_dur:<11} | {r.word_count:<7} | {prosody_info:<9} | {notes}"
            print(row)

        print("-" * 115)
        print(f"Total Evaluated: {len(reports)} | PASS: {pass_count} | WARN: {warn_count} | FAIL: {fail_count}")
        print("=" * 115 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Podcast Processing CI & QA Validator")
    parser.add_argument("--audit", action="store_true", default=True, help="Run audit on all processed files")
    parser.add_argument("--file", help="Validate specific markdown file")
    parser.add_argument("--check-report", type=Path, help="Fail if a committed QA report is empty or contains FAIL entries")

    args = parser.parse_args()
    if args.check_report:
        report_data = json.loads(args.check_report.read_text(encoding="utf-8"))
        if not report_data:
            logger.error("QA report is empty; refusing to pass.")
            sys.exit(1)
        failed = [entry for entry in report_data if entry.get("status") == "FAIL"]
        if failed:
            logger.error("QA report contains %d failed episode(s).", len(failed))
            sys.exit(1)
        logger.info("QA report accepted: %d episodes, no FAIL entries.", len(report_data))
        sys.exit(0)

    validator = PodcastValidator()

    if args.file:
        rep = validator.validate_episode(Path(args.file))
        print(json.dumps(rep.to_dict(), indent=2))
        if rep.status == "FAIL":
            sys.exit(1)
    else:
        reports = validator.audit_all()
        if not reports:
            logger.error("No transcript artifacts were evaluated; refusing to report a passing QA run.")
            sys.exit(1)
        if any(r.status == "FAIL" for r in reports):
            sys.exit(1)
