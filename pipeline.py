"""Autonomous, end-to-end podcast audio processing, prosody transcription, preaching extraction & Drive sync pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from downloader import (
    DEFAULT_AUDIO_DIR,
    DEFAULT_FEED_URL,
    DEFAULT_PROCESSED_DIR,
    Episode,
    download_audio,
    fetch_episodes,
)
from drive_sync import DriveUploader
from notebooklm_sync import NotebookLMSync
from transcriber import ProsodyTranscriber
from trimmer import DEFAULT_TRIMMED_DIR, PreachingTrimmer, SermonBoundary, get_audio_duration

load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

MANIFEST_PATH = Path(__file__).parent / "manifest.json"
INDEX_MD_PATH = Path(__file__).parent / "INDEX.md"
INDEX_CSV_PATH = Path(__file__).parent / "index.csv"


@dataclass
class ProcessingRecord:
    index: int
    guid: str
    title: str
    date_iso: str
    pub_date: str
    duration: str
    audio_file: str
    audio_size_mb: float
    md_file: str
    status: str  # "SUCCESS", "FAILED", "PARTIAL", "SKIPPED"
    trimmed_audio_file: Optional[str] = None
    preaching_start: Optional[str] = None
    preaching_end: Optional[str] = None
    speaker_name: Optional[str] = None
    error: Optional[str] = None
    drive_file_id: Optional[str] = None
    drive_link: Optional[str] = None
    transcription_time_s: Optional[float] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class PodcastPipeline:
    """Orchestrates end-to-end podcast downloading, prosody transcription, preaching trimming & Drive sync."""

    def __init__(
        self,
        feed_url: str = DEFAULT_FEED_URL,
        audio_dir: Path = DEFAULT_AUDIO_DIR,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        trimmed_dir: Path = DEFAULT_TRIMMED_DIR,
        model_name: Optional[str] = None,
        drive_folder_id: Optional[str] = None,
        notebook_id: Optional[str] = None,
    ):
        self.feed_url = feed_url
        self.audio_dir = audio_dir
        self.processed_dir = processed_dir
        self.trimmed_dir = trimmed_dir
        self.drive_uploader = DriveUploader(folder_id=drive_folder_id)
        self.transcriber = ProsodyTranscriber(preferred_model=model_name)
        self.trimmer = PreachingTrimmer()
        self.notebooklm_syncer = NotebookLMSync(notebook_id=notebook_id)
        self.manifest: Dict[str, dict] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, dict]:
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing manifest: {e}")
        return {}

    def _save_manifest(self):
        """Atomically persist manifest.json to prevent corruption on interruption."""
        tmp_path = MANIFEST_PATH.with_suffix(".tmp.json")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, MANIFEST_PATH)
            self.export_indexes()
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            logger.warning(f"Could not save manifest: {e}")

    def export_indexes(self):
        """Export master INDEX.md and index.csv, and sync to Google Drive."""
        valid_fields = set(ProcessingRecord.__dataclass_fields__)
        records = sorted(
            [
                ProcessingRecord(**{k: v for k, v in data.items() if k in valid_fields})
                for data in self.manifest.values()
            ],
            key=lambda r: (r.date_iso, r.index),
            reverse=True,
        )
        if not records:
            return

        # 1. Export index.csv
        try:
            tmp_csv = INDEX_CSV_PATH.with_suffix(".tmp.csv")
            with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Index",
                    "Date",
                    "Sermon Title",
                    "Primary Speaker",
                    "Service Duration",
                    "Preaching Start",
                    "Preaching End",
                    "Audio Size (MB)",
                    "Status",
                    "Transcript Markdown",
                    "Trimmed Audio File",
                    "Completed At",
                    "GUID",
                ])
                for r in records:
                    writer.writerow([
                        r.index,
                        r.date_iso,
                        r.title,
                        r.speaker_name or "Unknown",
                        r.duration or "N/A",
                        r.preaching_start or "00:00:00",
                        r.preaching_end or "End",
                        f"{r.audio_size_mb:.1f}",
                        r.status,
                        r.md_file,
                        r.trimmed_audio_file or "",
                        r.completed_at or "",
                        r.guid,
                    ])
            os.replace(tmp_csv, INDEX_CSV_PATH)
        except Exception as e:
            logger.warning(f"Failed to export index.csv: {e}")

        # 2. Export INDEX.md
        try:
            success_count = sum(1 for r in records if r.status in ["SUCCESS", "PARTIAL"])
            md_lines = [
                "# Christ Chapel Podcast Archive Master Index",
                "",
                f"- **Total Cataloged Episodes:** {len(records)}",
                f"- **Successfully Processed & Trimmed:** {success_count}",
                f"- **Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "| # | Date | Sermon Title | Primary Speaker | Service Length | Preaching Segment | Status | Transcript | Preaching Audio |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for r in records:
                speaker = r.speaker_name or "John C. Wood"
                title_clean = r.title.replace("|", "-")
                preach_seg = f"`{r.preaching_start or '00:00:00'}` → `{r.preaching_end or 'End'}`"
                status_badge = "✅ SUCCESS" if r.status == "SUCCESS" else f"⚠️ {r.status}"
                md_link = f"[{r.md_file}](Transcripts/{r.md_file})" if r.md_file else "N/A"
                audio_link = (
                    f"[{r.trimmed_audio_file}](TrimmedAudio/{r.trimmed_audio_file})"
                    if r.trimmed_audio_file
                    else "N/A"
                )
                row = f"| {r.index} | **{r.date_iso}** | {title_clean} | {speaker} | {r.duration} | {preach_seg} | {status_badge} | {md_link} | {audio_link} |"
                md_lines.append(row)

            md_content = "\n".join(md_lines) + "\n"
            tmp_md = INDEX_MD_PATH.with_suffix(".tmp.md")
            tmp_md.write_text(md_content, encoding="utf-8")
            os.replace(tmp_md, INDEX_MD_PATH)
        except Exception as e:
            logger.warning(f"Failed to export INDEX.md: {e}")

        # 3. Sync INDEX.md and index.csv to Google Drive Root
        if self.drive_uploader.local_drive_path and self.drive_uploader.local_drive_path.exists():
            try:
                if INDEX_MD_PATH.exists():
                    shutil.copy2(INDEX_MD_PATH, self.drive_uploader.local_drive_path / "INDEX.md")
                if INDEX_CSV_PATH.exists():
                    shutil.copy2(INDEX_CSV_PATH, self.drive_uploader.local_drive_path / "index.csv")
                logger.info("Synced INDEX.md and index.csv to Google Drive root folder.")
            except Exception as e:
                logger.warning(f"Failed copying index files to Drive: {e}")
    def process_episode(
        self,
        episode: Episode,
        force: bool = False,
        skip_transcription: bool = False,
    ) -> ProcessingRecord:
        """Execute the end-to-end processing pipeline for a single episode."""
        logger.info("=" * 70)
        logger.info(f"Processing Episode [{episode.index}]: '{episode.title}' ({episode.date_iso})")
        logger.info("=" * 70)

        md_dest = self.processed_dir / episode.md_filename
        trimmed_dest = self.trimmed_dir / f"{Path(episode.audio_filename).stem} - Preaching.mp3"
        existing_record = self.manifest.get(episode.guid)

        if not force and md_dest.exists() and md_dest.stat().st_size > 200 and trimmed_dest.exists():
            if existing_record and existing_record.get("status") in ["SUCCESS", "PARTIAL"]:
                logger.info(f"Episode already fully processed & trimmed: {md_dest.name}")
                return ProcessingRecord(**existing_record)

        # Step 1: Audio Download & Extraction
        try:
            audio_path, downloaded = download_audio(
                episode=episode,
                audio_dir=self.audio_dir,
                processed_dir=self.processed_dir,
                skip_existing=not force,
                force=force,
            )
            audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            logger.error(f"Failed to download/extract audio for '{episode.title}': {e}")
            record = ProcessingRecord(
                index=episode.index,
                guid=episode.guid,
                title=episode.title,
                date_iso=episode.date_iso,
                pub_date=episode.pub_date,
                duration=episode.duration,
                audio_file=episode.audio_filename,
                audio_size_mb=0.0,
                md_file=episode.md_filename,
                status="FAILED",
                error=f"Download error: {e}",
            )
            self.manifest[episode.guid] = record.to_dict()
            self._save_manifest()
            return record

        if skip_transcription:
            record = ProcessingRecord(
                index=episode.index,
                guid=episode.guid,
                title=episode.title,
                date_iso=episode.date_iso,
                pub_date=episode.pub_date,
                duration=episode.duration,
                audio_file=audio_path.name,
                audio_size_mb=round(audio_size_mb, 2),
                md_file=episode.md_filename,
                status="DOWNLOADED",
            )
            self.manifest[episode.guid] = record.to_dict()
            self._save_manifest()
            return record

        # Step 2: Prosody Transcription via Gemini
        t0 = time.time()
        md_path = md_dest
        try:
            if not md_dest.exists() or md_dest.stat().st_size < 200 or force:
                raw_transcript = self.transcriber.transcribe_audio_file(
                    audio_path=audio_path,
                    episode=episode,
                )
                transcription_time = round(time.time() - t0, 1)
                md_path = self.transcriber.create_formatted_markdown(
                    raw_transcript=raw_transcript,
                    episode=episode,
                    audio_path=audio_path,
                    output_dir=self.processed_dir,
                )
            else:
                logger.info(f"Loaded existing transcript: {md_dest.name}")
                transcription_time = existing_record.get("transcription_time_s") if existing_record else 0.0

        except Exception as e:
            logger.error(f"Failed transcription for '{episode.title}': {e}")
            record = ProcessingRecord(
                index=episode.index,
                guid=episode.guid,
                title=episode.title,
                date_iso=episode.date_iso,
                pub_date=episode.pub_date,
                duration=episode.duration,
                audio_file=audio_path.name,
                audio_size_mb=round(audio_size_mb, 2),
                md_file=episode.md_filename,
                status="FAILED",
                error=f"Transcription error: {e}",
            )
            self.manifest[episode.guid] = record.to_dict()
            self._save_manifest()
            return record

        # Step 3: Sermon Boundary Detection & Preaching Audio Extraction
        boundary: Optional[SermonBoundary] = None
        trimmed_path: Optional[Path] = None
        trimming_failed = False
        try:
            transcript_text = md_path.read_text(encoding="utf-8")
            total_dur = get_audio_duration(audio_path)
            boundary = self.trimmer.detect_boundaries_from_transcript(transcript_text, total_dur)
            trimmed_path = self.trimmer.extract_preaching_audio(
                raw_audio_path=audio_path,
                boundary=boundary,
                output_dir=self.trimmed_dir,
                force=force,
            )
        except Exception as e:
            trimming_failed = True
            logger.warning(f"Could not trim preaching audio for '{episode.title}': {e}")

        # Step 4: Google Drive Sync (Transcripts & Trimmed Audio)
        drive_info = self.drive_uploader.upload_markdown(
            file_path=md_path,
            title=f"{episode.date_iso} - {episode.title}",
            description=f"Prosody transcript for: {episode.title}",
        )
        if trimmed_path and trimmed_path.exists():
            self.drive_uploader.upload_audio(
                file_path=trimmed_path,
                title=f"{episode.date_iso} - {episode.title} - Preaching",
                description=f"Preaching section only ({boundary.start_timestamp if boundary else 'N/A'})",
            )

        drive_file_id = drive_info.get("file_id") if drive_info else None
        drive_link = (
            drive_info.get("web_view_link")
            or f"https://drive.google.com/drive/folders/{self.drive_uploader.transcripts_folder_id}"
        )

        # Step 5: Direct NotebookLM Ingestion (if authenticated)
        if self.notebooklm_syncer.is_available:
            try:
                self.notebooklm_syncer.sync_transcript(md_path)
            except Exception as nlm_err:
                logger.warning(f"NotebookLM auto-import note for '{episode.title}': {nlm_err}")

        speaker_final = boundary.speaker_name if (boundary and boundary.speaker_name and boundary.speaker_name != "Preacher") else (episode.author or "John C. Wood")
        final_status = "PARTIAL" if trimming_failed else "SUCCESS"

        record = ProcessingRecord(
            index=episode.index,
            guid=episode.guid,
            title=episode.title,
            date_iso=episode.date_iso,
            pub_date=episode.pub_date,
            duration=episode.duration,
            audio_file=audio_path.name,
            audio_size_mb=round(audio_size_mb, 2),
            md_file=md_path.name,
            status=final_status,
            trimmed_audio_file=trimmed_path.name if trimmed_path else None,
            preaching_start=boundary.start_timestamp if boundary else None,
            preaching_end=boundary.end_timestamp if boundary else None,
            speaker_name=speaker_final,
            drive_file_id=drive_file_id,
            drive_link=drive_link,
            transcription_time_s=transcription_time,
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.manifest[episode.guid] = record.to_dict()
        self._save_manifest()
        return record

    def run_pipeline(
        self,
        limit: int = 0,
        dry_run: bool = False,
        force: bool = False,
    ) -> List[ProcessingRecord]:
        """
        Run the batch pipeline with concurrent background prefetching of upcoming audio downloads.
        """
        episodes = fetch_episodes(self.feed_url)
        if not episodes:
            logger.warning("No episodes found in feed.")
            return []

        if dry_run:
            logger.info(">>> DRY RUN MODE: Processing first episode only <<<")
            target_episodes = [episodes[0]]
        elif limit > 0:
            target_episodes = episodes[:limit]
        else:
            target_episodes = episodes

        logger.info(f"Targeting {len(target_episodes)} episode(s) for processing with concurrent prefetch.")
        results: List[ProcessingRecord] = []

        def prefetch_worker(next_ep: Episode):
            """Background worker to prefetch next audio file while current episode transcribes."""
            try:
                logger.info(f"⚡ [Prefetch] Downloading next audio in background: [{next_ep.index}] '{next_ep.title}'...")
                download_audio(
                    episode=next_ep,
                    audio_dir=self.audio_dir,
                    processed_dir=self.processed_dir,
                    skip_existing=not force,
                    force=force,
                )
            except Exception as err:
                logger.warning(f"Prefetch download note for '{next_ep.title}': {err}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as prefetch_executor:
            prefetch_future = None

            for idx, ep in enumerate(target_episodes):
                # Trigger background prefetch for the NEXT episode (idx + 1)
                if idx + 1 < len(target_episodes):
                    next_episode = target_episodes[idx + 1]
                    prefetch_future = prefetch_executor.submit(prefetch_worker, next_episode)

                record = self.process_episode(ep, force=force)
                results.append(record)

                # Wait for next episode's prefetch to finish before moving to next iteration
                if prefetch_future:
                    try:
                        prefetch_future.result(timeout=600)
                    except Exception:
                        pass

        self.print_summary_table(results)
        return results

    def print_summary_table(self, records: List[ProcessingRecord]):
        """Output a clean, structured summary table of results."""
        print("\n" + "=" * 125)
        print(" " * 45 + "PIPELINE EXECUTION SUMMARY")
        print("=" * 125)
        header = (
            f"{'#':<3} | {'Date':<10} | {'Title':<35} | {'Speaker':<15} | {'Preaching Cut':<18} | {'Status':<8}"
        )
        print(header)
        print("-" * 125)

        success_count = 0
        failed_count = 0

        for r in records:
            short_title = (r.title[:32] + "...") if len(r.title) > 35 else r.title
            speaker = (r.speaker_name[:12] + "...") if (r.speaker_name and len(r.speaker_name) > 15) else (r.speaker_name or "John C. Wood")
            preach_seg = f"{r.preaching_start or '00:00:00'} → {r.preaching_end or 'End'}"

            if r.status in ["SUCCESS", "PARTIAL"]:
                success_count += 1
            else:
                failed_count += 1

            row = f"{r.index:<3} | {r.date_iso:<10} | {short_title:<35} | {speaker:<15} | {preach_seg:<18} | {r.status:<8}"
            print(row)

        print("-" * 125)
        print(f"Total: {len(records)} | Success: {success_count} | Failed: {failed_count}")
        print("=" * 125 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Podcast Download, Transcription, Preaching Trimmer & Sync Pipeline")
    parser.add_argument("--feed", default=DEFAULT_FEED_URL, help="RSS feed URL")
    parser.add_argument("--dry-run", action="store_true", help="Process 1st episode end-to-end as dry run")
    parser.add_argument("--limit", type=int, default=0, help="Number of episodes to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force re-processing of already completed items")
    parser.add_argument("--model", default=None, help="Preferred Gemini model name")
    parser.add_argument("--drive-folder", default=None, help="Google Drive folder ID")
    parser.add_argument("--notebook-id", default=None, help="Target NotebookLM notebook ID")

    args = parser.parse_args()

    pipeline = PodcastPipeline(
        feed_url=args.feed,
        model_name=args.model,
        drive_folder_id=args.drive_folder,
        notebook_id=args.notebook_id,
    )

    pipeline.run_pipeline(
        limit=args.limit,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
