"""Downloader module for podcast RSS feeds with direct audio extraction."""

from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional

import feedparser
import static_ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize static ffmpeg binaries if system ffmpeg is not present
static_ffmpeg.add_paths()

DEFAULT_FEED_URL = "https://s3.amazonaws.com/ccstarchives/xml/videopodcast.xml"
DEFAULT_AUDIO_DIR = Path(__file__).parent / "RawAudio"
DEFAULT_PROCESSED_DIR = Path(__file__).parent / "ProcessedMD"


@dataclass
class Episode:
    index: int
    guid: str
    title: str
    author: str
    summary: str
    description: str
    pub_date: str
    date_iso: str
    duration: str
    media_url: str
    media_length: Optional[int]
    media_type: str
    audio_filename: str
    md_filename: str

    def to_dict(self) -> dict:
        return asdict(self)


def sanitize_filename(name: str, max_length: int = 120) -> str:
    """Sanitize title for safe filesystem filenames."""
    # Replace dangerous characters with hyphens
    clean = re.sub(r'[\\/*?:"<>|]', "-", name)
    # Collapse multiple spaces or hyphens
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"-{2,}", "-", clean)
    clean = clean.strip(" .-_")
    if not clean:
        clean = "Untitled_Episode"
    return clean[:max_length]


def parse_pub_date(entry: feedparser.FeedParserDict) -> tuple[str, str]:
    """Extract standard ISO date (YYYY-MM-DD) and raw date string."""
    pub_raw = entry.get("published", "")
    dt = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            dt = datetime.datetime(*entry.published_parsed[:6])
        except Exception:
            pass

    if dt is None and pub_raw:
        try:
            dt = parsedate_to_datetime(pub_raw)
        except Exception:
            pass

    if dt:
        date_iso = dt.strftime("%Y-%m-%d")
    else:
        date_iso = "UNKNOWN_DATE"
    return date_iso, pub_raw


def fetch_episodes(feed_url: str = DEFAULT_FEED_URL) -> List[Episode]:
    """Fetch and parse all episodes from the given RSS feed URL."""
    logger.info(f"Fetching RSS feed from: {feed_url}")
    feed = feedparser.parse(feed_url)
    if feed.bozo and not feed.entries:
        raise ValueError(f"Failed to parse RSS feed from {feed_url}: {feed.bozo_exception}")

    episodes: List[Episode] = []
    for idx, entry in enumerate(feed.entries):
        title = entry.get("title", f"Episode {idx + 1}").strip()
        guid = entry.get("guid", entry.get("id", f"ep_{idx + 1}"))
        author = entry.get("author", entry.get("itunes_author", "Unknown Author"))
        summary = entry.get("summary", entry.get("itunes_summary", "")).strip()
        description = entry.get("description", "").strip()
        duration = entry.get("itunes_duration", "")

        date_iso, pub_raw = parse_pub_date(entry)

        # Enclosure
        media_url = ""
        media_length = None
        media_type = ""
        enclosures = entry.get("enclosures", [])
        if enclosures:
            enc = enclosures[0]
            media_url = enc.get("href", "")
            try:
                media_length = int(enc.get("length")) if enc.get("length") else None
            except (ValueError, TypeError):
                media_length = None
            media_type = enc.get("type", "")

        # Fallback to link if no enclosure
        if not media_url:
            media_url = entry.get("link", "")

        clean_title = sanitize_filename(title)
        audio_filename = f"{date_iso} - {clean_title}.mp3"
        md_filename = f"{date_iso} - {clean_title}.md"

        episodes.append(
            Episode(
                index=idx + 1,
                guid=guid,
                title=title,
                author=author,
                summary=summary,
                description=description,
                pub_date=pub_raw,
                date_iso=date_iso,
                duration=duration,
                media_url=media_url,
                media_length=media_length,
                media_type=media_type,
                audio_filename=audio_filename,
                md_filename=md_filename,
            )
        )

    logger.info(f"Parsed {len(episodes)} episodes from feed.")
    return episodes


def download_audio(
    episode: Episode,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    skip_existing: bool = True,
    force: bool = False,
    audio_bitrate: str = "64k",
) -> tuple[Path, bool]:
    """
    Download and convert the episode media to MP3 format directly using ffmpeg.
    Returns (audio_path, was_downloaded).
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dest_path = audio_dir / episode.audio_filename
    md_path = processed_dir / episode.md_filename

    # Skip logic
    if not force and skip_existing:
        if dest_path.exists() and dest_path.stat().st_size > 1024:
            logger.info(f"Audio file already exists: {dest_path.name}")
            return dest_path, False
        if md_path.exists() and md_path.stat().st_size > 100:
            logger.info(f"Processed MD already exists for episode: {episode.title} -> {md_path.name}")
            # If MD exists, we might not strictly need audio, but if requested we return dest_path
            if dest_path.exists():
                return dest_path, False

    if not episode.media_url:
        raise ValueError(f"No media URL found for episode: {episode.title}")

    logger.info(f"Downloading/Extracting audio for [{episode.index}] '{episode.title}' from {episode.media_url}")
    tmp_path = dest_path.with_suffix(".tmp.mp3")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in PATH or static_ffmpeg location.")

    # ffmpeg command: stream over HTTP directly, extract audio, convert to standard MP3
    cmd = [
        ffmpeg_bin,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        episode.media_url,
        "-vn",  # No video
        "-acodec",
        "libmp3lame",
        "-ab",
        audio_bitrate,
        "-ar",
        "24000",  # 24kHz is optimal quality/size for speech
        "-ac",
        "1",  # Mono for speech (reduces size 50% with zero quality loss for sermon speech)
        str(tmp_path),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error(f"ffmpeg extraction failed: {e.stderr}")
        # Fallback to yt-dlp if direct ffmpeg fails
        logger.info("Attempting fallback with yt-dlp...")
        yt_cmd = [
            "yt-dlp",
            "-x",
            "--audio-format",
            "mp3",
            "--audio-quality",
            "64K",
            "-o",
            str(tmp_path),
            episode.media_url,
        ]
        subprocess.run(yt_cmd, check=True)

    if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Extracted audio file is empty or missing: {tmp_path}")

    # Rename tmp to final destination
    tmp_path.replace(dest_path)
    elapsed = time.time() - t0
    file_size_mb = dest_path.stat().st_size / (1024 * 1024)
    logger.info(f"Extracted '{dest_path.name}' ({file_size_mb:.2f} MB) in {elapsed:.1f}s")

    return dest_path, True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download podcast episode audio")
    parser.add_argument("--feed", default=DEFAULT_FEED_URL, help="RSS feed URL")
    parser.add_argument("--limit", type=int, default=1, help="Number of episodes to download (0 for all)")
    parser.add_argument("--force", action="store_true", help="Force re-download even if exists")
    args = parser.parse_args()

    episodes = fetch_episodes(args.feed)
    to_download = episodes if args.limit == 0 else episodes[: args.limit]

    for ep in to_download:
        path, downloaded = download_audio(ep, force=args.force)
        print(f"[{'DOWNLOADED' if downloaded else 'CACHED'}] {path}")
