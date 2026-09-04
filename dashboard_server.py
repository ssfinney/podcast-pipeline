"""Real-time web dashboard server for monitoring podcast downloading, transcription, trimming & Drive sync."""

from __future__ import annotations

import html
import http.server
import json
import logging
import math
import os
import re
import socketserver
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "RawAudio"
PROCESSED_DIR = BASE_DIR / "ProcessedMD"
TRIMMED_DIR = BASE_DIR / "TrimmedAudio"
MANIFEST_PATH = BASE_DIR / "manifest.json"
INDEX_CSV_PATH = BASE_DIR / "index.csv"
DASHBOARD_HTML_PATH = BASE_DIR / "dashboard.html"
INDEX_MD_PATH = BASE_DIR / "INDEX.md"
DRIVE_LOCAL_PATH = Path(
    os.getenv(
        "GOOGLE_DRIVE_LOCAL_PATH",
        "/Users/stephenfinney/Library/CloudStorage/GoogleDrive-ssfinney92@gmail.com/My Drive/Christ Chapel Podcasts",
    )
)

PORT = int(os.getenv("DASHBOARD_PORT", "8420"))
DOWNLOAD_STALE_SEC = 300
TRANSCRIBE_STALE_SEC = 900

# Cache feed entries
FEED_CACHE_FILE = BASE_DIR / "feed_cache.json"

# Cache storage calculations
_STORAGE_CACHE = {"timestamp": 0.0, "data": {"raw_mb": 0.0, "trimmed_mb": 0.0, "md_mb": 0.0, "total_mb": 0.0}}
_FEED_CACHE_LOCK = threading.Lock()




def get_cached_storage_stats() -> Dict[str, float]:
    """Return storage stats with 30s TTL cache to avoid heavy recursive stats on every 3s poll."""
    now = time.time()
    if now - _STORAGE_CACHE["timestamp"] < 30.0:
        return _STORAGE_CACHE["data"]

    def dir_size_mb(path: Path) -> float:
        if not path.exists():
            return 0.0
        total = 0
        for file_path in path.glob("**/*"):
            try:
                if file_path.is_file():
                    total += file_path.stat().st_size
            except OSError:
                continue
        return total / (1024 * 1024)

    raw_mb = dir_size_mb(AUDIO_DIR)
    trimmed_mb = dir_size_mb(TRIMMED_DIR)
    md_mb = dir_size_mb(PROCESSED_DIR)

    stats = {
        "raw_mb": round(raw_mb, 1),
        "trimmed_mb": round(trimmed_mb, 1),
        "md_mb": round(md_mb, 2),
        "total_mb": round(raw_mb + trimmed_mb + md_mb, 1),
    }
    _STORAGE_CACHE["timestamp"] = now
    _STORAGE_CACHE["data"] = stats
    return stats


def get_feed_entries() -> List[dict]:
    """Load or refresh the RSS feed cache once per process."""
    with _FEED_CACHE_LOCK:
        return _get_feed_entries_locked()


def _get_feed_entries_locked() -> List[dict]:
    """Load or cache all episodes from RSS feed. Caller holds the cache lock."""
    try:
        cache_stat = FEED_CACHE_FILE.stat()
        if time.time() - cache_stat.st_mtime < 3600:
            with open(FEED_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError):
        pass

    from downloader import DEFAULT_FEED_URL, fetch_episodes

    try:
        eps = fetch_episodes(DEFAULT_FEED_URL)
        entries = [e.to_dict() for e in eps]
        # Atomic persist so concurrent readers never see partial JSON
        tmp_cache = FEED_CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_cache, "w", encoding="utf-8") as f:
            json.dump(entries, f)
        os.replace(tmp_cache, FEED_CACHE_FILE)
        return entries
    except Exception as e:
        tmp_cache_path = FEED_CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
        tmp_cache_path.unlink(missing_ok=True)
        logger.warning(f"Error fetching feed: {e}")
        return []


def _safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def detect_current_active_stage() -> Dict[str, Any]:
    """Detect what the background pipeline is actively doing by inspecting disk state."""
    active_info = {
        "is_active": False,
        "stage": "IDLE",
        "stage_label": "Idle / Waiting",
        "episode_index": None,
        "episode_title": None,
        "episode_date": None,
        "details": "",
        "percent": 0,
        "chunks_done": 0,
        "chunks_total": 0,
    }

    # 1. Check if downloading audio (.tmp.mp3 modified recently)
    now = time.time()
    tmp_audios = []
    for path in AUDIO_DIR.glob("*.tmp.mp3"):
        stat = _safe_stat(path)
        if stat and now - stat.st_mtime < DOWNLOAD_STALE_SEC:
            tmp_audios.append((path, stat))
    if tmp_audios:
        tmp_file, tmp_stat = tmp_audios[0]
        size_mb = tmp_stat.st_size / (1024 * 1024)
        active_info.update({
            "is_active": True,
            "stage": "DOWNLOADING",
            "stage_label": "Downloading & Extracting Audio",
            "episode_title": tmp_file.stem.replace(".tmp", ""),
            "details": f"Streaming audio from S3 ({size_mb:.1f} MB downloaded)",
            "percent": min(95, int((size_mb / 40.0) * 100)),
        })
        return active_info

    # 2. Check if transcribing chunks (.chunk_cache_*)
    chunk_dirs = []
    for path in AUDIO_DIR.glob(".chunk_cache_*"):
        stat = _safe_stat(path)
        if stat and now - stat.st_mtime < TRANSCRIBE_STALE_SEC:
            chunk_dirs.append((path, stat))
    if chunk_dirs:
        cdir, _ = max(chunk_dirs, key=lambda item: item[1].st_mtime)
        title_with_suffix = re.sub(r"^\.chunk_cache_", "", cdir.name)
        suffix_match = re.search(r"_(\d+)s$", title_with_suffix)
        chunk_duration_sec = int(suffix_match.group(1)) if suffix_match else 480
        title = re.sub(r"_\d+s$", "", title_with_suffix)
        txt_chunks = len(list(cdir.glob("chunk_*.txt")))
        mp3_chunks = len(list(cdir.glob("chunk_*.mp3")))
        source_audio = AUDIO_DIR / f"{title}.mp3"
        source_stat = _safe_stat(source_audio)
        if source_stat:
            estimated_duration = (source_stat.st_size * 8) / 64_000
            total_estimate = max(mp3_chunks, 1, math.ceil(estimated_duration / chunk_duration_sec))
        else:
            total_estimate = max(mp3_chunks, 1)
        active_info.update({
            "is_active": True,
            "stage": "TRANSCRIBING",
            "stage_label": "Transcribing with Gemini (Prosody Extraction)",
            "episode_title": title,
            "chunks_done": txt_chunks,
            "chunks_total": total_estimate,
            "details": f"Processing chunk {txt_chunks + 1} of {total_estimate} ({txt_chunks}/{total_estimate} complete)",
            "percent": int((txt_chunks / total_estimate) * 100) if total_estimate else 50,
        })
        return active_info

    # 3. Check if trimming (.tmp.mp3 in TrimmedAudio)
    for tmp_file in TRIMMED_DIR.glob("*.tmp.mp3"):
        tmp_stat = _safe_stat(tmp_file)
        if not tmp_stat:
            continue
        size_mb = tmp_stat.st_size / (1024 * 1024)
        active_info.update({
            "is_active": True,
            "stage": "TRIMMING",
            "stage_label": "Trimming Sermon & Extracting Preaching Cut",
            "episode_title": tmp_file.stem.replace(".tmp", "").replace(" - Preaching", ""),
            "details": f"Encoding trimmed preaching audio with ffmpeg ({size_mb:.1f} MB)",
            "percent": 85,
        })
        return active_info

    return active_info


def get_pipeline_status() -> Dict[str, Any]:
    """Compile comprehensive pipeline metrics and episode lists."""
    feed_items = get_feed_entries()
    total_episodes = len(feed_items) or 720

    manifest = {}
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            pass

    processed_records = list(manifest.values())
    success_count = sum(1 for r in processed_records if r.get("status") in ["SUCCESS", "PARTIAL"])
    failed_count = sum(1 for r in processed_records if r.get("status") == "FAILED")

    active_job = detect_current_active_stage()

    # Calculate total preaching hours
    total_preaching_sec = 0.0
    for r in processed_records:
        if r.get("status") in ["SUCCESS", "PARTIAL"]:
            start_ts = r.get("preaching_start")
            end_ts = r.get("preaching_end")
            if start_ts and end_ts:
                from trimmer import parse_timestamp_to_seconds

                dur = max(0, parse_timestamp_to_seconds(end_ts) - parse_timestamp_to_seconds(start_ts))
                total_preaching_sec += dur
            # Missing boundaries are excluded rather than replaced with an
            # invented 40-minute estimate.

    preach_hours = int(total_preaching_sec // 3600)
    preach_mins = int((total_preaching_sec % 3600) // 60)

    storage_stats = get_cached_storage_stats()

    # Compile enriched episode queue
    episodes_summary = []
    for item in feed_items:
        guid = item.get("guid")
        rec = manifest.get(guid)
        status = "COMPLETED" if (rec and rec.get("status") in ["SUCCESS", "PARTIAL"]) else ("FAILED" if (rec and rec.get("status") == "FAILED") else "QUEUED")
        if active_job.get("is_active") and active_job.get("episode_title") and (item.get("title") in active_job.get("episode_title") or item.get("date_iso") in active_job.get("episode_title")):
            status = "IN_PROGRESS"

        episodes_summary.append({
            "index": item.get("index"),
            "date": item.get("date_iso"),
            "title": item.get("title"),
            "author": item.get("author") or (rec.get("speaker_name") if rec else "John C. Wood"),
            "duration": item.get("duration"),
            "status": status,
            "preaching_start": rec.get("preaching_start") if rec else None,
            "preaching_end": rec.get("preaching_end") if rec else None,
            "md_file": rec.get("md_file") if rec else None,
            "trimmed_file": rec.get("trimmed_audio_file") if rec else None,
            "drive_link": rec.get("drive_link") if rec else "https://drive.google.com/drive/folders/1bGjKB1GcSGIP1ZEpcLzUQPZiEUyDzTtT",
            "audio_size_mb": rec.get("audio_size_mb") if rec else None,
            "completed_at": rec.get("completed_at") if rec else None,
        })

    return {
        "total_episodes": total_episodes,
        "processed_count": success_count,
        "failed_count": failed_count,
        "percent_complete": round((success_count / total_episodes) * 100, 1) if total_episodes else 0.0,
        "active_job": active_job,
        "total_preaching_time": f"{preach_hours}h {preach_mins}m",
        "storage": storage_stats,
        "drive_synced": DRIVE_LOCAL_PATH.exists(),
        "drive_path": str(DRIVE_LOCAL_PATH),
        "drive_links": {
            "root": "https://drive.google.com/drive/folders/1emYUaJPU0_5qoGgNphNW51A12wwrUWhz",
            "transcripts": "https://drive.google.com/drive/folders/1bGjKB1GcSGIP1ZEpcLzUQPZiEUyDzTtT",
            "trimmed_audio": "https://drive.google.com/drive/folders/1J_ms3tdJipsRmcIeYGjnbV7jqd5GIerL",
        },
        "episodes": episodes_summary,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }




class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML_PATH.read_bytes())
            return

        elif path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            data = get_pipeline_status()
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            return

        elif path == "/api/transcript":
            query = urllib.parse.parse_qs(parsed.query)
            file_name = query.get("file", [""])[0]
            safe_name = os.path.basename(urllib.parse.unquote(file_name))
            md_file = PROCESSED_DIR / safe_name
            if safe_name and md_file.is_file():
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(md_file.read_bytes())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Transcript not found")
            return

        else:
            self.send_response(404)
            self.end_headers()


def run_dashboard_server(port: int = PORT):
    class _Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    host = os.getenv("DASHBOARD_HOST", "127.0.0.1")
    with _Server((host, port), DashboardRequestHandler) as httpd:
        logger.info(f"Dashboard server live at: http://{host}:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_dashboard_server()
