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


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Christ Chapel Podcast Pipeline Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-glow: rgba(59, 130, 246, 0.35);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.3);
      --warning: #f59e0b;
      --danger: #ef4444;
      --accent: #8b5cf6;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: var(--bg);
      background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
      line-height: 1.5;
    }

    .container { max-width: 1380px; margin: 0 auto; }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-icon {
      width: 44px; height: 44px;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 22px; box-shadow: 0 0 20px var(--primary-glow);
    }
    .brand h1 { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; }
    .brand p { font-size: 12px; color: var(--text-muted); }

    .header-actions { display: flex; align-items: center; gap: 12px; }
    .badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 12px; border-radius: 20px;
      font-size: 12px; font-weight: 600;
      background: var(--card-bg); border: 1px solid var(--card-border);
    }
    .badge.active { color: var(--success); border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.1); }
    .pulse-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background-color: var(--success);
      box-shadow: 0 0 10px var(--success);
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }

    .btn {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 8px 16px; border-radius: 8px;
      font-size: 13px; font-weight: 600; cursor: pointer;
      text-decoration: none; transition: all 0.2s;
      border: 1px solid var(--card-border);
      background: var(--card-bg); color: var(--text);
    }
    .btn:hover { background: rgba(255, 255, 255, 0.1); }
    .btn-primary { background: var(--primary); border-color: var(--primary); color: #fff; }
    .btn-primary:hover { background: #2563eb; }

    /* Top Metric Cards */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px; margin-bottom: 24px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      backdrop-filter: blur(12px);
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .metric-title { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
    .metric-value { font-size: 28px; font-weight: 800; color: var(--text); }
    .metric-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

    /* Progress Bar */
    .progress-track {
      width: 100%; height: 8px; background: rgba(255, 255, 255, 0.08);
      border-radius: 4px; overflow: hidden; margin-top: 10px;
    }
    .progress-fill {
      height: 100%; background: linear-gradient(90deg, var(--primary), var(--accent));
      width: 0%; transition: width 0.5s ease;
    }

    /* Active Processing Spotlight Card */
    .active-card {
      background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
      border: 1px solid rgba(59, 130, 246, 0.3);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }
    .active-card::before {
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, var(--primary), var(--accent), var(--success));
    }
    .active-header {
      display: flex; justify-content: space-between; align-items: flex-start;
      margin-bottom: 16px;
    }
    .active-tag {
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      padding: 4px 10px; border-radius: 6px;
      background: rgba(59, 130, 246, 0.15); color: var(--primary);
      border: 1px solid rgba(59, 130, 246, 0.3);
      display: inline-flex; align-items: center; gap: 6px;
    }
    .active-title { font-size: 18px; font-weight: 700; margin-top: 6px; }
    .active-details { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

    /* Stepper */
    .stepper {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 12px; margin-top: 20px;
    }
    .step {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid var(--card-border);
      border-radius: 12px; padding: 12px;
      display: flex; align-items: center; gap: 10px;
      transition: all 0.3s;
    }
    .step.active {
      border-color: var(--primary);
      background: rgba(59, 130, 246, 0.12);
      box-shadow: 0 0 15px var(--primary-glow);
    }
    .step.done {
      border-color: var(--success);
      background: rgba(16, 185, 129, 0.08);
    }
    .step-num {
      width: 28px; height: 28px; border-radius: 50%;
      background: rgba(255, 255, 255, 0.1);
      display: flex; align-items: center; justify-content: center;
      font-size: 12px; font-weight: 700;
    }
    .step.active .step-num { background: var(--primary); color: #fff; }
    .step.done .step-num { background: var(--success); color: #fff; }
    .step-text { font-size: 12px; font-weight: 600; }
    .step-sub { font-size: 10px; color: var(--text-muted); }

    /* Table Section */
    .section-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 16px; flex-wrap: wrap; gap: 12px;
    }
    .section-title { font-size: 18px; font-weight: 700; }
    .search-box {
      background: var(--card-bg); border: 1px solid var(--card-border);
      border-radius: 8px; padding: 8px 14px;
      color: var(--text); font-size: 13px; outline: none;
      min-width: 260px;
    }
    .search-box:focus { border-color: var(--primary); }

    .filter-tabs { display: flex; gap: 8px; }
    .tab {
      padding: 6px 14px; border-radius: 8px;
      font-size: 12px; font-weight: 600; cursor: pointer;
      background: var(--card-bg); border: 1px solid var(--card-border);
      color: var(--text-muted);
    }
    .tab.active { background: var(--primary); color: #fff; border-color: var(--primary); }

    .table-container {
      overflow-x: auto;
      border: 1px solid var(--card-border);
      border-radius: 14px;
      background: var(--card-bg);
    }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th {
      background: rgba(255, 255, 255, 0.03);
      padding: 14px 16px; font-weight: 600; font-size: 11px;
      text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px;
      border-bottom: 1px solid var(--card-border);
    }
    td { padding: 14px 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); vertical-align: middle; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    .status-badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 8px; border-radius: 6px;
      font-size: 11px; font-weight: 700;
    }
    .status-badge.COMPLETED { background: rgba(16, 185, 129, 0.15); color: var(--success); }
    .status-badge.IN_PROGRESS { background: rgba(59, 130, 246, 0.15); color: var(--primary); }
    .status-badge.QUEUED { background: rgba(255, 255, 255, 0.06); color: var(--text-muted); }
    .status-badge.FAILED { background: rgba(239, 68, 68, 0.15); color: var(--danger); }

    .mono { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

    /* Modal */
    .modal-overlay {
      position: fixed; inset: 0; background: rgba(0, 0, 0, 0.75);
      display: none; align-items: center; justify-content: center;
      padding: 24px; z-index: 1000; backdrop-filter: blur(8px);
    }
    .modal {
      background: #0f172a; border: 1px solid var(--card-border);
      border-radius: 16px; width: 100%; max-width: 900px; max-height: 85vh;
      display: flex; flex-direction: column; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    .modal-header {
      padding: 20px 24px; border-bottom: 1px solid var(--card-border);
      display: flex; justify-content: space-between; align-items: center;
    }
    .modal-title { font-size: 16px; font-weight: 700; }
    .modal-body {
      padding: 24px; overflow-y: auto; font-size: 13px; line-height: 1.7;
      white-space: pre-wrap; font-family: 'Plus Jakarta Sans', sans-serif;
      color: #cbd5e1;
    }
    .modal-close {
      background: transparent; border: none; color: var(--text-muted);
      font-size: 20px; cursor: pointer; padding: 4px 8px;
    }
    .modal-close:hover { color: var(--text); }
    .sr-only {
      position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
      overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
    }
    .btn:focus-visible, .tab:focus-visible, .search-box:focus-visible, .modal-close:focus-visible {
      outline: 3px solid #93c5fd; outline-offset: 2px;
    }
    .badge.error { color: #fecaca; border-color: rgba(239, 68, 68, 0.5); background: rgba(239, 68, 68, 0.12); }
    .active-card.idle { border-color: var(--card-border); }
    .active-card.idle::before { background: var(--text-muted); }
    .refresh-state { min-height: 18px; margin-top: 8px; color: var(--text-muted); font-size: 11px; text-align: right; }
    table { min-width: 900px; }

    @media (max-width: 760px) {
      body { padding: 14px; }
      header { align-items: flex-start; flex-direction: column; gap: 16px; }
      .header-actions { width: 100%; flex-wrap: wrap; }
      .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .stepper { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .section-header > div:last-child, .filter-tabs, .search-box { width: 100%; }
      .filter-tabs { overflow-x: auto; padding-bottom: 4px; }
      .tab { flex: 0 0 auto; }
      .search-box { min-width: 0; }
      .modal-overlay { padding: 10px; }
      .modal-header, .modal-body { padding: 16px; }
    }
    @media (max-width: 460px) {
      .metrics-grid, .stepper { grid-template-columns: 1fr; }
      .card, .active-card { padding: 16px; }
      .brand h1 { font-size: 18px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <div class="brand-icon" aria-hidden="true">🎙️</div>
        <div>
          <h1>Christ Chapel Podcast Pipeline</h1>
          <p>Autonomous Audio Extraction, Prosody Transcription & Drive Sync</p>
        </div>
      </div>
      <div class="header-actions">
        <div class="badge" id="processBadge" role="status" aria-live="polite">
          <div class="pulse-dot" aria-hidden="true"></div>
          <span id="processStatusText">Connecting…</span>
        </div>
        <a href="https://drive.google.com/drive/folders/1emYUaJPU0_5qoGgNphNW51A12wwrUWhz" target="_blank" rel="noopener noreferrer" class="btn btn-primary">
          📂 Google Drive
        </a>
      </div>
    </header>

    <!-- Top Metrics -->
    <div class="metrics-grid">
      <div class="card">
        <div class="metric-title">Overall Progress</div>
        <div class="metric-value"><span id="processedCount">0</span> <span style="font-size: 16px; color: var(--text-muted); font-weight: 500;">/ <span id="totalCount">720</span></span></div>
        <div class="progress-track" id="overallProgressTrack" role="progressbar" aria-label="Overall catalog progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div class="progress-fill" id="overallProgressBar"></div></div>
        <div class="metric-sub" id="percentText">0% Cataloged</div>
      </div>
      <div class="card">
        <div class="metric-title">Preaching Extracted</div>
        <div class="metric-value" id="preachingTime">0h 0m</div>
        <div class="metric-sub">Isolated sermon cuts</div>
      </div>
      <div class="card">
        <div class="metric-title">Drive Mirror</div>
        <div class="metric-value" style="font-size: 20px;">Configured</div>
        <div class="metric-sub">Availability checked during each sync</div>
      </div>
      <div class="card">
        <div class="metric-title">Storage Processed</div>
        <div class="metric-value" id="storageTotal">0 MB</div>
        <div class="metric-sub"><span id="storageRaw">0</span> MB raw · <span id="storageTrim">0</span> MB trimmed</div>
      </div>
    </div>

    <!-- Active Spotlight -->
    <div class="active-card idle" id="activeCard" role="status" aria-live="polite">
      <div class="active-header">
        <div>
          <div class="active-tag" id="activeTag">
            <div class="pulse-dot" style="background: var(--primary); box-shadow: 0 0 10px var(--primary);"></div>
            <span id="activeStageName">Processing</span>
          </div>
          <div class="active-title" id="activeTitle">Loading current episode...</div>
          <div class="active-details" id="activeDetails">Initializing stream...</div>
        </div>
      </div>

      <div class="stepper">
        <div class="step" id="step1">
          <div class="step-num">1</div>
          <div>
            <div class="step-text">Download Audio</div>
            <div class="step-sub">S3 Stream to MP3</div>
          </div>
        </div>
        <div class="step" id="step2">
          <div class="step-num">2</div>
          <div>
            <div class="step-text">Gemini Prosody</div>
            <div class="step-sub">Vocal Stress & Cadence</div>
          </div>
        </div>
        <div class="step" id="step3">
          <div class="step-num">3</div>
          <div>
            <div class="step-text">Trim Sermon</div>
            <div class="step-sub">Extract Preaching Cut</div>
          </div>
        </div>
        <div class="step" id="step4">
          <div class="step-num">4</div>
          <div>
            <div class="step-text">Google Drive Sync</div>
            <div class="step-sub">Mirror to Cloud</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Master Table -->
    <div class="section-header">
      <div class="section-title">Episode Catalog & Processing Queue</div>
      <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <div class="filter-tabs" role="group" aria-label="Filter episodes by status">
          <button type="button" class="tab active" aria-pressed="true" onclick="setFilter('ALL', this)">All (<span id="tabCountAll">0</span>)</button>
          <button type="button" class="tab" aria-pressed="false" onclick="setFilter('COMPLETED', this)">Completed (<span id="tabCountDone">0</span>)</button>
          <button type="button" class="tab" aria-pressed="false" onclick="setFilter('IN_PROGRESS', this)">Active (<span id="tabCountActive">0</span>)</button>
          <button type="button" class="tab" aria-pressed="false" onclick="setFilter('QUEUED', this)">Queued (<span id="tabCountQueued">0</span>)</button>
        </div>
        <label for="searchInput" class="sr-only">Search episodes by title, date, or speaker</label>
        <input type="search" class="search-box" id="searchInput" placeholder="Search title, date, or speaker" autocomplete="off" oninput="renderTable()">
      </div>
      <div class="refresh-state" id="refreshState" role="status" aria-live="polite">Connecting to dashboard…</div>
    </div>

    <div class="table-container">
      <table>
        <caption class="sr-only">Podcast episodes and processing status</caption>
        <thead>
          <tr>
            <th style="width: 50px;">#</th>
            <th style="width: 100px;">Date</th>
            <th>Sermon Title</th>
            <th style="width: 140px;">Speaker</th>
            <th style="width: 150px;">Preaching Cut</th>
            <th style="width: 110px;">Status</th>
            <th style="width: 140px;">Actions</th>
          </tr>
        </thead>
        <tbody id="episodesTableBody" aria-live="polite">
          <tr><td colspan="7" style="text-align: center; padding: 40px; color: var(--text-muted);">Loading episode catalog…</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Transcript Modal -->
  <div class="modal-overlay" id="transcriptModal" aria-hidden="true">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modalTitle" tabindex="-1">
      <div class="modal-header">
        <div class="modal-title" id="modalTitle">Transcript Preview</div>
        <button type="button" class="modal-close" id="modalClose" aria-label="Close transcript preview" onclick="closeModal()">&times;</button>
      </div>
      <div class="modal-body" id="modalBody">Loading…</div>
    </div>
  </div>

  <script>
    let allEpisodes = [];
    let currentFilter = 'ALL';
    let refreshInFlight = false;
    let lastFocusedElement = null;

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      })[character]);
    }

    async function fetchStatus() {
      if (refreshInFlight) return;
      refreshInFlight = true;
      const refreshState = document.getElementById('refreshState');
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        updateDashboard(data);
        refreshState.textContent = `Updated ${new Date().toLocaleTimeString()}`;
      } catch (err) {
        console.error('Failed to fetch status:', err);
        refreshState.textContent = `Dashboard update failed: ${err.message}. Retrying automatically.`;
        const badge = document.getElementById('processBadge');
        badge.className = 'badge error';
        document.getElementById('processStatusText').textContent = 'Connection problem';
      } finally {
        refreshInFlight = false;
      }
    }

    function updateDashboard(data) {
      document.getElementById('processedCount').textContent = data.processed_count;
      document.getElementById('totalCount').textContent = data.total_episodes;
      document.getElementById('percentText').textContent = `${data.percent_complete}% Cataloged`;
      document.getElementById('overallProgressBar').style.width = `${data.percent_complete}%`;
      document.getElementById('overallProgressTrack').setAttribute('aria-valuenow', data.percent_complete);
      document.getElementById('preachingTime').textContent = data.total_preaching_time;

      if (data.storage) {
        document.getElementById('storageTotal').textContent = `${data.storage.total_mb} MB`;
        document.getElementById('storageRaw').textContent = data.storage.raw_mb;
        document.getElementById('storageTrim').textContent = data.storage.trimmed_mb;
      }

      const act = data.active_job;
      const activeCard = document.getElementById('activeCard');
      const processBadge = document.getElementById('processBadge');
      const processText = document.getElementById('processStatusText');
      const steps = [
        document.getElementById('step1'),
        document.getElementById('step2'),
        document.getElementById('step3'),
        document.getElementById('step4')
      ];
      steps.forEach(step => {
        step.className = 'step';
        step.removeAttribute('aria-current');
      });

      if (act && act.is_active) {
        activeCard.classList.remove('idle');
        processBadge.className = 'badge active';
        processText.textContent = 'File operation active';
        document.getElementById('activeStageName').textContent = act.stage_label;
        document.getElementById('activeTitle').textContent = act.episode_title || 'Active episode';
        document.getElementById('activeDetails').textContent = act.details || 'Processing';

        const stageIndex = {
          DOWNLOADING: 0,
          TRANSCRIBING: 1,
          TRIMMING: 2,
          SYNCING: 3
        }[act.stage];
        if (stageIndex !== undefined) {
          steps.forEach((step, index) => {
            if (index < stageIndex) step.className = 'step done';
            if (index === stageIndex) {
              step.className = 'step active';
              step.setAttribute('aria-current', 'step');
            }
          });
        }
      } else {
        activeCard.classList.add('idle');
        processBadge.className = 'badge';
        processText.textContent = 'No active file operation';
        document.getElementById('activeStageName').textContent = 'Idle';
        document.getElementById('activeTitle').textContent = 'No file operation detected';
        document.getElementById('activeDetails').textContent =
          'The backfill may be scanning completed episodes between observable stages.';
      }

      allEpisodes = data.episodes || [];
      updateTabCounts();
      renderTable();
    }

    function updateTabCounts() {
      const done = allEpisodes.filter(episode => episode.status === 'COMPLETED').length;
      const active = allEpisodes.filter(episode => episode.status === 'IN_PROGRESS').length;
      const queued = allEpisodes.filter(episode => episode.status === 'QUEUED').length;
      document.getElementById('tabCountAll').textContent = allEpisodes.length;
      document.getElementById('tabCountDone').textContent = done;
      document.getElementById('tabCountActive').textContent = active;
      document.getElementById('tabCountQueued').textContent = queued;
    }

    function setFilter(filter, button) {
      currentFilter = filter;
      document.querySelectorAll('.tab').forEach(tab => {
        const selected = tab === button;
        tab.classList.toggle('active', selected);
        tab.setAttribute('aria-pressed', String(selected));
      });
      renderTable();
    }

    function renderTable() {
      const query = document.getElementById('searchInput').value.trim().toLowerCase();
      const tbody = document.getElementById('episodesTableBody');
      const filtered = allEpisodes.filter(episode => {
        if (currentFilter !== 'ALL' && episode.status !== currentFilter) return false;
        if (!query) return true;
        return [episode.title, episode.date, episode.author]
          .some(value => String(value || '').toLowerCase().includes(query));
      });

      if (!filtered.length) {
        const message = allEpisodes.length
          ? 'No episodes match the current search and filter.'
          : 'No episodes are available yet.';
        tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 40px; color: var(--text-muted);">${message}</td></tr>`;
        return;
      }

      const statusLabels = {
        COMPLETED: 'Complete',
        IN_PROGRESS: 'In progress',
        QUEUED: 'Queued',
        FAILED: 'Failed'
      };
      tbody.innerHTML = filtered.map(episode => {
        const status = Object.hasOwn(statusLabels, episode.status) ? episode.status : 'QUEUED';
        const cut = episode.preaching_start && episode.preaching_end
          ? `<span class="mono" style="color: #93c5fd;">${escapeHtml(episode.preaching_start)} → ${escapeHtml(episode.preaching_end)}</span>`
          : '<span style="color: var(--text-muted); font-size: 12px;">Not extracted</span>';
        const viewButton = episode.md_file
          ? `<button type="button" class="btn read-transcript" style="padding: 4px 8px; font-size: 11px;" data-file="${escapeHtml(encodeURIComponent(episode.md_file))}" data-title="${escapeHtml(encodeURIComponent(episode.title || 'Transcript'))}">Read transcript</button>`
          : '<span style="color: var(--text-muted); font-size: 12px;">No transcript</span>';
        const driveLink = episode.drive_link
          ? `<a href="${escapeHtml(episode.drive_link)}" target="_blank" rel="noopener noreferrer" class="btn" style="padding: 4px 8px; font-size: 11px;">Open Drive</a>`
          : '';

        return `
          <tr>
            <td class="mono" style="color: var(--text-muted);">${escapeHtml(episode.index)}</td>
            <td class="mono" style="font-weight: 600;">${escapeHtml(episode.date)}</td>
            <td style="font-weight: 600;">${escapeHtml(episode.title)}</td>
            <td style="color: var(--text-muted);">${escapeHtml(episode.author || 'Unknown')}</td>
            <td>${cut}</td>
            <td><span class="status-badge ${status}">${statusLabels[status]}</span></td>
            <td><div style="display: flex; gap: 6px; flex-wrap: wrap;">${viewButton}${driveLink}</div></td>
          </tr>`;
      }).join('');

      tbody.querySelectorAll('.read-transcript').forEach(button => {
        button.addEventListener('click', () => viewTranscript(button.dataset.file, button.dataset.title));
      });
    }

    async function viewTranscript(mdFile, title) {
      const modal = document.getElementById('transcriptModal');
      lastFocusedElement = document.activeElement;
      document.getElementById('modalTitle').textContent = decodeURIComponent(title);
      document.getElementById('modalBody').textContent = 'Loading transcript…';
      modal.style.display = 'flex';
      modal.setAttribute('aria-hidden', 'false');
      document.body.style.overflow = 'hidden';
      document.getElementById('modalClose').focus();

      try {
        const res = await fetch(`/api/transcript?file=${mdFile}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        document.getElementById('modalBody').textContent = await res.text();
      } catch (err) {
        document.getElementById('modalBody').textContent =
          `Transcript could not be loaded (${err.message}). Close this dialog and try again.`;
      }
    }

    function closeModal() {
      const modal = document.getElementById('transcriptModal');
      modal.style.display = 'none';
      modal.setAttribute('aria-hidden', 'true');
      document.body.style.overflow = '';
      if (lastFocusedElement) lastFocusedElement.focus();
    }

    document.getElementById('transcriptModal').addEventListener('click', event => {
      if (event.target.id === 'transcriptModal') closeModal();
    });
    document.addEventListener('keydown', event => {
      const modalOpen = document.getElementById('transcriptModal').getAttribute('aria-hidden') === 'false';
      if (event.key === 'Escape' && modalOpen) closeModal();
      if (event.key === 'Tab' && modalOpen) {
        event.preventDefault();
        document.getElementById('modalClose').focus();
      }
    });
    setInterval(() => {
      if (!document.hidden) fetchStatus();
    }, 5000);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden) fetchStatus();
    });
    fetchStatus();
  </script>
</body>
</html>
"""


class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html_file = BASE_DIR / "dashboard.html"
            content = html_file.read_bytes() if html_file.exists() else DASHBOARD_HTML.encode("utf-8")
            self.wfile.write(content)
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
