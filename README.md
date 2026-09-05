# 🎙️ Podflow: Podcast Audio & Prosody Pipeline

[![CI Pipeline](https://github.com/ssfinney/podcast-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ssfinney/podcast-pipeline/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Google GenAI](https://img.shields.io/badge/Gemini-3.1%20%2F%203.5-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous, end-to-end podcast processing and transcription engine designed to ingest audio/video RSS feeds, stream and extract bandwidth-optimized MP3s, generate verbatim speech-to-text with rich vocal prosody using Google Gemini, detect service boundaries to extract preaching-only audio cuts, synchronize formatted Markdown artifacts directly to Google Drive for **NotebookLM**, and provide real-time web telemetry.

---

## 🌟 Key Features

- **Direct HTTP Stream Ingestion:** Streams remote enclosures (including 350MB+ MP4 videos) directly via `ffmpeg`, transcoding on-the-fly to speech-optimized 24kHz mono 64kbps MP3s without storing massive video containers locally.
- **Multimodal Prosody Transcription:** Captures **vocal stress** (bolding), **pitch/tone inflection tags** (`[rising pitch]`, `[soft tone]`, `[whispering]`, `[laughing]`), and measured **cadence pauses** (`[pause: 1.5s]`) with speaker timecodes.
- **Intelligent Preaching Trimmer:** Uses LLM reasoning to identify church service order (silence $\to$ worship $\to$ announcements $\to$ giving $\to$ **preaching** $\to$ dismissal) and slices clean sermon cuts using `ffmpeg`.
- **Parallel Chunking & Crash Recovery:** Slices audio into 5-minute segments, transcribes concurrently via `ThreadPoolExecutor`, and maintains persistent disk caches (`.chunk_cache_<stem>/`) so interrupted jobs resume without repeating completed chunks.
- **Google Drive & NotebookLM Integration:** Automatically syncs Markdown transcripts to `Transcripts/` and sermon audio to `TrimmedAudio/` in Google Drive, maintaining a master `INDEX.md` and `index.csv`.
- **Real-Time Web Dashboard:** Built-in web UI on `http://localhost:8420` tracking active processing stages, chunk progress, KPI metrics, and transcript modal viewing.
- **Automated CI / QA Guardrails:** Multi-tier validation engine (`validator.py`) enforcing audio integrity, boundary sanity, prosody density, and hallucination loop prevention.

---

## 🏗️ Architecture & Data Flow

```
[RSS Feed XML]
       │
       ▼
[downloader.py] ──────────► Streams S3 video enclosure directly via ffmpeg
       │                    (Converts on-the-fly to 24kHz mono 64kbps MP3)
       ▼
 [RawAudio/*.mp3]
       │
       ▼
[transcriber.py] ─────────► Slices into 5-minute chunks with zero-reencode copy
       │                    Parallel Gemini API transcription (ThreadPoolExecutor)
       ▼                    Timestamp normalization & prosody capture
[ProcessedMD/*.md]
       │
       ▼
 [trimmer.py] ────────────► Analyzes transcript with Gemini to detect sermon boundaries
       │                    Extracts preaching-only cut using ffmpeg
       ▼
[TrimmedAudio/*.mp3]
       │
       ▼
[drive_sync.py] ──────────► Syncs Transcripts/ and TrimmedAudio/ to Google Drive
       │                    (via local CloudStorage mount or Drive API v3)
       ▼
 [pipeline.py] ───────────► Updates manifest.json ledger, regenerates INDEX.md / index.csv
       │
       ▼
[dashboard_server.py] ────► Serves real-time web UI and REST API on http://localhost:8420
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)
- `ffmpeg` (automatically provided via bundled `static-ffmpeg` if not in PATH)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ssfinney/podcast-pipeline.git
cd podcast-pipeline

# Sync dependencies with uv
uv sync
```

### 3. Configuration
Copy the template and set your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
GOOGLE_DRIVE_FOLDER_ID=your_drive_folder_id
```

### Free local transcription and prosody

The Gemini backend remains the default. To run transcription and acoustic
prosody analysis without an API quota:

```bash
uv sync --extra local-prosody
uv run python pipeline.py --backend local --local-model small.en --limit 1
```

The local backend uses faster-whisper word timestamps plus measured pitch,
intensity, duration, and silence. `small.en` with CPU INT8 is the conservative
default for Intel Macs. Use `--local-model turbo` on faster hardware.

Optional speaker diarization uses the free pyannote Community-1 model. Accept
its Hugging Face model terms, set `HF_TOKEN`, and enable it with
`LOCAL_DIARIZATION=true`. Install it separately with
`uv sync --extra local-prosody --extra diarization`.

To create three non-destructive previews beside the existing Gemini outputs:

```bash
uv run python prosody_compare.py --limit 3
```

Previews and `report.json` are written under `ProsodyPOC/`; the manifest,
Drive files, and existing transcripts are not modified.

---

## 🛠️ Usage

### Run End-to-End Pipeline
```bash
# Single-episode dry run
uv run python pipeline.py --dry-run

# Batch process next N episodes
uv run python pipeline.py --limit 10

# Continuous backfill across full archive
uv run python pipeline.py

# Specify a custom RSS feed URL
uv run python pipeline.py --feed "https://example.com/podcast.xml"
```

### Preaching Trimmer
```bash
# Trim preaching from all transcribed episodes
uv run python trimmer.py --all

# Trim specific audio file
uv run python trimmer.py --audio "RawAudio/2026-08-30 - Sermon.mp3"
```

### QA Guardrails & Validation
```bash
# Run full QA audit across all processed files
uv run python validator.py

# Run pytest unit test suite
uv run pytest -v
```

### Launch Telemetry Dashboard
```bash
uv run python dashboard_server.py
```
Open **[http://localhost:8420](http://localhost:8420)** in your browser for real-time stage tracking and transcript inspection.

---

## 📚 NotebookLM Ingestion Guide

1. Navigate to [NotebookLM](https://notebooklm.google.com/).
2. Create a new notebook (e.g. *Christ Chapel Sermons*).
3. Select **Add Source** $\to$ **Google Drive**.
4. Select the synced **`Christ Chapel Podcasts/Transcripts`** folder.
5. NotebookLM will index the transcripts with vocal prosody tags for theological Q&A, study guides, and multi-source synthesis.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
