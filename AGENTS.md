# Repository Guidelines

## Project Overview
Autonomous, end-to-end podcast processing and transcription pipeline designed to ingest church service video podcasts from an RSS feed, stream and transcode audio to bandwidth-optimized MP3s, generate verbatim speech-to-text transcripts with rich vocal prosody using Google Gemini, detect sermon boundaries to isolate the preaching audio, synchronize artifacts with Google Drive (for NotebookLM ingestion), and provide live web telemetry.

---

## Architecture & Data Flow

The pipeline operates as a staged, resilient data processing engine with persistent disk caching and an append-only JSON ledger:

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

### Key Modules & Responsibilities
- **`downloader.py`**: RSS feed ingestion via `feedparser`, filename sanitization, direct HTTP audio streaming via `ffmpeg` with `yt-dlp` fallback.
- **`transcriber.py`**: Audio chunking (5-minute slices), parallel thread pool execution, persistent chunk disk caching (`.chunk_cache_<stem>/`), timestamp re-anchoring, multi-model fallback cascade on rate limits, remote file cleanup, and YAML frontmatter Markdown generation.
- **`trimmer.py`**: LLM-driven liturgical boundary detection (worship $\to$ announcements $\to$ giving $\to$ **preaching** $\to$ dismissal) and sample-accurate ffmpeg audio slicing.
- **`drive_sync.py`**: Dual-mode Google Drive synchronization supporting local macOS CloudStorage mirror and Drive v3 REST API.
- **`pipeline.py`**: Master CLI orchestrator, progress ledger management (`manifest.json`), and automatic export of `INDEX.md` and `index.csv`.
- **`validator.py`**: CI / QA guardrails engine auditing audio integrity, boundary sanity, prosody density, and hallucination loops.
- **`dashboard_server.py`**: Lightweight threaded HTTP server serving real-time telemetry, stage progression, transcript inspection, and storage stats on port `8420`.

---

## Key Directories

```
.
├── RawAudio/         # Downloaded full-service 64kbps MP3s & temporary chunk caches
├── ProcessedMD/      # Finalized prosody-annotated Markdown transcripts with YAML frontmatter
├── TrimmedAudio/     # Preaching-only isolated MP3 sermon cuts
├── .venv/            # Local virtual environment managed by uv (Python 3.12)
└── tests/            # Test suite (or test_pipeline.py in root)
```

---

## Development Commands

Always use `uv` and `uv run` for executing scripts, tests, and tools within the project environment.

### Environment Setup
```bash
# Initialize / sync dependencies with uv
uv sync

# Set OpenSSL path if building native extensions on macOS Intel
export OPENSSL_DIR=/Users/stephenfinney/.local/Cellar/openssl@3/3.6.3
```

### Pipeline Execution
```bash
# Run dry run on the first episode end-to-end
uv run python pipeline.py --dry-run

# Run batch processing for the next N episodes
uv run python pipeline.py --limit 10

# Run continuous backfill across all 720 episodes
uv run python pipeline.py

# Force re-processing of already completed episodes
uv run python pipeline.py --force
```

### Trimming & Verification
```bash
# Run preaching trimmer across all transcribed episodes
uv run python trimmer.py --all

# Run QA guardrails audit across all processed episodes
uv run python validator.py
```

### Testing
```bash
# Run pytest test suite
uv run pytest -v

# Run single test file
uv run pytest test_pipeline.py
```

### Dashboard & Telemetry
```bash
# Launch the web dashboard server (http://localhost:8420)
uv run python dashboard_server.py
```

---

## Code Conventions & Common Patterns

### 1. Typing & Data Structures
- Always enable `from __future__ import annotations`.
- Use `@dataclass` for domain entities (`Episode`, `ProcessingRecord`, `SermonBoundary`, `CheckResult`, `EpisodeQAReport`) and provide `.to_dict()` serialization methods.
- Use explicit type annotations (`pathlib.Path`, `Optional[str]`, `List[dict]`, `Dict[str, Any]`).

### 2. Multi-Model Fallback & API Resilience
When communicating with Google GenAI models, use a cascading model list to handle transient 503 load spikes and 429 rate limits:
```python
DEFAULT_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    "gemini-3.1-flash-lite-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.1-pro-preview",
]
```
- On `404` or zero quota (`limit: 0`): Skip model immediately and advance to next fallback.
- On `503 UNAVAILABLE` or `429 RESOURCE_EXHAUSTED`: Apply short linear/exponential backoff before cycling models.
- Always delete remote Gemini files via `client.files.delete(name=uploaded_file.name)` in `finally` blocks.

### 3. Audio Processing & Chunking Patterns
- Initialize bundled ffmpeg binaries across all media scripts:
  ```python
  import static_ffmpeg
  static_ffmpeg.add_paths()
  ```
- Use `ffmpeg -ss ... -t ... -c copy` for fast zero-reencode chunk slicing.
- Re-anchor chunk-relative timestamps `[MM:SS]` to full-recording elapsed time `[HH:MM:SS]` using `adjust_chunk_timestamps(text, start_sec)`.
- Use atomic temporary files (`*.tmp.mp3`) before replacing destination files (`tmp_path.replace(dest_path)`).

### 4. Parallel Processing & Interruption Tolerance
- Use `concurrent.futures.ThreadPoolExecutor(max_workers=3)` for concurrent API chunk transcription.
- Cache intermediate chunk transcripts to `.chunk_cache_<stem>/chunk_XXX.txt` so interrupted runs resume instantly from disk without re-billing API tokens.

### 5. Google Drive Synchronization
- Support dual-mode sync: direct local file copy to `GOOGLE_DRIVE_LOCAL_PATH` (macOS CloudStorage FileProvider) with fallback/extension to Google Drive REST API v3.

---

## Important Files

| File | Purpose |
|---|---|
| `pipeline.py` | Central entry point and orchestrator for downloading, transcription, trimming, and syncing. |
| `downloader.py` | RSS ingestion and direct audio extraction. |
| `transcriber.py` | Speech-to-text with prosody tagging and chunk management. |
| `trimmer.py` | Sermon boundary detector and preaching MP3 extractor. |
| `drive_sync.py` | Google Drive cloud/local synchronization adapter. |
| `validator.py` | Automated CI / QA validation and guardrails engine. |
| `dashboard_server.py` | Real-time web UI and JSON telemetry server (port 8420). |
| `manifest.json` | Single source of truth JSON database tracking all processed episodes. |
| `INDEX.md` & `index.csv` | Running master catalog linking sermon dates, speakers, cuts, transcripts, and audio. |
| `MILESTONES.md` | Feature specification and taxonomy for milestone event historical extraction. |
| `pyproject.toml` | Build specification, tool configurations, and package dependencies. |
| `.env` | Local environment configuration (`GEMINI_API_KEY`, Drive folder IDs). |

---

## Runtime & Tooling Preferences

- **Python Runtime:** Python `3.12` pinned via `.python-version` and `pyproject.toml` (`requires-python = ">=3.12"`).
- **Package Manager:** `uv` (`uv run`, `uv add`, `uv sync`). Never use raw `pip` unless explicitly requested.
- **Media Binaries:** `static-ffmpeg` (v3.0+) handles cross-platform binary paths automatically; avoid invoking Homebrew heavy source builds.
- **Process Supervision:** Long-running jobs (`podcast-backfill`, `podcast-dashboard`) are managed through `hub` or persistent daemon wrappers.

---

## Testing & QA

- **Test Suite:** Run with `uv run pytest -v`.
- **QA Guardrails Audit:** Run with `uv run python validator.py`.
- **Validation Criteria:**
  - **Audio Integrity:** File size $\ge 2.0\text{ MB}$, valid MP3 header, duration $\ge 10\text{ minutes}$.
  - **Trimming Sanity:** Preaching start $> 60\text{s}$, sermon length between $10\text{m}$ and $90\text{m}$, sermon ratio between $20\%$ and $90\%$ of total service.
  - **Prosody Density:** Transcript contains $\ge 10$ bold stress markers (`**word**`), $\ge 5$ inline acoustic tags (`[pause: ...]`, `[rising pitch]`, `[whispering]`), and valid speaker timecodes (`### Speaker [HH:MM:SS]`).
  - **Hallucination Detection:** Flags consecutive repeated n-grams ($\ge 8$ repeats) to prevent runaway model loops.
  - **Cloud Mirroring:** Verifies presence and matching byte counts in Google Drive folders.
