"""Transcriber module using Google GenAI SDK for high-throughput, resilient prosody-aware audio transcription."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import static_ffmpeg

from downloader import Episode

# Initialize ffmpeg paths
static_ffmpeg.add_paths()

load_dotenv()
logger = logging.getLogger(__name__)

# Model preference order: favor the available 3.7 Flash model, then stable
# Flash fallbacks before the lower-cost lite models and Pro preview.
DEFAULT_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
]

PROSODY_TRANSCRIPTION_PROMPT = """Transcribe the audio verbatim while capturing full vocal prosody, emotion, and cadence:
* **Vocal Stress:** Wrap strongly emphasized or punchy words in **bold**.
* **Pitch & Inflection:** Insert bracketed inline tags for noticeable pitch shifts or tone (e.g., [rising pitch / skeptical], [pitch drop / definitive], [whispering], [laughing]).
* **Pauses & Cadence:** Annotate deliberate silences or hesitation with duration (e.g., [pause: 1.5s]).
* **Structure:** Format with speaker labels and timestamps: ### Speaker Name [HH:MM:SS]."""

PROSODY_CHUNK_PROMPT = """Transcribe this audio segment verbatim while capturing full vocal prosody, emotion, and cadence:
* **Vocal Stress:** Wrap strongly emphasized or punchy words in **bold**.
* **Pitch & Inflection:** Insert bracketed inline tags for noticeable pitch shifts or tone (e.g., [rising pitch / skeptical], [pitch drop / definitive], [whispering], [laughing]).
* **Pauses & Cadence:** Annotate deliberate silences or hesitation with duration (e.g., [pause: 1.5s]).
* **Structure:** Format with speaker labels and timestamps: ### Speaker Name [HH:MM:SS]."""

# Global lock to serialize audio upload bursts and prevent upstream bandwidth congestion
_UPLOAD_LOCK = threading.Lock()


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def adjust_chunk_timestamps(chunk_text: str, start_sec: float) -> str:
    """Adjust chunk-relative timestamps [HH:MM:SS] to absolute recording time."""
    if start_sec <= 0:
        return chunk_text

    def replace_ts(match):
        ts_str = match.group(1)
        parts = ts_str.split(":")
        try:
            if len(parts) == 3:
                s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                s = int(parts[0]) * 60 + int(parts[1])
            else:
                return match.group(0)

            abs_s = start_sec + s
            h = int(abs_s // 3600)
            m = int((abs_s % 3600) // 60)
            sec = int(abs_s % 60)
            return f"[{h:02d}:{m:02d}:{sec:02d}]"
        except Exception:
            return match.group(0)

    pattern = r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]"
    return re.sub(pattern, replace_ts, chunk_text)


def get_audio_duration(audio_path: Path) -> float:
    """Get the duration of an audio file in seconds via ffprobe."""
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return 0.0
    try:
        cmd = [ffprobe_bin, "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(res.stdout)
        return float(data.get("format", {}).get("duration", 0.0))
    except Exception as e:
        logger.warning(f"Could not determine audio duration for {audio_path}: {e}")
        return 0.0


class ProsodyTranscriber:
    """Manages audio upload, Gemini prosody transcription, parallel chunking, caching, and file cleanup."""

    def __init__(self, api_key: Optional[str] = None, preferred_model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment or .env file.")
        self.client = genai.Client(api_key=self.api_key)
        self.preferred_model = preferred_model or os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

    def _extract_response_text(self, response) -> str:
        """Extract text cleanly from Gemini response object."""
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        if response.candidates:
            for cand in response.candidates:
                if cand.content and cand.content.parts:
                    texts = []
                    for part in cand.content.parts:
                        if hasattr(part, "text") and part.text:
                            texts.append(part.text)
                    if texts:
                        return "\n".join(texts).strip()
        return ""

    def _transcribe_single_file(
        self,
        file_path: Path,
        prompt: str,
        preferred_model: Optional[str] = None,
        max_model_retries: int = 5,
    ) -> str:
        """
        Generate prosody transcription for an audio file/chunk with serialized upload and parallel inference.
        """
        uploaded_file = None
        models_to_try = []
        pref = preferred_model or self.preferred_model
        if pref:
            models_to_try.append(pref)
        for m in DEFAULT_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

        last_error = None
        try:
            for model_name in models_to_try:
                for attempt in range(1, max_model_retries + 1):
                    try:
                        # Serialize file uploads to avoid bandwidth congestion
                        if not uploaded_file:
                            with _UPLOAD_LOCK:
                                logger.info(f"[{file_path.stem}] Uploading to Gemini Files API ({file_path.stat().st_size / (1024*1024):.2f} MB)...")
                                uploaded_file = self.client.files.upload(file=str(file_path))

                            # Polling is executed outside the upload lock so other workers are not blocked!
                            poll_start = time.time()
                            while uploaded_file.state.name == "PROCESSING":
                                if time.time() - poll_start > 180:
                                    raise TimeoutError(f"Timeout waiting for audio file {uploaded_file.name} to process.")
                                time.sleep(1.0)
                                uploaded_file = self.client.files.get(name=uploaded_file.name)

                            if uploaded_file.state.name != "ACTIVE":
                                raise RuntimeError(f"Audio file failed processing on Gemini: state={uploaded_file.state.name}")

                        logger.info(f"[{file_path.stem}] Transcribing via '{model_name}' (attempt {attempt}/{max_model_retries})...")
                        t0 = time.time()
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=[uploaded_file, prompt],
                            config=types.GenerateContentConfig(
                                temperature=0.2,
                                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            ),
                        )
                        text = self._extract_response_text(response)
                        if text:
                            elapsed = time.time() - t0
                            logger.info(f"[{file_path.stem}] Success with '{model_name}' ({len(text)} chars in {elapsed:.1f}s).")
                            return text
                        logger.warning(f"[{file_path.stem}] Empty response text from {model_name}.")

                    except Exception as e:
                        err_str = str(e)
                        last_error = e
                        if "limit: 0" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                            logger.warning(f"[{file_path.stem}] '{model_name}' has 0 quota or not found. Skipping model.")
                            break
                        elif "503" in err_str or "UNAVAILABLE" in err_str or "timeout" in err_str.lower() or "handshake" in err_str.lower() or "ssl" in err_str.lower():
                            wait_s = min(60, attempt * 6)
                            logger.warning(f"[{file_path.stem}] Network/API glitch ({err_str[:60]}...). Retrying in {wait_s}s...")
                            time.sleep(wait_s)
                            if "ssl" in err_str.lower() or "handshake" in err_str.lower():
                                uploaded_file = None
                        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                            wait_s = min(60, attempt * 8)
                            logger.warning(f"[{file_path.stem}] '{model_name}' rate limited (429). Retrying in {wait_s}s...")
                            time.sleep(wait_s)
                        else:
                            time.sleep(3)

            raise RuntimeError(f"All models failed for {file_path.name}: {last_error}")

        finally:
            if uploaded_file and uploaded_file.name:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                except Exception as del_err:
                    logger.warning(f"Failed to delete remote file {uploaded_file.name}: {del_err}")

    def _process_chunk_worker(
        self,
        idx: int,
        total_chunks: int,
        start_sec: float,
        seg_len: float,
        audio_path: Path,
        cache_dir: Path,
        ffmpeg_bin: str,
        model_name: str,
    ) -> tuple[int, str]:
        """Worker function to process and transcribe a single chunk with cleanup."""
        chunk_txt_file = cache_dir / f"chunk_{idx:03d}.txt"
        chunk_audio_file = cache_dir / f"chunk_{idx:03d}.mp3"

        if chunk_txt_file.exists() and chunk_txt_file.stat().st_size > 20:
            cached_text = chunk_txt_file.read_text(encoding="utf-8")
            logger.info(f"Chunk {idx + 1}/{total_chunks} loaded from cache ({len(cached_text)} chars).")
            return idx, cached_text

        logger.info(f"Extracting chunk {idx + 1}/{total_chunks} [{format_timestamp(start_sec)} -> {format_timestamp(start_sec + seg_len)}]...")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            str(audio_path),
            "-t",
            str(seg_len),
            "-c",
            "copy",
            str(chunk_audio_file),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)

        try:
            prompt = PROSODY_CHUNK_PROMPT
            logger.info(f"Transcribing chunk {idx + 1}/{total_chunks} via {model_name}...")
            raw_chunk_text = self._transcribe_single_file(chunk_audio_file, prompt, preferred_model=model_name)
            chunk_txt_file.write_text(raw_chunk_text, encoding="utf-8")
            return idx, raw_chunk_text
        finally:
            if chunk_audio_file.exists():
                chunk_audio_file.unlink()

    def transcribe_audio_file(
        self,
        audio_path: Path,
        episode: Optional[Episode] = None,
        chunk_duration_sec: int = 480,  # 8 minutes per chunk
        max_workers: int = 2,  # 2 workers for peak connection & rate limit stability
    ) -> str:
        """
        Transcribe full audio file using parallel chunks with disk caching and single-pass timestamp normalization.
        """
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise FileNotFoundError(f"Audio file not found or empty: {audio_path}")

        duration = get_audio_duration(audio_path)
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        logger.info(f"Audio file: '{audio_path.name}' ({file_size_mb:.2f} MB, {format_timestamp(duration)})")

        if duration > 0 and duration <= chunk_duration_sec:
            logger.info("Audio duration is <= 8 minutes, transcribing in single pass.")
            return self._transcribe_single_file(audio_path, PROSODY_TRANSCRIPTION_PROMPT)

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for audio chunking.")

        if duration <= 0:
            duration = (audio_path.stat().st_size * 8) / 64000

        num_chunks = math.ceil(duration / chunk_duration_sec)
        logger.info(f"Processing audio in {num_chunks} chunks ({chunk_duration_sec//60}m each) with {max_workers} parallel workers.")

        cache_dir = audio_path.parent / f".chunk_cache_{audio_path.stem}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        chunk_results: Dict[int, str] = {}

        # 1. Check existing cached chunk texts
        for idx in range(num_chunks):
            chunk_txt = cache_dir / f"chunk_{idx:03d}.txt"
            if chunk_txt.exists() and chunk_txt.stat().st_size > 20:
                chunk_results[idx] = chunk_txt.read_text(encoding="utf-8")

        # 2. Collect chunks needing processing
        needed_chunks = []
        for idx in range(num_chunks):
            if idx not in chunk_results:
                start_sec = idx * chunk_duration_sec
                seg_len = min(chunk_duration_sec, duration - start_sec)
                needed_chunks.append((idx, start_sec, seg_len, self.preferred_model))

        if needed_chunks:
            logger.info(f"{len(chunk_results)}/{num_chunks} chunks already cached, {len(needed_chunks)} chunks to process.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_chunk_worker,
                        idx=idx,
                        total_chunks=num_chunks,
                        start_sec=start_sec,
                        seg_len=seg_len,
                        audio_path=audio_path,
                        cache_dir=cache_dir,
                        ffmpeg_bin=ffmpeg_bin,
                        model_name=model,
                    ): idx
                    for idx, start_sec, seg_len, model in needed_chunks
                }

                for future in concurrent.futures.as_completed(futures):
                    try:
                        idx, text = future.result(timeout=300)
                        chunk_results[idx] = text
                    except Exception as e:
                        logger.warning(f"Chunk transcription error/timeout: {e}")
                        raise
        else:
            logger.info(f"All {num_chunks} chunks loaded from cache.")

        # 3. Assemble and normalize timestamps in a single pass
        full_transcripts = []
        for i in range(num_chunks):
            raw_text = chunk_results.get(i, "")
            start_sec = i * chunk_duration_sec
            adjusted_text = adjust_chunk_timestamps(raw_text, start_sec)
            full_transcripts.append(adjusted_text)

        full_transcript = "\n\n".join(full_transcripts)

        # Cleanup cache on complete success
        shutil.rmtree(cache_dir, ignore_errors=True)
        return full_transcript

    def create_formatted_markdown(
        self,
        raw_transcript: str,
        episode: Episode,
        audio_path: Path,
        output_dir: Path,
    ) -> Path:
        """Format transcript with frontmatter and metadata header, then save to Markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / episode.md_filename

        header = f"""---
title: "{episode.title.replace('"', "'")}"
date: "{episode.date_iso}"
author: "{episode.author.replace('"', "'")}"
duration: "{episode.duration}"
media_url: "{episode.media_url}"
audio_file: "{audio_path.name}"
guid: "{episode.guid}"
---

# {episode.title}

- **Date:** {episode.pub_date} ({episode.date_iso})
- **Author/Speaker:** {episode.author}
- **Duration:** {episode.duration}
- **Source Enclosure:** [{episode.media_url}]({episode.media_url})

## Summary
{episode.summary or episode.description or 'No summary provided.'}

---

## Verbatim Prosody Transcript

{raw_transcript.strip()}
"""
        out_path.write_text(header, encoding="utf-8")
        logger.info(f"Saved processed Markdown to: {out_path}")
        return out_path
