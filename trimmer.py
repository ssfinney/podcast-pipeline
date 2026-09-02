"""Audio trimmer module to detect and extract only the preaching/sermon portion of a podcast/service."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
import static_ffmpeg

# Initialize static ffmpeg
static_ffmpeg.add_paths()

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_TRIMMED_DIR = Path(__file__).parent / "TrimmedAudio"
DEFAULT_PROCESSED_DIR = Path(__file__).parent / "ProcessedMD"
DEFAULT_AUDIO_DIR = Path(__file__).parent / "RawAudio"


@dataclass
class SermonBoundary:
    start_timestamp: str  # "HH:MM:SS"
    end_timestamp: str  # "HH:MM:SS"
    start_seconds: float
    end_seconds: float
    speaker_name: str
    first_words: str
    last_words: str
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_timestamp_to_seconds(ts: str) -> float:
    """Parse 'HH:MM:SS' or 'MM:SS' into float seconds."""
    if not ts:
        return 0.0
    ts = re.sub(r"[^\d:]", "", ts.strip())
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except (ValueError, TypeError):
        pass
    return 0.0


def format_seconds_to_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def clean_text_for_boundary_detection(text: str) -> str:
    """
    Strip heavy inline acoustic tags while retaining full text context for accurate boundary detection.
    Gemini 3.7 / 3.5 / 3.1 Flash natively handles 1M+ token contexts (~4MB text).
    """
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    clean = re.sub(
        r"\[(pause:[^\]]+|singing|music|rising pitch[^\]]*|pitch drop[^\]]*|whispering|laughing|cheering|applause)\]",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()

    # Retain full transcript unless excessively huge (> 400k chars)
    if len(clean) > 400000:
        clean = clean[:280000] + "\n\n... [MIDDLE TEACHING PORTION] ...\n\n" + clean[-100000:]
    return clean


class PreachingTrimmer:
    """Detects preaching boundaries and extracts sermon-only audio files."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self.client = genai.Client(api_key=self.api_key)

    def detect_boundaries_from_transcript(
        self,
        transcript_text: str,
        total_duration_sec: float = 0.0,
    ) -> SermonBoundary:
        """
        Analyze transcript with Gemini to identify the start and end of the preaching message.
        """
        optimized_transcript = clean_text_for_boundary_detection(transcript_text)

        prompt = f"""You are an expert audio editor analyzing a Christian church service / podcast transcript.
Identify the exact START timestamp and END timestamp where the MAIN PREACHING / SERMON message begins and ends.

Order of a typical service:
1. Opening silence or background music (0:00 to ~0:05)
2. Worship with band and singing (~15 to 30 mins)
3. Welcome, announcements, baby dedications, or children dismiss (~5 to 15 mins)
4. Giving, offering prayer, or special ministry spotlight (~3 to 8 mins)
5. Transition prayer / Scripture reading / introduction of speaker
6. PREACHING / SERMON / PRIMARY TEACHING MESSAGE (>>> THIS IS THE ONLY PART WE WANT TO EXTRACT <<<)
7. Closing altar call / final worship chorus / dismissal announcements / exit chatter

Instructions:
- preaching_start_timestamp: The moment the main preacher/speaker steps up and begins reading scripture or preaching the sermon message.
- preaching_end_timestamp: The moment the main preacher finishes the sermon / concludes the final sermon prayer (before closing song / post-service announcements).
- Total service length is {format_seconds_to_timestamp(total_duration_sec)}.

Transcript:
{optimized_transcript}

Return strict JSON:
{{
  "preaching_start_timestamp": "HH:MM:SS",
  "preaching_end_timestamp": "HH:MM:SS",
  "preaching_start_sec": <float_seconds>,
  "preaching_end_sec": <float_seconds>,
  "speaker_name": "<name of preacher>",
  "first_words": "<first spoken words of sermon>",
  "last_words": "<closing words of sermon>",
  "reasoning": "<explanation of the selected start and end points>"
}}"""

        for model_name in ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]:
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                raw_json = resp.text.strip()
                # Clean any markdown code fences
                raw_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_json, flags=re.DOTALL).strip()
                data = json.loads(raw_json)

                start_ts = data.get("preaching_start_timestamp", "00:00:00")
                end_ts = data.get("preaching_end_timestamp", format_seconds_to_timestamp(total_duration_sec))

                start_sec = float(data.get("preaching_start_sec", parse_timestamp_to_seconds(start_ts)))
                end_sec = float(data.get("preaching_end_sec", parse_timestamp_to_seconds(end_ts)))

                if total_duration_sec > 0:
                    if end_sec <= start_sec or end_sec > total_duration_sec + 30:
                        end_sec = total_duration_sec
                    # Reject implausibly short sermon spans (e.g. < 10 mins on a normal service)
                    if total_duration_sec >= 1800 and (end_sec - start_sec < 600):
                        raise ValueError(
                            f"Implausible sermon span {format_seconds_to_timestamp(start_sec)} -> "
                            f"{format_seconds_to_timestamp(end_sec)} for a {format_seconds_to_timestamp(total_duration_sec)} service"
                        )

                boundary = SermonBoundary(
                    start_timestamp=format_seconds_to_timestamp(start_sec),
                    end_timestamp=format_seconds_to_timestamp(end_sec),
                    start_seconds=start_sec,
                    end_seconds=end_sec,
                    speaker_name=data.get("speaker_name", "Preacher"),
                    first_words=data.get("first_words", ""),
                    last_words=data.get("last_words", ""),
                    reasoning=data.get("reasoning", ""),
                )
                logger.info(
                    f"Detected sermon boundary: {boundary.start_timestamp} -> {boundary.end_timestamp} "
                    f"by {boundary.speaker_name}"
                )
                return boundary
            except Exception as e:
                logger.warning(f"Error detecting sermon boundaries with {model_name}: {e}")

        # Fallback based on reasonable church service liturgy
        fallback_start = total_duration_sec * 0.35 if total_duration_sec > 0 else 1800.0
        fallback_end = total_duration_sec if total_duration_sec > 0 else 5400.0
        return SermonBoundary(
            start_timestamp=format_seconds_to_timestamp(fallback_start),
            end_timestamp=format_seconds_to_timestamp(fallback_end),
            start_seconds=fallback_start,
            end_seconds=fallback_end,
            speaker_name="Preacher",
            first_words="N/A",
            last_words="N/A",
            reasoning="Fallback estimate based on audio length.",
        )

    def extract_preaching_audio(
        self,
        raw_audio_path: Path,
        boundary: SermonBoundary,
        output_dir: Path = DEFAULT_TRIMMED_DIR,
        force: bool = False,
    ) -> Path:
        """
        Extract only the preaching portion of the audio file using fast zero-reencode stream copy.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = raw_audio_path.stem
        out_filename = f"{stem} - Preaching.mp3"
        dest_path = output_dir / out_filename

        if not force and dest_path.exists() and dest_path.stat().st_size > 1024:
            logger.info(f"Trimmed preaching audio already exists: {dest_path.name}")
            return dest_path

        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for audio trimming.")

        duration_to_cut = max(1.0, boundary.end_seconds - boundary.start_seconds)
        logger.info(
            f"Trimming preaching audio for '{raw_audio_path.name}' "
            f"[{boundary.start_timestamp} -> {boundary.end_timestamp}] ({duration_to_cut/60:.1f} minutes)..."
        )

        tmp_path = dest_path.with_suffix(".tmp.mp3")

        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(boundary.start_seconds),
            "-i",
            str(raw_audio_path),
            "-t",
            str(duration_to_cut),
            "-c",
            "copy",
            str(tmp_path),
        ]

        t0 = time.time()
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)

        if not tmp_path.exists() or tmp_path.stat().st_size < 1024:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to generate trimmed audio: {tmp_path}")

        tmp_path.replace(dest_path)
        elapsed = time.time() - t0
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        logger.info(f"Extracted preaching audio: '{dest_path.name}' ({size_mb:.2f} MB) in {elapsed:.2f}s")
        return dest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preaching Audio Trimmer")
    parser.add_argument("--audio", help="Path to raw audio file")
    parser.add_argument("--transcript", help="Path to transcript Markdown file")
    parser.add_argument("--all", action="store_true", help="Trim all episodes in ProcessedMD")
    parser.add_argument("--force", action="store_true", help="Force re-trimming")

    args = parser.parse_args()
    trimmer = PreachingTrimmer()

    if args.audio:
        audio_p = Path(args.audio)
        md_p = Path(args.transcript) if args.transcript else DEFAULT_PROCESSED_DIR / f"{audio_p.stem}.md"
        text = md_p.read_text(encoding="utf-8") if md_p.exists() else ""
        dur = get_audio_duration(audio_p)
        boundary = trimmer.detect_boundaries_from_transcript(text, dur)
        trimmed = trimmer.extract_preaching_audio(audio_p, boundary, force=args.force)
        print(f"Trimmed preaching audio saved: {trimmed} ({boundary.start_timestamp} -> {boundary.end_timestamp})")

    elif args.all:
        for md_file in sorted(DEFAULT_PROCESSED_DIR.glob("*.md")):
            audio_file = DEFAULT_AUDIO_DIR / f"{md_file.stem}.mp3"
            if audio_file.exists():
                text = md_file.read_text(encoding="utf-8")
                dur = get_audio_duration(audio_file)
                boundary = trimmer.detect_boundaries_from_transcript(text, dur)
                trimmed = trimmer.extract_preaching_audio(audio_file, boundary, force=args.force)
                print(f"[TRIMMED] {trimmed.name} ({boundary.start_timestamp} -> {boundary.end_timestamp}) by {boundary.speaker_name}")
