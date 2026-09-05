"""Offline Whisper transcription with evidence-based prosody annotations."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from downloader import Episode
from transcriber import ProsodyTranscriber, format_timestamp, get_audio_duration

logger = logging.getLogger(__name__)


@dataclass
class WordProsody:
    text: str
    start: float
    end: float
    rms_db: float = -80.0
    pitch_start_hz: Optional[float] = None
    pitch_end_hz: Optional[float] = None
    speaker: str = "Speaker"


def _median(values: list[float], default: float = 0.0) -> float:
    return statistics.median(values) if values else default


def _robust_center_scale(values: list[float], floor: float) -> tuple[float, float]:
    """Return median and MAD scale that remain stable for short passages."""
    if not values:
        return 0.0, floor
    center = statistics.median(values)
    mad = statistics.median(abs(item - center) for item in values)
    return center, max(floor, 1.4826 * mad)


def _pitch_delta(word: WordProsody) -> float:
    if not word.pitch_start_hz or not word.pitch_end_hz:
        return 0.0
    if word.pitch_start_hz <= 0 or word.pitch_end_hz <= 0:
        return 0.0
    return 12.0 * math.log2(word.pitch_end_hz / word.pitch_start_hz)


def render_prosody_words(
    words: list[WordProsody],
    pause_threshold_sec: float = 0.65,
    pitch_threshold_semitones: float = 2.5,
    emphasis_score_threshold: float = 1.8,
    heading_interval_sec: float = 45.0,
) -> str:
    """Render timestamped Markdown from measured word-level acoustic features."""
    if not words:
        return ""

    by_speaker: dict[str, list[WordProsody]] = {}
    for word in words:
        by_speaker.setdefault(word.speaker, []).append(word)

    speaker_stats: dict[
        str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = {}
    for speaker, speaker_words in by_speaker.items():
        intensities = [item.rms_db for item in speaker_words]
        rates = [
            (item.end - item.start) / max(1, len(re.sub(r"\W", "", item.text)))
            for item in speaker_words
        ]
        pitches = [
            _median([item.pitch_start_hz or 0.0, item.pitch_end_hz or 0.0])
            for item in speaker_words
            if item.pitch_start_hz and item.pitch_end_hz
        ]
        speaker_stats[speaker] = (
            _robust_center_scale(intensities, 1.5),
            _robust_center_scale(rates, 0.015),
            _robust_center_scale(pitches, 8.0),
        )

    rendered: list[str] = []
    line_tokens: list[str] = []
    current_speaker: Optional[str] = None
    last_heading_at = -heading_interval_sec
    previous_end: Optional[float] = None

    def flush_line() -> None:
        if not line_tokens:
            return
        line = " ".join(line_tokens)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        rendered.append(line.strip())
        line_tokens.clear()

    for word in words:
        speaker_changed = word.speaker != current_speaker
        heading_due = word.start - last_heading_at >= heading_interval_sec
        if speaker_changed or heading_due:
            flush_line()
            if rendered:
                rendered.append("")
            rendered.append(f"### {word.speaker} [{format_timestamp(word.start)}]")
            current_speaker = word.speaker
            last_heading_at = word.start

        if previous_end is not None:
            gap = max(0.0, word.start - previous_end)
            if gap >= pause_threshold_sec:
                line_tokens.append(f"[pause: {gap:.1f}s]")

        duration_rate = (word.end - word.start) / max(
            1, len(re.sub(r"\W", "", word.text))
        )
        intensity_stats, duration_stats, pitch_stats = speaker_stats[word.speaker]
        intensity_z = (word.rms_db - intensity_stats[0]) / intensity_stats[1]
        duration_z = (duration_rate - duration_stats[0]) / duration_stats[1]
        pitch_mid = _median([word.pitch_start_hz or 0.0, word.pitch_end_hz or 0.0])
        pitch_z = (
            (pitch_mid - pitch_stats[0]) / pitch_stats[1] if pitch_mid > 0 else 0.0
        )
        emphasis_score = intensity_z + 0.5 * duration_z + 0.35 * abs(pitch_z)

        pitch_delta = _pitch_delta(word)
        if pitch_delta >= pitch_threshold_semitones:
            line_tokens.append("[rising pitch]")
        elif pitch_delta <= -pitch_threshold_semitones:
            line_tokens.append("[pitch drop]")

        token = word.text.strip()
        lexical = re.sub(r"\W", "", token)
        if len(lexical) >= 3 and emphasis_score >= emphasis_score_threshold:
            token = f"**{token}**"
        line_tokens.append(token)
        previous_end = word.end

    flush_line()
    return "\n".join(rendered).strip()


class LocalProsodyTranscriber(ProsodyTranscriber):
    """Drop-in transcriber backed by faster-whisper, librosa, and optional pyannote."""

    def __init__(
        self, model_name: Optional[str] = None, diarize: Optional[bool] = None
    ):
        self.backend = "local"
        self.api_key = None
        self.client = None
        self.preferred_model = model_name or os.getenv(
            "LOCAL_WHISPER_MODEL", "small.en"
        )
        self.last_model_used: Optional[str] = None
        self.diarize = (
            diarize
            if diarize is not None
            else os.getenv("LOCAL_DIARIZATION", "false").lower() == "true"
        )
        self._whisper_model = None

    def _load_whisper(self):
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Local transcription dependencies are missing. "
                "Run: uv sync --extra local-prosody"
            ) from exc
        device = os.getenv("LOCAL_WHISPER_DEVICE", "cpu")
        compute_type = os.getenv(
            "LOCAL_WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16"
        )
        self._whisper_model = WhisperModel(
            self.preferred_model, device=device, compute_type=compute_type
        )
        return self._whisper_model

    @staticmethod
    def _measure_words(wav_path: Path, words: list[WordProsody]) -> list[WordProsody]:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Local prosody dependencies are missing. "
                "Run: uv sync --extra local-prosody"
            ) from exc

        sample_rate = 16_000
        hop_length = 320
        audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
        rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            frame_length=1024,
            hop_length=hop_length,
        )

        for word in words:
            start_frame = max(0, int(word.start * sample_rate / hop_length))
            end_frame = min(
                len(rms_db),
                max(start_frame + 1, int(word.end * sample_rate / hop_length)),
            )
            word.rms_db = float(np.nanmedian(rms_db[start_frame:end_frame]))
            voiced_pitch = f0[start_frame:end_frame]
            voiced_pitch = voiced_pitch[np.isfinite(voiced_pitch)]
            if len(voiced_pitch) >= 2:
                midpoint = max(1, len(voiced_pitch) // 2)
                word.pitch_start_hz = float(np.nanmedian(voiced_pitch[:midpoint]))
                word.pitch_end_hz = float(np.nanmedian(voiced_pitch[midpoint:]))
        return words

    def _transcribe_chunk(self, wav_path: Path) -> list[WordProsody]:
        model = self._load_whisper()
        segments, _ = model.transcribe(
            str(wav_path),
            language="en",
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        words: list[WordProsody] = []
        for segment in segments:
            for item in segment.words or []:
                if (
                    item.start is not None
                    and item.end is not None
                    and item.word.strip()
                ):
                    words.append(
                        WordProsody(
                            item.word.strip(), float(item.start), float(item.end)
                        )
                    )
        return self._measure_words(wav_path, words)

    def _speaker_intervals(self, audio_path: Path) -> list[tuple[float, float, str]]:
        if not self.diarize:
            return []
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "LOCAL_DIARIZATION=true requires HF_TOKEN after accepting "
                "the pyannote model terms."
            )
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote is missing. Run: "
                "uv sync --extra local-prosody --extra diarization"
            ) from exc
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", token=token
        )
        output = pipeline(str(audio_path))
        return [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in output.speaker_diarization.itertracks(
                yield_label=True
            )
        ]

    @staticmethod
    def _assign_speakers(
        words: list[WordProsody], intervals: list[tuple[float, float, str]]
    ) -> None:
        for word in words:
            midpoint = (word.start + word.end) / 2
            for start, end, speaker in intervals:
                if start <= midpoint <= end:
                    word.speaker = speaker
                    break

    def transcribe_audio_file(
        self,
        audio_path: Path,
        episode: Optional[Episode] = None,
        chunk_duration_sec: int = 120,
        max_workers: int = 1,
    ) -> str:
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise FileNotFoundError(f"Audio file not found or empty: {audio_path}")
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            duration = (audio_path.stat().st_size * 8) / 64_000
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg not found for local audio decoding.")

        cache_dir = audio_path.parent / f".local_chunk_cache_{audio_path.stem}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        all_words: list[WordProsody] = []
        chunk_count = math.ceil(duration / chunk_duration_sec)
        logger.info(
            "Local transcription: %d chunks with %s", chunk_count, self.preferred_model
        )

        for index in range(chunk_count):
            offset = index * chunk_duration_sec
            length = min(chunk_duration_sec, duration - offset)
            cache_path = cache_dir / f"chunk_{index:03d}.json"
            if cache_path.exists():
                chunk_words = [
                    WordProsody(**item)
                    for item in json.loads(cache_path.read_text(encoding="utf-8"))
                ]
            else:
                wav_path = cache_dir / f"chunk_{index:03d}.wav"
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-ss",
                        str(offset),
                        "-i",
                        str(audio_path),
                        "-t",
                        str(length),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(wav_path),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=120,
                )
                try:
                    chunk_words = self._transcribe_chunk(wav_path)
                    cache_path.write_text(
                        json.dumps([asdict(item) for item in chunk_words]),
                        encoding="utf-8",
                    )
                finally:
                    wav_path.unlink(missing_ok=True)
            for word in chunk_words:
                word.start += offset
                word.end += offset
            all_words.extend(chunk_words)

        self._assign_speakers(all_words, self._speaker_intervals(audio_path))
        self.last_model_used = f"local:{self.preferred_model}"
        transcript = render_prosody_words(all_words)
        shutil.rmtree(cache_dir, ignore_errors=True)
        return transcript

    def audit_transcript(self, *args, **kwargs) -> dict:
        raise RuntimeError(
            "Gemini transcript auditing is unavailable with the local backend; "
            "use validator.py."
        )
