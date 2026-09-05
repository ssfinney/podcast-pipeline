"""Unit and integration tests for the podcast pipeline, trimmer, and validator."""

from pathlib import Path
import pytest
from downloader import sanitize_filename, parse_pub_date, Episode
from trimmer import parse_timestamp_to_seconds, format_seconds_to_timestamp
from transcriber import ProsodyTranscriber, adjust_chunk_timestamps
from local_transcriber import WordProsody, render_prosody_words
from validator import detect_repetition_loops, PodcastValidator


def test_sanitize_filename():
    raw = "The Healing Properties of Psalm 23 (Part 5): Paths of Righteousness?"
    clean = sanitize_filename(raw)
    assert "?" not in clean
    assert ":" not in clean
    assert "The Healing Properties" in clean


def test_timestamp_conversions():
    assert parse_timestamp_to_seconds("01:23:45") == 5025.0
    assert parse_timestamp_to_seconds("00:50:50") == 3050.0
    assert format_seconds_to_timestamp(3050.0) == "00:50:50"


def test_adjust_chunk_timestamps():
    chunk_text = "### Speaker [00:03:14] Welcome to church."
    adjusted = adjust_chunk_timestamps(chunk_text, start_sec=1800)  # 30 mins offset
    assert "[00:33:14]" in adjusted


def test_local_prosody_renderer_uses_measured_features():
    words = [
        WordProsody("we", 0.0, 0.2, -25.0, 120.0, 121.0),
        WordProsody("believe", 0.25, 0.75, -24.0, 120.0, 122.0),
        WordProsody("grace", 1.6, 2.2, -12.0, 120.0, 155.0),
        WordProsody(".", 2.2, 2.25, -25.0, None, None),
    ]
    rendered = render_prosody_words(words)
    assert "### Speaker [00:00:00]" in rendered
    assert "[pause: 0.9s]" in rendered
    assert "[rising pitch]" in rendered
    assert "**grace**" in rendered


def test_local_prosody_renderer_marks_speaker_changes():
    words = [
        WordProsody("Welcome", 0.0, 0.4, -20.0, 120.0, 120.0, "SPEAKER_00"),
        WordProsody("Amen", 1.0, 1.4, -20.0, 180.0, 180.0, "SPEAKER_01"),
    ]
    rendered = render_prosody_words(words)
    assert "### SPEAKER_00 [00:00:00]" in rendered
    assert "### SPEAKER_01 [00:00:01]" in rendered


def test_repetition_detector():
    normal_text = "God is good all the time and all the time God is good."
    assert detect_repetition_loops(normal_text) is None

    loop_text = (
        "and he said and he said and he said and he said and he said and he said and he said and he said."
    )
    assert detect_repetition_loops(loop_text) is not None


def test_transcript_validation_flow(tmp_path):
    # Test validator with a realistic sample transcript fixture
    processed_dir = tmp_path / "ProcessedMD"
    processed_dir.mkdir()
    sample_md = processed_dir / "2026-08-30 - Test Sermon.md"

    body_lines = [
        "---",
        'title: "Test Sermon"',
        'date: "2026-08-30"',
        'author: "John C. Wood"',
        'duration: "1:30:00"',
        "---",
        "",
        "# Test Sermon",
        "",
        "## Summary",
        "A sample test sermon summary.",
        "",
        "---",
        "",
        "## Verbatim Prosody Transcript",
        "",
        "### Speaker [00:00:00]",
        "Good **morning** church family! [pause: 1.0s] [cheering]",
        "We are so **grateful** to be here today. [rising pitch]",
        "The **Lord** is our **shepherd** and we shall not **want**.",
        "He **leadeth** me in paths of **righteousness** for his name sake. [pause: 0.5s]",
        "Yea, though I **walk** through the valley of the shadow of death, [pause: 1.0s]",
        "I will **fear** no **evil**, for thou art with me. [whispering]",
        "Thy rod and thy staff they **comfort** me. [pitch drop]",
        "Surely **goodness** and **mercy** shall follow me all the days of my life. [applause]",
    ]

    sample_md.write_text("\n".join(body_lines), encoding="utf-8")

    validator = PodcastValidator(processed_dir=processed_dir, audio_dir=tmp_path, trimmed_dir=tmp_path)
    reports = validator.audit_all(output_json=False)
    assert len(reports) == 1
    assert reports[0].status in ["PASS", "WARN"]



def test_audit_transcript_parses_json_and_records_model(monkeypatch, tmp_path):
    transcriber = ProsodyTranscriber.__new__(ProsodyTranscriber)
    transcriber.last_model_used = "gemini-3.7-flash"

    def fake_transcribe(audio_path, prompt, preferred_model, max_model_retries):
        assert preferred_model == "gemini-3.7-flash"
        return "```json\n{\"needs_reprocess\": true, \"confidence\": 0.9, \"issues\": [\"omission\"]}\n```"

    monkeypatch.setattr(transcriber, "_transcribe_single_file", fake_transcribe)
    audio_path = tmp_path / "episode.mp3"
    result = transcriber.audit_transcript(audio_path, "existing transcript")

    assert result["needs_reprocess"] is True
    assert result["confidence"] == 0.9
    assert result["recommendation"] == "reprocess"
    assert result["model_used"] == "gemini-3.7-flash"
if __name__ == "__main__":
    pytest.main(["-v", __file__])
