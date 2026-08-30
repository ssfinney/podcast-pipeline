"""Unit and integration tests for the podcast pipeline, trimmer, and validator."""

from pathlib import Path
import pytest
from downloader import sanitize_filename, parse_pub_date, Episode
from trimmer import parse_timestamp_to_seconds, format_seconds_to_timestamp
from transcriber import adjust_chunk_timestamps
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
    adjusted = adjust_chunk_timestamps(chunk_text, start_sec=1800) # 30 mins offset
    assert "[00:33:14]" in adjusted


def test_repetition_detector():
    normal_text = "God is good all the time and all the time God is good."
    assert detect_repetition_loops(normal_text) is None

    loop_text = "and he said and he said and he said and he said and he said and he said and he said and he said and he said and he said."
    assert detect_repetition_loops(loop_text) is not None


def test_processed_transcripts_exist():
    md_files = list(Path("ProcessedMD").glob("*.md"))
    assert len(md_files) >= 3


def test_validator_run():
    validator = PodcastValidator()
    reports = validator.audit_all(output_json=False)
    assert len(reports) >= 3
    assert all(r.status in ["PASS", "WARN"] for r in reports)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
