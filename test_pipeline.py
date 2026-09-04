"""Unit and integration tests for the podcast pipeline, trimmer, and validator."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import downloader as downloader_module
import dashboard_server as dashboard_module
import drive_sync as drive_sync_module
import pipeline as pipeline_module
from downloader import sanitize_filename
from pipeline import PodcastPipeline
from transcriber import ProsodyTranscriber, adjust_chunk_timestamps
from trimmer import PreachingTrimmer, format_seconds_to_timestamp, parse_timestamp_to_seconds
from validator import PodcastValidator, detect_repetition_loops


def test_sanitize_filename():
    raw = "The Healing Properties of Psalm 23 (Part 5): Paths of Righteousness?"
    clean = sanitize_filename(raw)
    assert "?" not in clean
    assert ":" not in clean
    assert "The Healing Properties" in clean


def test_fetch_episodes_preserves_url_resolved_guid(tmp_path, monkeypatch):
    xml = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title><item>
    <title>Episode</title><guid>1234</guid><pubDate>Sun, 30 August 2026 12:00:00 EST</pubDate>
    <enclosure url="https://example.com/video.mp4" type="video/mp4"/>
    </item></channel></rss>"""

    class Response:
        content = xml

        def raise_for_status(self):
            return None

    monkeypatch.setattr(downloader_module.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(downloader_module, "FEED_CACHE_PATH", tmp_path / "feed_cache.json")

    episodes = downloader_module.fetch_episodes("https://example.com/xml/feed.xml", max_retries=1)
    assert episodes[0].guid == "https://example.com/xml/1234"



def test_dashboard_feed_cache_is_atomic_and_reused(tmp_path, monkeypatch):
    cache_path = tmp_path / "feed_cache.json"
    monkeypatch.setattr(dashboard_module, "FEED_CACHE_FILE", cache_path)
    calls = 0

    class CachedEpisode:
        def to_dict(self):
            return {"guid": "g", "title": "Episode"}

    def fetch(_):
        nonlocal calls
        calls += 1
        return [CachedEpisode()]

    monkeypatch.setattr(downloader_module, "fetch_episodes", fetch)
    assert dashboard_module._get_feed_entries_locked() == [{"guid": "g", "title": "Episode"}]
    assert json.loads(cache_path.read_text(encoding="utf-8"))[0]["guid"] == "g"
    assert not list(tmp_path.glob("*.tmp"))

    monkeypatch.setattr(downloader_module, "fetch_episodes", lambda _: pytest.fail("fresh cache was ignored"))
    assert dashboard_module._get_feed_entries_locked()[0]["guid"] == "g"
    assert calls == 1


def test_dashboard_chunk_progress_uses_created_chunk_count(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_module, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(dashboard_module, "TRIMMED_DIR", tmp_path / "TrimmedAudio")
    cache_dir = tmp_path / ".chunk_cache_Episode_480s"
    cache_dir.mkdir()
    for index in range(3):
        (cache_dir / f"chunk_{index:03d}.mp3").write_bytes(b"audio")
    (cache_dir / "chunk_000.txt").write_text("complete transcript", encoding="utf-8")
    (tmp_path / "Episode.mp3").write_bytes(b"small")

    active = dashboard_module.detect_current_active_stage()
    assert active["stage"] == "TRANSCRIBING"
    assert active["chunks_done"] == 1
    assert active["chunks_total"] == 3


def test_drive_mirror_skips_identical_file_copy(tmp_path, monkeypatch):
    source = tmp_path / "episode.md"
    source.write_text("transcript", encoding="utf-8")
    drive_root = tmp_path / "Drive"
    destination_dir = drive_root / "Transcripts"
    destination_dir.mkdir(parents=True)
    destination = destination_dir / source.name
    drive_sync_module.shutil.copy2(source, destination)

    uploader = drive_sync_module.DriveUploader.__new__(drive_sync_module.DriveUploader)
    uploader.local_drive_path = drive_root
    uploader.transcripts_folder_id = "transcripts"
    uploader.trimmed_audio_folder_id = "audio"
    uploader.folder_id = "root"
    uploader._api_authenticated = False
    uploader.service = None
    monkeypatch.setattr(
        drive_sync_module.shutil,
        "copy2",
        lambda *_: pytest.fail("identical mirror file was copied"),
    )

    result = uploader.upload_file(source)
    assert result["local_drive_dest"] == str(destination)

def test_timestamp_conversions():
    assert parse_timestamp_to_seconds("01:23:45") == 5025.0
    assert parse_timestamp_to_seconds("00:50:50") == 3050.0
    assert format_seconds_to_timestamp(3050.0) == "00:50:50"


def test_adjust_chunk_timestamps():
    chunk_text = "### Speaker [00:03:14] Welcome to church."
    adjusted = adjust_chunk_timestamps(chunk_text, start_sec=1800)  # 30 mins offset
    assert "[00:33:14]" in adjusted


def test_repetition_detector():
    normal_text = "God is good all the time and all the time God is good."
    assert detect_repetition_loops(normal_text) is None

    loop_text = " ".join(["and he said"] * 50)
    finding = detect_repetition_loops(loop_text)
    assert finding is not None
    assert finding[1] >= 50


def test_transcript_validation_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("CI", "true")
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
    transcriber.preferred_model = "gemini-3.7-flash"
    transcriber.last_model_used = None

    def fake_transcribe(audio_path, prompt, preferred_model, max_model_retries):
        assert preferred_model == "gemini-3.7-flash"
        transcriber.last_model_used = "gemini-2.5-flash"
        return "```json\n{\"needs_reprocess\": true, \"confidence\": 0.9, \"issues\": [\"omission\"]}\n```"

    monkeypatch.setattr(transcriber, "_transcribe_single_file", fake_transcribe)
    audio_path = tmp_path / "episode.mp3"
    result = transcriber.audit_transcript(audio_path, "existing transcript")

    assert result["needs_reprocess"] is True
    assert result["confidence"] == 0.9
    assert result["recommendation"] == "reprocess"
    assert result["model_used"] == "gemini-2.5-flash"


def test_manifest_save_preserves_other_process_updates(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"g": {"status": "FAILED"}}), encoding="utf-8")
    monkeypatch.setattr(pipeline_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(PodcastPipeline, "export_indexes", lambda self: None)

    first = PodcastPipeline.__new__(PodcastPipeline)
    first.manifest = {"g": {"status": "FAILED"}}
    first._dirty_guids = set()
    second = PodcastPipeline.__new__(PodcastPipeline)
    second.manifest = {"g": {"status": "FAILED"}}
    second._dirty_guids = set()

    first._write_record("g", {"status": "SUCCESS"})
    second._write_record("h", {"status": "SUCCESS"})

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == {"g": {"status": "SUCCESS"}, "h": {"status": "SUCCESS"}}


def test_transcriber_does_not_carry_client_advance_state(tmp_path):
    class FakeModels:
        def __init__(self, responses):
            self.responses = iter(responses)

        def generate_content(self, **kwargs):
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return response

    blocked = SimpleNamespace(vertexai=True, models=FakeModels([RuntimeError("403 PERMISSION_DENIED")]))
    working = SimpleNamespace(
        vertexai=True,
        models=FakeModels([SimpleNamespace(text=""), SimpleNamespace(text="ok")]),
    )
    transcriber = ProsodyTranscriber.__new__(ProsodyTranscriber)
    transcriber.clients = [blocked, working]
    transcriber.preferred_model = "first"
    transcriber.last_model_used = None
    transcriber._extract_response_text = lambda response: response.text
    audio_path = tmp_path / "chunk.mp3"
    audio_path.write_bytes(b"audio")

    assert transcriber._transcribe_single_file(audio_path, "prompt", max_model_retries=1) == "ok"


def test_trimmer_marks_full_service_fallback_and_stops_retrying():
    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "preaching_start_sec": 0,
                        "preaching_end_sec": 3600,
                        "speaker_name": "Preacher",
                    }
                )
            )

    models = FakeModels()
    trimmer = PreachingTrimmer.__new__(PreachingTrimmer)
    trimmer.clients = [SimpleNamespace(models=models), SimpleNamespace(models=models)]

    boundary = trimmer.detect_boundaries_from_transcript("### Speaker [00:00:00]\nText", 3600)
    assert boundary.is_fallback is True
    assert models.calls == 1


def test_validator_non_ci_missing_audio_is_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    processed_dir = tmp_path / "ProcessedMD"
    processed_dir.mkdir()
    sample = processed_dir / "2026-01-01 - Test.md"
    sample.write_text("---\ntitle: \"Test\"\nauthor: \"A\"\n---\n" + ("word " * 150), encoding="utf-8")

    report = PodcastValidator(
        processed_dir=processed_dir,
        audio_dir=tmp_path / "RawAudio",
        trimmed_dir=tmp_path / "TrimmedAudio",
        local_drive_path=tmp_path / "Drive",
    ).validate_episode(sample)
    assert report.status == "FAIL"


def test_repetition_detector_long_cycles():
    """Sentence-level detection catches cycles exceeding n-gram window."""
    sentence = "This is a fabricated hallucination sentence that keeps repeating"
    text = ". ".join([sentence] * 20) + "."
    finding = detect_repetition_loops(text)
    assert finding is not None
    _, count = finding
    assert count >= 8


def test_validator_timestamp_helper():
    from validator import _timestamp_seconds
    assert _timestamp_seconds("01:23:45") == 5025.0
    assert _timestamp_seconds("50:30") == 3030.0
    assert _timestamp_seconds("") == 0.0
    assert _timestamp_seconds(None) == 0.0


def test_canonicalize_speaker():
    from pipeline import canonicalize_speaker

    assert canonicalize_speaker("Pastor John") == "John C. Wood"
    assert canonicalize_speaker("John Wood") == "John C. Wood"
    assert canonicalize_speaker("Nicholas Gilchrist") == "Nick Gilchrist"
    assert canonicalize_speaker(None) == "Unknown"
    assert canonicalize_speaker("  Jane Doe  ") == "Jane Doe"


def test_markdown_index_text_escaping():
    from pipeline import _escape_markdown_text

    assert _escape_markdown_text(r"A [title] | path\name") == r"A \[title\] \| path\\name"


def test_persisted_boundary_repairs_only_material_mismatches(tmp_path, monkeypatch):
    from pipeline import _persisted_boundary

    record = {
        "preaching_start": "00:30:00",
        "preaching_end": "01:10:00",
        "speaker_name": "John C. Wood",
        "status": "SUCCESS",
    }
    trimmed_path = tmp_path / "trimmed.mp3"
    trimmed_path.write_bytes(b"audio")

    monkeypatch.setattr(pipeline_module, "get_audio_duration", lambda _: 2380.0)
    assert _persisted_boundary(record, trimmed_path) is None

    monkeypatch.setattr(pipeline_module, "get_audio_duration", lambda _: 2200.0)
    boundary = _persisted_boundary(record, trimmed_path)
    assert boundary is not None
    assert boundary.start_seconds == 1800.0
    assert boundary.end_seconds == 4200.0


def test_boundary_content_error_triggers_fallback():
    """When all models return full-service boundaries, fallback is used."""

    class AlwaysFullServiceModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(
                text=json.dumps({
                    "preaching_start_timestamp": "00:00:00",
                    "preaching_end_timestamp": "01:30:00",
                    "preaching_start_sec": 0,
                    "preaching_end_sec": 5400,
                    "speaker_name": "Preacher",
                })
            )

    trimmer = PreachingTrimmer.__new__(PreachingTrimmer)
    models = AlwaysFullServiceModels()
    trimmer.clients = [SimpleNamespace(models=models)]

    boundary = trimmer.detect_boundaries_from_transcript(
        "### Speaker [00:00:00]\nLong sermon text...",
        total_duration_sec=5400,
    )
    assert boundary.is_fallback is True
    assert boundary.start_seconds == pytest.approx(5400 * 0.35)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
