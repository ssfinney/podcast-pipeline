"""Generate non-destructive local prosody previews for existing audio files."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from local_transcriber import LocalProsodyTranscriber


def metrics(text: str) -> dict[str, int]:
    return {
        "words": len(re.findall(r"\b\w+\b", text)),
        "stress_tags": len(re.findall(r"\*\*[^*]+\*\*", text)),
        "prosody_tags": len(
            re.findall(
                r"\[(?:pause:|rising pitch|pitch drop)[^\]]*\]",
                text,
                re.IGNORECASE,
            )
        ),
        "speaker_headings": len(
            re.findall(r"^### .+ \[\d{2}:\d{2}:\d{2}\]$", text, re.MULTILINE)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare local prosody output with existing Gemini transcripts"
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--audio-dir", type=Path, default=Path("RawAudio"))
    parser.add_argument("--processed-dir", type=Path, default=Path("ProcessedMD"))
    parser.add_argument("--output-dir", type=Path, default=Path("ProsodyPOC"))
    parser.add_argument("--diarize", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transcriber = LocalProsodyTranscriber(model_name=args.model, diarize=args.diarize)
    report: list[dict] = []
    for audio_path in sorted(args.audio_dir.glob("*.mp3"), reverse=True)[: args.limit]:
        started = time.time()
        local_text = transcriber.transcribe_audio_file(audio_path)
        output_path = args.output_dir / f"{audio_path.stem}.local.md"
        output_path.write_text(local_text + "\n", encoding="utf-8")
        existing_path = args.processed_dir / f"{audio_path.stem}.md"
        existing_text = (
            existing_path.read_text(encoding="utf-8") if existing_path.exists() else ""
        )
        report.append(
            {
                "audio": audio_path.name,
                "model": transcriber.last_model_used,
                "elapsed_seconds": round(time.time() - started, 1),
                "existing": metrics(existing_text),
                "local": metrics(local_text),
                "preview": str(output_path),
            }
        )
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
