"""NotebookLM automated source sync module using notebooklm-py."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("notebooklm_sync")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_NOTEBOOK_ID = os.getenv("NOTEBOOKLM_NOTEBOOK_ID", "6d56cc10-c5a0-4795-85f1-da3830d98b85")
DEFAULT_PROCESSED_DIR = Path(__file__).parent / "ProcessedMD"
STORAGE_FILE = Path.home() / ".notebooklm" / "profiles" / "default" / "storage_state.json"


def is_notebooklm_authenticated() -> bool:
    """Check if notebooklm-py has saved authentication credentials."""
    return STORAGE_FILE.exists() and STORAGE_FILE.stat().st_size > 100


class NotebookLMSync:
    """Manages automated source ingestion into Google NotebookLM."""

    def __init__(self, notebook_id: Optional[str] = None):
        self.notebook_id = notebook_id or DEFAULT_NOTEBOOK_ID

    @property
    def is_available(self) -> bool:
        return is_notebooklm_authenticated()

    def sync_transcript(self, md_path: Path) -> bool:
        """
        Upload a Markdown transcript as a source into the NotebookLM notebook via CLI.
        Returns True on success, False if unauthenticated or error.
        """
        if not md_path.exists():
            logger.warning(f"Transcript file not found: {md_path}")
            return False

        if not self.is_available:
            logger.info(
                "NotebookLM is not authenticated. Run 'uv run notebooklm login' once to enable auto-import."
            )
            return False

        logger.info(f"Syncing '{md_path.name}' to NotebookLM (notebook: {self.notebook_id})...")
        cmd = [
            "notebooklm",
            "source",
            "add",
            str(md_path),
            "-n",
            self.notebook_id,
            "--title",
            md_path.stem,
            "--json",
        ]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            logger.info(f"Successfully added '{md_path.name}' to NotebookLM: {res.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to add '{md_path.name}' to NotebookLM: {e.stderr}")
            return False
        except Exception as e:
            logger.warning(f"Error syncing to NotebookLM: {e}")
            return False

    def sync_all(self, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> int:
        """Sync all processed Markdown transcripts to NotebookLM."""
        if not self.is_available:
            print("\n" + "=" * 70)
            print("  NOTEBOOKLM AUTHENTICATION REQUIRED")
            print("=" * 70)
            print("  To enable fully automated source ingestion into NotebookLM:")
            print("  Run: uv run notebooklm login")
            print("  (Sign in to your Google Account once in the opened window)")
            print("=" * 70 + "\n")
            return 0

        md_files = sorted(processed_dir.glob("*.md"))
        success_count = 0
        logger.info(f"Syncing {len(md_files)} transcripts to NotebookLM ({self.notebook_id})...")

        for f in md_files:
            if self.sync_transcript(f):
                success_count += 1

        logger.info(f"NotebookLM sync complete: {success_count}/{len(md_files)} uploaded.")
        return success_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NotebookLM Auto-Import CLI")
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK_ID, help="Target Notebook ID")
    parser.add_argument("--all", action="store_true", default=True, help="Sync all transcripts in ProcessedMD")
    parser.add_argument("--file", help="Sync a specific markdown transcript")
    parser.add_argument("--login", action="store_true", help="Launch NotebookLM browser login")

    args = parser.parse_args()

    if args.login:
        print("Launching NotebookLM login via Playwright Chromium...")
        subprocess.run(["notebooklm", "login"])
    elif args.file:
        syncer = NotebookLMSync(notebook_id=args.notebook)
        syncer.sync_transcript(Path(args.file))
    else:
        syncer = NotebookLMSync(notebook_id=args.notebook)
        syncer.sync_all()
