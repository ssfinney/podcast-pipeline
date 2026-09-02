"""NotebookLM automated source sync module using notebooklm-py with deduplication & existing source checks."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

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


def normalize_title(title: str) -> str:
    """Normalize source title for robust duplicate matching."""
    t = title.strip()
    if t.endswith(".md"):
        t = t[:-3].strip()
    # Normalize hyphens and multiple spaces
    t = re.sub(r"[\s\-_]+", " ", t).lower()
    return t


class NotebookLMSync:
    """Manages automated source ingestion into Google NotebookLM with built-in deduplication."""

    def __init__(self, notebook_id: Optional[str] = None):
        self.notebook_id = notebook_id or DEFAULT_NOTEBOOK_ID
        self._existing_sources_cache: Optional[Set[str]] = None
        self._cache_timestamp: float = 0.0

    @property
    def is_available(self) -> bool:
        return is_notebooklm_authenticated()

    def get_existing_source_titles(self, force_refresh: bool = False) -> Set[str]:
        """Fetch and cache normalized titles of existing sources in the notebook."""
        now = time.time()
        if not force_refresh and self._existing_sources_cache is not None and (now - self._cache_timestamp < 60.0):
            return self._existing_sources_cache

        if not self.is_available:
            return set()

        cmd = [
            "notebooklm",
            "source",
            "list",
            "-n",
            self.notebook_id,
            "--json",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(res.stdout)
            titles = set()
            for s in data.get("sources", []):
                raw_title = s.get("title", "")
                if raw_title:
                    titles.add(normalize_title(raw_title))
            self._existing_sources_cache = titles
            self._cache_timestamp = now
            return titles
        except Exception as e:
            logger.warning(f"Could not list NotebookLM sources: {e}")
            return self._existing_sources_cache or set()

    def sync_transcript(self, md_path: Path) -> bool:
        """
        Upload a Markdown transcript as a source into the NotebookLM notebook.
        Skips gracefully if a source by the same title already exists.
        """
        if not md_path.exists():
            logger.warning(f"Transcript file not found: {md_path}")
            return False

        if not self.is_available:
            logger.info("NotebookLM is not authenticated. Run 'uv run notebooklm login' once.")
            return False

        norm_name = normalize_title(md_path.stem)
        existing_titles = self.get_existing_source_titles()

        if norm_name in existing_titles:
            logger.info(f"Source '{md_path.stem}' already exists in NotebookLM. Skipping duplicate upload.")
            return True

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
            logger.info(f"Successfully added '{md_path.name}' to NotebookLM.")
            # Update in-memory cache
            if self._existing_sources_cache is not None:
                self._existing_sources_cache.add(norm_name)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to add '{md_path.name}' to NotebookLM: {e.stderr}")
            return False
        except Exception as e:
            logger.warning(f"Error syncing to NotebookLM: {e}")
            return False

    def replace_transcript(self, md_path: Path) -> bool:
        """Replace matching NotebookLM sources before uploading a revised transcript."""
        if not md_path.exists() or not self.is_available:
            return False

        norm_name = normalize_title(md_path.stem)
        try:
            res = subprocess.run(
                ["notebooklm", "source", "list", "-n", self.notebook_id, "--json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            sources = json.loads(res.stdout).get("sources", [])
        except Exception as e:
            logger.warning(f"Could not list NotebookLM sources for replacement: {e}")
            return False

        matching_ids = [
            s.get("id")
            for s in sources
            if s.get("id") and normalize_title(s.get("title", "")) == norm_name
        ]
        for source_id in matching_ids:
            try:
                subprocess.run(
                    ["notebooklm", "source", "delete", source_id, "-y", "-n", self.notebook_id],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"Could not delete existing NotebookLM source {source_id}: {e}")
                return False

        self._existing_sources_cache = None
        return self.sync_transcript(md_path)

    def clean_duplicates(self) -> int:
        """Find and remove duplicate sources in the target notebook."""
        if not self.is_available:
            logger.warning("NotebookLM not authenticated.")
            return 0

        cmd = ["notebooklm", "source", "list", "-n", self.notebook_id, "--json"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            data = json.loads(res.stdout)
            sources = data.get("sources", [])
        except Exception as e:
            logger.error(f"Failed to list sources for cleanup: {e}")
            return 0

        seen: Dict[str, dict] = {}
        duplicates: List[dict] = []

        for s in sources:
            n_title = normalize_title(s.get("title", ""))
            if n_title in seen:
                duplicates.append(s)
            else:
                seen[n_title] = s

        deleted_count = 0
        for dup in duplicates:
            sid = dup.get("id")
            title = dup.get("title")
            logger.info(f"Deleting duplicate source in NotebookLM: {sid} ({title})...")
            del_cmd = ["notebooklm", "source", "delete", sid, "-y", "-n", self.notebook_id]
            res = subprocess.run(del_cmd, capture_output=True, text=True, timeout=30)
            if res.returncode == 0:
                deleted_count += 1
            else:
                logger.warning(f"Failed deleting duplicate {sid}: {res.stderr}")

        self._existing_sources_cache = None  # Invalidate cache
        logger.info(f"Cleaned {deleted_count} duplicate sources from NotebookLM.")
        return deleted_count

    def sync_all(self, processed_dir: Path = DEFAULT_PROCESSED_DIR) -> int:
        """Sync all processed Markdown transcripts to NotebookLM, skipping existing ones."""
        if not self.is_available:
            print("\n" + "=" * 70)
            print("  NOTEBOOKLM AUTHENTICATION REQUIRED")
            print("=" * 70)
            print("  To enable fully automated source ingestion into NotebookLM:")
            print("  Run: uv run notebooklm login")
            print("=" * 70 + "\n")
            return 0

        md_files = sorted(processed_dir.glob("*.md"))
        existing_titles = self.get_existing_source_titles(force_refresh=True)
        logger.info(f"Found {len(existing_titles)} existing sources in NotebookLM. Syncing {len(md_files)} local files...")

        success_count = 0
        for f in md_files:
            if self.sync_transcript(f):
                success_count += 1

        logger.info(f"NotebookLM sync check complete: {success_count}/{len(md_files)} up-to-date.")
        return success_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NotebookLM Auto-Import & Deduplication CLI")
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK_ID, help="Target Notebook ID")
    parser.add_argument("--all", action="store_true", help="Sync all transcripts in ProcessedMD (skipping existing)")
    parser.add_argument("--clean", action="store_true", help="Remove duplicate sources from NotebookLM")
    parser.add_argument("--file", help="Sync a specific markdown transcript")
    parser.add_argument("--login", action="store_true", help="Launch NotebookLM browser login")

    args = parser.parse_args()
    syncer = NotebookLMSync(notebook_id=args.notebook)

    if args.login:
        print("Launching NotebookLM login via Playwright Chromium...")
        subprocess.run(["notebooklm", "login"])
    elif args.clean:
        syncer.clean_duplicates()
    elif args.file:
        syncer.sync_transcript(Path(args.file))
    elif args.all:
        syncer.sync_all()
    else:
        syncer.sync_all()
