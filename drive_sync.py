"""Google Drive integration module for uploading and syncing processed podcast transcripts and trimmed audio."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.file"]
DEFAULT_LOCAL_DRIVE_ROOT = Path(
    os.getenv(
        "GOOGLE_DRIVE_LOCAL_PATH",
        "/Users/stephenfinney/Library/CloudStorage/GoogleDrive-ssfinney92@gmail.com/My Drive/Christ Chapel Podcasts",
    )
)


class DriveUploader:
    """Manages syncing transcripts and audio to Google Drive (via CloudStorage local mount or v3 API)."""

    def __init__(
        self,
        folder_id: Optional[str] = None,
        local_drive_path: Optional[Path] = None,
        credentials_file: Optional[Path] = None,
        token_file: Optional[Path] = None,
    ):
        self.folder_id = folder_id or os.getenv("GOOGLE_DRIVE_FOLDER_ID", "1emYUaJPU0_5qoGgNphNW51A12wwrUWhz")
        self.transcripts_folder_id = os.getenv("GOOGLE_DRIVE_TRANSCRIPTS_FOLDER_ID", "1bGjKB1GcSGIP1ZEpcLzUQPZiEUyDzTtT")
        self.trimmed_audio_folder_id = os.getenv("GOOGLE_DRIVE_TRIMMED_AUDIO_FOLDER_ID", "1J_ms3tdJipsRmcIeYGjnbV7jqd5GIerL")
        self.local_drive_path = local_drive_path or DEFAULT_LOCAL_DRIVE_ROOT

        self.credentials_file = credentials_file or Path(os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"))
        self.token_file = token_file or Path(os.getenv("GOOGLE_TOKEN_FILE", "token.json"))
        self.service = None
        self._api_authenticated = False
        self._init_service()

    def _init_service(self):
        """Try initializing Google Drive REST API if credentials exist, otherwise utilize DriveFS local sync."""
        try:
            import httplib2
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_httplib2 import AuthorizedHttp
            from googleapiclient.discovery import build
            creds = None
            if self.token_file.exists():
                try:
                    creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
                except Exception as e:
                    logger.debug(f"Token file load error: {e}")

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except Exception:
                        creds = None
                elif self.credentials_file.exists() and os.getenv("DRIVE_INTERACTIVE_AUTH") == "1":
                    try:
                        from google_auth_oauthlib.flow import InstalledAppFlow

                        flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), SCOPES)
                        creds = flow.run_local_server(port=0)
                        with open(self.token_file, "w") as token:
                            token.write(creds.to_json())
                    except Exception as flow_err:
                        logger.warning(f"Interactive Drive auth failed: {flow_err}")
                        creds = None

            if creds and creds.valid:
                authorized_http = AuthorizedHttp(creds, http=httplib2.Http(timeout=60))
                self.service = build("drive", "v3", http=authorized_http)
                self._api_authenticated = True
                logger.info("Google Drive REST API authenticated.")
        except Exception as init_err:
            logger.debug(f"Drive API init exception: {init_err}")

        if self.local_drive_path and self.local_drive_path.parent.exists():
            (self.local_drive_path / "Transcripts").mkdir(parents=True, exist_ok=True)
            (self.local_drive_path / "TrimmedAudio").mkdir(parents=True, exist_ok=True)
            logger.info(f"Google Drive active at: {self.local_drive_path}")

    @property
    def is_available(self) -> bool:
        return (self.local_drive_path and self.local_drive_path.exists()) or (self._api_authenticated and self.service is not None)

    def upload_file(
        self,
        file_path: Path,
        subfolder: str = "Transcripts",
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Upload/sync file to Google Drive.
        Prefers direct local desktop sync if available; falls back to REST API to avoid duplicate uploads.
        """
        if not file_path.exists():
            return None

        result = {
            "name": file_path.name,
            "folder_id": self.transcripts_folder_id if subfolder == "Transcripts" else self.trimmed_audio_folder_id,
            "local_drive_dest": None,
            "file_id": None,
            "web_view_link": None,
        }

        # 1. Prefer local Google Drive folder if available (atomic temp copy, zero duplicate API calls)
        if self.local_drive_path and self.local_drive_path.exists():
            dest_dir = self.local_drive_path / subfolder
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / file_path.name
            staging_file = dest_dir / f".{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
            try:
                src_stat = file_path.stat()
                if dest_file.exists():
                    dst_stat = dest_file.stat()
                    if dst_stat.st_size == src_stat.st_size and dst_stat.st_mtime_ns == src_stat.st_mtime_ns:
                        logger.debug(f"Google Drive mirror already current ({subfolder}): {dest_file.name}")
                        result["local_drive_dest"] = str(dest_file)
                        result["web_view_link"] = f"https://drive.google.com/drive/folders/{result['folder_id']}"
                        return result
                shutil.copy2(file_path, staging_file)
                os.replace(staging_file, dest_file)
                logger.info(f"Synced to Google Drive ({subfolder}): {dest_file.name}")
                result["local_drive_dest"] = str(dest_file)
                result["web_view_link"] = f"https://drive.google.com/drive/folders/{result['folder_id']}"
                return result
            except Exception as e:
                staging_file.unlink(missing_ok=True)
                logger.warning(f"Error copying to local Google Drive: {e}")

        # 2. Upload via Drive REST API only if local sync is not available
        if self._api_authenticated and self.service:
            try:
                from googleapiclient.http import MediaFileUpload

                mime = "text/markdown" if file_path.suffix == ".md" else "audio/mpeg"
                target_folder_id = result["folder_id"] or self.folder_id

                file_metadata = {"name": title or file_path.name, "mimeType": mime}
                if target_folder_id:
                    file_metadata["parents"] = [target_folder_id]
                if description:
                    file_metadata["description"] = description

                media = MediaFileUpload(str(file_path), mimetype=mime, resumable=True)
                uploaded = (
                    self.service.files()
                    .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
                    .execute()
                )
                result["file_id"] = uploaded.get("id")
                result["web_view_link"] = uploaded.get("webViewLink")
                logger.info(f"Uploaded via Drive API: ID={result['file_id']}")
                return result
            except Exception as e:
                logger.warning(f"API upload failed: {e}")

        logger.error(f"All Drive transports unavailable or failed for {file_path.name}")
        return None

    def upload_markdown(
        self,
        file_path: Path,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[dict]:
        return self.upload_file(file_path, subfolder="Transcripts", title=title, description=description)

    def upload_audio(
        self,
        file_path: Path,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[dict]:
        return self.upload_file(file_path, subfolder="TrimmedAudio", title=title, description=description)
