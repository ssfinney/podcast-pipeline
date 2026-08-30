"""Operational service management CLI for podcast background backfill and dashboard services."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent
PID_FILE = BASE_DIR / ".backfill.pid"
LOG_FILE = BASE_DIR / "backfill.log"


def get_running_process(name_pattern: str = "pipeline.py") -> Optional[int]:
    """Find running pipeline process PID via pgrep."""
    try:
        res = subprocess.run(["pgrep", "-f", f"python.*{name_pattern}"], capture_output=True, text=True)
        pids = [int(p.strip()) for p in res.stdout.strip().split("\n") if p.strip()]
        # Exclude our own PID
        my_pid = os.getpid()
        active = [p for p in pids if p != my_pid]
        return active[0] if active else None
    except Exception:
        return None


def get_dashboard_process() -> Optional[int]:
    """Find running dashboard process PID."""
    return get_running_process("dashboard_server.py")


def service_status():
    """Report status of backfill and dashboard services."""
    backfill_pid = get_running_process("pipeline.py")
    dashboard_pid = get_dashboard_process()

    print("\n" + "=" * 65)
    print(" " * 20 + "SERVICE STATUS OVERVIEW")
    print("=" * 65)
    print(f"  Podcast Backfill Engine : {'🟢 RUNNING (PID ' + str(backfill_pid) + ')' if backfill_pid else '🔴 STOPPED'}")
    print(f"  Web Telemetry Dashboard : {'🟢 RUNNING (PID ' + str(dashboard_pid) + ' | http://localhost:8420)' if dashboard_pid else '🔴 STOPPED'}")
    print("=" * 65)

    # Inspect current disk state
    from dashboard_server import detect_current_active_stage, get_pipeline_status

    status = get_pipeline_status()
    act = status.get("active_job", {})
    print(f"  Cataloged Progress      : {status.get('processed_count')}/{status.get('total_episodes')} ({status.get('percent_complete')}%)")
    print(f"  Preaching Extracted     : {status.get('total_preaching_time')}")
    print(f"  Current Stage           : {act.get('stage_label', 'Idle')}")
    if act.get("episode_title"):
        print(f"  Active Episode          : {act.get('episode_title')}")
        print(f"  Details                 : {act.get('details')}")
    print("=" * 65 + "\n")


def service_stop(graceful_timeout_sec: int = 15):
    """Gracefully stop the running backfill process."""
    pid = get_running_process("pipeline.py")
    if not pid:
        print("No active podcast backfill process found.")
        return

    print(f"Sending SIGTERM to process {pid}...")
    try:
        os.kill(pid, signal.SIGTERM)
        t0 = time.time()
        while time.time() - t0 < graceful_timeout_sec:
            if not get_running_process("pipeline.py"):
                print("Process stopped gracefully.")
                return
            time.sleep(1)
        print("Process did not exit in time; sending SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        print("Process terminated.")
    except ProcessLookupError:
        print("Process already stopped.")


def service_start(limit: int = 0):
    """Start background backfill process."""
    if get_running_process("pipeline.py"):
        print("Backfill process is already running.")
        return

    cmd = ["uv", "run", "python", "pipeline.py"]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])

    print(f"Starting background backfill: {' '.join(cmd)}")
    log_fd = open(LOG_FILE, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=log_fd,
        stderr=subprocess.STREQUAL if hasattr(subprocess, "STREQUAL") else log_fd,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"Started podcast backfill with PID: {proc.pid}")


def service_reload():
    """Gracefully reload: stop active backfill and start with new code."""
    print("=== Graceful Rolling Service Reload ===")
    service_stop()
    time.sleep(2)
    service_start()
    time.sleep(2)
    service_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Podcast Service Management CLI")
    parser.add_argument("action", choices=["status", "start", "stop", "reload", "logs"], help="Action to perform")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for start action")
    parser.add_argument("--lines", type=int, default=30, help="Lines of logs to tail")

    args = parser.parse_args()

    if args.action == "status":
        service_status()
    elif args.action == "stop":
        service_stop()
    elif args.action == "start":
        service_start(limit=args.limit)
    elif args.action == "reload":
        service_reload()
    elif args.action == "logs":
        if LOG_FILE.exists():
            lines = LOG_FILE.read_text(encoding="utf-8").strip().split("\n")
            print("\n".join(lines[-args.lines :]))
        else:
            print("No log file found. Run 'manage_service.py start' to initiate logging.")
