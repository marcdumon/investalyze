"""Run investalyze CLI commands as subprocesses and capture their output for the control panel.

One job runs at a time: every CLI writes to the same DuckDB file, so overlapping jobs would
contend for its single-writer lock. `MANAGER` is a process-wide singleton; the control panel
page polls it on a `dcc.Interval` instead of holding any state itself.
"""

import re
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
_MAX_LINES = 5000
_MAX_HISTORY = 20

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(line: str) -> str:
    """Remove ANSI color codes emitted by the CLIs' console formatter."""
    return _ANSI_RE.sub('', line)


@dataclass
class Job:
    """One subprocess run: its command, captured output, and outcome."""

    title: str
    argv: list[str]
    started: datetime
    finished: datetime | None = None
    returncode: int | None = None
    lines: list[str] = field(default_factory=list)


class JobManager:
    """Runs one CLI subprocess at a time and keeps its output for live and historical display."""

    def __init__(self) -> None:
        """Start with no job running and empty history."""
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self.current: Job | None = None
        self.history: list[Job] = []

    def is_running(self) -> bool:
        """True while a subprocess is currently executing."""
        with self._lock:
            return self._process is not None

    def start(self, title: str, argv: list[str]) -> bool:
        """Launch `[sys.executable, *argv]` as a new job. Returns False if a job is already running."""
        with self._lock:
            if self._process is not None:
                return False
            job = Job(title=title, argv=argv, started=datetime.now())
            process = subprocess.Popen(
                [sys.executable, *argv],
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._process = process
            self.current = job
        threading.Thread(target=self._read_output, args=(process, job), daemon=True).start()
        return True

    def _read_output(self, process: subprocess.Popen, job: Job) -> None:
        """Stream the subprocess's combined stdout/stderr into `job.lines` until it exits."""
        for raw_line in process.stdout:
            line = strip_ansi(raw_line.rstrip('\n'))
            job.lines.append(line)
            if len(job.lines) > _MAX_LINES:
                del job.lines[: len(job.lines) - _MAX_LINES]
        job.returncode = process.wait()
        job.finished = datetime.now()
        with self._lock:
            self._process = None
            self.current = None
            self.history.insert(0, job)
            del self.history[_MAX_HISTORY:]

    def cancel(self) -> bool:
        """Send SIGTERM to the running subprocess, if any. Returns False if nothing is running."""
        with self._lock:
            if self._process is None:
                return False
            self._process.send_signal(signal.SIGTERM)
            return True


MANAGER = JobManager()
