from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
import sys


class _TeeTextStream:
    """Mirror writes to both the original stream and a log file stream."""

    def __init__(self, original_stream, log_stream) -> None:
        self._original_stream = original_stream
        self._log_stream = log_stream

    def write(self, text: str) -> int:
        written_original = self._original_stream.write(text)
        self._log_stream.write(text)
        return written_original

    def flush(self) -> None:
        self._original_stream.flush()
        self._log_stream.flush()

    @property
    def encoding(self):
        return getattr(self._original_stream, "encoding", "utf-8")

    def isatty(self) -> bool:
        return bool(getattr(self._original_stream, "isatty", lambda: False)())


def install_terminal_output_logger(log_filename: str = "terminal_out.txt") -> Path:
    """Redirect stdout/stderr to both console and a workspace log file."""
    workspace_dir = Path(__file__).resolve().parent
    log_path = workspace_dir / log_filename

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_path, mode="w", encoding="utf-8")

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    command_text = " ".join(sys.argv) if sys.argv else "python"
    log_file.write(f"Run started: {run_timestamp}\n")
    log_file.write(f"Command: {command_text}\n")
    log_file.write("-" * 60 + "\n")
    log_file.flush()

    sys.stdout = _TeeTextStream(original_stdout, log_file)
    sys.stderr = _TeeTextStream(original_stderr, log_file)

    def _cleanup() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_file.close()

    atexit.register(_cleanup)
    return log_path
