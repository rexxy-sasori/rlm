"""
JSONL Trace Logger for fine-grained LLM call tracing.

Writes trace events in JSONL format compatible with the RLM visualizer.
Each event is a single JSON line with a "type" field to distinguish event types.
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any


class JSONLTraceLogger:
    """
    Logger for fine-grained LLM call tracing in JSONL format.

    Each trace event is written as a single JSON line, making it:
    - Append-friendly for streaming
    - Safe for concurrent writes
    - Compatible with the RLM visualizer

    Usage:
        logger = JSONLTraceLogger("/path/to/logs")
        logger.start_run("run-001", model_config={...})
        logger.start_session("run-001", "session-001")
        logger.log_llm_query("run-001", "session-001", {...})
        logger.log_llm_query_batched("run-001", "session-001", {...})
        logger.end_session("run-001", "session-001")
        logger.end_run("run-001")
    """

    def __init__(self, log_dir: str | None = None):
        self.log_dir = log_dir
        self._file_handles: dict[str, Any] = {}
        self._lock = threading.Lock()

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

    def _get_log_path(self, run_id: str) -> str:
        """Get the log file path for a run."""
        if not self.log_dir:
            raise ValueError("log_dir not set")
        return os.path.join(self.log_dir, f"{run_id}.jsonl")

    def _get_file_handle(self, run_id: str) -> Any:
        """Get or create a file handle for the run."""
        with self._lock:
            if run_id not in self._file_handles:
                log_path = self._get_log_path(run_id)
                self._file_handles[run_id] = open(log_path, "a", buffering=1)
            return self._file_handles[run_id]

    def _write_event(self, run_id: str, event: dict) -> None:
        """Write a single event to the log file."""
        if not self.log_dir:
            return

        f = self._get_file_handle(run_id)
        with self._lock:
            json.dump(event, f, separators=(",", ":"))
            f.write("\n")
            f.flush()

    def start_run(
        self,
        run_id: str,
        model_config: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Log the start of a run."""
        event = {
            "type": "run_start",
            "run_id": run_id,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if model_config:
            event["model_config"] = model_config
        self._write_event(run_id, event)

    def end_run(
        self,
        run_id: str,
        timestamp: str | None = None,
    ) -> None:
        """Log the end of a run."""
        event = {
            "type": "run_end",
            "run_id": run_id,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        self._write_event(run_id, event)

        # Close and remove file handle
        with self._lock:
            if run_id in self._file_handles:
                self._file_handles[run_id].close()
                del self._file_handles[run_id]

    def start_session(
        self,
        run_id: str,
        session_id: str,
        parent_session_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Log the start of a session (concurrent request)."""
        event = {
            "type": "session_start",
            "run_id": run_id,
            "session_id": session_id,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if parent_session_id:
            event["parent_session_id"] = parent_session_id
        self._write_event(run_id, event)

    def end_session(
        self,
        run_id: str,
        session_id: str,
        total_calls: int = 0,
        total_duration_ms: int = 0,
        timestamp: str | None = None,
    ) -> None:
        """Log the end of a session."""
        event = {
            "type": "session_end",
            "run_id": run_id,
            "session_id": session_id,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "total_calls": total_calls,
            "total_duration_ms": total_duration_ms,
        }
        self._write_event(run_id, event)

    def log_llm_query(
        self,
        run_id: str,
        session_id: str,
        call_id: str,
        model: str,
        prompt: str | dict[str, Any],
        response: str,
        depth: int = 0,
        prompt_length: int | None = None,
        response_length: int | None = None,
        tokens: dict[str, int] | None = None,
        duration_ms: int = 0,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
        attempt: int = 1,
    ) -> None:
        """Log a single LLM query call.

        Args:
            success: Whether the call succeeded
            error: Error message if the call failed
            attempt: Attempt number (for retry tracking)
        """
        event = {
            "type": "llm_query",
            "run_id": run_id,
            "session_id": session_id,
            "call_id": call_id,
            "model": model,
            "depth": depth,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "success": success,
            "attempt": attempt,
        }

        if prompt_length is not None:
            event["prompt_length"] = prompt_length
        if response_length is not None:
            event["response_length"] = response_length
        if tokens:
            event["tokens"] = tokens
        if metadata:
            event["metadata"] = metadata
        if error:
            event["error"] = error

        self._write_event(run_id, event)

    def log_llm_query_batched(
        self,
        run_id: str,
        session_id: str,
        call_id: str,
        model: str,
        batch_size: int,
        prompts: list[str | dict[str, Any]],
        responses: list[str],
        depth: int = 0,
        prompt_lengths: list[int] | None = None,
        response_lengths: list[int] | None = None,
        tokens: dict[str, int] | None = None,
        duration_ms: int = 0,
        timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
        attempt: int = 1,
    ) -> None:
        """Log a batched LLM query call.

        Args:
            success: Whether the call succeeded
            error: Error message if the call failed
            attempt: Attempt number (for retry tracking)
        """
        event = {
            "type": "llm_query_batched",
            "run_id": run_id,
            "session_id": session_id,
            "call_id": call_id,
            "model": model,
            "depth": depth,
            "batch_size": batch_size,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "success": success,
            "attempt": attempt,
        }

        if prompt_lengths is not None:
            event["prompt_lengths"] = prompt_lengths
        if response_lengths is not None:
            event["response_lengths"] = response_lengths
        if tokens:
            event["tokens"] = tokens
        if metadata:
            event["metadata"] = metadata
        if error:
            event["error"] = error

        self._write_event(run_id, event)

    def close(self) -> None:
        """Close all open file handles."""
        with self._lock:
            for f in self._file_handles.values():
                f.close()
            self._file_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Global trace logger instance
_global_trace_logger: JSONLTraceLogger | None = None


def set_global_trace_logger(logger: JSONLTraceLogger | None) -> None:
    """Set the global trace logger instance."""
    global _global_trace_logger
    _global_trace_logger = logger


def get_global_trace_logger() -> JSONLTraceLogger | None:
    """Get the global trace logger instance."""
    return _global_trace_logger


def generate_call_id() -> str:
    """Generate a unique call ID."""
    return f"call-{uuid.uuid4().hex[:12]}"


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return f"session-{uuid.uuid4().hex[:12]}"


def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
