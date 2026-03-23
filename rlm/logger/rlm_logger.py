"""
Logger for RLM iterations.

Captures run metadata and iterations in memory so they can be attached to
RLMChatCompletion.metadata. Optionally writes the same data to JSON-lines files.
Supports session-based logging for fine-grained call tracing.
"""

import json
import os
import uuid
from datetime import datetime

from rlm.core.types import RLMIteration, RLMMetadata


class RLMLogger:
    """
    Captures trajectory (run metadata + iterations) for each completion.
    By default only captures in memory; set log_dir to also save to disk.

    - log_dir=None: trajectory is available via get_trajectory() and can be
      attached to RLMChatCompletion.metadata (no disk write).
    - log_dir="path": same capture plus appends to a JSONL file per run.
    - session_mode=True: saves each session to a separate file in a run directory.
    """

    def __init__(self, log_dir: str | None = None, file_name: str = "rlm", session_mode: bool = True):
        self._save_to_disk = log_dir is not None
        self.log_dir = log_dir
        self.file_name = file_name
        self.session_mode = session_mode
        self.log_file_path: str | None = None
        self.run_dir: str | None = None
        
        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if session_mode:
                # Create a run directory for session-based logging
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_id = str(uuid.uuid4())[:8]
                self.run_dir = os.path.join(log_dir, f"run_{timestamp}_{run_id}")
                os.makedirs(self.run_dir, exist_ok=True)
                self.log_file_path = None  # Will be set per session
            else:
                # Legacy mode: single JSONL file
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_id = str(uuid.uuid4())[:8]
                self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._run_metadata: dict | None = None
        self._iterations: list[dict] = []
        self._iteration_count = 0
        self._metadata_logged = False
        self._current_session_id: str | None = None

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Capture run metadata (and optionally write to file)."""
        if self._metadata_logged:
            return

        self._run_metadata = metadata.to_dict()
        self._metadata_logged = True

        if self._save_to_disk:
            entry = {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **self._run_metadata,
            }
            if self.session_mode and self.run_dir:
                # Save metadata to run directory
                metadata_path = os.path.join(self.run_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(entry, f, indent=2)
            elif self.log_file_path:
                with open(self.log_file_path, "a") as f:
                    json.dump(entry, f)
                    f.write("\n")

    def start_session(self, session_id: str) -> None:
        """Start a new session for logging."""
        self._current_session_id = session_id
        self._iterations = []
        self._iteration_count = 0
        
        if self._save_to_disk and self.session_mode and self.run_dir:
            # Set log file path for this session
            self.log_file_path = os.path.join(self.run_dir, f"session_{session_id}.jsonl")

    def log(self, iteration: RLMIteration) -> None:
        """Capture one iteration (and optionally append to file)."""
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }
        self._iterations.append(entry)

        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def end_session(self, session_id: str, final_answer: str | None = None, execution_time: float | None = None) -> None:
        """End a session and save summary."""
        if self._save_to_disk and self.session_mode and self.run_dir:
            # Save session summary
            summary = {
                "type": "summary",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "total_iterations": self._iteration_count,
                "final_answer": final_answer,
                "execution_time": execution_time,
            }
            summary_path = os.path.join(self.run_dir, f"session_{session_id}_summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        
        self._current_session_id = None

    def clear_iterations(self) -> None:
        """Reset iterations for the next completion (trajectory is per completion)."""
        self._iterations = []
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        """Return captured run_metadata + iterations for the current completion, or None if no metadata yet."""
        if self._run_metadata is None:
            return None
        return {
            "run_metadata": self._run_metadata,
            "iterations": list(self._iterations),
        }

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
