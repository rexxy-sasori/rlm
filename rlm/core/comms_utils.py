"""
Communication utilities for RLM socket protocol.

Protocol: 4-byte big-endian length prefix + JSON payload.
Used for communication between LMHandler and environment subprocesses.
"""

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any

from rlm.core.types import RLMChatCompletion

# =============================================================================
# Message Dataclasses
# =============================================================================


@dataclass
class LMRequest:
    """Request message sent to the LM Handler.

    Supports both single prompt (prompt field) and batched prompts (prompts field).
    Includes tracing fields for fine-grained call logging.
    """

    prompt: str | dict[str, Any] | None = None
    prompts: list[str | dict[str, Any]] | None = None
    model: str | None = None
    depth: int = 0
    # Tracing fields for call logging
    session_id: str | None = None
    request_id: str | None = None
    call_type: str | None = None  # "llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched"
    run_id: str | None = None  # Run ID for grouping sessions in JSONL traces

    @property
    def is_batched(self) -> bool:
        """Check if this is a batched request."""
        return self.prompts is not None and len(self.prompts) > 0

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        d = {}
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.prompts is not None:
            d["prompts"] = self.prompts
        if self.model is not None:
            d["model"] = self.model
        d["depth"] = self.depth
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.request_id is not None:
            d["request_id"] = self.request_id
        if self.call_type is not None:
            d["call_type"] = self.call_type
        if self.run_id is not None:
            d["run_id"] = self.run_id
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "LMRequest":
        """Create from dict."""
        return cls(
            prompt=data.get("prompt"),
            prompts=data.get("prompts"),
            model=data.get("model"),
            depth=data.get("depth", -1),  # TODO: Default should throw an error
            session_id=data.get("session_id"),
            request_id=data.get("request_id"),
            call_type=data.get("call_type"),
            run_id=data.get("run_id"),
        )


@dataclass
class LMResponse:
    """Response message from the LM Handler.

    Supports both single response (chat_completion) and batched responses (chat_completions).
    Includes tracing fields for fine-grained call logging.
    """

    error: str | None = None
    chat_completion: RLMChatCompletion | None = None
    chat_completions: list[RLMChatCompletion] | None = None
    # Tracing fields for call logging
    request_id: str | None = None
    duration_ms: int | None = None
    timestamp: str | None = None  # ISO format timestamp

    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None

    @property
    def is_batched(self) -> bool:
        """Check if this is a batched response."""
        return self.chat_completions is not None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        result = {}
        if self.error is not None:
            result["error"] = self.error
            result["chat_completion"] = None
            result["chat_completions"] = None
        elif self.chat_completions is not None:
            result["chat_completions"] = [c.to_dict() for c in self.chat_completions]
            result["chat_completion"] = None
            result["error"] = None
        elif self.chat_completion is not None:
            result["chat_completion"] = self.chat_completion.to_dict()
            result["chat_completions"] = None
            result["error"] = None
        else:
            result["error"] = "No chat completion or error provided."
            result["chat_completion"] = None
            result["chat_completions"] = None
        
        # Add tracing fields
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "LMResponse":
        """Create from dict."""
        chat_completions = None
        if data.get("chat_completions"):
            chat_completions = [RLMChatCompletion.from_dict(c) for c in data["chat_completions"]]

        chat_completion = None
        if data.get("chat_completion"):
            chat_completion = RLMChatCompletion.from_dict(data["chat_completion"])

        return cls(
            error=data.get("error"),
            chat_completion=chat_completion,
            chat_completions=chat_completions,
            request_id=data.get("request_id"),
            duration_ms=data.get("duration_ms"),
            timestamp=data.get("timestamp"),
        )

    @classmethod
    def success_response(cls, chat_completion: RLMChatCompletion, request_id: str | None = None, duration_ms: int | None = None, timestamp: str | None = None) -> "LMResponse":
        """Create a successful single response."""
        return cls(chat_completion=chat_completion, request_id=request_id, duration_ms=duration_ms, timestamp=timestamp)

    @classmethod
    def batched_success_response(cls, chat_completions: list[RLMChatCompletion], request_id: str | None = None, duration_ms: int | None = None, timestamp: str | None = None) -> "LMResponse":
        """Create a successful batched response."""
        return cls(chat_completions=chat_completions, request_id=request_id, duration_ms=duration_ms, timestamp=timestamp)

    @classmethod
    def error_response(cls, error: str, request_id: str | None = None) -> "LMResponse":
        """Create an error response."""
        return cls(error=error, request_id=request_id)


# =============================================================================
# Socket Protocol Helpers
# =============================================================================


def socket_send(sock: socket.socket, data: dict) -> None:
    """Send a length-prefixed JSON message over socket.

    Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
    """
    payload = json.dumps(data).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def socket_recv(sock: socket.socket) -> dict:
    """Receive a length-prefixed JSON message from socket.

    Protocol: 4-byte big-endian length prefix + UTF-8 JSON payload.
    Returns empty dict if connection closed before length received.

    Raises:
        ConnectionError: If connection closes mid-message.
    """
    raw_len = sock.recv(4)
    if not raw_len:
        return {}

    length = struct.unpack(">I", raw_len)[0]
    payload = b""
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            raise ConnectionError("Connection closed before message complete")
        payload += chunk

    return json.loads(payload.decode("utf-8"))


def socket_request(address: tuple[str, int], data: dict, timeout: int = 300) -> dict:
    """Send a request and receive a response over a new socket connection.

    Opens a new TCP connection, sends the request, waits for response, then closes.

    Args:
        address: (host, port) tuple to connect to.
        data: Dictionary to send as JSON.
        timeout: Socket timeout in seconds (default 300).

    Returns:
        Response dictionary.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(address)
        socket_send(sock, data)
        return socket_recv(sock)


# =============================================================================
# Typed Request Helpers
# =============================================================================


def send_lm_request(
    address: tuple[str, int], request: LMRequest, timeout: int = 300, depth: int | None = None
) -> LMResponse:
    """Send an LM request and return typed response.

    Args:
        address: (host, port) tuple of LM Handler server.
        request: LMRequest to send.
        timeout: Socket timeout in seconds.
        depth: Optional depth to override request depth.

    Returns:
        LMResponse with content or error.
    """
    try:
        if depth is not None:
            request.depth = depth
        response_data = socket_request(address, request.to_dict(), timeout)
        return LMResponse.from_dict(response_data)
    except Exception as e:
        return LMResponse.error_response(f"Request failed: {e}")


def send_lm_request_batched(
    address: tuple[str, int],
    prompts: list[str | dict[str, Any]],
    model: str | None = None,
    timeout: int = 300,
    depth: int = 0,
) -> list[LMResponse]:
    """Send a batched LM request and return a list of typed responses.

    Args:
        address: (host, port) tuple of LM Handler server.
        prompts: List of prompts to send.
        model: Optional model name to use.
        timeout: Socket timeout in seconds.
        depth: Depth for routing (default 0).

    Returns:
        List of LMResponse objects, one per prompt, in the same order.
    """
    try:
        request = LMRequest(prompts=prompts, model=model, depth=depth)
        response_data = socket_request(address, request.to_dict(), timeout)
        response = LMResponse.from_dict(response_data)

        if not response.success:
            # Return error responses for all prompts
            return [LMResponse.error_response(response.error)] * len(prompts)

        if response.chat_completions is None:
            return [LMResponse.error_response("No completions returned")] * len(prompts)

        # Convert batched response to list of individual responses
        return [
            LMResponse.success_response(chat_completion)
            for chat_completion in response.chat_completions
        ]
    except Exception as e:
        return [LMResponse.error_response(f"Request failed: {e}")] * len(prompts)
