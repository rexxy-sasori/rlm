"""
LMHandler - Routes LLM requests from the RLM process and environment subprocesses.

Uses a multi-threaded socket server. Protocol: 4-byte length prefix + JSON payload.
"""

import asyncio
import time
from datetime import datetime, timezone
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Lock, Thread

from rlm.clients.base_lm import BaseLM
from rlm.core.comms_utils import LMRequest, LMResponse, socket_recv, socket_send
from rlm.core.types import RLMChatCompletion, UsageSummary
from rlm.logger.trace_logger import JSONLTraceLogger, generate_call_id, get_global_trace_logger


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self):
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                response = LMResponse.error_response("Request must be a JSON object")
                self._safe_send(response)
                return

            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler  # type: ignore

            if request.is_batched:
                # Batched request: process multiple prompts concurrently
                response = self._handle_batched(request, handler)
            elif request.prompt:
                # Single request: process one prompt
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts' in request.")

            self._safe_send(response)

        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - this is expected during parallel execution
            # when workers complete and close their sockets. Silently ignore.
            pass

        except Exception as e:
            # Try to send error response, but don't fail if socket is broken
            response = LMResponse.error_response(str(e))
            self._safe_send(response)

    def _safe_send(self, response: LMResponse) -> bool:
        """Send response, returning False if the socket is broken."""
        try:
            socket_send(self.connection, response.to_dict())
            return True
        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - silently ignore
            return False

    def _handle_single(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a single prompt request with call tracing."""
        client = handler.get_client(request.model, request.depth)

        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        trace_logger = get_global_trace_logger()
        call_id = request.request_id or generate_call_id()
        prompt_length = len(request.prompt) if isinstance(request.prompt, str) else len(str(request.prompt))

        try:
            content = client.completion(request.prompt)
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)

            # Log successful call
            if trace_logger and request.session_id and request.run_id:
                model_usage = client.get_last_usage()
                tokens = {
                    "prompt": model_usage.total_input_tokens,
                    "completion": model_usage.total_output_tokens,
                }
                trace_logger.log_llm_query(
                    run_id=request.run_id,
                    session_id=request.session_id,
                    call_id=call_id,
                    model=request.model or client.model_name,
                    prompt=request.prompt if isinstance(request.prompt, str) else str(request.prompt),
                    response=content,
                    depth=request.depth,
                    prompt_length=prompt_length,
                    response_length=len(content),
                    tokens=tokens,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    success=True,
                )

            # Legacy call logging for backward compatibility
            if request.session_id:
                call_data = {
                    "request_id": request.request_id,
                    "call_type": request.call_type or "llm_query",
                    "model": request.model or client.model_name,
                    "depth": request.depth,
                    "prompt_length": prompt_length,
                    "timestamp": timestamp,
                    "duration_ms": duration_ms,
                }
                handler.log_call(request.session_id, call_data)

            model_usage = client.get_last_usage()
            root_model = request.model or client.model_name
            usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
            return LMResponse.success_response(
                chat_completion=RLMChatCompletion(
                    root_model=root_model,
                    prompt=request.prompt,
                    response=content,
                    usage_summary=usage_summary,
                    execution_time=end_time - start_time,
                ),
                request_id=request.request_id,
                duration_ms=duration_ms,
                timestamp=timestamp,
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)
            error_msg = str(e)

            # Log failed call
            if trace_logger and request.session_id and request.run_id:
                trace_logger.log_llm_query(
                    run_id=request.run_id,
                    session_id=request.session_id,
                    call_id=call_id,
                    model=request.model or client.model_name,
                    prompt=request.prompt if isinstance(request.prompt, str) else str(request.prompt),
                    response="",
                    depth=request.depth,
                    prompt_length=prompt_length,
                    response_length=0,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    success=False,
                    error=error_msg,
                )

            # Legacy call logging for backward compatibility
            if request.session_id:
                call_data = {
                    "request_id": request.request_id,
                    "call_type": request.call_type or "llm_query",
                    "model": request.model or client.model_name,
                    "depth": request.depth,
                    "prompt_length": prompt_length,
                    "timestamp": timestamp,
                    "duration_ms": duration_ms,
                    "error": error_msg,
                }
                handler.log_call(request.session_id, call_data)

            raise

    def _handle_batched(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a batched prompts request using async for concurrency with call tracing."""
        client = handler.get_client(request.model, request.depth)

        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        trace_logger = get_global_trace_logger()
        call_id = request.request_id or generate_call_id()
        prompt_lengths = [
            len(p) if isinstance(p, str) else len(str(p))
            for p in request.prompts
        ]

        async def run_all():
            tasks = [client.acompletion(prompt) for prompt in request.prompts]
            return await asyncio.gather(*tasks)

        try:
            results = asyncio.run(run_all())
            end_time = time.perf_counter()

            total_time = end_time - start_time
            duration_ms = int(total_time * 1000)
            response_lengths = [len(r) for r in results]

            # Log successful batched call
            if trace_logger and request.session_id and request.run_id:
                model_usage = client.get_last_usage()
                tokens = {
                    "prompt": model_usage.total_input_tokens,
                    "completion": model_usage.total_output_tokens,
                }
                trace_logger.log_llm_query_batched(
                    run_id=request.run_id,
                    session_id=request.session_id,
                    call_id=call_id,
                    model=request.model or client.model_name,
                    batch_size=len(request.prompts),
                    prompts=request.prompts,
                    responses=results,
                    depth=request.depth,
                    prompt_lengths=prompt_lengths,
                    response_lengths=response_lengths,
                    tokens=tokens,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    success=True,
                )

            # Legacy call logging for backward compatibility
            if request.session_id:
                call_data = {
                    "request_id": request.request_id,
                    "call_type": request.call_type or "llm_query_batched",
                    "model": request.model or client.model_name,
                    "depth": request.depth,
                    "batch_size": len(request.prompts),
                    "prompt_lengths": prompt_lengths,
                    "timestamp": timestamp,
                    "duration_ms": duration_ms,
                }
                handler.log_call(request.session_id, call_data)

            model_usage = client.get_last_usage()
            root_model = request.model or client.model_name
            usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})

            chat_completions = [
                RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=content,
                    usage_summary=usage_summary,
                    execution_time=total_time / len(request.prompts),  # approximate per-prompt time
                )
                for prompt, content in zip(request.prompts, results, strict=True)
            ]

            return LMResponse.batched_success_response(
                chat_completions=chat_completions,
                request_id=request.request_id,
                duration_ms=duration_ms,
                timestamp=timestamp,
            )

        except Exception as e:
            end_time = time.perf_counter()
            duration_ms = int((end_time - start_time) * 1000)
            error_msg = str(e)

            # Log failed batched call
            if trace_logger and request.session_id and request.run_id:
                trace_logger.log_llm_query_batched(
                    run_id=request.run_id,
                    session_id=request.session_id,
                    call_id=call_id,
                    model=request.model or client.model_name,
                    batch_size=len(request.prompts),
                    prompts=request.prompts,
                    responses=[],
                    depth=request.depth,
                    prompt_lengths=prompt_lengths,
                    response_lengths=[],
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                    success=False,
                    error=error_msg,
                )

            # Legacy call logging for backward compatibility
            if request.session_id:
                call_data = {
                    "request_id": request.request_id,
                    "call_type": request.call_type or "llm_query_batched",
                    "model": request.model or client.model_name,
                    "depth": request.depth,
                    "batch_size": len(request.prompts),
                    "prompt_lengths": prompt_lengths,
                    "timestamp": timestamp,
                    "duration_ms": duration_ms,
                    "error": error_msg,
                }
                handler.log_call(request.session_id, call_data)

            raise


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """
    Handles all LM calls from the RLM main process and environment subprocesses.

    Uses a multi-threaded socket server for concurrent requests.
    Protocol: 4-byte big-endian length prefix + JSON payload.
    Includes call tracing for fine-grained logging of LLM calls.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,  # auto-assign available port
        other_backend_client: BaseLM | None = None,
    ):
        self.default_client = client
        self.other_backend_client = other_backend_client
        self.clients: dict[str, BaseLM] = {}
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port

        # Call tracing: session_id -> list of call data
        self._session_call_logs: dict[str, list[dict]] = {}
        self._session_lock = Lock()

        self.register_client(client.model_name, client)

    def start_session(self, session_id: str) -> None:
        """Start a new session for call tracing."""
        with self._session_lock:
            self._session_call_logs[session_id] = []

    def log_call(self, session_id: str, call_data: dict) -> None:
        """Log a call for the given session."""
        with self._session_lock:
            if session_id in self._session_call_logs:
                self._session_call_logs[session_id].append(call_data)

    def end_session(self, session_id: str) -> list[dict]:
        """End a session and return the call trace."""
        with self._session_lock:
            return self._session_call_logs.pop(session_id, [])

    def get_session_trace(self, session_id: str) -> list[dict]:
        """Get the current call trace for a session without ending it."""
        with self._session_lock:
            return list(self._session_call_logs.get(session_id, []))

    def register_client(self, model_name: str, client: BaseLM) -> None:
        """Register a client for a specific model name."""
        self.clients[model_name] = client

    def get_client(self, model: str | None = None, depth: int = 0) -> BaseLM:
        """Get client by model name or depth, or return default.

        Routing logic:
        - depth=0: use default_client (main backend)
        - depth=1: use other_backend_client if it exists, otherwise default_client
        - If model is specified and exists in clients, use that (overrides depth routing)
        """
        if model and model in self.clients:
            return self.clients[model]

        # Route based on depth
        if depth == 1 and self.other_backend_client is not None:
            return self.other_backend_client

        return self.default_client

    @property
    def port(self) -> int:
        """Get the actual port (useful when auto-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        """Get (host, port) tuple for connecting."""
        return (self.host, self.port)

    def start(self) -> tuple[str, int]:
        """Start the socket server in a background thread. Returns (host, port)."""
        if self._server is not None:
            return self.address

        self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self  # type: ignore

        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        return self.address

    def stop(self):
        """Stop the socket server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def completion(self, prompt: str, model: str | None = None) -> str:
        """Direct completion call (for main process use)."""
        return self.get_client(model).completion(prompt)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def get_usage_summary(self) -> UsageSummary:
        """Get the usage summary for all clients, merged into a single dict."""
        merged = {}
        # Include default client
        default_summary = self.default_client.get_usage_summary()
        merged.update(default_summary.model_usage_summaries)
        # Include other backend client if it exists
        if self.other_backend_client is not None:
            other_summary = self.other_backend_client.get_usage_summary()
            merged.update(other_summary.model_usage_summaries)
        # Include all registered clients
        for client in self.clients.values():
            client_summary = client.get_usage_summary()
            merged.update(client_summary.model_usage_summaries)
        return UsageSummary(model_usage_summaries=merged)
