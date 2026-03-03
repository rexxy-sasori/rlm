"""
Load testing script for RLM using the Oolong dataset.

This script loads the Oolong dataset and uses an async parallel manager
to test RLM's performance under concurrent load.

Usage:
    PORTKEY_API_KEY=... python examples/oolong_load_test.py [options]

Examples:
    # Basic usage with defaults
    PORTKEY_API_KEY=... python examples/oolong_load_test.py

    # Test with 20 samples and 10 concurrent requests
    PORTKEY_API_KEY=... python examples/oolong_load_test.py -n 20 -c 10

    # Use a different model with deeper recursion
    PORTKEY_API_KEY=... python examples/oolong_load_test.py --model "@openai/gpt-4o" --max-depth 2

    # Enable context compaction for long conversations
    PORTKEY_API_KEY=... python examples/oolong_load_test.py --compaction

    # Run without rich output (plain text)
    PORTKEY_API_KEY=... python examples/oolong_load_test.py --no-rich

Features:
- Loads oolongbench/oolong-real dataset from HuggingFace
- Configurable concurrency level (parallel requests)
- Configurable sample size for testing
- Performance metrics: throughput, latency percentiles, error rates
- Progress tracking with optional rich console output
- Optional context compaction to reduce token usage for long conversations
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed. Run: pip install datasets")
    sys.exit(1)

# Try to import rich, but make it optional
try:
    from rich.console import Console
    from rich.progress import Progress, TaskID
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # type: ignore
    Progress = None  # type: ignore
    TaskID = None  # type: ignore
    Table = None  # type: ignore

from rlm import RLM
from rlm.logger import RLMLogger


@dataclass
class LoadTestResult:
    """Result of a single RLM completion call."""

    sample_id: str
    success: bool
    latency: float
    tokens_input: int = 0
    tokens_output: int = 0
    error: str | None = None
    response: str = ""
    metadata: dict | None = None


@dataclass
class LoadTestMetrics:
    """Aggregated metrics from load testing."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    latencies: list[float] = field(default_factory=list)
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    errors: list[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def throughput(self) -> float:
        """Requests per second."""
        elapsed = self.end_time - self.start_time
        return self.total_requests / elapsed if elapsed > 0 else 0.0

    @property
    def success_rate(self) -> float:
        """Percentage of successful requests."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def mean_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.mean(self.latencies)

    @property
    def median_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class SimpleConsole:
    """Fallback console when rich is not available."""

    def print(self, message: str) -> None:
        # Strip rich formatting tags
        import re
        cleaned = re.sub(r'\[.*?\]', '', message)
        print(cleaned)


class SimpleProgress:
    """Fallback progress bar when rich is not available."""

    def __init__(self, total: int, description: str = ""):
        self.total = total
        self.current = 0
        self.description = description

    def __enter__(self):
        print(f"{self.description} (0/{self.total})")
        return self

    def __exit__(self, *args):
        print(f"Completed {self.total} items")

    def advance(self, task_id: Any, amount: int = 1) -> None:
        self.current += amount
        if self.current % max(1, self.total // 10) == 0 or self.current == self.total:
            print(f"  Progress: {self.current}/{self.total}")

    def add_task(self, description: str, total: int) -> int:
        return 0


class AsyncParallelManager:
    """
    Manages async parallel execution of RLM calls for load testing.

    Uses asyncio.Semaphore to control concurrency level.
    """

    def __init__(
        self,
        rlm: RLM,
        max_concurrency: int = 5,
        console: Any | None = None,
        use_rich: bool = True,
    ):
        self.rlm = rlm
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = console or (Console() if self.use_rich else SimpleConsole())
        self.results: list[LoadTestResult] = []
        self._progress: Any = None
        self._task_id: Any = None

    async def run_load_test(
        self,
        dataset_samples: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> LoadTestMetrics:
        """
        Run load test on the provided dataset samples.

        Args:
            dataset_samples: List of samples from the Oolong dataset
            show_progress: Whether to show progress bar

        Returns:
            LoadTestMetrics with aggregated results
        """
        metrics = LoadTestMetrics()
        metrics.start_time = time.perf_counter()
        self.results = []

        if show_progress:
            if self.use_rich and Progress is not None:
                with Progress(console=self.console) as progress:
                    self._progress = progress
                    self._task_id = progress.add_task(
                        f"[cyan]Processing {len(dataset_samples)} samples...",
                        total=len(dataset_samples),
                    )
                    await self._process_all(dataset_samples, metrics)
            else:
                with SimpleProgress(len(dataset_samples), f"Processing {len(dataset_samples)} samples...") as progress:
                    self._progress = progress
                    self._task_id = 0
                    await self._process_all(dataset_samples, metrics)
        else:
            await self._process_all(dataset_samples, metrics)

        metrics.end_time = time.perf_counter()
        return metrics

    async def _process_all(
        self,
        dataset_samples: list[dict[str, Any]],
        metrics: LoadTestMetrics,
    ) -> None:
        """Process all samples with controlled concurrency."""
        tasks = [
            self._process_single(sample, metrics)
            for sample in dataset_samples
        ]
        await asyncio.gather(*tasks)

    async def _process_single(
        self,
        sample: dict[str, Any],
        metrics: LoadTestMetrics,
    ) -> None:
        """Process a single sample with semaphore-controlled concurrency."""
        async with self.semaphore:
            result = await self._call_rlm(sample)
            self.results.append(result)

            # Update metrics
            metrics.total_requests += 1
            if result.success:
                metrics.successful_requests += 1
                metrics.latencies.append(result.latency)
                metrics.total_tokens_input += result.tokens_input
                metrics.total_tokens_output += result.tokens_output
            else:
                metrics.failed_requests += 1
                if result.error:
                    metrics.errors.append(result.error)

            # Update progress
            if self._progress is not None:
                if self.use_rich and hasattr(self._progress, 'advance'):
                    self._progress.advance(self._task_id)
                elif hasattr(self._progress, 'advance'):
                    self._progress.advance(self._task_id)

    async def _call_rlm(self, sample: dict[str, Any]) -> LoadTestResult:
        """Make a single RLM completion call."""
        sample_id = sample.get("id", "unknown")
        context = sample.get("context_window_text", "")
        question = sample.get("question", "")

        # Build the prompt
        prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the context provided.
If you need to perform calculations or analysis, use the REPL environment.
Return your final answer with FINAL_VAR."""

        start_time = time.perf_counter()

        try:
            # Run RLM completion in thread pool since RLM is synchronous
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Uses default executor
                lambda: self.rlm.completion(prompt),
            )

            end_time = time.perf_counter()
            latency = end_time - start_time

            # Extract token usage from result
            tokens_input = 0
            tokens_output = 0
            if result.usage_summary and result.usage_summary.model_usage_summaries:
                for model, usage in result.usage_summary.model_usage_summaries.items():
                    tokens_input += usage.total_input_tokens
                    tokens_output += usage.total_output_tokens

            return LoadTestResult(
                sample_id=sample_id,
                success=True,
                latency=latency,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                response=result.response,
                metadata=result.metadata,
            )

        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time

            return LoadTestResult(
                sample_id=sample_id,
                success=False,
                latency=latency,
                error=str(e),
            )


def print_metrics(metrics: LoadTestMetrics, console: Any, use_rich: bool = True) -> None:
    """Print load test metrics in a formatted table."""
    if use_rich and RICH_AVAILABLE and Table is not None:
        table = Table(title="Load Test Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(metrics.total_requests))
        table.add_row("Successful Requests", str(metrics.successful_requests))
        table.add_row("Failed Requests", str(metrics.failed_requests))
        table.add_row("Success Rate", f"{metrics.success_rate:.2f}%")
        table.add_row("", "")
        table.add_row("Throughput", f"{metrics.throughput:.2f} req/s")
        table.add_row("Total Time", f"{metrics.end_time - metrics.start_time:.2f}s")
        table.add_row("", "")
        table.add_row("Mean Latency", f"{metrics.mean_latency:.2f}s")
        table.add_row("Median Latency", f"{metrics.median_latency:.2f}s")
        table.add_row("P95 Latency", f"{metrics.p95_latency:.2f}s")
        table.add_row("P99 Latency", f"{metrics.p99_latency:.2f}s")
        table.add_row("", "")
        table.add_row("Total Input Tokens", str(metrics.total_tokens_input))
        table.add_row("Total Output Tokens", str(metrics.total_tokens_output))
        table.add_row("Total Tokens", str(metrics.total_tokens_input + metrics.total_tokens_output))

        console.print(table)
    else:
        # Plain text output
        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Total Requests:        {metrics.total_requests}")
        print(f"Successful Requests:   {metrics.successful_requests}")
        print(f"Failed Requests:       {metrics.failed_requests}")
        print(f"Success Rate:          {metrics.success_rate:.2f}%")
        print("")
        print(f"Throughput:            {metrics.throughput:.2f} req/s")
        print(f"Total Time:            {metrics.end_time - metrics.start_time:.2f}s")
        print("")
        print(f"Mean Latency:          {metrics.mean_latency:.2f}s")
        print(f"Median Latency:        {metrics.median_latency:.2f}s")
        print(f"P95 Latency:           {metrics.p95_latency:.2f}s")
        print(f"P99 Latency:           {metrics.p99_latency:.2f}s")
        print("")
        print(f"Total Input Tokens:    {metrics.total_tokens_input}")
        print(f"Total Output Tokens:   {metrics.total_tokens_output}")
        print(f"Total Tokens:          {metrics.total_tokens_input + metrics.total_tokens_output}")
        print("=" * 60)

    # Print errors if any
    if metrics.errors:
        error_msg = "\nErrors encountered:" if not use_rich else "\n[red]Errors encountered:[/red]"
        console.print(error_msg)
        error_counts: dict[str, int] = {}
        for error in metrics.errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        for error, count in error_counts.items():
            console.print(f"  - {error[:100]}... ({count} times)")


def load_oolong_dataset(
    subset: str = "dnd",
    split: str = "validation",
    num_samples: int | None = None,
    console: Any | None = None,
    use_rich: bool = True,
) -> list[dict[str, Any]]:
    """
    Load the Oolong dataset from HuggingFace.

    Args:
        subset: Dataset subset (e.g., 'dnd')
        split: Dataset split (e.g., 'validation')
        num_samples: Number of samples to load (None for all)
        console: Console for output
        use_rich: Whether to use rich formatting

    Returns:
        List of dataset samples
    """
    if console is None:
        console = Console() if use_rich and RICH_AVAILABLE else SimpleConsole()

    console.print(f"[cyan]Loading oolongbench/oolong-real dataset...[/cyan]" if use_rich else "Loading oolongbench/oolong-real dataset...")
    console.print(f"  Subset: {subset}")
    console.print(f"  Split: {split}")
    if num_samples:
        console.print(f"  Samples: {num_samples}")

    try:
        dataset = load_dataset(
            "oolongbench/oolong-real",
            subset,
            split=split,
            streaming=False,
            download_mode="reuse_dataset_if_exists"
        )
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]" if use_rich else f"Error loading dataset: {e}")
        console.print("[yellow]Make sure you have the datasets library installed:[/yellow]" if use_rich else "Make sure you have the datasets library installed:")
        console.print("  pip install datasets")
        raise

    # Convert to list and optionally limit samples
    samples = list(dataset)
    if num_samples:
        samples = samples[:num_samples]

    console.print(f"[green]Loaded {len(samples)} samples[/green]" if use_rich else f"Loaded {len(samples)} samples")
    return samples


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load testing script for RLM using the Oolong dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with Portkey (cloud)
    PORTKEY_API_KEY=... python examples/oolong_load_test.py

    # Local vLLM deployment (no API key needed)
    python examples/oolong_load_test.py --base-url http://localhost:8000/v1 --model llama3.1

    # Local Ollama deployment
    python examples/oolong_load_test.py --base-url http://localhost:11434/v1 --model llama3.1

    # Different models for root and sub-calls (e.g., strong model for reasoning, fast for sub-tasks)
    python examples/oolong_load_test.py \
        --base-url http://localhost:8000/v1 --model llama3.1-70b \
        --sub-base-url http://localhost:8001/v1 --sub-model llama3.1-8b \
        --max-depth 2

    # Test with 20 samples and 10 concurrent requests
    python examples/oolong_load_test.py -n 20 -c 10 --base-url http://localhost:8000/v1 --model llama3.1

    # Enable context compaction for long conversations
    python examples/oolong_load_test.py --base-url http://localhost:8000/v1 --model llama3.1 --compaction

    # Run without rich output (plain text)
    python examples/oolong_load_test.py --base-url http://localhost:8000/v1 --model llama3.1 --no-rich
        """
    )

    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=10,
        help="Number of dataset samples to test (default: 10)"
    )

    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=2,
        help="Maximum number of concurrent RLM calls (default: 5)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="@openai/gpt-5-nano",
        help="Model to use for testing (default: @openai/gpt-5-nano)"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Maximum RLM recursion depth (default: 1)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum RLM iterations per request (default: 5)"
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="dnd",
        help="Oolong dataset subset (default: dnd)"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Oolong dataset split (default: validation)"
    )

    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich output formatting (use plain text)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose RLM output"
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for local OpenAI-compatible API (e.g., http://localhost:8000/v1 for vLLM)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional for local deployments, uses OPENAI_API_KEY env var if not set)"
    )

    parser.add_argument(
        "--sub-model",
        type=str,
        default=None,
        help="Different model for sub-calls (rlm_query). Uses root model if not specified."
    )

    parser.add_argument(
        "--sub-base-url",
        type=str,
        default=None,
        help="Different base URL for sub-calls. Uses root base-url if not specified."
    )

    parser.add_argument(
        "--sub-api-key",
        type=str,
        default=None,
        help="Different API key for sub-calls. Uses root api-key if not specified."
    )

    parser.add_argument(
        "--compaction",
        action="store_true",
        help="Enable context compaction to reduce token usage for long conversations"
    )

    parser.add_argument(
        "--compaction-threshold-pct",
        type=float,
        default=0.85,
        help="Context compaction threshold as percentage of model context limit (default: 0.85)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    use_rich = not args.no_rich and RICH_AVAILABLE
    console = Console() if use_rich else SimpleConsole()

    # Determine backend configuration
    if args.base_url:
        # Local/OpenAI-compatible deployment
        backend = "openai"
        backend_kwargs: dict[str, Any] = {
            "model_name": args.model,
            "base_url": args.base_url,
        }
        # API key is optional for local deployments
        if args.api_key:
            backend_kwargs["api_key"] = args.api_key
        elif os.environ.get("OPENAI_API_KEY"):
            backend_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY")
        # For local deployments, no API key needed (vLLM, Ollama, etc.)
    else:
        # Use Portkey (cloud) - requires API key
        backend = "portkey"
        api_key = os.environ.get("PORTKEY_API_KEY")
        if not api_key:
            console.print("[red]Error: PORTKEY_API_KEY not set. Set it or use --base-url for local deployment.[/red]" if use_rich else "Error: PORTKEY_API_KEY not set. Set it or use --base-url for local deployment.")
            sys.exit(1)
        backend_kwargs = {
            "model_name": args.model,
            "api_key": api_key,
        }

    # Configure sub-model (for rlm_query calls) if specified
    other_backends = None
    other_backend_kwargs = None
    if args.sub_model or args.sub_base_url:
        # Sub-model uses openai backend if base_url specified, otherwise same as root
        if args.sub_base_url:
            other_backends = ["openai"]
            other_backend_kwargs = [{
                "model_name": args.sub_model or args.model,
                "base_url": args.sub_base_url,
            }]
            # Add API key for sub-model if specified
            if args.sub_api_key:
                other_backend_kwargs[0]["api_key"] = args.sub_api_key
            elif args.api_key:
                other_backend_kwargs[0]["api_key"] = args.api_key
            elif os.environ.get("OPENAI_API_KEY"):
                other_backend_kwargs[0]["api_key"] = os.environ.get("OPENAI_API_KEY")
        else:
            # Same backend type as root, different model
            other_backends = [backend]
            other_backend_kwargs = [backend_kwargs.copy()]
            other_backend_kwargs[0]["model_name"] = args.sub_model

    # Print configuration
    console.print("[bold blue]RLM Load Testing with Oolong Dataset[/bold blue]" if use_rich else "RLM Load Testing with Oolong Dataset")
    console.print("=" * 60)
    console.print(f"Backend: {backend}")
    if args.base_url:
        console.print(f"Base URL: {args.base_url}")
    console.print(f"Root Model: {args.model}")
    if args.sub_model or args.sub_base_url:
        console.print(f"Sub Model: {args.sub_model or args.model}")
        if args.sub_base_url:
            console.print(f"Sub Base URL: {args.sub_base_url}")
    console.print(f"Max Depth: {args.max_depth}")
    console.print(f"Max Iterations: {args.max_iterations}")
    console.print(f"Max Concurrency: {args.concurrency}")
    console.print(f"Num Samples: {args.num_samples}")
    console.print(f"Subset: {args.subset}")
    console.print(f"Split: {args.split}")
    console.print(f"Compaction: {'Enabled' if args.compaction else 'Disabled'}")
    if args.compaction:
        console.print(f"Compaction Threshold: {args.compaction_threshold_pct:.2%}")
    console.print("=" * 60)
    console.print()

    # Load dataset
    samples = load_oolong_dataset(
        subset=args.subset,
        split=args.split,
        num_samples=args.num_samples,
        console=console,
        use_rich=use_rich,
    )
    console.print()

    # Initialize RLM
    logger = RLMLogger()
    rlm_kwargs: dict[str, Any] = {
        "backend": backend,
        "backend_kwargs": backend_kwargs,
        "environment": "local",
        "max_depth": args.max_depth,
        "max_iterations": args.max_iterations,
        "logger": logger,
        "verbose": args.verbose,
        "compaction": args.compaction,
        "compaction_threshold_pct": args.compaction_threshold_pct,
    }
    # Add sub-model configuration if specified
    if other_backends and other_backend_kwargs:
        rlm_kwargs["other_backends"] = other_backends
        rlm_kwargs["other_backend_kwargs"] = other_backend_kwargs

    rlm = RLM(**rlm_kwargs)

    # Create async parallel manager
    manager = AsyncParallelManager(
        rlm=rlm,
        max_concurrency=args.concurrency,
        console=console,
        use_rich=use_rich,
    )

    # Run load test
    console.print(f"[cyan]Starting load test with {args.concurrency} concurrent requests...[/cyan]" if use_rich else f"Starting load test with {args.concurrency} concurrent requests...")
    console.print()

    metrics = asyncio.run(manager.run_load_test(samples, show_progress=True))

    # Print results
    console.print()
    print_metrics(metrics, console, use_rich=use_rich)

    # Show all sample responses with detailed info
    header = "\n[bold]All Request Details:[/bold]" if use_rich else "\nAll Request Details:"
    console.print(header)
    for i, result in enumerate(manager.results):
        if use_rich:
            status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        else:
            status = "[OK]" if result.success else "[FAIL]"
        console.print(f"\n{status} Request {i+1}/{len(manager.results)} - Sample {result.sample_id}")
        console.print(f"  Latency: {result.latency:.2f}s")
        console.print(f"  Input Tokens: {result.tokens_input:,}")
        console.print(f"  Output Tokens: {result.tokens_output:,}")
        console.print(f"  Total Tokens: {result.tokens_input + result.tokens_output:,}")
        if result.success:
            preview = result.response[:200].replace("\n", " ")
            console.print(f"  Response: {preview}...")
            
            # Show detailed iteration metrics and code
            if result.metadata and "iterations" in result.metadata:
                iterations = result.metadata.get("iterations", [])
                console.print(f"  Iterations: {len(iterations)}")
                for j, iteration in enumerate(iterations):
                    iteration_time = iteration.get("iteration_time", 0)
                    console.print(f"\n    Iteration {j+1}: {iteration_time:.2f}s")
                    
                    # Show code blocks
                    code_blocks = iteration.get("code_blocks", [])
                    if code_blocks:
                        console.print(f"    Code Blocks: {len(code_blocks)}")
                        for k, code_block in enumerate(code_blocks):
                            code = code_block.get("code", "")
                            result_data = code_block.get("result", {})
                            execution_time = result_data.get("execution_time", 0)
                            stdout = result_data.get("stdout", "")
                            stderr = result_data.get("stderr", "")
                            
                            console.print(f"\n      Code Block {k+1}: {execution_time:.2f}s")
                            console.print(f"      Code: {code[:500].replace('\n', ' ')}...")
                            if stdout:
                                console.print(f"      Stdout: {stdout[:200].replace('\n', ' ')}...")
                            if stderr:
                                console.print(f"      Stderr: {stderr[:200].replace('\n', ' ')}...")
        else:
            console.print(f"  Error: {result.error}")


if __name__ == "__main__":
    main()
