#!/usr/bin/env python3
"""
Analysis script for RLM trace logs (JSONL format).

Analyzes llm_query and llm_query_batched events to provide insights on:
- Call counts and success rates
- Cache hit rates (if available)
- Timing statistics
- Model usage distribution
- Error analysis

Usage:
    python analyze_traces.py /path/to/trace.jsonl
    python analyze_traces.py /path/to/trace.jsonl --output summary.json
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_trace_file(file_path: str) -> list[dict]:
    """Parse a JSONL trace file."""
    events = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
    return events


def analyze_traces(events: list[dict]) -> dict[str, Any]:
    """Analyze trace events and compute statistics."""
    
    # Filter to only LLM query events
    query_events = [e for e in events if e.get('type') in ('llm_query', 'llm_query_batched')]
    
    if not query_events:
        return {"error": "No llm_query or llm_query_batched events found"}
    
    # Basic counts
    total_calls = len(query_events)
    single_calls = len([e for e in query_events if e.get('type') == 'llm_query'])
    batched_calls = len([e for e in query_events if e.get('type') == 'llm_query_batched'])
    
    # Success/failure analysis
    successful_calls = len([e for e in query_events if e.get('success', True)])
    failed_calls = total_calls - successful_calls
    success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
    
    # Error analysis
    errors = defaultdict(int)
    for e in query_events:
        if not e.get('success', True) and e.get('error'):
            # Categorize errors
            error_msg = e['error']
            if 'timeout' in error_msg.lower():
                errors['timeout'] += 1
            elif 'connection' in error_msg.lower() or 'network' in error_msg.lower():
                errors['connection'] += 1
            elif 'rate' in error_msg.lower() or 'limit' in error_msg.lower():
                errors['rate_limit'] += 1
            else:
                errors['other'] += 1
    
    # Timing statistics
    durations = [e.get('duration_ms', 0) for e in query_events]
    avg_duration = sum(durations) / len(durations) if durations else 0
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    sorted_durations = sorted(durations)
    p50_duration = sorted_durations[len(sorted_durations) // 2] if sorted_durations else 0
    p95_duration = sorted_durations[int(len(sorted_durations) * 0.95)] if sorted_durations else 0
    p99_duration = sorted_durations[int(len(sorted_durations) * 0.99)] if sorted_durations else 0
    
    # Model usage
    model_counts = defaultdict(int)
    for e in query_events:
        model = e.get('model', 'unknown')
        model_counts[model] += 1
    
    # Depth distribution
    depth_counts = defaultdict(int)
    for e in query_events:
        depth = e.get('depth', 0)
        depth_counts[depth] += 1
    
    # Session analysis
    session_calls = defaultdict(int)
    session_durations = defaultdict(list)
    for e in query_events:
        session_id = e.get('session_id', 'unknown')
        session_calls[session_id] += 1
        session_durations[session_id].append(e.get('duration_ms', 0))
    
    # Batch size analysis (for batched calls)
    batch_sizes = []
    total_batch_items = 0
    for e in query_events:
        if e.get('type') == 'llm_query_batched':
            batch_size = e.get('batch_size', 0)
            batch_sizes.append(batch_size)
            total_batch_items += batch_size
    
    avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0
    
    # Token analysis (if available)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    calls_with_tokens = 0
    for e in query_events:
        tokens = e.get('tokens', {})
        if tokens:
            calls_with_tokens += 1
            total_prompt_tokens += tokens.get('prompt', 0)
            total_completion_tokens += tokens.get('completion', 0)
    
    # Compute prompt/completion lengths
    prompt_lengths = []
    response_lengths = []
    for e in query_events:
        if 'prompt_length' in e:
            prompt_lengths.append(e['prompt_length'])
        if 'response_length' in e:
            response_lengths.append(e['response_length'])
        # For batched calls
        if 'prompt_lengths' in e:
            prompt_lengths.extend(e['prompt_lengths'])
        if 'response_lengths' in e:
            response_lengths.extend(e['response_lengths'])
    
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    
    # Build summary
    summary = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_events": len(events),
        "total_llm_calls": total_calls,
        "call_types": {
            "single": single_calls,
            "batched": batched_calls,
        },
        "success_rate": {
            "successful": successful_calls,
            "failed": failed_calls,
            "rate_percent": round(success_rate, 2),
        },
        "errors": dict(errors) if errors else None,
        "timing_ms": {
            "average": round(avg_duration, 2),
            "min": min_duration,
            "max": max_duration,
            "p50": p50_duration,
            "p95": p95_duration,
            "p99": p99_duration,
            "total_duration_ms": sum(durations),
        },
        "models_used": dict(model_counts),
        "depth_distribution": dict(depth_counts),
        "session_stats": {
            "total_sessions": len(session_calls),
            "avg_calls_per_session": round(total_calls / len(session_calls), 2) if session_calls else 0,
        },
        "batch_stats": {
            "total_batched_calls": batched_calls,
            "avg_batch_size": round(avg_batch_size, 2),
            "total_items_in_batches": total_batch_items,
        } if batched_calls else None,
        "token_stats": {
            "calls_with_token_data": calls_with_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        } if calls_with_tokens else None,
        "length_stats": {
            "avg_prompt_length": round(avg_prompt_length, 2),
            "avg_response_length": round(avg_response_length, 2),
        } if prompt_lengths or response_lengths else None,
    }
    
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Print a formatted summary to stdout."""
    print("=" * 70)
    print("RLM TRACE ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Analysis Time: {summary.get('analysis_timestamp', 'N/A')}")
    print()
    
    print("CALL STATISTICS")
    print("-" * 70)
    print(f"Total LLM Calls: {summary['total_llm_calls']}")
    print(f"  - Single calls (llm_query): {summary['call_types']['single']}")
    print(f"  - Batched calls (llm_query_batched): {summary['call_types']['batched']}")
    print()
    
    print("SUCCESS RATE")
    print("-" * 70)
    sr = summary['success_rate']
    print(f"Successful: {sr['successful']}")
    print(f"Failed: {sr['failed']}")
    print(f"Success Rate: {sr['rate_percent']}%")
    if summary.get('errors'):
        print("\nError Breakdown:")
        for error_type, count in summary['errors'].items():
            print(f"  - {error_type}: {count}")
    print()
    
    print("TIMING STATISTICS (ms)")
    print("-" * 70)
    timing = summary['timing_ms']
    print(f"Average: {timing['average']}")
    print(f"Min: {timing['min']}")
    print(f"Max: {timing['max']}")
    print(f"P50: {timing['p50']}")
    print(f"P95: {timing['p95']}")
    print(f"P99: {timing['p99']}")
    print(f"Total Duration: {timing['total_duration_ms']:,} ms ({timing['total_duration_ms']/1000:.2f} s)")
    print()
    
    print("MODEL USAGE")
    print("-" * 70)
    for model, count in summary['models_used'].items():
        percentage = (count / summary['total_llm_calls'] * 100) if summary['total_llm_calls'] > 0 else 0
        print(f"  {model}: {count} calls ({percentage:.1f}%)")
    print()
    
    print("DEPTH DISTRIBUTION")
    print("-" * 70)
    for depth, count in sorted(summary['depth_distribution'].items()):
        print(f"  Depth {depth}: {count} calls")
    print()
    
    print("SESSION STATISTICS")
    print("-" * 70)
    ss = summary['session_stats']
    print(f"Total Sessions: {ss['total_sessions']}")
    print(f"Avg Calls per Session: {ss['avg_calls_per_session']}")
    print()
    
    if summary.get('batch_stats'):
        print("BATCH STATISTICS")
        print("-" * 70)
        bs = summary['batch_stats']
        print(f"Total Batched Calls: {bs['total_batched_calls']}")
        print(f"Average Batch Size: {bs['avg_batch_size']}")
        print(f"Total Items in Batches: {bs['total_items_in_batches']}")
        print()
    
    if summary.get('token_stats'):
        print("TOKEN STATISTICS")
        print("-" * 70)
        ts = summary['token_stats']
        print(f"Calls with Token Data: {ts['calls_with_token_data']}")
        print(f"Total Prompt Tokens: {ts['total_prompt_tokens']:,}")
        print(f"Total Completion Tokens: {ts['total_completion_tokens']:,}")
        print(f"Total Tokens: {ts['total_tokens']:,}")
        print()
    
    if summary.get('length_stats'):
        print("LENGTH STATISTICS")
        print("-" * 70)
        ls = summary['length_stats']
        print(f"Average Prompt Length: {ls['avg_prompt_length']}")
        print(f"Average Response Length: {ls['avg_response_length']}")
        print()
    
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze RLM trace logs (JSONL format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_traces.py /root/rlm-traces/run-20250322-143052-a1b2c3d4.jsonl
    python analyze_traces.py trace.jsonl --output summary.json
    python analyze_traces.py trace.jsonl --format json
        """
    )
    
    parser.add_argument('trace_file', help='Path to the JSONL trace file')
    parser.add_argument('-o', '--output', help='Output file for JSON summary')
    parser.add_argument('-f', '--format', choices=['text', 'json'], default='text',
                       help='Output format (default: text)')
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.trace_file).exists():
        print(f"Error: File not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse and analyze
    print(f"Parsing trace file: {args.trace_file}", file=sys.stderr)
    events = parse_trace_file(args.trace_file)
    print(f"Found {len(events)} events", file=sys.stderr)
    
    summary = analyze_traces(events)
    
    # Output results
    if args.format == 'json':
        output = json.dumps(summary, indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Summary written to: {args.output}")
        else:
            print(output)
    else:
        print_summary(summary)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nJSON summary written to: {args.output}")


if __name__ == '__main__':
    main()
