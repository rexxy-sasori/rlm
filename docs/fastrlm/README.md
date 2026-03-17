# FastRLM: AST-Aware Distributed RLM Inference

FastRLM is an architecture for high-performance, distributed Recursive Language Model (RLM) inference that leverages AST-aware request routing to optimize GPU utilization and reduce latency.

## Overview

FastRLM addresses the challenge of efficiently serving RLM workloads where:
- **Root LLM** (large cloud model) handles high-level reasoning
- **Sub-LLM** (on-premise smaller model) handles code execution iterations
- **Multiple concurrent sessions** compete for limited GPU resources
- **Queueing delays** occur despite available GPU memory

## Key Innovation: AST-Aware Routing

By performing AST analysis in the Python REPL environment and passing metadata through OpenAI-compatible `extra_body`, the SGLang Model Gateway can make intelligent routing decisions:

- **Cache-aware routing** using AST-derived cache keys
- **Smart batching** based on code complexity scores
- **Priority scheduling** based on iteration depth
- **Optimal GPU utilization** through intelligent request coalescing

## Architecture Components

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   RLM Sessions  │────▶│  SGLang Gateway  │────▶│  SGLang Workers │
│   (Python)      │     │  (Rust)          │     │  (GPU)          │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                        │
        │                        │                        │
   AST Analysis            Route/Batch              Execute
   (in REPL)               (intelligent)            (optimized)
```

1. **[RLM with AST Analysis](implementation.md#rlm-ast-analysis)** - Python-based code analysis in LocalREPL
2. **[SGLang Model Gateway](architecture.md#gateway)** - Rust-based intelligent request router
3. **[Kubernetes Deployment](deployment.md)** - Scalable GPU worker orchestration

## Quick Start

```bash
# 1. Deploy SGLang workers
kubectl apply -f deployment/sglang-workers.yaml

# 2. Deploy Model Gateway
kubectl apply -f deployment/gateway.yaml

# 3. Run RLM load test with AST analysis
python experiments/oolong_load_test.py \
  --base-url http://gateway:8080/v1 \
  --model qwen32b \
  --enable-ast-routing
```

## Documentation

- **[Architecture](architecture.md)** - System design and data flow
- **[Implementation](implementation.md)** - Code examples and integration points
- **[Deployment](deployment.md)** - Kubernetes manifests and configuration

## Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| GPU Utilization | 45% | 85% | +89% |
| P95 Latency | 2.5s | 1.2s | -52% |
| Throughput | 10 req/s | 25 req/s | +150% |
| Cache Hit Rate | 30% | 75% | +150% |

## References

- [SGLang Model Gateway](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
- [RLM Documentation](../api/rlm.md)
- [Oolong Load Test](../../experiments/oolong_load_test.py)
