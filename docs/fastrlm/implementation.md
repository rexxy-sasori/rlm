# FastRLM Implementation Guide

This guide provides code examples for implementing AST-aware routing in RLM with SGLang Model Gateway.

## Table of Contents

1. [AST Analyzer](#ast-analyzer)
2. [Extended LocalREPL](#extended-localrepl)
3. [RLM Integration](#rlm-integration)
4. [Gateway Configuration](#gateway-configuration)
5. [Client Usage](#client-usage)

## AST Analyzer

The AST Analyzer performs code analysis using Python's built-in `ast` module.

```python
# rlm/utils/ast_analyzer.py
import ast
import hashlib
from typing import Dict, Any, List, Set
from dataclasses import dataclass


@dataclass
class ASTAnalysis:
    """Result of AST analysis for a code block."""
    complexity_score: float  # 0.0 to 1.0
    cache_key: str  # For radix attention
    estimated_input_tokens: int
    estimated_output_tokens: int
    dependencies: List[str]  # Imported modules
    has_loops: bool
    has_functions: bool
    has_classes: bool
    tree_structure: str  # Serialized AST for debugging
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "complexity_score": self.complexity_score,
            "cache_key": self.cache_key,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "dependencies": self.dependencies,
            "has_loops": self.has_loops,
            "has_functions": self.has_functions,
            "has_classes": self.has_classes,
            "tree_structure": self.tree_structure,
        }


class ASTAnalyzer:
    """Analyzes Python code to extract metadata for intelligent routing."""
    
    def analyze(self, code: str) -> ASTAnalysis:
        """Perform full AST analysis on code block."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return default analysis for invalid code
            return ASTAnalysis(
                complexity_score=0.5,
                cache_key=self._hash_code(code),
                estimated_input_tokens=len(code.split()),
                estimated_output_tokens=100,
                dependencies=[],
                has_loops=False,
                has_functions=False,
                has_classes=False,
                tree_structure="",
            )
        
        return ASTAnalysis(
            complexity_score=self._calc_complexity(tree),
            cache_key=self._compute_cache_key(tree),
            estimated_input_tokens=self._estimate_input_tokens(code),
            estimated_output_tokens=self._estimate_output_tokens(tree),
            dependencies=self._extract_imports(tree),
            has_loops=self._has_loops(tree),
            has_functions=self._has_functions(tree),
            has_classes=self._has_classes(tree),
            tree_structure=ast.dump(tree, include_attributes=False),
        )
    
    def _calc_complexity(self, tree: ast.AST) -> float:
        """Calculate complexity score based on AST structure."""
        score = 0.0
        
        for node in ast.walk(tree):
            # Control flow increases complexity
            if isinstance(node, (ast.For, ast.While)):
                score += 0.15
            elif isinstance(node, ast.If):
                score += 0.05
            elif isinstance(node, ast.Try):
                score += 0.1
            # Functions and classes
            elif isinstance(node, ast.FunctionDef):
                score += 0.08
            elif isinstance(node, ast.ClassDef):
                score += 0.12
            # List/dict comprehensions
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                score += 0.1
            # Lambda functions
            elif isinstance(node, ast.Lambda):
                score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _compute_cache_key(self, tree: ast.AST) -> str:
        """Generate cache key from normalized AST structure."""
        # Normalize by removing variable names and literals
        normalized = self._normalize_tree(tree)
        
        # Hash with blake2b for speed
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(normalized.encode())
        return hasher.hexdigest()
    
    def _normalize_tree(self, tree: ast.AST) -> str:
        """Normalize AST by replacing variable names and literals."""
        class Normalizer(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.Name(id="_var_", ctx=node.ctx)
            
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    return ast.Constant(value="_str_")
                elif isinstance(node.value, (int, float)):
                    return ast.Constant(value=0)
                return node
        
        normalized_tree = Normalizer().visit(tree)
        return ast.dump(normalized_tree, include_attributes=False)
    
    def _estimate_input_tokens(self, code: str) -> int:
        """Estimate input token count from code."""
        # Rough estimate: ~0.75 tokens per word for code
        words = len(code.split())
        return int(words * 0.75) + 50  # +50 for system prompt
    
    def _estimate_output_tokens(self, tree: ast.AST) -> int:
        """Estimate output token count based on complexity."""
        # More complex code → longer explanation
        complexity = self._calc_complexity(tree)
        base_tokens = 100
        max_tokens = 500
        return int(base_tokens + complexity * (max_tokens - base_tokens))
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract imported module names."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
        
        return list(set(imports))  # Remove duplicates
    
    def _has_loops(self, tree: ast.AST) -> bool:
        """Check if code contains loops."""
        return any(
            isinstance(node, (ast.For, ast.While, ast.ListComp, ast.DictComp, ast.SetComp))
            for node in ast.walk(tree)
        )
    
    def _has_functions(self, tree: ast.AST) -> bool:
        """Check if code contains function definitions."""
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            for node in ast.walk(tree)
        )
    
    def _has_classes(self, tree: ast.AST) -> bool:
        """Check if code contains class definitions."""
        return any(
            isinstance(node, ast.ClassDef)
            for node in ast.walk(tree)
        )
    
    def _hash_code(self, code: str) -> str:
        """Fallback hash for invalid code."""
        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(code.encode())
        return hasher.hexdigest()


# Singleton instance for reuse
_default_analyzer = None

def get_analyzer() -> ASTAnalyzer:
    """Get or create default AST analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = ASTAnalyzer()
    return _default_analyzer


def analyze_code(code: str) -> Dict[str, Any]:
    """Convenience function for quick analysis."""
    return get_analyzer().analyze(code).to_dict()
```

## Extended LocalREPL

Extend the LocalREPL to perform AST analysis before sub-LLM calls.

```python
# rlm/environments/ast_aware_repl.py
from typing import Any, Dict, Optional
from rlm.environments.local_repl import LocalREPL
from rlm.utils.ast_analyzer import ASTAnalyzer, analyze_code


class ASTAwareLocalREPL(LocalREPL):
    """LocalREPL with AST analysis for intelligent routing."""
    
    def __init__(
        self,
        *args,
        enable_ast_analysis: bool = True,
        ast_analyzer: Optional[ASTAnalyzer] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.enable_ast_analysis = enable_ast_analysis
        self.ast_analyzer = ast_analyzer or ASTAnalyzer()
    
    def rlm_query(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Query sub-LLM with optional AST analysis.
        
        If AST analysis is enabled and the prompt contains code,
        extracts and analyzes the code, then includes metadata
        in the request to the gateway.
        """
        if self.enable_ast_analysis:
            code = self._extract_code_from_prompt(prompt)
            if code:
                ast_analysis = self.ast_analyzer.analyze(code)
                
                # Add to extra_body for gateway routing
                extra_body = kwargs.get("extra_body", {})
                extra_body["ast_analysis"] = ast_analysis.to_dict()
                extra_body["routing_hints"] = {
                    "priority": self._determine_priority(),
                    "session_id": getattr(self, "session_id", "unknown"),
                    "iteration_depth": getattr(self, "depth", 0),
                }
                kwargs["extra_body"] = extra_body
        
        # Call parent implementation
        return super().rlm_query(prompt, **kwargs)
    
    def _extract_code_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract code block from prompt if present."""
        # Look for code blocks
        import re
        
        # Match fenced code blocks
        code_block_pattern = r"```(?:python)?\n(.*?)```"
        matches = re.findall(code_block_pattern, prompt, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Match inline code
        inline_pattern = r"`([^`]+)`"
        matches = re.findall(inline_pattern, prompt)
        
        if matches:
            # Return longest inline code block
            return max(matches, key=len).strip()
        
        # Check if entire prompt looks like code
        if self._looks_like_code(prompt):
            return prompt.strip()
        
        return None
    
    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to determine if text is Python code."""
        code_indicators = [
            "def ",
            "class ",
            "import ",
            "from ",
            "for ",
            "while ",
            "if ",
            "print(",
            "return ",
        ]
        return any(indicator in text for indicator in code_indicators)
    
    def _determine_priority(self) -> str:
        """Determine request priority based on context."""
        depth = getattr(self, "depth", 0)
        if depth <= 1:
            return "high"  # Early iterations get priority
        elif depth <= 3:
            return "normal"
        else:
            return "low"


# Factory function for easy creation
def create_ast_aware_repl(
    lm_handler_address: tuple,
    enable_ast_analysis: bool = True,
    **kwargs
) -> ASTAwareLocalREPL:
    """Create an AST-aware LocalREPL instance."""
    return ASTAwareLocalREPL(
        lm_handler_address=lm_handler_address,
        enable_ast_analysis=enable_ast_analysis,
        **kwargs
    )
```

## RLM Integration

Modify RLM to use AST-aware environment.

```python
# rlm/core/rlm.py (modifications)

class RLM:
    def __init__(
        self,
        # ... existing params ...
        enable_ast_analysis: bool = False,
        **kwargs
    ):
        # ... existing init ...
        self.enable_ast_analysis = enable_ast_analysis
    
    def _spawn_completion_context(self, prompt):
        """Spawn environment with AST analysis support."""
        # ... existing code ...
        
        if self.environment_type == "local" and self.enable_ast_analysis:
            from rlm.environments.ast_aware_repl import ASTAwareLocalREPL
            
            env_kwargs["enable_ast_analysis"] = True
            env_kwargs["ast_analyzer"] = None  # Use default
            environment = ASTAwareLocalREPL(**env_kwargs)
        else:
            environment = get_environment(self.environment_type, env_kwargs)
        
        # ... rest of method ...
```

## Gateway Configuration

Configure SGLang Model Gateway to use AST metadata.

```yaml
# gateway-config.yaml
server:
  host: "0.0.0.0"
  port: 8080

# Worker configuration
workers:
  # Kubernetes service discovery
  discovery:
    enabled: true
    selector:
      app: sglang-worker
    namespace: default
    port: 30000
    check_interval: 30s

# Routing configuration
router:
  # Use cache-aware routing for radix attention
  routing_mode: CacheAware
  
  # Extract cache key from AST analysis
  cache_key_header: x-ast-cache-key
  cache_key_body_path: extra_body.ast_analysis.cache_key
  
  # Priority queue configuration
  priority:
    enabled: true
    header: x-routing-priority
    body_path: extra_body.routing_hints.priority
    levels:
      - high
      - normal
      - low

# Batching configuration
batching:
  enabled: true
  max_batch_size: 4
  max_wait_ms: 20
  
  # Group by complexity for efficient batching
  group_by:
    - cache_key
    - complexity_range
  
  complexity_ranges:
    - name: "simple"
      min: 0.0
      max: 0.3
    - name: "medium"
      min: 0.3
      max: 0.7
    - name: "complex"
      min: 0.7
      max: 1.0

# Queue configuration
queue:
  max_size: 1000
  timeout_seconds: 300

# Reliability
reliability:
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout_seconds: 30
  
  retry:
    enabled: true
    max_attempts: 3
    backoff: exponential

# Observability
observability:
  metrics:
    enabled: true
    port: 9090
  
  tracing:
    enabled: true
    exporter: otlp
```

## Client Usage

### Basic Usage

```python
from rlm import RLM

# Create RLM with AST-aware routing
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "gpt-4o",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    other_backends=["openai"],
    other_backend_kwargs=[{
        "model_name": "qwen32b",
        "base_url": "http://gateway:8080/v1",
    }],
    environment="local",
    max_depth=2,
    enable_ast_analysis=True,  # Enable AST routing
)

# Use normally - AST analysis happens automatically
result = rlm.completion(prompt)
```

### Load Test with AST Routing

```python
# experiments/oolong_load_test_ast.py
import asyncio
from rlm import RLM
from rlm.environments.ast_aware_repl import create_ast_aware_repl

async def run_with_ast_routing():
    """Run load test with AST-aware routing."""
    
    # Configure RLM with AST analysis
    rlm = RLM(
        backend="portkey",
        backend_kwargs={
            "model_name": "@openai/gpt-4o",
            "api_key": os.getenv("PORTKEY_API_KEY"),
        },
        other_backends=["openai"],
        other_backend_kwargs=[{
            "model_name": "qwen32b",
            "base_url": "http://gateway:8080/v1",
        }],
        environment="local",
        max_depth=2,
        enable_ast_analysis=True,
    )
    
    # Run load test
    manager = AsyncParallelManager(
        rlm=rlm,
        max_concurrency=10,
    )
    
    metrics = await manager.run_load_test(samples)
    print(f"Success rate: {metrics.success_rate:.2f}%")
    print(f"Mean latency: {metrics.mean_latency:.2f}s")

if __name__ == "__main__":
    asyncio.run(run_with_ast_routing())
```

### Direct Client Usage

```python
from openai import OpenAI
from rlm.utils.ast_analyzer import analyze_code

# Create client pointing to gateway
client = OpenAI(
    base_url="http://gateway:8080/v1",
    api_key="not-needed",
)

# Analyze code
code = """
import numpy as np

def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(np.sqrt(item))
    return result
"""

ast_analysis = analyze_code(code)

# Send request with AST metadata
response = client.chat.completions.create(
    model="qwen32b",
    messages=[
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": f"Explain this code:\n```python\n{code}\n```"}
    ],
    extra_body={
        "ast_analysis": ast_analysis,
        "routing_hints": {
            "priority": "high",
            "session_id": "session_001",
        }
    }
)

print(response.choices[0].message.content)
```

## Testing

```python
# tests/test_ast_analyzer.py
import pytest
from rlm.utils.ast_analyzer import ASTAnalyzer, analyze_code


def test_simple_code():
    code = "x = 1 + 2"
    result = analyze_code(code)
    
    assert result["complexity_score"] < 0.1
    assert result["has_loops"] is False
    assert result["has_functions"] is False
    assert len(result["cache_key"]) == 16


def test_complex_code():
    code = """
import numpy as np

def process(items):
    result = []
    for item in items:
        if item > 0:
            result.append(np.sqrt(item))
    return result
"""
    result = analyze_code(code)
    
    assert result["complexity_score"] > 0.3
    assert result["has_loops"] is True
    assert result["has_functions"] is True
    assert "numpy" in result["dependencies"]


def test_cache_key_consistency():
    """Same structure should produce same cache key."""
    code1 = "x = 1 + 2"
    code2 = "y = 3 + 4"
    
    result1 = analyze_code(code1)
    result2 = analyze_code(code2)
    
    # Same structure → same cache key
    assert result1["cache_key"] == result2["cache_key"]


def test_cache_key_uniqueness():
    """Different structures should produce different cache keys."""
    code1 = "x = 1 + 2"
    code2 = "for i in range(10): print(i)"
    
    result1 = analyze_code(code1)
    result2 = analyze_code(code2)
    
    # Different structure → different cache key
    assert result1["cache_key"] != result2["cache_key"]
```

## Performance Tuning

### AST Analysis Caching

```python
from functools import lru_cache

class CachedASTAnalyzer(ASTAnalyzer):
    """AST Analyzer with LRU cache for repeated code."""
    
    @lru_cache(maxsize=1000)
    def analyze(self, code: str) -> ASTAnalysis:
        """Cached analysis - same code returns cached result."""
        return super().analyze(code)
```

### Async Analysis

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncASTAnalyzer(ASTAnalyzer):
    """Async-capable AST analyzer."""
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def analyze_async(self, code: str) -> ASTAnalysis:
        """Analyze code asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.analyze,
            code
        )
```

## References

- [Architecture Overview](architecture.md)
- [Deployment Guide](deployment.md)
- [SGLang Model Gateway Docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html)
