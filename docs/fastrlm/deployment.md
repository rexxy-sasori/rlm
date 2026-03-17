# FastRLM Deployment Guide

This guide provides Kubernetes manifests and configuration for deploying FastRLM with SGLang Model Gateway.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Namespace Setup](#namespace-setup)
3. [SGLang Workers](#sglang-workers)
4. [Model Gateway](#model-gateway)
5. [RLM Load Test](#rlm-load-test)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

- Kubernetes cluster with GPU support (NVIDIA GPUs)
- `kubectl` configured to access your cluster
- NVIDIA GPU Operator installed
- Sufficient GPU resources (1+ GPUs for workers)

## Namespace Setup

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: fastrlm
  labels:
    app: fastrlm
    environment: production
```

```bash
kubectl apply -f namespace.yaml
```

## SGLang Workers

### ConfigMap for Worker Configuration

```yaml
# sglang-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sglang-config
  namespace: fastrlm
data:
  # Model configuration
  MODEL_NAME: "Qwen/Qwen2.5-32B-Instruct"
  TP_SIZE: "1"  # Tensor parallelism per pod
  
  # Server configuration
  PORT: "30000"
  HOST: "0.0.0.0"
  
  # Memory and performance
  MEM_FRACTION: "0.85"
  MAX_NUM_REQS: "4"
  MAX_TOTAL_TOKENS: "32768"
  
  # Logging
  LOG_LEVEL: "INFO"
```

### Deployment with Multiple Replicas

```yaml
# sglang-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-qwen32b
  namespace: fastrlm
  labels:
    app: sglang-worker
    model: qwen32b
spec:
  replicas: 3  # Adjust based on available GPUs
  selector:
    matchLabels:
      app: sglang-worker
      model: qwen32b
  template:
    metadata:
      labels:
        app: sglang-worker
        model: qwen32b
    spec:
      nodeSelector:
        nvidia.com/gpu.present: "true"  # Only schedule on GPU nodes
      
      containers:
      - name: sglang
        image: lmsysorg/sglang:latest
        command:
        - python
        - -m
        - sglang.launch_server
        args:
        - --model
        - "$(MODEL_NAME)"
        - --tp
        - "$(TP_SIZE)"
        - --port
        - "$(PORT)"
        - --host
        - "$(HOST)"
        - --mem-fraction-static
        - "$(MEM_FRACTION)"
        - --max-num-reqs
        - "$(MAX_NUM_REQS)"
        - --max-total-tokens
        - "$(MAX_TOTAL_TOKENS)"
        
        envFrom:
        - configMapRef:
            name: sglang-config
        
        ports:
        - containerPort: 30000
          name: http
          protocol: TCP
        
        resources:
          limits:
            nvidia.com/gpu: 1  # Each pod gets 1 GPU
            memory: "32Gi"
            cpu: "8"
          requests:
            memory: "16Gi"
            cpu: "4"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 60
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 30
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          runAsNonRoot: false
      
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 100Gi
      
      # Optional: Use host network for better performance
      # hostNetwork: true
      
      # Optional: Tolerations for GPU nodes
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service for Workers

```yaml
# sglang-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-qwen32b
  namespace: fastrlm
  labels:
    app: sglang-worker
    model: qwen32b
spec:
  type: ClusterIP
  selector:
    app: sglang-worker
    model: qwen32b
  ports:
  - port: 30000
    targetPort: 30000
    protocol: TCP
    name: http
```

### Horizontal Pod Autoscaler (Optional)

```yaml
# sglang-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-qwen32b
  namespace: fastrlm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sglang-qwen32b
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: sglang_queue_length
      target:
        type: AverageValue
        averageValue: "4"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

## Model Gateway

### RBAC for Service Discovery

```yaml
# gateway-rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sglang-gateway
  namespace: fastrlm
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: pod-reader
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["pods/status"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: sglang-gateway-pod-reader
subjects:
- kind: ServiceAccount
  name: sglang-gateway
  namespace: fastrlm
roleRef:
  kind: ClusterRole
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### ConfigMap for Gateway Configuration

```yaml
# gateway-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gateway-config
  namespace: fastrlm
data:
  config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8080
      request_timeout_seconds: 300
    
    # Worker discovery via Kubernetes
    service_discovery:
      enabled: true
      selector:
        app: sglang-worker
        model: qwen32b
      namespace: fastrlm
      port: 30000
      check_interval: 30s
      health_check:
        enabled: true
        path: /health
        interval: 10s
        timeout: 5s
    
    # Routing configuration
    router:
      routing_mode: CacheAware
      
      # Extract cache key from AST analysis
      cache_key_extractor:
        type: body_path
        path: extra_body.ast_analysis.cache_key
      
      # Priority queue
      priority:
        enabled: true
        extractor:
          type: body_path
          path: extra_body.routing_hints.priority
        default: normal
        levels:
          - high
          - normal
          - low
    
    # Batching configuration
    batching:
      enabled: true
      max_batch_size: 4
      max_wait_ms: 20
      group_by:
        - cache_key
        - complexity_range
      complexity_extractor:
        type: body_path
        path: extra_body.ast_analysis.complexity_score
    
    # Request queue
    queue:
      max_size: 1000
      timeout_seconds: 300
    
    # Reliability
    reliability:
      circuit_breaker:
        enabled: true
        failure_threshold: 5
        success_threshold: 3
        timeout_seconds: 30
      
      retry:
        enabled: true
        max_attempts: 3
        backoff: exponential
        initial_delay_ms: 100
        max_delay_ms: 1000
    
    # Observability
    observability:
      metrics:
        enabled: true
        port: 9090
        path: /metrics
      
      tracing:
        enabled: true
        exporter: otlp
        endpoint: http://jaeger:4317
      
      logging:
        level: info
        format: json
```

### Gateway Deployment

```yaml
# gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-gateway
  namespace: fastrlm
  labels:
    app: sglang-gateway
spec:
  replicas: 1  # Gateway is stateless, can scale horizontally
  selector:
    matchLabels:
      app: sglang-gateway
  template:
    metadata:
      labels:
        app: sglang-gateway
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: sglang-gateway
      
      containers:
      - name: gateway
        image: lmsysorg/sgl-model-gateway:latest
        command:
        - sglang-router
        - launch
        - --config
        - /config/config.yaml
        
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        
        volumeMounts:
        - name: config
          mountPath: /config
          readOnly: true
        
        env:
        - name: RUST_LOG
          value: "info"
        - name: RUST_BACKTRACE
          value: "1"
      
      volumes:
      - name: config
        configMap:
          name: gateway-config
```

### Gateway Service

```yaml
# gateway-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-gateway
  namespace: fastrlm
  labels:
    app: sglang-gateway
spec:
  type: ClusterIP
  selector:
    app: sglang-gateway
  ports:
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: http
  - port: 9090
    targetPort: 9090
    protocol: TCP
    name: metrics
```

### Gateway Ingress (Optional)

```yaml
# gateway-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sglang-gateway
  namespace: fastrlm
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - fastrlm.example.com
    secretName: fastrlm-tls
  rules:
  - host: fastrlm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sglang-gateway
            port:
              number: 8080
```

## RLM Load Test

### ConfigMap for Load Test Configuration

```yaml
# loadtest-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: loadtest-config
  namespace: fastrlm
data:
  # Root LLM configuration
  ROOT_MODEL: "@openai/gpt-4o"
  ROOT_BACKEND: "portkey"
  
  # Sub-LLM configuration (via gateway)
  SUB_MODEL: "qwen32b"
  SUB_BASE_URL: "http://sglang-gateway:8080/v1"
  
  # Test configuration
  NUM_SAMPLES: "100"
  CONCURRENCY: "10"
  MAX_DEPTH: "2"
  MAX_ITERATIONS: "5"
  
  # Enable AST analysis
  ENABLE_AST_ANALYSIS: "true"
  
  # Dataset configuration
  DATASET_SUBSET: "dnd"
  DATASET_SPLIT: "validation"
```

### Load Test Job

```yaml
# loadtest-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: fastrlm-load-test
  namespace: fastrlm
  labels:
    app: fastrlm-load-test
spec:
  parallelism: 1
  completions: 1
  template:
    metadata:
      labels:
        app: fastrlm-load-test
    spec:
      restartPolicy: OnFailure
      
      containers:
      - name: load-test
        image: your-registry/fastrlm:latest  # Build from Dockerfile
        command:
        - python
        - experiments/oolong_load_test.py
        args:
        - --num-samples
        - "$(NUM_SAMPLES)"
        - --concurrency
        - "$(CONCURRENCY)"
        - --max-depth
        - "$(MAX_DEPTH)"
        - --max-iterations
        - "$(MAX_ITERATIONS)"
        - --base-url
        - "$(SUB_BASE_URL)"
        - --model
        - "$(SUB_MODEL)"
        - --enable-ast-analysis
        
        envFrom:
        - configMapRef:
            name: loadtest-config
        
        env:
        - name: PORTKEY_API_KEY
          valueFrom:
            secretKeyRef:
              name: fastrlm-secrets
              key: portkey-api-key
        
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
        
        volumeMounts:
        - name: results
          mountPath: /results
      
      volumes:
      - name: results
        emptyDir: {}
```

### Dockerfile for Load Test

```dockerfile
# Dockerfile.loadtest
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install RLM package
COPY . .
RUN pip install -e .

# Set environment
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "experiments/oolong_load_test.py"]
```

## Monitoring

### Prometheus ServiceMonitor

```yaml
# prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sglang-gateway
  namespace: fastrlm
  labels:
    app: sglang-gateway
spec:
  selector:
    matchLabels:
      app: sglang-gateway
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

### Grafana Dashboard ConfigMap

```yaml
# grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fastrlm-dashboard
  namespace: fastrlm
  labels:
    grafana_dashboard: "true"
data:
  fastrlm.json: |
    {
      "dashboard": {
        "title": "FastRLM Metrics",
        "panels": [
          {
            "title": "Request Rate",
            "targets": [
              {
                "expr": "rate(sglang_gateway_requests_total[1m])"
              }
            ]
          },
          {
            "title": "Latency (P95)",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, sglang_gateway_request_duration_seconds_bucket)"
              }
            ]
          },
          {
            "title": "Batch Size",
            "targets": [
              {
                "expr": "sglang_gateway_batch_size"
              }
            ]
          },
          {
            "title": "Cache Hit Rate",
            "targets": [
              {
                "expr": "sglang_gateway_cache_hit_rate"
              }
            ]
          },
          {
            "title": "Worker Health",
            "targets": [
              {
                "expr": "sglang_worker_health"
              }
            ]
          }
        ]
      }
    }
```

## Troubleshooting

### Check Worker Status

```bash
# Check if workers are running
kubectl get pods -n fastrlm -l app=sglang-worker

# Check worker logs
kubectl logs -n fastrlm -l app=sglang-worker --tail=100

# Check worker resources
kubectl top pods -n fastrlm -l app=sglang-worker
```

### Check Gateway Status

```bash
# Check gateway pod
kubectl get pods -n fastrlm -l app=sglang-gateway

# Check gateway logs
kubectl logs -n fastrlm -l app=sglang-gateway --tail=100

# Check gateway metrics
curl http://sglang-gateway:9090/metrics
```

### Test Connectivity

```bash
# Port forward gateway
kubectl port-forward -n fastrlm svc/sglang-gateway 8080:8080

# Test health endpoint
curl http://localhost:8080/health

# Test chat completions
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen32b",
    "messages": [{"role": "user", "content": "Hello"}],
    "extra_body": {
      "ast_analysis": {
        "complexity_score": 0.1,
        "cache_key": "test123"
      }
    }
  }'
```

### Common Issues

#### Workers Not Discovered

```bash
# Check RBAC
kubectl auth can-i list pods --as=system:serviceaccount:fastrlm:sglang-gateway

# Check gateway logs for discovery errors
kubectl logs -n fastrlm -l app=sglang-gateway | grep -i discovery
```

#### GPU Out of Memory

```bash
# Check GPU memory usage
kubectl exec -n fastrlm -it <worker-pod> -- nvidia-smi

# Reduce batch size or memory fraction
kubectl edit configmap -n fastrlm sglang-config
```

#### High Latency

```bash
# Check queue depth
curl http://sglang-gateway:9090/metrics | grep queue

# Check worker load
kubectl top pods -n fastrlm
```

## Cleanup

```bash
# Delete all resources
kubectl delete namespace fastrlm

# Or delete individual components
kubectl delete -f sglang-deployment.yaml
kubectl delete -f gateway-deployment.yaml
kubectl delete -f loadtest-job.yaml
```

## References

- [Architecture Overview](architecture.md)
- [Implementation Guide](implementation.md)
- [SGLang Documentation](https://docs.sglang.io/)
- [Kubernetes GPU Guide](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
