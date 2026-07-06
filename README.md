# GPU LLM Router

A minimal, queue-based LLM inference routing prototype. Client requests are
queued in Redis, picked up by a worker, and forwarded to a single vLLM
instance running on a GPU. The backend exposes a unified API for submitting
inference requests and reading back GPU / vLLM / queue metrics.

Built to demonstrate a queuing architecture in front of a GPU-backed LLM —
one GPU, one model, async request handling instead of blocking on inference.

## Architecture

```
Client → POST /infer → Backend → Redis Queue → Worker → vLLM → GPU
                          ↓                       ↓
                    returns request_id      stores result in Redis
                          ↓
Client → GET /status/{request_id} → Backend → returns result

GET /metrics → Backend → scrapes DCGM (GPU) + vLLM + Redis queue → unified JSON
```

- **vLLM** serves `Qwen/Qwen2.5-7B-Instruct` (alias `qwen`) via an
  OpenAI-compatible REST API, with a full GPU allocated to it.
- **Redis** is the queue between the API and the worker — `/infer` never
  blocks on the model, it just enqueues and returns a `request_id`.
- **FastAPI backend** receives requests, runs the queue worker as an asyncio
  background task in the same process, and exposes `/infer`, `/status`, and
  `/metrics`.
- **DCGM Exporter** (assumed pre-installed via the NVIDIA GPU Operator)
  supplies live GPU hardware metrics (utilization, temperature, power, VRAM).

## API

### `POST /infer`

Queues a prompt for inference. Returns immediately — does not wait for the
model.

Request:
```json
{
  "prompt": "string",
  "max_tokens": 256,
  "temperature": 0.7
}
```

Response:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "result": null,
  "latency_ms": null
}
```

### `GET /status/{request_id}`

Poll until `status` is `completed` or `failed`.

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "text": "A GPU (Graphics Processing Unit) is a...",
    "tokens_generated": 45,
    "finish_reason": "stop",
    "tokens_per_second": 62.7
  },
  "latency_ms": 718.4
}
```

### `GET /metrics`

Unified snapshot combining GPU hardware metrics (DCGM), vLLM's own
Prometheus metrics, and live Redis queue depth.

```json
{
  "gpu": {
    "utilization_percent": 0,
    "temperature_c": 42,
    "power_usage_w": 14.08,
    "memory_used_mib": 90145,
    "memory_free_mib": 7104
  },
  "vllm": {
    "requests_running": 0,
    "requests_waiting": 0,
    "kv_cache_usage_perc": 0.0,
    "last_tokens_per_second": 62.7
  },
  "queue": {
    "depth": 0
  }
}
```

`memory_used_mib` stays high even at idle — vLLM pre-allocates most of the
GPU's VRAM for the KV cache on startup. This is expected, not a leak.

## Getting Started

### Prerequisites

- `kubectl` configured against a Kubernetes cluster
- An NVIDIA GPU node in that cluster with the **NVIDIA GPU Operator**
  installed (device plugin + DCGM Exporter) — this is what exposes
  `nvidia.com/gpu` as a schedulable resource and provides the GPU metrics
  endpoint the backend scrapes
- Enough GPU VRAM to load Qwen2.5-7B-Instruct (~24GB+ recommended — vLLM
  will pre-allocate most of whatever is available for its KV cache)

### Clone the repo

```bash
git clone https://github.com/<your-username>/gpu-llm-router.git
cd gpu-llm-router
```

### Repo layout

```
gpu-llm-router/
├── backend/
│   ├── main.py           # FastAPI app + Redis queue worker
│   └── requirements.txt
├── k8s/
│   ├── vllm.yaml         # vLLM Deployment + Service
│   ├── redis.yaml        # Redis Deployment + Service
│   └── api.yaml          # Backend Deployment + Service
└── README.md
```

## Deployment

With the repo cloned and the prerequisites above in place, deploy the three
components in order:

```bash
# 1. Deploy vLLM (downloads the model on first start — can take several minutes)
kubectl apply -f k8s/vllm.yaml

# 2. Deploy Redis
kubectl apply -f k8s/redis.yaml

# 3. Deploy the backend
#    The backend's code is injected via ConfigMap rather than a custom
#    Docker image, so build the ConfigMap first (run from the repo root):
kubectl create configmap backend-code \
  --from-file=main.py=backend/main.py \
  --from-file=requirements.txt=backend/requirements.txt

kubectl apply -f k8s/api.yaml
```

Verify everything is running:
```bash
kubectl get pods
kubectl get svc
```

## Example Usage

The backend is a `ClusterIP` service, so from outside the cluster you'll
need a port-forward first:

```bash
kubectl port-forward svc/backend-service 8000:8000
```

**Send a request:**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a GPU is in one sentence.", "max_tokens": 100}'
```
```json
{"request_id": "86c380fe-0a7f-45e6-a4a9-3e6cb9672081", "status": "queued", "result": null, "latency_ms": null}
```

**Check the result** (using the `request_id` from above):
```bash
curl http://localhost:8000/status/86c380fe-0a7f-45e6-a4a9-3e6cb9672081
```
```json
{
  "request_id": "86c380fe-0a7f-45e6-a4a9-3e6cb9672081",
  "status": "completed",
  "result": {
    "text": "A GPU (Graphics Processing Unit) is a specialized processor designed to handle parallel computations efficiently.",
    "tokens_generated": 21,
    "finish_reason": "stop",
    "tokens_per_second": 58.3
  },
  "latency_ms": 360.2
}
```

**Check live metrics:**
```bash
curl http://localhost:8000/metrics | python3 -m json.tool
```
```json
{
  "gpu": {
    "utilization_percent": 0,
    "temperature_c": 42,
    "power_usage_w": 14.08,
    "memory_used_mib": 90145,
    "memory_free_mib": 7104
  },
  "vllm": {
    "requests_running": 0,
    "requests_waiting": 0,
    "kv_cache_usage_perc": 0.0,
    "last_tokens_per_second": 58.3
  },
  "queue": {
    "depth": 0
  }
}
```

## Tech Stack

- **vLLM** — OpenAI-compatible inference server
- **Redis** — request queue
- **FastAPI** + **httpx** + **redis-py (asyncio)** — backend API and worker
- **NVIDIA DCGM Exporter** — GPU hardware metrics
