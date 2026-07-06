import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REDIS_URL = "redis://redis-service:6379"
VLLM_URL = "http://vllm-service:8000"
DCGM_URL = "http://nvidia-dcgm-exporter.gpu-operator:9400/metrics"
QUEUE_KEY = "inference:queue"
RESULT_TTL = 3600

redis_client = None


class InferRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class InferResult(BaseModel):
    text: str
    tokens_generated: int
    finish_reason: str #completed naturally,token length reached or cancelled
    tokens_per_second: float


class InferResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[InferResult] = None #Set to None as filed empty when request is queued
    latency_ms: Optional[float] = None


async def worker_loop():
    while True:
        try:
            item = await redis_client.brpop(QUEUE_KEY, timeout=1)
            if item is None:
                continue

            _, raw = item
            request = json.loads(raw)
            request_id = request["request_id"]

            await redis_client.set(f"inference:status:{request_id}", "processing")

            start = time.time()
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{VLLM_URL}/v1/chat/completions",
                    json={
                        "model": "qwen",
                        "messages": [{"role": "user", "content": request["prompt"]}],
                        "max_tokens": request["max_tokens"],
                        "temperature": request["temperature"],
                    },
                )
            latency_ms = (time.time() - start) * 1000
            data = response.json()
            choice = data["choices"][0]
            tokens_generated = data["usage"]["completion_tokens"]

            tokens_per_second = round(tokens_generated / (latency_ms / 1000), 2)
            await redis_client.set("metrics:last_tokens_per_second", tokens_per_second)

            await redis_client.set(
                f"inference:result:{request_id}",
                json.dumps({
                    "status": "completed",
                    "result": {
                        "text": choice["message"]["content"],
                        "tokens_generated": tokens_generated,
                        "finish_reason": choice["finish_reason"],
                        "tokens_per_second": tokens_per_second,
                    },
                    "latency_ms": latency_ms,
                }),
                ex=RESULT_TTL,
            )

        except Exception as e:
            print(f"Worker error: {e}")
            if "request_id" in locals():
                await redis_client.set(
                    f"inference:result:{request_id}",
                    json.dumps({"status": "failed", "error": str(e)}),
                    ex=RESULT_TTL,
                )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL)
    asyncio.create_task(worker_loop())
    yield
    await redis_client.aclose()


app = FastAPI(title="GPU LLM Router", lifespan=lifespan)


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    request_id = str(uuid.uuid4())
    await redis_client.lpush(
        QUEUE_KEY,
        json.dumps({
            "request_id": request_id,
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }),
    )
    await redis_client.set(f"inference:status:{request_id}", "queued")
    return InferResponse(request_id=request_id, status="queued")


@app.get("/status/{request_id}", response_model=InferResponse)
async def get_status(request_id: str):
    raw = await redis_client.get(f"inference:result:{request_id}")
    if raw:
        data = json.loads(raw)
        return InferResponse(
            request_id=request_id,
            status=data["status"],
            result=data.get("result"),
            latency_ms=data.get("latency_ms"),
        )

    status = await redis_client.get(f"inference:status:{request_id}")
    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")

    return InferResponse(request_id=request_id, status=status.decode())


@app.get("/metrics")
async def get_metrics():
    queue_depth = await redis_client.llen(QUEUE_KEY)

    last_tps_raw = await redis_client.get("metrics:last_tokens_per_second")
    last_tps = float(last_tps_raw) if last_tps_raw else None

    vllm_data = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{VLLM_URL}/metrics")
        vllm_data = parse_vllm_metrics(r.text)
    except Exception as e:
        vllm_data = {"error": str(e)}

    gpu_data = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(DCGM_URL)
        gpu_data = parse_gpu_metrics(r.text)
    except Exception as e:
        gpu_data = {"error": str(e)}

    return {
        "gpu": gpu_data,
        "vllm": {**vllm_data, "last_tokens_per_second": last_tps},
        "queue": {"depth": queue_depth},
    }


def parse_vllm_metrics(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("vllm:num_requests_running{"):
            result["requests_running"] = float(line.split()[-1])
        elif line.startswith("vllm:num_requests_waiting{"):
            result["requests_waiting"] = float(line.split()[-1])
        elif line.startswith("vllm:kv_cache_usage_perc{"):
            result["kv_cache_usage_perc"] = float(line.split()[-1])
    return result


def parse_gpu_metrics(text: str) -> dict:
    result = {}
    used = free = reserved = None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("DCGM_FI_DEV_GPU_UTIL{"):
            result["utilization_percent"] = float(line.split()[-1])
        elif line.startswith("DCGM_FI_DEV_GPU_TEMP{"):
            result["temperature_c"] = float(line.split()[-1])
        elif line.startswith("DCGM_FI_DEV_POWER_USAGE{"):
            result["power_usage_w"] = round(float(line.split()[-1]), 2)
        elif line.startswith("DCGM_FI_DEV_FB_USED{"):
            used = float(line.split()[-1])
            result["memory_used_mib"] = used
        elif line.startswith("DCGM_FI_DEV_FB_FREE{"):
            free = float(line.split()[-1])
            result["memory_free_mib"] = free
        elif line.startswith("DCGM_FI_DEV_FB_RESERVED{"):
            reserved = float(line.split()[-1])

    if used is not None and free is not None and reserved is not None:
        result["memory_total_mib"] = used + free + reserved

    return result
