# vLLM OpenAI-Compatible LLM Inference Server

A complete guide to deploying and running the Qwen3-8B language model using vLLM on Modal with an OpenAI-compatible API interface.

## Table of Contents
- [Overview](#overview)
- [What is vLLM?](#what-is-vllm)
- [Project Architecture](#project-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Usage](#usage)
- [Testing](#testing)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Overview

This project demonstrates how to deploy a production-ready Large Language Model (LLM) inference server using vLLM on Modal's cloud platform. The server exposes an OpenAI-compatible API, making it easy to integrate with existing tools and libraries that support OpenAI's interface.

### What You'll Build
- A scalable LLM inference server running Qwen3-8B model
- OpenAI-compatible REST API endpoints
- Automatic scaling and cold-start optimization
- Persistent caching for model weights and compilation artifacts

## What is vLLM?

**vLLM** (Very Large Language Model) is an open-source library designed for fast and efficient LLM inference and serving. It provides:

- **High throughput**: Optimized for serving many requests simultaneously
- **Efficient memory management**: Uses PagedAttention to manage KV cache efficiently
- **Dynamic batching**: Automatically batches requests for better GPU utilization
- **OpenAI compatibility**: Drop-in replacement for OpenAI API endpoints
- **Multi-GPU support**: Tensor parallelism for distributing large models

### Why vLLM?
Traditional LLM inference can be slow and resource-intensive. vLLM optimizes this through:
- **PagedAttention**: Inspired by virtual memory in operating systems, efficiently manages attention key-value cache
- **Continuous batching**: Processes multiple requests in parallel without waiting for all to complete
- **Optimized kernels**: Custom CUDA kernels for faster matrix operations

## Project Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                         Modal Cloud                          │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              vLLM Inference Server                  │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │         Qwen3-8B-FP8 Model                   │  │    │
│  │  │     (Cached in HuggingFace Volume)          │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                                                      │    │
│  │  ┌─────────────────────────────────────────────┐  │    │
│  │  │      OpenAI-Compatible API Server           │  │    │
│  │  │      (Port 8000)                             │  │    │
│  │  │  - /v1/chat/completions                      │  │    │
│  │  │  - /v1/completions                           │  │    │
│  │  │  - /health                                    │  │    │
│  │  │  - /docs (Swagger UI)                        │  │    │
│  │  └─────────────────────────────────────────────┘  │    │
│  │                                                      │    │
│  │  GPU: H100 (80GB VRAM)                              │    │
│  │  Volumes: HF Cache + vLLM Cache                     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTPS
                            ▼
                    ┌──────────────┐
                    │   Clients    │
                    │  (OpenAI SDK)│
                    └──────────────┘
```

### Key Technologies

1. **Modal**: Cloud platform for running containerized applications with GPU support
2. **vLLM**: High-performance inference engine
3. **Qwen3-8B-FP8**: 8-billion parameter language model with 8-bit quantization
4. **CUDA 12.8**: GPU acceleration framework
5. **OpenAI API**: Standard interface for LLM interactions

## Prerequisites

### Knowledge Requirements
- Basic Python programming
- Understanding of REST APIs
- Familiarity with command line tools

### System Requirements
- Python 3.12+
- Modal account (sign up at [modal.com](https://modal.com))
- OpenAI Python library (for client interactions)

### Installation

1. **Install Modal CLI**:
```bash
pip install modal
```

2. **Authenticate with Modal**:
```bash
modal setup
```

3. **Install OpenAI client** (for testing):
```bash
pip install openai==1.76.0
```

4. **Clone or create the project file**:
Save the provided code as `vllm_inference.py`

## Configuration

### Container Image Setup

The project uses a custom Docker image with:
```python
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.2",
        "huggingface_hub[hf_transfer]==0.35.0",
        "flashinfer-python==0.3.1",
        "torch==2.8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

**What this does**:
- Starts with NVIDIA CUDA base image (provides GPU drivers)
- Installs Python 3.12
- Installs vLLM and dependencies using `uv` (fast package installer)
- Enables HuggingFace's fast transfer protocol for downloading models

### Model Configuration

```python
MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"
```

**Model Details**:
- **Qwen3-8B**: 8-billion parameter model with reasoning capabilities
- **FP8 Quantization**: 8-bit floating-point format reduces memory usage
- **Requires**: H100/H200/B200 GPU for native FP8 support
- **Memory**: ~8GB VRAM for model weights + KV cache

**Changing Models**: Replace `MODEL_NAME` with any HuggingFace model ID that vLLM supports.

### Volume Configuration

**HuggingFace Cache Volume**:
```python
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
```
Stores downloaded model weights to avoid re-downloading on every server restart.

**vLLM Cache Volume**:
```python
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
```
Stores JIT-compiled CUDA kernels and optimizations for faster subsequent startups.

### Performance Tuning: FAST_BOOT

```python
FAST_BOOT = True
```

**Trade-off**:
- `True`: Faster container startup (1-2 minutes), slightly lower throughput
- `False`: Slower startup (3-5 minutes), maximum throughput

**When to use**:
- `True`: Services that frequently scale to zero (cold starts)
- `False`: Services with always-running replicas

**Technical Details**:
- Controls Torch compilation and CUDA graph capture
- CUDA graphs optimize GPU kernel launches but take time to build
- Torch compilation JIT-compiles operations for better performance

### Server Configuration

```python
@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
```

**Parameters Explained**:
- `gpu=f"H100:{N_GPU}"`: Request N H100 GPUs (default: 1)
- `scaledown_window=15 * MINUTES`: Keep server alive for 15 minutes after last request
- `timeout=10 * MINUTES`: Maximum time to wait for container startup
- `max_inputs=32`: Handle up to 32 concurrent requests per replica
- `volumes`: Mount persistent storage for caches

### vLLM Server Command

```python
cmd = [
    "vllm",
    "serve",
    "--uvicorn-log-level=info",
    MODEL_NAME,
    "--revision", MODEL_REVISION,
    "--served-model-name", MODEL_NAME,
    "--host", "0.0.0.0",
    "--port", str(VLLM_PORT),
]
cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
cmd += ["--tensor-parallel-size", str(N_GPU)]
```

**Key Flags**:
- `serve`: Start OpenAI-compatible server mode
- `--enforce-eager`: Disable CUDA graphs and compilation (for FAST_BOOT)
- `--tensor-parallel-size`: Split model across multiple GPUs if N_GPU > 1
- `--host 0.0.0.0`: Listen on all network interfaces

## Deployment

### Deploy to Modal

```bash
modal deploy vllm_inference.py
```

**What happens**:
1. Modal builds the container image (first time: ~10-15 minutes)
2. Downloads model weights to HuggingFace cache volume
3. Starts vLLM server process
4. Returns a public HTTPS URL

**Output**:
```
✓ Created objects.
├── 🔨 Created mount /Users/you/project/vllm_inference.py
├── 🔨 Created image vllm_image
└── 🔨 Created function serve

View app at: https://modal.com/apps/your-workspace/example-vllm-inference

Web endpoint: https://your-workspace--example-vllm-inference-serve.modal.run
```

## Usage

### Interactive API Documentation

Visit the Swagger UI at:
```
https://your-workspace--example-vllm-inference-serve.modal.run/docs
```

This provides:
- Interactive API explorer
- Request/response schemas
- Try-it-out functionality
- cURL command generation

### Health Check

```bash
curl https://your-workspace--example-vllm-inference-serve.modal.run/health
```

**Response**:
```json
{"status": "ok"}
```

### Python Client Example

```python
from openai import OpenAI

# Point to your Modal endpoint
client = OpenAI(
    base_url="https://your-workspace--example-vllm-inference-serve.modal.run/v1",
    api_key="not-needed"  # Modal handles auth
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-FP8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### cURL Example

```bash
curl -X POST \
  https://your-workspace--example-vllm-inference-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-FP8",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

### Streaming Responses

The server supports Server-Sent Events (SSE) for streaming:

```python
async def stream_response():
    async with aiohttp.ClientSession() as session:
        payload = {
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "model": "llm",
            "stream": True
        }
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as resp:
            async for line in resp.content:
                # Process each chunk as it arrives
                print(line.decode())
```

## Testing

### Built-in Test Function

Run the included test:
```bash
modal run vllm_inference.py
```

**What it does**:
1. Spins up a fresh server replica
2. Runs health check
3. Sends two test conversations:
   - First: "You are a pirate who went to Harvard"
   - Second: "You are Jar Jar Binks"
4. Streams and prints responses
5. Shuts down after completion

**Custom test content**:
```bash
modal run vllm_inference.py --content "Explain machine learning"
```

### Load Testing

The project includes a `locust` load test setup:

```bash
modal run openai_compatible/load_test.py
```

**What it tests**:
- Concurrent request handling
- Throughput under load
- Latency percentiles
- Server stability

## Performance Optimization

### Concurrent Request Handling

```python
@modal.concurrent(max_inputs=32)
```

**Tuning Guidelines**:
- Start with 32 for 8B models on H100
- Decrease if seeing OOM (out of memory) errors
- Increase if GPU utilization is low
- Monitor with Modal's built-in metrics

### Scaling Strategy

**Scale-down window**:
```python
scaledown_window=15 * MINUTES
```
- Keeps container warm for 15 minutes after last request
- Adjust based on traffic patterns
- Longer window = less cold starts, higher idle cost

**Autoscaling**:
Modal automatically creates replicas based on:
- Queue depth
- Request rate
- `max_inputs` setting

### GPU Selection

**H100 (Recommended)**:
- 80GB VRAM
- Native FP8 support
- Best performance/cost

**A100 Alternative**:
- 40GB/80GB variants
- No native FP8 (uses emulation)
- Lower cost, slightly slower

**Change GPU**:
```python
gpu=f"A100:{N_GPU}"  # or "H200", "B200"
```

### Multi-GPU Deployment

For larger models or higher throughput:

```python
N_GPU = 2  # Split model across 2 GPUs
```

**How it works**:
- `--tensor-parallel-size` splits weight matrices
- Each GPU handles part of computation
- Synchronized via NVLink/PCIe

## Troubleshooting

### Common Issues

**1. Container startup timeout**
```
Error: Container failed to start within 10 minutes
```
**Solution**: Increase timeout or enable FAST_BOOT
```python
timeout=20 * MINUTES,
FAST_BOOT = True
```

**2. Out of Memory (OOM)**
```
CUDA out of memory
```
**Solutions**:
- Reduce `max_inputs` (fewer concurrent requests)
- Use smaller model or quantization
- Increase GPU VRAM (upgrade to H100)

**3. Cold start latency**
```
First request takes 2-3 minutes
```
**Solutions**:
- Set `FAST_BOOT = True`
- Increase `scaledown_window` to keep containers warm
- Use Modal's keep-warm feature (see docs)

**4. Model download fails**
```
Error downloading model from HuggingFace
```
**Solutions**:
- Check MODEL_NAME and MODEL_REVISION
- Verify HuggingFace Hub access
- Ensure sufficient volume space

**5. vLLM compilation errors**
```
Torch compilation failed
```
**Solutions**:
- Enable `--enforce-eager` (FAST_BOOT=True)
- Check CUDA compatibility
- Update vLLM version

### Monitoring

**Modal Dashboard**:
- View logs in real-time
- Monitor GPU utilization
- Track request latency
- See error rates

**Access logs**:
```bash
modal logs example-vllm-inference
```

## Advanced Configuration

### Custom System Prompts

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful coding assistant specializing in Python."
    },
    {
        "role": "user",
        "content": "Write a function to sort a list"
    }
]
```

### Generation Parameters

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-FP8",
    messages=messages,
    temperature=0.7,      # Creativity (0-2)
    max_tokens=500,       # Response length limit
    top_p=0.9,           # Nucleus sampling
    frequency_penalty=0.0,
    presence_penalty=0.0
)
```

### JSON Mode

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B-FP8",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)
```

## Cost Optimization

### Strategies

1. **Adjust scale-down window**: Balance cold starts vs idle cost
2. **Use FAST_BOOT**: Reduce startup time when scaling from zero
3. **Right-size GPU**: Don't use H100 if A100 suffices
4. **Batch requests**: Use vLLM's continuous batching automatically
5. **Monitor utilization**: Use Modal metrics to optimize `max_inputs`

### Estimated Costs

Modal pricing (as of 2024):
- H100: ~$4/hour
- A100 (80GB): ~$3/hour
- A100 (40GB): ~$2/hour

Factor in:
- Active compute time
- Scale-down idle time
- Storage for volumes

## Next Steps

### Extend Functionality

1. **Add authentication**: Implement API key validation
2. **Custom endpoints**: Add specialized routes for your use case
3. **Monitoring**: Integrate with observability tools
4. **Rate limiting**: Control request frequency per user
5. **Fine-tuned models**: Deploy your own trained models

### Related Resources

- [Modal Documentation](https://modal.com/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Qwen Model Card](https://huggingface.co/Qwen/Qwen3-8B-FP8)
- [Modal Examples Repository](https://github.com/modal-labs/modal-examples)

## Support

- **Modal Discord**: Community support and discussions
- **GitHub Issues**: Report bugs or request features
- **Modal Support**: Email support@modal.com

---

**License**: MIT (check project repository for details)

**Last Updated**: October 2025