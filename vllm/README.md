# vLLM OpenAI-Compatible LLM Inference Server

A production-ready inference server for deploying language models with an OpenAI-compatible API using vLLM on Modal's cloud platform.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Deploy Server
```bash
modal deploy vllm_inference.py
```

### 3. Test It Works
```bash
modal run vllm_inference.py test
```

### 4. Use the API
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-workspace--vllm-qwen-inference-serve.modal.run/v1",
    api_key="sk-vllm-test-key-12345",
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Features

✅ **OpenAI-Compatible API** - Drop-in replacement for OpenAI SDK  
✅ **High Throughput** - Dynamic batching with continuous serving  
✅ **Automatic Scaling** - Modal handles auto-scaling based on load  
✅ **Fast Startup** - FAST_BOOT enabled for quick cold starts (1-2 min)  
✅ **Persistent Caching** - Model weights cached across deployments  
✅ **Streaming Support** - Server-Sent Events for real-time responses  
✅ **API Authentication** - Basic API key authentication  
✅ **Environment Configuration** - All settings via environment variables  

## Architecture

```
Modal Cloud
├── vLLM Inference Server
│   ├── Qwen3-0.6B Model (cached in HuggingFace Volume)
│   ├── OpenAI-Compatible API Endpoints
│   │   ├── /v1/chat/completions
│   │   ├── /v1/completions
│   │   └── /v1/models
│   └── GPU: H100 (80GB VRAM)
└── → HTTPS endpoints for client access
```

## Configuration via Environment Variables

All settings are configurable through environment variables. Set before deployment:

```bash
# Model Configuration
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"           # Model from HuggingFace Hub
export VLLM_MODEL_REVISION="main"                  # Optional: specific revision

# Server Configuration
export VLLM_API_KEY="sk-your-secure-api-key"      # API key (CHANGE IN PRODUCTION)
export VLLM_N_GPU=1                               # Number of H100 GPUs
export VLLM_FAST_BOOT=true                        # Enable fast startup
export VLLM_PORT=8000                             # Server port
export VLLM_SCALEDOWN_MINUTES=10                  # Keep-alive after last request
export VLLM_TIMEOUT_MINUTES=10                    # Container startup timeout
```

### Supported Models

Any model from HuggingFace Hub supported by vLLM:

```bash
# Qwen Series (Recommended)
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"
export VLLM_MODEL_NAME="Qwen/Qwen3-1B"
export VLLM_MODEL_NAME="Qwen/Qwen3-3B"
export VLLM_MODEL_NAME="Qwen/Qwen3-7B"

# Other Popular Models
export VLLM_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
export VLLM_MODEL_NAME="meta-llama/Llama-2-7b-hf"
export VLLM_MODEL_NAME="tiiuae/falcon-7b-instruct"
```

## Deployment

### Deploy with Default Configuration
```bash
modal deploy vllm_inference.py
```

### Deploy with Custom Model
```bash
VLLM_MODEL_NAME="Qwen/Qwen3-1B" \
VLLM_API_KEY="sk-prod-key-12345" \
modal deploy vllm_inference.py
```

### Deploy with Multiple GPUs
```bash
VLLM_MODEL_NAME="Qwen/Qwen3-7B" \
VLLM_N_GPU=2 \
VLLM_FAST_BOOT=false \
modal deploy vllm_inference.py
```

First deployment takes 10-15 minutes (Docker build + model download).

## Usage Examples

### Python Client (Recommended)

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="https://your-endpoint/v1",
    api_key="sk-vllm-test-key-12345",
)

# Non-streaming
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=200,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Environment Variables in Python
```python
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=f"{os.getenv('VLLM_BASE_URL')}/v1",
    api_key=os.getenv('VLLM_API_KEY'),
)
```

### cURL
```bash
curl -X POST \
  https://your-endpoint/v1/chat/completions \
  -H "Authorization: Bearer sk-vllm-test-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

### Example Client Script
```bash
export VLLM_BASE_URL="https://your-endpoint"
export VLLM_API_KEY="sk-vllm-test-key-12345"

python client_example.py --stream
```

## Project Files

| File | Purpose |
|------|---------|
| `vllm_inference.py` | Main inference server with Modal deployment |
| `client_example.py` | Example client for testing the API |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `README.md` | This file |

## Testing

### Run Built-in Tests
```bash
modal run vllm_inference.py test
```

Or with custom prompt:
```bash
modal run vllm_inference.py test "Explain machine learning"
```

### Monitor Deployment
```bash
modal logs vllm-qwen-inference
```

## Performance Tuning

### Concurrent Requests
Default: 16 concurrent requests per replica
- Increase if GPU utilization is low
- Decrease if seeing OOM (out of memory) errors

### Keep-Alive Duration
```bash
export VLLM_SCALEDOWN_MINUTES=10  # Balance: cold starts vs idle cost
```

### FAST_BOOT Mode
```bash
export VLLM_FAST_BOOT=true   # Faster startup (1-2 min), default
export VLLM_FAST_BOOT=false  # Maximum performance, slower startup
```

## Cost Estimation

Modal pricing (2025):
- **H100 GPU**: ~$4/hour (compute only, when running)
- **Storage**: ~$0.50/month per 100GB

**Example monthly costs:**
- 1 hour/day usage: ~$120
- 8 hours/day usage: ~$960
- Always on: ~$2,880

Adjust `VLLM_SCALEDOWN_MINUTES` to optimize costs.

## Troubleshooting

### Model Download Fails
```bash
# Error: Model not found on HuggingFace
# Solution: Verify model name format
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"  # ✓ Correct
export VLLM_MODEL_NAME="qwen3"            # ✗ Wrong
```

### Out of Memory (OOM)
```bash
# Solutions:
export VLLM_FAST_BOOT=true        # Disable compilation
export VLLM_N_GPU=2               # Use more GPUs
# Or reduce concurrent requests in vllm_inference.py (max_inputs)
```

### API Key Authentication Fails
```bash
# Make sure API key matches:
import os
print(f"API Key: {os.getenv('VLLM_API_KEY')}")

# Include Bearer token in requests:
headers = {"Authorization": f"Bearer {api_key}"}
```

### Container Startup Timeout
```bash
# Increase timeout:
export VLLM_TIMEOUT_MINUTES=20
```

## Production Deployment

### Secure API Key Management
```bash
# Bad ❌
API_KEY = "sk-prod-key-12345"

# Good ✅
API_KEY = os.getenv("VLLM_API_KEY")
```

### Use `.env.local` for Development
```bash
# Create .env.local (not version controlled)
echo "VLLM_API_KEY=sk-dev-key-12345" > .env.local
```

### Environment-Specific Keys
```bash
# Development
export VLLM_API_KEY="sk-dev-key-..."

# Production
export VLLM_API_KEY="sk-prod-key-..."
```

### Modal Secrets (Advanced)
```bash
# Create Modal secret
modal secret create vllm-secrets VLLM_API_KEY="sk-prod-key-12345"

# Reference in vllm_inference.py (advanced usage)
@app.function(secrets=[modal.Secret.from_name("vllm-secrets")])
```

## Common Commands

```bash
# Deploy
modal deploy vllm_inference.py

# Deploy with custom config
VLLM_MODEL_NAME="Qwen/Qwen3-1B" modal deploy vllm_inference.py

# Test
modal run vllm_inference.py test

# View logs
modal logs vllm-qwen-inference

# Remove deployment
modal remove vllm-qwen-inference

# Use client with env vars
export VLLM_BASE_URL="https://your-endpoint"
python client_example.py
```

## Next Steps

1. ✅ Deploy the server
2. ✅ Test with sample requests
3. ✅ Integrate with your application
4. ✅ Monitor performance in Modal dashboard
5. ✅ Optimize concurrency and scaling settings
6. ✅ Consider upgrading to larger model if needed

## Support & Resources

- **Modal Documentation**: https://modal.com/docs
- **vLLM Documentation**: https://docs.vllm.ai
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
- **HuggingFace Hub Models**: https://huggingface.co/models
- **Qwen Models**: https://huggingface.co/Qwen

## License

MIT (see project repository for details)

---

**Last Updated**: October 2025
