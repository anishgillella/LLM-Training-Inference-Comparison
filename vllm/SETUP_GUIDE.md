# Setup & Configuration Guide

## Environment Variables

All configuration is managed through environment variables. This is the recommended approach for production deployments.

### Server Configuration

Set these before deploying to Modal:

```bash
# Model to deploy from HuggingFace Hub
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"

# Specific model revision (optional, defaults to latest)
export VLLM_MODEL_REVISION="main"

# API Key for authentication (IMPORTANT: Change in production!)
export VLLM_API_KEY="your-secure-api-key-here"

# Number of GPUs (H100 recommended)
export VLLM_N_GPU=1

# Enable fast startup (recommended: true)
export VLLM_FAST_BOOT=true

# Server port
export VLLM_PORT=8000

# Keep server alive this many minutes after last request
export VLLM_SCALEDOWN_MINUTES=10

# Container startup timeout in minutes
export VLLM_TIMEOUT_MINUTES=10
```

### Supported Models from HuggingFace

Popular models you can use:

```bash
# Small models (recommended for testing)
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"
export VLLM_MODEL_NAME="Qwen/Qwen3-1B"

# Medium models
export VLLM_MODEL_NAME="Qwen/Qwen3-3B"
export VLLM_MODEL_NAME="Qwen/Qwen3-7B"

# Other popular models
export VLLM_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1"
export VLLM_MODEL_NAME="meta-llama/Llama-2-7b-hf"
export VLLM_MODEL_NAME="tiiuae/falcon-7b-instruct"
```

## Option 1: Using `.env` File

Create a `.env` file in the project root:

```bash
# vllm_inference.py configuration
VLLM_MODEL_NAME=Qwen/Qwen3-0.6B
VLLM_API_KEY=sk-my-secret-key-12345
VLLM_N_GPU=1
VLLM_FAST_BOOT=true
VLLM_SCALEDOWN_MINUTES=10

# Client configuration
VLLM_BASE_URL=https://your-workspace--vllm-qwen-inference-serve.modal.run
```

Then load it before deployment:

```bash
# Unix/Linux/Mac
source .env

# Windows (PowerShell)
Get-Content .env | ForEach-Object { $_.split('=') | ForEach-Object { if ($_) { [Environment]::SetEnvironmentVariable($_.split('=')[0], $_.split('=')[1]) } } }
```

Or use the `python-dotenv` library:

```python
from dotenv import load_dotenv
load_dotenv()

# Now all env vars are loaded
```

## Option 2: Export Variables Directly

```bash
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"
export VLLM_API_KEY="sk-my-secret-key-12345"
export VLLM_N_GPU=1

modal deploy vllm_inference.py
```

## Option 3: Command Line

```bash
VLLM_MODEL_NAME="Qwen/Qwen3-0.6B" \
VLLM_API_KEY="sk-my-secret-key-12345" \
modal deploy vllm_inference.py
```

## Deployment with Custom Configuration

### Example: Deploy Qwen 1B Model with Custom API Key

```bash
export VLLM_MODEL_NAME="Qwen/Qwen3-1B"
export VLLM_API_KEY="sk-prod-key-$(date +%s)"
export VLLM_FAST_BOOT=true
export VLLM_SCALEDOWN_MINUTES=15

modal deploy vllm_inference.py
```

### Example: Deploy Larger Model with Multiple GPUs

```bash
export VLLM_MODEL_NAME="Qwen/Qwen3-7B"
export VLLM_N_GPU=2  # Use 2 GPUs for tensor parallelism
export VLLM_FAST_BOOT=false  # Disable for better performance
export VLLM_TIMEOUT_MINUTES=20  # Allow more startup time

modal deploy vllm_inference.py
```

## Client Configuration

### Using Environment Variables

```bash
# Set your endpoint and API key
export VLLM_BASE_URL="https://your-workspace--vllm-qwen-inference-serve.modal.run"
export VLLM_API_KEY="sk-my-secret-key-12345"
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"

# Run client with env vars
python client_example.py
```

### Using Command Line Arguments

```bash
python client_example.py \
  https://your-workspace--vllm-qwen-inference-serve.modal.run \
  --api-key sk-my-secret-key-12345 \
  --model Qwen/Qwen3-0.6B \
  --stream
```

### Using `.env` File

Create `.env`:

```
VLLM_BASE_URL=https://your-workspace--vllm-qwen-inference-serve.modal.run
VLLM_API_KEY=sk-my-secret-key-12345
VLLM_MODEL_NAME=Qwen/Qwen3-0.6B
```

Then in Python:

```python
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=f"{os.getenv('VLLM_BASE_URL')}/v1",
    api_key=os.getenv('VLLM_API_KEY'),
)

response = client.chat.completions.create(
    model=os.getenv('VLLM_MODEL_NAME'),
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## Production Best Practices

### 1. Secure API Keys

Never hardcode API keys. Use environment variables:

```bash
# Bad ❌
API_KEY = "sk-prod-key-12345"

# Good ✅
API_KEY = os.getenv("VLLM_API_KEY")
```

### 2. Use `.env.local` for Local Development

```bash
# Create .env.local (not version controlled)
echo "VLLM_API_KEY=sk-dev-key-12345" > .env.local
```

### 3. Different Keys for Different Environments

```bash
# Development
export VLLM_API_KEY="sk-dev-key-..."

# Staging
export VLLM_API_KEY="sk-staging-key-..."

# Production
export VLLM_API_KEY="sk-prod-key-..."
```

### 4. Modal Secrets

For production, use Modal's secrets management:

```bash
# Create a secret in Modal
modal secret create vllm-secrets VLLM_API_KEY="sk-prod-key-12345"

# Reference in vllm_inference.py
@app.function(
    secrets=[modal.Secret.from_name("vllm-secrets")]
)
```

### 5. Use `.gitignore` for `.env` Files

```bash
# Already configured in .gitignore
.env
.env.local
.env.*.local
```

## Deployment Checklist

Before deploying to production:

- [ ] Set `VLLM_API_KEY` to a strong, unique key
- [ ] Choose appropriate model with `VLLM_MODEL_NAME`
- [ ] Set `VLLM_N_GPU` based on model size
- [ ] Adjust `VLLM_SCALEDOWN_MINUTES` for your usage pattern
- [ ] Test with `modal run vllm_inference.py test`
- [ ] Verify API key authentication works
- [ ] Monitor startup time and adjust `FAST_BOOT` if needed
- [ ] Check GPU utilization in Modal dashboard
- [ ] Set up monitoring and alerts

## Troubleshooting

### Model Not Found

```bash
# Error: Model not found on HuggingFace
export VLLM_MODEL_NAME="Qwen/Qwen3-0.6B"  # Correct format

# Not like this:
export VLLM_MODEL_NAME="qwen3"  # ❌ Wrong
```

### Environment Variables Not Picked Up

```bash
# Make sure to export them
export VLLM_API_KEY="my-key"

# Not just set locally
VLLM_API_KEY="my-key"  # ❌ Won't work

# Or use command line
VLLM_API_KEY="my-key" modal deploy vllm_inference.py  # ✅ Works
```

### API Key Authentication Fails

```python
# Check the key is correct
import os
print(f"API Key: {os.getenv('VLLM_API_KEY')}")

# Make sure to include "Bearer" in requests
headers = {"Authorization": f"Bearer {api_key}"}

# Or use OpenAI client (automatic)
client = OpenAI(api_key=api_key)
```

## Quick Reference

```bash
# Deploy with Qwen 0.6B (default)
modal deploy vllm_inference.py

# Deploy with Qwen 1B
VLLM_MODEL_NAME="Qwen/Qwen3-1B" modal deploy vllm_inference.py

# Deploy with custom API key
VLLM_API_KEY="sk-my-key" modal deploy vllm_inference.py

# Test deployment
modal run vllm_inference.py test

# View logs
modal logs vllm-qwen-inference

# Use client with env vars
export VLLM_BASE_URL="https://your-endpoint"
python client_example.py
```

---

For more information:
- [Modal Documentation](https://modal.com/docs)
- [vLLM Documentation](https://docs.vllm.ai)
- [HuggingFace Hub Models](https://huggingface.co/models)
