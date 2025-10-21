# Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Deploy to Modal

Deploy the inference server to Modal:

```bash
modal deploy vllm_inference.py
```

**First-time deployment may take 10-15 minutes:**
- Docker image builds (~5 minutes)
- Model weights download (~5-10 minutes)
- vLLM compilation (~2-5 minutes)

**Expected output:**
```
✓ Created objects.
├── 🔨 Created function serve
├── 🔨 Created image vllm_image

Web endpoint: https://your-workspace--vllm-qwen-inference-serve.modal.run
```

### 3. Test the Deployment

Run the built-in test function:

```bash
modal run vllm_inference.py test
```

Or with a custom prompt:

```bash
modal run vllm_inference.py test "Explain machine learning in simple terms"
```

### 4. Use the Server

#### Option A: Python Client (Recommended)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-workspace--vllm-qwen-inference-serve.modal.run/v1",
    api_key="sk-vllm-test-key-12345",  # Must match API_KEY in vllm_inference.py
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
)

print(response.choices[0].message.content)
```

#### Option B: cURL

```bash
curl -X POST \
  https://your-workspace--vllm-qwen-inference-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm-test-key-12345" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

#### Option C: Example Client

```bash
python client_example.py https://your-workspace--vllm-qwen-inference-serve.modal.run --stream
```

## Configuration Reference

### Model Settings

In `vllm_inference.py`:

```python
MODEL_NAME = "Qwen/Qwen3-0.6B"           # Model to deploy
N_GPU = 1                               # Number of H100 GPUs
FAST_BOOT = True                        # Fast startup (recommended)
API_KEY = "sk-vllm-test-key-12345"      # API key for auth
SCALEDOWN_MINUTES = 10                  # Keep-alive after last request
```

### API Key

The API key is sent in requests as:

```bash
Authorization: Bearer sk-vllm-test-key-12345
```

Or with OpenAI client:

```python
client = OpenAI(api_key="sk-vllm-test-key-12345")
```

## Monitoring

### View Logs

```bash
modal logs vllm-qwen-inference
```

### View Real-time Metrics

Visit: https://modal.com/apps (dashboard)

### Get Server Status

```python
import requests

response = requests.get(
    "https://your-workspace--vllm-qwen-inference-serve.modal.run/v1/models",
    headers={"Authorization": "Bearer sk-vllm-test-key-12345"}
)
print(response.json())
```

## Troubleshooting

### Server Won't Start

**Error:** `Container failed to start within 10 minutes`

**Solution:** Increase timeout or enable faster startup
```python
timeout=20 * MINUTES
FAST_BOOT = True
```

### Model Download Fails

**Error:** `Error downloading model from HuggingFace`

**Solution:**
- Check model name (e.g., "Qwen/Qwen3-0.6B" is correct)
- Verify HuggingFace Hub access
- Check volume storage space

### Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solution:**
- Reduce `max_inputs` (fewer concurrent requests)
- Use a smaller model
- Upgrade GPU (H100 has 80GB VRAM)

### API Key Authentication Fails

**Error:** `403 Unauthorized`

**Solution:**
- Verify API key matches `API_KEY` in `vllm_inference.py`
- Include `Authorization: Bearer` header
- Or use OpenAI client with correct `api_key`

## Performance Tips

1. **Concurrent Requests**: `max_inputs=16` is tuned for 0.6B model
   - Increase if GPU utilization is low
   - Decrease if seeing OOM errors

2. **Streaming**: Use streaming for faster time-to-first-token
   - Stream=True for real-time responses
   - Stream=False for simple requests

3. **Batch Requests**: vLLM automatically batches concurrent requests
   - No manual batching needed
   - Higher concurrency = better GPU utilization

4. **Keep-Warm**: Adjust scale-down window
   - Longer window = less cold starts, higher idle cost
   - Shorter window = cheaper, slower cold starts

## Stopping the Server

```bash
modal remove vllm-qwen-inference
```

## Upgrading the Model

To use a different model, update in `vllm_inference.py`:

```python
MODEL_NAME = "Qwen/Qwen3-1B"  # or any other HuggingFace model
```

Then redeploy:

```bash
modal deploy vllm_inference.py
```

## Cost Estimation

Modal pricing (2025):
- H100: ~$4/hour (compute only, when running)
- Storage: ~$0.50/month per 100GB

**Example monthly cost:**
- 1 hour/day usage: ~$120
- 8 hours/day usage: ~$960
- Always on: ~$2,880

Adjust `SCALEDOWN_MINUTES` to balance cost vs latency.

## Next Steps

1. ✅ Deploy the server
2. ✅ Test with sample requests
3. ✅ Integrate with your application
4. ✅ Monitor performance in Modal dashboard
5. ✅ Optimize concurrency and scale-down settings
6. ✅ Consider upgrading to larger model (1B, 7B, etc.)

## Support

- **Modal Docs**: https://modal.com/docs
- **vLLM Docs**: https://docs.vllm.ai
- **OpenAI API Reference**: https://platform.openai.com/docs/api-reference
