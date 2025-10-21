# Quick Start Guide

## 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

## 2️⃣ Deploy Server
```bash
modal deploy vllm_inference.py
```

Wait for completion (10-15 minutes on first deploy). You'll get a URL like:
```
https://your-workspace--vllm-qwen-inference-serve.modal.run
```

## 3️⃣ Test It Works
```bash
modal run vllm_inference.py test
```

## 4️⃣ Use the API

### Python
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

### cURL
```bash
curl -X POST \
  https://your-workspace--vllm-qwen-inference-serve.modal.run/v1/chat/completions \
  -H "Authorization: Bearer sk-vllm-test-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## ⚙️ Configuration

Edit `vllm_inference.py` to change:
- `MODEL_NAME` - Different model
- `API_KEY` - Your API key
- `N_GPU` - Number of GPUs
- `SCALEDOWN_MINUTES` - Cost vs latency tradeoff

## 📖 Full Guide
See `DEPLOYMENT_GUIDE.md` for detailed instructions.

## 🆘 Issues?

**Server won't start?**
```python
FAST_BOOT = True  # Already enabled
timeout = 20 * MINUTES  # Increase if needed
```

**Out of memory?**
```python
max_inputs = 8  # Reduce concurrent requests
```

**Wrong API key error?**
- Make sure API key matches `API_KEY` in `vllm_inference.py`
- Include `Authorization: Bearer <key>` header

---

**That's it! You're running an LLM inference server.** 🚀
