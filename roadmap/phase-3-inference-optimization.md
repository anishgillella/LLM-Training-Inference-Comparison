# Phase 3: Inference Optimization (Weeks 6-7)

> Master techniques to make your models faster, cheaper, and more efficient in production

---

## 🎯 Phase Overview

In this phase, you'll learn how to optimize LLM inference for real-world deployment:

1. **Decoding Strategies** - Control generation quality and diversity
2. **Flash Attention & KV Cache** - Speed up attention computation
3. **Production Serving** - Deploy with vLLM or Text Generation Inference

**Duration:** 2-3 weeks part-time
**Cost:** $0-10
**Hardware:** Google Colab Free or Pro
**Prerequisites:** A trained model from Phase 1 or 2

---

## 📋 Stage 8: Decoding Strategies

### Goal
Understand and implement different text generation strategies to balance quality, diversity, and speed.

### What You'll Learn
- Greedy decoding vs sampling
- Temperature scaling
- Top-k and nucleus (top-p) sampling
- Beam search
- Repetition penalties
- How decoding affects alignment

### Why Decoding Matters

The same model can produce very different outputs based on decoding strategy:

```python
Prompt: "The future of AI is"

Greedy: "the future of AI is the future of the world."  # Boring, repetitive
Temperature 0.7: "the future of AI is bright and full of possibilities."  # Balanced
Temperature 1.5: "the future of AI is quantum banana moonwalk!"  # Too random
```

### Tasks Checklist
- [ ] Implement greedy decoding (baseline)
- [ ] Try temperature sampling (0.3, 0.7, 1.0, 1.5)
- [ ] Implement top-k sampling (k=10, 50)
- [ ] Implement nucleus (top-p) sampling (p=0.9, 0.95)
- [ ] Try beam search (beam size = 4, 8)
- [ ] Add repetition penalty
- [ ] Compare quality vs diversity tradeoffs
- [ ] Find best settings for your use case

### Decoding Methods Explained

#### 1. Greedy Decoding
```python
# Always pick the most likely token
def greedy_decode(model, prompt, max_length=50):
    for _ in range(max_length):
        logits = model(prompt)
        next_token = argmax(logits)  # Pick highest probability
        prompt = append(prompt, next_token)
    return prompt

# Pros: Fast, deterministic
# Cons: Boring, repetitive, no diversity
```

#### 2. Temperature Sampling
```python
# Scale logits to control randomness
def temperature_sample(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits)
    next_token = sample(probs)
    return next_token

# Temperature = 0.1: Almost greedy (very focused)
# Temperature = 0.7: Balanced (recommended default)
# Temperature = 1.0: Unmodified probabilities
# Temperature = 1.5: More random and creative
```

#### 3. Top-k Sampling
```python
# Only sample from top k most likely tokens
def top_k_sample(logits, k=50):
    top_k_logits, top_k_indices = topk(logits, k)
    probs = softmax(top_k_logits)
    next_token_idx = sample(probs)
    next_token = top_k_indices[next_token_idx]
    return next_token

# k = 1: Greedy
# k = 10: Very focused
# k = 50: Balanced
# k = 100+: More diverse
```

#### 4. Nucleus (Top-p) Sampling
```python
# Sample from smallest set of tokens with cumulative prob > p
def nucleus_sample(logits, p=0.9):
    sorted_probs, sorted_indices = sort(softmax(logits))
    cumsum = cumulative_sum(sorted_probs)
    
    # Find cutoff where cumsum exceeds p
    nucleus_mask = cumsum <= p
    nucleus_probs = sorted_probs[nucleus_mask]
    
    # Renormalize and sample
    nucleus_probs = nucleus_probs / sum(nucleus_probs)
    next_token_idx = sample(nucleus_probs)
    return sorted_indices[next_token_idx]

# p = 0.9: Focused (recommended)
# p = 0.95: Balanced
# p = 0.99: More diverse
```

#### 5. Beam Search
```python
# Keep top N candidate sequences at each step
def beam_search(model, prompt, beam_size=4):
    beams = [(prompt, 0.0)]  # (sequence, score)
    
    for _ in range(max_length):
        new_beams = []
        for seq, score in beams:
            logits = model(seq)
            top_k_probs, top_k_tokens = topk(softmax(logits), beam_size)
            
            for prob, token in zip(top_k_probs, top_k_tokens):
                new_seq = append(seq, token)
                new_score = score + log(prob)
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1])[-beam_size:]
    
    return beams[0][0]  # Return best sequence

# Pros: Better quality for specific tasks (translation, summarization)
# Cons: Slower, less diverse, can be repetitive
```

### Code Example - Comparing Decoding Strategies

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./dpo_final")
tokenizer = AutoTokenizer.from_pretrained("./dpo_final")

prompt = "The future of artificial intelligence is"

# 1. Greedy Decoding
output_greedy = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    do_sample=False  # Greedy
)

# 2. Temperature Sampling
output_temp07 = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    do_sample=True,
    temperature=0.7
)

output_temp15 = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    do_sample=True,
    temperature=1.5
)

# 3. Top-k Sampling
output_topk = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    do_sample=True,
    top_k=50,
    temperature=1.0
)

# 4. Nucleus (Top-p) Sampling
output_topp = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

# 5. Beam Search
output_beam = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    num_beams=4,
    early_stopping=True
)

# 6. Combined: Top-p + Temperature + Repetition Penalty (Recommended!)
output_best = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,  # Penalize repetitions
    no_repeat_ngram_size=3  # No 3-gram repetition
)

# Print all outputs
for name, output in [
    ("Greedy", output_greedy),
    ("Temp 0.7", output_temp07),
    ("Temp 1.5", output_temp15),
    ("Top-k", output_topk),
    ("Top-p", output_topp),
    ("Beam", output_beam),
    ("Combined", output_best)
]:
    print(f"\n{name}:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("-" * 60)
```

### Recommended Settings by Use Case

| Use Case | Temperature | Top-p | Top-k | Beam | Repetition Penalty |
|----------|-------------|-------|-------|------|-------------------|
| **Factual Q&A** | 0.1-0.3 | - | - | - | 1.0 |
| **Creative Writing** | 0.8-1.2 | 0.95 | - | - | 1.1 |
| **Code Generation** | 0.2-0.5 | 0.9 | 50 | - | 1.0 |
| **Chatbot** | 0.7 | 0.9 | - | - | 1.2 |
| **Translation** | 0.3 | - | - | 4-8 | 1.0 |
| **Summarization** | 0.5 | 0.9 | - | 4 | 1.1 |

### Evaluation Script

```python
def evaluate_decoding_strategy(model, tokenizer, test_prompts, config):
    """Evaluate quality and diversity of different decoding strategies"""
    
    outputs = []
    for prompt in test_prompts:
        output = model.generate(
            **tokenizer(prompt, return_tensors="pt"),
            **config  # decoding parameters
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(text)
    
    # Metrics
    diversity = compute_self_bleu(outputs)  # Lower = more diverse
    repetition = compute_repetition_rate(outputs)  # Lower = less repetitive
    quality = compute_perplexity(model, outputs)  # Lower = better
    
    return {
        "diversity": diversity,
        "repetition": repetition,
        "quality": quality,
        "examples": outputs[:3]
    }

# Compare strategies
strategies = {
    "greedy": {"do_sample": False},
    "temp_0.7": {"do_sample": True, "temperature": 0.7},
    "top_p": {"do_sample": True, "top_p": 0.9, "temperature": 0.7},
    "beam": {"num_beams": 4},
}

for name, config in strategies.items():
    results = evaluate_decoding_strategy(model, tokenizer, test_prompts, config)
    print(f"\n{name}: Diversity={results['diversity']:.2f}, Repetition={results['repetition']:.2%}, Quality={results['quality']:.2f}")
```

### Success Criteria
- ✅ Understand when to use each decoding method
- ✅ Can tune parameters for desired output characteristics
- ✅ Know recommended settings for different tasks
- ✅ Can evaluate quality vs diversity tradeoffs

### Estimated Time
2-3 days

---

## 📋 Stage 9: Flash Attention & KV Cache Optimization

### Goal
Understand and implement attention optimizations that make inference 2-4x faster with 3-10x less memory.

### What You'll Learn
- Why attention is the bottleneck
- Key-Value (KV) cache mechanics
- Flash Attention algorithm
- Memory-efficient attention
- Batching for throughput

### The Attention Bottleneck

Standard attention has O(n²) complexity:

```python
# Standard self-attention (simplified)
Q = input @ W_q  # Query
K = input @ W_k  # Key
V = input @ W_v  # Value

# O(n²) memory and computation!
attention_weights = softmax(Q @ K.T / sqrt(d))
output = attention_weights @ V

# For 2048 tokens: 2048² = 4M operations
# For 4096 tokens: 4096² = 16M operations (4x more!)
```

### Key-Value (KV) Cache

**Problem:** In autoregressive generation, we recompute attention for all previous tokens every time.

**Solution:** Cache the Key and Value tensors!

```python
# Without KV cache:
# Step 1: Compute attention for token 1
# Step 2: Compute attention for tokens 1-2 (recompute token 1!)
# Step 3: Compute attention for tokens 1-3 (recompute tokens 1-2!)
# ...

# With KV cache:
# Step 1: Compute K1, V1, cache them
# Step 2: Compute K2, V2, reuse K1, V1 from cache
# Step 3: Compute K3, V3, reuse K1, V1, K2, V2 from cache
# ...

# Result: 2-3x faster generation!
```

### Flash Attention

Flash Attention reorganizes attention computation to:
- Minimize memory reads/writes (use faster SRAM vs slower HBM)
- Compute attention in blocks
- Avoid materializing the full attention matrix

**Result:** 2-4x speedup, 3-10x less memory!

### Tasks Checklist
- [ ] Enable KV cache in your model
- [ ] Measure speedup from KV cache
- [ ] Install Flash Attention 2
- [ ] Enable Flash Attention in model
- [ ] Benchmark with and without Flash Attention
- [ ] Try different batch sizes
- [ ] Measure memory usage

### Code Example - Enabling KV Cache

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model = AutoModelForCausalLM.from_pretrained("./dpo_final")
tokenizer = AutoTokenizer.from_pretrained("./dpo_final")

prompt = "Write a story about a robot learning to"

# Benchmark WITHOUT KV cache
start = time.time()
output_no_cache = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=200,
    use_cache=False  # Disable KV cache
)
time_no_cache = time.time() - start

# Benchmark WITH KV cache (default)
start = time.time()
output_with_cache = model.generate(
    **tokenizer(prompt, return_tensors="pt"),
    max_length=200,
    use_cache=True  # Enable KV cache
)
time_with_cache = time.time() - start

print(f"Without KV cache: {time_no_cache:.2f}s")
print(f"With KV cache: {time_with_cache:.2f}s")
print(f"Speedup: {time_no_cache / time_with_cache:.2f}x")

# Expected: 2-3x speedup with KV cache
```

### Code Example - Flash Attention 2

```python
# Install Flash Attention 2
# !pip install flash-attn --no-build-isolation

from transformers import AutoModelForCausalLM
import torch

# Load model with Flash Attention 2
model_flash = AutoModelForCausalLM.from_pretrained(
    "./dpo_final",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Enable Flash Attention!
    device_map="auto"
)

# Benchmark
prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Warmup
for _ in range(5):
    _ = model_flash.generate(**inputs, max_length=100)

# Standard attention
model_standard = AutoModelForCausalLM.from_pretrained(
    "./dpo_final",
    torch_dtype=torch.float16,
    device_map="auto"
)

import time

# Time standard attention
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    _ = model_standard.generate(**inputs, max_length=100)
torch.cuda.synchronize()
time_standard = (time.time() - start) / 20

# Time Flash Attention
torch.cuda.synchronize()
start = time.time()
for _ in range(20):
    _ = model_flash.generate(**inputs, max_length=100)
torch.cuda.synchronize()
time_flash = (time.time() - start) / 20

print(f"Standard Attention: {time_standard:.3f}s per generation")
print(f"Flash Attention 2: {time_flash:.3f}s per generation")
print(f"Speedup: {time_standard / time_flash:.2f}x")

# Expected: 1.5-3x speedup depending on sequence length
```

### Memory Profiling

```python
import torch

def profile_memory(model, tokenizer, prompt, max_length=200):
    """Measure peak memory usage during generation"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return peak_memory_mb

# Compare memory usage
mem_standard = profile_memory(model_standard, tokenizer, prompt)
mem_flash = profile_memory(model_flash, tokenizer, prompt)

print(f"Standard Attention: {mem_standard:.0f} MB")
print(f"Flash Attention 2: {mem_flash:.0f} MB")
print(f"Memory Savings: {(1 - mem_flash/mem_standard)*100:.1f}%")

# Expected: 20-50% memory savings with Flash Attention
```

### Batch Inference for Throughput

```python
def batch_generate(model, tokenizer, prompts, batch_size=4):
    """Generate for multiple prompts in batches"""
    
    outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Tokenize batch (with padding)
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        
        # Generate for entire batch at once
        batch_outputs = model.generate(
            **inputs,
            max_length=100,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Decode
        texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        outputs.extend(texts)
    
    return outputs

# Benchmark sequential vs batched
prompts = ["Write a poem about AI" for _ in range(20)]

# Sequential
start = time.time()
for prompt in prompts:
    _ = model.generate(**tokenizer(prompt, return_tensors="pt").to("cuda"), max_length=50)
time_sequential = time.time() - start

# Batched (batch_size=4)
start = time.time()
_ = batch_generate(model, tokenizer, prompts, batch_size=4)
time_batched = time.time() - start

print(f"Sequential: {time_sequential:.2f}s ({len(prompts)/time_sequential:.2f} prompts/sec)")
print(f"Batched: {time_batched:.2f}s ({len(prompts)/time_batched:.2f} prompts/sec)")
print(f"Speedup: {time_sequential / time_batched:.2f}x")

# Expected: 2-4x throughput improvement with batching
```

### Success Criteria
- ✅ KV cache enabled and providing 2-3x speedup
- ✅ Flash Attention installed and working
- ✅ Measurable speedup from Flash Attention (1.5-3x)
- ✅ Understanding of memory/speed tradeoffs
- ✅ Batch inference working for throughput

### Estimated Time
3-4 days

---

## 📋 Stage 10: Production Serving with vLLM

### Goal
Deploy your model for production inference using state-of-the-art serving infrastructure.

### What You'll Learn
- vLLM and Text Generation Inference (TGI)
- Continuous batching
- PagedAttention
- Serving APIs
- Throughput optimization
- Production best practices

### Why vLLM/TGI?

**Standard HuggingFace Transformers:**
- Single request at a time
- Naive batching
- ~20-50 tokens/sec

**vLLM or TGI:**
- Continuous batching (handle requests as they arrive)
- PagedAttention (KV cache memory management)
- Optimized kernels
- ~200-500 tokens/sec

**Result: 10-20x better throughput!**

### Tasks Checklist
- [ ] Install vLLM
- [ ] Set up basic serving endpoint
- [ ] Test single requests
- [ ] Test concurrent requests
- [ ] Benchmark throughput vs HuggingFace
- [ ] Try different batch sizes
- [ ] Measure latency vs throughput tradeoffs

### Code Example - vLLM Setup

```bash
# Install vLLM
pip install vllm
```

```python
from vllm import LLM, SamplingParams

# 1. Load model with vLLM
llm = LLM(
    model="./dpo_final",
    tensor_parallel_size=1,  # Number of GPUs
    max_model_len=2048,  # Max context length
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
)

# 2. Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# 3. Single prompt
prompt = "Explain quantum computing in simple terms."
output = llm.generate(prompt, sampling_params)
print(output[0].outputs[0].text)

# 4. Batch of prompts (this is where vLLM shines!)
prompts = [
    "What is machine learning?",
    "How does a neural network work?",
    "Explain backpropagation.",
    "What is gradient descent?",
] * 10  # 40 prompts

import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

print(f"Generated {len(prompts)} responses in {elapsed:.2f}s")
print(f"Throughput: {len(prompts) / elapsed:.2f} prompts/sec")

# Expected: 5-20 prompts/sec depending on model size and GPU
```

### Code Example - vLLM API Server

```bash
# Start vLLM API server (OpenAI-compatible!)
python -m vllm.entrypoints.openai.api_server \
    --model ./dpo_final \
    --port 8000
```

```python
# Client code (OpenAI-compatible API!)
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy"  # vLLM doesn't need key

response = openai.Completion.create(
    model="./dpo_final",
    prompt="What is the meaning of life?",
    max_tokens=100,
    temperature=0.7,
)

print(response.choices[0].text)
```

### Throughput Benchmarking

```python
import concurrent.futures
import time
import requests

def send_request(prompt):
    """Send a single request to vLLM server"""
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "./dpo_final",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7,
        }
    )
    return response.json()

# Test concurrent requests
prompts = ["Write a short story about AI" for _ in range(100)]

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(send_request, prompts))
elapsed = time.time() - start

print(f"Completed {len(prompts)} requests in {elapsed:.2f}s")
print(f"Throughput: {len(prompts) / elapsed:.2f} req/sec")
print(f"Average latency: {elapsed / len(prompts):.2f}s per request")
```

### Comparison: HuggingFace vs vLLM

```python
# Benchmark: Standard HuggingFace Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_hf = AutoModelForCausalLM.from_pretrained("./dpo_final").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("./dpo_final")

prompts = ["Write a poem about nature" for _ in range(50)]

# HuggingFace (sequential)
start = time.time()
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    _ = model_hf.generate(**inputs, max_length=100)
time_hf = time.time() - start

# vLLM (batched with continuous batching)
from vllm import LLM, SamplingParams

llm = LLM(model="./dpo_final")
sampling_params = SamplingParams(max_tokens=100)

start = time.time()
_ = llm.generate(prompts, sampling_params)
time_vllm = time.time() - start

print(f"HuggingFace: {time_hf:.2f}s ({len(prompts)/time_hf:.2f} prompts/sec)")
print(f"vLLM: {time_vllm:.2f}s ({len(prompts)/time_vllm:.2f} prompts/sec)")
print(f"Speedup: {time_hf / time_vllm:.2f}x")

# Expected: 5-15x speedup with vLLM
```

### Production Deployment Checklist

- [ ] Set up vLLM API server
- [ ] Configure appropriate batch size
- [ ] Set GPU memory utilization (0.8-0.9)
- [ ] Add request queuing and rate limiting
- [ ] Monitor latency and throughput
- [ ] Set up logging and error handling
- [ ] Load testing with concurrent requests
- [ ] Auto-scaling based on load (optional)

### Alternative: Text Generation Inference (TGI)

```bash
# HuggingFace's official serving solution
# Docker-based, production-ready

docker run -p 8080:80 \
    --gpus all \
    --shm-size 1g \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id ./dpo_final \
    --num-shard 1
```

### Success Criteria
- ✅ vLLM server running and responding
- ✅ 5-15x throughput improvement vs HuggingFace
- ✅ Can handle concurrent requests
- ✅ Latency acceptable for use case
- ✅ Understanding of production considerations

### Estimated Time
3-4 days

---

## 📊 Phase 3 Evaluation & Comparison

### Comprehensive Benchmarking

```python
def full_inference_benchmark(model_path):
    """
    Comprehensive comparison of all inference optimizations
    """
    
    test_prompt = "Write a detailed explanation of machine learning"
    
    configs = {
        "Baseline (HF)": {
            "use_kv_cache": False,
            "flash_attention": False,
            "framework": "hf"
        },
        "HF + KV Cache": {
            "use_kv_cache": True,
            "flash_attention": False,
            "framework": "hf"
        },
        "HF + Flash Attn": {
            "use_kv_cache": True,
            "flash_attention": True,
            "framework": "hf"
        },
        "vLLM": {
            "framework": "vllm"
        },
    }
    
    results = {}
    
    for name, config in configs.items():
        # Measure latency, throughput, memory
        latency = measure_latency(model_path, test_prompt, config)
        throughput = measure_throughput(model_path, [test_prompt]*20, config)
        memory = measure_memory(model_path, test_prompt, config)
        
        results[name] = {
            "latency_ms": latency * 1000,
            "throughput_tok_per_sec": throughput,
            "memory_mb": memory,
        }
    
    return results

# Run benchmark
results = full_inference_benchmark("./dpo_final")

# Print results
print("\n" + "="*60)
print("Inference Optimization Results")
print("="*60)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Latency: {metrics['latency_ms']:.0f}ms")
    print(f"  Throughput: {metrics['throughput_tok_per_sec']:.0f} tokens/sec")
    print(f"  Memory: {metrics['memory_mb']:.0f} MB")
```

### Expected Improvements

| Optimization | Speedup | Memory Savings | Complexity |
|--------------|---------|----------------|------------|
| KV Cache | 2-3x | Minimal | ⭐ Easy |
| Flash Attention 2 | 1.5-3x | 20-50% | ⭐⭐ Medium |
| Batching | 2-4x | - | ⭐ Easy |
| vLLM | 5-15x | 2-3x | ⭐⭐ Medium |
| **Combined** | **10-30x** | **50-70%** | ⭐⭐ Medium |

### Deliverables for Phase 3

1. **Benchmarks:**
   - [ ] Decoding strategy comparison
   - [ ] KV cache speedup measurements
   - [ ] Flash Attention benchmarks
   - [ ] vLLM throughput tests

2. **Code:**
   - [ ] Inference scripts with different strategies
   - [ ] vLLM deployment setup
   - [ ] Benchmarking harness

3. **Documentation:**
   - [ ] Recommended decoding settings for each task
   - [ ] Performance comparison charts
   - [ ] Production deployment guide

---

## 🎯 Phase 3 Success Criteria

By the end of Phase 3, you should be able to:

- ✅ Choose appropriate decoding strategies for different tasks
- ✅ Enable and measure KV cache benefits
- ✅ Use Flash Attention for speedup
- ✅ Deploy models with vLLM for production
- ✅ Benchmark and optimize inference performance
- ✅ Understand latency/throughput/memory tradeoffs

---

## 🚀 What's Next?

Congratulations on completing Phase 3! Your models are now optimized for production.

**Next Steps:**
- **Phase 4:** Explore advanced topics (Speculative Decoding, Multi-modal, etc.)
- **Apply to your domain:** Build a real product with your optimized model

---

## 📚 Additional Resources for Phase 3

### Papers
- **Flash Attention:** "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- **PagedAttention:** "Efficient Memory Management for LLM Serving" (Kwon et al., 2023)

### Tools
- vLLM documentation
- Text Generation Inference (TGI)
- HuggingFace Optimum

---

Ready for Phase 4? Proceed to [Phase 4: Advanced Topics](./phase-4-advanced-topics.md)


