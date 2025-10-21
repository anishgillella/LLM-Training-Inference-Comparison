# Phase 1: Core Training Methods (Weeks 1-3)

> Learn the fundamental training techniques that power modern LLMs

---

## 🎯 Phase Overview

In this phase, you'll learn the essential training methods that form the foundation of all modern LLM work:

1. **Supervised Fine-Tuning (SFT)** - Adapt base models to specific tasks
2. **LoRA/QLoRA** - Parameter-efficient fine-tuning
3. **Knowledge Distillation** - Create smaller, faster models
4. **Quantization Basics** - Compress models for deployment

**Duration:** 3 weeks part-time (10-15 hrs/week)
**Cost:** $0-5
**Hardware:** Google Colab Free (T4 GPU)
**Model:** Start with TinyLlama-1.1B or GPT-2 Medium (355M)

---

## 📋 Stage 1: Supervised Fine-Tuning (SFT)

### Goal
Create a baseline instruction-following model by fine-tuning a base model on task-specific data.

### What You'll Learn
- Causal language modeling loss
- Tokenization and attention masks
- Gradient accumulation for memory efficiency
- Mixed precision training (FP16/BF16)
- Learning rate scheduling
- Model evaluation basics

### Tasks Checklist
- [ ] Choose your domain (summarization, dialogue, code, or rewriting)
- [ ] Load base model (TinyLlama-1.1B recommended)
- [ ] Prepare instruction dataset with proper formatting
- [ ] Set up training loop with HuggingFace Trainer
- [ ] Configure: learning rate, batch size, gradient accumulation
- [ ] Train for 1-2 epochs with validation
- [ ] Save checkpoints to Google Drive

### Code Example - SFT Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Load instruction dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:5000]")

# 3. Configure training
training_args = TrainingArguments(
    output_dir="./sft_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,  # Mixed precision for speed
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=50,
)

# 4. Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

# 5. Train!
trainer.train()

# 6. Save
model.save_pretrained("./sft_final")
tokenizer.save_pretrained("./sft_final")
```

### Success Criteria
- ✅ Model generates coherent responses to instructions
- ✅ Validation loss decreases steadily
- ✅ No overfitting on training set
- ✅ Clear improvement over base model

### Estimated Time
5-7 days (including debugging and experimentation)

### Recommended Datasets
- **Easy start:** `databricks/databricks-dolly-15k` (instruction following)
- **Dialogue:** `Anthropic/hh-rlhf` (already has preferences!)
- **Summarization:** `cnn_dailymail` or `reddit_tifu`
- **Code:** `openai_humaneval` or `mbpp`

---

## 📋 Stage 2: LoRA/QLoRA - Parameter-Efficient Fine-Tuning

### Goal
Learn to fine-tune models efficiently by training only a small fraction of parameters.

### What You'll Learn
- Low-Rank Adaptation (LoRA) architecture
- How adapter layers work
- 4-bit quantization with QLoRA
- Memory optimization techniques
- Merging adapters at inference time
- Comparison: full fine-tuning vs LoRA

### Why LoRA Matters
- ✅ Train only 0.5-2% of parameters (vs 100% in full fine-tuning)
- ✅ 3-10x less memory needed
- ✅ Similar performance to full fine-tuning (95-100%)
- ✅ Can train multiple adapters for different tasks
- ✅ Faster training and iteration

### Tasks Checklist
- [ ] Understand LoRA architecture
- [ ] Install PEFT library (`pip install peft`)
- [ ] Configure LoRA with appropriate rank (r=8, r=16)
- [ ] Train with LoRA on same task as SFT
- [ ] Compare training time and memory vs full fine-tuning
- [ ] Try QLoRA (4-bit quantization + LoRA)
- [ ] Merge adapter weights for deployment

### Code Example - LoRA Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # See how few params we're training!

# 4. Load dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:5000]")

# 5. Train (same as before, but much faster!)
training_args = TrainingArguments(
    output_dir="./lora_model",
    num_train_epochs=3,  # Can train for more epochs with LoRA
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,  # Higher LR often works better with LoRA
    fp16=True,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=training_args,
)

trainer.train()

# 6. Save (only adapter weights, very small!)
model.save_pretrained("./lora_adapter")
```

### QLoRA - Even More Efficient!

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load model in 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for training
model = prepare_model_for_kbit_training(model)

# Then apply LoRA as before...
# (rest of code is the same)
```

### Comparison Table

| Method | Trainable Params | Memory | Training Time | Final Performance |
|--------|------------------|--------|---------------|-------------------|
| Full Fine-Tuning | 100% | 12GB | 4 hours | 100% (baseline) |
| LoRA (r=16) | ~0.5% | 4GB | 1.5 hours | 97-99% |
| QLoRA (4-bit) | ~0.5% | 2GB | 2 hours | 95-98% |

### Success Criteria
- ✅ LoRA adapter file is <100MB (vs multi-GB full model)
- ✅ Training uses significantly less memory
- ✅ Performance within 5% of full fine-tuning
- ✅ Can load and merge adapter successfully

### Estimated Time
3-5 days

---

## 📋 Stage 3: Knowledge Distillation

### Goal
Train a small "student" model to mimic a large "teacher" model, achieving 5-10x speedup with only 10-20% quality loss.

### Why Distillation is Crucial
**This is how the industry actually works!**

- Meta: Llama-2-70B → Llama-2-7B → Llama-2-1.3B
- Google: Gemini Ultra → Gemini Pro → Gemini Nano
- Mistral: Mixtral-8x7B → Mistral-7B variants
- OpenAI: GPT-4 → GPT-3.5 (likely distillation-based)

### What You'll Learn
- Teacher-student training paradigm
- Soft labels vs hard labels
- Temperature scaling
- Distillation loss functions
- Quality/speed tradeoffs
- Why distillation beats training small models from scratch

### Key Concept: Soft Labels

```python
# Traditional training: Hard labels
Label: "Paris" (100% probability, 0% for others)

# Distillation: Soft labels (from teacher)
Teacher outputs with temperature T=2:
  "Paris": 0.7
  "France": 0.15
  "Lyon": 0.10
  "Berlin": 0.05

→ Student learns rich relationships, not just correct answer!
```

### Tasks Checklist
- [ ] Fine-tune teacher model (TinyLlama-1.1B)
- [ ] Choose student model (GPT-2 Medium 355M)
- [ ] Generate soft labels from teacher on training data
- [ ] Implement distillation loss (KL divergence)
- [ ] Train student on soft labels
- [ ] Compare: student vs teacher quality
- [ ] Measure inference speed difference
- [ ] Analyze quality/speed tradeoff

### Code Example - Knowledge Distillation

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 1. Load teacher (already fine-tuned)
teacher = AutoModelForCausalLM.from_pretrained("./sft_final")
teacher.eval()

# 2. Load student (smaller model)
student = AutoModelForCausalLM.from_pretrained("gpt2-medium")  # 355M params

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

# 3. Distillation training class
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, temperature=2.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get student outputs
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Distillation loss (soft targets)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Student loss (hard targets)
        student_loss = student_outputs.loss
        
        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return (loss, student_outputs) if return_outputs else loss

# 4. Train with distillation
training_args = TrainingArguments(
    output_dir="./distilled_student",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    fp16=True,
)

trainer = DistillationTrainer(
    model=student,
    teacher_model=teacher,
    temperature=2.0,
    alpha=0.5,  # Balance between soft and hard targets
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

### Evaluation: Teacher vs Student

```python
import time

# Test inference speed
def measure_speed(model, prompt, num_runs=10):
    start = time.time()
    for _ in range(num_runs):
        outputs = model.generate(prompt, max_length=100)
    end = time.time()
    return (end - start) / num_runs

teacher_speed = measure_speed(teacher, test_prompt)
student_speed = measure_speed(student, test_prompt)

print(f"Teacher: {teacher_speed:.3f}s per generation")
print(f"Student: {student_speed:.3f}s per generation")
print(f"Speedup: {teacher_speed / student_speed:.2f}x")

# Expected results:
# Teacher (1.1B): ~0.5s, 4GB VRAM
# Student (355M): ~0.15s, 1.5GB VRAM
# Speedup: ~3-4x, Quality: 85-90% of teacher
```

### Success Criteria
- ✅ Student is 3-5x faster than teacher
- ✅ Student uses 2-3x less memory
- ✅ Student achieves 80-90% of teacher quality
- ✅ Clear improvement over training student from scratch

### Real-World Impact
```
Before distillation:
- Teacher: 1.1B params, 50 tokens/sec, 8GB VRAM
- Deployment: ❌ Too slow/expensive for edge

After distillation:
- Student: 355M params, 200 tokens/sec, 2GB VRAM
- Deployment: ✅ Can run on phones, laptops, edge devices!
```

### Estimated Time
4-6 days

---

## 📋 Stage 4: Quantization Basics

### Goal
Learn to compress models by reducing numerical precision from 32-bit to 8-bit or 4-bit.

### What You'll Learn
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Different quantization formats (INT8, NF4, GPTQ)
- Memory/quality tradeoffs
- When to use which method

### Quantization Benefits

| Precision | Size | Memory | Speed | Quality Loss |
|-----------|------|--------|-------|--------------|
| FP32 (baseline) | 100% | 4GB | 1x | 0% |
| FP16 | 50% | 2GB | 1.5x | <0.1% |
| INT8 | 25% | 1GB | 2-3x | 0.5-2% |
| NF4 (4-bit) | 12.5% | 0.5GB | 2-4x | 2-5% |

### Tasks Checklist
- [ ] Install bitsandbytes (`pip install bitsandbytes`)
- [ ] Load model in 8-bit quantization
- [ ] Load model in 4-bit quantization (NF4)
- [ ] Measure memory usage for each
- [ ] Run inference benchmarks
- [ ] Evaluate quality degradation
- [ ] Compare: FP16 vs INT8 vs NF4

### Code Example - 8-bit Quantization

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load in 8-bit
model_8bit = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    load_in_8bit=True,
    device_map="auto"
)

print(f"8-bit model loaded!")
# Uses ~2x less memory than FP16
```

### Code Example - 4-bit Quantization (NF4)

```python
# Load in 4-bit with NF4 (better than standard 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Nested quantization
)

model_4bit = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"4-bit model loaded!")
# Uses ~4x less memory than FP16!
```

### Benchmarking Script

```python
import torch
import time

def benchmark_model(model, tokenizer, prompt, num_runs=20):
    """Measure inference speed and memory"""
    
    # Warm up
    for _ in range(3):
        _ = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=100)
    
    # Measure speed
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        _ = model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device), max_length=100)
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs
    
    # Measure memory
    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return avg_time, memory_mb

# Test each precision
results = {}
for name, model in [("FP16", model_fp16), ("INT8", model_8bit), ("NF4", model_4bit)]:
    time_per_gen, memory = benchmark_model(model, tokenizer, "Hello, how are you?")
    results[name] = {"time": time_per_gen, "memory": memory}
    print(f"{name}: {time_per_gen:.3f}s, {memory:.0f}MB")
```

### Expected Results (TinyLlama-1.1B)

```
FP16:  0.120s per generation, 2400MB VRAM
INT8:  0.095s per generation, 1200MB VRAM (2x smaller, 1.3x faster)
NF4:   0.110s per generation, 600MB VRAM (4x smaller, 1.1x faster)

Quality (perplexity on validation set):
FP16:  8.5 (baseline)
INT8:  8.6 (+1% degradation)
NF4:   8.9 (+5% degradation)
```

### When to Use Each

- **FP16:** Default for training and inference, good balance
- **INT8:** Production inference, minimal quality loss, 2x compression
- **NF4:** Maximum compression for deployment, especially with QLoRA
- **FP32:** Only for research/debugging, rarely needed

### Success Criteria
- ✅ Successfully load models in 8-bit and 4-bit
- ✅ Measure 2x and 4x memory reduction respectively
- ✅ Quality degradation <5%
- ✅ Understand when to use each precision

### Estimated Time
2-3 days

---

## 📊 Phase 1 Evaluation & Comparison

### Final Milestone: Compare All Methods

Create a comprehensive comparison of all techniques learned:

| Model | Params | Training Time | Memory (Train) | Memory (Infer) | Quality | Use Case |
|-------|--------|---------------|----------------|----------------|---------|----------|
| Base (no training) | 1.1B | - | - | 4GB | Baseline | - |
| Full SFT | 1.1B | 4 hrs | 12GB | 4GB | 100% | Research |
| LoRA SFT | 1.1B | 1.5 hrs | 4GB | 4GB | 98% | Production |
| QLoRA SFT | 1.1B | 2 hrs | 2GB | 4GB | 96% | Consumer GPU |
| Distilled (355M) | 355M | 3 hrs | 6GB | 1.5GB | 85% | Edge/Mobile |
| Distilled + INT8 | 355M | - | - | 0.8GB | 84% | Edge/Mobile |
| Distilled + NF4 | 355M | - | - | 0.4GB | 82% | Mobile/IoT |

### Deliverables for Phase 1

1. **Models:**
   - [ ] SFT fine-tuned model
   - [ ] LoRA adapter (separate from base)
   - [ ] QLoRA adapter
   - [ ] Distilled student model
   - [ ] Quantized versions (8-bit, 4-bit)

2. **Code:**
   - [ ] Training scripts for each method
   - [ ] Evaluation scripts
   - [ ] Benchmarking code

3. **Documentation:**
   - [ ] Training logs and metrics
   - [ ] Comparison charts
   - [ ] Example outputs from each model
   - [ ] Lessons learned writeup

4. **Analysis:**
   - [ ] Which method worked best for your use case?
   - [ ] What are the tradeoffs?
   - [ ] When would you use each in production?

---

## 🎯 Phase 1 Success Criteria

By the end of Phase 1, you should be able to:

- ✅ Fine-tune any LLM on custom data
- ✅ Use LoRA/QLoRA for efficient training
- ✅ Create distilled models for deployment
- ✅ Apply quantization for compression
- ✅ Measure and compare model performance
- ✅ Understand memory/quality/speed tradeoffs
- ✅ Choose the right technique for different scenarios

---

## 🚀 What's Next?

Congratulations on completing Phase 1! You now have the foundation for all modern LLM training.

**Next Steps:**
- **Phase 2:** Learn alignment methods (Reward Modeling, DPO, RLHF)
- **Phase 3:** Master inference optimization (Flash Attention, vLLM, Speculative Decoding)
- **Phase 4:** Explore advanced topics (Multi-modal, Constitutional AI, etc.)

**Or take a break and:**
- Polish your Phase 1 portfolio project
- Write a blog post about what you learned
- Apply these techniques to your own domain

---

## 📚 Additional Resources for Phase 1

### Papers
- **LoRA:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA:** "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **Distillation:** "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

### Tutorials
- HuggingFace PEFT documentation
- HuggingFace TRL examples
- bitsandbytes quantization guide

### Communities
- HuggingFace Discord
- r/LocalLLaMA
- EleutherAI Discord

---

Ready for Phase 2? Proceed to [Phase 2: Alignment Methods](./phase-2-alignment-methods.md)


