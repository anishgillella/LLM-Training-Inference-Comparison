# Phase 2: Alignment Methods (Weeks 4-5)

> Learn how to align LLMs with human preferences and values

---

## 🎯 Phase Overview

In this phase, you'll learn the cutting-edge alignment techniques that turn base models into helpful, harmless, and honest assistants:

1. **Reward Modeling** - Train models to predict human preferences
2. **Direct Preference Optimization (DPO)** - Align models directly from preference data
3. **RLHF with PPO** (Optional, Advanced) - Full reinforcement learning pipeline

**Duration:** 2-3 weeks part-time
**Cost:** $0-10
**Hardware:** Google Colab Free → Pro (upgrade recommended for week 2)
**Prerequisites:** Complete Phase 1 (you need a fine-tuned model!)

---

## 📋 Stage 5: Reward Modeling

### Goal
Train a model to predict which of two outputs humans would prefer, creating a learned "quality score" function.

### What You'll Learn
- Bradley-Terry preference model
- Pairwise ranking loss
- Reward model calibration
- Preference data collection and formatting
- How to evaluate preference accuracy

### Why Reward Modeling Matters
This is the bridge between human preferences and model behavior:
```
Human: "Output A is better than Output B"
    ↓
Reward Model: learns to predict this
    ↓
Scores: A = 0.8, B = 0.3
    ↓
Used in: RLHF training, Best-of-N sampling, evaluation
```

### Key Concept: From Classification to Ranking

```python
# Traditional classification: Is this output good?
Label: Good (1) or Bad (0)
Problem: Subjective, inconsistent

# Preference ranking: Which output is better?
Input: (prompt, output_A, output_B)
Label: A is better (1) or B is better (0)
Result: More consistent, easier to label
```

### Tasks Checklist
- [ ] Choose preference dataset (recommend: `Anthropic/hh-rlhf`)
- [ ] Load your SFT model from Phase 1
- [ ] Modify architecture to add scalar reward head
- [ ] Implement pairwise ranking loss
- [ ] Train on preference pairs
- [ ] Validate accuracy on held-out preferences
- [ ] Analyze failure cases
- [ ] Test reward model on new outputs

### Dataset Format

```python
# Preference pair format
{
    "prompt": "How do I make a cake?",
    "chosen": "Here's a simple recipe: 1. Mix flour...",  # Better output
    "rejected": "Idk, Google it."  # Worse output
}

# The reward model learns: reward(chosen) > reward(rejected)
```

### Code Example - Reward Model Training

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from trl import RewardTrainer
from datasets import load_dataset
import torch

# 1. Load your SFT model (from Phase 1)
model = AutoModelForSequenceClassification.from_pretrained(
    "./sft_final",  # Your fine-tuned model from Phase 1
    num_labels=1,  # Output a single reward score
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("./sft_final")

# 2. Load preference dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# 3. Preprocess: tokenize chosen and rejected
def preprocess_function(examples):
    # Tokenize chosen outputs
    tokenized_chosen = tokenizer(examples["chosen"], truncation=True, max_length=512)
    # Tokenize rejected outputs
    tokenized_rejected = tokenizer(examples["rejected"], truncation=True, max_length=512)
    
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["test"].map(preprocess_function, batched=True)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    remove_unused_columns=False,
)

# 5. Create RewardTrainer (handles pairwise loss automatically!)
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 6. Train!
trainer.train()

# 7. Save
model.save_pretrained("./reward_model_final")
```

### Reward Model Loss (Bradley-Terry Model)

```python
# Conceptual explanation of the loss
def reward_loss(chosen_rewards, rejected_rewards):
    """
    Maximize the probability that chosen has higher reward than rejected.
    
    P(chosen > rejected) = sigmoid(reward(chosen) - reward(rejected))
    Loss = -log(P(chosen > rejected))
    """
    return -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
```

### Evaluation Script

```python
def evaluate_reward_model(model, tokenizer, test_pairs):
    """Test if reward model correctly ranks preferences"""
    
    correct = 0
    total = len(test_pairs)
    
    for pair in test_pairs:
        # Get rewards for chosen and rejected
        chosen_inputs = tokenizer(pair["chosen"], return_tensors="pt", truncation=True)
        rejected_inputs = tokenizer(pair["rejected"], return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            chosen_reward = model(**chosen_inputs).logits.item()
            rejected_reward = model(**rejected_inputs).logits.item()
        
        # Check if chosen has higher reward
        if chosen_reward > rejected_reward:
            correct += 1
    
    accuracy = correct / total
    print(f"Reward Model Accuracy: {accuracy:.2%}")
    return accuracy

# Test on validation set
accuracy = evaluate_reward_model(model, tokenizer, eval_dataset[:100])

# Expected: >65% on hh-rlhf (random baseline is 50%)
```

### Success Criteria
- ✅ >65% accuracy on validation preferences
- ✅ Reward scores correlate with quality (better outputs = higher rewards)
- ✅ Model consistently distinguishes chosen vs rejected
- ✅ Sensible reward scores on manual test cases

### Common Issues & Solutions

**Problem:** Low accuracy (~50-55%)
- Solution: Train longer, check learning rate, ensure data is shuffled

**Problem:** Rewards are all similar
- Solution: Reduce regularization, increase model capacity

**Problem:** Rewards are extreme (very high or very low)
- Solution: Add reward normalization, gradient clipping

### Estimated Time
4-6 days

---

## 📋 Stage 6: Direct Preference Optimization (DPO)

### Goal
Align your SFT model with human preferences using DPO - a simpler, more stable alternative to RLHF.

### What You'll Learn
- DPO objective function
- Why DPO is simpler than RLHF (no separate reward model needed!)
- KL divergence as a constraint
- Beta hyperparameter tuning
- Reference model vs policy model
- Preventing reward hacking

### Why DPO is Revolutionary

**Traditional RLHF Pipeline:**
```
1. Train SFT model
2. Train reward model (separate!)
3. Use RL (PPO) to optimize policy with reward model
4. Deal with instability, reward hacking, etc.
```

**DPO Pipeline:**
```
1. Train SFT model
2. Train directly on preferences (no separate reward model!)
3. Done! ✅

Result: Same or better alignment, much simpler!
```

### Key Concept: Implicit Reward Modeling

DPO doesn't need a separate reward model because it optimizes the policy directly:

```python
# RLHF: Maximize reward while staying close to reference
reward(output) - β * KL(policy || reference)

# DPO: Equivalent but optimized directly on preferences
log(σ(β * log(π(chosen)/π_ref(chosen)) - β * log(π(rejected)/π_ref(rejected))))
```

### Tasks Checklist
- [ ] Load your SFT model (this will be the initial policy)
- [ ] Create a copy as the "reference model" (frozen)
- [ ] Load preference dataset
- [ ] Configure DPO trainer with appropriate β
- [ ] Train with preference data
- [ ] Monitor KL divergence from reference
- [ ] Generate outputs and compare with SFT baseline
- [ ] Evaluate preference win rate

### Code Example - DPO Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# 1. Load your SFT model (from Phase 1)
model = AutoModelForCausalLM.from_pretrained("./sft_final")
tokenizer = AutoTokenizer.from_pretrained("./sft_final")

# 2. Load reference model (frozen copy of SFT model)
ref_model = AutoModelForCausalLM.from_pretrained("./sft_final")

# 3. Load preference dataset
dataset = load_dataset("Anthropic/hh-rlhf")

# 4. DPO Configuration
training_args = DPOConfig(
    output_dir="./dpo_model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,  # Lower LR than SFT!
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    beta=0.1,  # KL penalty coefficient (tune this!)
    max_length=512,
    max_prompt_length=256,
    remove_unused_columns=False,
)

# 5. Create DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# 6. Train!
dpo_trainer.train()

# 7. Save aligned model
model.save_pretrained("./dpo_final")
```

### Understanding Beta (β)

Beta controls how much the policy can diverge from the reference:

```python
# Small β (0.01-0.1): Stay close to reference
# → More conservative, slower learning
# → Use when: You trust your SFT model

# Medium β (0.1-0.5): Balanced
# → Standard choice
# → Use when: Starting with DPO

# Large β (0.5-1.0): Allow more divergence
# → Faster learning, but risk reward hacking
# → Use when: Need aggressive alignment
```

### Monitoring Training

```python
# Key metrics to watch during DPO training:

1. Loss: Should decrease steadily
   - If stuck: increase learning rate or beta

2. KL Divergence: Should stay reasonable (<5.0)
   - If too high: decrease beta or learning rate
   - If zero: increase beta

3. Rewards (implicit): Chosen should be > Rejected
   - Monitor the gap

4. Evaluation accuracy: >60% means it's learning
```

### Evaluation: SFT vs DPO

```python
def compare_models(sft_model, dpo_model, tokenizer, test_prompts):
    """Compare SFT baseline with DPO-aligned model"""
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        
        # Generate from SFT
        inputs = tokenizer(prompt, return_tensors="pt")
        sft_output = sft_model.generate(**inputs, max_length=100, temperature=0.7)
        sft_text = tokenizer.decode(sft_output[0], skip_special_tokens=True)
        
        # Generate from DPO
        dpo_output = dpo_model.generate(**inputs, max_length=100, temperature=0.7)
        dpo_text = tokenizer.decode(dpo_output[0], skip_special_tokens=True)
        
        print(f"\nSFT Output: {sft_text}")
        print(f"\nDPO Output: {dpo_text}")
        print(f"\n{'='*60}")

# Test prompts
test_prompts = [
    "How can I help someone who is feeling depressed?",
    "What's the best way to learn Python?",
    "Explain quantum computing to a 10-year-old.",
]

compare_models(sft_model, dpo_model, tokenizer, test_prompts)
```

### Expected Improvements

After DPO training, outputs should be:
- ✅ More helpful and detailed
- ✅ Better formatted (if preferences favor formatting)
- ✅ More aligned with human preferences
- ✅ Less likely to be harmful or misleading

### Win Rate Evaluation

```python
def calculate_win_rate(model_a, model_b, tokenizer, test_set, judge_model):
    """
    Compare two models using GPT-4 or reward model as judge
    Returns: win rate for model_a
    """
    wins = 0
    total = len(test_set)
    
    for example in test_set:
        prompt = example["prompt"]
        
        # Generate from both models
        output_a = generate(model_a, tokenizer, prompt)
        output_b = generate(model_b, tokenizer, prompt)
        
        # Judge which is better
        winner = judge_model.compare(prompt, output_a, output_b)
        
        if winner == "A":
            wins += 1
    
    win_rate = wins / total
    return win_rate

# Expected: DPO should beat SFT 60-75% of the time
win_rate = calculate_win_rate(dpo_model, sft_model, tokenizer, test_set, reward_model)
print(f"DPO vs SFT Win Rate: {win_rate:.1%}")
```

### Success Criteria
- ✅ DPO model preferred over SFT in 60-75% of comparisons
- ✅ KL divergence stays reasonable (< 5.0)
- ✅ Maintains instruction-following capability
- ✅ Doesn't degrade on general tasks

### Estimated Time
5-7 days

---

## 📋 Stage 7 (Optional): RLHF with PPO

### ⚠️ Warning: Advanced!
Only attempt after DPO success. PPO is significantly more complex and unstable.

### Goal
Implement the full RLHF pipeline using Proximal Policy Optimization (PPO) with your reward model.

### What You'll Learn
- Policy gradient methods
- Value function estimation
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Why RLHF is hard (reward hacking, mode collapse, instability)
- How to debug RL training

### Why PPO After DPO?
**DPO is simpler and often better!** But learning PPO helps you:
- Understand how ChatGPT was originally trained
- Learn reinforcement learning fundamentals
- Appreciate why DPO is revolutionary
- Handle cases where DPO isn't enough

### RLHF Pipeline Overview

```
1. SFT Model (actor/policy)
   ↓
2. Reward Model (critic)
   ↓
3. PPO Training Loop:
   - Generate outputs from policy
   - Score with reward model
   - Compute advantages
   - Update policy to maximize reward
   - Constrain with KL penalty (stay close to SFT)
   ↓
4. Aligned Model
```

### Tasks Checklist
- [ ] Ensure reward model is working well (>70% accuracy)
- [ ] Set up PPO trainer from TRL
- [ ] Configure reward model as environment
- [ ] Add KL penalty term (crucial!)
- [ ] Implement advantage estimation (GAE)
- [ ] Train with careful monitoring
- [ ] Watch for reward hacking
- [ ] Watch for mode collapse
- [ ] Compare with DPO model

### Code Example - PPO Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset
import torch

# 1. Load SFT model as policy (with value head for PPO)
model = AutoModelForCausalLMWithValueHead.from_pretrained("./sft_final")
tokenizer = AutoTokenizer.from_pretrained("./sft_final")

# 2. Load reward model
reward_model = AutoModelForSequenceClassification.from_pretrained("./reward_model_final")
reward_model.eval()

# 3. Load reference model for KL penalty
ref_model = AutoModelForCausalLM.from_pretrained("./sft_final")
ref_model.eval()

# 4. PPO Configuration
ppo_config = PPOConfig(
    model_name="./sft_final",
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=4,
    ppo_epochs=4,  # Number of optimization epochs per batch
    init_kl_coef=0.2,  # KL penalty coefficient
    target_kl=6.0,  # Target KL divergence
    cliprange=0.2,  # PPO clip range
    vf_coef=0.1,  # Value function coefficient
)

# 5. Create PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# 6. Load prompts for generation
dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

# 7. Training loop
for epoch in range(1):
    for batch in dataset:
        # Extract prompts
        query_tensors = [tokenizer.encode(prompt, return_tensors="pt")[0] for prompt in batch["prompt"]]
        
        # Generate responses
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, max_new_tokens=100)
            response_tensors.append(response)
        
        # Compute rewards using reward model
        texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        rewards = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                reward = reward_model(**inputs).logits.item()
            rewards.append(torch.tensor(reward))
        
        # PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log metrics
        print(f"Reward: {stats['ppo/mean_scores']:.3f}, KL: {stats['ppo/mean_non_score_reward']:.3f}")

# 8. Save aligned model
model.save_pretrained("./ppo_final")
```

### Key PPO Concepts

**1. Advantage Function:**
```python
# How much better is this action than average?
Advantage = Q(state, action) - V(state)
         = reward + γ * V(next_state) - V(state)

# Positive advantage: reinforce this action
# Negative advantage: discourage this action
```

**2. Clipped Objective:**
```python
# Prevent too-large policy updates
ratio = π_new(action) / π_old(action)
clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε = 0.2 typically

loss = -min(ratio * advantage, clipped_ratio * advantage)
```

**3. KL Penalty:**
```python
# Prevent drifting too far from reference
total_reward = reward_model_score - β * KL(π || π_ref)

# If KL gets too high, increase β
# If KL is too low, decrease β
```

### Monitoring PPO Training

Watch for these issues:

**1. Reward Hacking:**
```
Symptom: Reward spikes up, but outputs are nonsense
Cause: Model exploits reward model bugs
Fix: Increase KL penalty, improve reward model
```

**2. Mode Collapse:**
```
Symptom: Model generates same output repeatedly
Cause: Reward model favors certain patterns
Fix: Add diversity bonus, reduce learning rate
```

**3. Instability:**
```
Symptom: Reward/loss oscillates wildly
Cause: Learning rate too high, batch size too small
Fix: Reduce LR, increase batch size, clip gradients
```

### Debugging Script

```python
def monitor_ppo_health(stats):
    """Check if PPO training is healthy"""
    
    # Check 1: Reward should increase gradually
    if stats['ppo/mean_scores'] > 10.0:
        print("⚠️  WARNING: Reward too high - possible reward hacking!")
    
    # Check 2: KL should stay reasonable
    if stats['ppo/mean_kl'] > 10.0:
        print("⚠️  WARNING: KL too high - model drifting from reference!")
    
    # Check 3: Policy loss should be reasonable
    if abs(stats['ppo/loss/policy']) > 5.0:
        print("⚠️  WARNING: Policy loss too high - training unstable!")
    
    # Check 4: Value loss
    if stats['ppo/loss/value'] > 10.0:
        print("⚠️  WARNING: Value function not learning well!")
    
    print(f"✅ Health check passed. Reward: {stats['ppo/mean_scores']:.2f}, KL: {stats['ppo/mean_kl']:.2f}")
```

### Success Criteria
- ✅ Training is stable (no wild oscillations)
- ✅ Reward increases gradually
- ✅ KL divergence stays < 10.0
- ✅ Outputs are coherent and improved
- ✅ Comparable or better than DPO

### When PPO Doesn't Work
It's okay if PPO is challenging! Many practitioners prefer DPO because:
- Simpler to implement
- More stable training
- Often similar or better results
- No separate reward model needed

**If PPO isn't working:** Stick with DPO! It's what most companies use today.

### Estimated Time
1-2 weeks (high variance due to debugging!)

---

## 📊 Phase 2 Evaluation & Comparison

### Final Milestone: Compare All Alignment Methods

| Model | Win Rate vs SFT | Training Complexity | Stability | Production Use |
|-------|----------------|---------------------|-----------|----------------|
| **SFT Baseline** | 0% (baseline) | ⭐ Easy | ✅ Stable | Common |
| **Reward Model** | N/A (scoring only) | ⭐⭐ Medium | ✅ Stable | Common |
| **DPO** | 60-75% | ⭐⭐ Medium | ✅ Stable | **Very Common** ⭐ |
| **PPO/RLHF** | 55-70% | ⭐⭐⭐⭐ Very Hard | ⚠️  Unstable | Less Common |

**Industry Trend:** Most companies are moving from PPO → DPO due to simplicity and stability.

### Comprehensive Evaluation

```python
def full_alignment_evaluation(models_dict, test_set, reward_model):
    """
    Compare SFT, DPO, and PPO on multiple dimensions
    
    models_dict = {
        "sft": sft_model,
        "dpo": dpo_model,
        "ppo": ppo_model,  # if you have it
    }
    """
    results = {}
    
    for name, model in models_dict.items():
        # 1. Win rate (head-to-head)
        wins = 0
        for example in test_set:
            output = generate(model, example["prompt"])
            score = reward_model.score(example["prompt"], output)
            wins += (score > baseline_score)
        
        # 2. Reward scores
        avg_reward = compute_avg_reward(model, test_set, reward_model)
        
        # 3. Safety evaluation
        safety_score = evaluate_safety(model, safety_prompts)
        
        # 4. Helpfulness
        helpfulness = evaluate_helpfulness(model, help_prompts)
        
        results[name] = {
            "win_rate": wins / len(test_set),
            "avg_reward": avg_reward,
            "safety": safety_score,
            "helpfulness": helpfulness,
        }
    
    return results

# Run full evaluation
results = full_alignment_evaluation(
    {"sft": sft_model, "dpo": dpo_model},
    test_set,
    reward_model
)

print(results)
```

### Deliverables for Phase 2

1. **Models:**
   - [ ] Reward model checkpoint
   - [ ] DPO-aligned model
   - [ ] (Optional) PPO-aligned model

2. **Analysis:**
   - [ ] Reward model accuracy report
   - [ ] DPO training curves and metrics
   - [ ] Comparison: SFT vs DPO vs PPO
   - [ ] Example outputs showing improvement

3. **Documentation:**
   - [ ] Training logs
   - [ ] Hyperparameter tuning notes
   - [ ] Failure cases and insights
   - [ ] Lessons learned

---

## 🎯 Phase 2 Success Criteria

By the end of Phase 2, you should be able to:

- ✅ Train reward models from preference data
- ✅ Implement DPO alignment
- ✅ Understand why DPO is simpler than RLHF
- ✅ (Optional) Implement PPO/RLHF
- ✅ Evaluate alignment quality
- ✅ Understand reward hacking and how to prevent it
- ✅ Know when to use each alignment method

---

## 🚀 What's Next?

Congratulations on completing Phase 2! You now understand modern alignment techniques.

**Next Steps:**
- **Phase 3:** Master inference optimization (Flash Attention, vLLM)
- **Phase 4:** Explore advanced topics (Speculative Decoding, Multi-modal)

**Or take a break and:**
- Compare your aligned model with ChatGPT
- Run user studies to validate improvements
- Write about the DPO vs PPO tradeoffs

---

## 📚 Additional Resources for Phase 2

### Papers
- **DPO:** "Direct Preference Optimization" (Rafailov et al., 2023)
- **InstructGPT:** "Training language models to follow instructions" (OpenAI, 2022)
- **PPO:** "Proximal Policy Optimization" (Schulman et al., 2017)
- **Reward Modeling:** "Learning to Summarize from Human Feedback" (Stiennon et al., 2020)

### Implementations
- TRL library documentation
- DPO reference implementation
- OpenRLHF (open-source RLHF)

### Communities
- r/LocalLLaMA (alignment discussions)
- EleutherAI Discord
- HuggingFace Discord

---

Ready for Phase 3? Proceed to [Phase 3: Inference Optimization](./phase-3-inference-optimization.md)


