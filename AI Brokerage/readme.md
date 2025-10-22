# 🧠 AI Brokerage — Voice & Email Automation Grid

> **AI-powered commercial insurance brokerage system**  
> Built to demonstrate Harper-style modular AI infrastructure — combining voice automation, email intelligence, post-training alignment, and inference optimization under the REAL framework (Reliable · Experience-focused · Accurate · Low-latency).

---

## 📋 Overview

AI Brokerage is a modular, production-style MVP that simulates how an AI-native insurance brokerage would operate.

The system consists of multiple **independent agents** — each responsible for its own workflow:
- 🎙️ **Voice Agent** — handles live insurance conversations
- 💌 **Email Agent** — manages follow-up and standalone communications
- 🧠 **Inference Service** — performs structured insurance reasoning
- 🧩 **Alignment Service** — improves models via post-training workflows (DPO / RLHF / GRPO / RLAIF)
- 🌐 **Data Enrichment Layer** — fetches external business, risk, and compliance data asynchronously
- 📊 **Monitoring Layer** — tracks metrics and enforces the REAL framework

Every agent runs independently, communicating through APIs or Redis streams, so you can scale or deploy each component without coupling them.

---

## 🧱 Folder Structure

ai-brokerage/
│
├── voice_agent/ # Real-time call handling
├── email_agent/ # Automated & inbound email workflows
├── inference_service/ # vLLM-based reasoning backend
├── alignment_service/ # Post-training pipelines (DPO, GRPO, RLHF)
├── data_enrichment/ # Async context fetchers
├── monitoring/ # Logfire dashboards and REAL metrics
└── README.md

---

## ⚙️ Core Components

### 🎙️ Voice Agent
- Built with **Retell** for quick deployment (upgradeable to LiveKit for custom infra).
- Uses **Deepgram** for STT and **ElevenLabs / Cartesia** for TTS.
- Responds within **sub-700 ms latency**, using low-latency streaming.
- Logs every interaction to **Logfire** for latency, token, and empathy metrics.
- Does **not** depend on inference or alignment loops — self-contained runtime.
- Deployed on **Vercel** for edge-optimized voice endpoints.

---

### 🎤 Voice Workflows

#### 1. **Outbound Prospecting Agent** (MVP Starting Point)
- **Purpose**: Cold outreach, lead qualification, objection handling, appointment booking
- **Conversation Flow**: 
  - Warm intro → Pain point discovery → Value prop → Objection handling → CTA (book call/get quote)
  - Typical duration: 3-7 minutes
- **Key Metrics**: Lead quality score, booking rate, call duration, objection resolution rate
- **Integration**: Syncs qualified leads to email agent for follow-up sequences
- **Deployment**: Outbound dialer using Twilio/Retell with predictive pacing

#### 2. **Form-Filling / Underwriting Agent** (Phase 2)
- **Purpose**: Multi-turn insurance application conversations (20-30 min)
- **Conversation Flow**:
  - Business info collection → Risk assessment questions → Coverage recommendations → Pricing
  - Multi-turn context preservation across long dialogues
- **Key Metrics**: Application completion rate, accuracy of collected data, time-to-complete
- **Integration**: Feeds into inference service for underwriting recommendations
- **Escalation**: Routes complex questions to human underwriters

#### 3. **Claims Intake Agent (FNOL)** (Phase 2-3)
- **Purpose**: Empathetic first notice of loss collection, 24/7 availability
- **Conversation Flow**:
  - Incident empathy & clarification → Details collection → Coverage determination → Next steps
  - Emotional intelligence scoring via judge
- **Key Metrics**: FNOL data completeness, customer satisfaction score, escalation rate
- **Integration**: Auto-populates claims system, triggers workflows

#### 4. **Policy Service Agent** (Phase 2-3)
- **Purpose**: 24/7 policy servicing (coverage explanations, certificate generation, endorsements)
- **Conversation Flow**:
  - Intent detection (coverage question vs. change request) → Resolution → Transaction confirmation
- **Key Metrics**: Resolution without escalation rate, CSAT, average handle time
- **Multi-modal**: Voice → SMS for confirmations → Email for documents

---

### 🧠 Evaluation & Alignment Infrastructure

#### **MLLM-as-a-Judge System** (Phase 1 — Critical)
- **Core Function**: Automated evaluation of voice agent responses for compliance, accuracy, and tone
- **Judge Scores Track**:
  - `compliance_score` (0-1) — regulatory adherence (state-specific insurance rules)
  - `accuracy_score` (0-1) — factual correctness of insurance info
  - `clarity_score` (0-1) — understandability for non-experts
  - `empathy_score` (0-1) — emotional appropriateness (especially for claims)
  - `conversion_score` (0-1) — likelihood to move deal forward
- **Implementation**: Run GPT-4o + domain-specific rubrics against conversation logs
- **Feedback Loop**: Judge scores feed directly into RLHF/DPO training data
- **Latency Requirement**: <5 second judgment per 500-token conversation

#### **Insurance Domain Benchmarks**
Synthetic evaluation datasets covering:
- **High-risk scenarios** (complex underwriting, objection handling, claims)
- **Compliance violations** (states with different regulations)
- **Tone deviations** (too aggressive, not empathetic enough)
- **Factual errors** (wrong coverage recommendations, incorrect pricing)

Example benchmark structure:
```json
{
  "benchmark_id": "bench_commercial_auto_101",
  "conversation": [...],
  "ground_truth": {
    "coverage_needed": ["Commercial Auto", "Cargo Protection"],
    "compliance_state": "CA",
    "risk_level": "medium"
  },
  "judge_rubric": {
    "compliance": {...},
    "accuracy": {...},
    "empathy": {...}
  }
}
```

#### **Production Monitoring Dashboard**
Track in real-time:
- Model performance degradation (judge scores trending down)
- Cost per successful interaction
- Latency percentiles (p50, p95, p99)
- Escalation rate anomalies
- Failure mode clustering (group failures by type)

---

### 🛠️ Platform Tools

#### **1. Internal No-Code Voice Workflow Builder**
- Drag-and-drop interface for non-technical team members to:
  - Design conversation flows (branching logic)
  - Modify system prompts and tone
  - Set qualification thresholds
  - Configure escalation rules
- Backend: Flow → YAML → FastAPI → Voice Agent
- Permission model: Teams can edit templates but not deploy without approval

#### **2. Conversation Analytics Dashboard**
Real-time metrics for each voice workflow:
- Completion rate (% of calls reaching goal)
- Call duration distribution
- Top objection patterns
- Booking/qualification rate
- Customer satisfaction (post-call survey scores)
- Cost per successful interaction
- A/B testing results (voice persona vs. script changes)

#### **3. A/B Testing Framework**
- Test different voice personas (professional, friendly, concise)
- Test different prompt versions with statistical significance
- Automatic routing: 50/50 split, track outcomes
- Winner detection: p-value < 0.05, convert to default

#### **4. Voice Agent Templates Library**
Pre-built conversation templates for:
- Cold outreach (various industries)
- Objection handling playbooks
- Claims intake standardized flows
- Policy inquiry templates
- Upsell/cross-sell conversation starters

---

### 🔌 Telephony & Real-Time Infrastructure

#### **For MVP (Retell-based)**
- Retell handles SIP/media orchestration
- Twilio integration for outbound calling (predictive dialer)
- Single-model deployment (GPT-4o for all agents initially)

#### **For Scale (Custom Infrastructure)**
- **Telephony Protocols**:
  - SIP (Session Initiation Protocol) for call signaling
  - RTP (Real-Time Protocol) for audio streaming
  - WebRTC for browser-based testing
- **Media Server**: LiveKit for WebRTC handling
- **Voice Orchestration**: Pipecat for flexible pipeline customization
- **Multi-model Routing**: Route conversations to different models based on complexity (GPT-4o vs. Llama 70B vs. specialized underwriting model)

#### **Audio Processing Pipeline**
- **VAD** (Voice Activity Detection) — detect speech vs. silence
- **AEC** (Acoustic Echo Cancellation) — remove echo in calls
- **Noise Suppression** — reduce background noise
- **Speaker Diarization** — track who's speaking (agent vs. customer)

#### **Durable Workflows (Temporal.io)**
For long conversations (20-30 min underwriting):
- Checkpoint conversation state every 5 minutes
- Resume from checkpoint if connection drops
- Retry failed API calls with exponential backoff
- Track workflow history for audit/compliance

---

### 💌 Email Agent
- Two functions:
  1. Sends automated follow-ups after voice calls (policy summaries, payment reminders).
  2. Handles standalone inbound/outbound email threads asynchronously.
- Uses GPT-4o for text generation, validated through **LLM-as-a-Judge** for compliance and clarity.
- Persists threads and metadata in **Postgres**.
- Runs on **Vercel** or **Railway**, using background tasks or Redis queue for async delivery.

---

### 🧠 Inference Service
- Independent REST API (`/infer`) for structured insurance reasoning tasks.
- Built on **vLLM**, using OpenAI-compatible API schema.
- Implements:
  - **Speculative decoding** for faster generation.
  - **KV-cache optimization** and **continuous batching**.
  - **PagedAttention** and **INT8/INT4 quantization** to reduce GPU load.
- Supports **multi-model routing** for cost-latency balancing.
- Serves as the backend for voice or email agents when they need reasoning.

---

### 🧩 Alignment Service
- Runs on **Modal GPUs** for training and post-training.
- Implements the core Harper-style techniques (non-redundant set):
  - **RLHF** — Reinforcement Learning from Human Feedback
  - **RLAIF** — Reinforcement Learning from AI Feedback
  - **GRPO** — Gradient Reward Policy Optimization
  - **DPO** — Direct Preference Optimization
- Uses feedback from Logfire metrics and LLM-as-a-Judge scores as training rewards.
- Produces versioned checkpoints for the inference service to deploy.
- Tracks experiments and model metrics through **Modal logs** and **Logfire**.

---

### 🌐 Data Enrichment Layer
- Async service that fetches contextual data in parallel using `asyncio.gather()`.
- Integrations:
  - Mock **risk scoring** API  
  - **Location & compliance** lookups  
  - Synthetic **business registry** data
- Returns enriched JSON payloads for inference or email context.
- Demonstrates Harper’s principle: *“Start Practical, Scale Smart.”*

---

### 📊 Monitoring Layer
- Built on **Logfire**, which centralizes:
  - p50/p95/p99 latency tracking
  - Model accuracy and judge scores
  - Voice call durations and completion rates
  - Cost per call and GPU utilization
- Exports structured JSON logs for easy observability.
- Drives the REAL metrics system.

---

## 🔍 REAL Framework Mapping

| REAL Pillar | Implementation |
|--------------|----------------|
| **Reliable** | Redis retry queues, Postgres persistence, Modal job durability |
| **Experience-focused** | Voice empathy & latency metrics, email clarity scoring |
| **Accurate** | Judge-based grading, alignment feedback, quantization validation |
| **Low-latency** | vLLM optimizations, Vercel edge endpoints, async enrichment |

---

## ☁️ Deployment Plan

| Environment | Platform | Purpose |
|--------------|-----------|----------|
| **Local Dev** | Docker Compose | Full stack simulation |
| **Training & Alignment** | Modal | GPU-based post-training loops |
| **Frontend / Agents** | Vercel | Voice and email micro-services |
| **Storage & Cache** | Supabase / Railway | Postgres + Redis Streams |
| **Monitoring** | Logfire | Metrics, logs, and REAL dashboards |

---

## 🧠 Synthetic Data Generation

Synthetic datasets simulate real commercial insurance scenarios:

### 1. **Voice & Email Dialogues**
```json
{
  "conversation_id": "conv_024",
  "turns": [
    {"role": "customer", "text": "I run a logistics company and need coverage for my trucks."},
    {"role": "agent", "text": "You'll need commercial auto insurance and cargo protection."}
  ],
  "metadata": {"risk_score": 0.84, "industry": "Logistics"}
}
```

### 2. **Form-Filling Data**
```json
{
  "application_id": "app_101",
  "form": {
    "Business Type": "Bakery",
    "Employees": 8,
    "Building Ownership": "Lease"
  }
}
```

### 3. **Evaluation Pairs**
Each entry contains model output + judge score + feedback signal for post-training.
```json
{
  "eval_id": "eval_42",
  "model_output": "Your business needs general liability and property coverage.",
  "judge_score": 0.92,
  "feedback": "Clear, actionable recommendation with compliance validated."
}
```

---

## 🔄 Phase-by-Phase Development Process

Each phase has **clear objectives, distinct deliverables, measurable success criteria, and dependencies** to ensure no ambiguity.

---

### **Phase 1 — Voice Agent MVP** (3-4 weeks)

**🎯 Objectives:**
- Build a working outbound prospecting agent that can make real calls
- Establish evaluation infrastructure (MLLM-as-a-Judge)
- Create visibility into voice quality and business metrics
- Prove sub-700ms latency is achievable

**📦 Deliverables:**
- ✅ Retell + FastAPI integration (webhook handlers for call events)
- ✅ Deepgram STT + ElevenLabs TTS streaming pipeline
- ✅ MLLM-as-a-Judge scoring system with 5 rubrics (compliance, accuracy, clarity, empathy, conversion)
- ✅ Conversation analytics dashboard (React + Postgres queries)
- ✅ Logfire monitoring setup (latency p50/p95/p99, cost per call, judge scores)
- ✅ Twilio outbound dialer integration
- ✅ Synthetic prospecting dataset (500+ conversations)

**🎯 Success Criteria:**
- Calls connect within 2 seconds
- STT → Response → TTS latency < 700ms (p95)
- Judge evaluation completes in < 5 seconds per conversation
- Dashboard shows real-time metrics
- ≥80% of 20 test calls are scorable (judge doesn't error)
- Cost per call tracked (< $0.50 per call with low-cost models)

**📊 Scope Boundaries:**
- **ONLY** outbound prospecting workflow (no email, no underwriting, no claims)
- **ONLY** GPT-4o for inference (no fine-tuning yet)
- **ONLY** dashboard visualization (no internal no-code builder)
- **NOT** production-scale (10-100 concurrent calls max)

**⚙️ Dependencies:**
- None (this is the foundation)

**🔗 Next Phase Gate:**
- Judge system passes accuracy test on holdout dataset
- Latency meets sub-700ms target
- Dashboard is accessible and shows real data

---

### **Phase 2 — Platform Tools & Analytics** (2-3 weeks)

**🎯 Objectives:**
- Enable non-technical teams (sales, ops) to modify voice workflows
- Implement A/B testing for conversation variants
- Build comprehensive failure analysis tools
- Support Phase 3 form-filling agent development

**📦 Deliverables:**
- ✅ Internal No-Code Voice Workflow Builder (React UI)
  - Drag-drop conversation flow editor
  - System prompt + tone controls
  - Intent/entity mapper
  - Escalation rule configuration
  - YAML export → FastAPI deployment
- ✅ A/B Testing Framework
  - Test setup wizard (pick 2 variants)
  - Automatic 50/50 traffic split
  - Statistical significance calculator (p-value < 0.05)
  - Auto-winner detection + rollout
- ✅ Enhanced Analytics Dashboard
  - Failure clustering (group similar failures)
  - Judge score drill-downs (why did this call fail?)
  - Objection pattern detection
  - Cost per booking calculated
- ✅ Voice Agent Templates Library
  - 3 pre-built prospecting templates (by industry)
  - Objection handling playbooks
  - One-click deployment

**🎯 Success Criteria:**
- Sales team can modify a prompt without engineer approval
- A/B test runs to statistical significance in <100 calls
- Dashboard surfaces top 3 failure patterns automatically
- 3 templates are deployable in < 5 minutes
- Non-technical user can create new variant in < 10 minutes

**📊 Scope Boundaries:**
- **ONLY** for prospecting agent (not email/underwriting yet)
- **ONLY** internal tools (no customer-facing UI)
- **NOT** complex branching logic (simple conversation trees only)

**⚙️ Dependencies:**
- Phase 1 must be complete (judge system, monitoring, basic agent)
- Requires Postgres schema for storing workflow definitions
- Requires user auth system for permissions

**🔗 Next Phase Gate:**
- Non-technical team member successfully deploys a workflow variant
- A/B test completes with statistical significance
- Failure analysis identifies ≥5 patterns in demo data

---

### **Phase 3 — Multi-Workflow Expansion** (3-4 weeks)

**🎯 Objectives:**
- Build form-filling agent for 20-30 min underwriting conversations
- Add claims intake (FNOL) agent for 24/7 availability
- Implement multi-turn context preservation
- Set up intelligent escalation to human agents

**📦 Deliverables:**
- ✅ Form-Filling / Underwriting Agent
  - Multi-turn conversation (20-30 min typical duration)
  - Context preservation across long dialogues (Temporal.io for durability)
  - Structured data extraction (business info, risk assessment, coverage needs)
  - Integration with inference service for underwriting recommendations
  - Escalation to human underwriters for complex cases
- ✅ Claims Intake Agent (FNOL)
  - Empathetic conversation flow for incident reporting
  - Emotional intelligence scoring via judge
  - Structured FNOL data collection
  - Integration with claims system (auto-populate fields)
- ✅ Multi-Modal Orchestration
  - Voice → SMS confirmations (Twilio SMS)
  - Voice → Email document requests
  - Seamless context passing between channels
- ✅ Intelligent Routing System
  - Detect conversation complexity (use judge scores)
  - Route to specialized agents (prospecting vs. underwriting vs. claims)
  - Route to human agents when confidence too low
- ✅ Extended Logfire Dashboards
  - Per-workflow metrics (prospecting, underwriting, claims separate)
  - Escalation rate tracking
  - FNOL completeness scoring

**🎯 Success Criteria:**
- Underwriting calls average 20+ minutes
- Context preserved across 500+ turns (no reset)
- Judge scores available within 5 seconds per turn
- Escalations < 10% for non-complex calls
- Claims intake captures ≥95% of required fields
- SMS/email integrations work end-to-end

**📊 Scope Boundaries:**
- **ONLY** three workflows (prospecting, underwriting, claims)
- **ONLY** single-model inference (GPT-4o)
- **NOT** custom infrastructure yet (still using Retell + Twilio)
- **NOT** alignment training yet

**⚙️ Dependencies:**
- Phase 1 complete (base voice infrastructure)
- Phase 2 complete (platform tools for easier workflow creation)
- Temporal.io setup required
- Postgres schema expanded for workflow state

**🔗 Next Phase Gate:**
- Form-filling calls complete with >90% success rate
- FNOL integration passes end-to-end test
- Multi-modal routing works (voice → SMS → email)
- All three workflows have separate dashboards with distinct metrics

---

### **Phase 4 — Inference Optimization** (2-3 weeks)

**🎯 Objectives:**
- Deploy vLLM inference service for controlled latency + cost
- Implement speculative decoding, KV-cache, quantization
- Add multi-model routing (cost vs. latency tradeoffs)
- Reduce per-call inference cost

**📦 Deliverables:**
- ✅ vLLM Server Setup
  - OpenAI-compatible API (`/v1/chat/completions`)
  - Speculative decoding enabled
  - KV-cache optimization
  - PagedAttention + INT8/INT4 quantization
- ✅ Multi-Model Router
  - Route based on conversation complexity
  - GPT-4o for complex underwriting
  - Llama 70B for standard prospecting (cheaper)
  - Specialized insurance model (if fine-tuned in Phase 5)
- ✅ Inference Cost Dashboard
  - Cost per token by model
  - Cost per successful interaction
  - ROI analysis (cost vs. booking rate)
- ✅ Latency Optimization Benchmarks
  - Token generation speed
  - Streaming latency
  - End-to-end call latency (STT → inference → TTS)

**🎯 Success Criteria:**
- vLLM achieves < 50ms per token generation
- Specialized models achieve 20% cost reduction vs. GPT-4o
- Multi-model routing improves cost/quality tradeoff
- Dashboard shows cost breakdown per workflow

**📊 Scope Boundaries:**
- **ONLY** inference layer optimization (no training yet)
- **ONLY** vLLM (no custom CUDA kernels)
- **ONLY** quantization to INT8/INT4 (no GGML)

**⚙️ Dependencies:**
- Phase 1-3 complete (all three workflows working)
- GPU infrastructure access (Modal, Lambda Labs, or custom)
- Inference logs from Phase 1-3

**🔗 Next Phase Gate:**
- vLLM server passes latency benchmarks
- Multi-model router makes correct routing decisions
- Cost reduced by ≥20% vs. direct GPT-4o calls

---

### **Phase 5 — Model Fine-Tuning & Alignment** (4-6 weeks)

**🎯 Objectives:**
- Collect human-annotated feedback on Phase 1-4 conversations
- Fine-tune open-source models (Llama 70B, etc.) on insurance domain
- Use RLHF/DPO/GRPO to improve on judge scores
- Measure improvement vs. GPT-4o baseline

**📦 Deliverables:**
- ✅ Insurance Preference Dataset
  - 1,000+ human-annotated conversation pairs
  - Judge scores + human ratings (do they align?)
  - Failure mode classifications
  - Compliance violation examples
- ✅ Alignment Training Pipeline (Modal)
  - RLHF using judge scores as reward signal
  - DPO training on preference pairs
  - GRPO for multi-objective optimization (cost + quality)
  - Model versioning + checkpoint management
- ✅ Fine-Tuned Model Variants
  - Insurance-optimized Llama 70B (cheaper than GPT-4o, better domain knowledge)
  - Specialized underwriting model (form-filling expert)
  - Specialized claims model (empathy focus)
- ✅ Model Evaluation Report
  - Judge scores pre/post fine-tuning
  - Comparison vs. GPT-4o on insurance benchmarks
  - Cost/quality tradeoff analysis

**🎯 Success Criteria:**
- Fine-tuned models achieve ≥90% judge score (compliance + accuracy)
- Judge scores correlate ≥0.8 with human ratings
- Specialized models outperform GPT-4o on their domain
- Cost/call reduced by 40%+ while maintaining quality
- Alignment training converges in < 2 days on Modal

**📊 Scope Boundaries:**
- **ONLY** Llama-based models (not training GPT-4o)
- **ONLY** RLHF/DPO/GRPO (no supervised fine-tuning in isolation)
- **ONLY** insurance domain (not general)

**⚙️ Dependencies:**
- Phase 1-4 complete (data collection, evaluation infra)
- Judge system fully calibrated (≥0.8 correlation with humans)
- Preference dataset labeled by domain experts
- Modal GPU access

**🔗 Next Phase Gate:**
- Judge scores improve by ≥5 points post-training
- Human evaluators prefer fine-tuned model ≥70% of time
- Training converges without divergence

---

### **Phase 6 — Custom Infrastructure & Scale** (4-8 weeks)

**🎯 Objectives:**
- Replace Retell with custom LiveKit + Pipecat infrastructure
- Handle 1,000+ concurrent calls
- Add SIP/WebRTC for enterprise integrations
- Achieve true low-latency voice pipelines

**📦 Deliverables:**
- ✅ Custom Voice Pipeline (LiveKit + Pipecat)
  - WebRTC media server setup
  - Real-time audio streaming
  - Audio processing (VAD, AEC, noise suppression)
  - Speaker diarization
- ✅ Telephony Integration
  - SIP protocol support (enterprise PBX)
  - Twilio/Telnyx backend
  - Predictive dialer with call pacing
  - Call recording + compliance logging
- ✅ Scalable Architecture
  - Multi-region deployment
  - Load balancing across voice servers
  - State management for 1,000+ concurrent calls
  - Failover + recovery mechanisms
- ✅ Audio Processing Pipeline
  - VAD (Voice Activity Detection)
  - AEC (Acoustic Echo Cancellation)
  - Noise suppression algorithms
  - Quality metrics (MOS score)
- ✅ Enterprise Features
  - Call recording + secure storage
  - Compliance audit logs
  - Call routing policies (by geography, skill, queue)
  - SLA tracking (answer time, handling time)

**🎯 Success Criteria:**
- Handle ≥1,000 concurrent calls
- Sub-700ms latency at scale
- 99.9% uptime SLA
- Custom infrastructure matches Retell cost
- Enterprise SIP customers can connect

**📊 Scope Boundaries:**
- **ONLY** infrastructure for prospecting + underwriting + claims (no new workflows)
- **ONLY** open-source tools (LiveKit, Pipecat)
- **NOT** proprietary media servers

**⚙️ Dependencies:**
- Phase 1-5 complete (all workflows optimized)
- Infrastructure budget approved
- DevOps team to manage deployment

**🔗 Next Phase Gate:**
- Load test passes 1,000 concurrent calls
- Latency remains < 700ms under load
- Failover test succeeds

---

### **Phase 7 — Cloud Deployment & Operations** (1-2 weeks)

**🎯 Objectives:**
- Move from local dev → production cloud infrastructure
- Set up monitoring, alerting, auto-scaling
- Document operations runbook
- Prepare for scaling to 10,000+ calls/day

**📦 Deliverables:**
- ✅ Production Deployment
  - Voice agents → Vercel (edge)
  - Inference service → Modal GPUs
  - Database → Supabase (Postgres + PgVector)
  - Cache → Redis (Railway or Upstash)
  - Monitoring → Logfire dashboards
- ✅ Auto-Scaling Setup
  - Horizontal scaling for voice agents
  - GPU auto-scaling for inference
  - Queue management (Redis streams)
  - Retry queues with exponential backoff
- ✅ Observability & Alerting
  - Latency alerts (p99 > 1s)
  - Cost anomaly detection (unexpected spikes)
  - Judge score degradation alerts
  - Escalation rate anomaly alerts
  - Model availability alerts
- ✅ Operations Runbook
  - Incident response procedures
  - Rollback strategies
  - Model deployment workflow
  - On-call rotation setup
- ✅ Performance Baselines
  - Benchmark all three workflows
  - Establish SLOs (Service Level Objectives)
  - Cost tracking by workflow/customer

**🎯 Success Criteria:**
- All systems deployed to production
- Monitoring dashboard accessible
- Auto-scaling test passes
- ≥2 successful incident simulations resolved
- Cost tracking accurate within 1%
- 99.5% uptime achieved over 1 week

**📊 Scope Boundaries:**
- **ONLY** core workflows (prospecting, underwriting, claims)
- **ONLY** cloud-native infrastructure
- **NOT** new features (operations focus only)

**⚙️ Dependencies:**
- Phase 1-6 complete
- Production environment setup
- Security audit complete
- DevOps infrastructure ready

**🔗 Next Phase Gate:**
- Production deployment successful
- Monitoring shows all green
- Zero critical incidents in first week

---

## 📋 Phase Dependency Map

```
Phase 1 (Voice MVP + Judge)
    ↓
Phase 2 (Platform Tools)
    ↓
Phase 3 (Multi-Workflow Expansion)
    ↓
Phase 4 (Inference Optimization) ← Parallel: Phase 5 (Alignment)
    ↓
Phase 6 (Custom Infrastructure)
    ↓
Phase 7 (Cloud Deployment)
```

**Timeline Summary:**
- **Phase 1:** 3-4 weeks (foundation)
- **Phase 2:** 2-3 weeks (enablement)
- **Phase 3:** 3-4 weeks (features)
- **Phase 4 + 5:** 6-9 weeks (optimization + training, can overlap)
- **Phase 6:** 4-8 weeks (scale)
- **Phase 7:** 1-2 weeks (operations)

**Total MVP-to-Scale: ~6-7 months**

---

## 🎯 Harper Role Alignment

This project is structured to demonstrate **both combined Harper roles**:

### **AI Engineer - Voice** (Primary MVP Focus)
✅ **Covered in MVP**:
- Outbound prospecting agent (Phase 1)
- Sub-700ms latency optimization
- Retell → LiveKit infrastructure progression
- Deepgram STT + ElevenLabs TTS integration
- Twilio telephony integration
- Production monitoring via Logfire

🔜 **Phases 2-3**:
- Form-filling agent (20-30 min conversations)
- Claims intake (FNOL) agent
- Policy service agent (24/7)
- Multi-modal orchestration (voice → SMS → email)
- Custom LiveKit + Pipecat infrastructure
- No-code workflow builder for non-technical teams

### **AI Research Engineer** (Phase 1 Parallel Track)
✅ **Covered in MVP**:
- MLLM-as-a-Judge evaluation system (Phase 1, critical)
- Insurance domain benchmarks + rubrics
- Production monitoring dashboard
- Conversation analytics for failure mode analysis

🔜 **Phases 2-4**:
- Synthetic dataset generation at scale
- RLHF/DPO training pipelines (Phase 4)
- Cost-latency optimization research
- Model routing strategies

---

## 📚 Skill Requirements by Component

| Component | Skills Required | Difficulty | Estimate |
|-----------|-----------------|------------|----------|
| **Voice Agent (Retell)** | FastAPI, async Python, Retell API, Twilio | Beginner→Intermediate | 1-2 weeks |
| **MLLM-as-a-Judge** | Prompt engineering, GPT-4o, eval rubrics, Logfire | Intermediate | 1 week |
| **Analytics Dashboard** | React/Next.js, Postgres queries, real-time metrics | Intermediate | 1-2 weeks |
| **Inference Service** | vLLM, CUDA optimization, quantization | Advanced | 2-3 weeks |
| **Alignment Training** | RLHF/DPO, Modal GPU jobs, verl framework | Advanced | 2-4 weeks |
| **Custom Infrastructure** | LiveKit, Pipecat, WebRTC, SIP protocols | Expert | 4-8 weeks |

---

## 🔧 No-Code & Low-Code Tools Clarification

Based on the Harper job postings, here's the distinction:

### **Low-Code Platforms** (Used for MVP Rapid Development)
These are **existing products** you use to build quickly:
- **VAPI** or **Retell** — Low-code voice agent builders (dashboard-based, minimal coding)
- **Twilio** — Telephony APIs (code-light integration)
- **Logfire** — Observability dashboard (pre-built dashboards)

### **Internal No-Code Tools** (You'll Build in Phase 2)
These are **custom tools** for your non-technical operations/sales teams:

1. **Voice Workflow Builder** (React-based)
   - Drag-drop conversation flow designer
   - Intent/entity mapper
   - System prompt editor with tone controls
   - Escalation rule configuration
   - No coding required — exports to YAML → FastAPI

2. **A/B Testing Dashboard**
   - Visual test setup (choose 2 prompts/personas)
   - Statistical significance calculator
   - Auto-winner detection and rollout

3. **Conversation Analytics Portal**
   - Real-time call metrics
   - Failure clustering & root cause
   - Judge score drilldowns
   - Cost analysis by workflow type

4. **Agent Template Gallery**
   - Pre-built flows for common insurance workflows
   - One-click deployment to production
   - Version control & rollback

**MVP includes**: Low-code platforms (#1) + analytics dashboard (#2)  
**Phase 2+**: Internal no-code builder + A/B testing + template gallery

---

## ✅ Why These Choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Voice Infra | Retell → LiveKit | Low-code start, scalable to custom WebRTC |
| TTS / STT | ElevenLabs / Deepgram | Lowest latency + highest quality voices |
| Serving | vLLM | Proven sub-second latency, KV caching, speculative decoding |
| Post-Training | verl + RLHF/DPO/GRPO/RLAIF | Modern, efficient, matches Harper JD stack |
| Monitoring | Logfire | Built-in observability for AI pipelines |
| Deployment | Modal + Vercel | Separation of GPU compute and real-time serving |
| Data Enrichment | Async microservices | True Harper-style "AI Grid" scalability |

---

## 🚀 Future Work

- Expand data enrichment to include real insurance APIs.
- Integrate compliance validation for US state policies.
- Build customer-facing React dashboard for insights.
- Add real-time evaluation dashboards in Logfire.
- Experiment with TensorRT-LLM inference optimization.

---

## 🧩 Summary

AI Brokerage demonstrates how an AI-native commercial insurance system can run multiple intelligent agents — voice, email, inference, alignment — all under the REAL framework.

Each agent is autonomous, reliable, and measurable, proving end-to-end understanding of Harper's engineering philosophy:

> **"Start practical, scale smart, and ship daily."**

---

### 📌 Project Metadata

**Author:** Anish Gillella

**Deployment Targets:** Modal (GPU) · Vercel (Edge) · Logfire (Monitoring)

**Tech Stack:** FastAPI · vLLM · verl · Retell · Deepgram · ElevenLabs · Redis · Postgres

---

### ✅ Next Steps

If this looks aligned, I can:
1. Generate the **matching folder scaffolding** (`__init__.py`, `main.py`, placeholder APIs per agent).
2. Or write the **synthetic data generation script** (`generate_insurance_data.py`) that follows this README.

Would you like me to start with the **folder scaffolding** next?