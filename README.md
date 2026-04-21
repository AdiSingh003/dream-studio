<div align="center">

```
██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
██║  ██║██████╔╝█████╗  ███████║██╔████╔██║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

### *An End-to-End Text-to-Image Generation Studio*

> **DreamForge** is a production-grade generative AI studio built on Stable Diffusion v1.5, featuring LoRA domain adaptation, ASHA-guided hyperparameter optimisation, magnitude-based pruning, and a full MLOps observability stack — all packaged in a deployable Docker Compose environment and deployed on **Google Cloud Platform with NVIDIA L4 GPU**.

<br/>

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-ff4b4b?logo=streamlit)](http://35.237.254.142:8503)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194e2?logo=mlflow)](http://35.237.254.142:5012)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-e6522c?logo=prometheus)](http://35.237.254.142:9092)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?logo=github)](https://github.com/AdiSingh003/dream-studio)

---

</div>

---

## 🔭 Overview

Large-scale latent diffusion models excel at broad visual generation — but they systematically under-represent **culturally specific imagery**. Generic prompts to Stable Diffusion v1.5 fail to reproduce the warmth of Diwali diyas, the chromatic chaos of Holi, or the grandeur of Ganesh Chaturthi processions.

**DreamForge** solves this through a complete, reproducible MLOps pipeline:

```
Dataset  ──►  LoRA Fine-Tuning  ──►  ASHA HPO  ──►  Pruning  ──►  Inference  ──►  Monitoring
   │                │                    │               │              │               │
IndianFestivals   PEFT on           Optuna +        L1-norm        Streamlit      MLflow +
  (3,500+ imgs)   UNet attn.    SuccessiveHalving   pruning        UI + API      Prometheus
```

The system is designed from first principles for **reproducibility**, **extensibility**, and **production readiness** — not just research demos.

---

## 🚀 Live Deployment (GCP NVIDIA L4)

DreamForge is deployed on **Google Cloud Platform** (`g2-standard-4` instance, NVIDIA L4 24 GB VRAM) as a three-container Docker Compose stack.

| Service | URL | Description |
|---|---|---|
| 🎨 **Streamlit UI** | [http://35.237.254.142:8503](http://35.237.254.142:8503) | Image generation interface |
| 📊 **MLflow UI** | [http://35.237.254.142:5012](http://35.237.254.142:5012) | Experiment tracking & run comparison |
| 🔥 **Prometheus** | [http://35.237.254.142:9092](http://35.237.254.142:9092) | Real-time metrics & alerts |

**Source Code:** [https://github.com/AdiSingh003/dream-studio](https://github.com/AdiSingh003/dream-studio)

### Production Metrics (Observed on GCP L4)

| Metric | Value |
|---|---|
| GPU | NVIDIA L4 |
| VRAM Total | 22,563 MB |
| VRAM Used (model loaded) | 2,653 MB (11.8%) |
| Avg Generation Latency | 4.9 s |
| Avg Steps/s | 6.9 |
| P50 Latency | 2.7 s |
| P95 Latency | 10.5 s |
| P99 Latency | 11.5 s |
| Error Rate | 0.0% |

---

## 📸 Screenshots

### Streamlit UI — LoRA Fine-Tuned Image Generation
> Prompt: *"ganesh chatuthi celebrations"* · Scheduler: DPM++ 2M · Model: Fine-tuned (LoRA) · Size: 512×512

![Streamlit UI](lora_finetuned_image.png)

### MLOps Monitoring Dashboard
> Real-time GPU status, latency trends, scheduler distribution, and performance metrics

![MLOps Dashboard](mlops_monitoring_dashboard_in_app.png)

### MLflow Experiment Tracking
> 7 logged runs with generation_time_s, steps_per_second, images_generated, pixels_generated

![MLflow Dashboard](mlflow_dashboard.png)

---

## 🏗 Architecture

DreamForge is decomposed into four clean tiers:

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND TIER                            │
│   Streamlit UI (port 8503) · Plotly Dashboards             │
│   Generation Controls · Session Gallery · Monitoring Tab   │
├─────────────────────────────────────────────────────────────┤
│                    PIPELINE TIER                            │
│   ModelManager (Singleton, thread-safe, scheduler swap)    │
│   InferenceEngine · PromptProcessor · ImageProcessor       │
│   BatchProcessor (PriorityQueue + async worker)            │
├─────────────────────────────────────────────────────────────┤
│                   MONITORING TIER                           │
│   MetricsCollector (Prometheus, port 8012)                 │
│   MLflowTracker (runs, params, artifacts, port 5012)       │
│   DriftDetector (latency + error rate sliding windows)     │
├─────────────────────────────────────────────────────────────┤
│                    TRAINING TIER                            │
│   LoRATrainer (PEFT · AMP · cosine LR · noise offset)     │
│   hpo_objective (Optuna + ASHA SuccessiveHalvingPruner)    │
│   data.py · prepare_mini_data.py                           │
└─────────────────────────────────────────────────────────────┘
```

### Docker Compose Stack (GCP Deployment)

```
┌─────────────────────────────────────────────────────────────┐
│                     dreamforge_net (bridge)                 │
│                                                             │
│  dreamforge_app        mlflow             prometheus        │
│  :8503 (Streamlit)     :5012 (UI)         :9092 (UI)        │
│  :8012 (metrics) ──────────────────────► scrapes :8012      │
│                        SQLite + volumes                     │
└─────────────────────────────────────────────────────────────┘
```

### ModelManager

- **Singleton pattern** with `threading.Lock` — one model load per process
- Device resolution cascade: `CUDA → Apple MPS → CPU`
- Three-level fallback during `from_pretrained` for models lacking fp16 safetensors
- **Scheduler hot-swapping** across DDIM, Euler-A, DPM++ 2M, PNDM, LMS, DPM++ SDE

### InferenceEngine

- Accepts `GenerationConfig` dataclass; always returns `GenerationResult`
- Explicit CUDA OOM handling with `torch.cuda.empty_cache()` and actionable error messages
- The UI layer **never receives an unhandled exception**

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+, CUDA 12.1+ recommended
pip install -r requirements.txt
```

### Docker Compose (Recommended)

```bash
git clone https://github.com/AdiSingh003/dream-studio.git
cd dream-studio

# Create .env file
cp .env.example .env
# Add your HF token: HUGGING_FACE_HUB_TOKEN=hf_your_token_here

# Build and run all services
docker compose build --build-arg HF_TOKEN=hf_your_token_here
docker compose up -d
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8503 |
| MLflow UI | http://localhost:5012 |
| Prometheus Metrics | http://localhost:9092 |

### Local Development

```bash
# 1. Prepare dataset
python training/data.py

# 2. (Optional) Mini-dataset for fast HPO iterations
python training/prepare_mini_data.py

# 3. Run HPO
python training/hpo_objective.py --n-trials 10

# 4. Fine-tune with best trial config
python training/train.py --config configs/best_trial.yaml

# 5. Prune the adapter
python training/prune.py --sparsity 0.20

# 6. Launch the UI
streamlit run app/main.py
```

---

## 🎨 LoRA Fine-Tuning

### Dataset: IndianFestivals

DreamForge fine-tunes on the [`AIMLOps-C4-G16/IndianFestivals`](https://huggingface.co/datasets/AIMLOps-C4-G16/IndianFestivals) dataset — **3,500+ images** spanning seven festival categories with richly engineered text prompts:

| Festival | Prompt Themes |
|---|---|
| 🪔 **Diwali** | Clay oil diyas, fireworks, rangoli patterns, golden illumination |
| 🎨 **Holi** | Coloured powder, vibrant spring celebration, joyful crowds |
| 🕌 **Eid** | Mosque architecture, crescent moon, traditional attire, lanterns |
| 🐘 **Ganesh Chaturthi** | Lord Ganesha idol, colourful decorations, traditional worship |
| 🇮🇳 **Independence Day** | National flag, patriotic parade, saffron–white–green |
| ☀️ **Lohri** | Bonfire, folk dance, winter harvest celebration |
| 🎄 **Christmas** | Festive lights, trees, winter celebration |

**Split:** 80% train / 10% val / 10% test (stratified by class)

### LoRA Formulation

LoRA injects trainable rank-decomposed matrices into the UNet attention layers while freezing the 860 M-parameter backbone:

```
h = W₀x + (α/r) · B·A·x

where:
  W₀  ∈ ℝᵈˣᵏ  — frozen pre-trained weight
  A   ∈ ℝʳˣᵏ  — initialised from 𝒩(0, σ²)
  B   ∈ ℝᵈˣʳ  — initialised to zero (zero-delta init)
  r           — rank (controls adapter capacity)
  α           — scaling factor
```

**Target layers** (6 per UNet transformer block):
`to_q · to_k · to_v · to_out.0 · ff.net.0.proj · ff.net.2`

### Best Trial Configuration

| Parameter | Value | Notes |
|---|---|---|
| LoRA Rank `r` | **16** | Also evaluated: 4, 8 |
| LoRA Alpha `α` | **32** | Scale = α/r = 2.0 |
| Dropout | 0.05 | Applied to LoRA layers |
| Trainable Params | **~3.4 M** | vs. 860 M frozen UNet |
| Learning Rate | 1.34 × 10⁻⁴ | AdamW + cosine schedule |
| Warmup Ratio | 0.05 | 5% of total steps |
| Batch Size | 4 | Per-GPU |
| Gradient Accumulation | 2 | Effective batch = 8 |
| Mixed Precision | fp16 | `torch.cuda.amp.GradScaler` |
| Noise Offset | 0.05 | Improves dark-scene fidelity |
| Epochs | 10 | Best checkpoint by val loss |

### Training Objective

The denoising score-matching loss in latent space:

```
ℒ = 𝔼[‖ε − εθ(zₜ, t, τθ(c))‖²]

where:
  z₀ = E(x)    — VAE-encoded latent
  ε ~ 𝒩(0, I)  — ground-truth noise
  τθ(c)        — CLIP text embedding (frozen)
```

The VAE encoder, VAE decoder, and CLIP text encoder are **all frozen**. Only LoRA parameters inside the UNet are updated.

---

## ⚡ ASHA Hyperparameter Optimisation

### Algorithm

Asynchronous Successive Halving (ASHA) launches many trials concurrently and **early-stops unpromising ones** — eliminating the synchronisation barrier of standard SHA. A trial is promoted to the next rung only if it ranks in the top `1/η` fraction of all trials reaching that rung.

### Search Space

```python
r       ∈  {4, 8, 16}
α       ∈  {16, 32, 64}
p_drop  ~  Uniform(0.0, 0.2)
η_lr    ~  LogUniform(1e-5, 2e-4)
B       ∈  {1, 2, 4}
```

### Integration

```python
# Non-invasive loop integration
if trial is not None:
    trial.report(train_loss, global_step)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

### Top-5 Trial Results

| Trial | `r` | `α` | `η_lr` | `B` | Val Loss | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **7** ⭐ | 16 | 32 | 1.34×10⁻⁴ | 4 | **0.1823** | Completed |
| 3 | 16 | 64 | 9.8×10⁻⁵ | 2 | 0.1941 | Completed |
| 9 | 8 | 32 | 1.12×10⁻⁴ | 4 | 0.2017 | Completed |
| 1 | 4 | 16 | 5.6×10⁻⁵ | 2 | 0.2209 | Completed |
| 5 | 4 | 32 | 2.3×10⁻⁵ | 1 | 0.2451 | Pruned (rung 1) |

> ⭐ **Trial 7** is used for the full 10-epoch fine-tune. ASHA pruned **6 of 10 trials** at the first rung, achieving a **4.8× wall-clock speedup** over exhaustive grid search.

---

## ✂️ Post-Training Pruning

DreamForge applies **ℓ₁-norm magnitude pruning** to LoRA adapter matrices A and B after training. The frozen backbone is never touched.

```python
import torch.nn.utils.prune as prune

def prune_lora_weights(unet, sparsity=0.20):
    lora_layers = [
        (m, 'weight') for n, m in unet.named_modules()
        if 'lora' in n.lower() and hasattr(m, 'weight')
    ]
    prune.global_unstructured(
        lora_layers,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    # Make masks permanent before saving
    for module, param in lora_layers:
        prune.remove(module, param)
    return unet
```

### Sparsity–Quality Trade-off

| Sparsity | Val Loss | Adapter Size Reduction | Recommendation |
|:---:|:---:|:---:|:---:|
| 0% | 0.1823 | — | Baseline |
| 10% | 0.1829 | 9% | Conservative |
| **20%** | **0.1847** | **18%** | ✅ **Recommended** |
| 30% | 0.1891 | 26% | Acceptable |
| 50% | 0.2103 | 40% | Degraded |
| 70% | 0.2614 | 57% | ❌ Not recommended |

> **20% sparsity is the sweet spot**: 18% memory reduction for only a 1.3% relative increase in validation loss.

---

## 📊 MLOps Stack

### Prometheus Metrics  `→ app port 8012 · UI port 9092`

| Metric | Type | Description |
|---|---|---|
| `sd_generations_total` | Counter | Cumulative generations (labelled by status + scheduler) |
| `sd_generation_duration_seconds` | Histogram | Latency distribution (10 buckets, 5–300 s) |
| `sd_steps_per_second` | Gauge | Current inference throughput |
| `sd_active_generations` | Gauge | Concurrent in-flight requests |
| `sd_gpu_memory_used_bytes` | Gauge | CUDA allocated memory |

> The app exports metrics on port **8012** inside the Docker network. The Prometheus container scrapes `dreamforge:8012` and exposes the UI on host port **9092**.

**Alert Rules:**

| Alert | Condition | Severity |
|---|---|---|
| `DreamForgeAppDown` | Exporter unreachable > 1 min | 🔴 Critical |
| `HighGenerationErrorRate` | Error rate > 10% over 5 min | 🟡 Warning |
| `HighGenerationLatency` | Avg latency > 60 s over 5 min | 🟡 Warning |

### MLflow Experiment Tracking  `→ port 5012`

Every generation and training run is logged as an MLflow run with:
- **Parameters:** prompt length, steps, guidance scale, scheduler, seed, LoRA mode
- **Metrics:** latency, steps/s, pixel count, validation loss
- **Artifacts:** generated images, prompt text
- **Backend:** SQLite (`mlflow.db`) + host-mounted artifact storage

### Statistical Drift Detection

```
Δ_lat = (L̃_recent − L̃_baseline) / L̃_baseline × 100%

Alerts:
  Δ_lat ≥ 25%  →  ⚠️  Warning
  Δ_lat ≥ 75%  →  🚨  Critical
  Δ_err change > threshold  →  Error rate drift alert
```

### Model Registry

File-backed version tracking with four stage labels:

```
experimental  →  staging  →  production  →  archived
```

Each version stores: HuggingFace model ID · MLflow run ID · evaluation metrics · provenance tags — serialised to a human-readable `registry.json`.

---

## 🔬 Experimental Results

**Training Hardware:** NVIDIA RTX 3090 (24 GB VRAM) · Intel Core i9-12900K · 64 GB DDR5 · Ubuntu 22.04
**Production Hardware:** GCP `g2-standard-4` · NVIDIA L4 (24 GB VRAM) · Ubuntu 22.04
**Software:** PyTorch 2.4.0 · CUDA 12.1 · HuggingFace Diffusers 0.30.3 · PEFT 0.12.0

### Fine-Tuning Results

| Model | Val Loss | Test Loss | Trainable Params | Train Time |
|---|:---:|:---:|:---:|:---:|
| SD v1.5 (zero-shot) | 0.2501 | 0.2548 | 0 | — |
| LoRA r = 4 | 0.2209 | 0.2241 | 0.9 M | 1.2 h |
| LoRA r = 8 | 0.2017 | 0.2039 | 1.7 M | 1.6 h |
| **LoRA r = 16 (ours)** | **0.1823** | **0.1856** | **3.4 M** | **2.3 h** |

> **27.1% validation loss reduction** over the zero-shot baseline using only **0.39%** of UNet parameters.

### GCP L4 Production Inference

| Metric | Value |
|---|---|
| Avg Generation Latency | 4.9 s |
| Avg Steps/s | 6.9 |
| P50 Latency | 2.7 s |
| P95 Latency | 10.5 s |
| P99 Latency | 11.5 s |
| VRAM Utilisation | 11.8% (2653 / 22563 MB) |
| Error Rate | 0.0% |
| MLflow Runs Logged | 7 (4 generation + 3 model_load) |

### Scheduler Performance (512×512, 20 steps)

| Scheduler | Latency (s) | Steps/s | Best For |
|---|:---:|:---:|---|
| DDIM | 4.8 | 4.2 | General use, deterministic |
| **Euler A** | **4.6** | **4.3** | Creative variety |
| DPM++ 2M | 5.1 | 3.9 | Highest quality/step |
| PNDM | 4.7 | 4.3 | Stable baseline |
| LMS | 5.3 | 3.8 | Smooth results |
| DPM++ SDE | 6.2 | 3.2 | Stochastic quality |

### MLOps Overhead

| Component | Overhead | Notes |
|---|---|---|
| MLflow logging | ~0.3 s/generation | 6% overhead at 5 s baseline |
| Prometheus update | < 1 ms | Negligible |
| Drift detection | < 2 ms | 100-sample window |

---

## ⚙️ Configuration Reference

```yaml
# configs/best_trial.yaml

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - to_q
    - to_k
    - to_v
    - to_out.0
    - ff.net.0.proj
    - ff.net.2

training:
  learning_rate: 1.34e-4
  warmup_ratio: 0.05
  batch_size: 4
  gradient_accumulation_steps: 2
  mixed_precision: fp16
  noise_offset: 0.05
  epochs: 10

pruning:
  sparsity: 0.20
  method: l1_unstructured
  target: lora_only

monitoring:
  mlflow_uri: http://mlflow:5012
  prometheus_port: 8012
  drift_window_size: 100
  drift_latency_warning_pct: 25
  drift_latency_critical_pct: 75
```

---

## 🙏 Acknowledgements

We thank the **AIMLOps-C4-G16 team** for releasing the IndianFestivals dataset on Hugging Face, and the **HuggingFace Diffusers** and **PEFT** teams for their exceptional open-source libraries.

---

<div align="center">

**Built with ❤️ at IIT Jodhpur · M.Tech Artificial Intelligence**

*Aditya Pratap Singh (M25CSA002) · Gadiya Mahek Shankesh (M25CSA011)*

<br/>

```
"From festival prompts to photorealistic pixels — DreamForge bridges the gap."
```

</div>