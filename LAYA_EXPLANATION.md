# Technical Deep-Dive: Laya System 1 Decision Engine

## Executive Overview
**Laya** is a high-speed, non-autoregressive decision engine designed for single-forward-pass typed evaluation (`choice`, `score`, `noul`) over arbitrary state documents (emails, support tickets, JSON, unstructured text) in **over 100 languages**.

Unlike traditional Generative AI models (LLMs) that generate text token-by-token using expensive autoregressive decoding loops:
- **Laya generates no text tokens.**
- It executes **one single neural network forward pass** per request.
- Returns calibrated probability distributions over your exact schema in **~33 milliseconds** on an RTX 3090 / Apple Silicon GPU.

---

## 1. System 1 vs System 2 AI Architectures

| Characteristic | System 2 (Generative LLMs - Llama, GPT-4) | System 1 (Non-Autoregressive - Laya) |
|---|---|---|
| **Mechanism** | Autoregressive token-by-token generation | Single forward classification pass |
| **Inference Time** | 500 ms – 5,000+ ms | **25 ms – 40 ms** |
| **Parsing/Schema** | Requires regex/JSON repair parser | **Natively typed** (`choice`, `score`, `noul`) |
| **Hallucination Risk**| High (can generate invalid JSON or hallucinate text) | **Zero** (no free-form text generation) |
| **GPU VRAM Impact** | High KV-Cache memory consumption | **Static model footprint (~1.2 GB for 3 models)** |
| **Training Objective**| Next-token prediction (Cross-Entropy) | **RLCD (Reinforcement Learning against Strictly Proper Scoring Rules)** |

---

## 2. Checkpoints & Preloaded Router Architecture

Laya features three specialized encoder models:

1. **`laya` (ModernBERT-large, 421M params)**
   - Optimized for high-accuracy English processing.
   - Context Window: 512 tokens.
2. **`laya-multilingual` (mmBERT-base, 322M params)**
   - Supports 100+ languages and non-Latin scripts (Devanagari, Kanji, Cyrillic, Arabic).
   - 2x faster execution (~32.8 ms). Context Window: 1024 tokens.
3. **`laya-typed-decisions` (ModernBERT-large, 421M params)**
   - Specialized for multi-step structured workflow decision trees.

### Preloaded Router (`Router(preload=True)`)
- At runtime, the built-in **`Router`** inspects the input state in **<0.5 ms** (pure Python script/unicode analysis).
- Automatically routes non-Latin or non-English text to `laya-multilingual` while keeping English queries on `laya`.
- Prevents script confidence hallucination (e.g., preventing the English model from confidently misclassifying Khmer or Hindi text).
- **VRAM Footprint**: Preloading all 3 models takes **~1.2 GB VRAM**, leaving remaining VRAM / Unified RAM free for concurrent workloads.

---

## 3. Apple Silicon MacBook & MLX Integration 🍎

Laya natively supports **Apple Silicon MacBooks** (M1/M2/M3/M4 Pro/Max/Ultra) via PyTorch Metal Performance Shaders (`device="mps"`) and Apple **MLX** array representations:

- **Apple Unified Memory Advantage**: On MacBooks, CPU and GPU share the same high-speed LPDDR5/LPDDR5X RAM pool. All 3 Laya checkpoints reside in unified RAM without PCIe transfer overhead.
- **Apple MLX Integration**: Probability vectors from Laya can be converted directly into native Apple MLX arrays (`mlx.core.array`) for zero-copy composition into downstream MLX pipelines:
```python
import mlx.core as mx

# Convert Laya probabilities to Apple MLX Array
dept_probs = res["answers"]["department"]["probabilities"]
mlx_probs = mx.array(list(dept_probs.values()))
print("Apple MLX Array:", mlx_probs)
```

---

## 4. Hardware Telemetry & Performance Benchmark

| **NVIDIA RTX 3090** | PyTorch CUDA 13.2 | ~4.4 GB Reserved | **~9.2 ms** |
| **Apple Silicon MacBook (M1-M4)** | PyTorch MPS + MLX | ~1.2 GB Unified RAM | **~25.0 ms** |
| **CPU (Intel / AMD / ARM)** | PyTorch CPU | ~1.2 GB System RAM | **~190.0 ms** |

---

## 5. Input Token Length & Context Scaling (Up to 8,192 Tokens)

### Default Token Budgets
- `laya` (ModernBERT-large): `max_len = 512` tokens (`head_max_len = 192` options budget).
- `laya-multilingual` (mmBERT-base): `max_len = 1024` tokens (`head_max_len = 256` options budget).

### Scaling to 8,192 Tokens at Runtime
ModernBERT & mmBERT support an architectural limit of **8,192 tokens**. You can dynamically scale context windows at runtime without retraining:

```python
import laya

agent = laya.load("convaiinnovations/laya-multilingual")

# Expand total context to 8,192 tokens (7,168 tokens for document state ~ 5,500 words)
agent.cfg["max_len"] = 8192
agent.cfg["head_max_len"] = 1024

# Single pass prediction over long document
result = agent.predict(long_document_state, questions)
```
