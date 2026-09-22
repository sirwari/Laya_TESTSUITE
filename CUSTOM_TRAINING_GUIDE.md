# Retraining & Fine-Tuning Laya on Custom Data

This guide explains how to fine-tune **Laya** (`convaiinnovations/laya` or `convaiinnovations/laya-multilingual`) on your own support tickets, enterprise routing workflows, or domain-specific classification schemas.

---

## 💡 Why Fine-Tune Laya?

While Laya models achieve zero-shot performance out-of-the-box, fine-tuning on your proprietary historical tickets or domain schemas:
1. **Pushes Accuracy to Production Grade**: Increases classification accuracy (e.g. from 80% to 95%+) on domain-specific jargon, internal ticket IDs, or specialized taxonomy.
2. **Sharpens Confidence Calibration**: Aligns probability scores directly with your team's historical SLA targets.
3. **Optimizes Decision Latency**: Retains single-pass sub-35 ms execution while learning custom enterprise routing logic.

---

## 1. Preparing Your Training Dataset

Laya fine-tuning accepts labeled JSON data where each sample maps an input **state** (e.g. ticket subject + body) to **typed questions** and **gold target labels**.

### Example Dataset JSON (`custom_tickets_train.json`):
```json
[
  {
    "state": {
      "subject": "504 Gateway Timeout during PostgreSQL migration",
      "body": "All queries to customer database are failing since 14:00 UTC."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {
          "database_infra": "PostgreSQL, DB timeouts, queries",
          "billing": "invoices, payment failures",
          "security": "unauthorized logins, 2FA"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "Rate severity level",
        "criteria": ["low", "medium", "critical outage"]
      }
    },
    "gold": {
      "department": "database_infra",
      "severity": 2
    }
  }
]
```

---

## 2. RLCD Loss Function (Proper Scoring Rules)

Laya is fine-tuned using **RLCD (Reinforcement Learning with Strictly Proper Scoring Rules)**. Rather than standard cross-entropy which can cause overconfident misclassifications:

- **Brier Score Reward Loss** minimizes squared error between predicted probabilities $P(y)$ and ground truth target $Y$:
$$\text{Brier Loss} = \sum_{i=1}^{K} (P(y_i) - Y_i)^2$$
- Policy gradients weight log probabilities by calibrated proper-scoring rewards.

---

## 3. Fine-Tuning Script (`examples/train_custom_laya.py`)

Run the standalone fine-tuning script on your local GPU (NVIDIA RTX 3090, Apple Silicon MPS, or Kaggle/Colab GPUs):

```bash
/home/geo/Laya_Test/venv/bin/python examples/train_custom_laya.py
```

### Core Code Snippet:
```python
import laya
from laya.common import build_sequence, collate_items, QTYPES
from safetensors.torch import save_file

# 1. Load base model
agent = laya.load("convaiinnovations/laya")
model = agent.model.cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# 2. Training Loop with RLCD Loss
model.train()
for sample in dataset:
    optimizer.zero_grad()
    # Build sequence & forward pass
    logits, act = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    loss = rlcd_brier_loss(logits, gold_target_tensor)
    loss.backward()
    optimizer.step()

# 3. Save Custom Checkpoint
save_file(model.state_dict(), "./custom_laya_checkpoint/model.safetensors")
```

---

## 4. Loading Your Custom Checkpoint in Production

Once training completes and saves your model to `./custom_laya_checkpoint`:

### Direct Load:
```python
import laya

# Load your custom fine-tuned checkpoint
agent = laya.load("./custom_laya_checkpoint")
res = agent.predict(state, questions)
```

### Attach to Production Router:
```python
from laya import Router

router = Router(preload=True)

# Attach your custom fine-tuned agent as an explicit route
custom_agent = laya.load("./custom_laya_checkpoint")
router.attach("custom_enterprise", custom_agent)

# Dispatch requests to your custom checkpoint
res = router.predict(state, questions, model="custom_enterprise")
```

---

## 5. Multi-GPU Distributed Fine-Tuning (DDP)

For large datasets (>10,000 cases), use PyTorch Distributed Data Parallel (`torchrun`):
```bash
torchrun --nproc_per_node=2 notebooks/train_ddp.py
```
*(Reference implementation is available in `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`).*
