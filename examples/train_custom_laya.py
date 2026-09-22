#!/usr/bin/env python3
"""
Custom Fine-Tuning Guide & Reference Script for Laya System 1 Engine.

Demonstrates how to fine-tune Laya ('convaiinnovations/laya', 421M params)
on custom labeled support ticket datasets or domain-specific typed decision schemas.

Training Objective:
  - RLCD (Reinforcement Learning against Strictly Proper Scoring Rules - Brier Score Reward).
  - Multi-task policy gradient over 'choice', 'score', and 'noul' heads.
  - Temperature calibration fitting post-training.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import laya
from laya.common import build_sequence, render_options, collate_items, QTYPES, build_model
from laya.agent import Agent

# ==============================================================================
# 1. Custom Dataset Format
# ==============================================================================
# Each training sample contains:
#   - state: Dict or string document (e.g. support ticket subject + body)
#   - question: Dict defining question schema ('type', 'instructions', 'criteria')
#   - gold_target: String matching the correct criteria key (for choice), level int (for score), or bool (for noul)

MOCK_CUSTOM_DATASET = [
    {
        "state": {"subject": "Database timeout error", "body": "Queries to PostgreSQL are failing with 504 Gateway Timeout."},
        "question_id": "dept",
        "question": {
            "type": "choice",
            "instructions": "Which department handles this?",
            "criteria": {"tech_support": "database, server, API crash", "billing": "invoices", "sales": "contracts"}
        },
        "gold_target": "tech_support"
    },
    {
        "state": {"subject": "Need invoice copy for Q3", "body": "Please email us the PDF invoice for transaction #8812."},
        "question_id": "dept",
        "question": {
            "type": "choice",
            "instructions": "Which department handles this?",
            "criteria": {"tech_support": "database, server", "billing": "invoices, payment", "sales": "contracts"}
        },
        "gold_target": "billing"
    },
    {
        "state": {"subject": "We want to upgrade to Enterprise 500 seats", "body": "Can someone from sales call us today to finalize contract terms?"},
        "question_id": "dept",
        "question": {
            "type": "choice",
            "instructions": "Which department handles this?",
            "criteria": {"tech_support": "database", "billing": "invoices", "sales": "enterprise sales, contracts, upgrades"}
        },
        "gold_target": "sales"
    },
    {
        "state": {"subject": "Cancel our subscription immediately", "body": "Your service is too slow. Refund us or we switch to Competitor X."},
        "question_id": "churn",
        "question": {
            "type": "noul",
            "instructions": "Does the customer threaten to cancel or switch?"
        },
        "gold_target": "yes"
    }
]

# ==============================================================================
# 2. RLCD Loss Calculation (Reinforcement Learning with Proper Scoring Rules)
# ==============================================================================
def rlcd_brier_loss(logits, gold_indices, t_scale=1.0):
    """
    Computes Brier Score Reward Policy Gradient Loss for calibrated probabilities.
    Brier Score = sum((p_i - y_i)^2)
    """
    scaled_logits = logits / t_scale
    probs = torch.softmax(scaled_logits, dim=-1)
    
    batch_size, num_options = logits.shape
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, gold_indices.unsqueeze(1), 1.0)
    
    # Brier score error vector
    brier_error = torch.sum((probs - one_hot) ** 2, dim=-1)
    
    # Policy gradient loss weighted by log probability of true target
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    gold_log_probs = log_probs.gather(1, gold_indices.unsqueeze(1)).squeeze(1)
    
    loss = torch.mean(brier_error * (-gold_log_probs))
    return loss

# ==============================================================================
# 3. Custom Fine-Tuning Execution Pipeline
# ==============================================================================
def train_custom_laya(dataset, output_dir="./custom_laya_checkpoint", epochs=3, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n[FINE-TUNING] Target Device: {device}")
    print(f"[FINE-TUNING] Loading base model 'convaiinnovations/laya'...")

    # Load agent configuration & tokenizer
    base_agent = laya.load("convaiinnovations/laya")
    tok = base_agent.tok
    model = base_agent.model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"[FINE-TUNING] Preparing {len(dataset)} custom training samples...")
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for sample in dataset:
            optimizer.zero_grad()
            
            state = sample["state"]
            qdef = sample["question"]
            gold = sample["gold_target"]

            # Format internal sequence
            q_internal = base_agent._to_internal(qdef)
            seq, markers = build_sequence(tok, state, q_internal, max_len=512, head_max_len=192)
            
            item = {
                "ids": seq,
                "markers": markers,
                "qtype": QTYPES[q_internal["t"]]
            }
            
            batch = collate_items([[item]], tok.pad_token_id)
            
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            marker_pos = batch["marker_pos"].to(device)
            marker_mask = batch["marker_mask"].to(device)
            qtype = batch["qtype"].to(device)

            logits, act = model(input_ids, attn_mask, marker_pos, marker_mask, qtype)
            
            # Map gold target to option index
            options = list(qdef["criteria"].keys()) if qdef["type"] == "choice" else ["no", "yes"]
            gold_idx = options.index(gold) if gold in options else 0
            gold_tensor = torch.tensor([gold_idx], device=device)

            k = len(markers)
            logits_k = logits[:, :k]
            
            loss = rlcd_brier_loss(logits_k, gold_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch}/{epochs} | Avg RLCD Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoint
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "encoder"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)

    # Save configuration & weights
    with open(os.path.join(output_dir, "rl_agent_config.json"), "w") as f:
        json.dump(base_agent.cfg, f, indent=2)

    tok.save_pretrained(os.path.join(output_dir, "tokenizer"))
    model.encoder.config.save_pretrained(os.path.join(output_dir, "encoder"))

    from safetensors.torch import save_file
    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))

    print(f"\n[FINE-TUNING COMPLETE] Saved custom checkpoint to: '{output_dir}'")
    print(f"You can now load your custom model using:")
    print(f"  agent = laya.load('{output_dir}')")
    print(f"  router.attach('custom', agent)")

if __name__ == "__main__":
    train_custom_laya(MOCK_CUSTOM_DATASET)
