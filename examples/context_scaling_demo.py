#!/usr/bin/env python3
"""
Context Length Scaling & Long-Document Evaluation Demonstration for Laya.

Demonstrates how to dynamically expand Laya's token context window
from default 512/1024 tokens up to 8,192 tokens for evaluating long documents.
"""

import time
import torch
import laya
from laya import Router

def demonstrate_context_scaling():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print(" 📏 LAYA CONTEXT LENGTH SCALING & LONG-DOCUMENT EVALUATION")
    print(f" Target Device: {device.upper()}")
    print("=" * 80)

    # 1. Load single checkpoint directly
    print("\n[1/3] Loading 'convaiinnovations/laya-multilingual'...")
    agent = laya.load("convaiinnovations/laya-multilingual", device=device)

    print(f"  Default max_len      : {agent.cfg.get('max_len')} tokens")
    print(f"  Default head_max_len : {agent.cfg.get('head_max_len')} tokens")

    # 2. Dynamically override context window limits to 8,192 tokens
    print("\n[2/3] Dynamically expanding context window to 8,192 tokens...")
    agent.cfg["max_len"] = 8192
    agent.cfg["head_max_len"] = 1024

    print(f"  Updated max_len      : {agent.cfg['max_len']} tokens")
    print(f"  Updated head_max_len : {agent.cfg['head_max_len']} tokens")
    print(f"  Available Document State Budget: {agent.cfg['max_len'] - agent.cfg['head_max_len']} tokens (~5,500 words)")

    # 3. Generate a massive long-document state (~3,000 words)
    long_document = {
        "title": "ANNUAL ENTERPRISE COMPLIANCE & INCIDENT AUDIT REPORT 2026",
        "body": "Section 1: Executive Overview. " + ("The enterprise infrastructure experienced routine traffic with 99.99% uptime. " * 300) +
                "\nSection 2: Security Event. At 03:00 UTC an unauthorized PII exfiltration attempt was detected and blocked by SIEM firewall rules. " +
                ("All secondary subnet protocols were verified clean and operational. " * 200) +
                "\nSection 3: Final Audit Recommendation. Escalate to CISO for mandatory SOC2 compliance review."
    }

    questions = {
        "audit_classification": {
            "type": "choice",
            "instructions": "Classify annual audit report status.",
            "criteria": {
                "security_escalation": "Contains security event or PII breach requiring CISO review",
                "clean_audit": "Clean routine audit with 99.99% uptime",
                "financial_dispute": "Billing or accounting audit"
            }
        },
        "requires_ciso_review": {
            "type": "noul",
            "instructions": "Does the report require CISO executive review?"
        }
    }

    # 4. Predict over 8,192-token expanded context
    print("\n[3/3] Executing single forward pass prediction over 8,192-token context...")
    t0 = time.perf_counter()
    result = agent.predict(long_document, questions)
    dur_ms = (time.perf_counter() - t0) * 1000

    print(f"\n=== PREDICTION COMPLETED IN {dur_ms:.2f} ms ===")
    for qid, ans in result.items():
        if isinstance(ans, dict) and ans.get("type") == "choice":
            print(f" [{qid.upper()}] Choice: {ans['choice']} (Confidence: {ans['confidence']*100:.1f}%)")
        elif isinstance(ans, dict) and ans.get("type") == "noul":
            prob_yes = ans.get("noul", 0.0)
            decision = "YES" if prob_yes >= 0.5 else "NO"
            print(f" [{qid.upper()}] Decision: {decision} (Prob Yes: {prob_yes*100:.1f}%)")

    print("=" * 80)

if __name__ == "__main__":
    demonstrate_context_scaling()
