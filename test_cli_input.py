#!/usr/bin/env python3
"""
Interactive CLI for testing Laya Non-Autoregressive System 1 Decision Engine on RTX 3090.
Allows user to enter custom input text / JSON and test typed questions live.
"""

import os
import sys
import json
import time
import torch
import laya
from laya import Router

def print_header():
    print("=" * 70)
    print("      LAYA SYSTEM 1 DECISION ENGINE - INTERACTIVE CLI TEST")
    print("=" * 70)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[HARDWARE] Running on CUDA Device: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("[HARDWARE] Running on CPU (CUDA not detected)")
    print("=" * 70)

def main():
    print_header()
    print("\n[INFO] Initializing Laya Router with preloaded checkpoints on GPU...")
    t0 = time.time()
    router = Router(preload=True, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Router ready in {(time.time() - t0):.2f} seconds.\n")

    default_questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, duplicate charges, payment failures, refunds",
                "technical": "app crashes, API errors, server downtime, bugs",
                "sales": "demo requests, pricing inquiry, enterprise contracts",
                "other": "general inquiries or feedback"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["low priority", "medium urgency", "critical outage or blocker"]
        },
        "churn_threat": {
            "type": "noul",
            "instructions": "Does the user explicitly threaten to cancel or switch to a competitor?"
        }
    }

    while True:
        print("\n" + "-" * 70)
        print("ENTER TEST INPUT STATE")
        print("Type or paste customer message / ticket text (or type 'exit' to quit):")
        print("-" * 70)
        
        user_input = input("> ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting Laya Interactive CLI. Goodbye!")
            break
        
        if not user_input:
            user_input = "Hi team, we were charged twice for March invoice #4411. Please issue a refund today or we will cancel our enterprise subscription."
            print(f"[DEFAULT INPUT USED]: {user_input}")

        state = {"body": user_input}
        
        print("\nEvaluating Laya prediction in single forward pass...")
        start_t = time.perf_counter()
        result = router.predict(state, default_questions)
        dur_ms = (time.perf_counter() - start_t) * 1000

        routing_meta = result.get("routing", {})
        answers = result.get("answers", {})

        print("\n" + "=" * 70)
        print(f" RESULTS (Forward Pass Time: {dur_ms:.2f} ms)")
        print("=" * 70)
        print(f" Routed Checkpoint : {routing_meta.get('model', 'N/A').upper()}")
        print(f" Routing Reason     : {routing_meta.get('reason', 'N/A')}")
        print("-" * 70)
        
        for q_key, val in answers.items():
            q_type = val.get("type", "choice")
            if q_type == "choice":
                choice = val.get("choice")
                conf = val.get("confidence", 0.0)
                print(f" [{q_key.upper()}] (Choice) -> {choice}  (Confidence: {conf*100:.1f}%)")
            elif q_type == "score":
                score = val.get("score")
                legend = val.get("legend", {})
                print(f" [{q_key.upper()}] (Score)  -> Level {score} | Legend: {legend}")
            elif q_type == "noul":
                prob_yes = val.get("noul", 0.0)
                decision = "yes" if prob_yes >= 0.5 else "no"
                conf = val.get("confidence", 0.0)
                print(f" [{q_key.upper()}] (Noul)   -> Decision: '{decision}' (Prob 'yes': {prob_yes*100:.1f}% | Conf: {conf*100:.1f}%)")

        print("=" * 70)

if __name__ == "__main__":
    main()
