#!/usr/bin/env python3
"""
Apple Silicon MacBook & MLX Test Script for Laya System 1 Decision Engine.
Runs preloaded Laya Router on Apple MPS / MLX.
"""

import sys
import time
import torch
import laya
from laya import Router

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

def main():
    print("=" * 70)
    print(" 🍎 LAYA APPLE SILICON MACBOOK & MLX TEST RUNNER")
    print("=" * 70)
    
    # 1. Device resolution
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"[DEVICE] Detected Apple Silicon GPU (Metal Performance Shaders - MPS)")
    else:
        device = "cpu"
        print(f"[DEVICE] Running on CPU (MPS not detected)")

    if HAS_MLX:
        print(f"[MLX] Apple MLX Core Active Device: {mx.default_device()}")
    else:
        print("[MLX] Apple MLX package not installed (run 'pip install mlx')")
    print("=" * 70)

    # 2. Preload models into Apple Silicon Unified Memory
    print("\n[INFO] Preloading Laya model checkpoints into Unified Memory...")
    t0 = time.time()
    router = Router(preload=True, device=device)
    print(f"[INFO] 3 Checkpoints preloaded in {(time.time() - t0):.2f} seconds.\n")

    # 3. Test State and Typed Questions
    state = {
        "body": "We were billed twice on invoice #4411 ($2,400). Please refund immediately or we will cancel our plan."
    }

    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "billing": "invoices, payment failures, refunds",
                "technical": "app crashes, bugs",
                "sales": "pricing, demos"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate urgency level",
            "criteria": ["low priority", "medium priority", "critical blocker or cancellation threat"]
        },
        "churn_threat": {
            "type": "noul",
            "instructions": "Does the user threaten to cancel or switch services?"
        }
    }

    # Warmup
    _ = router.predict(state, questions)

    # Prediction pass
    start_t = time.perf_counter()
    result = router.predict(state, questions)
    dur_ms = (time.perf_counter() - start_t) * 1000

    answers = result.get("answers", {})
    routing = result.get("routing", {})

    print(f"=== INFERENCE TIME: {dur_ms:.2f} ms (Single Forward Pass) ===")
    print(f" Checkpoint : {routing.get('model', 'N/A').upper()}")
    print(f" Reason     : {routing.get('reason', 'N/A')}\n")

    for qid, ans in answers.items():
        if ans["type"] == "choice":
            print(f" [{qid.upper()}] Choice: {ans['choice']} (Confidence: {ans['confidence']*100:.1f}%)")
        elif ans["type"] == "score":
            print(f" [{qid.upper()}] Score Level: {ans['score']} | Legend: {ans['legend']}")
        elif ans["type"] == "noul":
            prob_yes = ans.get("noul", 0.0)
            decision = "yes" if prob_yes >= 0.5 else "no"
            print(f" [{qid.upper()}] Decision: {decision.upper()} (Prob Yes: {prob_yes*100:.1f}%)")

    # MLX Array Integration Demonstration
    if HAS_MLX and "department" in answers:
        dept_probs = answers["department"]["probabilities"]
        mlx_arr = mx.array(list(dept_probs.values()))
        print("\n--- Apple MLX Array Output ---")
        print("MLX Probabilities Array :", mlx_arr)
        print(f"Max Confidence via MLX   : {mx.max(mlx_arr).item()*100:.1f}%")

    print("=" * 70)

if __name__ == "__main__":
    main()
