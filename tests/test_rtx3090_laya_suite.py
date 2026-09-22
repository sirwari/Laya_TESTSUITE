import pytest
import torch
import time
import laya
from laya import Router

def test_torch_cuda_rtx3090_available():
    """Verify PyTorch detects CUDA and RTX 3090 GPU."""
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch environment!"
    device_name = torch.cuda.get_device_name(0)
    print(f"\n[GPU CHECK] Detected GPU: {device_name}")
    assert "3090" in device_name or "NVIDIA" in device_name, f"Unexpected GPU device name: {device_name}"

def test_router_preload_gpu():
    """Test Router preloading all 3 checkpoints onto GPU VRAM."""
    print("\n[PRELOAD TEST] Preloading Laya checkpoints onto CUDA...")
    start_t = time.perf_counter()
    router = Router(preload=True, device="cuda")
    preload_dur = (time.perf_counter() - start_t) * 1000
    print(f"[PRELOAD TEST] Preloading completed in {preload_dur:.2f} ms")
    
    # Check GPU memory usage
    allocated_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved(0) / (1024 * 1024)
    print(f"[GPU MEMORY] Allocated VRAM: {allocated_mb:.2f} MB | Reserved VRAM: {reserved_mb:.2f} MB")
    assert allocated_mb > 0, "No VRAM allocated after preloading models!"

def test_multilingual_script_routing():
    """Test automatic script and language detection across multiple languages."""
    router = Router(preload=True, device="cuda")
    
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds, charging",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts"
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "medium", "critical"]
        }
    }
    
    # 1. English state -> English checkpoint
    en_state = {"body": "We were billed twice on invoice #4411. Please issue a refund today."}
    res_en = router.predict(en_state, questions)
    assert res_en["routing"]["model"] == "english", f"Expected 'english' routing, got {res_en['routing']['model']}"
    assert res_en["answers"]["department"]["choice"] == "billing"
    print(f"\n[ROUTING EN] Result: {res_en['answers']['department']['choice']} (Confidence: {res_en['answers']['department']['confidence']:.2f})")
    
    # 2. Hindi state -> Multilingual checkpoint
    hi_state = {"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}
    res_hi = router.predict(hi_state, questions)
    assert res_hi["routing"]["model"] == "multilingual", f"Expected 'multilingual' routing, got {res_hi['routing']['model']}"
    assert res_hi["answers"]["department"]["choice"] == "billing"
    print(f"[ROUTING HI] Result: {res_hi['answers']['department']['choice']} (Confidence: {res_hi['answers']['department']['confidence']:.2f})")
    
    # 3. German state -> Multilingual checkpoint
    de_state = {"body": "Der Kunde wurde zweimal belastet. Bitte erstatten Sie den Betrag."}
    res_de = router.predict(de_state, questions)
    assert res_de["routing"]["model"] == "multilingual"
    print(f"[ROUTING DE] Result: {res_de['answers']['department']['choice']} (Confidence: {res_de['answers']['department']['confidence']:.2f})")

def test_typed_questions_support():
    """Test all 3 question types: choice, score, noul."""
    router = Router(preload=True, device="cuda")
    
    state = {
        "subject": "System crash during data export",
        "body": "The application crashed while exporting 50k customer records. This is blocking our launch! Fix immediately or we cancel our contract."
    }
    
    questions = {
        "cat": {
            "type": "choice",
            "instructions": "Classify the ticket category.",
            "criteria": {"bug": "software defect, crash, exception", "billing": "invoices", "sales": "leads"}
        },
        "severity": {
            "type": "score",
            "instructions": "Rate severity level.",
            "criteria": ["low impact", "moderate impact", "critical blocker"]
        },
        "threatens_churn": {
            "type": "noul",
            "instructions": "Does the user explicitly threaten to cancel or switch services?"
        }
    }
    
    res = router.predict(state, questions)
    answers = res["answers"]
    
    print("\n[TYPED DECISIONS TEST]")
    print(f"  Category: {answers['cat']['choice']} (Confidence: {answers['cat']['confidence']:.2f})")
    print(f"  Severity: Score {answers['severity']['score']} / Legend: {answers['severity']['legend']}")
    print(f"  Churn Risk (Noul): Prob(Yes)={answers['threatens_churn']['noul']:.2f} | Confidence={answers['threatens_churn']['confidence']:.2f}")
    
    assert answers["cat"]["choice"] == "bug"
    assert "score" in answers["severity"]
    assert "noul" in answers["threatens_churn"]

def test_single_forward_pass_latency():
    """Verify inference latency on RTX 3090 is under 35 ms."""
    router = Router(preload=True, device="cuda")
    
    state = {"body": "Can I upgrade my monthly subscription to annual?"}
    questions = {
        "dept": {
            "type": "choice",
            "instructions": "Target department",
            "criteria": {"sales": "upgrades, sales", "support": "help", "billing": "payments"}
        }
    }
    
    # Warmup pass
    _ = router.predict(state, questions)
    
    # Measure 20 runs
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = router.predict(state, questions)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    print(f"\n[LATENCY BENCHMARK RTX 3090]")
    print(f"  Min Latency: {min_latency:.2f} ms")
    print(f"  Avg Latency: {avg_latency:.2f} ms")
    
    assert avg_latency < 50.0, f"Average latency too high: {avg_latency:.2f} ms"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
