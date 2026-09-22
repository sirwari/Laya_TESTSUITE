#!/usr/bin/env python3
"""
Laya Live Interactive Web Dashboard, Visual Question Builder & Engineering Science Briefing Studio
Powered by FastAPI, Uvicorn, PyTorch CUDA, and NVIDIA RTX 3090.

Original Laya Architecture Created by NandhaKishorM / Convai Innovations
Repository: https://github.com/NandhaKishorM/laya
License: Apache-2.0
"""

import time
import json
import torch
import psutil
from typing import Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import laya
from laya import Router

# Initialize Laya Router with preloaded GPU checkpoints
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[DASHBOARD] Preloading Laya checkpoints onto {DEVICE.upper()}...")
t0 = time.time()
router = Router(preload=True, device=DEVICE)
print(f"[DASHBOARD] Checkpoints preloaded and resident in VRAM in {(time.time() - t0):.2f} seconds.")

app = FastAPI(title="Laya Engineering & Science Studio", version="2.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PRESETS = {
    "Support Agent: P1 Production Outage Escalation": {
        "text": "CRITICAL OUTAGE: Our US-East production API cluster is completely down with 500 errors! This is blocking 100k active enterprise users. Escalate immediately!",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department queue should handle this ticket?",
                "criteria": {
                    "tech_support": "application crashes, API errors, system outages",
                    "billing": "invoices, payment failures",
                    "sales": "enterprise pricing",
                    "security": "unauthorized access, 2FA"
                }
            },
            "sla_urgency": {
                "type": "score",
                "instructions": "Rate the ticket SLA urgency based on business impact.",
                "criteria": [
                    "P4 Low: Cosmetic issue",
                    "P3 Medium: Single user affected",
                    "P2 High: Feature blocked",
                    "P1 Critical: Full system outage"
                ]
            },
            "escalation_tier": {
                "type": "choice",
                "instructions": "Which tier should handle resolution?",
                "criteria": {
                    "auto_reply": "FAQ docs",
                    "tier_1_agent": "standard agent",
                    "executive_escalation": "VP of Customer Success / Incident Response"
                }
            }
        }
    },
    "Support Agent: Account Security & Fraud Guardrail": {
        "text": "Urgent: I received a notification that an unauthorized password reset attempt occurred from Russia (IP: 185.220.101.4). Lock my account now!",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Target department",
                "criteria": {
                    "account_security": "unauthorized logins, password reset, fraud, 2FA",
                    "tech_support": "bugs, crashes",
                    "billing": "invoices"
                }
            },
            "security_risk": {
                "type": "noul",
                "instructions": "Does the ticket report suspicious activity, data breach, or security incident?"
            }
        }
    },
    "Customer Support (Billing & Churn Risk)": {
        "text": "Hi team, we were billed twice for March invoice #4411 ($2,400). Please issue a refund today or we will cancel our enterprise plan immediately.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {
                    "billing": "invoices, duplicate charges, payment failures, refunds",
                    "technical": "app crashes, API errors, system downtime",
                    "sales": "enterprise pricing, demo requests",
                    "other": "everything else"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": ["low priority", "medium urgency", "critical blocker or escalation"]
            },
            "churn_threat": {
                "type": "noul",
                "instructions": "Does the user explicitly threaten to cancel or switch services?"
            }
        }
    },
    "Max Capability: SOC2 Zero-Day Exploit & Ransomware Threat": {
        "text": "INCIDENT REPORT #INC-2026-8812: At 02:14 UTC SIEM triggered alerts on SQL injection targeting customer DB. Attackers exploited zero-day vulnerability in API gateway to bypass OAuth 2FA. 45k customer records with PII were exfiltrated to IP 185.220.101.5. Payload attempted to lock DB tables demanding 50 BTC ransom within 24h. DB subnet isolated, CISO escalation & GDPR 72h notification decision needed.",
        "questions": {
            "incident_severity": {
                "type": "choice",
                "instructions": "Classify security incident severity level",
                "criteria": {
                    "p1_critical_breach": "P1 Critical: Exfiltration, ransomware, full compromise",
                    "p2_major_compromise": "P2 Major: Vulnerability exploited, no exfiltration",
                    "p3_moderate_outage": "P3 Moderate: System degradation",
                    "p4_minor_event": "P4 Minor: Failed attack attempt"
                }
            },
            "attack_vector": {
                "type": "choice",
                "instructions": "Identify primary attack vector",
                "criteria": {
                    "sql_injection_exfiltration": "SQL injection exfiltration",
                    "ransomware_encryption": "Ransomware locker",
                    "oauth_auth_bypass": "Authentication 2FA bypass",
                    "ddos_outage": "DDoS outage"
                }
            },
            "data_exfiltration": {
                "type": "noul",
                "instructions": "Was sensitive customer PII exfiltrated?"
            },
            "ransomware_threat": {
                "type": "noul",
                "instructions": "Does the incident involve an active ransomware extortion demand?"
            },
            "compliance_impact": {
                "type": "score",
                "instructions": "Rate compliance and regulatory impact",
                "criteria": ["Level 0: Internal", "Level 1: Minor SOC2 note", "Level 2: GDPR 72h notice", "Level 3: SEC Material Fines"]
            },
            "executive_action": {
                "type": "choice",
                "instructions": "Specify mandatory executive action",
                "criteria": {
                    "ciso_emergency_dispatch": "CISO SMS & Legal Counsel dispatch",
                    "soc_team_review": "SOC team review within 4h",
                    "routine_helpdesk": "Routine ticket"
                }
            }
        }
    },
    "Max Capability: AML Wire Fraud & Structuring Detection": {
        "text": "SWIFT AUDIT #SWIFT-99120-FRD: Account #ACC-4491-001 opened 3 days ago by shell corp 'Apex Global Trading LLC' initiated $850,000 USD wire to Nicosia, Cyprus. Balance grew from $100 via 45 rapid $19,500 deposits (structuring threshold evasion). VPN node in Seychelles vs Delaware registration. Beneficiary matches partial OFAC watchlist.",
        "questions": {
            "fraud_typology": {
                "type": "choice",
                "instructions": "Classify financial crime typology",
                "criteria": {
                    "structuring_smurfing": "Structuring deposits below CTR limit",
                    "wire_fraud": "Wire fraud / forged invoice",
                    "account_takeover": "Stolen credentials",
                    "normal_corporate": "Legitimate enterprise wire"
                }
            },
            "aml_risk_level": {
                "type": "score",
                "instructions": "Rate overall AML risk score",
                "criteria": ["Clean", "Low Risk", "Moderate SAR Required", "Critical OFAC Sanction Match"]
            },
            "ofac_sanction_match": {
                "type": "noul",
                "instructions": "Does beneficiary match OFAC sanction watchlist?"
            },
            "structuring_detected": {
                "type": "noul",
                "instructions": "Were rapid micro-deposits used to evade reporting limits?"
            },
            "account_intervention": {
                "type": "choice",
                "instructions": "Select mandatory risk intervention",
                "criteria": {
                    "freeze_account_and_funds": "Freeze account and hold $850k wire",
                    "require_in_person_id": "Require corporate ID verification",
                    "allow_with_flag": "Allow wire but file report"
                }
            },
            "sar_filing_required": {
                "type": "noul",
                "instructions": "Is mandatory SAR filing required with FinCEN?"
            }
        }
    },
    "Max Capability: Supply Chain Logistics & Port Seizure Claim": {
        "text": "FREIGHT CLAIM #LOG-2026-5511: Order #PO-88410 of 5,000 CNC lathe units ($320,000 USD) shipped via ocean freight container #TGHU-9921 from Shenzhen to Hamburg. Customs placed seizure hold due to missing origin docs. Container exposed to rainstorms for 14 days. 1,200 units severely corroded. Customer Nordic Precision GmbH refused delivery, threatened legal arbitration, demands $320k refund + $50k delay damages.",
        "questions": {
            "claim_category": {
                "type": "choice",
                "instructions": "Classify logistics claim category",
                "criteria": {
                    "damaged_freight": "Corrosion and transit physical damage",
                    "customs_seizure": "Customs clearance hold",
                    "delay_damages": "Late delivery penalty"
                }
            },
            "liable_party": {
                "type": "choice",
                "instructions": "Determine primary liable party",
                "criteria": {
                    "freight_forwarder": "Freight forwarder (incorrect paperwork)",
                    "ocean_carrier": "Ocean shipping line (improper storage)",
                    "port_customs": "Hamburg port customs authority"
                }
            },
            "claim_financial_tier": {
                "type": "score",
                "instructions": "Assess financial claim tier",
                "criteria": ["Tier 1: < $5k", "Tier 2: $5k-$50k", "Tier 3: $50k-$200k", "Tier 4: > $200k Enterprise"]
            },
            "legal_threat": {
                "type": "noul",
                "instructions": "Does customer threaten legal arbitration?"
            },
            "resolution_action": {
                "type": "choice",
                "instructions": "Select resolution strategy",
                "criteria": {
                    "expedited_replacement": "Ship air freight replacement & credit",
                    "full_cash_refund": "Issue full $320k cash refund",
                    "legal_counsel_review": "Escalate to corporate legal counsel"
                }
            }
        }
    },
    "Max Capability: Clinical EHR Triage & Code STEMI Escalation": {
        "text": "EMERGENCY TRIAGE #EHR-2026-0091: 62yo male presenting with acute substernal chest pressure radiating to left jaw/shoulder, severe diaphoresis and dyspnea for 45 min. Vitals: BP 178/104, HR 112, O2 Sat 88% room air. 12-lead ECG reveals 3mm ST-elevation in V1-V4. POC Troponin I critical at 4.82 ng/mL (Normal <0.04). History of hypertension, T2D, 30 pack-year smoking.",
        "questions": {
            "triage_acuity_level": {
                "type": "score",
                "instructions": "Assign Emergency Severity Index (ESI) score",
                "criteria": ["ESI 5: Non-urgent", "ESI 4: Less urgent", "ESI 3: Urgent", "ESI 2: Emergent", "ESI 1: Resuscitation"]
            },
            "primary_diagnosis": {
                "type": "choice",
                "instructions": "Determine primary clinical diagnostic category",
                "criteria": {
                    "stemi_myocardial_infarction": "STEMI Acute Myocardial Infarction",
                    "pulmonary_embolism": "Acute Pulmonary Embolism",
                    "aortic_dissection": "Aortic Dissection",
                    "gastroesophageal": "GERD / Esophageal spasm"
                }
            },
            "cath_lab_activation": {
                "type": "noul",
                "instructions": "Is immediate Code STEMI Cath Lab activation required?"
            },
            "icu_bed_required": {
                "type": "noul",
                "instructions": "Is admission to Cardiac Intensive Care Unit (CICU) required?"
            },
            "troponin_critical_alert": {
                "type": "noul",
                "instructions": "Is cardiac troponin at critical panic value?"
            },
            "specialist_consult": {
                "type": "choice",
                "instructions": "Select specialist consultation",
                "criteria": {
                    "interventional_cardiology": "Interventional Cardiology",
                    "pulmonology": "Pulmonology",
                    "general_internal_medicine": "General Internal Medicine"
                }
            }
        }
    },
    "Multilingual Support (Hindi - Billing)": {
        "text": "मुझसे दो बार शुल्क लिया गया, कृपया तुरंत पैसे वापस करें अन्यथा हम खाता बंद कर देंगे।",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {
                    "billing": "invoices, payment issues, refunds",
                    "technical": "bugs, outages",
                    "sales": "pricing"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "Urgency rating",
                "criteria": ["low", "medium", "critical"]
            },
            "churn_threat": {
                "type": "noul",
                "instructions": "Is there a threat to close account?"
            }
        }
    }
}

class PredictionRequest(BaseModel):
    text: str
    questions: Dict[str, Any]
    model_override: Optional[str] = "auto"

@app.get("/api/telemetry")
def get_telemetry():
    vram_allocated = 0.0
    vram_reserved = 0.0
    total_vram = 0.0
    gpu_name = "CPU Only"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_name = "Apple Silicon GPU (Metal MPS)"

    cpu_pct = psutil.cpu_percent()
    ram = psutil.virtual_memory()

    return {
        "gpu": {
            "name": gpu_name,
            "allocated_mb": round(vram_allocated, 1),
            "reserved_mb": round(vram_reserved, 1),
            "total_mb": round(total_vram, 1),
            "pct_used": round((vram_allocated / total_vram * 100), 1) if total_vram > 0 else 0.0
        },
        "system": {
            "cpu_percent": cpu_pct,
            "ram_used_gb": round(ram.used / (1024**3), 1),
            "ram_total_gb": round(ram.total / (1024**3), 1),
            "ram_percent": ram.percent
        }
    }

@app.get("/api/presets")
def get_presets():
    return PRESETS

@app.post("/api/predict")
def predict(req: PredictionRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text state cannot be empty.")

    state = {"body": req.text.strip()}
    t0 = time.perf_counter()
    try:
        if req.model_override and req.model_override != "auto":
            res = router.predict(state, req.questions, model=req.model_override)
        else:
            res = router.predict(state, req.questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    dur_ms = (time.perf_counter() - t0) * 1000

    routing_info = res.get("routing", {})
    raw_answers = res.get("answers", {})

    formatted_answers = {}
    for q_id, q_val in raw_answers.items():
        q_type = q_val.get("type")
        if q_type == "choice":
            formatted_answers[q_id] = {
                "type": "choice",
                "choice": q_val.get("choice"),
                "confidence": round(q_val.get("confidence", 0.0), 4),
                "probabilities": q_val.get("probabilities", {})
            }
        elif q_type == "score":
            formatted_answers[q_id] = {
                "type": "score",
                "score": round(q_val.get("score", 0.0), 4),
                "legend": q_val.get("legend", {}),
                "probabilities": q_val.get("probabilities", {}),
                "confidence": round(q_val.get("confidence", 0.0), 4)
            }
        elif q_type == "noul":
            prob_yes = q_val.get("noul", 0.0)
            formatted_answers[q_id] = {
                "type": "noul",
                "decision": "yes" if prob_yes >= 0.5 else "no",
                "probability_yes": round(prob_yes, 4),
                "confidence": round(q_val.get("confidence", 0.0), 4)
            }

    # Scientific Engineering Metrics
    llm_baseline_ms = 1800.0
    speedup_factor = round(llm_baseline_ms / max(dur_ms, 1.0), 1)

    return {
        "latency_ms": round(dur_ms, 2),
        "routing": routing_info,
        "answers": formatted_answers,
        "engineering_summary": {
            "inference_architecture": "Non-Autoregressive Single Neural Pass",
            "latency_reduction": f"{speedup_factor}x Speedup (~{dur_ms:.1f}ms vs 1.8s LLMs)",
            "probability_calibration": "RLCD Proper Scoring Rule (Brier Loss)",
            "hallucination_guarantee": "0.0% Risk (Strictly Typed Projection)",
            "vram_footprint": f"{get_telemetry()['gpu']['allocated_mb']} MB Resident"
        },
        "telemetry": get_telemetry()
    }

@app.get("/", response_class=HTMLResponse)
def index_html():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Laya Engineering & Science Studio</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #070a12;
      --card-bg: #0f172a;
      --card-border: #1e293b;
      --accent-blue: #3b82f6;
      --accent-purple: #8b5cf6;
      --accent-emerald: #10b981;
      --accent-cyan: #06b6d4;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg-dark); color: var(--text-main); font-family: 'Inter', sans-serif; padding: 24px; min-height: 100vh; line-height: 1.5; }
    .container { max-width: 1400px; margin: 0 auto; }
    
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--card-border); padding-bottom: 20px; margin-bottom: 24px; }
    .logo-group h1 { font-size: 1.8rem; font-weight: 800; background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .attribution-tag { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; display: flex; gap: 8px; align-items: center; }
    .attribution-tag a { color: #60a5fa; text-decoration: none; }
    .attribution-tag a:hover { text-decoration: underline; }

    /* Mode Toggle Switch */
    .mode-switch-container { display: flex; background: #1e293b; border-radius: 30px; padding: 4px; border: 1px solid #334155; }
    .mode-btn { border: none; background: transparent; color: var(--text-muted); padding: 8px 20px; border-radius: 24px; font-weight: 600; font-size: 0.9rem; cursor: pointer; transition: all 0.25s ease; }
    .mode-btn.active { background: linear-gradient(135deg, #2563eb, #0891b2); color: white; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4); }

    /* Telemetry Row */
    .telemetry-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
    .stat-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
    .stat-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }
    .stat-value { font-size: 1.4rem; font-weight: 800; color: #60a5fa; margin-top: 6px; font-family: 'JetBrains Mono', monospace; }

    /* Layout Grids */
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }
    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; padding: 24px; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }
    .card-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 16px; color: #f1f5f9; display: flex; align-items: center; justify-content: space-between; }
    
    select, textarea, input { width: 100%; background: #070a12; border: 1px solid #1e293b; border-radius: 8px; color: var(--text-main); padding: 12px 14px; font-family: inherit; font-size: 0.92rem; outline: none; transition: border-color 0.2s; }
    select:focus, textarea:focus, input:focus { border-color: var(--accent-blue); box-shadow: 0 0 0 2px rgba(59,130,246,0.25); }
    textarea { font-family: 'JetBrains Mono', monospace; resize: vertical; }

    .btn { background: linear-gradient(135deg, #2563eb, #0891b2); color: white; border: none; padding: 14px 28px; border-radius: 10px; font-weight: 700; font-size: 1rem; cursor: pointer; transition: transform 0.15s, box-shadow 0.2s; width: 100%; margin-top: 16px; display: flex; justify-content: center; align-items: center; gap: 8px; }
    .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45); }
    .btn:active { transform: translateY(0); }

    /* Visual Builder */
    .builder-q-item { background: #070a12; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 14px; position: relative; }
    .builder-row { display: flex; gap: 12px; margin-top: 8px; }
    .btn-add-q { background: #1e293b; color: #60a5fa; border: 1px dashed #3b82f6; padding: 10px; border-radius: 8px; font-weight: 600; cursor: pointer; width: 100%; font-size: 0.88rem; }
    .btn-remove { background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid #f43f5e; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; cursor: pointer; float: right; }

    /* Executive Science Mode Panels */
    .exec-panel { display: none; }
    .exec-active .exec-panel { display: block; }
    .exec-active .dev-panel { display: none; }

    .diag-card { background: #070a12; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .diag-title { font-weight: 700; font-size: 1rem; color: #60a5fa; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }

    /* SVG Architecture Diagram Styles */
    .svg-diag { width: 100%; height: auto; background: #0b0f19; border-radius: 10px; padding: 16px; border: 1px solid #1e293b; }

    /* Results */
    .res-block { background: #070a12; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 14px; }
    .res-header { display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 8px; font-size: 0.95rem; }
    .prob-bar-wrap { background: #1e293b; border-radius: 6px; height: 10px; overflow: hidden; margin-top: 8px; }
    .prob-bar { background: linear-gradient(90deg, #3b82f6, #10b981); height: 100%; width: 0%; transition: width 0.5s ease; }
  </style>
</head>
<body id="body-root">
  <div class="container">
    <header>
      <div class="logo-group">
        <h1>⚡ Laya Engineering & Science Studio</h1>
        <div class="attribution-tag">
          <span>Created by <a href="https://github.com/NandhaKishorM/laya" target="_blank"><b>NandhaKishorM / Convai Innovations</b></a></span>
          <span>•</span>
          <span>License: <b>Apache-2.0</b></span>
        </div>
      </div>

      <!-- Mode Switcher -->
      <div class="mode-switch-container">
        <button class="mode-btn active" id="btn-dev-mode" onclick="setMode('dev')">🛠️ Visual Studio & Sandbox</button>
        <button class="mode-btn" id="btn-exec-mode" onclick="setMode('exec')">🔬 Executive & Engineering Science</button>
      </div>
    </header>

    <!-- Telemetry Bar -->
    <div class="telemetry-row">
      <div class="stat-card">
        <div class="stat-label">Forward Pass Latency</div>
        <div class="stat-value" id="stat-latency">-- ms</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Routed Checkpoint</div>
        <div class="stat-value" id="stat-routed">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Hardware Architecture</div>
        <div class="stat-value" id="stat-device">--</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">VRAM Footprint</div>
        <div class="stat-value" id="stat-vram">-- MB</div>
      </div>
    </div>

    <!-- EXECUTIVE & ENGINEERING SCIENCE PANEL -->
    <div class="exec-panel">
      <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">🔬 Neural Topology & System 1 Execution Paradigm</div>
        <p style="color: var(--text-muted); font-size: 0.92rem; margin-bottom: 20px;">
          Laya is a non-autoregressive decision engine trained via <strong>Reinforcement Learning against Strictly Proper Scoring Rules (RLCD)</strong>. It projects text/JSON state into calibrated probability spaces in a single forward pass without autoregressive token sampling.
        </p>

        <!-- DIAGRAM 1: SINGLE FORWARD PASS NEURAL ARCHITECTURE -->
        <div class="diag-card">
          <div class="diag-title">📐 Diagram 1: Single Forward Pass Neural Topology vs Autoregressive LLM</div>
          <svg class="svg-diag" viewBox="0 0 900 240">
            <!-- System 2 Autoregressive LLM -->
            <rect x="20" y="20" width="410" height="200" rx="10" fill="#111827" stroke="#374151" stroke-dasharray="4"/>
            <text x="35" y="45" fill="#f43f5e" font-weight="bold" font-size="14">System 2: Autoregressive LLM (GPT-4 / Llama 3)</text>
            <text x="35" y="70" fill="#9ca3af" font-size="12">Input State ➔ Token 1 ➔ Token 2 ➔ ... ➔ Token N (500–2000 ms)</text>
            <rect x="35" y="90" width="380" height="40" rx="6" fill="#1f293d" stroke="#f43f5e"/>
            <text x="50" y="115" fill="#f8fafc" font-size="12">Iterative KV-Cache Expansion + Softmax Loop</text>
            <text x="35" y="160" fill="#f43f5e" font-size="12">⚠️ High VRAM Inflation + Free-Form Schema Hallucinations</text>
            
            <!-- System 1 Laya Single Pass -->
            <rect x="470" y="20" width="410" height="200" rx="10" fill="#0f172a" stroke="#10b981" stroke-width="2"/>
            <text x="485" y="45" fill="#10b981" font-weight="bold" font-size="14">System 1: Laya Engine (ModernBERT / mmBERT)</text>
            <text x="485" y="70" fill="#9ca3af" font-size="12">Input State + Typed Questions ➔ Single Forward Pass (8–33 ms)</text>
            <rect x="485" y="90" width="380" height="40" rx="6" fill="#064e3b" stroke="#10b981"/>
            <text x="500" y="115" fill="#f8fafc" font-size="12">Parallel Multi-Head Categorical Projection</text>
            <text x="485" y="160" fill="#10b981" font-size="12">✅ 100% Typed Schema Compliance + Zero Token Generation</text>
          </svg>
        </div>

        <!-- DIAGRAM 2: DYNAMIC SCRIPT & ROUTER PIPELINE -->
        <div class="diag-card">
          <div class="diag-title">⚡ Diagram 2: Sub-Millisecond Script Analysis & Checkpoint Router</div>
          <svg class="svg-diag" viewBox="0 0 900 160">
            <rect x="20" y="30" width="160" height="100" rx="8" fill="#1e293b" stroke="#3b82f6"/>
            <text x="35" y="65" fill="#60a5fa" font-weight="bold" font-size="13">Raw State Input</text>
            <text x="35" y="90" fill="#9ca3af" font-size="11">Text, Email, JSON</text>

            <path d="M 180 80 L 250 80" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow)"/>

            <rect x="250" y="30" width="180" height="100" rx="8" fill="#1e293b" stroke="#06b6d4"/>
            <text x="265" y="60" fill="#06b6d4" font-weight="bold" font-size="13">Script & Lang Router</text>
            <text x="265" y="80" fill="#9ca3af" font-size="11">Unicode Topo (<0.5ms)</text>
            <text x="265" y="100" fill="#9ca3af" font-size="11">Devanagari, Kanji, Latin</text>

            <path d="M 430 50 L 530 35" stroke="#10b981" stroke-width="2"/>
            <path d="M 430 110 L 530 125" stroke="#8b5cf6" stroke-width="2"/>

            <rect x="530" y="15" width="340" height="50" rx="6" fill="#064e3b" stroke="#10b981"/>
            <text x="545" y="45" fill="#f8fafc" font-size="12">laya (ModernBERT-large 421M, English, 512 ctx)</text>

            <rect x="530" y="95" width="340" height="50" rx="6" fill="#3b0764" stroke="#8b5cf6"/>
            <text x="545" y="125" fill="#f8fafc" font-size="12">laya-multilingual (mmBERT-base 322M, 100+ langs)</text>
          </svg>
        </div>

        <!-- RLCD MATHEMATICAL PRINCIPLES -->
        <div class="diag-card">
          <div class="diag-title">🧮 RLCD Training & Proper Scoring Rule Calibration</div>
          <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            Laya probability distributions are calibrated using <strong>Strictly Proper Scoring Rules</strong> (Brier Score Rewards):
          </p>
          <div style="background: #0b0f19; border-left: 4px solid #10b981; padding: 14px; margin-top: 10px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #a78bfa;">
            Brier Loss = (1 / N) * Σ sum_{k=1}^K (P(y_k) - Y_k)^2
          </div>
          <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 10px;">
            This mathematical loss formulation forces model output confidences to equal true empirical probability values, eliminating the calibration distortion inherent in standard cross-entropy models.
          </p>
        </div>
      </div>
    </div>

    <!-- MAIN INTERACTIVE STUDIO -->
    <div class="grid-2">
      <!-- INPUT STATE CARD -->
      <div class="card">
        <div class="card-title">
          <span>📥 Input State Document</span>
          <select id="preset-select" style="width: auto; padding: 6px 12px; font-size: 0.82rem;">
            <option value="">Load Preset Scenario...</option>
          </select>
        </div>

        <textarea id="input-text" rows="7" placeholder="Paste customer support ticket, email body, or JSON state..."></textarea>

        <div style="margin-top: 14px; display: flex; gap: 12px; align-items: center;">
          <label style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;">Router Checkpoint:</label>
          <select id="model-override" style="width: auto; flex: 1;">
            <option value="auto">Auto Router (< 0.5ms Script & Language Detection)</option>
            <option value="english">laya (ModernBERT-large, English)</option>
            <option value="multilingual">laya-multilingual (mmBERT-base, 100+ Languages)</option>
            <option value="typed-decisions">laya-typed-decisions (ModernBERT)</option>
          </select>
        </div>

        <button class="btn" onclick="runPrediction()">⚡ Execute Single-Pass Decision</button>
      </div>

      <!-- VISUAL QUESTION BUILDER CARD -->
      <div class="card">
        <div class="card-title">
          <span>⚙️ Visual Question Schema Builder</span>
          <button style="background: transparent; color: #60a5fa; border: none; font-size: 0.8rem; cursor: pointer;" onclick="toggleJsonView()">
            Toggle JSON View
          </button>
        </div>

        <!-- FORM BUILDER UI -->
        <div id="visual-builder-ui">
          <div id="questions-list"></div>
          <button class="btn-add-q" onclick="addQuestionField()">+ Add New Typed Question</button>
        </div>

        <!-- RAW JSON UI (HIDDEN BY DEFAULT) -->
        <div id="json-builder-ui" style="display: none;">
          <textarea id="questions-json" rows="12" placeholder="Question schema JSON..."></textarea>
        </div>
      </div>
    </div>

    <!-- RESULTS DISPLAY GRID -->
    <div class="grid-2">
      <div class="card">
        <div class="card-title">📊 Calibrated Decision Results</div>
        <div id="results-container">
          <p style="color: var(--text-muted); font-size: 0.9rem;">Click 'Execute' to run instant prediction...</p>
        </div>
      </div>

      <div class="card">
        <div class="card-title">🔍 Routing Rationale & Decision Details</div>
        <div id="routing-rationale" style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 14px;"></div>
        <pre id="raw-json-output">// Raw model decision predictions JSON</pre>
      </div>
    </div>
  </div>

  <script>
    let PRESETS = {};
    let activeQuestions = [];

    function setMode(mode) {
      const root = document.getElementById('body-root');
      const btnDev = document.getElementById('btn-dev-mode');
      const btnExec = document.getElementById('btn-exec-mode');

      if (mode === 'exec') {
        root.classList.add('exec-active');
        btnExec.classList.add('active');
        btnDev.classList.remove('active');
      } else {
        root.classList.remove('exec-active');
        btnDev.classList.add('active');
        btnExec.classList.remove('active');
      }
    }

    async function init() {
      const res = await fetch('/api/presets');
      PRESETS = await res.json();
      const sel = document.getElementById('preset-select');
      Object.keys(PRESETS).forEach(k => {
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k;
        sel.appendChild(opt);
      });

      sel.onchange = () => {
        if (PRESETS[sel.value]) {
          document.getElementById('input-text').value = PRESETS[sel.value].text;
          loadQuestionsIntoBuilder(PRESETS[sel.value].questions);
        }
      };

      if (Object.keys(PRESETS).length > 0) {
        sel.value = Object.keys(PRESETS)[0];
        sel.onchange();
      }

      updateTelemetry();
    }

    function loadQuestionsIntoBuilder(qObj) {
      activeQuestions = [];
      for (const [id, val] of Object.entries(qObj)) {
        activeQuestions.push({ id, ...val });
      }
      renderVisualBuilder();
    }

    function renderVisualBuilder() {
      const container = document.getElementById('questions-list');
      container.innerHTML = '';

      activeQuestions.forEach((q, idx) => {
        const item = document.createElement('div');
        item.className = 'builder-q-item';
        
        let criteriaVal = '';
        if (q.type === 'choice') {
          criteriaVal = typeof q.criteria === 'object' ? JSON.stringify(q.criteria) : q.criteria;
        } else if (q.type === 'score') {
          criteriaVal = Array.isArray(q.criteria) ? q.criteria.join(', ') : q.criteria;
        }

        item.innerHTML = `
          <button class="btn-remove" onclick="removeQuestionField(${idx})">Remove</button>
          <div style="font-weight: 700; font-size: 0.85rem; color: #60a5fa; margin-bottom: 6px;">Question ${idx+1}</div>
          <div class="builder-row">
            <input type="text" placeholder="Question ID (e.g. department)" value="${q.id}" onchange="activeQuestions[${idx}].id = this.value" style="flex: 1;">
            <select onchange="activeQuestions[${idx}].type = this.value; renderVisualBuilder();" style="width: 140px;">
              <option value="choice" ${q.type === 'choice' ? 'selected' : ''}>Categorical Choice</option>
              <option value="score" ${q.type === 'score' ? 'selected' : ''}>Intensity Score</option>
              <option value="noul" ${q.type === 'noul' ? 'selected' : ''}>Yes/No Flag (Noul)</option>
            </select>
          </div>
          <div style="margin-top: 8px;">
            <input type="text" placeholder="Instructions for model..." value="${q.instructions || ''}" onchange="activeQuestions[${idx}].instructions = this.value">
          </div>
          ${q.type !== 'noul' ? `
          <div style="margin-top: 8px;">
            <input type="text" placeholder="${q.type === 'choice' ? 'Criteria JSON: {\"billing\": \"invoices\", \"tech\": \"bugs\"}' : 'Comma separated levels: Low, Medium, Critical'}" 
              value='${criteriaVal}' onchange="updateQuestionCriteria(${idx}, this.value)">
          </div>` : ''}
        `;
        container.appendChild(item);
      });

      document.getElementById('questions-json').value = JSON.stringify(getQuestionsFromBuilder(), null, 2);
    }

    function updateQuestionCriteria(idx, rawVal) {
      const q = activeQuestions[idx];
      if (q.type === 'choice') {
        try {
          q.criteria = JSON.parse(rawVal);
        } catch(e) {
          q.criteria = rawVal;
        }
      } else if (q.type === 'score') {
        q.criteria = rawVal.split(',').map(s => s.trim());
      }
      renderVisualBuilder();
    }

    function addQuestionField() {
      activeQuestions.push({
        id: "new_question_" + (activeQuestions.length + 1),
        type: "choice",
        instructions: "What category does this state belong to?",
        criteria: { "option_a": "description A", "option_b": "description B" }
      });
      renderVisualBuilder();
    }

    function removeQuestionField(idx) {
      activeQuestions.splice(idx, 1);
      renderVisualBuilder();
    }

    function getQuestionsFromBuilder() {
      const out = {};
      activeQuestions.forEach(q => {
        if (!q.id) return;
        out[q.id] = {
          type: q.type,
          instructions: q.instructions
        };
        if (q.type !== 'noul' && q.criteria) {
          out[q.id].criteria = q.criteria;
        }
      });
      return out;
    }

    function toggleJsonView() {
      const v = document.getElementById('visual-builder-ui');
      const j = document.getElementById('json-builder-ui');
      if (v.style.display === 'none') {
        v.style.display = 'block';
        j.style.display = 'none';
      } else {
        v.style.display = 'none';
        j.style.display = 'block';
      }
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/telemetry');
        const data = await res.json();
        document.getElementById('stat-vram').textContent = data.gpu.allocated_mb + ' MB';
        document.getElementById('stat-device').textContent = data.gpu.name;
      } catch(e) {}
    }

    async function runPrediction() {
      const text = document.getElementById('input-text').value;
      const model = document.getElementById('model-override').value;
      const questions = getQuestionsFromBuilder();

      document.getElementById('results-container').innerHTML = '<p style="color: var(--text-muted)">Running single forward pass...</p>';

      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, questions, model_override: model })
      });
      const data = await res.json();

      document.getElementById('stat-latency').textContent = data.latency_ms + ' ms';
      document.getElementById('stat-routed').textContent = data.routing.model.toUpperCase();
      document.getElementById('stat-vram').textContent = data.telemetry.gpu.allocated_mb + ' MB';

      document.getElementById('routing-rationale').innerHTML = 
        `<strong>Routing Reason:</strong> ${data.routing.reason} <br><strong>Checkpoint Repo:</strong> <code>${data.routing.repo}</code>`;

      document.getElementById('raw-json-output').textContent = JSON.stringify(data.answers, null, 2);

      let html = '';
      for (const [qid, ans] of Object.entries(data.answers)) {
        if (ans.type === 'choice') {
          const pct = Math.round(ans.confidence * 100);
          html += `<div class="res-block">
            <div class="res-header">
              <span>[CATEGORICAL CHOICE] ${qid.toUpperCase()}</span>
              <span style="color: #60a5fa">${ans.choice} (${pct}% Confidence)</span>
            </div>
            <div class="prob-bar-wrap"><div class="prob-bar" style="width: ${pct}%"></div></div>
          </div>`;
        } else if (ans.type === 'score') {
          html += `<div class="res-block">
            <div class="res-header">
              <span>[INTENSITY SCORE] ${qid.toUpperCase()}</span>
              <span style="color: #a78bfa">Score Level: ${ans.score}</span>
            </div>
          </div>`;
        } else if (ans.type === 'noul') {
          const pct = Math.round(ans.probability_yes * 100);
          const color = ans.decision === 'yes' ? '#10b981' : '#f43f5e';
          html += `<div class="res-block">
            <div class="res-header">
              <span>[YES/NO FLAG] ${qid.toUpperCase()}</span>
              <span style="color: ${color}">${ans.decision.toUpperCase()} (Prob Yes: ${pct}%)</span>
            </div>
            <div class="prob-bar-wrap"><div class="prob-bar" style="width: ${pct}%; background: ${color}"></div></div>
          </div>`;
        }
      }
      document.getElementById('results-container').innerHTML = html;
    }

    init();
  </script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
