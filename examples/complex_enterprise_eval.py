#!/usr/bin/env python3
"""
Complex Enterprise & Scientific Evaluation Suite for Laya System 1 Decision Engine.
Pushes model capabilities to maximum context and parallel typed question budgets:
  - 4 Massive Real-World Enterprise Scenarios (SOC2 Breach, AML Wire Fraud, Supply Chain Logistics, Clinical EHR Triage).
  - Up to 7 Parallel Typed Questions per scenario (choice, score, noul).
  - Evaluates latency, confidence, script routing, and decision accuracy.
"""

import time
import json
import torch
import laya
from laya import Router

# Define 4 Massive Enterprise Scenarios
COMPLEX_SCENARIOS = [
    {
        "name": "Scenario 1: SOC2 / ISO 27001 Security Breach & Zero-Day Exploit",
        "state": {
            "title": "INCIDENT REPORT #INC-2026-8812: Zero-Day Exploit & Database Compromise",
            "body": """
            At 02:14:09 UTC, our automated SIEM system triggered high-severity alerts indicating suspicious SQL injection patterns targeting the core customer billing database. Attackers exploited an unpatched zero-day vulnerability in our external API gateway to bypass OAuth 2FA authentication mechanisms.

            Post-incident forensics confirm that 45,000 customer records containing plaintext PII, hashed password digests, and credit card last-4 digits were exfiltrated to an external IP address (185.220.101.5) originating from a known bulletproof hosting provider. Furthermore, the attacker executed a malicious payload attempting to lock database tables with an encrypted ransomware header demanding 50 BTC within 24 hours.

            Current Status: Database subnet isolated, secondary API nodes disabled, SOC team currently conducting memory dump analysis. Immediate executive decision required regarding regulatory notifications (GDPR 72-hour disclosure rule) and mandatory CISO escalation.
            """
        },
        "questions": {
            "incident_severity": {
                "type": "choice",
                "instructions": "Classify the overall security incident severity.",
                "criteria": {
                    "p1_critical_breach": "P1 Critical: Active data exfiltration, ransomware threat, full system compromise",
                    "p2_major_compromise": "P2 Major: Vulnerability exploited, limited data access, no exfiltration",
                    "p3_moderate_outage": "P3 Moderate: System instability or service degradation without breach",
                    "p4_minor_event": "P4 Minor: Failed attack attempt, blocked port scan",
                    "p5_false_alarm": "P5 False Alarm: Benign security alert or internal test"
                }
            },
            "attack_vector": {
                "type": "choice",
                "instructions": "Identify the primary attack vector used by the adversary.",
                "criteria": {
                    "sql_injection_exfiltration": "SQL injection leading to database extraction",
                    "ransomware_encryption": "Ransomware locker payload",
                    "oauth_auth_bypass": "Authentication / 2FA bypass",
                    "ddos_outage": "Volumetric denial of service",
                    "insider_threat": "Malicious internal employee access"
                }
            },
            "data_exfiltration": {
                "type": "noul",
                "instructions": "Was sensitive customer PII or confidential data exfiltrated?"
            },
            "ransomware_threat": {
                "type": "noul",
                "instructions": "Does the incident involve an active ransomware extortion demand?"
            },
            "compliance_impact": {
                "type": "score",
                "instructions": "Rate regulatory and compliance impact.",
                "criteria": [
                    "Level 0: Internal event only",
                    "Level 1: Minor SOC2 audit note",
                    "Level 2: GDPR 72h mandatory authority notification",
                    "Level 3: Severe regulatory fines, SEC material disclosure required"
                ]
            },
            "executive_action": {
                "type": "choice",
                "instructions": "Specify mandatory executive escalation action.",
                "criteria": {
                    "ciso_emergency_dispatch": "Immediate CISO SMS, Legal Counsel & Incident Response dispatch",
                    "soc_team_review": "Standard SOC team review within 4 hours",
                    "helpdesk_routine": "Routine helpdesk ticket assignment"
                }
            },
            "containment_step": {
                "type": "choice",
                "instructions": "Select primary technical containment measure.",
                "criteria": {
                    "isolate_database_subnet": "Isolate database network subnet and revoke all master keys",
                    "revoke_api_tokens": "Revoke API access tokens only",
                    "patch_vulnerability": "Apply software patch without taking database offline",
                    "monitor_traffic": "Continue passive traffic monitoring"
                }
            }
        }
    },
    {
        "name": "Scenario 2: High-Stakes Financial Fraud & Anti-Money Laundering (AML)",
        "state": {
            "title": "WIRE TRANSACTION AUDIT #SWIFT-99120-FRD: High-Volume Velocity Spike",
            "body": """
            On 2026-09-22 at 14:45 EST, account #ACC-4491-001 (opened 3 days ago under shell corporation 'Apex Global Trading LLC') initiated an outgoing SWIFT international wire transfer of $850,000 USD to a beneficiary bank in Nicosia, Cyprus.

            Risk Indicators:
            1. Velocity Anomaly: Account balance increased from $100 to $900,000 via 45 rapid micro-deposits ($19,500 each, structuring threshold evasion) within a 4-hour window.
            2. IP Mismatch: Transaction executed from a residential VPN node in Seychelles while corporate registration lists Delaware, USA.
            3. Beneficiary Analysis: Beneficiary name 'BlackSea Investments Ltd' matches partial sanction watchlist entries under OFAC Executive Order 14024.
            4. Documentation: Supporting invoice provided appears to be a forged PDF with altered tax ID numbers.
            """
        },
        "questions": {
            "fraud_typology": {
                "type": "choice",
                "instructions": "Classify the financial crime fraud typology.",
                "criteria": {
                    "structuring_smurfing": "Structuring / smurfing deposits below reporting thresholds",
                    "wire_fraud": "Wire fraud / forged commercial invoice",
                    "account_takeover": "Unauthorized account takeover via stolen credentials",
                    "normal_corporate": "Legitimate enterprise commercial wire"
                }
            },
            "aml_risk_level": {
                "type": "score",
                "instructions": "Rate overall Anti-Money Laundering (AML) risk score.",
                "criteria": [
                    "Level 0: Clean transaction",
                    "Level 1: Low risk anomaly",
                    "Level 2: Moderate risk SAR required",
                    "Level 3: Critical risk high-volume AML / Sanction match"
                ]
            },
            "ofac_sanction_match": {
                "type": "noul",
                "instructions": "Does the beneficiary match OFAC or international sanction watchlists?"
            },
            "structuring_detected": {
                "type": "noul",
                "instructions": "Were rapid micro-deposits used to evade currency transaction reporting (CTR) limits?"
            },
            "account_intervention": {
                "type": "choice",
                "instructions": "Select mandatory risk intervention action.",
                "criteria": {
                    "freeze_account_and_funds": "Freeze account and hold $850,000 outbound wire immediately",
                    "require_in_person_id": "Request notarized corporate ID verification",
                    "allow_with_flag": "Allow transaction but file post-wire report",
                    "approve_normal": "Approve wire transfer normally"
                }
            },
            "sar_filing_required": {
                "type": "noul",
                "instructions": "Is a mandatory Suspicious Activity Report (SAR) filing required with FinCEN?"
            },
            "investigation_priority": {
                "type": "score",
                "instructions": "Assign fraud analyst investigation priority.",
                "criteria": ["Routine Queue", "High Priority", "Urgent Immediate Fraud Escalation"]
            }
        }
    },
    {
        "name": "Scenario 3: Global Supply Chain Logistics, Freight & Claims Triage",
        "state": {
            "title": "FREIGHT CLAIM #LOG-2026-5511: Hamburg Port Customs Seizure & Goods Damage",
            "body": """
            Order #PO-88410 consisting of 5,000 specialized precision CNC lathe units valued at $320,000 USD shipped via ocean freight container #TGHU-9921 from Shenzhen to Hamburg port.

            Dispute Details:
            Upon arrival at Hamburg container terminal, customs officials placed a seizure hold on the container due to missing origin certification documents and incorrect HS classification codes submitted by the freight forwarder. The container remained exposed on the open quay during severe rainstorms for 14 days.

            Inspection Report:
            When released, 1,200 units exhibited severe saltwater corrosion and structural rust. The customer (Tier-1 Enterprise Customer 'Nordic Precision Machining GmbH') has refused delivery, issued a formal threat of legal arbitration under CISG international sales law, and demands an immediate $320,000 refund plus $50,000 in liquidated delay damages.
            """
        },
        "questions": {
            "claim_category": {
                "type": "choice",
                "instructions": "Classify the logistics claim category.",
                "criteria": {
                    "damaged_freight": "Corrosion and physical damage during transit",
                    "customs_seizure": "Regulatory customs seizure and clearance hold",
                    "delay_damages": "Late delivery penalty claim",
                    "missing_shipment": "Lost container"
                }
            },
            "liable_party": {
                "type": "choice",
                "instructions": "Determine primary liable party.",
                "criteria": {
                    "freight_forwarder": "Freight forwarder (incorrect customs paperwork)",
                    "ocean_carrier": "Ocean shipping line (improper container storage)",
                    "port_authority": "Hamburg port customs authority",
                    "customer_fault": "Customer mismanagement"
                }
            },
            "claim_financial_tier": {
                "type": "score",
                "instructions": "Assess financial claim tier.",
                "criteria": [
                    "Tier 1: < $5,000",
                    "Tier 2: $5,000 - $50,000",
                    "Tier 3: $50,000 - $200,000",
                    "Tier 4: Enterprise High-Value > $200,000"
                ]
            },
            "legal_threat": {
                "type": "noul",
                "instructions": "Does the customer explicitly threaten legal arbitration or litigation?"
            },
            "insurance_claim_required": {
                "type": "noul",
                "instructions": "Does this event require filing a marine cargo insurance claim?"
            },
            "resolution_action": {
                "type": "choice",
                "instructions": "Select commercial resolution strategy.",
                "criteria": {
                    "expedited_replacement_and_credit": "Ship replacement batch via air freight & issue partial credit",
                    "full_cash_refund": "Issue full $320,000 refund immediately",
                    "reject_claim": "Reject claim based on carrier bill of lading terms",
                    "legal_counsel_review": "Escalate to corporate legal counsel prior to response"
                }
            }
        }
    },
    {
        "name": "Scenario 4: Emergency Healthcare EHR Patient Triage & Clinical Escalation",
        "state": {
            "title": "EMERGENCY TRIAGE INTAKE #EHR-2026-0091: Acute Chest Pain & Cardiac Markers",
            "body": """
            Patient: 62-year-old male presenting to Emergency Department at 21:05 with sudden-onset severe substernal chest pressure radiating to the left jaw and shoulder, accompanied by profuse diaphoresis, nausea, and severe shortness of breath (dyspnea) lasting for 45 minutes.

            Vitals:
            - Blood Pressure: 178/104 mmHg
            - Heart Rate: 112 bpm (Sinus Tachycardia)
            - Oxygen Saturation: 88% on room air
            - Respiratory Rate: 26 breaths/min

            Diagnostic Results:
            Initial 12-lead ECG reveals 3mm ST-segment elevation in leads V1-V4 with reciprocal ST-depression in leads II, III, and aVF. Stat point-of-care High-Sensitivity Cardiac Troponin I is significantly elevated at 4.82 ng/mL (Normal < 0.04 ng/mL). Patient has past medical history of uncontrolled hypertension, type 2 diabetes, and 30 pack-year smoking history.
            """
        },
        "questions": {
            "triage_acuity_level": {
                "type": "score",
                "instructions": "Assign Emergency Severity Index (ESI) triage acuity score.",
                "criteria": [
                    "ESI 5: Non-urgent",
                    "ESI 4: Less urgent",
                    "ESI 3: Urgent",
                    "ESI 2: Emergent (high risk, severe pain)",
                    "ESI 1: Resuscitation (immediate life-saving intervention needed)"
                ]
            },
            "primary_diagnosis": {
                "type": "choice",
                "instructions": "Determine primary clinical diagnostic category.",
                "criteria": {
                    "stemi_myocardial_infarction": "ST-Elevation Myocardial Infarction (STEMI / Acute Coronary Syndrome)",
                    "pulmonary_embolism": "Acute Pulmonary Embolism",
                    "aortic_dissection": "Aortic Dissection",
                    "gastroesophageal_reflux": "GERD / Esophageal spasm",
                    "musculoskeletal": "Chest wall musculoskeletal pain"
                }
            },
            "cath_lab_activation": {
                "type": "noul",
                "instructions": "Is immediate Code STEMI Cardiac Catheterization Lab activation required?"
            },
            "icu_bed_required": {
                "type": "noul",
                "instructions": "Will the patient require admission to the Cardiac Intensive Care Unit (CICU)?"
            },
            "troponin_critical_alert": {
                "type": "noul",
                "instructions": "Is the cardiac troponin result at a critical panic value?"
            },
            "specialist_consult": {
                "type": "choice",
                "instructions": "Select primary medical specialist consultation.",
                "criteria": {
                    "interventional_cardiology": "Interventional Cardiology (Emergency Percutaneous Coronary Intervention)",
                    "pulmonology": "Pulmonology consult",
                    "gastroenterology": "GI consult",
                    "general_medicine": "General Internal Medicine floor admission"
                }
            },
            "nursing_care_level": {
                "type": "score",
                "instructions": "Assign nursing care intensity level.",
                "criteria": ["Standard Ward", "Telemetry Unit", "Continuous 1:1 Intensive Nursing"]
            }
        }
    }
]

def run_complex_suite():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 85)
    print(f" 🔬 LAYA SYSTEM 1 MAXIMUM CAPABILITY EVALUATION SUITE")
    print(f" Target Device: {device.upper()} | Resident Model Checkpoints: 3 Preloaded")
    print("=" * 85)

    router = Router(preload=True, device=device)

    total_questions_evaluated = 0
    start_suite_time = time.perf_counter()

    for idx, scenario in enumerate(COMPLEX_SCENARIOS, 1):
        print(f"\n[{idx}/4] {scenario['name']}")
        print("-" * 85)
        
        state = scenario["state"]
        questions = scenario["questions"]
        
        # Warmup pass
        _ = router.predict(state, questions)
        
        # Benchmark single forward pass latency
        t0 = time.perf_counter()
        result = router.predict(state, questions)
        dur_ms = (time.perf_counter() - t0) * 1000

        answers = result["answers"]
        routing = result["routing"]
        num_q = len(questions)
        total_questions_evaluated += num_q

        print(f"  ⚡ Inference Latency : {dur_ms:.2f} ms for {num_q} parallel questions ({dur_ms/num_q:.2f} ms / question)")
        print(f"  🔀 Routed Checkpoint : {routing['model'].upper()} ({routing['repo']})")
        print(f"  💡 Routing Reason     : {routing['reason']}")
        print(f"  📊 TYPED DECISIONS OUTPUT:")

        for qid, ans in answers.items():
            q_type = ans["type"]
            if q_type == "choice":
                conf = ans.get("confidence", 0.0) * 100
                print(f"     • [{qid.upper()}] (Choice) -> {ans['choice']} (Confidence: {conf:.1f}%)")
            elif q_type == "score":
                conf = ans.get("confidence", 0.0) * 100
                print(f"     • [{qid.upper()}] (Score)  -> Score Level: {ans['score']} (Confidence: {conf:.1f}%) | Legend: {ans['legend']}")
            elif q_type == "noul":
                prob = ans.get("noul", 0.0) * 100
                decision = "YES" if prob >= 50 else "NO"
                print(f"     • [{qid.upper()}] (Noul)   -> Decision: {decision} (Prob Yes: {prob:.1f}%)")

    total_suite_time = (time.perf_counter() - start_suite_time) * 1000
    print("\n" + "=" * 85)
    print(f" ✅ MAXIMUM SUITE EVALUATION COMPLETE")
    print(f" Total Decisions Evaluated : {total_questions_evaluated} Parallel Typed Questions")
    print(f" Total Wall-Clock Execution: {total_suite_time:.2f} ms across 4 massive scenarios")
    print("=" * 85)

if __name__ == "__main__":
    run_complex_suite()
