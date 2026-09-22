#!/usr/bin/env python3
"""
Production Support Ticket Routing Agent powered by Laya System 1 Decision Engine.
Evaluates incoming tickets in <35 ms across 5 dimensions:
  1. Department Queue (choice)
  2. SLA Urgency Level (score)
  3. Escalation Tier (choice)
  4. Churn Risk Flag (noul)
  5. Security/Fraud Alert (noul)
"""

import time
import json
import torch
import laya
from laya import Router

class SupportRoutingAgent:
    def __init__(self, preload: bool = True, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"[ROUTING AGENT] Initializing Laya Router on '{device.upper()}'...")
        self.router = Router(preload=preload, device=device)
        
        # Define multi-dimensional ticket evaluation schema
        self.questions = {
            "department": {
                "type": "choice",
                "instructions": "Which department queue should handle this ticket?",
                "criteria": {
                    "billing": "invoices, payment failures, credit card updates, refund requests",
                    "tech_support": "application crashes, API errors, timeouts, bug reports, system outages",
                    "account_security": "compromised accounts, password resets, 2FA issues, fraud, unauthorized access",
                    "sales_growth": "enterprise plan inquiries, demo requests, seat upgrades, pricing",
                    "general": "general feedback, documentation questions, basic inquiry"
                }
            },
            "sla_urgency": {
                "type": "score",
                "instructions": "Rate the ticket SLA urgency based on business impact.",
                "criteria": [
                    "P4 Low: Minor question, cosmetic bug, non-blocking feedback",
                    "P3 Medium: Single user affected, standard inquiry",
                    "P2 High: Multiple users affected, feature blocked, cancellation risk",
                    "P1 Critical: Full system outage, severe data breach, enterprise production blocked"
                ]
            },
            "escalation_tier": {
                "type": "choice",
                "instructions": "What agent tier or automated resolution workflow should handle this?",
                "criteria": {
                    "auto_reply": "frequently asked questions with simple documentation answers",
                    "tier_1_agent": "standard customer support agent for routine requests",
                    "tier_2_specialist": "senior technical support engineer or billing specialist",
                    "executive_escalation": "VP of Customer Success or Security Incident Response Team"
                }
            },
            "churn_threat": {
                "type": "noul",
                "instructions": "Does the customer explicitly threaten to cancel their contract or switch to a competitor?"
            },
            "security_risk": {
                "type": "noul",
                "instructions": "Does the ticket report suspicious activity, data breach, or security incident?"
            }
        }

    def route_ticket(self, ticket: dict) -> dict:
        start_t = time.perf_counter()
        
        # Run single-pass forward prediction
        prediction = self.router.predict(ticket, self.questions)
        dur_ms = (time.perf_counter() - start_t) * 1000

        answers = prediction["answers"]
        routing = prediction["routing"]

        dept = answers["department"]["choice"]
        dept_conf = answers["department"]["confidence"]
        sla_score = answers["sla_urgency"]["score"]
        tier = answers["escalation_tier"]["choice"]
        churn_prob = answers["churn_threat"]["noul"]
        security_prob = answers["security_risk"]["noul"]

        # Actionable Routing Logic based on Laya typed decisions
        action = "standard_queue"
        if security_prob > 0.70:
            action = "EMERGENCY_SECURITY_LOCKDOWN"
        elif churn_prob > 0.65 or sla_score >= 2.5:
            action = "HIGH_PRIORITY_RETENTION_ESCALATION"
        elif dept_conf < 0.40:
            action = "HUMAN_TRIAGE_REVIEW"

        return {
            "ticket_id": ticket.get("id", "UNKNOWN"),
            "processing_time_ms": round(dur_ms, 2),
            "routed_checkpoint": routing["model"].upper(),
            "routing_reason": routing["reason"],
            "decision": {
                "department": dept,
                "department_confidence": f"{dept_conf*100:.1f}%",
                "sla_urgency_score": sla_score,
                "escalation_tier": tier,
                "churn_threat": "YES" if churn_prob >= 0.5 else "NO",
                "churn_probability": f"{churn_prob*100:.1f}%",
                "security_risk": "YES" if security_prob >= 0.5 else "NO",
                "security_probability": f"{security_prob*100:.1f}%",
                "recommended_action": action
            }
        }

def run_demonstration():
    agent = SupportRoutingAgent()

    test_tickets = [
        {
            "id": "TICK-101",
            "from": "cto@enterprise.com",
            "subject": "CRITICAL: API Outage on US-East production cluster",
            "body": "Our entire infrastructure is down! API requests are returning 500 Internal Server Error. This is blocking 100k users. Fix immediately!"
        },
        {
            "id": "TICK-102",
            "from": "billing@acme.org",
            "subject": "Duplicate charge on invoice #9942",
            "body": "We were billed $4,500 twice for this month's renewal. Please credit our card back today or we will cancel our plan."
        },
        {
            "id": "TICK-103",
            "from": "user@gmail.com",
            "subject": "Suspicious login attempt from unknown IP",
            "body": "I received an email saying someone logged into my account from Russia. I did not authorize this! Lock my account."
        },
        {
            "id": "TICK-104",
            "from": "dev@startup.io",
            "subject": "How to configure webhooks?",
            "body": "Hi, where can I find the API documentation for subscribing to event webhooks in Node.js?"
        }
    ]

    print("\n" + "=" * 80)
    print(" 🚀 RUNNING SUPPORT TICKET ROUTING AGENT DEMONSTRATION")
    print("=" * 80)

    for ticket in test_tickets:
        res = agent.route_ticket(ticket)
        print(f"\n[TICKET {res['ticket_id']}] {ticket['subject']}")
        print(f"  Latency     : {res['processing_time_ms']} ms (Routed to '{res['routed_checkpoint']}')")
        print(f"  Department  : {res['decision']['department']} ({res['decision']['department_confidence']})")
        print(f"  SLA Score   : Level {res['decision']['sla_urgency_score']}")
        print(f"  Tier        : {res['decision']['escalation_tier']}")
        print(f"  Churn Threat: {res['decision']['churn_threat']} ({res['decision']['churn_probability']})")
        print(f"  Security    : {res['decision']['security_risk']} ({res['decision']['security_probability']})")
        print(f"  👉 ACTION   : {res['decision']['recommended_action']}")
        print("-" * 80)

if __name__ == "__main__":
    run_demonstration()
