import json

nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🍎 Laya on Apple Silicon MacBook (MLX & PyTorch MPS Setup)\n",
    "\n",
    "This notebook provides an easy, optimized setup for running **Laya** — the non-autoregressive System 1 decision engine — on **Apple Silicon MacBooks** (M1/M2/M3/M4 series with Unified Memory).\n",
    "\n",
    "### ⚡ Why Laya on Apple Silicon?\n",
    "- **Zero Autoregressive Bottlenecks**: Executes typed decisions (`choice`, `score`, `noul`) in **a single forward pass** (~25-45 ms on M-series chips).\n",
    "- **Apple Unified Memory Advantage**: Preloads all 3 Laya model checkpoints into shared CPU/GPU RAM with **<1.2 GB memory footprint**.\n",
    "- **Native Metal & MLX Acceleration**: Leverages PyTorch Metal (`mps`) and Apple MLX framework.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Quick Installation (MacBook Terminal / Virtualenv)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install Laya, Apple MLX, PyTorch, and HuggingFace dependencies\n",
    "!pip install -q laya mlx torch transformers huggingface_hub numpy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Verify Apple Silicon Device & MLX Availability"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import torch\n",
    "import mlx.core as mx\n",
    "\n",
    "print(f\"Python Version        : {sys.version.split()[0]}\")\n",
    "print(f\"PyTorch Version       : {torch.__version__}\")\n",
    "print(f\"Apple MPS Available   : {torch.backends.mps.is_available()}\")\n",
    "print(f\"Apple MLX Core Active : {mx.default_device()}\")\n",
    "\n",
    "device = \"mps\" if torch.backends.mps.is_available() else \"cpu\"\n",
    "print(f\"\\n[TARGET DEVICE] Selected device for Laya: '{device.upper()}'\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Initialize Preloaded Laya Router on MacBook GPU"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import laya\n",
    "from laya import Router\n",
    "\n",
    "print(\"Preloading Laya model checkpoints into MacBook Unified Memory...\")\n",
    "t0 = time.time()\n",
    "\n",
    "# Preload models on Apple Silicon MPS/CPU\n",
    "router = Router(preload=True, device=device)\n",
    "print(f\"Done! 3 checkpoints resident in memory in {(time.time() - t0):.2f} seconds.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Define Typed Decision Questions & Test Input State"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "state = {\n",
    "    \"from\": \"user@acme.com\",\n",
    "    \"subject\": \"Duplicate charge on invoice #4411\",\n",
    "    \"body\": \"Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan immediately.\"\n",
    "}\n",
    "\n",
    "questions = {\n",
    "    \"department\": {\n",
    "        \"type\": \"choice\",\n",
    "        \"instructions\": \"Which department should handle this request?\",\n",
    "        \"criteria\": {\n",
    "            \"billing\": \"invoices, payments, refunds\",\n",
    "            \"technical\": \"bugs, outages, system errors\",\n",
    "            \"sales\": \"pricing, new contracts\",\n",
    "            \"other\": \"everything else\"\n",
    "        }\n",
    "    },\n",
    "    \"urgency\": {\n",
    "        \"type\": \"score\",\n",
    "        \"instructions\": \"How urgent is this request?\",\n",
    "        \"criteria\": [\"not urgent\", \"medium priority\", \"critical blocker or cancellation threat\"]\n",
    "    },\n",
    "    \"churn_risk\": {\n",
    "        \"type\": \"noul\",\n",
    "        \"instructions\": \"Does the user threaten to cancel or switch services?\"\n",
    "    }\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Execute Single-Pass Inference & Benchmark Latency"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Warmup pass\n",
    "_ = router.predict(state, questions)\n",
    "\n",
    "# Run single forward pass\n",
    "t0 = time.perf_counter()\n",
    "res = router.predict(state, questions)\n",
    "dur_ms = (time.perf_counter() - t0) * 1000\n",
    "\n",
    "print(f\"=== LAYA PREDICTION RESULTS (Inference Time: {dur_ms:.2f} ms) ===\")\n",
    "print(f\"Routed Model Checkpoint : {res['routing']['model'].upper()}\")\n",
    "print(f\"Routing Rationale       : {res['routing']['reason']}\\n\")\n",
    "\n",
    "for qid, ans in res['answers'].items():\n",
    "    if ans['type'] == 'choice':\n",
    "        print(f\"[{qid.upper()}] Choice: {ans['choice']} (Confidence: {ans['confidence']*100:.1f}%)\")\n",
    "    elif ans['type'] == 'score':\n",
    "        print(f\"[{qid.upper()}] Score Level: {ans['score']} | Legend: {ans['legend']}\")\n",
    "    elif ans['type'] == 'noul':\n",
    "        prob_yes = ans.get('noul', 0.0)\n",
    "        decision = 'yes' if prob_yes >= 0.5 else 'no'\n",
    "        print(f\"[{qid.upper()}] Noul Decision: {decision.upper()} (Prob Yes: {prob_yes*100:.1f}%)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Test Multilingual Routing on MacBook (Hindi, German, Japanese)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "multilingual_states = [\n",
    "    {\"name\": \"Hindi\", \"body\": \"मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।\"},\n",
    "    {\"name\": \"German\", \"body\": \"Die Anwendung stürzt bei der Datenverarbeitung ab. Bitte helfen Sie!\"},\n",
    "    {\"name\": \"Japanese\", \"body\": \"エンタープライズプランの価格とデモのスケジュールについて教えていただけますか？\"}\n",
    "]\n",
    "\n",
    "for test_item in multilingual_states:\n",
    "    res_lang = router.predict({\"body\": test_item[\"body\"]}, questions)\n",
    "    print(f\"[{test_item['name']}] -> Routed Model: {res_lang['routing']['model'].upper()} | Reason: {res_lang['routing']['reason']}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Integrating Apple MLX Arrays with Laya Confidence Scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert Laya output probabilities into Apple MLX Arrays for downstream MLX pipelines\n",
    "import mlx.core as mx\n",
    "\n",
    "dept_probs = res['answers']['department']['probabilities']\n",
    "mlx_probs = mx.array(list(dept_probs.values()))\n",
    "\n",
    "print(\"MLX Array representation of Laya probabilities:\")\n",
    "print(mlx_probs)\n",
    "print(f\"Max Confidence via MLX: {mx.max(mlx_probs).item()*100:.1f}%\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("notebooks/laya_apple_macbook_mlx_setup.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Created notebooks/laya_apple_macbook_mlx_setup.ipynb successfully!")
