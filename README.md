---
title: DreamCode-RSI + Laya
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Self-improving agent exploration via replay worlds
---

# 🚀 DreamCode-RSI + Laya
### Recursive Self-Improving Agent Routing and Exploration Engine through Replay Worlds

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange.svg)](https://gradio.app)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast%20python%20packaging-de5fe9.svg)](https://github.com/astral-sh/uv)

---

## 🔬 Core Scientific Thesis
Inspired by the **Dream-RSI** paper (*Tong Zheng et al., 2026*), **DreamCode-RSI + Laya** decouples candidate generation from search-strategy improvement:
> **The system improves HOW it searches (exploration policy), not just WHAT it finds (code solutions).**

Traditional agentic search pays the full inference and evaluation cost on every online iteration. In contrast, DreamCode-RSI records discovery trees from online exploration and freezes them into **Replay Simulators**. Alternative candidate exploration policies ($\pi_0, \pi_1, \pi_2$) are then evaluated **off-policy** across these historical worlds without calling the generative model or running new code.

---

## 🏛️ Two-Level Controller Architecture

```
User Problem (e.g. Matrix Mult Optimization)
         │
         ▼
┌────────────────────────────────────────┐
│  Layer A: Laya Routing (System 1)      │
│  - Non-autoregressive typed routing    │
│  - Specialist choice + confidence gate │
│  - Fallback rule when conf < 0.70      │
└──────────────────┬─────────────────────┘
                   │ Selected Specialist (algorithm / math / gpu / debug)
                   ▼
┌────────────────────────────────────────┐
│  Layer B: Dream-RSI Meta-Exploration   │
│  - Branch opening, batching, stopping  │
│  - Online tree rollout in sandbox      │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Multi-Stage Isolated Evaluator        │
│  - AST static check & safety linting   │
│  - Assertions & unit test verification │
│  - Micro-benchmarks & normalized score │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Replay Simulator Pool                 │
│  - Frozen historical discovery worlds  │
│  - Causal frontier disclosure          │
└──────────────────┬─────────────────────┘
                   │ Replay reports
                   ▼
┌────────────────────────────────────────┐
│  Policy Improvement Agent (LLM)        │
│  - Revises Python exploration policies │
└──────────────────┬─────────────────────┘
                   │ Candidate policy
                   ▼
┌────────────────────────────────────────┐
│  Independent 5-Stage Promotion Gate    │
│  - Static check -> Interface test ->   │
│    Train replay -> Val replay -> Gate  │
│  (Promoted policy runs next online rd) │
└────────────────────────────────────────┘
```

---

## 🌟 Key Features

1. **Deterministic Replay Worlds**: Turns completed search trees into offline simulators where candidate branch-selection strategies can be benchmarked at zero marginal inference cost.
2. **Laya Decision Model**: High-speed, non-autoregressive decision layer routing tasks to `algorithm`, `math`, `gpu`, and `debug` specialists with calibrated probabilities and confidence scores.
3. **Multi-Stage Sandboxing**: Hardened code executor running under Linux `rlimit` (CPU, virtual memory, pids, file size) and Docker container isolation (`--network none`, `--cpus="2"`, `--memory="512m"`).
4. **Independent 5-Stage Promotion Gate**: Prevents self-delusion. Candidates are tested on held-out validation worlds before promotion.
5. **Stateful LangGraph Orchestration**: Explicit separation between the **Online Exploration Graph** and the **Offline Dreaming Graph**.
6. **Zero-Friction Research Dashboard**: Interactive Gradio demo displaying live routing cards, tree structures, replay comparisons, and policy evolution timelines.

---

## 📊 Replay Scoring Objective

$$S(\pi) = Q_{\text{best}} - \beta_{\text{cost}} \cdot A + \beta_{\text{parallel}} \cdot \left(\frac{A}{\max(1, R)}\right)$$

- $Q_{\text{best}}$: Highest solution quality found by the policy in replay.
- $A$: Number of branch attempts revealed.
- $R$: Number of exploration rounds taken.
- $\beta_{\text{cost}} = 0.01$: Cost penalty discouraging wasteful branching.
- $\beta_{\text{parallel}} = 0.05$: Efficiency bonus rewarding effective parallel worker batching.

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.12+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, automatic subprocess fallback with resource limits enabled)

### 2. Installation
```bash
git clone https://github.com/your-username/dreamcode-rsi.git
cd dreamcode-rsi
uv sync --extra dev
```

### 3. Launch Interactive Gradio Dashboard
```bash
uv run python -m app.gradio_app
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

### 4. Launch FastAPI Server
```bash
uv run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 5. Run Full Test Suite
```bash
uv run pytest tests/ -v
```

---

## 🐳 Hugging Face Spaces Deployment

The repository is pre-configured for Docker-based Hugging Face Spaces:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) selecting **Docker** SDK.
2. Add your Space remote and push:
```bash
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<YOUR_SPACE_NAME>
git push --force space main
```
The Space will automatically build the container and serve the Gradio dashboard at port 7860.

