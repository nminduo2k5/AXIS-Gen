# AXIS-Gen — AI-Native Test Generation for Industrial Access Control

Experimental code for the paper **"AI-Native Test Generation for Industrial Access Control: LLM-Guided XACML Request Synthesis"** (ICAI-FAI 2026, CMC University, Hanoi).

The project compares 4 methods for generating **XACML test requests** for an ABAC policy of an industrial control system (ICS), and measures their quality with **mutation testing**.

## Table of Contents

1. [Overview](#1-overview)
2. [Directory Structure](#2-directory-structure)
3. [The Policy Under Test](#3-the-policy-under-test)
4. [Mutation Operators](#4-mutation-operators)
5. [Metrics](#5-metrics)
6. [Installation](#6-installation)
7. **[Running the Experiment with Groq (standard way)](#7-running-the-experiment-with-groq-standard-way)**
8. [Running with Other Backends](#8-running-with-other-backends-gemini--openai--anthropic--surrogate)
9. [The Surrogate Backend — Scientific Integrity Note](#9-the-surrogate-backend--scientific-integrity-note)
10. [How the Pipeline Handles LLM Failures](#10-how-the-pipeline-handles-llm-failures)
11. [Main Outputs](#11-main-outputs)
12. [Security](#12-security)
13. [Troubleshooting](#13-troubleshooting)
14. [Methodological References](#14-methodological-references)

---

## 1. Overview

An access control policy can contain faults (wrong effect, wrong condition, missing rule, ...). A good test suite is one that **detects the most faults with the fewest requests**.

How it is measured:
1. From the original policy, create **51 mutants** (each mutant is the original policy plus exactly 1 injected fault).
2. Run a set of requests against the original policy and against every mutant.
3. A mutant is **killed** if at least one request yields a decision different from the original policy.
4. **Mutation score** = number of killed mutants / total number of mutants.

The four request generation methods (budget of at most 40 requests):

| Method | Description |
|---|---|
| **Random** | Uniform random sampling over the attribute value domains |
| **Pairwise (2-way)** | Generates combinations covering every pair of values (classic baseline, X-CREATE style). Stops once all pairs are covered, so it typically uses < 40 requests (~30) |
| **LLM-Guided** | Gives the policy in natural language to an LLM and asks it to generate requests that trigger each rule, flip each condition (MC/DC), try boundary values, and test interactions between rules |
| **LLM-Guided + Symbolic** (AXIS-Gen) | Exactly the LLM-Guided requests **plus** "conflicting rule pair" requests found by exhaustive grid search over the whole attribute space (2 requests for this policy) |

**Paired ablation design:** for each seed the pipeline calls the LLM **only once**; `LLM-Guided+Symbolic` **reuses exactly that result** and only appends the symbolic requests (trimming from the end to keep the budget). The difference between the two methods is therefore due to the symbolic step, not to LLM sampling noise.

Why the symbolic step exists: an LLM struggles to find requests where **two rules apply at the same time** (needed to kill mutants that change the combining algorithm — RCM). The symbolic step fills exactly this gap.

---

## 2. Directory Structure

```
.
├── main_experiment.py     # Runs 1 backend / 1 model: generate requests, score, export tables/figures/report
├── run_groq_compare.py    # Runs several Groq models in turn + comparison table (standard way with Groq)
├── generators.py          # 4 request generators + LLM clients (Groq, Gemini, OpenAI, Anthropic, Surrogate)
├── policy_model.py        # XACML/ABAC model, PDP (deny/permit-overrides, first-applicable), XML export
├── ics_policy.py          # Water-treatment plant policy (6 rules) — the object under test
├── mutation.py            # 7 mutation operators, generates 51 mutants
├── harness.py             # Runs test suites, computes mutation score, rule coverage, efficiency
├── .env                   # "Router" file: selects the backend for main_experiment.py via ENV_FILE
├── .env.groq              # Groq config          (contains API key — do NOT commit)
├── .env.gemini            # Gemini config        (contains API key — do NOT commit)
├── .env.openai            # OpenAI GPT config    (contains API key — do NOT commit)
├── .env.anthropic         # Claude config        (contains API key — do NOT commit)
├── .env.surrogate         # Runs offline, no key needed
├── IEEE_Conference_Template/   # LaTeX paper (conference_101719.tex) + figures/data used in the paper
└── results/               # Created automatically on run
    ├── surrogate/                     # Surrogate backend results
    └── groq/
        ├── model_comparison.csv       # Model comparison table (written by run_groq_compare.py)
        └── <model-name>/              # One directory per Groq model, e.g. openai_gpt-oss-120b/
            ├── data/      raw_results.json, aggregate_stats.json, experiment_results.xlsx
            ├── tables/    Tables I–IV (CSV)
            ├── figures/   fig1–fig5 (PNG)
            ├── xacml/     policy.xml + sample XACML 3.0 requests
            ├── report/    Word report (requires Node.js + docx)
            └── logs/      experiment_<timestamp>.log
```

---

## 3. The Policy Under Test

Policy `ICS-WaterTreatment-ABAC-v1`, combining algorithm **deny-overrides**, 6 rules:

| Rule | Effect | Meaning |
|---|---|---|
| R1 | PERMIT | Operator reads HMI/PLC in the control/supervisory zone |
| R2 | PERMIT | Engineer reads/writes in the field–supervisory zone, unless in emergency mode |
| R3 | DENY | Overriding the safety interlock is forbidden when `safety_state = normal` |
| R4 | PERMIT | Engineer with sufficient clearance (≥4), on site, may override the interlock during maintenance |
| R5 | DENY | Vendor is forbidden from accessing the level-0 field zone |
| R6 | PERMIT | Vendor reads the historian during production |

Attribute space: 9 attributes (role, clearance_level, location, type, zone, criticality, action, safety_state, plant_mode) → **92,160 combinations**.

A "default-deny" rule is deliberately **not** added, because under deny-overrides it would dominate every decision and mask the effect of the other mutations.

---

## 4. Mutation Operators

| Code | Name | Injected fault |
|---|---|---|
| RCM | Rule Combining Mutation | Changes the combining algorithm |
| CEM | Effect Mutation | Flips Permit ↔ Deny of a rule |
| CPM | Comparison Mutation | Inverts a comparison operator (eq→neq, ≥→<, ...) |
| CVM | Constant Value Mutation | Shifts a numeric threshold by ±1 |
| TRM | Target Removal | Removes one condition from the Target (widens the scope) |
| LOM | Logical Operator Mutation | AND → OR |
| MRD | Missing Rule Deletion | Deletes an entire rule |

> Note: the `RCM_first-applicable` mutant is an **equivalent mutant** for this policy (it gives decisions identical to the original policy on all 92,160 points), so the maximum achievable mutation score is 50/51 ≈ **98.0%**.

---

## 5. Metrics

- **Mutation score** — fraction of mutants killed (primary metric).
- **Rule coverage** — fraction of rules "touched" at least once.
- **Requests-to-20-killed** — number of requests needed to kill 20 mutants (efficiency).
- **Decision distribution** — share of PERMIT/DENY/NOT_APPLICABLE (guards against all-DENY test suites).
- **Survivor analysis** — which mutants survive at a large budget (300 requests, **using the surrogate**).

---

## 6. Installation

Requirement: Python ≥ 3.10.

```bash
pip install lxml matplotlib openpyxl numpy python-dotenv
pip install openai            # for Groq and OpenAI (Groq uses an OpenAI-compatible API)
pip install google-genai      # if using Gemini
pip install anthropic         # if using Claude
```

Optional (Word report export): install Node.js, then `npm install -g docx`. Without this step the pipeline still runs normally; it only prints `Cannot find module 'docx'` at the last step and skips the `.docx` file (other results are unaffected).

---

## 7. Running the Experiment with Groq (standard way)

### 7.1. Setup (one time)

1. Get a free API key at https://console.groq.com/keys (format `gsk_...`).
2. Open `.env.groq`, fill in the key and set the number of seeds:

   ```ini
   LLM_BACKEND=groq
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
   LLM_MODEL=openai/gpt-oss-120b
   N_SEEDS=5
   ```

   - `N_SEEDS=1` is the trial mode (fast, uses little quota); `N_SEEDS=5` is the full run that produces the numbers for the paper.
   - You do not need to edit `.env` when using `run_groq_compare.py` — the script selects `.env.groq` itself.

### 7.2. Quick trial run (recommended first)

Set `N_SEEDS=1` in `.env.groq`, set `ENV_FILE=.env.groq` in `.env`, then:

```bash
python main_experiment.py
```

Check the log for:
- `Parsed request count:` ≈ 40 (not 0);
- `domain-repair: 0/360 values invalid` (or very low);
- LLM-Guided has a higher mutation score than Random.

If everything looks fine, set `N_SEEDS=5` and continue with step 7.3.

### 7.3. Full run over several models — the standard command

```bash
python run_groq_compare.py
```

This runs the **entire experiment (5 seeds)** for each model in turn, each model in its own process:

| Default model | Note |
|---|---|
| `openai/gpt-oss-20b` | Small model |
| `openai/gpt-oss-120b` | Large model |

Before running, the script asks Groq which models the key can use and **automatically skips** models that do not exist. Each model takes about 3–5 minutes. A failing model does not stop the remaining ones.

**Run only one or a few specific models** (set the `GROQ_MODELS` variable, comma-separated):

```powershell
# PowerShell
$env:GROQ_MODELS="openai/gpt-oss-120b"; python run_groq_compare.py
```
```bat
:: CMD
set GROQ_MODELS=openai/gpt-oss-120b
python run_groq_compare.py
```
```bash
# bash
GROQ_MODELS="openai/gpt-oss-120b" python run_groq_compare.py
```

The comparison table `results/groq/model_comparison.csv` **merges every model that already has results** (including earlier runs), so you can run models one at a time and merge them gradually.

> **You don't always have to run `run_groq_compare.py`:** `python main_experiment.py` runs only **one** model (taken from `LLM_MODEL`) and writes to `results/groq/` (no per-model subdirectory, no comparison table). Use it for trial runs; use `run_groq_compare.py` for the official numbers.

### 7.4. Results

After the run, the console prints a summary table and saves:

- `results/groq/model_comparison.csv` — model comparison table: mutation score, rule coverage, number of valid / failed seeds, average request count, rate of out-of-domain LLM values;
- `results/groq/<model-name>/` — tables, figures, xlsx, XACML and detailed logs for each model (see the structure in section 2).

The two baselines Random and Pairwise **do not depend on the model**, so they have identical numbers in every model directory — this is normal.

### 7.5. Groq-specific notes

- **The free tier has tokens-per-minute and tokens-per-day limits.** If you get `429`, wait a few minutes and re-run that model.
- **`qwen/qwen3.8-27b` does not work on the free tier:** the limit is 1000 output tokens/minute, while generating 40 requests needs about 5000. The pipeline stops immediately with the message "request exceeds this model's per-minute token limit". You need to upgrade to Groq Dev Tier; afterwards add it with `GROQ_MODELS=openai/gpt-oss-120b,qwen/qwen3.8-27b`.
- The list of Groq models changes frequently. Older models (such as Llama 3.x) may return `404 model_not_found`. See the current models at https://console.groq.com/docs/models — or let `run_groq_compare.py` print the available list itself.
- For `gpt-oss` models the client sets `reasoning_effort="low"` so that the model does not spend all its tokens on hidden reasoning (otherwise the JSON output tends to be truncated).
- `temperature = 0.7`, and `seed` is passed to the API; however, cloud inference **does not guarantee bit-exact reproducibility** — two runs with the same configuration may differ by a few mutation-score percentage points. Therefore report mean ± std over 5 seeds and do not draw conclusions from small differences.

**How to report Groq numbers in the paper (important):**
- Report the **exact numbers in the result files** (mean ± std across seeds, no rounding or "about"), and state clearly that each model was run only **once** (5 seeds), so the run-to-run variation has not been measured.
- The difference between the two models (20b and 120b) and between `LLM-Guided` and `LLM-Guided+Symbolic` (+0.8 to +1.6 points) is smaller than the standard deviation, so **do not conclude** that either model is better or that the symbolic step is statistically significant; only describe it.
- The official numbers are those of the last run with the current code (paired ablation). Results of older runs were overwritten and are not usable.

---

## 8. Running with Other Backends (Gemini / OpenAI / Anthropic / Surrogate)

These backends run via `main_experiment.py`. Open the `.env` file and **leave only one `ENV_FILE` line uncommented**:

```ini
# ENV_FILE=.env.surrogate
# ENV_FILE=.env.groq
# ENV_FILE=.env.openai
ENV_FILE=.env.gemini
# ENV_FILE=.env.anthropic
```

Then run:

```bash
python main_experiment.py
```

Each `.env.<backend>` file looks like this:

```ini
LLM_BACKEND=gemini
GEMINI_API_KEY=<your key>
LLM_MODEL=gemini-3.5-flash
N_SEEDS=1        # 1 = debug mode; change to 5 for the real run
```

| Backend | Key variable | Default model | Get a key |
|---|---|---|---|
| groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | https://console.groq.com/keys |
| gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` | https://aistudio.google.com/app/apikey |
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | https://platform.openai.com/api-keys |
| anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | https://console.anthropic.com/settings/keys |
| surrogate | (not needed) | — | — |

`LLM_MODEL` overrides the default model. Anthropic additionally supports `ANTHROPIC_WORKSPACE_ID` if the key is not bound to a workspace. Results are written to `results/<backend>/`; backends do not overwrite one another.

Recommended workflow for every backend: run `surrogate` first to check the pipeline offline → run `N_SEEDS=1` with the real LLM → change to `N_SEEDS=5`.

Default configuration (in `main_experiment.py`): 5 seeds, a budget of 40 requests per method, an efficiency threshold of 20 mutants.

---

## 9. The Surrogate Backend — Scientific Integrity Note

`HeuristicSurrogateLLMClient` is **not a real LLM**. It deterministically executes the prompt strategy (trigger rules, flip each condition, try boundary values) to test the pipeline and the measurement method without an API.

- Surrogate numbers **must not be presented as results of a real LLM** in the paper — they are only a reference for the "ceiling" of the strategy.
- Survivor analysis (`analyse_survivors`) and the sample XACML requests always use the surrogate, regardless of the selected backend.

---

## 10. How the Pipeline Handles LLM Failures

| Situation | Behavior |
|---|---|
| Wrong/expired key (401), wrong model (404), blocked project (403) | **Stops immediately** with a clear message, instead of producing a misleading 0% result |
| OpenAI: out of credits (`insufficient_quota`) | **Stops immediately** ("out of credits") |
| Groq/OpenAI: `Request too large` (exceeds the model's per-minute token limit) | **Stops immediately** — retrying does not help |
| 429 from ordinary rate limiting | The SDK retries automatically (up to 3 times, following `retry-after`); Gemini retries 3 times, waiting 20–90 seconds |
| 5xx / overload | Retry with exponential backoff |
| LLM returns empty (`[]`) / non-JSON | **Retries up to 3 times** with a different sampling seed before treating it as a failure |
| Truncated output (ran out of tokens) | Warns and salvages the JSON objects that are already complete |
| `<think>...</think>` block of reasoning models | Automatically stripped before parsing the JSON |
| LLM returns too many requests | Trimmed to exactly N to keep the budget fair (too few: kept as is, noted in the log) |
| Out-of-domain values | Normalized if only the format differs (`"3"`→3); otherwise replaced with a random value and the rate is **counted** |
| Seed on which the LLM fails | Marked `failed`, **excluded from the statistics** with a warning |
| LLM fails on **2 consecutive seeds** | **Aborts the whole run** (systemic error such as exhausted quota); no all-zero report is produced |

LLM quality statistics (number of requests returned, out-of-domain value rate, number of attempts, failed seeds) are recorded in `aggregate_stats.json`, under `llm_diagnostics`.

---

## 11. Main Outputs

- **Table I** — the 6 policy rules; **Table II** — the 51 mutants
- **Table III** — aggregated results (mean ± std) for the 4 methods
- **Table IV** — per-seed × method details
- **Fig 1** mutation score · **Fig 2** rule coverage · **Fig 3** kill progress · **Fig 4** decision distribution · **Fig 5** kill heatmap by operator
- `experiment_results.xlsx` (6 sheets), `policy.xml` and sample XACML 3.0 requests, Word report
- `results/groq/model_comparison.csv` — comparison of Groq models

---

## 12. Security

- `.env*` files contain API keys and are blocked by `.gitignore`. **Do not commit them or zip them together for sending.**
- If a key has ever been exposed, regenerate it immediately on the provider's site.
- Only `.env.surrogate` (which has no key) should be committed to the repo.

---

## 13. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `404 model_not_found` (Groq) | The model was removed or the key lacks access. See `https://console.groq.com/docs/models`; `run_groq_compare.py` prints the list of available models itself |
| `429 ... Request too large ... OTPM` (Groq, e.g. Qwen) | The free tier's tokens-per-minute limit is smaller than one answer. Use another model or upgrade to Dev Tier |
| `429 rate_limit_exceeded` (Groq) | Out of tokens per minute or per day. Wait, then re-run that model |
| `401 invalid_api_key` (Groq) | Wrong key in `.env.groq` |
| `403 PERMISSION_DENIED ... project has been denied access` (Gemini) | The key's Google project is blocked. Create a key in a **new project** or use another account |
| `401 ... API key has expired` (OpenAI) | The key expired. Create a new key and check your credit |
| `429 insufficient_quota` (OpenAI) | Billing credit exhausted |
| LLM-Guided result = 0% or few requests | The LLM failed on a seed; check the log for `FAILED` / `retrying` and `llm_diagnostics` |
| Model truncated (`finish_reason=length`) | A reasoning model used up its tokens; see section 7.5 |
| `ModuleNotFoundError: lxml` (or openai, dotenv, ...) | `pip install lxml openai python-dotenv` (see section 6) |
| `Cannot find module 'docx'` | `npm install -g docx` (only affects the Word report) |
| Warning about "automatic function calling (AFC)" | Harmless, from the Gemini SDK |

---

## 14. Methodological References

- Martin & Xie (2007), *A fault model and mutation testing of access control policies*.
- Bertolino et al. (2010/2012), XACMUT and X-CREATE — test generation and mutation for XACML.
- Xu et al., the rule-pair coverage criterion.
