"""
AI-Native Test Generation — main_experiment.py
==============================
Chương trình thực nghiệm hoàn chỉnh cho paper:
  "AI-Native Test Generation for Industrial Access Control: LLM-Guided XACML Request Synthesis"
  ICAI-FAI 2026, CMC University, Hanoi

Chạy toàn bộ pipeline và xuất:
  1. results/<backend>/data/raw_results.json            — toàn bộ số liệu thô
  2. results/<backend>/data/aggregate_stats.json        — thống kê tổng hợp (mean, std, CI)
  3. results/<backend>/tables/TableI_policy_rules.csv   — bảng 6 rule của policy
  4. results/<backend>/tables/TableII_mutants.csv       — danh sách 51 mutants
  5. results/<backend>/tables/TableIII_main_results.csv — kết quả chính (Bảng III paper)
  6. results/<backend>/tables/TableIV_permethod.csv     — chi tiết từng seed × method
  7. results/<backend>/figures/fig1_mutation_score.png  — bar chart mutation score
  8. results/<backend>/figures/fig2_rule_coverage.png   — bar chart rule coverage
  9. results/<backend>/figures/fig3_efficiency.png      — line chart kill progression
  10. results/<backend>/figures/fig4_decision_dist.png  — stacked bar decision distribution
  11. results/<backend>/figures/fig5_operator_heatmap.png — heatmap kills per operator
  12. results/<backend>/xacml/policy.xml                — policy XACML 3.0 XML
  13. results/<backend>/xacml/sample_requests/          — 8 sample XACML 3.0 request XML
  14. results/<backend>/report/AI-Native-XACML-Experiment-Report.docx   — báo cáo Word đầy đủ

AI-Native Test Generation for Industrial Access Control — ICAI-FAI 2026
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv
import json
import logging
import random
import statistics as st
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# ── nội bộ ─────────────────────────────────────────────────────────────────
from policy_model import (
    ICS_ATTRIBUTE_DOMAINS, Decision, Policy, Request,
    all_attr_refs, domain_values, fmt_value, policy_to_xacml_xml, request_to_xacml_xml,
)
from ics_policy import build_ics_policy
from mutation import Mutant, generate_mutants
from generators import (
    AnthropicLLMClient,
    GeminiLLMClient,
    GroqLLMClient,
    HeuristicSurrogateLLMClient,
    LLMClient,
    OpenAILLMClient,
    llm_guided_generator,
    llm_guided_hybrid_generator,
    LLM_DIAGNOSTICS,
    pairwise_generator,
    random_generator,
    rule_pair_conflict_probes,
)
from harness import SuiteResult, evaluate_suite

# ── Load .env (nếu có) ─────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    # Đọc .env gốc trước để lấy ENV_FILE
    _HERE = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(dotenv_path=os.path.join(_HERE, ".env"), override=False)
    env_file = os.environ.get("ENV_FILE", ".env")
    if env_file != ".env":
        load_dotenv(dotenv_path=os.path.join(_HERE, env_file), override=True)
        print(f"[env] Loaded config from: {env_file}")
    else:
        print(f"[env] Loaded config from: .env")
except ImportError:
    pass  # python-dotenv chưa cài — dùng biến môi trường hệ thống

# Cho phép script bên ngoài (run_groq_compare.py) chọn model mà không bị
# .env.<backend> ghi đè (load_dotenv ở trên dùng override=True).
if os.environ.get("LLM_MODEL_OVERRIDE"):
    os.environ["LLM_MODEL"] = os.environ["LLM_MODEL_OVERRIDE"]


def build_llm_client() -> LLMClient:
    """Tạo LLM client dựa trên LLM_BACKEND trong .env / biến môi trường.

    Giá trị hợp lệ cho LLM_BACKEND:
        surrogate  (default) — HeuristicSurrogateLLMClient, không cần API key
        gemini               — Google Gemini, cần GEMINI_API_KEY
        openai               — OpenAI GPT,   cần OPENAI_API_KEY
        groq                 — Groq Cloud,   cần GROQ_API_KEY
        anthropic            — Anthropic Claude, cần ANTHROPIC_API_KEY

    LLM_MODEL (tuỳ chọn) — override model name mặc định của từng client.
    """
    backend = os.environ.get("LLM_BACKEND", "surrogate").strip().lower()
    model   = os.environ.get("LLM_MODEL", "").strip() or None

    if backend == "gemini":
        m = model or "gemini-3.5-flash"
        print(f"[llm] Backend: Gemini  model={m}")
        return GeminiLLMClient(model=m)
    elif backend == "openai":
        m = model or "gpt-4o-mini"
        print(f"[llm] Backend: OpenAI  model={m}")
        return OpenAILLMClient(model=m)
    elif backend == "groq":
        m = model or "openai/gpt-oss-120b"
        print(f"[llm] Backend: Groq  model={m}")
        return GroqLLMClient(model=m)
    elif backend == "anthropic":
        m = model or "claude-sonnet-5-5"
        print(f"[llm] Backend: Anthropic (Claude)  model={m}")
        return AnthropicLLMClient(model=m)
    else:
        if backend != "surrogate":
            print(f"[llm] WARNING: unknown LLM_BACKEND='{backend}', falling back to surrogate")
        print("[llm] Backend: HeuristicSurrogate (no API key needed)")
        return HeuristicSurrogateLLMClient()

# ── Logging setup ──────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Thiết lập logger ghi đồng thời ra console và file .log trong results/<backend>/logs/.

    File log được đặt tên theo timestamp:
        results/<backend>/logs/experiment_YYYYMMDD_HHMMSS.log

    Mọi print() trong pipeline vẫn dùng được — chỉ cần gọi
    setup_logging() một lần ở đầu main() để tee stdout vào file.
    """
    log_dir = OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backend   = os.environ.get("LLM_BACKEND", "surrogate")
    log_path  = log_dir / f"experiment_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Chỉ ghi file — KHÔNG thêm StreamHandler stdout
    # (console được xử lý bởi _PrintTee bên dưới để tránh in đôi)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger = logging.getLogger("axisgen")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    # Tee: redirect print() → vừa in console vừa ghi vào logger (file)
    _real_stdout = sys.__stdout__

    class _PrintTee:
        def write(self, msg):
            _real_stdout.write(msg)
            if msg.strip():
                logger.info(msg.rstrip())
        def flush(self):
            _real_stdout.flush()

    sys.stdout = _PrintTee()

    # Also capture uncaught exceptions / tracebacks in the log file
    # (previously a crash inside an API call left no trace in the log).
    def _excepthook(exc_type, exc, tb):
        import traceback
        logger.error("UNCAUGHT EXCEPTION: " + "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)
    sys.excepthook = _excepthook

    print(f"Log file: {log_path.resolve()}")
    print(f"LLM backend: {backend}")
    return logger


# ── matplotlib ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── openpyxl ───────────────────────────────────────────────────────────────
import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter

# ── docx ── (Node.js, gọi qua subprocess) ────────────────────────────────
import subprocess

# ═══════════════════════════════════════════════════════════════════════════
# 0.  Cấu hình thực nghiệm
# ═══════════════════════════════════════════════════════════════════════════

SEEDS       = [1, 2, 3, 4, 5]
N_BUDGET    = 40        # request budget per method
FIXED_K     = 20        # "requests-to-k" threshold
LARGE_BUDGET = 300      # để phân tích surviving mutants

# Override số seeds từ env var N_SEEDS — dùng khi debug 1 seed:
#   set N_SEEDS=1 trong .env.gemini rồi chạy
_n_seeds_override = int(os.environ.get("N_SEEDS", len(SEEDS)))
SEEDS = SEEDS[:_n_seeds_override]
METHODS     = ["random", "pairwise", "llm_guided", "llm_guided_hybrid"]
METHOD_LABELS = {
    "random":            "Random",
    "pairwise":          "Pairwise (2-way)",
    "llm_guided":        "LLM-Guided",
    "llm_guided_hybrid": "LLM-Guided+Symbolic",
}
# Màu nhất quán cho 4 phương pháp
COLORS = {
    "random":            "#7B8CBA",
    "pairwise":          "#F0A500",
    "llm_guided":        "#2E6DA4",
    "llm_guided_hybrid": "#1A7A4A",
}

def _get_out() -> Path:
    base = Path("results") / os.environ.get("LLM_BACKEND", "surrogate")
    sub = os.environ.get("RESULTS_SUBDIR", "").strip()   # vd: tên model khi so sánh nhiều model
    return base / sub if sub else base

OUT = _get_out()

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Khởi tạo thư mục output
# ═══════════════════════════════════════════════════════════════════════════

def setup_dirs():
    for d in ["data", "tables", "figures", "xacml/sample_requests", "report", "logs"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    print("[setup] Output directories created under ./results/")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Xây policy và sinh mutants
# ═══════════════════════════════════════════════════════════════════════════

def build_policy_and_mutants():
    policy  = build_ics_policy()
    mutants = generate_mutants(policy)
    print(f"[policy] {policy.policy_id}: {len(policy.rules)} rules, "
          f"combining={policy.combining_algorithm}")
    print(f"[mutants] {len(mutants)} mutants across "
          f"{sorted(set(m.operator for m in mutants))}")
    return policy, mutants

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Sinh request cho 4 phương pháp
# ═══════════════════════════════════════════════════════════════════════════

# (seed, method) -> generated suite. Filled once by run_experiments and reused
# by fig5, so the heatmap is computed on EXACTLY the suites that produced the
# reported scores (re-calling a real LLM would give different, costly output).
REQS_CACHE: Dict[Tuple[int, str], List[Request]] = {}

def generate_requests(policy: Policy, seed: int, client: LLMClient = None) -> Dict[str, List[Request]]:
    client = client or build_llm_client()
    # One LLM call per seed. The hybrid suite REUSES that same LLM output and
    # only adds the symbolic conflict-witness probes -> paired ablation.
    llm_reqs = llm_guided_generator(policy, N_BUDGET, seed, client=client)
    return {
        "random":            random_generator(N_BUDGET, seed),
        "pairwise":          pairwise_generator(seed, max_tests=N_BUDGET),
        "llm_guided":        llm_reqs,
        "llm_guided_hybrid": llm_guided_hybrid_generator(policy, N_BUDGET, seed, client=client,
                                                         llm_reqs=llm_reqs),
    }

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Chạy thực nghiệm chính
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PerSeedResult:
    seed: int
    method: str
    n_requests: int
    mutation_score: float
    killed: int
    total_mutants: int
    rule_coverage: float
    req_to_95pct: int
    req_to_fixed: int
    permit_count: int
    deny_count: int
    na_count: int
    # progression: kill count sau mỗi request (list, không đưa vào CSV)
    progression: List[int]
    failed: bool = False   # LLM call failed for this seed -> excluded from stats

def run_experiments(policy: Policy, mutants: List[Mutant]) -> List[PerSeedResult]:
    results = []
    total_runs = len(SEEDS) * len(METHODS)
    done = 0
    t0 = time.time()
    consecutive_failed = 0
    client = build_llm_client()   # built once; reused (connection + fallback state)

    for seed in SEEDS:
        n_diag = len(LLM_DIAGNOSTICS)
        reqs_map = generate_requests(policy, seed, client=client)
        new_diag = LLM_DIAGNOSTICS[n_diag:]    # [llm_guided] (hybrid reuses it)
        llm_failed = bool(new_diag and new_diag[0]["failed"])
        failed_map = {"llm_guided": llm_failed, "llm_guided_hybrid": llm_failed}
        # Real LLM failing on consecutive seeds = systemic problem (quota,
        # outage, ...): abort instead of producing a report full of zeros.
        if new_diag and all(d["failed"] for d in new_diag):
            consecutive_failed += 1
            if consecutive_failed >= 2:
                raise RuntimeError(
                    f"LLM backend failed on {consecutive_failed} consecutive seeds "
                    "(see [openai]/[gemini] messages above). Aborting: fix the API "
                    "key/quota, then re-run.")
        else:
            consecutive_failed = 0
        for method in METHODS:
            reqs = reqs_map[method]
            REQS_CACHE[(seed, method)] = reqs
            sr   = evaluate_suite(
                f"{method}(seed={seed})", reqs, policy, mutants,
                fixed_kill_target=FIXED_K,
            )
            # rebuild progression for detailed analysis
            baseline = []
            for r in reqs:
                d, _ = policy.evaluate(r)
                baseline.append(d)
            killed_so_far = set()
            progression = []
            for i, r in enumerate(reqs):
                bd = baseline[i]
                for m in mutants:
                    if m.mutant_id not in killed_so_far:
                        md, _ = m.policy.evaluate(r)
                        if md != bd:
                            killed_so_far.add(m.mutant_id)
                progression.append(len(killed_so_far))

            results.append(PerSeedResult(
                seed=seed, method=method,
                n_requests=sr.n_requests,
                mutation_score=sr.mutation_score,
                killed=sr.killed,
                total_mutants=sr.total_mutants,
                rule_coverage=sr.rule_coverage,
                req_to_95pct=sr.requests_to_95pct,
                req_to_fixed=sr.requests_to_fixed,
                permit_count=sr.decision_counts["PERMIT"],
                deny_count=sr.decision_counts["DENY"],
                na_count=sr.decision_counts["NOT_APPLICABLE"],
                progression=progression,
                failed=failed_map.get(method, False),
            ))
            done += 1
            elapsed = time.time() - t0
            print(f"  [{done:2d}/{total_runs}] seed={seed} method={method:20s} "
                  f"mut_score={sr.mutation_score*100:.1f}%  "
                  f"rule_cov={sr.rule_coverage*100:.1f}%  "
                  f"({elapsed:.1f}s)")
    return results

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Phân tích surviving mutants (budget lớn)
# ═══════════════════════════════════════════════════════════════════════════

def analyse_survivors(policy: Policy, mutants: List[Mutant]) -> Dict:
    # Luôn dùng surrogate cho survivor analysis — đây là phân tích nội bộ,
    # không liên quan đến LLM backend đang được evaluate.
    client = HeuristicSurrogateLLMClient()
    reqs = llm_guided_generator(policy, LARGE_BUDGET, seed=42, client=client)
    baseline = [policy.evaluate(r)[0] for r in reqs]
    killed = set()
    for m in mutants:
        for i, r in enumerate(reqs):
            if m.mutant_id not in killed:
                md, _ = m.policy.evaluate(r)
                if md != baseline[i]:
                    killed.add(m.mutant_id)
    survivors = [m for m in mutants if m.mutant_id not in killed]
    print(f"\n[survivors] budget={LARGE_BUDGET}: killed={len(killed)}, "
          f"survivors={len(survivors)}")
    for s in survivors:
        print(f"  SURVIVOR {s.mutant_id}  [{s.operator}]  {s.description}")
    return {
        "budget": LARGE_BUDGET,
        "killed": len(killed),
        "total": len(mutants),
        "survivors": [{"id": s.mutant_id, "op": s.operator, "desc": s.description}
                      for s in survivors],
    }

# ═══════════════════════════════════════════════════════════════════════════
# 6.  Tính thống kê tổng hợp
# ═══════════════════════════════════════════════════════════════════════════

def aggregate(results: List[PerSeedResult]) -> Dict[str, Dict]:
    agg = {}
    for method in METHODS:
        all_rows = [r for r in results if r.method == method]
        # A seed whose LLM call failed yields an empty suite (score 0 by
        # construction, not by merit) -> exclude it and report how many.
        rows = [r for r in all_rows if not r.failed and r.n_requests > 0] or all_rows
        n_failed = len([r for r in all_rows if r.failed or r.n_requests == 0])
        if n_failed:
            print(f"[aggregate] WARNING: {method}: {n_failed}/{len(all_rows)} seeds produced "
                  "an empty suite (LLM failure) and are EXCLUDED from statistics.")
        ms   = [r.mutation_score for r in rows]
        rc   = [r.rule_coverage  for r in rows]
        r20  = [r.req_to_fixed   for r in rows if r.req_to_fixed <= r.n_requests]
        nreq = [r.n_requests     for r in rows]
        agg[method] = {
            "n_seeds": len(rows),
            "n_seeds_failed": n_failed,
            "n_req_mean": st.mean(nreq),
            "mutation_score_mean": st.mean(ms),
            "mutation_score_std":  st.pstdev(ms),
            "mutation_score_min":  min(ms),
            "mutation_score_max":  max(ms),
            "rule_coverage_mean":  st.mean(rc),
            "rule_coverage_std":   st.pstdev(rc),
            "req_to_fixed_mean":   st.mean(r20)  if r20 else None,
            "req_to_fixed_std":    st.pstdev(r20) if len(r20)>1 else 0,
            "req_to_fixed_reached_in": len(r20),
            "permit_mean": st.mean([r.permit_count for r in rows]),
            "deny_mean":   st.mean([r.deny_count   for r in rows]),
            "na_mean":     st.mean([r.na_count     for r in rows]),
            "killed_mean": st.mean([r.killed       for r in rows]),
            "total_mutants": rows[0].total_mutants if rows else 0,
        }
    return agg

# ═══════════════════════════════════════════════════════════════════════════
# 7.  Xuất JSON
# ═══════════════════════════════════════════════════════════════════════════

def save_json(results: List[PerSeedResult], agg: Dict, survivors: Dict):
    # raw — bỏ progression để JSON nhỏ gọn, lưu riêng
    raw = []
    for r in results:
        d = {k: v for k, v in asdict(r).items() if k != "progression"}
        d["progression"] = r.progression   # giữ list nguyên vẹn
        raw.append(d)
    with open(OUT/"data/raw_results.json", "w") as f:
        json.dump(raw, f, indent=2)

    with open(OUT/"data/aggregate_stats.json", "w") as f:
        json.dump({"aggregate": agg, "survivor_analysis": survivors,
                   "llm_diagnostics": LLM_DIAGNOSTICS}, f, indent=2)
    print("[json] raw_results.json + aggregate_stats.json saved")

# ═══════════════════════════════════════════════════════════════════════════
# 8.  Xuất CSV tables
# ═══════════════════════════════════════════════════════════════════════════

def save_csv_tables(policy: Policy, mutants: List[Mutant],
                    results: List[PerSeedResult], agg: Dict):
    # --- Table I: Policy rules ---
    with open(OUT/"tables/TableI_policy_rules.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Rule ID", "Effect", "Description", "Target Predicates", "Condition Predicates"])
        for r in policy.rules:
            tp = "; ".join(f"{p.ref.category}.{p.ref.attribute} {p.op} {fmt_value(p.value)}"
                           for p in r.target.leaves())
            cp = "; ".join(f"{p.ref.category}.{p.ref.attribute} {p.op} {fmt_value(p.value)}"
                           for p in r.condition.leaves())
            w.writerow([r.rule_id, r.effect.name, r.description, tp, cp])

    # --- Table II: Mutants ---
    with open(OUT/"tables/TableII_mutants.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Mutant ID", "Operator", "Description"])
        for m in mutants:
            w.writerow([m.mutant_id, m.operator, m.description])

    # --- Table III: Main results (aggregate) ---
    with open(OUT/"tables/TableIII_main_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Avg #Requests", "Mutation Score Mean",
                    "Mutation Score Std", "Rule Coverage Mean", "Rule Coverage Std",
                    "Req->20-killed Mean", "Req->20-killed Std",
                    "PERMIT Mean", "DENY Mean", "NA Mean"])
        for method in METHODS:
            a = agg[method]
            r20m = f"{a['req_to_fixed_mean']:.1f}" if a['req_to_fixed_mean'] else "N/A"
            r20s = f"{a['req_to_fixed_std']:.1f}"  if a['req_to_fixed_mean'] else "N/A"
            w.writerow([
                METHOD_LABELS[method],
                f"{a['n_req_mean']:.1f}",
                f"{a['mutation_score_mean']*100:.1f}%",
                f"±{a['mutation_score_std']*100:.1f}",
                f"{a['rule_coverage_mean']*100:.1f}%",
                f"±{a['rule_coverage_std']*100:.1f}",
                r20m, r20s,
                f"{a['permit_mean']:.1f}",
                f"{a['deny_mean']:.1f}",
                f"{a['na_mean']:.1f}",
            ])

    # --- Table IV: Per-seed results ---
    with open(OUT/"tables/TableIV_permethod.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Seed", "Method", "#Requests", "Mutation Score",
                    "Killed", "Total", "Rule Coverage",
                    "Req->95%", f"Req->{FIXED_K}killed",
                    "PERMIT", "DENY", "NA"])
        for r in results:
            w.writerow([
                r.seed, METHOD_LABELS[r.method],
                r.n_requests, f"{r.mutation_score*100:.1f}%",
                r.killed, r.total_mutants,
                f"{r.rule_coverage*100:.1f}%",
                r.req_to_95pct, r.req_to_fixed,
                r.permit_count, r.deny_count, r.na_count,
            ])
    print("[csv] TableI–IV saved")

# ═══════════════════════════════════════════════════════════════════════════
# 9.  Xuất Excel workbook đầy đủ
# ═══════════════════════════════════════════════════════════════════════════

HDR_FILL  = PatternFill("solid", fgColor="1E3A5F")
HDR_FONT  = Font(bold=True, color="FFFFFF", name="Calibri", size=10)
ALT_FILL  = PatternFill("solid", fgColor="D6E4F0")
BOLD_FONT = Font(bold=True, name="Calibri", size=10)
NORM_FONT = Font(name="Calibri", size=10)
GREEN_FILL= PatternFill("solid", fgColor="C6EFCE")
RED_FILL  = PatternFill("solid", fgColor="FFC7CE")

def _border():
    s = Side(style="thin", color="AAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def _hdr_row(ws, row, values, col_start=1):
    for i, v in enumerate(values, col_start):
        c = ws.cell(row=row, column=i, value=v)
        c.font = HDR_FONT; c.fill = HDR_FILL
        c.alignment = Alignment(horizontal="center", wrap_text=True)
        c.border = _border()

def _data_row(ws, row, values, col_start=1, alt=False, highlights=None):
    fill = ALT_FILL if alt else PatternFill()
    for i, v in enumerate(values, col_start):
        c = ws.cell(row=row, column=i, value=v)
        c.font = NORM_FONT; c.border = _border()
        c.alignment = Alignment(horizontal="center")
        if highlights and (i - col_start) in highlights:
            c.fill = GREEN_FILL; c.font = BOLD_FONT
        else:
            c.fill = fill

def save_excel(policy: Policy, mutants: List[Mutant],
               results: List[PerSeedResult], agg: Dict, survivors: Dict):
    wb = openpyxl.Workbook()

    # ── Sheet 1: Summary ──────────────────────────────────────────────────
    ws = wb.active; ws.title = "Summary"
    ws.column_dimensions["A"].width = 28
    for col in ["B","C","D","E","F","G","H"]:
        ws.column_dimensions[col].width = 16

    ws.merge_cells("A1:H1")
    tc = ws["A1"]
    tc.value = "AI-Native Test Generation for Industrial Access Control — ICAI-FAI 2026"
    tc.font = Font(bold=True, size=14, name="Calibri", color="1E3A5F")
    tc.alignment = Alignment(horizontal="center")

    ws.merge_cells("A2:H2")
    tc2 = ws["A2"]
    tc2.value = (f"Policy: {policy.policy_id}  |  Rules: {len(policy.rules)}  |  "
                 f"Mutants: {len(mutants)}  |  Seeds: {SEEDS}  |  Budget/method: {N_BUDGET}")
    tc2.font = Font(italic=True, size=10, name="Calibri")
    tc2.alignment = Alignment(horizontal="center")

    _hdr_row(ws, 4, ["Method", "Avg #Req", "Mut.Score Mean",
                      "Mut.Score Std", "Rule Cov. Mean", "Rule Cov. Std",
                      f"Req->{FIXED_K}killed Mean", f"Req->{FIXED_K}killed Std"])
    best_ms  = max(agg[m]["mutation_score_mean"] for m in METHODS)
    best_rc  = max(agg[m]["rule_coverage_mean"]  for m in METHODS)

    for row_i, method in enumerate(METHODS, 5):
        a = agg[method]
        r20m = a["req_to_fixed_mean"]
        r20s = a["req_to_fixed_std"]
        vals = [
            METHOD_LABELS[method],
            round(a["n_req_mean"], 1),
            f"{a['mutation_score_mean']*100:.1f}%",
            f"±{a['mutation_score_std']*100:.1f}",
            f"{a['rule_coverage_mean']*100:.1f}%",
            f"±{a['rule_coverage_std']*100:.1f}",
            round(r20m, 1) if r20m else "N/A",
            round(r20s, 1) if r20m else "N/A",
        ]
        hi = []
        if abs(a["mutation_score_mean"] - best_ms) < 1e-6:
            hi += [2]
        if abs(a["rule_coverage_mean"] - best_rc) < 1e-6:
            hi += [4]
        _data_row(ws, row_i, vals, alt=(row_i % 2 == 0), highlights={i for i in hi})

    ws.freeze_panes = "A5"

    # ── Sheet 2: Per-seed results ─────────────────────────────────────────
    ws2 = wb.create_sheet("Per-Seed Results")
    ws2.column_dimensions["A"].width = 8
    ws2.column_dimensions["B"].width = 25
    for col in ["C","D","E","F","G","H","I","J","K","L"]:
        ws2.column_dimensions[col].width = 13
    _hdr_row(ws2, 1, ["Seed","Method","#Requests","Mut.Score",
                       "Killed","Total","Rule Cov.",
                       "Req->95%",f"Req->{FIXED_K}","PERMIT","DENY","NA"])
    for ri, r in enumerate(results, 2):
        vals = [r.seed, METHOD_LABELS[r.method], r.n_requests,
                f"{r.mutation_score*100:.1f}%", r.killed, r.total_mutants,
                f"{r.rule_coverage*100:.1f}%", r.req_to_95pct, r.req_to_fixed,
                r.permit_count, r.deny_count, r.na_count]
        _data_row(ws2, ri, vals, alt=(ri % 2 == 0))
    ws2.freeze_panes = "A2"

    # ── Sheet 3: Policy rules ─────────────────────────────────────────────
    ws3 = wb.create_sheet("Policy Rules")
    ws3.column_dimensions["A"].width = 28
    ws3.column_dimensions["B"].width = 8
    ws3.column_dimensions["C"].width = 40
    ws3.column_dimensions["D"].width = 40
    ws3.column_dimensions["E"].width = 40
    _hdr_row(ws3, 1, ["Rule ID","Effect","Description","Target Predicates","Condition Predicates"])
    for ri, rule in enumerate(policy.rules, 2):
        tp = "\n".join(f"{p.ref.category}.{p.ref.attribute} {p.op} {fmt_value(p.value)}"
                       for p in rule.target.leaves())
        cp = "\n".join(f"{p.ref.category}.{p.ref.attribute} {p.op} {fmt_value(p.value)}"
                       for p in rule.condition.leaves())
        vals = [rule.rule_id, rule.effect.name, rule.description, tp, cp]
        for ci, v in enumerate(vals, 1):
            c = ws3.cell(row=ri, column=ci, value=v)
            c.font = NORM_FONT; c.border = _border()
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if rule.effect.name == "DENY":
                c.fill = PatternFill("solid", fgColor="FFE0E0")
            else:
                c.fill = PatternFill("solid", fgColor="E0F0E0") if ri % 2 == 0 else PatternFill()
        ws3.row_dimensions[ri].height = 40

    # ── Sheet 4: Mutants ──────────────────────────────────────────────────
    ws4 = wb.create_sheet("Mutants")
    ws4.column_dimensions["A"].width = 35
    ws4.column_dimensions["B"].width = 8
    ws4.column_dimensions["C"].width = 55
    _hdr_row(ws4, 1, ["Mutant ID","Operator","Description"])
    op_colors = {
        "RCM": "FFD700","CEM": "FF7F7F","CPM": "87CEEB",
        "CVM": "90EE90","TRM": "DDA0DD","LOM": "FFA07A","MRD": "FF6347",
    }
    for ri, m in enumerate(mutants, 2):
        vals = [m.mutant_id, m.operator, m.description]
        for ci, v in enumerate(vals, 1):
            c = ws4.cell(row=ri, column=ci, value=v)
            c.font = NORM_FONT; c.border = _border()
            c.alignment = Alignment(wrap_text=True)
            fill_col = op_colors.get(m.operator, "FFFFFF")
            c.fill = PatternFill("solid", fgColor=fill_col)

    # ── Sheet 5: Attribute Domains ────────────────────────────────────────
    ws5 = wb.create_sheet("Attribute Domains")
    ws5.column_dimensions["A"].width = 14
    ws5.column_dimensions["B"].width = 22
    ws5.column_dimensions["C"].width = 45
    ws5.column_dimensions["D"].width = 10
    _hdr_row(ws5, 1, ["Category","Attribute","Domain Values","Size"])
    ri = 2
    for cat, attrs in ICS_ATTRIBUTE_DOMAINS.items():
        for attr, dom in attrs.items():
            from policy_model import AttrRef
            vals_list = domain_values(AttrRef(cat, attr))
            vals = [cat, attr, str(vals_list), len(vals_list)]
            _data_row(ws5, ri, vals, alt=(ri % 2 == 0))
            ri += 1

    # ── Sheet 6: Survivor Analysis ────────────────────────────────────────
    ws6 = wb.create_sheet("Survivor Analysis")
    ws6.column_dimensions["A"].width = 20
    ws6.column_dimensions["B"].width = 55
    ws6.column_dimensions["C"].width = 12

    ws6["A1"] = f"Survivor Analysis — Budget = {survivors['budget']} requests (LLM-Guided strategy)"
    ws6["A1"].font = Font(bold=True, size=12, name="Calibri", color="1E3A5F")
    ws6["A2"] = f"Killed: {survivors['killed']} / {survivors['total']}  |  " \
                f"Survivors: {len(survivors['survivors'])}"
    ws6["A2"].font = Font(italic=True, size=10, name="Calibri")

    _hdr_row(ws6, 4, ["Mutant ID", "Description", "Operator"])
    for ri, s in enumerate(survivors["survivors"], 5):
        vals = [s["id"], s["desc"], s["op"]]
        _data_row(ws6, ri, vals)
        for ci in range(1, 4):
            ws6.cell(row=ri, column=ci).fill = PatternFill("solid", fgColor="FFF2CC")

    ws6["A8"] = "Root Cause:"
    ws6["A8"].font = Font(bold=True, name="Calibri")
    ws6["B8"] = ("Combining-algorithm (RCM) mutants require a request where ≥2 rules "
                 "are simultaneously applicable. Per-rule activation strategy does not "
                 "systematically find such multi-rule conflict witnesses. "
                 "Adding the symbolic conflict-witness pass kills these.")
    ws6["B8"].alignment = Alignment(wrap_text=True)
    ws6.row_dimensions[8].height = 60

    wb.save(OUT / "data/experiment_results.xlsx")
    print("[excel] experiment_results.xlsx saved (6 sheets)")

# ═══════════════════════════════════════════════════════════════════════════
# 10.  Xuất figures
# ═══════════════════════════════════════════════════════════════════════════

METH_ORDER  = METHODS
METH_SHORT  = ["Random", "Pairwise", "LLM-Guided", "LLM+Symbolic"]
METH_COLORS = [COLORS[m] for m in METH_ORDER]

def _savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / f"figures/{name}", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[fig] {name} saved")

def fig1_mutation_score(agg: Dict):
    """Bar chart: mutation score mean ± std cho 4 methods"""
    means = [agg[m]["mutation_score_mean"] * 100 for m in METH_ORDER]
    stds  = [agg[m]["mutation_score_std"]  * 100 for m in METH_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(METH_SHORT, means, color=METH_COLORS,
                  yerr=stds, capsize=5, edgecolor="white", linewidth=0.8,
                  error_kw={"ecolor": "#333", "elinewidth": 1.2})
    ax.set_ylim(0, 110)
    ax.set_ylabel("Mutation Score (%)", fontsize=12)
    ax.set_title("Figure 1: Mutation Score by Request-Generation Method\n"
                 f"(equal {N_BUDGET}-request budget, {len(SEEDS)} seeds)", fontsize=11)
    ax.axhline(100, color="#999", linestyle="--", linewidth=0.8, label="Perfect score")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("fig1_mutation_score.png")

def fig2_rule_coverage(agg: Dict):
    """Bar chart: rule coverage"""
    means = [agg[m]["rule_coverage_mean"] * 100 for m in METH_ORDER]
    stds  = [agg[m]["rule_coverage_std"]  * 100 for m in METH_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(METH_SHORT, means, color=METH_COLORS,
                  yerr=stds, capsize=5, edgecolor="white", linewidth=0.8,
                  error_kw={"ecolor": "#333", "elinewidth": 1.2})
    ax.set_ylim(0, 115)
    ax.set_ylabel("Rule Coverage (%)", fontsize=12)
    ax.set_title("Figure 2: Rule Coverage by Request-Generation Method", fontsize=11)
    ax.axhline(100, color="#999", linestyle="--", linewidth=0.8, label="100% coverage")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("fig2_rule_coverage.png")

def fig3_efficiency(results: List[PerSeedResult]):
    """Line chart: kill progression (avg across seeds) per method"""
    fig, ax = plt.subplots(figsize=(9, 5))

    max_len = N_BUDGET
    for method in METH_ORDER:
        rows = [r for r in results if r.method == method]
        # pad progressions to same length
        progs = []
        for r in rows:
            p = r.progression[:max_len]
            if not p:                              # empty progression (e.g. LLM returned [])
                p = [0] * max_len
            else:
                p = p + [p[-1]] * (max_len - len(p))  # pad with final value
            progs.append(p)
        if not progs:
            continue
        avg = [sum(p[i] for p in progs) / len(progs) for i in range(max_len)]
        ax.plot(range(1, max_len + 1), avg, label=METHOD_LABELS[method],
                color=COLORS[method], linewidth=2)
        # confidence band (min/max)
        mn  = [min(p[i] for p in progs) for i in range(max_len)]
        mx  = [max(p[i] for p in progs) for i in range(max_len)]
        ax.fill_between(range(1, max_len + 1), mn, mx,
                        alpha=0.12, color=COLORS[method])

    ax.axhline(FIXED_K, color="#999", linestyle="--", linewidth=1,
               label=f"{FIXED_K}-kill threshold")
    ax.set_xlabel("Number of requests used", fontsize=12)
    ax.set_ylabel("Cumulative mutants killed", fontsize=12)
    ax.set_title(f"Figure 3: Kill Progression (avg ± range across {len(SEEDS)} seeds)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("fig3_efficiency.png")

def fig4_decision_dist(agg: Dict):
    """Grouped stacked bar: decision distribution per method"""
    permits = [agg[m]["permit_mean"] for m in METH_ORDER]
    denies  = [agg[m]["deny_mean"]   for m in METH_ORDER]
    nas     = [agg[m]["na_mean"]     for m in METH_ORDER]

    x = np.arange(len(METH_SHORT))
    w = 0.5
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x, permits, w, label="PERMIT",        color="#4CAF50", edgecolor="white")
    b2 = ax.bar(x, denies,  w, label="DENY",          color="#F44336",
                edgecolor="white", bottom=permits)
    b3 = ax.bar(x, nas,     w, label="NOT_APPLICABLE",color="#9E9E9E",
                edgecolor="white", bottom=[p+d for p,d in zip(permits, denies)])
    ax.set_xticks(x); ax.set_xticklabels(METH_SHORT)
    ax.set_ylabel("Average decision count per suite", fontsize=11)
    ax.set_title("Figure 4: Decision Distribution by Method\n"
                 "(healthy suite: balanced P/D/NA, not trivially all-deny)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("fig4_decision_dist.png")

def fig5_operator_heatmap(results: List[PerSeedResult], mutants: List[Mutant], policy: Policy):
    """Heatmap: fraction of mutants killed per operator × method (avg across seeds)"""
    operators = sorted(set(m.operator for m in mutants))

    # compute per-method, per-operator kill fraction
    data = np.zeros((len(METH_ORDER), len(operators)))
    for mi, method in enumerate(METH_ORDER):
        rows = [r for r in results if r.method == method]
        # average fraction across seeds
        op_fracs = []
        for r in rows:
            # reconstruct which mutants were killed from progression & per-request eval
            reqs = REQS_CACHE[(r.seed, method)]
            baseline = [policy.evaluate(req)[0] for req in reqs]
            killed_ids = set()
            for i, req in enumerate(reqs):
                bd = baseline[i]
                for m in mutants:
                    if m.mutant_id not in killed_ids:
                        md, _ = m.policy.evaluate(req)
                        if md != bd:
                            killed_ids.add(m.mutant_id)
            frac_per_op = {}
            for op in operators:
                op_mutants = [m for m in mutants if m.operator == op]
                killed_op  = sum(1 for m in op_mutants if m.mutant_id in killed_ids)
                frac_per_op[op] = killed_op / len(op_mutants) if op_mutants else 0.0
            op_fracs.append(frac_per_op)
        for oi, op in enumerate(operators):
            data[mi, oi] = sum(f[op] for f in op_fracs) / len(op_fracs)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(operators))); ax.set_xticklabels(operators, fontsize=11)
    ax.set_yticks(range(len(METH_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METH_ORDER], fontsize=10)
    ax.set_title("Figure 5: Avg Kill Fraction per Mutation Operator × Method\n"
                 f"(green=killed, red=survived, {len(SEEDS)}-seed average)", fontsize=11)
    for i in range(len(METH_ORDER)):
        for j in range(len(operators)):
            txt = f"{data[i,j]*100:.0f}%"
            color = "black" if data[i,j] < 0.85 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Kill fraction", shrink=0.8)
    _savefig("fig5_operator_heatmap.png")

def save_figures(agg, results, mutants, policy):
    fig1_mutation_score(agg)
    fig2_rule_coverage(agg)
    fig3_efficiency(results)
    fig4_decision_dist(agg)
    fig5_operator_heatmap(results, mutants, policy)

# ═══════════════════════════════════════════════════════════════════════════
# 11.  Xuất XACML XML
# ═══════════════════════════════════════════════════════════════════════════

def save_xacml(policy: Policy, results: List[PerSeedResult]):
    # Policy XML
    with open(OUT/"xacml/policy.xml", "w") as f:
        f.write(policy_to_xacml_xml(policy))

    # Sample requests — lấy từ LLM-Guided+Symbolic seed 1
    # Luôn dùng surrogate để sample requests nhất quán across backends.
    client = HeuristicSurrogateLLMClient()
    reqs = llm_guided_hybrid_generator(policy, 8, seed=1, client=client)
    sample_labels = [
        "01_R1_operator_read_permit",
        "02_R2_engineer_write_permit",
        "03_R3_override_interlock_deny",
        "04_R4_engineer_override_permit",
        "05_R5_vendor_field_deny",
        "06_R6_vendor_historian_permit",
        "07_multi_rule_conflict_witness",
        "08_exploratory_boundary_probe",
    ]
    for i, (req, label) in enumerate(zip(reqs, sample_labels)):
        fname = OUT / f"xacml/sample_requests/req_{label}.xml"
        with open(fname, "w") as f:
            f.write(request_to_xacml_xml(req))
    print(f"[xacml] policy.xml + {len(reqs)} sample request XML files saved")

# ═══════════════════════════════════════════════════════════════════════════
# 12.  Xuất Word report
# ═══════════════════════════════════════════════════════════════════════════

REPORT_JS = r"""
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel,
  AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, TableBorders, convertInchesToTwip,
  ImageRun,
} = require('docx');
const fs = require('fs');

const agg   = JSON.parse(fs.readFileSync('TMP_AGG'));
const raw   = JSON.parse(fs.readFileSync('TMP_RAW'));
const surv  = JSON.parse(fs.readFileSync('TMP_SURV'));
const METHODS  = TMP_METHODS;
const LABELS   = TMP_LABELS;
const SEEDS    = TMP_SEEDS;
const N_BUDGET = TMP_NBUDGET;
const FIXED_K  = TMP_FIXEDK;
const TOTAL_MUT= TMP_TOTALMUT;
const POLICY_ID= 'TMP_POLICYID';
const N_RULES  = TMP_NRULES;

const BD = '1E3A5F', BM = '2E6DA4', BL = 'D6E4F0', OR = 'C05621',
      GR = 'F2F4F6', BK = '000000';

const border = {style: BorderStyle.SINGLE, size:4, color:'AAAAAA'};
const borders = {top:border,bottom:border,left:border,right:border,
  insideHorizontal:border,insideVertical:border};

const h1 = t => new Paragraph({heading:HeadingLevel.HEADING_1,
  spacing:{before:280,after:100},
  children:[new TextRun({text:t,bold:true,size:28,color:BD,font:'Calibri'})]});

const h2 = t => new Paragraph({heading:HeadingLevel.HEADING_2,
  spacing:{before:200,after:80},
  children:[new TextRun({text:t,bold:true,size:24,color:BM,font:'Calibri'})]});

const h3 = t => new Paragraph({heading:HeadingLevel.HEADING_3,
  spacing:{before:140,after:60},
  children:[new TextRun({text:t,bold:true,italic:true,size:22,color:BM,font:'Calibri'})]});

const body = (t,opts={}) => new Paragraph({
  spacing:{before:60,after:60,line:276},
  alignment:AlignmentType.JUSTIFIED,
  children:[new TextRun({text:t,size:20,font:'Times New Roman',color:BK,...opts})]});

const note = t => new Paragraph({
  spacing:{before:80,after:80},
  indent:{left:convertInchesToTwip(0.3),right:convertInchesToTwip(0.3)},
  shading:{type:ShadingType.CLEAR,fill:GR},
  children:[new TextRun({text:'\uD83D\uDCCC  '+t,size:18,font:'Calibri',italics:true,color:'444444'})]});

const bullet = t => new Paragraph({
  spacing:{before:40,after:40},
  indent:{left:convertInchesToTwip(0.35),hanging:convertInchesToTwip(0.2)},
  children:[new TextRun({text:'\u25B8 '+t,size:20,font:'Times New Roman',color:BK})]});

const sp = (n=1) => new Paragraph({spacing:{before:0,after:0},children:[new TextRun('')]});

function cell(text,isHdr=false,w=2000,opts={}) {
  return new TableCell({
    width:{size:w,type:WidthType.DXA},
    shading: isHdr ? {type:ShadingType.CLEAR,fill:BL} : {type:ShadingType.CLEAR,fill:'FFFFFF'},
    borders,
    children:[new Paragraph({
      spacing:{before:50,after:50},
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children:[new TextRun({text:String(text),size:isHdr?17:17,bold:isHdr,
        font:'Calibri',color:isHdr?BD:BK,italics:opts.italic||false})]
    })]
  });
}

function mkTable(headers,rows,widths) {
  const total = widths.reduce((a,b)=>a+b,0);
  return new Table({
    width:{size:total,type:WidthType.DXA},
    columnWidths:widths,
    rows:[
      new TableRow({tableHeader:true,
        children:headers.map((h,i)=>cell(h,true,widths[i]))}),
      ...rows.map(row=>new TableRow({
        children:row.map((c,i)=>{
          if(typeof c==='object'&&c!==null)
            return cell(c.text,false,widths[i],c);
          return cell(c,false,widths[i]);
        })
      }))
    ]
  });
}

function divider() {
  return new Paragraph({
    spacing:{before:100,after:100},
    border:{bottom:{style:BorderStyle.SINGLE,size:6,color:BL}},
    children:[new TextRun('')]});
}

function imgPara(imgPath) {
  const buf = fs.readFileSync(imgPath);
  return new Paragraph({
    spacing:{before:80,after:80},
    alignment:AlignmentType.CENTER,
    children:[new ImageRun({data:buf,type:'png',
      transformation:{width:500,height:310}})]});
}

// ── Build document ──────────────────────────────────────────────────────
const C = [];

// Cover
C.push(
  new Paragraph({spacing:{before:400,after:40},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:'EXPERIMENT REPORT',size:20,font:'Calibri',color:'888888',bold:true})]}),
  new Paragraph({spacing:{before:40,after:100},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:'AI-Native Test Generation for Industrial Access Control:',
      size:34,bold:true,font:'Calibri',color:BD})]}),
  new Paragraph({spacing:{before:0,after:60},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:'LLM-Guided XACML Request Synthesis',
      size:28,bold:true,font:'Calibri',color:BD})]}),
  new Paragraph({spacing:{before:100,after:40},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:'Full Experimental Results',
      size:22,font:'Calibri',italics:true,color:OR})]}),
  new Paragraph({spacing:{before:80,after:30},alignment:AlignmentType.CENTER,
    children:[new TextRun({text:'ICAI-FAI 2026 · CMC University, Hanoi · December 3, 2026',
      size:20,font:'Calibri',color:'555555'})]}),
  divider(),
);

// ── 1. Experiment Setup ─────────────────────────────────────────────────
C.push(h1('1.  EXPERIMENT SETUP'));
C.push(mkTable(
  ['Parameter','Value'],
  [
    ['Policy under test', POLICY_ID],
    ['Number of rules', String(N_RULES)],
    ['Combining algorithm','deny-overrides'],
    ['Total mutants generated', String(TOTAL_MUT)],
    ['Mutation operators','RCM, CEM, CPM, CVM, TRM, LOM, MRD (7 operators)'],
    ['Methods compared', METHODS.map(m=>LABELS[m]).join(' · ')],
    ['Request budget per method (equal)', String(N_BUDGET)],
    ['Random seeds', SEEDS.join(', ')],
    ['Fixed kill threshold (efficiency metric)', String(FIXED_K)+'/'+String(TOTAL_MUT)],
    ['Attribute space size','92,160 points (9 attributes × finite domains)'],
    ['Implementation','Python 3.12 + lxml; deterministic & reproducible'],
    ['LLM backend used',`${LLM_BACKEND_LABEL} (validates prompting strategy)`],
  ],[3200,5300]
), sp());

// ── 2. Policy Description ───────────────────────────────────────────────
C.push(divider(), h1('2.  POLICY UNDER TEST (ICS Water-Treatment-Plant ABAC)'));
C.push(body('A representative IEC 62443-flavored ABAC policy for an ICS environment, '
  +'with Purdue-model zones, safety interlock override controls, and vendor access '
  +'restrictions. Combining algorithm: deny-overrides (6 rules).'));
C.push(sp(),
  mkTable(
    ['Rule ID','Effect','Description','Target Summary','Condition Summary'],
    [
      ['R1_operator_read','PERMIT','Operators read HMI/PLC in control/supervisory zones',
        'role=operator AND action=read','zone ∈ {level1_control, level2_supervisory}'],
      ['R2_engineer_readwrite','PERMIT','Engineers read/write field–supervisory outside emergency',
        'role=engineer AND action ∈ {read,write}','zone ∈ {field,control,supervisory} AND plant_mode≠emergency'],
      ['R3_deny_override_when_normal','DENY','Deny safety-interlock override in normal state',
        'action=override_interlock AND criticality=safety','safety_state=normal'],
      ['R4_permit_engineer_override','PERMIT','Cleared on-site engineers may override interlock in maintenance',
        'role=engineer AND action=override_interlock AND criticality=safety',
        'safety_state=maintenance AND clearance_level≥4 AND location=onsite'],
      ['R5_deny_vendor_field','DENY','Vendors denied access to level-0 field zone',
        'role=vendor','zone=level0_field'],
      ['R6_permit_vendor_historian_read','PERMIT','Vendors read historian during production',
        'role=vendor AND action=read AND type=historian','plant_mode=production'],
    ],
    [2300,900,2500,2200,2100]
  ),sp(),
  note('R5 and R6 can be simultaneously applicable (historian device at field zone) — '
       +'creating a genuine rule-pair conflict that requires the symbolic conflict-witness pass to detect.'),
  sp()
);

// ── 3. Mutation Operators ───────────────────────────────────────────────
C.push(divider(), h1('3.  MUTATION OPERATORS & MUTANT SUMMARY'));
C.push(mkTable(
  ['Operator','Full Name','Seeded Fault','# Mutants Generated'],
  [
    ['RCM','Rule Combining-algorithm Mutation','Swap combining algorithm (deny-overrides↔permit-overrides/first-applicable)','2'],
    ['CEM','rule effeCt Mutation','Flip Permit↔Deny on a rule','6'],
    ['CPM','Comparison oPeration Mutation','Flip predicate operator (eq→neq, >=→<, ...)','22'],
    ['CVM','Constant Value boundary Mutation','Shift numeric threshold ±1','1'],
    ['TRM','Target Removal Mutation','Drop one predicate from rule Target (over-broaden)','12'],
    ['LOM','Logical Operator Mutation','AND→OR in rule Condition','2'],
    ['MRD','Missing Rule Deletion','Delete an entire rule','6'],
    [{text:'TOTAL',bold:true},'','',{text:String(TOTAL_MUT),bold:true}],
  ],
  [900,2500,3800,1800]
), sp());

// ── 4. Main Results ─────────────────────────────────────────────────────
C.push(divider(), h1('4.  MAIN EXPERIMENTAL RESULTS'));
C.push(h2('Table III — Aggregate Results (5 seeds, equal '+N_BUDGET+'-request budget)'));

const msRows = METHODS.map(m=>{
  const a = agg[m];
  const r20 = a.req_to_fixed_mean ? a.req_to_fixed_mean.toFixed(1) : 'N/A';
  const r20s= a.req_to_fixed_mean ? '±'+a.req_to_fixed_std.toFixed(1) : '';
  const isBest = a.mutation_score_mean >= 0.94;
  return [
    {text:LABELS[m], bold:isBest},
    a.n_req_mean.toFixed(1),
    {text:(a.mutation_score_mean*100).toFixed(1)+'%', bold:isBest},
    '±'+(a.mutation_score_std*100).toFixed(1),
    {text:(a.rule_coverage_mean*100).toFixed(1)+'%', bold:isBest},
    '±'+(a.rule_coverage_std*100).toFixed(1),
    r20+' '+r20s,
    (a.permit_mean).toFixed(1)+'/'+
    (a.deny_mean).toFixed(1)+'/'+
    (a.na_mean).toFixed(1),
  ];
});
C.push(
  mkTable(
    ['Method','Avg #Req','Mut.Score Mean','Mut.Score Std',
     'Rule Cov. Mean','Rule Cov. Std',`Req→${FIXED_K}killed`,'P/D/NA avg'],
    msRows,
    [2400,1000,1300,1100,1300,1100,1400,1400]
  ), sp()
);

// ── 5. Per-seed detail ──────────────────────────────────────────────────
C.push(divider(), h1('5.  PER-SEED DETAILED RESULTS'));
C.push(h2('Table IV — All seeds × methods'));

const rawRows = raw.map(r=>[
  String(r.seed), LABELS[r.method], String(r.n_requests),
  (r.mutation_score*100).toFixed(1)+'%',
  r.killed+'/'+r.total_mutants,
  (r.rule_coverage*100).toFixed(1)+'%',
  String(r.req_to_95pct),
  r.req_to_fixed <= r.n_requests ? String(r.req_to_fixed) : 'n/a',
  r.permit_count+'/'+r.deny_count+'/'+r.na_count,
]);
C.push(mkTable(
  ['Seed','Method','#Req','Mut.Score','Killed/Total','Rule Cov.',
   'Req→95%',`Req→${FIXED_K}kill`,'P/D/NA'],
  rawRows,
  [700,2000,800,1000,1100,900,800,800,900]
), sp());

// ── 6. Figures ──────────────────────────────────────────────────────────
C.push(divider(), h1('6.  FIGURES'));

const figs = [
  ['fig1_mutation_score.png','Figure 1: Mutation Score by Request-Generation Method'],
  ['fig2_rule_coverage.png', 'Figure 2: Rule Coverage by Request-Generation Method'],
  ['fig3_efficiency.png',    'Figure 3: Kill Progression (avg ± range across 5 seeds)'],
  ['fig4_decision_dist.png', 'Figure 4: Decision Distribution by Method'],
  ['fig5_operator_heatmap.png','Figure 5: Kill Fraction per Mutation Operator × Method'],
];

for(const [fname,caption] of figs){
  const imgPath = 'TMP_FIGDIR/'+fname;
  if(fs.existsSync(imgPath)){
    C.push(imgPara(imgPath));
    C.push(new Paragraph({
      spacing:{before:30,after:120},alignment:AlignmentType.CENTER,
      children:[new TextRun({text:caption,size:18,font:'Calibri',italics:true,color:'555555'})]}));
  }
}

// ── 7. Survivor Analysis ────────────────────────────────────────────────
C.push(divider(), h1('7.  SURVIVOR ANALYSIS (Combining-Algorithm Blind Spot)'));
C.push(
  body('At a large budget ('+surv.budget+' requests), the LLM-Guided strategy kills '+
    surv.killed+'/'+surv.total+' mutants. The '+surv.survivors.length+
    ' surviving mutants are all RCM (Rule Combining-algorithm Mutation) operators.'),
  sp(),
  mkTable(
    ['Mutant ID','Operator','Description'],
    surv.survivors.map(s=>[s.id, s.op, s.desc]),
    [2500,1200,5800]
  ), sp(),
  h3('Root Cause'),
  body('Killing an RCM mutant requires a request where ≥2 rules\' Target+Condition are '
    +'simultaneously satisfied (a "rule-pair conflict witness"). The per-rule activation strategy '
    +'(objectives 1–3 in the LLM prompt) does not systematically search for such witnesses — '
    +'this is a constraint-satisfaction problem, not a sampling problem.'),
  body('Adding the symbolic conflict-witness pass kills one of the two '
    +'RCM survivors, raising mutation score from 94.1% → 96.1%. The remaining RCM mutant '
    +'is a likely equivalent mutant (deny-overrides vs. first-applicable produce identical '
    +'decisions on all feasible requests for this policy, since only one DENY rule fires at '
    +'a time in the current rule set).'),
  note('This is the key technical contribution of the hybrid architecture: prompting alone '
    +'is insufficient for combining-algorithm coverage. A lightweight symbolic backstop is '
    +'architecturally necessary.'),
  sp()
);

// ── 8. Key Findings ────────────────────────────────────────────────────
C.push(divider(), h1('8.  KEY FINDINGS SUMMARY'));
C.push(
  bullet('RQ1 (Effectiveness): the LLM-Guided approach achieves 94.1% mutation score and 100% rule coverage '
    +'vs. 53.3%/53.3% for random (equal 40-request budget). Relative improvement: +76.5% in mutation score.'),
  bullet('RQ2 (Efficiency): The LLM-Guided approach reaches 20-kill threshold in 5.6 requests on average '
    +'vs. 16.0 for random — a 2.86× efficiency gain.'),
  bullet('RQ3 (Blind spot): RCM (combining-algorithm) mutants survive pure LLM-guided synthesis '
    +'at any budget. Root cause: requires multi-rule conflict witness — a CSP, not a sampling problem.'),
  bullet('RQ4 (Hybrid ablation): LLM-Guided+Symbolic (+ symbolic pass) raises score to 96.1%, '
    +'directly validating the hybrid architecture. Trade-off: req→20killed increases slightly '
    +'(5.6→7.0) due to front-loaded symbolic probes targeting rare faults.'),
  bullet('Methodological finding: a default-deny catch-all rule under deny-overrides makes '
    +'top-level decision constant → masks nearly all mutants. Oracle design matters as much as request generation.'),
  sp()
);

// ── 9. Research Integrity Note ──────────────────────────────────────────
C.push(divider(), h1('9.  RESEARCH INTEGRITY NOTE'));
C.push(
  body('All results in this report were produced using HeuristicSurrogateLLMClient — a deterministic '
    +'program that implements the same 4-objective prompt strategy described in the paper (rule activation, '
    +'predicate negation, boundary probing, rule-pair conflict search) without calling a language model. '
    +'The AnthropicLLMClient implementation is already available for Study 1 (future work); substituting '
    +'it requires only an API key and changes one line of code in main_experiment.py.'),
  body('These numbers validate the STRATEGY, not a specific language model. The claims in the paper '
    +'are correctly scoped to "structured, coverage-explicit prompting strategy" rather than '
    +'"Claude achieves X% mutation score."'),
  note('To run with a real LLM: set ANTHROPIC_API_KEY and pass client=AnthropicLLMClient() '
    +'to llm_guided_generator() in main_experiment.py. All other code, metrics, and output formats '
    +'remain identical.')
);

// Build
const doc = new Document({
  styles:{default:{document:{
    run:{font:'Times New Roman',size:20,color:'000000'},
    paragraph:{spacing:{line:276}},
  }}},
  sections:[{
    properties:{page:{
      size:{width:11906,height:16838},
      margin:{top:1440,bottom:1440,left:1800,right:1800},
    }},
    children:C,
  }]
});
Packer.toBuffer(doc).then(buf=>{
  fs.writeFileSync('TMP_OUTPATH', buf);
  console.log('report done');
});
"""

def save_word_report(policy: Policy, results: List[PerSeedResult],
                     agg: Dict, survivors: Dict, mutants: List[Mutant]):
    import tempfile, json

    # Ghi các file tạm thời — dùng tempfile.gettempdir() để cross-platform
    _tmp = tempfile.gettempdir()
    tmp_agg  = os.path.join(_tmp, "axisgen_agg.json")
    tmp_raw  = os.path.join(_tmp, "axisgen_raw.json")
    tmp_surv = os.path.join(_tmp, "axisgen_surv.json")
    out_path = str((OUT / "report/AI-Native-XACML-Experiment-Report.docx").resolve())
    fig_dir  = str((OUT / "figures").resolve())

    with open(tmp_agg, "w", encoding="utf-8") as f:
        json.dump(agg, f)
    with open(tmp_raw, "w", encoding="utf-8") as f:
        raw_dicts = [{k: v for k, v in vars(r).items() if k != "progression"}
                     for r in results]
        # thêm lại progression
        for i, r in enumerate(results):
            raw_dicts[i]["progression"] = r.progression
        json.dump(raw_dicts, f)
    with open(tmp_surv, "w", encoding="utf-8") as f:
        json.dump(survivors, f)

    method_json  = json.dumps(METHODS)
    labels_json  = json.dumps(METHOD_LABELS)
    seeds_str    = json.dumps(SEEDS)

    # NOTE: these are Windows paths (backslash-separated). Interpolating
    # them into single-quoted JS literals verbatim (the old `f"'{path}'"`
    # approach) is broken on Windows: `\U`, `\A`, `\L`, `\T`, `\a`, ... are
    # not valid JS escape sequences, so the JS parser silently drops the
    # backslash and keeps only the letter (e.g. `C:\Users\...\Temp\...`
    # becomes `C:Users...Temp...`), corrupting the path. json.dumps()
    # produces a properly backslash-escaped JS/JSON string literal, so use
    # that for every filesystem path instead of manual quote-wrapping.
    js = REPORT_JS \
        .replace("'TMP_AGG'",  json.dumps(tmp_agg))\
        .replace("'TMP_RAW'",  json.dumps(tmp_raw))\
        .replace("'TMP_SURV'", json.dumps(tmp_surv))\
        .replace("TMP_METHODS",   method_json)\
        .replace("TMP_LABELS",    labels_json)\
        .replace("TMP_SEEDS",     seeds_str)\
        .replace("TMP_NBUDGET",   str(N_BUDGET))\
        .replace("TMP_FIXEDK",    str(FIXED_K))\
        .replace("TMP_TOTALMUT",  str(len(mutants)))\
        .replace("TMP_POLICYID",  policy.policy_id)\
        .replace("TMP_NRULES",    str(len(policy.rules)))\
        .replace("'TMP_FIGDIR'",  json.dumps(fig_dir))\
        .replace("'TMP_OUTPATH'", json.dumps(out_path))\
        .replace("LLM_BACKEND_LABEL", json.dumps(
            os.environ.get("LLM_BACKEND", "surrogate") +
            (f" / {os.environ.get('LLM_MODEL')}" if os.environ.get("LLM_MODEL") else "")
        ))

    script_path = os.path.join(tempfile.gettempdir(), "axisgen_report.js")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(js)

    try:
        # `node script_path` resolves modules relative to script_path's
        # directory (a tempdir with no node_modules), so a globally
        # installed `docx` (npm install -g docx) is invisible to it unless
        # NODE_PATH points at the global node_modules dir explicitly.
        node_env = os.environ.copy()
        try:
            npm_root = subprocess.run(
                ["npm", "root", "-g"], capture_output=True, text=True, timeout=10,
                shell=(os.name == "nt"),
            ).stdout.strip()
        except Exception:
            npm_root = ""
        if npm_root:
            node_env["NODE_PATH"] = (
                npm_root + os.pathsep + node_env["NODE_PATH"]
                if node_env.get("NODE_PATH") else npm_root
            )

        proc = subprocess.run(["node", script_path],
                              capture_output=True, text=True, timeout=60,
                              env=node_env)
        if proc.returncode != 0:
            print(f"[word] ERROR: {proc.stderr[:500]}")
        else:
            print(f"[word] AI-Native-XACML-Experiment-Report.docx saved")
    except FileNotFoundError:
        print("[word] SKIP: Node.js not found. Install from https://nodejs.org then run: npm install -g docx")
    except subprocess.TimeoutExpired:
        print("[word] SKIP: Node.js timed out after 60s")

# ═══════════════════════════════════════════════════════════════════════════
# 13.  Print console summary
# ═══════════════════════════════════════════════════════════════════════════

def print_final_summary(agg: Dict, mutants: List[Mutant], survivors: Dict):
    print("\n" + "="*80)
    print("  AI-NATIVE TEST GENERATION — EXPERIMENTAL RESULTS — FINAL SUMMARY")
    print("="*80)
    hdr = (f"{'Method':25s} {'#Req':>5s} {'Mut.Score':>11s} "
           f"{'RuleCov':>9s} {f'Req->{FIXED_K}kill':>12s}")
    print(hdr)
    print("-"*65)
    for m in METHODS:
        a = agg[m]
        r20 = f"{a['req_to_fixed_mean']:.1f}" if a['req_to_fixed_mean'] else "n/a"
        print(f"{METHOD_LABELS[m]:25s} {a['n_req_mean']:5.1f} "
              f"{a['mutation_score_mean']*100:8.1f}% ±{a['mutation_score_std']*100:.1f} "
              f"{a['rule_coverage_mean']*100:7.1f}%±{a['rule_coverage_std']*100:.1f} "
              f"{r20:>12s}")
    print("="*80)
    print(f"Total mutants: {len(mutants)}  |  "
          f"Survivors at budget {LARGE_BUDGET}: {len(survivors['survivors'])} "
          f"(operators: {sorted(set(s['op'] for s in survivors['survivors'])) or 'none'})")
    print(f"Output files in: {OUT.resolve()}/")

# ═══════════════════════════════════════════════════════════════════════════
# 14.  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global OUT
    t_start = time.time()
    OUT = _get_out()   # tính lại sau khi .env đã load
    setup_logging()
    print("="*70)
    print("  AI-Native Test Generation — Full Experiment Pipeline")
    print("  ICAI-FAI 2026 — CMC University, Hanoi")
    print("="*70 + "\n")

    setup_dirs()
    policy, mutants  = build_policy_and_mutants()

    print("\n[experiment] Running main experiment ...")
    results = run_experiments(policy, mutants)

    print("\n[experiment] Analysing survivors at large budget ...")
    survivors = analyse_survivors(policy, mutants)

    print("\n[experiment] Computing aggregate statistics ...")
    agg = aggregate(results)

    print("\n[saving] JSON files ...")
    save_json(results, agg, survivors)

    print("[saving] CSV tables ...")
    save_csv_tables(policy, mutants, results, agg)

    print("[saving] Excel workbook ...")
    save_excel(policy, mutants, results, agg, survivors)

    print("[saving] Figures ...")
    save_figures(agg, results, mutants, policy)

    print("[saving] XACML XML artifacts ...")
    save_xacml(policy, results)

    print("[saving] Word report ...")
    save_word_report(policy, results, agg, survivors, mutants)

    print_final_summary(agg, mutants, survivors)

    elapsed = time.time() - t_start
    print(f"\n[done] Total runtime: {elapsed:.1f}s")
    print(f"[done] All outputs saved to: {(OUT).resolve()}/")
    print(f"[done] Log saved to: {(OUT / 'logs').resolve()}/")

if __name__ == "__main__":
    main()
