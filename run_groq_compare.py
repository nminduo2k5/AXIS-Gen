"""
Chạy toàn bộ thực nghiệm lần lượt với nhiều model Groq rồi tổng hợp so sánh.

    python run_groq_compare.py

Mỗi model chạy main_experiment.py như một tiến trình riêng, kết quả lưu ở
results/groq/<tên-model>/. Cuối cùng ghi results/groq/model_comparison.csv.
Một model lỗi (sai tên, hết quota...) không làm dừng các model còn lại.
Đổi danh sách model bằng biến môi trường GROQ_MODELS (phân cách bằng dấu phẩy).
"""
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",         # nhỏ  (20B)
    "openai/gpt-oss-120b",        # lớn  (120B)
    # "qwen/qwen3.8-27b" cần Groq Dev Tier: gói miễn phí giới hạn 1000 token
    # output/phút, không đủ cho một lần sinh 40 request. Thêm qua GROQ_MODELS.
]
MODELS = [m.strip() for m in os.environ.get("GROQ_MODELS", "").split(",") if m.strip()] or DEFAULT_MODELS
METHODS = ["random", "pairwise", "llm_guided", "llm_guided_hybrid"]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def available_models():
    """Hỏi Groq danh sách model mà key này thực sự dùng được (None nếu không hỏi được)."""
    try:
        from dotenv import dotenv_values
        from openai import OpenAI
        key = dotenv_values(HERE / ".env.groq").get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        return {m.id for m in client.models.list().data}
    except Exception as e:
        print(f"[compare] Không kiểm tra được danh sách model ({str(e)[:120]}); vẫn thử chạy.")
        return None


def main():
    ok = {}
    avail = available_models()
    if avail is not None:
        missing = [m for m in MODELS if m not in avail]
        if missing:
            print(f"[compare] BỎ QUA model không có trên Groq (key này): {missing}")
            print("[compare] Model đang có: " + ", ".join(sorted(avail)))
    for model in MODELS:
        if avail is not None and model not in avail:
            ok[model] = False
            continue
        print(f"\n{'=' * 70}\n  GROQ MODEL: {model}\n{'=' * 70}", flush=True)
        env = os.environ.copy()
        env.update({
            "ENV_FILE": ".env.groq",
            "LLM_MODEL_OVERRIDE": model,
            "RESULTS_SUBDIR": slug(model),
        })
        rc = subprocess.run([sys.executable, str(HERE / "main_experiment.py")],
                            cwd=HERE, env=env).returncode
        ok[model] = rc == 0
        if rc != 0:
            print(f"[compare] {model}: FAILED (exit code {rc}) — chuyển sang model tiếp theo.")

    # Bảng so sánh gộp MỌI model đã có kết quả (kể cả những lần chạy trước),
    # nên có thể chạy từng model riêng bằng GROQ_MODELS=<model> rồi gộp dần.
    rows = []
    done_dirs = sorted(p.parent.parent for p in (HERE / "results" / "groq").glob("*/data/aggregate_stats.json"))
    for model_dir in done_dirs:
        model = model_dir.name
        stats_path = model_dir / "data" / "aggregate_stats.json"
        agg = json.loads(stats_path.read_text(encoding="utf-8"))
        diag = [d for d in agg.get("llm_diagnostics", [])]
        tv = sum(d["total_values"] for d in diag) or 1
        invalid = 100.0 * sum(d["invalid_values"] for d in diag) / tv
        for meth in ("llm_guided", "llm_guided_hybrid"):
            a = agg["aggregate"][meth]
            rows.append([
                model, meth, a["n_seeds"], a.get("n_seeds_failed", 0),
                f"{a['n_req_mean']:.1f}",
                f"{a['mutation_score_mean'] * 100:.1f}", f"{a['mutation_score_std'] * 100:.1f}",
                f"{a['rule_coverage_mean'] * 100:.1f}",
                f"{invalid:.1f}",
            ])

    if not rows:
        print("[compare] Chưa có model nào chạy thành công — không tạo bảng so sánh.")
        sys.exit(1)

    out = HERE / "results" / "groq" / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Method", "Valid seeds", "Failed seeds", "Avg #Req",
                    "Mutation score % (mean)", "Std", "Rule coverage %", "Invalid LLM values %"])
        w.writerows(rows)

    print(f"\n{'=' * 78}\n  SO SÁNH CÁC MODEL GROQ\n{'=' * 78}")
    print(f"{'Model':28s} {'Method':20s} {'Seeds':>5s} {'#Req':>5s} {'MutScore%':>10s} {'RuleCov%':>9s} {'Invalid%':>8s}")
    for r in rows:
        print(f"{r[0]:28s} {r[1]:20s} {r[2]:>5} {r[4]:>5s} {r[5] + '±' + r[6]:>10s} {r[7]:>9s} {r[8]:>8s}")
    print(f"\n[compare] Đã lưu: {out}")
    failed = [m for m in MODELS if not ok.get(m)]
    if failed:
        print(f"[compare] Model thất bại: {failed}")


if __name__ == "__main__":
    main()
