# Hướng dẫn chạy thực nghiệm — AXIS-Gen ICAI-FAI 2026

> Môi trường trên máy:
> - Python: `C:\Users\Admin\miniconda3\python.exe` (Python 3.10 / conda env `duong`)
> - Conda environment: `duong`

---

## Cấu trúc file cấu hình

```
.env              ← file router, chỉ chọn backend nào dùng
.env.surrogate    ← chạy offline, không cần API key
.env.openai       ← chạy với OpenAI GPT
.env.gemini       ← chạy với Google Gemini
```

Kết quả mỗi backend được lưu vào thư mục riêng:

```
results/
├── surrogate/   ← kết quả surrogate
├── openai/      ← kết quả OpenAI
└── gemini/      ← kết quả Gemini
```

---

## Bước 0 — Cài dependencies (chạy 1 lần)

```powershell
conda activate duong
pip install matplotlib openpyxl lxml numpy python-dotenv openai google-genai
```

---

## Cách chọn backend

Mở file `.env`, bỏ comment dòng muốn dùng:

```ini
# Chọn 1 trong 3 dòng dưới, comment 2 dòng còn lại

ENV_FILE=.env.surrogate
# ENV_FILE=.env.openai
# ENV_FILE=.env.gemini
```

Sau đó chạy:

```powershell
conda activate duong
cd "C:\Users\Admin\Desktop\AI-Native-TestGen-ICAI-FAI-2026_Full-Experiment"
python main_experiment.py
```

---

## 1. Surrogate (offline, không cần API key)

**Dùng khi**: chạy thử, verify pipeline, không có API key.

**Bước 1** — Sửa `.env`:
```ini
ENV_FILE=.env.surrogate
```

**Bước 2** — Chạy:
```powershell
python main_experiment.py
```

**Thời gian**: ~45 giây

**Output**: `results/surrogate/`

**Kết quả kỳ vọng**:
```
Method                    #Req   Mut.Score     RuleCov   Req->20kill
Random                    40.0   53.3% ±3.8    53.3%        16.0
Pairwise (2-way)          30.4   56.9% ±5.9    60.0%        13.2
LLM-Guided                40.0   94.1% ±2.1   100.0%         5.6
LLM-Guided+Symbolic       40.0   96.1% ±2.1   100.0%         6.8
```

---

## 2. OpenAI (GPT-4o-mini)

**Yêu cầu**: tài khoản OpenAI có credits tại https://platform.openai.com/settings/organization/billing

**Bước 1** — Kiểm tra credits còn không:
```powershell
python -c "
from dotenv import load_dotenv; load_dotenv('.env.openai')
from openai import OpenAI; import os
c = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
r = c.chat.completions.create(model='gpt-4o-mini', max_tokens=5, messages=[{'role':'user','content':'hi'}])
print('OK:', r.choices[0].message.content)
"
```

**Bước 2** — Sửa `.env`:
```ini
ENV_FILE=.env.openai
```

**Bước 3** — Chạy:
```powershell
python main_experiment.py
```

**Thời gian**: ~5–15 phút (tùy tốc độ API)

**Output**: `results/openai/`

**Đổi model** (tuỳ chọn) — sửa trong `.env.openai`:
```ini
LLM_MODEL=gpt-4o          # mạnh hơn, đắt hơn
LLM_MODEL=gpt-4o-mini     # mặc định, rẻ nhất
LLM_MODEL=gpt-3.5-turbo   # cũ hơn
```

**Chi phí ước tính**: ~$0.10–0.50 USD với gpt-4o-mini cho 5 seeds

**Lỗi thường gặp**:

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| `429 credit_balance_exhausted` | Hết credits | Nạp tiền tại platform.openai.com/billing |
| `401 invalid_api_key` | Key sai | Kiểm tra lại key trong `.env.openai` |
| `404 model_not_found` | Sai tên model | Dùng `gpt-4o-mini` hoặc `gpt-4o` |

---

## 3. Gemini (Google AI)

**Yêu cầu**: API key từ https://aistudio.google.com/app/apikey và project Google Cloud đã enable **Generative Language API**

**Bước 1** — Kiểm tra key và model available:
```powershell
python -c "
from dotenv import load_dotenv; load_dotenv('.env.gemini')
from google import genai; import os
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
for m in client.models.list():
    if 'flash' in m.name and 'gemini' in m.name:
        print(m.name)
"
```

**Bước 2** — Điền model available vào `.env.gemini`:
```ini
LLM_MODEL=gemini-2.5-flash   # thay bằng model từ kết quả bước 1
```
Hoặc để trống `LLM_MODEL=` để pipeline tự detect.

**Bước 3** — Sửa `.env`:
```ini
ENV_FILE=.env.gemini
```

**Bước 4** — Chạy:
```powershell
python main_experiment.py
```

**Thời gian**: ~5–15 phút

**Output**: `results/gemini/`

**Lỗi thường gặp**:

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| `403 PERMISSION_DENIED: Your project has been denied access` | Project Google Cloud bị chặn | Xem hướng dẫn fix bên dưới |
| `404 model is no longer available` | Model đã bị deprecated | Để trống `LLM_MODEL=` để tự detect |
| `400 API key not valid` | Key sai hoặc hết hạn | Tạo key mới tại aistudio.google.com |

**Fix lỗi 403 Gemini**:
1. Vào https://aistudio.google.com/app/apikey → xem key thuộc project nào
2. Vào https://console.cloud.google.com → chọn đúng project
3. Vào **APIs & Services** → **Library** → tìm **Generative Language API** → **Enable**
4. Nếu vẫn lỗi: tạo key mới tại AI Studio (key cũ có thể bị flag)

---

## So sánh 3 backend

| | Surrogate | OpenAI | Gemini |
|--|-----------|--------|--------|
| Cần API key | Không | Có | Có |
| Chi phí | Miễn phí | ~$0.1–0.5/run | Miễn phí (free tier) |
| Thời gian | ~45s | ~5–15 phút | ~5–15 phút |
| Validate | Strategy | GPT model | Gemini model |
| Dùng cho paper | Study baseline | Study 1 | Study 1 |
| Output folder | `results/surrogate/` | `results/openai/` | `results/gemini/` |

---

## Xem kết quả sau khi chạy

```
results/<backend>/
├── data/
│   ├── raw_results.json          ← số liệu thô 5 seeds × 4 methods
│   └── aggregate_stats.json      ← mean/std
├── tables/
│   ├── TableI_policy_rules.csv
│   ├── TableII_mutants.csv
│   ├── TableIII_main_results.csv ← bảng chính dùng cho paper
│   └── TableIV_permethod.csv
├── figures/
│   ├── fig1_mutation_score.png
│   ├── fig2_rule_coverage.png
│   ├── fig3_efficiency.png
│   ├── fig4_decision_dist.png
│   └── fig5_operator_heatmap.png
├── xacml/
│   ├── policy.xml
│   └── sample_requests/
└── logs/
    └── experiment_YYYYMMDD_HHMMSS.log  ← log đầy đủ quá trình chạy
```

---

## Chạy nhanh — tóm tắt lệnh

```powershell
conda activate duong
cd "C:\Users\Admin\Desktop\AI-Native-TestGen-ICAI-FAI-2026_Full-Experiment"

# Surrogate (luôn work)
# → sửa .env: ENV_FILE=.env.surrogate
python main_experiment.py

# OpenAI (cần credits)
# → sửa .env: ENV_FILE=.env.openai
python main_experiment.py

# Gemini (cần fix project permissions)
# → sửa .env: ENV_FILE=.env.gemini
python main_experiment.py
```
