# AXIS-Gen — AI-Native Test Generation cho Kiểm soát truy cập công nghiệp

Mã nguồn thực nghiệm cho bài báo **"AI-Native Test Generation for Industrial Access Control: LLM-Guided XACML Request Synthesis"** (ICAI-FAI 2026, CMC University, Hà Nội).

Dự án so sánh 4 phương pháp sinh **request kiểm thử XACML** cho một chính sách ABAC của hệ thống điều khiển công nghiệp (ICS), và đo chất lượng bằng **mutation testing**.

## Mục lục

1. [Ý tưởng tổng quan](#1-ý-tưởng-tổng-quan)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Policy được kiểm thử](#3-policy-được-kiểm-thử)
4. [Các toán tử mutation](#4-các-toán-tử-mutation)
5. [Các chỉ số đo](#5-các-chỉ-số-đo)
6. [Cài đặt](#6-cài-đặt)
7. **[Chạy thực nghiệm với Groq (cách chuẩn)](#7-chạy-thực-nghiệm-với-groq-cách-chuẩn)**
8. [Chạy với backend khác](#8-chạy-với-backend-khác-gemini--openai--anthropic--surrogate)
9. [Backend Surrogate — lưu ý khoa học](#9-backend-surrogate--lưu-ý-về-tính-trung-thực-khoa-học)
10. [Cách pipeline xử lý lỗi LLM](#10-cách-pipeline-xử-lý-lỗi-llm)
11. [Đầu ra chính](#11-đầu-ra-chính)
12. [Bảo mật](#12-bảo-mật)
13. [Khắc phục sự cố](#13-khắc-phục-sự-cố)
14. [Tài liệu tham khảo](#14-tài-liệu-tham-khảo-phương-pháp)

---

## 1. Ý tưởng tổng quan

Chính sách truy cập (policy) có thể chứa lỗi (sai effect, sai điều kiện, thiếu rule...). Một bộ test tốt là bộ test **phát hiện được nhiều lỗi nhất với ít request nhất**.

Cách đo:
1. Từ policy gốc, tạo **51 mutant** (mỗi mutant là policy gốc + đúng 1 lỗi được gieo vào).
2. Chạy một bộ request trên policy gốc và trên từng mutant.
3. Mutant bị **kill** nếu có ít nhất một request cho quyết định khác policy gốc.
4. **Mutation score** = số mutant bị kill / tổng số mutant.

Bốn phương pháp sinh request (ngân sách tối đa 40 request):

| Phương pháp | Mô tả |
|---|---|
| **Random** | Lấy mẫu ngẫu nhiên đều trên miền giá trị thuộc tính |
| **Pairwise (2-way)** | Sinh tổ hợp phủ mọi cặp giá trị (baseline cổ điển, kiểu X-CREATE). Dừng khi đã phủ hết cặp nên thường dùng < 40 request (~30) |
| **LLM-Guided** | Đưa policy dạng ngôn ngữ tự nhiên cho LLM, yêu cầu sinh request kích hoạt từng rule, lật từng điều kiện (MC/DC), thử giá trị biên, thử tương tác giữa các rule |
| **LLM-Guided + Symbolic** (AXIS-Gen) | Đúng bộ request của LLM-Guided **cộng thêm** các request "cặp rule xung đột" tìm bằng duyệt lưới toàn bộ không gian thuộc tính (2 request với policy này) |

**Thiết kế ablation ghép cặp (paired):** với mỗi seed, pipeline chỉ gọi LLM **một lần**; `LLM-Guided+Symbolic` **dùng lại đúng kết quả đó** và chỉ thêm các request symbolic (cắt bớt cuối để giữ ngân sách). Nhờ vậy chênh lệch giữa hai phương pháp là do bước symbolic, không phải do nhiễu lấy mẫu của LLM.

Lý do có bước symbolic: LLM khó tìm được request mà **hai rule cùng áp dụng** (cần để giết các mutant đổi thuật toán kết hợp — RCM). Bước symbolic bù đúng khoảng trống này.

---

## 2. Cấu trúc thư mục

```
.
├── main_experiment.py     # Chạy 1 backend / 1 model: sinh request, chấm điểm, xuất bảng/hình/báo cáo
├── run_groq_compare.py    # Chạy nhiều model Groq lần lượt + bảng so sánh (cách chuẩn với Groq)
├── generators.py          # 4 bộ sinh request + client LLM (Groq, Gemini, OpenAI, Anthropic, Surrogate)
├── policy_model.py        # Mô hình XACML/ABAC, PDP (deny/permit-overrides, first-applicable), xuất XML
├── ics_policy.py          # Policy nhà máy xử lý nước (6 rule) — đối tượng được kiểm thử
├── mutation.py            # 7 toán tử mutation, sinh 51 mutant
├── harness.py             # Chạy bộ test, tính mutation score, rule coverage, hiệu quả
├── .env                   # File "router": chọn backend cho main_experiment.py qua ENV_FILE
├── .env.groq              # Cấu hình Groq          (chứa API key — KHÔNG commit)
├── .env.gemini            # Cấu hình Gemini       (chứa API key — KHÔNG commit)
├── .env.openai            # Cấu hình OpenAI GPT    (chứa API key — KHÔNG commit)
├── .env.anthropic         # Cấu hình Claude        (chứa API key — KHÔNG commit)
├── .env.surrogate         # Chạy offline, không cần key
├── IEEE_Conference_Template/   # Bài báo LaTeX (conference_101719.tex) + hình/số liệu dùng trong bài
└── results/               # Tự tạo khi chạy
    ├── surrogate/                     # Kết quả backend surrogate
    └── groq/
        ├── model_comparison.csv       # Bảng so sánh các model (do run_groq_compare.py ghi)
        └── <tên-model>/               # Mỗi model Groq một thư mục, vd: openai_gpt-oss-120b/
            ├── data/      raw_results.json, aggregate_stats.json, experiment_results.xlsx
            ├── tables/    TableI–IV (CSV)
            ├── figures/   fig1–fig5 (PNG)
            ├── xacml/     policy.xml + request mẫu XACML 3.0
            ├── report/    Báo cáo Word (cần Node.js + docx)
            └── logs/      experiment_<thời gian>.log
```

---

## 3. Policy được kiểm thử

Policy `ICS-WaterTreatment-ABAC-v1`, thuật toán kết hợp **deny-overrides**, 6 rule:

| Rule | Effect | Ý nghĩa |
|---|---|---|
| R1 | PERMIT | Operator đọc HMI/PLC ở vùng control/supervisory |
| R2 | PERMIT | Engineer đọc/ghi ở vùng field–supervisory, trừ khi ở chế độ khẩn cấp |
| R3 | DENY | Cấm override safety interlock khi `safety_state = normal` |
| R4 | PERMIT | Engineer đủ clearance (≥4), tại chỗ, được override interlock khi bảo trì |
| R5 | DENY | Vendor bị cấm truy cập vùng level-0 field |
| R6 | PERMIT | Vendor đọc historian khi đang sản xuất |

Không gian thuộc tính: 9 thuộc tính (role, clearance_level, location, type, zone, criticality, action, safety_state, plant_mode) → **92.160 tổ hợp**.

Cố ý **không** thêm rule "default-deny" vì dưới deny-overrides nó sẽ chi phối mọi quyết định và che mất tác dụng của các mutation khác.

---

## 4. Các toán tử mutation

| Mã | Tên | Lỗi được gieo |
|---|---|---|
| RCM | Rule Combining Mutation | Đổi thuật toán kết hợp |
| CEM | Effect Mutation | Đổi Permit ↔ Deny của một rule |
| CPM | Comparison Mutation | Đảo toán tử so sánh (eq→neq, ≥→<, ...) |
| CVM | Constant Value Mutation | Dịch ngưỡng số ±1 |
| TRM | Target Removal | Bỏ một điều kiện trong Target (mở rộng phạm vi) |
| LOM | Logical Operator Mutation | AND → OR |
| MRD | Missing Rule Deletion | Xóa cả rule |

> Lưu ý: mutant `RCM_first-applicable` là **mutant tương đương** với policy này (cho quyết định giống hệt policy gốc trên cả 92.160 điểm), nên mutation score tối đa đạt được là 50/51 ≈ **98.0%**.

---

## 5. Các chỉ số đo

- **Mutation score** — tỉ lệ mutant bị kill (chỉ số chính).
- **Rule coverage** — tỉ lệ rule được "chạm tới" ít nhất một lần.
- **Requests-to-20-killed** — số request cần để giết 20 mutant (hiệu quả).
- **Phân bố quyết định** — tỉ lệ PERMIT/DENY/NOT_APPLICABLE (chống bộ test toàn DENY).
- **Phân tích survivor** — mutant nào sống sót ở ngân sách lớn (300 request, **dùng surrogate**).

---

## 6. Cài đặt

Yêu cầu: Python ≥ 3.10.

```bash
pip install lxml matplotlib openpyxl numpy python-dotenv
pip install openai            # cho Groq và OpenAI (Groq dùng API tương thích OpenAI)
pip install google-genai      # nếu dùng Gemini
pip install anthropic         # nếu dùng Claude
```

Tùy chọn (xuất báo cáo Word): cài Node.js rồi `npm install -g docx`. Thiếu bước này pipeline vẫn chạy bình thường, chỉ báo lỗi `Cannot find module 'docx'` ở bước cuối và bỏ qua file `.docx` (kết quả khác không bị ảnh hưởng).

---

## 7. Chạy thực nghiệm với Groq (cách chuẩn)

### 7.1. Chuẩn bị (làm một lần)

1. Lấy API key miễn phí tại https://console.groq.com/keys (dạng `gsk_...`).
2. Mở file `.env.groq`, điền key và đặt số seed:

   ```ini
   LLM_BACKEND=groq
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
   LLM_MODEL=openai/gpt-oss-120b
   N_SEEDS=5
   ```

   - `N_SEEDS=1` là chế độ chạy thử (nhanh, ít tốn quota); `N_SEEDS=5` là chạy đầy đủ để lấy số liệu cho bài.
   - Không cần sửa file `.env` khi dùng `run_groq_compare.py` — script tự chọn `.env.groq`.

### 7.2. Chạy thử nhanh (khuyến nghị làm trước)

Đặt `N_SEEDS=1` trong `.env.groq`, đặt `ENV_FILE=.env.groq` trong `.env`, rồi:

```bash
python main_experiment.py
```

Kiểm tra trong log:
- `Parsed request count:` ≈ 40 (không phải 0);
- `domain-repair: 0/360 values invalid` (hoặc rất thấp);
- LLM-Guided có mutation score cao hơn Random.

Nếu ổn, đổi `N_SEEDS=5` và chạy tiếp bước 7.3.

### 7.3. Chạy đầy đủ nhiều model — lệnh chuẩn

```bash
python run_groq_compare.py
```

Lệnh này chạy **toàn bộ thực nghiệm (5 seed)** lần lượt cho từng model, mỗi model một tiến trình riêng:

| Model mặc định | Ghi chú |
|---|---|
| `openai/gpt-oss-20b` | Model nhỏ |
| `openai/gpt-oss-120b` | Model lớn |

Trước khi chạy, script hỏi Groq xem key dùng được model nào và **tự bỏ qua** model không tồn tại. Mỗi model mất khoảng 3–5 phút. Một model lỗi không làm dừng các model còn lại.

**Chỉ chạy một hoặc vài model cụ thể** (đặt biến `GROQ_MODELS`, phân cách bằng dấu phẩy):

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

Bảng so sánh `results/groq/model_comparison.csv` **gộp mọi model đã có kết quả** (kể cả các lần chạy trước), nên có thể chạy từng model riêng rồi gộp dần.

> **Không phải lúc nào cũng chạy `run_groq_compare.py`:** `python main_experiment.py` chỉ chạy **một** model (lấy từ `LLM_MODEL`) và ghi vào `results/groq/` (không có thư mục con theo model, không có bảng so sánh). Dùng nó để chạy thử; dùng `run_groq_compare.py` cho số liệu chính thức.

### 7.4. Kết quả

Sau khi chạy, console in bảng tổng hợp và lưu:

- `results/groq/model_comparison.csv` — bảng so sánh các model: mutation score, rule coverage, số seed hợp lệ / lỗi, số request trung bình, tỉ lệ giá trị LLM sai miền;
- `results/groq/<tên-model>/` — bảng, hình, xlsx, XACML, log chi tiết của từng model (xem cấu trúc ở mục 2).

Hai baseline Random và Pairwise **không phụ thuộc model**, nên có số liệu giống hệt nhau ở mọi thư mục model — điều này là bình thường.

### 7.5. Lưu ý riêng của Groq

- **Gói miễn phí có giới hạn token/phút và token/ngày.** Nếu gặp `429`, chờ vài phút rồi chạy lại model đó.
- **`qwen/qwen3.8-27b` không chạy được trên gói miễn phí:** giới hạn 1000 token output/phút, trong khi một lần sinh 40 request cần khoảng 5000. Pipeline dừng ngay với thông báo "request exceeds this model's per-minute token limit". Cần nâng cấp Groq Dev Tier; sau đó thêm bằng `GROQ_MODELS=openai/gpt-oss-120b,qwen/qwen3.8-27b`.
- Danh sách model Groq thay đổi thường xuyên. Model cũ (như Llama 3.x) có thể trả `404 model_not_found`. Xem model hiện có ở https://console.groq.com/docs/models — hoặc để `run_groq_compare.py` tự in ra danh sách khả dụng.
- Với các model `gpt-oss`, client đặt `reasoning_effort="low"` để model không dùng hết token cho phần suy luận ẩn (nếu không, JSON đầu ra dễ bị cắt cụt).
- `temperature = 0.7`, `seed` được truyền vào API; tuy nhiên suy luận trên dịch vụ đám mây **không đảm bảo lặp lại chính xác từng bit** — hai lần chạy cùng cấu hình có thể lệch vài điểm phần trăm mutation score. Vì vậy hãy báo cáo mean ± std trên 5 seed và không suy diễn từ chênh lệch nhỏ.

**Cách báo cáo số liệu Groq trong bài (quan trọng):**
- Ghi **đúng số liệu trong file kết quả** (mean ± std giữa các seed, không làm tròn hay dùng "khoảng"), và nêu rõ mỗi model chỉ chạy **một lần** (5 seed), nên độ dao động giữa các lần chạy chưa được đo.
- Chênh lệch giữa hai model (20b và 120b) và giữa `LLM-Guided` với `LLM-Guided+Symbolic` (+0.8 đến +1.6 điểm) nhỏ hơn độ lệch chuẩn, nên **không kết luận** model nào tốt hơn hay bước symbolic có ý nghĩa thống kê; chỉ mô tả.
- Số liệu chính thức là của lần chạy cuối bằng code hiện tại (ablation ghép cặp). Kết quả các lần chạy cũ đã bị ghi đè và không dùng được.

---

## 8. Chạy với backend khác (Gemini / OpenAI / Anthropic / Surrogate)

Các backend này chạy bằng `main_experiment.py`. Mở file `.env`, **chỉ để một dòng `ENV_FILE` không bị comment**:

```ini
# ENV_FILE=.env.surrogate
# ENV_FILE=.env.groq
# ENV_FILE=.env.openai
ENV_FILE=.env.gemini
# ENV_FILE=.env.anthropic
```

Rồi chạy:

```bash
python main_experiment.py
```

Mỗi file `.env.<backend>` có dạng:

```ini
LLM_BACKEND=gemini
GEMINI_API_KEY=<key của bạn>
LLM_MODEL=gemini-3.5-flash
N_SEEDS=1        # 1 = chế độ debug; đổi thành 5 khi chạy thật
```

| Backend | Biến key | Model mặc định | Lấy key |
|---|---|---|---|
| groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` | https://console.groq.com/keys |
| gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` | https://aistudio.google.com/app/apikey |
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | https://platform.openai.com/api-keys |
| anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | https://console.anthropic.com/settings/keys |
| surrogate | (không cần) | — | — |

`LLM_MODEL` ghi đè model mặc định. Anthropic có thêm `ANTHROPIC_WORKSPACE_ID` nếu key không gắn workspace. Kết quả ghi vào `results/<backend>/`, các backend không ghi đè lẫn nhau.

Quy trình khuyến nghị cho mọi backend: chạy `surrogate` trước để kiểm tra pipeline offline → chạy `N_SEEDS=1` với LLM thật → đổi `N_SEEDS=5`.

Cấu hình mặc định (trong `main_experiment.py`): 5 seed, ngân sách 40 request/phương pháp, ngưỡng hiệu quả 20 mutant.

---

## 9. Backend "Surrogate" — lưu ý về tính trung thực khoa học

`HeuristicSurrogateLLMClient` **không phải LLM thật**. Nó thực thi tất định chiến lược prompt (kích hoạt rule, lật từng điều kiện, thử giá trị biên) để kiểm tra pipeline và phương pháp đo mà không cần API.

- Số liệu từ surrogate **không được trình bày như kết quả của LLM thật** trong bài báo — chỉ là mốc tham chiếu cho "trần" của chiến lược.
- Phân tích survivor (`analyse_survivors`) và request mẫu XACML luôn dùng surrogate, bất kể backend đang chọn.

---

## 10. Cách pipeline xử lý lỗi LLM

| Tình huống | Hành vi |
|---|---|
| Key sai/hết hạn (401), model sai (404), project bị chặn (403) | **Dừng ngay** với thông báo rõ ràng, không cho ra kết quả 0% gây hiểu nhầm |
| OpenAI: hết credit (`insufficient_quota`) | **Dừng ngay** ("out of credits") |
| Groq/OpenAI: `Request too large` (vượt giới hạn token/phút của model) | **Dừng ngay** — thử lại không giúp được |
| 429 do giới hạn tốc độ thông thường | SDK tự thử lại (tối đa 3 lần, theo `retry-after`); Gemini thử lại 3 lần, chờ 20–90 giây |
| 5xx / quá tải | Thử lại với backoff lũy thừa |
| LLM trả rỗng (`[]`) / không phải JSON | **Thử lại tối đa 3 lần** với seed lấy mẫu khác trước khi coi là lỗi |
| Output bị cắt cụt (hết token) | Cảnh báo và cứu lại các object JSON đã hoàn chỉnh |
| Khối `<think>...</think>` của model suy luận | Tự loại bỏ trước khi đọc JSON |
| LLM trả thừa request | Cắt về đúng N để giữ ngân sách công bằng (trả thiếu: giữ nguyên, ghi nhận trong log) |
| Giá trị ngoài miền | Chuẩn hóa nếu chỉ lệch định dạng (`"3"`→3); còn lại thay bằng giá trị ngẫu nhiên và **đếm** tỉ lệ này |
| Seed mà LLM lỗi | Đánh dấu `failed`, **loại khỏi thống kê** kèm cảnh báo |
| LLM lỗi ở **2 seed liên tiếp** | **Dừng cả lần chạy** (lỗi hệ thống như hết quota), không xuất báo cáo toàn số 0 |

Thống kê chất lượng LLM (số request trả về, tỉ lệ giá trị sai miền, số lần thử, seed lỗi) được ghi trong `aggregate_stats.json`, mục `llm_diagnostics`.

---

## 11. Đầu ra chính

- **Table I** — 6 rule của policy; **Table II** — 51 mutant
- **Table III** — kết quả tổng hợp (mean ± std) cho 4 phương pháp
- **Table IV** — chi tiết từng seed × phương pháp
- **Fig 1** mutation score · **Fig 2** rule coverage · **Fig 3** tiến trình kill · **Fig 4** phân bố quyết định · **Fig 5** heatmap kill theo toán tử
- `experiment_results.xlsx` (6 sheet), `policy.xml` và request mẫu XACML 3.0, báo cáo Word
- `results/groq/model_comparison.csv` — so sánh các model Groq

---

## 12. Bảo mật

- File `.env*` chứa API key đã được chặn bởi `.gitignore`. **Không commit, không nén chung để gửi.**
- Key đã từng lộ thì tạo lại ngay trên trang của nhà cung cấp.
- Repo chỉ nên commit `.env.surrogate` (không có key).

---

## 13. Khắc phục sự cố

| Triệu chứng | Nguyên nhân / Cách xử lý |
|---|---|
| `404 model_not_found` (Groq) | Model đã bị gỡ hoặc key không có quyền. Xem `https://console.groq.com/docs/models`; `run_groq_compare.py` tự in danh sách model khả dụng |
| `429 ... Request too large ... OTPM` (Groq, vd Qwen) | Giới hạn token/phút của gói miễn phí nhỏ hơn một câu trả lời. Dùng model khác hoặc nâng cấp Dev Tier |
| `429 rate_limit_exceeded` (Groq) | Hết token/phút hoặc /ngày. Chờ rồi chạy lại model đó |
| `401 invalid_api_key` (Groq) | Sai key trong `.env.groq` |
| `403 PERMISSION_DENIED ... project has been denied access` (Gemini) | Google project của key bị chặn. Tạo key trong **project mới** hoặc dùng tài khoản khác |
| `401 ... API key has expired` (OpenAI) | Key hết hạn. Tạo key mới, kiểm tra credit |
| `429 insufficient_quota` (OpenAI) | Hết credit thanh toán |
| Kết quả LLM-Guided = 0% hoặc ít request | Seed LLM lỗi; xem log tìm `FAILED` / `retrying` và `llm_diagnostics` |
| Model bị cắt cụt (`finish_reason=length`) | Model suy luận dùng hết token; xem mục 7.5 |
| `ModuleNotFoundError: lxml` (hoặc openai, dotenv...) | `pip install lxml openai python-dotenv` (xem mục 6) |
| `Cannot find module 'docx'` | `npm install -g docx` (chỉ ảnh hưởng báo cáo Word) |
| Cảnh báo "automatic function calling (AFC)" | Vô hại, của SDK Gemini |

---

## 14. Tài liệu tham khảo phương pháp

- Martin & Xie (2007), *A fault model and mutation testing of access control policies*.
- Bertolino et al. (2010/2012), XACMUT và X-CREATE — sinh test và mutation cho XACML.
- Xu et al., tiêu chí rule-pair coverage.
