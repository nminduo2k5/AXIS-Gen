# AXIS-Gen — AI-Native Test Generation cho Kiểm soát truy cập công nghiệp

Mã nguồn thực nghiệm cho bài báo **"AI-Native Test Generation for Industrial Access Control: LLM-Guided XACML Request Synthesis"** (ICAI-FAI 2026, CMC University, Hà Nội).

Dự án so sánh 4 phương pháp sinh **request kiểm thử XACML** cho một chính sách ABAC của hệ thống điều khiển công nghiệp (ICS), và đo chất lượng bằng **mutation testing**.

---

## 1. Ý tưởng tổng quan

Chính sách truy cập (policy) có thể chứa lỗi (sai effect, sai điều kiện, thiếu rule...). Một bộ test tốt là bộ test **phát hiện được nhiều lỗi nhất với ít request nhất**.

Cách đo:
1. Từ policy gốc, tạo **51 mutant** (mỗi mutant là policy gốc + đúng 1 lỗi được gieo vào).
2. Chạy một bộ request trên policy gốc và trên từng mutant.
3. Mutant bị **kill** nếu có ít nhất một request cho quyết định khác policy gốc.
4. **Mutation score** = số mutant bị kill / tổng số mutant.

Bốn phương pháp sinh request (cùng ngân sách 40 request):

| Phương pháp | Mô tả |
|---|---|
| **Random** | Lấy mẫu ngẫu nhiên đều trên miền giá trị thuộc tính |
| **Pairwise (2-way)** | Sinh tổ hợp phủ mọi cặp giá trị (baseline cổ điển, kiểu X-CREATE) |
| **LLM-Guided** | Đưa policy dạng ngôn ngữ tự nhiên cho LLM, yêu cầu sinh request kích hoạt từng rule, lật từng điều kiện (MC/DC), thử giá trị biên, thử tương tác giữa các rule |
| **LLM-Guided + Symbolic** (AXIS-Gen) | LLM kết hợp bước tìm "cặp rule xung đột" bằng duyệt lưới toàn bộ không gian thuộc tính |

Lý do có phương pháp thứ 4: LLM/mẫu theo từng rule khó tìm được request mà **hai rule cùng áp dụng** (cần để giết các mutant đổi thuật toán kết hợp — RCM). Bước symbolic bù đúng khoảng trống này.

---

## 2. Cấu trúc thư mục

```
.
├── main_experiment.py     # Chương trình chính: chạy toàn bộ pipeline, xuất bảng/hình/báo cáo
├── generators.py          # 4 bộ sinh request + các client LLM (Gemini, OpenAI, Anthropic, Surrogate)
├── policy_model.py        # Mô hình XACML/ABAC, PDP (deny/permit-overrides, first-applicable), xuất XML
├── ics_policy.py          # Policy nhà máy xử lý nước (6 rule) — đối tượng được kiểm thử
├── mutation.py            # 7 toán tử mutation, sinh 51 mutant
├── harness.py             # Chạy bộ test, tính mutation score, rule coverage, hiệu quả
├── .env                   # File "router": chọn backend qua ENV_FILE
├── .env.gemini            # Cấu hình Gemini        (chứa API key — KHÔNG commit)
├── .env.openai            # Cấu hình OpenAI GPT    (chứa API key — KHÔNG commit)
├── .env.anthropic         # Cấu hình Claude        (chứa API key — KHÔNG commit)
├── .env.surrogate         # Chạy offline, không cần key
├── HUONG_DAN_CHAY.md      # Ghi chú chạy trên máy cụ thể
└── results/
    ├── surrogate/  gemini/  openai/  anthropic/    # Mỗi backend một thư mục kết quả
        ├── data/      raw_results.json, aggregate_stats.json, experiment_results.xlsx
        ├── tables/    TableI–IV (CSV)
        ├── figures/   fig1–fig5 (PNG)
        ├── xacml/     policy.xml + request mẫu XACML 3.0
        ├── report/    Báo cáo Word
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

---

## 5. Các chỉ số đo

- **Mutation score** — tỉ lệ mutant bị kill (chỉ số chính).
- **Rule coverage** — tỉ lệ rule được "chạm tới" ít nhất một lần.
- **Requests-to-20-killed** — số request cần để giết 20 mutant (hiệu quả).
- **Phân bố quyết định** — tỉ lệ PERMIT/DENY/NOT_APPLICABLE (chống bộ test toàn DENY).
- **Phân tích survivor** — mutant nào sống sót ở ngân sách lớn (300 request).

---

## 6. Cài đặt

Yêu cầu: Python ≥ 3.10.

```bash
pip install lxml matplotlib openpyxl numpy python-dotenv
pip install google-genai      # nếu dùng Gemini
pip install openai            # nếu dùng GPT
pip install anthropic         # nếu dùng Claude
```

Tùy chọn (xuất báo cáo Word): cài Node.js rồi `npm install -g docx`. Thiếu bước này pipeline vẫn chạy bình thường, chỉ bỏ qua file `.docx`.

---

## 7. Cấu hình backend

Mở file `.env`, **chỉ để một dòng `ENV_FILE` không bị comment**:

```ini
# ENV_FILE=.env.surrogate
# ENV_FILE=.env.openai
ENV_FILE=.env.gemini
# ENV_FILE=.env.anthropic
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
| gemini | `GEMINI_API_KEY` | `gemini-3.5-flash` | https://aistudio.google.com/app/apikey |
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | https://platform.openai.com/api-keys |
| anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | https://console.anthropic.com/settings/keys |
| surrogate | (không cần) | — | — |

`LLM_MODEL` (tùy chọn) ghi đè model mặc định. Anthropic có thêm `ANTHROPIC_WORKSPACE_ID` nếu key không gắn workspace.

---

## 8. Chạy thực nghiệm

```bash
python main_experiment.py
```

Khuyến nghị quy trình:
1. Chạy `ENV_FILE=.env.surrogate` để kiểm tra pipeline offline (không tốn API).
2. Đặt `N_SEEDS=1` với backend LLM thật, xem log có `Parsed request count` hợp lý.
3. Đổi `N_SEEDS=5` và chạy đầy đủ.

Cấu hình thực nghiệm mặc định (trong `main_experiment.py`): 5 seed, ngân sách 40 request/phương pháp, ngưỡng hiệu quả 20 mutant.

Kết quả ghi vào `results/<backend>/`. Kết quả các backend không ghi đè lẫn nhau.

---

## 9. Backend "Surrogate" — lưu ý về tính trung thực khoa học

`HeuristicSurrogateLLMClient` **không phải LLM thật**. Nó thực thi tất định chiến lược prompt (kích hoạt rule, lật từng điều kiện, thử giá trị biên) để kiểm tra pipeline và phương pháp đo mà không cần API.

- Số liệu từ surrogate **không được trình bày như kết quả của LLM thật** trong bài báo.
- Phân tích survivor (`analyse_survivors`) và request mẫu XACML luôn dùng surrogate, bất kể backend đang chọn.

---

## 10. Cách xử lý lỗi LLM của pipeline

| Tình huống | Hành vi |
|---|---|
| Key sai/hết hạn (401), model sai (404), project bị chặn (403) | **Dừng ngay** với thông báo rõ ràng, không cho ra kết quả 0% gây hiểu nhầm |
| 429 (Gemini) | Thử lại 3 lần, chờ 20–90 giây |
| 5xx / quá tải | Thử lại với backoff lũy thừa |
| Output bị cắt cụt (hết token) | Cảnh báo và cứu lại các object JSON đã hoàn chỉnh |
| LLM trả thừa request | Cắt về đúng N để giữ ngân sách công bằng |
| Giá trị ngoài miền | Chuẩn hóa nếu chỉ lệch định dạng (`"3"`→3); còn lại thay bằng giá trị ngẫu nhiên và **đếm** tỉ lệ này |
| Seed mà LLM lỗi | Đánh dấu `failed`, **loại khỏi thống kê** kèm cảnh báo |

Thống kê chất lượng LLM (số request trả về, tỉ lệ giá trị sai, seed lỗi) được ghi trong `aggregate_stats.json` mục `llm_diagnostics`, để báo cáo minh bạch trong bài.

Để kết quả tái lập được, `seed` được truyền vào API Gemini/OpenAI; cả ba backend dùng chung `temperature = 0.7` (riêng Anthropic dùng mặc định của API).

---

## 11. Đầu ra chính

- **Table I** — 6 rule của policy; **Table II** — 51 mutant
- **Table III** — kết quả tổng hợp (mean ± std) cho 4 phương pháp
- **Table IV** — chi tiết từng seed × phương pháp
- **Fig 1** mutation score · **Fig 2** rule coverage · **Fig 3** tiến trình kill · **Fig 4** phân bố quyết định · **Fig 5** heatmap kill theo toán tử
- `experiment_results.xlsx` (6 sheet), `policy.xml` và request mẫu XACML 3.0, báo cáo Word

---

## 12. Bảo mật

- File `.env*` chứa API key đã được chặn bởi `.gitignore`. **Không commit, không nén chung để gửi.**
- Key đã từng lộ thì tạo lại ngay trên trang của nhà cung cấp.
- Repo chỉ nên commit `.env.surrogate` (không có key).

---

## 13. Khắc phục sự cố

| Triệu chứng | Nguyên nhân / Cách xử lý |
|---|---|
| `403 PERMISSION_DENIED ... project has been denied access` (Gemini) | Google project của key bị chặn. Tạo key trong **project mới** hoặc dùng tài khoản khác |
| `401 ... API key has expired` (OpenAI) | Key hết hạn. Tạo key mới, kiểm tra credit |
| `429 insufficient_quota` (OpenAI) | Hết credit thanh toán |
| `404 model not found` | Sai tên model. Kiểm tra `LLM_MODEL` |
| Kết quả LLM-Guided = 0% | Seed LLM lỗi; xem log tìm `FAILED` và `llm_diagnostics` |
| `ModuleNotFoundError: lxml` | `pip install lxml` |
| `Cannot find module 'docx'` | `npm install -g docx` (chỉ ảnh hưởng báo cáo Word) |
| Cảnh báo "automatic function calling (AFC)" | Vô hại, của SDK Gemini |

---

## 14. Tài liệu tham khảo phương pháp

- Martin & Xie (2007), *A fault model and mutation testing of access control policies*.
- Bertolino et al. (2010/2012), XACMUT và X-CREATE — sinh test và mutation cho XACML.
- Xu et al., tiêu chí rule-pair coverage.
