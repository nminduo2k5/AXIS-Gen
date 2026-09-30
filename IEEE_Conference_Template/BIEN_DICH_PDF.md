# Hướng dẫn biên dịch PDF — AXIS-Gen ICAI-FAI 2026

> File chính: `AXIS_Gen_ICAI_FAI_2026.tex`
> Output: `AXIS_Gen_ICAI_FAI_2026.pdf`
> Máy đã có: **MiKTeX** (pdflatex 25.1)

---

## Cách 1 — Biên dịch bằng lệnh (PowerShell / CMD)

Mở terminal, `cd` vào đúng folder này rồi chạy **3 lần** liên tiếp:

```powershell
cd "C:\Users\Admin\Desktop\AI-Native-TestGen-ICAI-FAI-2026_Full-Experiment\IEEE_Conference_Template"

pdflatex -interaction=nonstopmode conference_101719.tex
pdflatex -interaction=nonstopmode conference_101719.tex
pdflatex -interaction=nonstopmode conference_101719.tex
```

> Phải chạy **3 lần** để cross-references (số trang, số hình, số bảng) được resolve đúng.

Kết quả cuối cùng sẽ in:
```
Output written on AXIS_Gen_ICAI_FAI_2026.pdf (5 pages, xxxxxx bytes).
```

---

## Cách 2 — Dùng latexmk (tự động, chỉ 1 lệnh)

```powershell
cd "C:\Users\Admin\Desktop\AI-Native-TestGen-ICAI-FAI-2026_Full-Experiment\IEEE_Conference_Template"

latexmk -pdf -interaction=nonstopmode AXIS_Gen_ICAI_FAI_2026.tex
```

`latexmk` tự phát hiện cần chạy bao nhiêu lần và dừng khi ổn định.

---

## Cách 3 — Dùng Overleaf (không cần cài gì)

1. Zip các file sau vào 1 file `.zip`:
   - `AXIS_Gen_ICAI_FAI_2026.tex`
   - `IEEEtran.cls`
   - Folder `results_groq/` (chỉ cần 3 file `fig_backends.png`, `fig_eff_120b.png`, `fig_heat_120b.png`)
   - Folder `results_surrogate/figures/` (5 file `.png`)

2. Vào https://overleaf.com → **New Project** → **Upload Project**

3. Upload file `.zip` vừa tạo

4. Overleaf tự compile, xem PDF ngay trên trình duyệt

---

## Cấu trúc file trong folder

```
IEEE_Conference_Template/
├── AXIS_Gen_ICAI_FAI_2026.tex    ← file LaTeX nguồn (EDIT file này)
├── AXIS_Gen_ICAI_FAI_2026.pdf    ← output PDF (sau khi compile)
├── IEEEtran.cls                   ← class IEEE (không sửa)
├── results_surrogate/
│   └── figures/
│       ├── fig1_mutation_score.png
│       ├── fig2_rule_coverage.png
│       ├── fig3_efficiency.png
│       ├── fig4_decision_dist.png
│       └── fig5_operator_heatmap.png
└── BIEN_DICH_PDF.md               ← file này
```

---

## Các file tạm được sinh ra sau khi compile

pdflatex tự động tạo ra — **không cần quan tâm**, không commit lên git:

| File | Mục đích |
|------|----------|
| `.aux` | Cross-reference data |
| `.log` | Log biên dịch chi tiết |
| `.out` | Bookmark hyperlinks |

Xóa sạch file tạm:
```powershell
Remove-Item *.aux, *.log, *.out -ErrorAction SilentlyContinue
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Fix |
|-----|-------------|-----|
| `! Undefined control sequence \url` | Thiếu package | Đã fix: `\usepackage{url}` có trong preamble |
| `File 'fig1_mutation_score.png' not found` | Sai đường dẫn figures | Chạy từ đúng folder `IEEE_Conference_Template/` |
| `LaTeX Warning: Citation undefined` | Chưa chạy đủ lần | Chạy thêm 1 lần nữa |
| `! LaTeX Error: File 'IEEEtran.cls' not found` | Chạy sai thư mục | `cd` vào đúng folder trước khi compile |
| Package chưa cài (MiKTeX) | MiKTeX thiếu package | MiKTeX tự download khi compile lần đầu, cần internet |

---

## Trước khi submit — checklist

- [ ] Điền đúng email tác giả (dòng `email@institution.edu` trong `.tex`)
- [ ] Điền `[Authors]` cho 2 bibitem b12, b13 (arXiv papers)
- [ ] Điền nội dung Acknowledgment
- [ ] Kiểm tra số trang ≤ 8 (hiện tại: 5 trang)
- [ ] Xóa dòng `\thanks{Submitted to...}` nếu không cần footnote
