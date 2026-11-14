# Báo cáo Cá nhân Day 7 — Retrieval, Context & Agentic RAG

- **Sinh viên:** Nguyễn Quang Hiệp
- **MSSV:** `2A202603509`
- **Lớp:** `L3A`
- **Ngày hoàn thành:** `2026-11-14`
- **Phạm vi thực hiện:** Phần Cơ bản (85 điểm) + Thử thách Nâng cao (+15 điểm bonus)
- **Nhà cung cấp mô hình:** OpenAI API (`gpt-4.1-mini` cho chat, rerank & Vision; `text-embedding-3-small` `1536` chiều cho Phần cơ bản kết nối Day 6; `text-embedding-3-large` `3072` chiều cho Thử thách nâng cao)

---

## 1. Dữ liệu và Benchmark (Phần cơ bản)

### 1.1 Kết nối dữ liệu từ Day 6

- Hệ thống kết nối trực tiếp tới cơ sở dữ liệu Qdrant đã lập chỉ mục ở Day 6 tại `C:\Users\hiepq\VinUni\K5-L3A-DAY06-NguyenQuangHiep-2A202603509-DataIngestion\starter\qdrant_db`, collection `day06_moet_chunks`.
- Tập dữ liệu gồm **210 chunk** trích xuất bằng **chiến lược `C` (`by_paragraph`)** từ 2 văn bản quy chế của Bộ GD&ĐT (`tt08-2021`: **99 chunk**; `tt18-2021`: **111 chunk**), sử dụng vector `text-embedding-3-small` (`1536` chiều).
- Script [`src/export_day6.py`](../../src/export_day6.py) xuất toàn bộ 210 chunk và 20 câu hỏi Day 6 sang schema Day 7 (`starter/data/basic/`). Theo [`starter/data/basic/export_report.json`](../../starter/data/basic/export_report.json), **19/20** câu hỏi khớp đúng 1 chunk gold duy nhất và **1 câu (`q19`)** là multi-evidence kéo dài trên 2 chunk liên tiếp của trang 13 (`para:tt18-2021:p13:c000` và `c001`).

### 1.2 Cấu trúc benchmark `dev` và `test`

Script [`src/build_basic_benchmark.py`](../../src/build_basic_benchmark.py) xây dựng bộ dữ liệu đánh giá tách biệt tại `starter/data/basic/questions.json` và `starter/data/basic/cases.json`:

| Tập dữ liệu | Tổng câu | Có đáp án | Thiếu bằng chứng (`unanswerable`) | Ghi chú phân tách |
| --- | ---: | ---: | ---: | --- |
| `dev` | 22 | 20 (`q01`–`q20`) | 2 (`u01`, `u02`) | Bộ câu hỏi từ Day 6 + 2 bẫy thiếu thông tin (`u01` hỏi mức tiền học bổng khuyến khích...) dùng để dò tham số và phân tích lỗi |
| `test` | 8 | 6 (`q21`–`q26`) | 2 (`u03`, `u04`) | Câu hỏi độc lập đặt trên các trang tài liệu chưa sử dụng ở Day 6 + 2 bẫy kiểm chứng từ chối trả lời (`abstain`) |

- **Quy tắc gán `filter` (`cap_dao_tao`):** `src/build_basic_benchmark.py` chỉ gán bộ lọc metadata khi câu hỏi chỉ định rõ bậc đào tạo — `{"cap_dao_tao": "dai_hoc"}` cho `q13, q18, q21` (`TT08/2021`) và `{"cap_dao_tao": "tien_si"}` cho `q02, q15, q24` (`TT18/2021`); các câu còn lại và toàn bộ câu `unanswerable` giữ `filter = None` để kiểm thử truy xuất liên văn bản.

Kiểm chứng tương thích trước khi chạy bằng [`src/check_setup.py`](../../src/check_setup.py) (`--live`) xác nhận kết nối Qdrant Day 6 và OpenAI API (`OPENAI_API_KEY` trong `.env`, không đưa vào Git theo `.gitignore`).

---

## 2. Retrieval và Thử nghiệm (7 cấu hình)

Các cấu hình retrieval được triển khai trong [`src/retrievers.py`](../../src/retrievers.py) và [`src/rerankers.py`](../../src/rerankers.py), đánh giá đồng nhất qua [`src/eval_basic_retrieval.py`](../../src/eval_basic_retrieval.py). Kết quả lấy nguyên văn từ `starter/results/basic_retrieval/dev/comparison.json` và `starter/results/basic_retrieval/test/comparison.json`.

### 2.1 Bảng tổng hợp kết quả Retrieval

#### Trên tập `dev` (`n = 20` câu hỏi retrieval)

| Cấu hình | Hit@1 | Hit@3 | Hit@3 Strict | Context Hit | Avg Context Chars | Median (ms) | Total (s) | LLM / Embed Tokens | Chi phí (USD) |
| --- | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `tfidf_baseline_800` | 15/20 (0.75) | 19/20 (0.95) | 18 | 16/20 (0.80) | 582.0 | 1.23 | 0.026 | 0 | \$0.000000 |
| `bm25_800` | 18/20 (0.90) | 19/20 (0.95) | 18 | 18/20 (0.90) | 599.8 | 2.08 | 0.043 | 0 | \$0.000000 |
| `dense_qdrant_800` | 17/20 (0.85) | 19/20 (0.95) | 18 | 16/20 (0.80) | 627.6 | 324.02 | 8.717 | 1,497 embed | \$0.000030 |
| `hybrid_rrf_800` | 19/20 (0.95) | 20/20 (1.00) | 19 | 18/20 (0.90) | 600.6 | 17.69 | 0.357 | 0 | \$0.000000 |
| `hybrid_rrf_pack2400` | 19/20 (0.95) | 20/20 (1.00) | 19 | 19/20 (0.95) | 2,176.2 | 17.45 | 0.408 | 0 | \$0.000000 |
| **`hybrid_rerank_pack2400`** | **20/20 (1.00)** | **20/20 (1.00)** | **20** | **20/20 (1.00)** | 2,133.4 | 1,043.02 | 23.926 | 35,526 llm | \$0.015002 |
| `query_rewrite_pack2400` | 15/20 (0.75) | 19/20 (0.95) | 18 | 19/20 (0.95) | 2,200.4 | 1,660.41 | 33.737 | 2,990 llm / 1,435 embed | \$0.002364 |

#### Trên tập `test` (`n = 6` câu hỏi retrieval)

| Cấu hình | Hit@1 | Hit@3 | Hit@3 Strict | Context Hit | Avg Context Chars | Median (ms) | Total (s) | LLM / Embed Tokens | Chi phí (USD) |
| --- | :---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `tfidf_baseline_800` | 6/6 (1.00) | 6/6 (1.00) | 6 | 6/6 (1.00) | 586.2 | 1.33 | 0.008 | 0 | \$0.000000 |
| `bm25_800` | 6/6 (1.00) | 6/6 (1.00) | 6 | 6/6 (1.00) | 586.2 | 2.54 | 0.016 | 0 | \$0.000000 |
| `dense_qdrant_800` | 5/6 (0.8333) | 6/6 (1.00) | 6 | 5/6 (0.8333) | 627.3 | 359.38 | 2.223 | 524 embed | \$0.000010 |
| `hybrid_rrf_800` | 6/6 (1.00) | 6/6 (1.00) | 6 | 6/6 (1.00) | 586.2 | 20.16 | 0.235 | 0 | \$0.000000 |
| `hybrid_rrf_pack2400` | 6/6 (1.00) | 6/6 (1.00) | 6 | 6/6 (1.00) | 2,203.0 | 25.24 | 0.225 | 0 | \$0.000000 |
| **`hybrid_rerank_pack2400`** | **6/6 (1.00)** | **6/6 (1.00)** | **6** | **6/6 (1.00)** | 2,046.7 | 883.34 | 5.405 | 11,064 llm | \$0.004631 |
| `query_rewrite_pack2400` | 5/6 (0.8333) | 5/6 (0.8333) | 5 | 5/6 (0.8333) | 2,245.7 | 1,519.05 | 9.613 | 956 llm / 438 embed | \$0.000739 |

> **Ghi chú đo `embed_tokens = 0` ở các cấu hình hybrid:** Trong một phiên chạy [`src/eval_basic_retrieval.py`](../../src/eval_basic_retrieval.py), `DenseQdrantRetriever` lưu bộ nhớ đệm vector câu truy vấn (`self._query_cache`). Vì vậy `hybrid_rrf_800`, `hybrid_rrf_pack2400` và `hybrid_rerank_pack2400` tái sử dụng trực tiếp vector đã nhúng từ lượt `dense_qdrant_800` (`1,497` embed tokens ở `dev`; `524` embed tokens ở `test`) nên ghi nhận `0` token embed phát sinh thêm, trong khi `query_rewrite_pack2400` tạo chuỗi truy vấn viết lại mới nên gọi API nhúng riêng (`1,435` / `438` embed tokens).

### 2.2 Phân tích đánh đổi (Tradeoffs) & Chọn cấu hình tối ưu

1. **Kết hợp Dense + Lexical (Hybrid RRF):** Giảm sai sót đơn lẻ của từng phương pháp. Trên `dev`, `dense_qdrant_800` đạt `17/20` Hit@1 (`16/20` Context Hit), `bm25_800` đạt `18/20` Hit@1, nhưng ghép lại thành `hybrid_rrf_800` đẩy Hit@1 lên `19/20` và Hit@3 đạt `20/20`.
2. **Đóng gói ngữ cảnh kề (`pack2400`) + LLM Rerank (`hybrid_rerank_pack2400`):** Nới budget từ 800 lên 2.400 ký tự kèm kề trước/sau và sắp xếp lại bằng LLM giúp đạt **20/20 trên mọi chỉ số (`Hit@1`, `Hit@3`, `Hit@3 Strict`, `Context Hit` = 1.00)** ở `dev` và **6/6** ở `test`. Chi phí đánh đổi rất nhỏ (`$0.015` trên 20 câu `dev`, trung vị `1.04 s/câu`), hoàn toàn xứng đáng làm công cụ chính cho Agent.
3. **Tại sao `query_rewrite_pack2400` lại tụt điểm?** Trên văn bản pháp quy có thuật ngữ chuẩn xác ("thang điểm 10", "phản biện độc lập", số Điều/khoản), viết lại câu truy vấn bằng LLM làm loãng từ khóa gốc và khiến Hit@1 giảm còn `15/20` trên `dev` (và `5/6` ở `test`), đồng thời chậm nhất (`1,660.41 ms`).

### 2.3 Phân tích 2 lỗi điển hình trên tập `dev`

- **Trường hợp 1 — `q10` ("Theo thang điểm 10 trong quy chế đào tạo đại học, mức điểm nào được xếp loại Xuất sắc, Giỏi, Khá..."):**
  - Trên bảng đánh giá retrieval độc lập (`starter/results/basic_retrieval/dev`), `tfidf_baseline_800`, `bm25_800`, và `dense_qdrant_800` đều **trượt cả Top-1 lẫn Top-3 và `context_hit=False`**: câu hỏi gốc ít trùng từ khóa định lượng với bảng xếp loại ở `para:tt08-2021:p11:c000`, nên `dense_qdrant_800` chỉ đưa các đoạn nói chung về điểm học phần/thang điểm 4 vào Top-3 (`p10:c003`, `p10:c004`, `p09:c004`) và chỉ đóng gói được `p10:c003`, `p10:c004` trong `budget=800`.
  - Khi đi qua Agent Baseline (`basic_agent/dev/baseline`), Agent tự viết lại truy vấn ngắn (`"thang điểm 10 xếp loại Xuất sắc Giỏi Khá"`), giúp `dense_qdrant` đẩy `p11:c000` vào **rank 3 (`score=0.378287`)**, nhưng do hai kết quả đứng trước (`p10:c004`, `p07:c004`) đã chiếm gần hết `budget=800`, `p11:c000` tiếp tục bị cắt khỏi ngữ cảnh đóng gói, khiến Baseline lặp 4 lần rồi `abstain` sai.
  - Khi dùng `hybrid_rrf_pack2400`, RRF đưa `p11:c000` vào ngữ cảnh (`context_hit=True`, dù `hit1=False`); đến `hybrid_rerank_pack2400`, LLM rerank nhận diện đúng định nghĩa từng ngưỡng điểm và đẩy `p11:c000` lên vị trí số 1 (`hit1=True`).
- **Trường hợp 2 — `q19` (hai khoản nối tiếp thuộc Điều 16 đánh giá luận án tại `tt18-2021` trang 13: `c000` và `c001`):**
  - Dù `dense_qdrant_800`, `hybrid_rrf_800`, `hybrid_rrf_pack2400` và `query_rewrite_pack2400` đều có `hit1=True` (bắt được `p13:c001`), tất cả đều bị **`context_hit=False`** vì `budget=800` cắt phần thứ hai hoặc `p13:c000` không vào top ứng viên cùng lúc trước khi đóng gói.
  - Chỉ `hybrid_rerank_pack2400` giữ được cả `para:tt18-2021:p13:c001` lẫn đoạn kề `para:tt18-2021:p13:c000` trong packed context (`context_hit=True`), đạt `hit3_strict=20/20`.

---

## 3. Agent và Đánh giá (Baseline vs Chosen)

Triển khai trong [`src/basic_agent.py`](../../src/basic_agent.py) và chạy qua [`src/run_basic_agent.py`](../../src/run_basic_agent.py) với model `gpt-4.1-mini`:

- **Baseline Agent:** Chỉ có 1 công cụ `search` sử dụng `dense_qdrant_800` (budget 800 ký tự, không đọc chunk lân cận, không kiểm tra từ khóa chính xác).
- **Chosen Agent:** Trang bị 3 công cụ (`search_hybrid` dùng `hybrid_rerank_pack2400` + `budget=2400`, `search_exact_keywords` để tìm số điều/tên riêng, `read_adjacent` để mở chunk trước/sau trên cùng hoặc chuyển trang), cơ chế nhắc thử lại khi thiếu hoặc sai ID trích dẫn (`invalid_or_missing_citations`).

### 3.1 Bảng so sánh Baseline vs Chosen trên `dev` (`n = 22`) và `test` (`n = 8`)

Số liệu trích xuất trực tiếp từ `starter/results/basic_agent/{dev,test}/{baseline,chosen}/{summary,stats}.json`:

| Chỉ số đánh giá | `dev` Baseline | `dev` Chosen | `test` Baseline | `test` Chosen |
| --- | :---: | :---: | :---: | :---: |
| Tổng số case (`dev=22`, `test=8`) | 22 | 22 | 8 | 8 |
| Lỗi API / Provider (`provider_errors`) | 0 | 0 | 0 | 0 |
| Gọi công cụ truy xuất (`tool_use`) | 22/22 | 22/22 | 8/8 | 8/8 |
| Điểm câu trả lời có căn cứ (`grounded_answer`, `mean` 0–2) | `{0:14, 1:0, 2:6}`, **0.60** | `{0:0, 1:0, 2:20}`, **2.00** | `{0:4, 1:0, 2:2}`, **0.6667** | `{0:0, 1:0, 2:6}`, **2.00** |
| Tính hợp lệ của citation (`citation_validity`) | 6/6 | **22/22** | 2/2 | **6/6** |
| Độ phủ citation câu trả lời (`answer_citation_coverage`) | 6/6 | **20/20** | 2/2 | **6/6** |
| Từ chối đúng câu bẫy (`abstention`) | 2/2 | **2/2** | 2/2 | **2/2** |
| Từ chối sai (`false_abstention`) | **14/20** | **0/20** | **4/6** | **0/6** |
| Trạng thái (`answered / abstain / error`) | 6 / 16 / 0 | **20 / 2 / 0** | 2 / 6 / 0 | **6 / 2 / 0** |
| Trung bình số lượt tìm kiếm (`avg_searches`) | 3.09 | **1.32** (`27 hybrid, 2 exact`) | 2.88 | **1.50** (`9 hybrid, 3 exact`) |
| Độ trễ trung vị / P95 (`median_ms / p95_ms`) | 4,707.35 / 5,473.59 | 4,051.73 / 5,878.96 | 4,637.29 / 4,957.40 | 3,618.79 / 3,776.83 |
| Tổng thời gian chạy (`total_seconds`) | 95.30 s | 102.96 s | 32.32 s | 33.10 s |
| Tổng token Agent / Embed / Rerank | 70,729 / 1,325 / 0 | 75,759 / 935 / 47,839 | 24,171 / 283 / 0 | 34,675 / 303 / 15,430 |
| Tổng chi phí ước tính (`total_estimated_cost_usd`) | \$0.032586 | \$0.055098 | \$0.011182 | \$0.021958 |

### 3.2 Phân tích lỗi Baseline và lý do Chosen vượt trội

- **Vòng lặp cạn lượt tìm (`search_limit`) vì context budget quá chật (800 ký tự):** Ở các câu như `q01` và `q10`, Baseline chỉ nạp được 1–2 đoạn ngắn vào context (`p11:c004`, `p12:c001` cho `q01`; `p10:c004`, `p07:c004` cho `q10`). Khi thông tin mục tiêu bị đẩy ra khỏi 800 ký tự, Agent lặp lại 4 lần tìm kiếm (`['search', 'search', 'search', 'search', 'search_limit']`) và cuối cùng phải `abstain` (tạo ra **14/20** false abstention trên `dev` và **4/6** trên `test`).
- **Thiếu chunk nối tiếp & lỗi citation (`q19`):** Baseline chỉ giữ được `para:tt18-2021:p13:c001` (bỏ sót `c000`), rồi gặp `invalid_or_missing_citations` và bỏ cuộc (`abstain`).
- **Cải thiện ở Chosen:** Nhờ `hybrid_rerank_pack2400` lấy trúng ngữ cảnh ngay lượt đầu (trung bình giảm từ `3.09` xuống `1.32` lượt tìm trên `dev`, và `2.88` xuống `1.50` trên `test`), Chosen trả lời đúng **20/20** (`mean=2.0`) trên `dev` và **6/6** (`mean=2.0`) trên `test` với **0** false abstention và **100%** `abstention` chuẩn trên câu bẫy (`u01–u04`), đồng thời giảm trung vị thời gian phản hồi xuống `4.05 s` (`dev`) và `3.62 s` (`test`).

### 3.3 Bảng chi tiết từng câu trên `test` và điểm ngữ cảnh `context_grade` (0/1/2)

Điểm `context_grade` (0 = thiếu/lạc đề, 1 = thiếu ý hoặc cắt đầu/cuối đoạn, 2 = đủ bằng chứng trực tiếp trong ngữ cảnh đóng gói) được chấm đầy đủ trong `starter/results/basic_agent/{dev,test}/{baseline,chosen}/reviews.json`:

- **Trung bình `context_grade` trên `dev` (`n = 22` / `q01–q20`):** Baseline đạt **`1.2273`** toàn bộ (tổng `27/44` điểm trên 22 câu) và **`1.30`** trên 20 câu có đáp án (`26/40` điểm); Chosen đạt **`2.00`** tuyệt đối (`44/44` điểm toàn bộ và `40/40` điểm câu có đáp án).
- **Trung bình `context_grade` trên `test` (`n = 8` / `q21–q26`):** Baseline đạt **`1.125`** toàn bộ (tổng `9/16` điểm trên 8 câu) và **`1.3333`** trên 6 câu có đáp án (`8/12` điểm); Chosen đạt **`1.875`** toàn bộ (`15/16` điểm) và **`2.00`** trên 6 câu có đáp án (`12/12` điểm).

| Query ID | Loại câu | Baseline: Tool / Packed Context / Kết quả | Baseline: `ctx` / `ans` | Chosen: Tool / Packed Context / Kết quả | Chosen: `ctx` / `ans` | Lý do chênh lệch |
| --- | --- | --- | :---: | --- | :---: | --- |
| `q21` | `TT08` (`dai_hoc`) | 4×`search` -> `[p18:c005]` -> `abstain` sai | `0` / `0` | 1×`search_hybrid` -> `[p04:c001, p18:c005, p01:c003]` -> Trả lời + cite `p04:c001` | `2` / `2` | Rank 3 `p04:c001` bị đẩy khỏi `budget=800` ở Baseline |
| `q22` | `TT08` (tổng hợp) | 4×`search` -> `[p09:c002]` -> `abstain` sai | `1` / `0` | 1×`search_hybrid` -> `[p09:c002, p08:c002, p09:c000]` -> Trả lời + cite `p09:c002` | `2` / `2` | Baseline cắt mất đầu Điều 9 (`p09:c000`), Chosen giữ đủ kề |
| `q23` | `TT08` (tổng hợp) | 1×`search` -> `[p16:c002]` -> Trả lời + cite `p16:c002` | `2` / `2` | 1×`search_hybrid` -> `[p16:c002, p02:c004, p13:c002]` -> Trả lời + cite `p16:c002` | `2` / `2` | Đều bắt đúng giới hạn 25% tín chỉ trao đổi |
| `q24` | `TT18` (`tien_si`) | 1×`search` -> `[p05:c001]` -> Trả lời + cite `p05:c001` | `2` / `2` | 1×`search_hybrid` -> `[p05:c001, p04:c006, ...]` -> Trả lời + cite `p05:c001` | `2` / `2` | Đều có filter `cap_dao_tao=tien_si` và đủ định mức 7/5/3 & 0,5 |
| `q25` | `TT18` (tổng hợp) | 1×`search` -> `[p06:c003]` -> `abstain` (lỗi thiếu citation) | `2` / `0` | 1×`search_hybrid` -> `[p06:c003, p22:c002, p06:c002]` -> Trả lời + cite `p06:c003` | `2` / `2` | Baseline tìm đủ `p06:c003` nhưng bỏ `[chunk-id]` nên bị từ chối |
| `q26` | `TT18` (tổng hợp) | 4×`search` -> `[p17:c003]` -> `abstain` sai | `1` / `0` | 1×`search_hybrid` -> `[p17:c001, p17:c003, p17:c002]` -> Trả lời + cite `p17:c001` | `2` / `2` | Rank 2 `p17:c001` (số 07 và 03) vượt `budget=800` sau `p17:c003` |
| `u03` | Unanswerable | 4×`search` (`[p11:c004]`) -> `abstain` đúng | `1` / `True` | 2×`hybrid` + 3×`exact` (6 chunks đa trang) -> `abstain` đúng | `2` / `True` | Chosen rà soát cả từ khóa chính xác lẫn đa điều khoản trước khi `abstain` |
| `u04` | Unanswerable | 4×`search` (`[p12:c003, p15:c000]`) -> `abstain` đúng | `0` / `True` | 1×`search_hybrid` (`[p18:c001, p11:c002, p12:c003]`) -> `abstain` đúng | `1` / `True` | Không có quy định ký túc xá/nội trú trong TT08/2021; cả hai từ chối chuẩn |

---

## 4. Thử thách Nâng cao (+15 điểm bonus)

### 4.1 Tài liệu mục tiêu & Quá trình lập chỉ mục đa phương thức

- **Nguồn tài liệu:** Báo cáo thường niên Tập đoàn FPT 2025 — *"Làm chủ công nghệ chiến lược"* ([`starter/data/advanced/manifest.json`](../../starter/data/advanced/manifest.json)), tải chính thức từ `https://fpt.com/api/media/BCTN_2025_ad403f779b.pdf` (cổng nhà đầu tư FPT) qua [`src/fetch_advanced_doc.py`](../../src/fetch_advanced_doc.py).
  - Kích thước: **232 trang** (`12.345.876` bytes, SHA256: `9449a7251d126b635988974dbc72c2773c7fff39f4e5e858b5f7d3928b4e0e6c`).
  - Mật độ đa phương thức: **101 trang** chứa ảnh nhúng (**162 ảnh**), **209 trang** chứa `>= 15` đối tượng đồ họa vector (biểu đồ, sơ đồ, infographic, bảng tài chính), tổng cộng **565.033 ký tự** text lớp PDF (`231/232` trang có `>= 100` ký tự).
- **Pipeline trích xuất & lập chỉ mục ([`src/advanced_ingest.py`](../../src/advanced_ingest.py)):**
  - Chạy song song (`--workers 4`, `timeout=90.0s`) toàn bộ **232/232 trang** (`failed_vision_pages = []`): kết xuất từng trang thành PNG, gọi `gpt-4.1-mini` Vision trích xuất bảng Markdown, giá trị biểu đồ, sơ đồ kiến trúc, ảnh sự kiện, sau đó tạo `835` text chunks + `408` Vision/multimodal chunks = **1.243 chunks**.
  - Nhúng toàn bộ 1.243 chunks bằng **`text-embedding-3-large` (`3072` chiều)** vào Qdrant collection `fpt_bctn_2025` (`starter/data/advanced/qdrant_db`).
  - Chi phí lập chỉ mục ghi nhận trong [`starter/data/advanced/ingest_manifest.json`](../../starter/data/advanced/ingest_manifest.json): Vision `1.125.391` tokens (`$0.594525`) + Embedding `553.976` tokens (`$0.072017`) = **`$0.666542`** (`embed_seconds = 18.9s`).

### 4.2 Kiến trúc Multimodal Agent & Tập benchmark tách biệt

- **Thiết kế Agent ([`src/advanced_agent.py`](../../src/advanced_agent.py)):**
  - 4 công cụ thực thi:
    1. `search_multimodal_hybrid`: Qdrant `text-embedding-3-large` + BM25 RRF + LLM rerank (`gpt-4.1-mini`), có tùy chọn lọc theo trang/loại chunk.
    2. `search_exact_keywords`: BM25 thuần cho mã số kế toán (`Mã 10`, `Mã 270`, `Mã 300`), số nghị quyết, tên riêng, URL.
    3. `read_page_full`: Đọc trọn bộ text chunk của một trang và trang liền kề khi đoạn trích bị cắt ngang.
    4. `inspect_page_image`: Mở trực tiếp ảnh PNG của trang và gọi Vision (`gpt-4.1-mini`) trả lời câu hỏi định hướng trên 5 vùng (`full|top|bottom|left|right`).
  - Đóng gói ngữ cảnh `budget = 8000` ký tự kèm cơ chế ưu tiên chunk `:v` (`+0.02` và `*1.22`), phạt đa dạng hóa theo trang (`-0.12 * pg_rank`) để không bị 1 trang dài đẩy trang kia ra trong câu `multi_hop`, và kiểm tra citation nội dòng.
- **Benchmark tách biệt theo trang ([`src/build_advanced_benchmark.py`](../../src/build_advanced_benchmark.py)):**
  - Bộ `dev`: **18 câu** (3 câu mỗi loại × 6 nhóm: `text`, `table`, `chart`, `image_diagram`, `multi_hop`, `unanswerable`).
  - Bộ `test` độc lập: **30 câu** (`6 table`, `5 image_diagram`, `5 chart`, `4 multi_hop`, `4 text`, `6 unanswerable`).
  - Kiểm chứng tự động qua `validate_advanced_benchmark`: giao tập trang nguồn giữa `dev` và `test` bằng rỗng (`page_overlap = []`), mọi câu `unanswerable` đều đã được rà toàn bộ 232 trang text + vision cache xác nhận không có dữ liệu.

### 4.3 Quá trình tinh chỉnh lặp trên tập `dev` (`v1 -> v5`)

Toàn bộ chỉnh sửa prompt, ngân sách ngữ cảnh, xử lý ảnh và kiểm tra citation nội dòng được thực hiện **dựa trên phát hiện ở `dev` và kiểm chứng hồi quy** (lưu lịch sử `v1_*`, `v2_*`, `v3_*` và bản hiện tại trong `starter/results/advanced/dev/`):

| Phiên bản | Thay đổi kỹ thuật chính | Answered / Abstain / Error | Retry Citation | Độ trễ trung vị | Chi phí `dev` | Kết quả & phát hiện lỗi |
| --- | --- | :---: | :---: | ---: | ---: | --- |
| `v1` | `budget=4500`, thứ tự chữ cái `:t < :v`, prompt cơ bản | 15 / 3 / 0 | **10/18** | 11,457.51 ms | \$0.229431 | `adv_dev_mh02` chỉ giữ chunk trang 138, đẩy mất trang 152 vì budget 4500 chật; `:t` đè `:v`; 10/18 case phải retry citation |
| `v2` | Tăng `budget=8000`, bonus `:v` (`+0.02`/`1.22`), phạt đa trang (`-0.12*rank`), nhắc rõ format `[chunk-id]` | 15 / 3 / 0 | **1/18** | 10,273.60 ms | \$0.189195 | Sửa xong `adv_dev_mh02` (đủ `p138+p152`); **còn lỗi `adv_dev_mh03` (`p082-083`)**: bỏ qua `inspect_page_image` trên card đồ họa & bảng nhiều cột nên ghép nhầm `"07 cuộc họp trực tiếp"` |
| `v3` | Bắt buộc dùng `inspect_page_image` cả cho card KPI/bảng nghị quyết nhiều cột | 14 / 4 / 0 | 1/18 | 9,610.63 ms | \$0.184474 | Sửa `adv_dev_mh03` đạt 100%; nhưng tạo **regression ở `adv_dev_t02` (`p057`)**: Vision `150 DPI` full-page bỏ sót đoạn văn chữ nhỏ phía trên card KPI và trả lời "không có % kế hoạch", khiến Agent phủ định chunk `p057:t004` hợp lệ và `abstain` sai |
| `v4` (Chốt khóa ban đầu) | Kết xuất ảnh live **`full=200 DPI`**, vùng crop **`240 DPI`**; cập nhật prompt Vision & Agent **không phủ định dữ kiện văn xuôi đã có rõ trong chunk `:t`** khi ảnh zoom toàn trang chưa đọc ra chữ nhỏ | **15 / 3 / 0** | 1/18 | 10,722.81 ms | \$0.195891 | **`18/18` (`100%`) chính xác trên `dev`** (`15/15` có đáp án đúng key facts + citation đúng trang nguồn; `3/3` unanswerable từ chối đúng) |
| **`v5` (Gia cố citation & loại spoiler benchmark)** | Siết `_extract_invalid_inline_citations` trong `submit_final_answer` (chặn tag `[chunk-id]` viết trong văn xuôi nhưng bị đẩy khỏi `packed_context_ids`), bổ sung Rule 4–5 kiểm tra trang tổng hợp/ưu tiên ID `:v`, và bỏ gợi ý đáp án (`FPT Semiconductor`, `NVIDIA`) khỏi câu hỏi `adv_test_mh04` | **15 / 3 / 0** | 0/18 (khớp sạch inline & list) | **10,661.78 ms** (P95 `18,167.84 ms`, `219.97 s`) | **\$0.187999** (`avg_tool_calls=2.33`) | Giữ vững **`18/18` (`100.0%`) trên `dev`**: `adv_dev_mh02` trích xuất chính xác `7.121.722` lượt học và `94%`; `3/3` unanswerable từ chối chuẩn |

### 4.4 Kết quả đánh giá trên tập `test` (`n = 30`) — `v4` vs `v5`

- **Lần chạy chốt khóa `v4-locked` (`src/run_advanced.py --split test` trước chỉnh sửa `v5`):**
  - `answered = 24`, `abstain = 6`, `error = 0`, `avg_tool_calls = 3.2` (`search_multimodal_hybrid: 40`, `inspect_page_image: 35`, `search_exact_keywords: 11`, `read_page_full: 10`), trung vị `11,739.43 ms`, P95 `21,919.34 ms`, `387.54 s`, tổng chi phí **`$0.434609`** (`870,812` agent + `1,434` embed + `145,290` rerank tokens).
  - Kết quả chấm nghiêm ngặt: **`27 / 30` (`90.0%`)** (`text: 4/4`, `table: 6/6`, `chart: 5/5`, `image_diagram: 4/5`, `multi_hop: 2/4`, `unanswerable: 6/6`; 3 lỗi là `id03`, `mh01`, `mh02`; riêng `id04` đủ nội dung và `citations[]` trang 111 nhưng còn rò rỉ 2 tag nội dòng `p118` chưa bị chặn; `mh04` khi đó vẫn còn từ khóa gợi ý trong câu hỏi).
- **Lần chạy cuối `v5` (lưu hiện hành trong `starter/results/advanced/test/{traces,stats,reviews,summary}.json`):**
  - Sau khi gia cố kiểm tra citation nội dòng + loại bỏ gợi ý ở `adv_test_mh04`, thực thi lại một lần `.\.venv\Scripts\python.exe src/run_advanced.py --split test` (không tinh chỉnh prompt theo lỗi `test` theo `RULES.md`):
  - **Thống kê thực thi (`test`, `n = 30`):** `answered = 25`, `abstain = 5`, `error = 0`; `avg_tool_calls = 3.27` (`search_multimodal_hybrid: 47`, `inspect_page_image: 38`, `search_exact_keywords: 7`, `read_page_full: 6`); độ trễ trung vị `12,494.85 ms`, P95 `24,851.11 ms`, tổng thời gian `458.18 s`; `979,094` agent tokens (`$0.419214`) + `1,692` embed tokens (`$0.000220`) + `170,068` rerank tokens (`$0.069926`) = **`$0.489360`**.

#### Tổng hợp theo hạng mục trên `test` (`v5` hiện hành trong `starter/results/advanced/test/summary.json`)

| Hạng mục | `v4-locked` | `v5` (Hiện hành) | Tỷ lệ (`v5`) | Ghi chú đối chiếu trực tiếp với báo cáo gốc & traces |
| --- | :---: | :---: | :---: | --- |
| `text` | 4 / 4 | **4 / 4** | 100.0% | Đúng mọi tỷ lệ nhân sự & mục tiêu 2026 (`t01`), NQ57 (`t02`), Broadband & bản quyền EPL (`t03`), CSR & Quỹ Vì Cộng đồng (`t04`) |
| `table` | 6 / 6 | **6 / 6** | 100.0% | Chính xác tới từng cổ phiếu/tỷ lệ cổ đông (`tbl01–02`), hệ số tài chính (`tbl03`), giải ngân dự án (`tbl04`), phát thải KNK (`tbl05`), và số liệu KQKD chính xác từng đồng (`tbl06`: `70.112.825.100.710`, `25.888.529.512.413`, `13.043.632.833.797`) |
| `chart` | 5 / 5 | **5 / 5** | 100.0% | Đọc đúng giá trị mọi cột/đường/biểu đồ tròn từ ảnh: DT & EPS 5 năm (`c01`), cơ cấu địa lý (`c02`), thị trường CNTT nước ngoài & `-0,8%` APAC (`c03`), chuyển đổi số (`c04`), AI/Data Analytics & năng suất (`c05`) |
| `image_diagram` | 4 / 5 | **4 / 5** | 80.0% | Đúng `id01`, `id02`, `id04` (**`v5` hết hẳn lỗi rò rỉ tag `p118`**, chỉ trích dẫn `p111:t000`, `p111:v000`), `id05`; **còn 1 lỗi ở `adv_test_id03`** (`p080`) |
| `multi_hop` | 2 / 4 | **2 / 4** | 50.0% | Đúng hoàn toàn `mh03` và **`mh04` bản khó không gợi ý** (tự tìm & ghép đủ `p030` + `p068`); **2 lỗi ở `mh01` và `mh02`** |
| `unanswerable` | 6 / 6 | **5 / 6** | 83.33% | Từ chối chuẩn `u01, u02, u03, u05, u06`; **`v5` phát sinh 1 lỗi ở `adv_test_u04`** (trả lời không có lương bình quân nhưng vẫn gọi `abstain=False` kèm doanh thu/người `93 triệu`) |
| **Tổng cộng** | **27 / 30 (90.0%)** | **26 / 30** | **86.67%** | `v5` siết chặt citation nội dòng (`id04` sạch) và bỏ spoiler `mh04`; giữ trung thực `26/30` trên bản `v5` và `27/30` trên `v4-locked` |

#### Bảng đánh giá đầy đủ 30 câu `test` (`v5`, đối chiếu `starter/results/advanced/test/reviews.json`)

| Query ID | Nhóm | Trang nguồn | Trang trích dẫn | Trạng thái (`status`) | Chấm nghiêm ngặt | Tóm tắt evidence & lý do |
| --- | --- | :---: | :---: | :---: | :---: | --- |
| `adv_test_t01` | `text` | `[8]` | `[8, 55]` | `answered` | **PASS** | Đủ tỷ lệ nhân sự & mục tiêu 2026 (`43,9%`, `53,2%`, `15,8%`, `15,0%`, `18,4%`, `24,4%`) |
| `adv_test_t02` | `text` | `[49]` | `[49]` | `answered` | **PASS** | Nêu đủ `Top 3 ASEAN`, `Top 50`, đầu tư `3% ngân sách` theo NQ57 |
| `adv_test_t03` | `text` | `[70]` | `[70]` | `answered` | **PASS** | Đúng `10.007` tỷ, `+13,7%`, bản quyền EPL `2025-2026` đến `2030-2031` |
| `adv_test_t04` | `text` | `[154]` | `[154]` | `answered` | **PASS** | Đúng `72,8` tỷ, `97.300` người, từ `2014`, `25,6` tỷ chương trình Ánh sáng |
| `adv_test_tbl01` | `table` | `[32]` | `[32]` | `answered` | **PASS** | Chính xác `254`, `1.241.598.059`, `72,88%`, `96.585.637` (`5,67%`), `749.771.552` (`44,01%`) |
| `adv_test_tbl02` | `table` | `[33]` | `[33]` | `answered` | **PASS** | Chính xác `Trương Gia Bình` (`117.347.966` / `6,89%`), `SCIC` (`96.585.637`), `Bùi Quang Ngọc` (`62.449.511` / `3,67%`) |
| `adv_test_tbl03` | `table` | `[59]` | `[59]` | `answered` | **PASS** | Đúng các hệ số tài chính (`41,8%/48,2%`, `47,1/49,0`, `22,9%/21,9%`) |
| `adv_test_tbl04` | `table` | `[75]` | `[75]` | `answered` | **PASS** | Đúng 3 dự án & tiến độ giải ngân (`6.000/2.823/47,1%`, `2.500/1.751/70,0%`, `2.500/749/30,0%`) |
| `adv_test_tbl05` | `table` | `[146]` | `[146]` | `answered` | **PASS** | Chính xác phát thải Scope 1, 2 & tổng (`13.217/4.826/4.383`, `109.696/138.748/123.569`, `122.913/143.574/127.952`) |
| `adv_test_tbl06` | `table` | `[169]` | `[169]` | `answered` | **PASS** | Đúng từng đồng KQKD (`70.112.825.100.710`, `25.888.529.512.413`, `13.043.632.833.797`) |
| `adv_test_c01` | `chart` | `[10]` | `[10]` | `answered` | **PASS** | Đọc đúng chuỗi DT (`35.657..70.113`) & EPS (`2.367..5.216`) 5 năm |
| `adv_test_c02` | `chart` | `[27]` | `[27]` | `answered` | **PASS** | Đúng cơ cấu 5 khu vực (`48,9%`, `22,8%`, `11,6%`, `11,1%`, `5,6%`) |
| `adv_test_c03` | `chart` | `[62]` | `[62]` | `answered` | **PASS** | Đủ 4 thị trường (`15.452`, `8.163`, `8.173`, `3.593`) và Châu Á–TBD giảm `0,8%` |
| `adv_test_c04` | `chart` | `[63]` | `[63]` | `answered` | **PASS** | Đúng chuỗi doanh thu CĐS (`5.522..16.751`), tăng trưởng `16,8%`, Cloud/AI `47,3%` |
| `adv_test_c05` | `chart` | `[65]` | `[65]` | `answered` | **PASS** | Đọc đúng `980, 1.221, 1.856, 2.612`, năm 2025 `1.116`, tăng năng suất `7,6%` |
| `adv_test_id01` | `image_diagram` | `[31]` | `[31]` | `answered` | **PASS** | Đọc đúng sơ đồ mô hình quản trị & đơn vị thành viên trên trang 31 |
| `adv_test_id02` | `image_diagram` | `[66]` | `[66]` | `answered` | **PASS** | Đọc đúng URL `https://nq57.mst.gov.vn/products`, `14 -> 18`, ông Nguyễn Hoàng Minh |
| `adv_test_id03` | `image_diagram` | `[80]` | `[80]` | `answered` | **FAIL** | Chỉ nêu Tổng Giám đốc là người đại diện theo pháp luật, **bỏ sót Chủ tịch HĐQT** |
| `adv_test_id04` | `image_diagram` | `[111]` | `[111]` | `answered` | **PASS** | Đủ 5 chuẩn ISO + `1,1 triệu giờ` AI; trích dẫn sạch `p111:t000, v000` (hết tag `p118` ở `v5`) |
| `adv_test_id05` | `image_diagram` | `[155]` | `[155]` | `answered` | **PASS** | Đọc đúng phông sân khấu `FPT Young Talents K24`, `2023-2025` & 4 mốc học bổng |
| `adv_test_mh01` | `multi_hop` | `[166, 167, 168]` | `[165, 167, 168]` | `answered` | **FAIL** | Đủ 3 số liệu (`88.141...`, `44.393...`, `43.748...`) & phép kiểm đúng, nhưng **gán sai `p165:t000` cho Mã 270** (nằm ở `p166`) và không inspect `p166` |
| `adv_test_mh02` | `multi_hop` | `[170, 171]` | `[170, 171]` | `answered` | **FAIL** | Đúng 2 con số 14 chữ số, ngày ký `18/03/2026` & PTGĐ Nguyễn Thế Phương, nhưng **tính nhẩm `90,1%` thay vì `90,24%`** |
| `adv_test_mh03` | `multi_hop` | `[144, 147]` | `[144, 147]` | `answered` | **PASS** | Đúng năng lượng tái tạo `842.749 / 187.453.013` (`0,45%`) & mục tiêu 2026 |
| `adv_test_mh04` | `multi_hop` | `[30, 68]` | `[30, 68]` | `answered` | **PASS** | Benchmark 2-hop đã bỏ gợi ý: tự tìm & ghép đủ `p030` (`FPT Semiconductor`, `NVIDIA`) + `p068` (`Restar`, `10 triệu chip`, `36/38`, `111 tỷ tokens`) |
| `adv_test_u01` | `unanswerable` | `[]` | `[]` | `abstain` | **PASS** | Từ chối đúng (không có số lượng GPU NVIDIA H100/H200 cụ thể) |
| `adv_test_u02` | `unanswerable` | `[]` | `[]` | `abstain` | **PASS** | Từ chối đúng (không có vốn đầu tư riêng khu HCM02) |
| `adv_test_u03` | `unanswerable` | `[]` | `[]` | `abstain` | **PASS** | Từ chối đúng (không có % thị phần thuê bao Internet cố định toàn quốc 2025) |
| `adv_test_u04` | `unanswerable` | `[]` | `[65, 137]` | `answered` | **FAIL** | Nói rõ không công bố lương bình quân nhưng vẫn gọi `abstain=False` kèm 3 trích dẫn (doanh thu/người `93 triệu`) |
| `adv_test_u05` | `unanswerable` | `[]` | `[]` | `abstain` | **PASS** | Từ chối đúng (không có số cửa hàng Long Châu mở mới tại BCTN FPT 2025) |
| `adv_test_u06` | `unanswerable` | `[]` | `[]` | `abstain` | **PASS** | Từ chối đúng (chưa có mức cổ tức cụ thể năm 2026) |

### 4.5 Phân tích lỗi trên `test` (`v5`), rà soát trace & giới hạn cải thiện

1. **`adv_test_id03` (Trang 80 — Mô hình quản trị & người đại diện theo pháp luật) — FAIL:**
   - Vision ở `inspect_page_image(80)` đọc phần mô tả dưới nhãn *"Người đại diện theo pháp luật của Tập đoàn"* nhưng ghép nhầm mô tả điều hành công tác HĐQT (thuộc **Chủ tịch HĐQT**) sang **Tổng Giám đốc**, bỏ sót Chủ tịch HĐQT là người đồng đại diện theo pháp luật. Nguyên nhân do sơ đồ trang 80 bố trí hộp chữ song song có badge vai trò nhỏ tách rời khối giải thích.
2. **`adv_test_mh01` (Trang 166–168 — Bảng cân đối kế toán hợp nhất) — FAIL:**
   - Agent thu thập đúng cả 3 giá trị (`Mã 270 = 88.141.991.634.625`, `Mã 300 = 44.393.950.887.086`, `Mã 400 = 43.748.040.747.539` từ `p167–168`) và kiểm chứng chính xác `44.393... + 43.748... = 88.141...`. Tuy nhiên Agent không gọi `inspect_page_image(166)` mà gán trích dẫn `[adv:fpt2025:p165:t000]` (vốn chỉ kết thúc ở Tài sản ngắn hạn) cho Tổng tài sản `Mã 270` nằm ở trang 166.
3. **`adv_test_mh02` (Trang 170–172 — Tỷ lệ chuyển đổi dòng tiền trên LNST) — FAIL (theo tiêu chuẩn số học nghiêm ngặt):**
   - Hệ thống trích xuất chính xác cả hai con số 14 chữ số (`LNST Mã 60 = 11.232.339.450.734 VNĐ`, `Lưu chuyển tiền thuần HĐKD Mã 20 = 10.136.043.915.911 VNĐ`) và người ký (`Phó Tổng Giám đốc Nguyễn Thế Phương`, ngày `18/03/2026`). Tuy nhiên khi tự tính nhẩm phép chia `10.136.043.915.911 / 11.232.339.450.734` (`90,2398%`), LLM viết `"khoảng 90,1%"` (lệch `0,14` điểm % so với `90,24%`).
4. **`adv_test_u04` (Câu bẫy tiền lương bình quân toàn CBNV 2025) — FAIL (hành vi abstention ở `v5`):**
   - Mặc dù trong nội dung trả lời Agent thừa nhận báo cáo không công bố mức lương bình quân, Agent vẫn gọi `submit_final_answer(abstain=False)` để báo thêm chỉ số *"doanh thu/người ~93 triệu đồng/tháng"* kèm 3 citation `[p065:t002, p065:v000, p137:t003]`, làm sai hợp đồng `expected_behavior: abstain`.
5. **Điểm tiến bộ xác nhận ở `v5` (`adv_test_id04` & `adv_test_mh04`):**
   - Bộ lọc `_extract_invalid_inline_citations` đã ngăn triệt để việc viết tag `[chunk-id]` bị evicted khỏi `packed_context_ids` (khắc phục lỗi rò rỉ tag `p118` ở `adv_test_id04` của `v4`). Đồng thời `adv_test_mh04` sau khi bỏ gợi ý `FPT Semiconductor` / `NVIDIA` khỏi đề vẫn tự tìm đủ cả trang `p030` lẫn `p068` và tổng hợp chính xác mọi `key_facts`.
6. **Giới hạn & hướng nâng cấp tiếp theo:**
   - Bổ sung công cụ **`python_calculator`** để thực thi phép chia/cộng tài chính xác định, loại bỏ hoàn toàn lỗi tính nhẩm như `adv_test_mh02`.
   - Khi câu hỏi nêu một khoảng trang báo cáo tài chính liên tiếp (`166–168`), hoặc có bước trích chỉ tiêu đầu kỳ/tổng cộng trên trang liền kề, nên kiểm tra đầy đủ ảnh/bảng từng trang trong khoảng đó (thay vì suy diễn từ trang `165` liền trước), giải quyết lỗi như `adv_test_mh01`.
   - Ràng buộc rule kiểm chứng `abstain=True` bắt buộc mỗi khi dữ kiện cốt lõi của câu hỏi không tồn tại trong toàn văn, không chuyển sang trả lời chỉ số thay thế gần nghĩa (`adv_test_u04`).

---

## 5. Giao diện Demo & Lệnh chạy tái lập

### 5.1 Kiến trúc Web Demo Interactive Studio ([`src/demo_server.py`](../../src/demo_server.py) & [`src/web/`](../../src/web/index.html))

- **Backend HTTP (`ThreadingHTTPServer`):** [`src/demo_server.py`](../../src/demo_server.py) khởi tạo `DemoAppContext` kết nối trực tiếp hai Qdrant store (`basic` + `advanced`) và cung cấp các API:
  - `GET /`, `GET /assets/*` phục vụ giao diện tĩnh tại [`src/web/index.html`](../../src/web/index.html), [`src/web/app.js`](../../src/web/app.js), [`src/web/styles.css`](../../src/web/styles.css).
  - `GET /api/overview`: tổng hợp dữ liệu corpus, bảng so sánh 7 cấu hình retrieval, benchmark `basic_agent`, tiến trình `advanced`, và trace đầy đủ.
  - `GET /api/chunks`, `GET /api/chunk/<id>`: duyệt/lọc chunk theo `kb`, `page`, `source_id`, `modality`.
  - `GET /api/advanced/page/<page_no>.png?region=cached|full|top|bottom|left|right`: trả ảnh PNG gốc hoặc crop live độ phân giải cao của từng trang BCTN FPT 2025.
  - `POST /api/ask`: thực thi live query trên `advanced` hoặc `basic` (chế độ `chosen`, `baseline`, hoặc `compare`), trả kèm telemetry token/chi phí và event trace từng bước.

### 5.2 Lệnh chạy tái lập & Danh mục kiểm chứng

Chạy từ thư mục gốc dự án bằng PowerShell (`.venv` đã cài đặt các gói trong [`requirements.txt`](../../requirements.txt) và cấu hình `OPENAI_API_KEY` trong `.env`):

```powershell
# 0. Kiểm tra cấu hình và kết nối API / Qdrant Day 6
.\.venv\Scripts\python.exe src/check_setup.py --live

# 1. Chạy toàn bộ bộ unit test (35 tests)
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. Phần cơ bản: Xuất dữ liệu Day 6, dựng benchmark & đánh giá 7 cấu hình retrieval
.\.venv\Scripts\python.exe src/export_day6.py
.\.venv\Scripts\python.exe src/build_basic_benchmark.py
.\.venv\Scripts\python.exe src/eval_basic_retrieval.py --split dev
.\.venv\Scripts\python.exe src/eval_basic_retrieval.py --split test

# 3. Phần cơ bản: Đánh giá Agent Baseline vs Chosen trên dev và test
.\.venv\Scripts\python.exe src/run_basic_agent.py --split dev --mode both
.\.venv\Scripts\python.exe src/run_basic_agent.py --split test --mode both

# 4. Thử thách nâng cao: Tải PDF, ingest 232 trang, dựng benchmark & đánh giá dev/test
.\.venv\Scripts\python.exe src/fetch_advanced_doc.py --output starter/data/advanced/raw/FPT_BCTN_2025.pdf --inspect-only
.\.venv\Scripts\python.exe src/advanced_ingest.py --workers 4
.\.venv\Scripts\python.exe src/build_advanced_benchmark.py
.\.venv\Scripts\python.exe src/run_advanced.py --split dev
.\.venv\Scripts\python.exe src/run_advanced.py --split test

# 5. Khởi chạy Web UI Interactive Demo (mặc định http://127.0.0.1:8765)
.\.venv\Scripts\python.exe src/demo_server.py --host 127.0.0.1 --port 8765
```

**Xác minh kiểm thử cuối cùng:** Lệnh `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` thực thi sau tất cả các chỉnh sửa mã nguồn cho kết quả **`Ran 35 tests — OK`** (`0` failures, `0` errors, `0` skipped), gồm `test_basic_setup.py`, `test_baseline.py`, `test_fixture.py`, `test_retrievers.py`, `test_rerankers.py`, `test_basic_agent.py`, `test_agentic.py`, `test_agent_metrics.py`, `test_advanced_ingest.py`, `test_advanced_agent.py`, và `test_demo_server.py`.
