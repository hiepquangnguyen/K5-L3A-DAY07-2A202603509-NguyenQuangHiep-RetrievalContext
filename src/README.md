# Code Day 7 (`src/`)

Các module triển khai Day 7 (Agentic RAG & Thử thách nâng cao):

- `export_day6.py`: Xuất 210 chunk chiến lược `C` (`by_paragraph`) và bộ câu hỏi từ Qdrant Day 6 sang schema Day 7 (`starter/data/basic/`).
- `build_basic_benchmark.py`: Tạo bộ benchmark Phần cơ bản gồm 22 câu `dev` (20 có đáp án + 2 thiếu bằng chứng) và 8 câu `test` độc lập trên các trang chưa dùng ở Day 6 (6 có đáp án + 2 thiếu bằng chứng), kiểm chứng khớp nguyên văn `gold` với `starter/baseline.py`.
- `check_setup.py`: Kiểm tra cấu hình `.env`, kết nối Qdrant Day 6 và (với `--live`) kiểm chứng OpenAI API key + tương thích embedding/chat.
- `fetch_advanced_doc.py`: Tải và kiểm tra cấu trúc PDF >= 100 trang (`pymupdf`) cho Thử thách nâng cao.

## Lệnh chạy nhanh (từ thư mục gốc)

```powershell
# 1. Cài đặt môi trường ảo (nếu chưa có)
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Tạo .env từ .env.example và điền OPENAI_API_KEY
copy .env.example .env

# 3. Kiểm tra kết nối Qdrant Day 6 và OpenAI API
.\.venv\Scripts\python.exe src/check_setup.py --live

# 4. Xuất dữ liệu Day 6 và dựng benchmark Phần cơ bản
.\.venv\Scripts\python.exe src/export_day6.py
.\.venv\Scripts\python.exe src/build_basic_benchmark.py

# 5. Chạy kiểm chứng test và TF-IDF baseline trên dev
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe starter/baseline.py --chunks starter/data/basic/chunks.json --questions starter/data/basic/questions.json --split dev --output starter/results/basic-dev-tfidf
```


## Web UI/UX Demo Server (`demo_server.py` + `web/`)

Giao diện web tương tác (4 tab: **Live Studio**, **Basic Benchmark**, **Advanced 232p**, **Corpus Explorer**) phục vụ hỏi đáp trực tiếp trên cả hai kho dữ liệu (**FPT BCTN 2025 – 232 trang Multimodal Vision** và **MOET Day 6 – Baseline vs Chosen**) kèm trình xem trace, ảnh trang PDF theo vùng crop (`cached`/`full`/`top`/`bottom`/`left`/`right`) và trình duyệt toàn bộ corpus.

```powershell
# Khởi chạy server demo (mặc định http://127.0.0.1:8765)
.\.venv\Scripts\python.exe src/demo_server.py --host 127.0.0.1 --port 8765
```

> **Lưu ý:** Server mở kết nối `QdrantLocal` theo chế độ lazy lock theo tiến trình; không chạy đồng thời script CLI khác truy cập cùng thư mục `qdrant_db` trong khi server đang giữ kết nối.
