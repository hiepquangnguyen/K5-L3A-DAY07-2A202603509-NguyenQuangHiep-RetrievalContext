"""Interactive UI/UX Demo Server for Day 7 Agentic RAG & Multimodal 232p FPT Studio."""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from advanced_agent import render_page_region_png_bytes, run_advanced_agent  # noqa: E402
from basic_agent import run_baseline_starter_agent, run_multitool_agent  # noqa: E402
from rerankers import OpenAIRerankRetriever  # noqa: E402
from retrievers import Bm25Retriever, DenseQdrantRetriever, HybridRrfRetriever  # noqa: E402

WEB_DIR = ROOT / "src" / "web"
DEFAULT_DAY6_QDRANT = (
    ROOT.parent / "K5-L3A-Day06-NguyenQuangHiep-2A202603509-DataIngestion" / "starter" / "qdrant_db"
).resolve()

PRESET_QUESTIONS: list[dict[str, Any]] = [
    {
        "id": "preset_adv_tbl06",
        "kb": "advanced",
        "mode": "advanced",
        "tag": "Table (BCTC từng đồng)",
        "question": "Theo Báo cáo kết quả hoạt động kinh doanh hợp nhất năm 2025 (Mẫu B 02 - DN/HN), Doanh thu thuần (Mã 10), Lợi nhuận gộp (Mã 20) và Tổng lợi nhuận kế toán trước thuế (Mã 50) chính xác đến từng đồng là bao nhiêu?",
    },
    {
        "id": "preset_adv_c03",
        "kb": "advanced",
        "mode": "advanced",
        "tag": "Chart (Biểu đồ tròn)",
        "question": "Theo biểu đồ tròn Doanh thu lĩnh vực dịch vụ CNTT cho thị trường nước ngoài năm 2025, doanh thu, tỷ trọng và tốc độ tăng/giảm của 4 thị trường Nhật Bản, Mỹ, Châu Á-Thái Bình Dương và Châu Âu là bao nhiêu; thị trường nào tăng trưởng âm?",
    },
    {
        "id": "preset_adv_id02",
        "kb": "advanced",
        "mode": "advanced",
        "tag": "Image / UI Screen",
        "question": "Quan sát ảnh chụp giao diện cổng thông tin và ảnh chân dung trong mục Dịch vụ CNTT trong nước: thanh địa chỉ hiển thị URL gì, công bố thêm bao nhiêu sản phẩm Made by FPT (nâng tổng số lên bao nhiêu), và nhân vật trong ảnh là ai?",
    },
    {
        "id": "preset_adv_mh03",
        "kb": "advanced",
        "mode": "advanced",
        "tag": "Multi-hop ESG",
        "question": "Đối chiếu Chính sách môi trường và Bảng tiêu thụ điện năm 2025: điện từ năng lượng tái tạo đạt bao nhiêu kWh trên tổng tiêu thụ điện (tỷ lệ %), và mục tiêu năm 2026 cho năng lượng tái tạo cùng tỷ lệ đèn LED là bao nhiêu?",
    },
    {
        "id": "preset_adv_u01",
        "kb": "advanced",
        "mode": "advanced",
        "tag": "Trap Unanswerable",
        "question": "Số lượng cụ thể GPU NVIDIA H100 hoặc H200 đang vận hành tại mỗi Nhà máy AI của FPT ở Việt Nam và Nhật Bản được công bố là bao nhiêu?",
    },
    {
        "id": "preset_basic_q10",
        "kb": "basic",
        "mode": "compare",
        "tag": "MOET So sánh q10",
        "question": "Theo thang điểm 10 trong quy chế đào tạo đại học, mức điểm nào được xếp loại Xuất sắc, Giỏi, Khá?",
    },
    {
        "id": "preset_basic_q19",
        "kb": "basic",
        "mode": "compare",
        "tag": "MOET Multi-chunk q19",
        "question": "Trong thời gian tối đa bao lâu kể từ khi luận án của nghiên cứu sinh đạt quy trình phản biện độc lập thì cơ sở đào tạo phải tổ chức họp Hội đồng đánh giá luận án, và những người thân nào của nghiên cứu sinh không được tham gia Hội đồng?",
    },
    {
        "id": "preset_basic_u01",
        "kb": "basic",
        "mode": "compare",
        "tag": "MOET Trap u01",
        "question": "Theo Quy chế đào tạo trình độ đại học và tiến sĩ trong bộ tài liệu, mức tiền học bổng khuyến khích học tập hàng tháng tối thiểu là bao nhiêu đồng?",
    },
]


def _read_json_safe(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _extract_case_list(payload: Any, split: str) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return list(payload.get(split) or [])
    if isinstance(payload, list):
        return [c for c in payload if c.get("split") == split]
    return []


class DemoAppContext:
    """Thread-safe data & lazy Qdrant retriever container powering the Day 7 demo UI."""

    def __init__(
        self,
        *,
        root: Path = ROOT,
        basic_qdrant_path: Path | None = None,
        adv_qdrant_path: Path | None = None,
    ) -> None:
        self.root = root
        default_basic_qdrant = Path(os.getenv("DAY6_QDRANT_PATH", str(DEFAULT_DAY6_QDRANT)))
        self.basic_qdrant_path = basic_qdrant_path or (
            default_basic_qdrant if default_basic_qdrant.is_absolute() else (root / default_basic_qdrant).resolve()
        )
        self.adv_qdrant_path = adv_qdrant_path or (root / "starter" / "data" / "advanced" / "qdrant_db")
        self.adv_pdf_path = root / "starter" / "data" / "advanced" / "raw" / "FPT_BCTN_2025.pdf"
        self.adv_pages_dir = root / "starter" / "data" / "advanced" / "pages"

        self.basic_chunks: list[dict[str, Any]] = _read_json_safe(
            root / "starter" / "data" / "basic" / "chunks.json", []
        )
        self.adv_chunks: list[dict[str, Any]] = _read_json_safe(
            root / "starter" / "data" / "advanced" / "chunks.json", []
        )
        self.basic_chunks_by_id = {c["id"]: c for c in self.basic_chunks}
        self.adv_chunks_by_id = {c["id"]: c for c in self.adv_chunks}

        self._basic_lock = threading.Lock()
        self._adv_lock = threading.Lock()
        self._basic_dense: DenseQdrantRetriever | None = None
        self._basic_bm25: Bm25Retriever | None = None
        self._basic_hybrid: HybridRrfRetriever | None = None
        self._adv_dense: DenseQdrantRetriever | None = None
        self._adv_bm25: Bm25Retriever | None = None
        self._adv_hybrid: HybridRrfRetriever | None = None

    def close(self) -> None:
        for dense in (self._basic_dense, self._adv_dense):
            if dense is not None:
                try:
                    dense.close()
                except Exception:
                    pass
        self._basic_dense = None
        self._adv_dense = None

    def _ensure_basic_retrievers(self) -> tuple[DenseQdrantRetriever, Bm25Retriever, HybridRrfRetriever]:
        if self._basic_bm25 is None:
            self._basic_bm25 = Bm25Retriever(self.basic_chunks)
        if self._basic_dense is None:
            self._basic_dense = DenseQdrantRetriever(
                qdrant_path=self.basic_qdrant_path,
                collection_name=os.getenv("DAY6_COLLECTION_NAME", "day06_moet_chunks"),
                strategy_filter=os.getenv("DAY6_STRATEGY", "C"),
                embed_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
            self._basic_hybrid = HybridRrfRetriever(self._basic_dense, self._basic_bm25, candidate_k=10, rrf_k=60)
        assert self._basic_hybrid is not None
        return self._basic_dense, self._basic_bm25, self._basic_hybrid

    def _ensure_adv_retrievers(self) -> tuple[DenseQdrantRetriever, Bm25Retriever, HybridRrfRetriever]:
        if self._adv_bm25 is None:
            self._adv_bm25 = Bm25Retriever(self.adv_chunks, name="bm25_advanced")
        if self._adv_dense is None:
            self._adv_dense = DenseQdrantRetriever(
                qdrant_path=self.adv_qdrant_path,
                collection_name=os.getenv("ADVANCED_COLLECTION_NAME", "fpt_bctn_2025"),
                strategy_filter=None,
                embed_model=os.getenv("ADVANCED_EMBEDDING_MODEL", "text-embedding-3-large"),
                name="dense_qdrant_large",
            )
            self._adv_hybrid = HybridRrfRetriever(
                self._adv_dense, self._adv_bm25, candidate_k=15, rrf_k=60, name="hybrid_rrf_advanced"
            )
        assert self._adv_hybrid is not None
        return self._adv_dense, self._adv_bm25, self._adv_hybrid

    def hydrate_trace(self, trace: dict[str, Any], *, kb: str) -> dict[str, Any]:
        """Attach resolved chunk payloads for citations, context_ids, and tool hits."""
        lookup = self.adv_chunks_by_id if kb == "advanced" else self.basic_chunks_by_id
        ids: list[str] = []
        seen: set[str] = set()

        def _push(cid: str) -> None:
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)

        for cid in trace.get("citations") or []:
            _push(cid)
        for cid in trace.get("context_ids") or []:
            _push(cid)
        for ev in trace.get("events") or []:
            for r in ev.get("results") or []:
                if isinstance(r, dict) and r.get("id"):
                    _push(r["id"])

        details: dict[str, dict[str, Any]] = {}
        pages: list[int] = []
        seen_pages: set[int] = set()
        for cid in ids:
            ch = lookup.get(cid)
            if not ch:
                continue
            meta = ch.get("metadata") or {}
            pg = int(ch.get("page") or 0)
            if kb == "advanced" and pg > 0 and pg not in seen_pages:
                seen_pages.add(pg)
                pages.append(pg)
            details[cid] = {
                "id": ch["id"],
                "source_id": ch.get("source_id"),
                "page": pg,
                "section": meta.get("section") or "",
                "modality": meta.get("modality") or "text",
                "text": ch.get("text") or "",
            }
        return {**trace, "chunk_details": details, "pages": pages}

    def run_live_query(
        self,
        *,
        question: str,
        kb: str = "advanced",
        mode: str = "advanced",
        filter_cap: str | None = None,
        use_reranker: bool = True,
    ) -> dict[str, Any]:
        clean_q = question.strip()
        if not clean_q:
            raise ValueError("Question must not be empty.")

        query_id = f"live_{int(time.time() * 1000)}"
        if kb == "advanced":
            return self._run_live_advanced(query_id, clean_q, use_reranker=use_reranker)
        if kb == "basic":
            return self._run_live_basic(query_id, clean_q, mode=mode, filter_cap=filter_cap)
        raise ValueError(f"Unsupported knowledge base: {kb}")

    def _run_live_advanced(self, query_id: str, question: str, *, use_reranker: bool) -> dict[str, Any]:
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        vision_model = os.getenv("ADVANCED_VISION_MODEL", "gpt-4.1-mini")
        embed_model = os.getenv("ADVANCED_EMBEDDING_MODEL", "text-embedding-3-large")
        with self._adv_lock:
            dense, bm25, hybrid = self._ensure_adv_retrievers()
            reranker = (
                OpenAIRerankRetriever(hybrid, candidate_k=12, model=chat_model, name="hybrid_rerank_advanced")
                if use_reranker
                else None
            )
            retriever = reranker if reranker is not None else hybrid
            before_tokens = dense.total_tokens
            before_cost = dense.total_cost_usd
            case = {"id": query_id, "split": "live", "question": question, "category": "live"}
            trace = run_advanced_agent(
                case,
                hybrid_retriever=retriever,
                lexical_retriever=bm25,
                chunks=self.adv_chunks,
                pdf_path=self.adv_pdf_path,
                pages_dir=self.adv_pages_dir,
                chat_model=chat_model,
                vision_model=vision_model,
                max_turns=7,
                top_k=5,
                budget=8000,
            )
            embed_tokens = dense.total_tokens - before_tokens
            embed_cost = max(0.0, dense.total_cost_usd - before_cost)
            rerank_tokens = (reranker.prompt_tokens + reranker.completion_tokens) if reranker else 0
            rerank_cost = reranker.cost_usd if reranker else 0.0
            llm_cost = float(trace.get("llm_cost_usd") or 0.0)
            trace["telemetry"] = {
                "kb": "advanced",
                "mode": "advanced",
                "chat_model": chat_model,
                "vision_model": vision_model,
                "embed_model": embed_model,
                "use_reranker": use_reranker,
                "embed_tokens": embed_tokens,
                "embed_cost_usd": round(embed_cost, 6),
                "rerank_tokens": rerank_tokens,
                "rerank_cost_usd": round(rerank_cost, 6),
                "agent_llm_tokens": int(trace.get("llm_tokens") or 0),
                "agent_cost_usd": round(llm_cost, 6),
                "total_cost_usd": round(llm_cost + embed_cost + rerank_cost, 6),
            }
            return {"kb": "advanced", "mode": "advanced", "primary": self.hydrate_trace(trace, kb="advanced")}

    def _run_live_basic(self, query_id: str, question: str, *, mode: str, filter_cap: str | None) -> dict[str, Any]:
        if mode not in ("baseline", "chosen", "compare"):
            raise ValueError(f"Unsupported basic mode: {mode}")
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        case: dict[str, Any] = {"id": query_id, "split": "live", "question": question}
        if filter_cap in ("dai_hoc", "tien_si"):
            case["filter"] = {"cap_dao_tao": filter_cap}

        with self._basic_lock:
            dense, bm25, hybrid = self._ensure_basic_retrievers()
            runs: dict[str, Any] = {}
            target_modes = ["baseline", "chosen"] if mode == "compare" else [mode]
            for m in target_modes:
                before_tokens = dense.total_tokens
                before_cost = dense.total_cost_usd
                if m == "baseline":
                    trace = run_baseline_starter_agent(case, dense, model=chat_model, max_searches=4, top_k=3, budget=800)
                    rerank_tokens = 0
                    rerank_cost = 0.0
                else:
                    reranker = OpenAIRerankRetriever(hybrid, candidate_k=10, model=chat_model, name="hybrid_rerank")
                    trace = run_multitool_agent(
                        case,
                        hybrid_retriever=reranker,
                        lexical_retriever=bm25,
                        chunks=self.basic_chunks,
                        model=chat_model,
                        max_turns=5,
                        top_k=3,
                        budget=2400,
                    )
                    rerank_tokens = reranker.prompt_tokens + reranker.completion_tokens
                    rerank_cost = reranker.cost_usd
                embed_tokens = dense.total_tokens - before_tokens
                embed_cost = max(0.0, dense.total_cost_usd - before_cost)
                llm_cost = float(trace.get("llm_cost_usd") or 0.0)
                trace["telemetry"] = {
                    "kb": "basic",
                    "mode": m,
                    "chat_model": chat_model,
                    "embed_model": embed_model,
                    "embed_tokens": embed_tokens,
                    "embed_cost_usd": round(embed_cost, 6),
                    "rerank_tokens": rerank_tokens,
                    "rerank_cost_usd": round(rerank_cost, 6),
                    "agent_llm_tokens": int(trace.get("llm_tokens") or 0),
                    "agent_cost_usd": round(llm_cost, 6),
                    "total_cost_usd": round(llm_cost + embed_cost + rerank_cost, 6),
                }
                runs[m] = self.hydrate_trace(trace, kb="basic")

            primary = runs.get("chosen") or runs["baseline"]
            return {"kb": "basic", "mode": mode, "primary": primary, "comparison": runs if mode == "compare" else None}

    def build_overview_payload(self) -> dict[str, Any]:
        """Assemble cached benchmark stats, traces, corpus metadata, and presets."""
        basic_cases = _read_json_safe(self.root / "starter" / "data" / "basic" / "cases.json", [])
        basic_questions = _read_json_safe(self.root / "starter" / "data" / "basic" / "questions.json", [])
        export_report = _read_json_safe(self.root / "starter" / "data" / "basic" / "export_report.json", {})

        basic_retrieval = {
            sp: _read_json_safe(self.root / "starter" / "results" / "basic_retrieval" / sp / "comparison.json", {})
            for sp in ("dev", "test")
        }

        basic_agent: dict[str, Any] = {}
        for sp in ("dev", "test"):
            basic_agent[sp] = {}
            for mode in ("baseline", "chosen"):
                base_dir = self.root / "starter" / "results" / "basic_agent" / sp / mode
                raw_traces = _read_json_safe(base_dir / "traces.json", [])
                basic_agent[sp][mode] = {
                    "summary": _read_json_safe(base_dir / "summary.json", {}),
                    "stats": _read_json_safe(base_dir / "stats.json", {}),
                    "reviews": _read_json_safe(base_dir / "reviews.json", []),
                    "traces": [self.hydrate_trace(t, kb="basic") for t in raw_traces],
                }

        adv_manifest = _read_json_safe(self.root / "starter" / "data" / "advanced" / "manifest.json", {})
        adv_ingest = _read_json_safe(self.root / "starter" / "data" / "advanced" / "ingest_manifest.json", {})
        adv_cases_raw = _read_json_safe(self.root / "starter" / "data" / "advanced" / "cases.json", {})

        adv_dev_dir = self.root / "starter" / "results" / "advanced" / "dev"
        adv_test_dir = self.root / "starter" / "results" / "advanced" / "test"

        iterations: list[dict[str, Any]] = []
        for tag, fname in (("v1", "v1_stats.json"), ("v2", "v2_stats.json"), ("v3", "v3_stats.json"), ("v4", "stats.json")):
            st = _read_json_safe(adv_dev_dir / fname)
            if st:
                tname = "traces.json" if tag == "v4" else f"{tag}_traces.json"
                tr = _read_json_safe(adv_dev_dir / tname, [])
                retry_count = sum(
                    1 for item in tr if any(ev.get("event") == "invalid_or_missing_citations" for ev in item.get("events", []))
                )
                iterations.append({**st, "version": tag, "citation_retries": retry_count})

        adv_splits: dict[str, Any] = {}
        for sp, dpath in (("dev", adv_dev_dir), ("test", adv_test_dir)):
            raw_traces = _read_json_safe(dpath / "traces.json", [])
            adv_splits[sp] = {
                "cases": _extract_case_list(adv_cases_raw, sp),
                "stats": _read_json_safe(dpath / "stats.json", {}),
                "summary": _read_json_safe(dpath / "summary.json", {}),
                "reviews": _read_json_safe(dpath / "reviews.json", []),
                "traces": [self.hydrate_trace(t, kb="advanced") for t in raw_traces],
            }

        moet_counts: dict[str, int] = {}
        for ch in self.basic_chunks:
            sid = str(ch.get("source_id") or "unknown")
            moet_counts[sid] = moet_counts.get(sid, 0) + 1

        return {
            "status": {
                "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
                "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
                "vision_model": os.getenv("ADVANCED_VISION_MODEL", "gpt-4.1-mini"),
                "basic_embed_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                "adv_embed_model": os.getenv("ADVANCED_EMBEDDING_MODEL", "text-embedding-3-large"),
                "basic_qdrant_ready": self.basic_qdrant_path.exists(),
                "adv_qdrant_ready": self.adv_qdrant_path.exists(),
                "adv_pdf_ready": self.adv_pdf_path.exists(),
                "adv_pages_count": len(list(self.adv_pages_dir.glob("p*.png"))) if self.adv_pages_dir.exists() else 0,
            },
            "presets": PRESET_QUESTIONS,
            "basic": {
                "export_report": export_report,
                "chunk_count": len(self.basic_chunks),
                "sources": moet_counts,
                "cases": basic_cases,
                "questions": basic_questions,
                "retrieval": basic_retrieval,
                "agent": basic_agent,
            },
            "advanced": {
                "manifest": adv_manifest,
                "ingest_manifest": adv_ingest,
                "chunk_count": len(self.adv_chunks),
                "dev_iterations": iterations,
                "splits": adv_splits,
            },
        }

    def query_chunks(
        self,
        *,
        kb: str,
        page: int | None = None,
        source_id: str | None = None,
        modality: str | None = None,
        q: str = "",
        limit: int = 40,
        offset: int = 0,
    ) -> dict[str, Any]:
        pool = self.adv_chunks if kb == "advanced" else self.basic_chunks
        needle = q.strip().lower()
        filtered: list[dict[str, Any]] = []
        for ch in pool:
            meta = ch.get("metadata") or {}
            mod = str(meta.get("modality") or "text")
            if page is not None and int(ch.get("page") or 0) != page:
                continue
            if source_id and ch.get("source_id") != source_id:
                continue
            if modality and modality != "all" and mod != modality:
                continue
            if needle:
                hay = f"{ch.get('id', '')} {meta.get('section', '')} {ch.get('text', '')}".lower()
                if needle not in hay:
                    continue
            filtered.append(
                {
                    "id": ch["id"],
                    "source_id": ch.get("source_id"),
                    "page": int(ch.get("page") or 0),
                    "section": meta.get("section") or "",
                    "modality": mod,
                    "text": ch.get("text") or "",
                }
            )
        sliced = filtered[offset : offset + max(1, min(limit, 100))]
        return {"total": len(filtered), "offset": offset, "limit": limit, "items": sliced}

    def get_page_png(self, page_no: int, *, region: str = "cached") -> bytes:
        total_pages = 232
        if page_no < 1 or page_no > total_pages:
            raise ValueError(f"Page {page_no} out of range (1..{total_pages}).")
        if region == "cached":
            cached_path = self.adv_pages_dir / f"p{page_no:03d}.png"
            if cached_path.exists():
                return cached_path.read_bytes()
            region = "full"
        if region not in ("full", "top", "bottom", "left", "right"):
            raise ValueError(f"Invalid region: {region}")
        return render_page_region_png_bytes(
            page_no,
            pdf_path=self.adv_pdf_path,
            pages_dir=self.adv_pages_dir,
            region=region,
        )


def make_handler(ctx: DemoAppContext) -> type[BaseHTTPRequestHandler]:
    """Create a HTTP request handler bound cleanly to DemoAppContext."""

    class DemoRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            sys.stderr.write(f"[demo] {self.address_string()} - {format % args}\n")

        def _send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(raw, "application/json; charset=utf-8", status=status)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = parse_qs(parsed.query)

            if path == "/":
                index_file = WEB_DIR / "index.html"
                if not index_file.exists():
                    self._send_json({"error": "Frontend index.html missing."}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(index_file.read_bytes(), "text/html; charset=utf-8")
                return

            if path.startswith("/assets/"):
                rel = path[len("/assets/") :]
                target = (WEB_DIR / rel).resolve()
                if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
                    self._send_json({"error": "Asset not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                ext_map = {
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".html": "text/html; charset=utf-8",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                }
                ctype = ext_map.get(target.suffix.lower())
                if ctype is None:
                    guessed, _ = mimetypes.guess_type(str(target))
                    ctype = f"{guessed or 'application/octet-stream'}; charset=utf-8"
                self._send_bytes(target.read_bytes(), ctype)
                return

            if path == "/api/overview":
                self._send_json(ctx.build_overview_payload())
                return

            if path == "/api/chunks":
                kb = (qs.get("kb") or ["advanced"])[0]
                page_raw = (qs.get("page") or [""])[0].strip()
                page_val = int(page_raw) if page_raw.isdigit() else None
                source_id = (qs.get("source_id") or [""])[0].strip() or None
                modality = (qs.get("modality") or [""])[0].strip() or None
                q = (qs.get("q") or [""])[0]
                limit = int((qs.get("limit") or ["40"])[0])
                offset = int((qs.get("offset") or ["0"])[0])
                self._send_json(
                    ctx.query_chunks(
                        kb=kb, page=page_val, source_id=source_id, modality=modality, q=q, limit=limit, offset=offset
                    )
                )
                return

            m_page = re.fullmatch(r"/api/advanced/page/(\d+)(?:\.png)?", path)
            if m_page:
                page_no = int(m_page.group(1))
                region = (qs.get("region") or ["cached"])[0]
                try:
                    png = ctx.get_page_png(page_no, region=region)
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(png, "image/png")
                return

            if path.startswith("/api/chunk/"):
                cid = path[len("/api/chunk/") :]
                ch = ctx.adv_chunks_by_id.get(cid) or ctx.basic_chunks_by_id.get(cid)
                if not ch:
                    self._send_json({"error": f"Unknown chunk: {cid}"}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_json(ch)
                return

            self._send_json({"error": f"Not found: {path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/ask":
                self._send_json({"error": f"Not found: {parsed.path}"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
                result = ctx.run_live_query(
                    question=str(body.get("question") or ""),
                    kb=str(body.get("kb") or "advanced"),
                    mode=str(body.get("mode") or "advanced"),
                    filter_cap=body.get("filter_cap") or None,
                    use_reranker=bool(body.get("use_reranker", True)),
                )
                self._send_json(result)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    return DemoRequestHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Day 7 Agentic RAG & Multimodal Studio web server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    ctx = DemoAppContext()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(ctx))
    print(f"Day 7 Agentic RAG Studio listening at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down demo server...", flush=True)
    finally:
        server.server_close()
        ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
