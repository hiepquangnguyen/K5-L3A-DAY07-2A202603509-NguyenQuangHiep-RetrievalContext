"""Full multimodal PDF ingestion pipeline for Advanced Challenge (FPT_BCTN_2025.pdf, 232 pages)."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from rerankers import estimate_chat_cost_usd  # noqa: E402
from retrievers import EMBED_PRICE_PER_1M_USD  # noqa: E402

VISION_SYSTEM_PROMPT = """Bạn là chuyên gia số hóa tài liệu đa phương thức (bảng biểu, biểu đồ, sơ đồ, hình ảnh báo cáo thường niên).
Hãy đọc kỹ ảnh trang PDF được cung cấp và trích xuất chính xác 100% dữ liệu nhìn thấy, KHÔNG suy đoán số liệu không hiển thị.
Trả về duy nhất JSON hợp lệ theo schema:
{
  "page_title": "Tiêu đề hoặc tên mục chính của trang",
  "content_types": ["text" | "table" | "chart" | "diagram" | "image"],
  "visual_summary": "Mô tả chi tiết mọi yếu tố trực quan: ảnh chụp, nhân sự/chức danh, sơ đồ khối, mũi tên quy trình, infographic, mốc thời gian, biểu tượng/nhãn.",
  "charts_and_tables": "Trích xuất ĐẦY ĐỦ số liệu của từng biểu đồ (loại biểu đồ, tiêu đề, đơn vị, trục X/Y, chú giải, TẤT CẢ giá trị theo từng năm/hạng mục) và từng bảng biểu (bảng Markdown đầy đủ hàng/cột/đơn vị). Nếu trang không có bảng hoặc biểu đồ thì để chuỗi rỗng."
}"""


def clean_pdf_text(raw: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in (raw or "").splitlines()]
    compact = "\n".join(ln for ln in lines if ln)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()


def split_text_windows(text: str, *, max_chars: int = 900, overlap: int = 150) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= max_chars:
        return [clean]
    paragraphs = [p.strip() for p in clean.split("\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            start = 0
            while start < len(para):
                end = min(len(para), start + max_chars)
                chunks.append(para[start:end].strip())
                if end == len(para):
                    break
                start = max(start + 1, end - overlap)
            continue
        cand = f"{cur}\n{para}".strip() if cur else para
        if len(cand) <= max_chars:
            cur = cand
        else:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 and len(cur) > overlap else ""
            cur = f"{tail}\n{para}".strip() if tail and len(f"{tail}\n{para}") <= max_chars else para
    if cur:
        chunks.append(cur)
    return chunks


def render_pdf_pages(pdf_path: Path, pages_dir: Path, *, dpi: int = 150, force: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    total = min(len(doc), limit) if limit else len(doc)
    records: list[dict[str, Any]] = []
    try:
        for idx in range(total):
            page_no = idx + 1
            page = doc[idx]
            img_path = pages_dir / f"p{page_no:03d}.png"
            if force or not img_path.exists():
                pix = page.get_pixmap(dpi=dpi)
                img_path.write_bytes(pix.tobytes("png"))
            records.append(
                {
                    "page": page_no,
                    "image_path": str(img_path.relative_to(ROOT)).replace("\\", "/"),
                    "abs_image_path": str(img_path),
                    "raw_text": clean_pdf_text(page.get_text("text")),
                    "num_images": len(page.get_images()),
                    "num_drawings": len(page.get_drawings()),
                }
            )
    finally:
        doc.close()
    return records


def extract_page_vision(
    page_rec: dict[str, Any],
    vision_cache_dir: Path,
    *,
    model: str,
    api_key: str,
    base_url: str,
    force: bool = False,
) -> dict[str, Any]:
    page_no = int(page_rec["page"])
    cache_file = vision_cache_dir / f"p{page_no:03d}.json"
    if not force and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        cached["cached"] = True
        return cached

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)
    b64 = base64.b64encode(Path(page_rec["abs_image_path"]).read_bytes()).decode("ascii")
    hint = page_rec["raw_text"][:500]
    user_text = (
        f"Trang {page_no}/232 của Báo cáo thường niên FPT 2025.\n"
        f"Gợi ý text thô đầu trang (chỉ dùng đối chiếu tên riêng nếu chữ nhỏ):\n{hint}"
    )

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": VISION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
                        ],
                    },
                ],
            )
            usage = getattr(resp, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            parsed = json.loads((resp.choices[0].message.content or "{}").strip())
            result = {
                "page": page_no,
                "page_title": str(parsed.get("page_title") or "").strip(),
                "content_types": [str(x) for x in (parsed.get("content_types") or [])],
                "visual_summary": str(parsed.get("visual_summary") or "").strip(),
                "charts_and_tables": str(parsed.get("charts_and_tables") or "").strip(),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cost_usd": round(estimate_chat_cost_usd(model, pt, ct), 6),
                "error": None,
            }
            cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["cached"] = False
            return result
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))

    return {
        "page": page_no,
        "page_title": "",
        "content_types": ["text"],
        "visual_summary": "",
        "charts_and_tables": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "error": f"{type(last_exc).__name__}: {last_exc}" if last_exc else "Unknown error",
        "cached": False,
    }


def build_multimodal_chunks(page_records: list[dict[str, Any]], vision_by_page: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    source_name = "FPT Báo cáo thường niên 2025 (232 trang)"
    for rec in page_records:
        page_no = int(rec["page"])
        vis = vision_by_page.get(page_no, {})
        title = vis.get("page_title") or f"Trang {page_no}"
        ctypes = vis.get("content_types") or ["text"]

        # 1. Text chunks from PyMuPDF
        text_parts = split_text_windows(rec["raw_text"], max_chars=900, overlap=150)
        for idx, part in enumerate(text_parts):
            cid = f"adv:fpt2025:p{page_no:03d}:t{idx:03d}"
            chunks.append(
                {
                    "id": cid,
                    "text": f"[Trang {page_no} - Text | {title}]\n{part}",
                    "source_id": "fpt-bctn-2025",
                    "source": source_name,
                    "page": page_no,
                    "metadata": {
                        "section": title,
                        "modality": "text",
                        "content_types": ctypes,
                        "image_path": rec["image_path"],
                        "num_images": rec["num_images"],
                        "num_drawings": rec["num_drawings"],
                    },
                }
            )

        # 2. Vision multimodal summary / charts / tables chunks
        vis_sections = []
        if vis.get("visual_summary"):
            vis_sections.append(f"Mô tả trực quan:\n{vis['visual_summary']}")
        if vis.get("charts_and_tables"):
            vis_sections.append(f"Số liệu biểu đồ & bảng biểu:\n{vis['charts_and_tables']}")
        vis_body = "\n\n".join(vis_sections).strip()
        if vis_body:
            vis_parts = split_text_windows(vis_body, max_chars=1400, overlap=150)
            for idx, part in enumerate(vis_parts):
                cid = f"adv:fpt2025:p{page_no:03d}:v{idx:03d}"
                chunks.append(
                    {
                        "id": cid,
                        "text": f"[Trang {page_no} - Vision/Multimodal | {title}]\n{part}",
                        "source_id": "fpt-bctn-2025",
                        "source": source_name,
                        "page": page_no,
                        "metadata": {
                            "section": title,
                            "modality": "multimodal",
                            "content_types": ctypes,
                            "image_path": rec["image_path"],
                            "num_images": rec["num_images"],
                            "num_drawings": rec["num_drawings"],
                        },
                    }
                )
    return chunks


def index_chunks_qdrant(
    chunks: list[dict[str, Any]],
    qdrant_dir: Path,
    collection_name: str,
    *,
    embed_model: str,
    api_key: str,
    base_url: str,
    batch_size: int = 64,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    vectors: list[list[float]] = []
    total_tokens = 0
    t0 = time.perf_counter()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        inputs = [c["text"] for c in batch]
        resp = client.embeddings.create(model=embed_model, input=inputs)
        vectors.extend([float(x) for x in item.embedding] for item in resp.data)
        usage = getattr(resp, "usage", None)
        if usage:
            total_tokens += int(getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
        print(f"[embed] {min(start + batch_size, len(chunks))}/{len(chunks)} chunks embedded", flush=True)

    embed_seconds = time.perf_counter() - t0
    dim = len(vectors[0]) if vectors else 3072
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(qdrant_dir))
    try:
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name)
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        points = []
        for c, vec in zip(chunks, vectors):
            meta = c.get("metadata") or {}
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["id"])),
                    vector=vec,
                    payload={
                        "chunk_id": c["id"],
                        "source_id": c["source_id"],
                        "source": c["source"],
                        "page": c["page"],
                        "section": meta.get("section", ""),
                        "modality": meta.get("modality", "text"),
                        "content_types": meta.get("content_types", []),
                        "image_path": meta.get("image_path", ""),
                        "text": c["text"],
                    },
                )
            )
        for start in range(0, len(points), 128):
            qdrant.upsert(collection_name=collection_name, points=points[start : start + 128], wait=True)
    finally:
        qdrant.close()

    rate = EMBED_PRICE_PER_1M_USD.get(embed_model, 0.13)
    return {
        "embed_model": embed_model,
        "dimensions": dim,
        "vectors_indexed": len(vectors),
        "embed_tokens": total_tokens,
        "embed_cost_usd": round((total_tokens / 1_000_000.0) * rate, 6),
        "embed_seconds": round(embed_seconds, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest FPT_BCTN_2025.pdf (232p) with Vision + text-embedding-3-large.")
    parser.add_argument("--pdf", type=Path, default=ROOT / "starter" / "data" / "advanced" / "raw" / "FPT_BCTN_2025.pdf")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "starter" / "data" / "advanced")
    parser.add_argument("--qdrant-dir", type=Path, default=ROOT / "starter" / "data" / "advanced" / "qdrant_db")
    parser.add_argument("--collection", default=os.getenv("ADVANCED_COLLECTION_NAME", "fpt_bctn_2025"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--vision-model", default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--embed-model", default=os.getenv("ADVANCED_EMBEDDING_MODEL", "text-embedding-3-large"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    t_start = time.perf_counter()
    pages_dir = args.out_dir / "pages"
    vision_cache_dir = args.out_dir / "vision_cache"
    vision_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step 1/4] Rendering PDF pages ({args.pdf}) at {args.dpi} DPI...", flush=True)
    page_records = render_pdf_pages(args.pdf, pages_dir, dpi=args.dpi, force=args.force, limit=args.limit)
    print(f"  Rendered/verified {len(page_records)} pages.", flush=True)

    print(f"[step 2/4] Running Vision extraction ({args.vision_model}, workers={args.workers})...", flush=True)
    t_vis = time.perf_counter()
    vision_results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                extract_page_vision,
                rec,
                vision_cache_dir,
                model=args.vision_model,
                api_key=api_key,
                base_url=base_url,
                force=args.force,
            )
            for rec in page_records
        ]
        for done_idx, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            res = fut.result()
            vision_results.append(res)
            if done_idx % 25 == 0 or done_idx == len(page_records) or res.get("error"):
                print(f"  [vision] {done_idx}/{len(page_records)} p{res['page']:03d} err={res.get('error')}", flush=True)

    vision_seconds = time.perf_counter() - t_vis
    vision_by_page = {r["page"]: r for r in vision_results}
    failed_pages = sorted(r["page"] for r in vision_results if r.get("error"))
    vision_cost_fresh = sum(float(r.get("cost_usd") or 0.0) for r in vision_results if not r.get("cached"))
    vision_cost_total = sum(float(r.get("cost_usd") or 0.0) for r in vision_results)
    vision_tokens_total = sum(int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0) for r in vision_results)

    print("[step 3/4] Building text + multimodal chunks...", flush=True)
    chunks = build_multimodal_chunks(page_records, vision_by_page)
    chunks_path = args.out_dir / "chunks.json"
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    text_chunks_n = sum(1 for c in chunks if c["metadata"]["modality"] == "text")
    mm_chunks_n = sum(1 for c in chunks if c["metadata"]["modality"] == "multimodal")
    print(f"  Built {len(chunks)} chunks (text={text_chunks_n}, multimodal={mm_chunks_n}).", flush=True)

    print(f"[step 4/4] Embedding & indexing into Qdrant ({args.collection}, model={args.embed_model})...", flush=True)
    embed_stats = index_chunks_qdrant(
        chunks,
        args.qdrant_dir,
        args.collection,
        embed_model=args.embed_model,
        api_key=api_key,
        base_url=base_url,
    )

    manifest = {
        "pdf": str(args.pdf.relative_to(ROOT)).replace("\\", "/"),
        "pages_processed": len(page_records),
        "failed_vision_pages": failed_pages,
        "text_chunks": text_chunks_n,
        "multimodal_chunks": mm_chunks_n,
        "total_chunks": len(chunks),
        "vision_model": args.vision_model,
        "vision_tokens_total": vision_tokens_total,
        "vision_cost_usd_total": round(vision_cost_total, 6),
        "vision_cost_usd_this_run": round(vision_cost_fresh, 6),
        "vision_seconds": round(vision_seconds, 2),
        **embed_stats,
        "total_cost_usd": round(vision_cost_total + embed_stats["embed_cost_usd"], 6),
        "total_elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "collection": args.collection,
        "qdrant_dir": str(args.qdrant_dir.relative_to(ROOT)).replace("\\", "/"),
    }
    (args.out_dir / "ingest_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nIngestion complete:\n" + json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
