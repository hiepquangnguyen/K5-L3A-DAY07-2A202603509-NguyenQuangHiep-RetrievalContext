"""Download and inspect candidate >=100-page multimodal PDF documents for the Advanced Challenge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz  # PyMuPDF
import httpx

ROOT = Path(__file__).resolve().parents[1]


def download_pdf(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    with httpx.Client(follow_redirects=True, timeout=120.0, headers=headers) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            hasher = hashlib.sha256()
            size = 0
            with dest.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
    return {"url": url, "path": str(dest), "bytes": size, "sha256": hasher.hexdigest()}


def inspect_pdf(pdf_path: Path) -> dict:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    text_pages = 0
    image_pages = 0
    drawing_pages = 0
    total_chars = 0
    total_images = 0
    sample_rich_pages: list[dict] = []

    for idx, page in enumerate(doc, start=1):
        txt = (page.get_text("text") or "").strip()
        imgs = page.get_images(full=True)
        drawings = page.get_drawings()
        c_len = len(txt)
        total_chars += c_len
        if c_len >= 100:
            text_pages += 1
        if imgs:
            image_pages += 1
            total_images += len(imgs)
        if len(drawings) >= 15:
            drawing_pages += 1
        if len(sample_rich_pages) < 15 and (imgs or len(drawings) >= 30) and c_len >= 150:
            sample_rich_pages.append(
                {
                    "page": idx,
                    "chars": c_len,
                    "images": len(imgs),
                    "drawings": len(drawings),
                    "preview": " ".join(txt.split())[:140],
                }
            )
    doc.close()
    return {
        "file": pdf_path.name,
        "total_pages": total_pages,
        "text_pages_ge_100chars": text_pages,
        "pages_with_embedded_images": image_pages,
        "total_embedded_images": total_images,
        "pages_with_vector_drawings_ge_15": drawing_pages,
        "total_extracted_chars": total_chars,
        "avg_chars_per_page": round(total_chars / max(total_pages, 1), 1),
        "sample_rich_pages": sample_rich_pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect a PDF for Advanced Challenge (>=100 pages).")
    parser.add_argument("--url", help="Direct PDF URL to download")
    parser.add_argument("--output", type=Path, required=True, help="Local PDF destination path")
    parser.add_argument("--inspect-only", action="store_true", help="Skip download and inspect existing PDF")
    args = parser.parse_args()

    meta: dict = {}
    if not args.inspect_only:
        if not args.url:
            raise ValueError("--url is required unless --inspect-only is set")
        meta["download"] = download_pdf(args.url, args.output)
    meta["inspection"] = inspect_pdf(args.output)

    report_path = args.output.with_suffix(".inspect.json")
    report_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
