"""Export Day 6 Qdrant chunks (single strategy) and benchmark questions to Day 7 JSON schema."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit  # noqa: E402
from convert_day6 import convert_question  # noqa: E402


def _default_day6_dir() -> Path:
    env_dir = os.getenv("DAY6_PROJECT_DIR")
    if env_dir:
        p = Path(env_dir)
        return p if p.is_absolute() else (ROOT / p).resolve()
    return (ROOT.parent / "K5-L3A-DAY06-NguyenQuangHiep-2A202603509-DataIngestion").resolve()


def export_qdrant_chunks(
    qdrant_path: Path,
    collection_name: str,
    strategy: str,
) -> list[dict[str, Any]]:
    """Scroll all chunks for one Day 6 strategy and map them to the Day 7 chunk contract."""
    from qdrant_client import QdrantClient, models

    if not qdrant_path.exists():
        raise FileNotFoundError(f"Qdrant path not found: {qdrant_path}")

    client = QdrantClient(path=str(qdrant_path))
    try:
        collections = {c.name for c in client.get_collections().collections}
        if collection_name not in collections:
            raise RuntimeError(f"Collection '{collection_name}' not found in {qdrant_path}")

        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))]
        )
        chunks: list[dict[str, Any]] = []
        offset = None
        while True:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = dict(rec.payload or {})
                chunk_id = str(payload.pop("chunk_id", rec.id))
                source_id = str(payload.pop("document_id", ""))
                source = str(payload.pop("source", ""))
                page = int(payload.pop("page", 0))
                text = str(payload.pop("text", ""))
                payload.pop("strategy", None)
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": text,
                        "source_id": source_id,
                        "source": source,
                        "page": page,
                        "metadata": payload,
                    }
                )
            if next_offset is None:
                break
            offset = next_offset
    finally:
        client.close()

    chunks.sort(key=lambda c: (c["source_id"], int(c["page"]), int(c["metadata"].get("chunk_index", 0)), c["id"]))
    return chunks


def export_questions(questions_path: Path, chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert Day 6 questions and verify whether each gold quote fits inside a single exported chunk."""
    raw = json.loads(questions_path.read_text(encoding="utf-8"))
    converted = [convert_question(q) for q in raw]
    issues: list[dict[str, Any]] = []
    for q in converted:
        matched = [c["id"] for c in chunks if hit(c, q["gold"])]
        if not matched:
            issues.append(
                {
                    "id": q["id"],
                    "split": q["split"],
                    "gold": q["gold"],
                    "reason": "evidence spans multiple chunks or differs slightly from chunk text",
                }
            )
    return converted, issues


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    day6_dir = _default_day6_dir()
    parser = argparse.ArgumentParser(description="Export Day 6 chunks and questions to Day 7 JSON format.")
    parser.add_argument("--qdrant-path", type=Path, default=day6_dir / "starter" / "qdrant_db")
    parser.add_argument("--collection", default=os.getenv("DAY6_COLLECTION_NAME", "day06_moet_chunks"))
    parser.add_argument("--strategy", default=os.getenv("DAY6_STRATEGY", "C"))
    parser.add_argument("--questions-path", type=Path, default=day6_dir / "benchmark" / "questions.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "starter" / "data" / "basic")
    args = parser.parse_args()

    chunks = export_qdrant_chunks(args.qdrant_path, args.collection, args.strategy)
    questions, issues = export_questions(args.questions_path, chunks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_out = args.output_dir / "chunks.json"
    questions_out = args.output_dir / "day6_questions_converted.json"
    report_out = args.output_dir / "export_report.json"

    chunks_out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    questions_out.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "qdrant_path": str(args.qdrant_path),
        "collection": args.collection,
        "strategy": args.strategy,
        "chunk_count": len(chunks),
        "question_count": len(questions),
        "single_chunk_gold_matches": len(questions) - len(issues),
        "gold_issues": issues,
    }
    report_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
