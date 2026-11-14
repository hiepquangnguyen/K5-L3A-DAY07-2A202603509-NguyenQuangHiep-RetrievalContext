"""Verify Day 7 environment: OpenAI API key, model tool-calling, embedding, and Day 6 Qdrant compatibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]


def _mask(key: str | None) -> str:
    if not key:
        return "MISSING"
    if len(key) <= 10:
        return "***"
    return f"{key[:7]}...{key[-4:]}"


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Verify OpenAI API and Day 6 Qdrant connection.")
    parser.add_argument("--live", action="store_true", help="Send a minimal live API probe to OpenAI.")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    day6_qdrant = os.getenv(
        "DAY6_QDRANT_PATH",
        str((ROOT.parent / "K5-L3A-DAY06-NguyenQuangHiep-2A202603509-DataIngestion/starter/qdrant_db").resolve()),
    )
    qdrant_path = Path(day6_qdrant)
    if not qdrant_path.is_absolute():
        qdrant_path = (ROOT / qdrant_path).resolve()
    collection_name = os.getenv("DAY6_COLLECTION_NAME", "day06_moet_chunks")
    strategy = os.getenv("DAY6_STRATEGY", "C")

    status: dict = {
        "openai_api_key": _mask(api_key),
        "openai_base_url": base_url,
        "chat_model": chat_model,
        "embedding_model": embed_model,
        "day6_qdrant_exists": qdrant_path.exists(),
    }

    if qdrant_path.exists():
        qc = QdrantClient(path=str(qdrant_path))
        try:
            info = qc.get_collection(collection_name)
            count_strat = qc.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))]
                ),
                exact=True,
            ).count
            status["qdrant_collection"] = collection_name
            status["qdrant_vector_size"] = getattr(info.config.params.vectors, "size", None)
            status["qdrant_strategy_count"] = count_strat
        finally:
            qc.close()

    if args.live:
        if not api_key or "your-openai-api-key" in api_key:
            raise RuntimeError("OPENAI_API_KEY is missing or placeholder in .env")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30.0)
        emb = client.embeddings.create(model=embed_model, input=["Quy chế đào tạo trình độ đại học"])
        vec = emb.data[0].embedding
        status["live_embed_dim"] = len(vec)

        if qdrant_path.exists() and len(vec) == status.get("qdrant_vector_size"):
            qc = QdrantClient(path=str(qdrant_path))
            try:
                res = qc.query_points(
                    collection_name=collection_name,
                    query=vec,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))]
                    ),
                    limit=3,
                    with_payload=True,
                )
                status["qdrant_probe_top1"] = {
                    "chunk_id": res.points[0].payload.get("chunk_id") if res.points else None,
                    "score": round(float(res.points[0].score), 6) if res.points else None,
                }
            finally:
                qc.close()

        chat = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Reply with OK only."}],
            max_tokens=10,
        )
        status["live_chat_reply"] = (chat.choices[0].message.content or "").strip()

    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
