"""Retrieval methods for Day 7: TF-IDF, BM25, Day 6 Qdrant Dense, Hybrid RRF, and LLM Rerank."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

from openai import OpenAI
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))

from baseline import TfidfIndex, tokenize  # noqa: E402

# Official OpenAI embedding rate (USD per 1M input tokens) for text-embedding-3-small.
EMBED_PRICE_PER_1M_USD = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


class RetrieverProtocol(Protocol):
    name: str

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        ...


def matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Apply metadata/top-level filters uniformly across lexical and hybrid retrievers."""
    if not filters:
        return True
    meta = chunk.get("metadata") or {}
    for key, expected in filters.items():
        if key == "document_id":
            actual = chunk.get("source_id", meta.get("document_id"))
        elif key in chunk:
            actual = chunk.get(key)
        else:
            actual = meta.get(key)
        if actual != expected:
            return False
    return True


def _with_ranks(rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        ranked.append({**row, "rank": idx, "strategy": strategy})
    return ranked


class FilteredTfidfRetriever:
    """Wrapper over starter.baseline.TfidfIndex supporting consistent metadata filtering."""

    def __init__(self, chunks: list[dict[str, Any]], name: str = "tfidf") -> None:
        self.name = name
        self.chunks = chunks
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        raw = self.index.search(query, top_k=len(self.chunks))
        filtered = [r for r in raw if matches_filters(r, filters)][:top_k]
        return _with_ranks(filtered, self.name)


class Bm25Retriever:
    """BM25Okapi lexical retriever over normalized Vietnamese tokens with metadata filtering."""

    def __init__(self, chunks: list[dict[str, Any]], name: str = "bm25") -> None:
        self.name = name
        self.chunks = chunks
        self._tokenized = [tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        scored: list[dict[str, Any]] = []
        for chunk, score in zip(self.chunks, scores):
            if score > 0 and matches_filters(chunk, filters):
                scored.append({**chunk, "score": float(score)})
        scored.sort(key=lambda c: (-c["score"], c["id"]))
        return _with_ranks(scored[:top_k], self.name)


class DenseQdrantRetriever:
    """Dense retriever querying the real Day 6 Qdrant local store via OpenAI embeddings."""

    def __init__(
        self,
        qdrant_path: Path,
        collection_name: str = "day06_moet_chunks",
        strategy_filter: str | None = "C",
        embed_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        name: str = "dense_qdrant",
    ) -> None:
        self.name = name
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        self.strategy_filter = strategy_filter
        self.embed_model = embed_model

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError("OPENAI_API_KEY is missing; cannot initialize DenseQdrantRetriever.")
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        self.openai_client = OpenAI(api_key=resolved_key, base_url=resolved_base, timeout=30.0)
        self.qdrant = QdrantClient(path=str(qdrant_path))
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.embed_seconds = 0.0
        self._query_cache: dict[str, list[float]] = {}

        info = self.qdrant.get_collection(self.collection_name)
        self.dimensions = getattr(info.config.params.vectors, "size", None)

    def close(self) -> None:
        self.qdrant.close()

    def embed_query(self, query: str) -> list[float]:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        t0 = time.perf_counter()
        resp = self.openai_client.embeddings.create(model=self.embed_model, input=[query])
        self.embed_seconds += time.perf_counter() - t0
        vec = [float(x) for x in resp.data[0].embedding]
        if self.dimensions and len(vec) != self.dimensions:
            raise RuntimeError(f"Embedding dimension mismatch: query={len(vec)} != index={self.dimensions}")
        tokens = int(getattr(resp.usage, "total_tokens", 0) or getattr(resp.usage, "prompt_tokens", 0) or 0)
        self.total_tokens += tokens
        rate = EMBED_PRICE_PER_1M_USD.get(self.embed_model, 0.0)
        self.total_cost_usd += (tokens / 1_000_000.0) * rate
        self._query_cache[query] = vec
        return vec

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        vec = self.embed_query(query)
        conditions: list[models.FieldCondition] = []
        if self.strategy_filter:
            conditions.append(
                models.FieldCondition(key="strategy", match=models.MatchValue(value=self.strategy_filter))
            )
        if filters:
            for k, v in filters.items():
                field_key = "document_id" if (k == "source_id" and self.strategy_filter) else k
                conditions.append(models.FieldCondition(key=field_key, match=models.MatchValue(value=v)))

        q_filter = models.Filter(must=conditions) if conditions else None
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vec,
            query_filter=q_filter,
            limit=top_k,
            with_payload=True,
        )
        hits: list[dict[str, Any]] = []
        for rank, pt in enumerate(response.points, start=1):
            payload = dict(pt.payload or {})
            chunk_id = str(payload.pop("chunk_id", pt.id))
            source_id = str(payload.pop("document_id", "") or payload.pop("source_id", ""))
            source = str(payload.pop("source", ""))
            page = int(payload.pop("page", 0))
            text = str(payload.pop("text", ""))
            payload.pop("strategy", None)
            hits.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source_id": source_id,
                    "source": source,
                    "page": page,
                    "metadata": payload,
                    "score": round(float(pt.score), 6),
                    "rank": rank,
                    "strategy": self.name,
                }
            )
        return hits


class HybridRrfRetriever:
    """Combine dense and lexical retrievers via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        dense: RetrieverProtocol,
        lexical: RetrieverProtocol,
        *,
        candidate_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        name: str = "hybrid_rrf",
    ) -> None:
        self.name = name
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        fetch_k = max(self.candidate_k, top_k)
        dense_rows = self.dense.search(query, filters=filters, top_k=fetch_k)
        lex_rows = self.lexical.search(query, filters=filters, top_k=fetch_k)

        by_id: dict[str, dict[str, Any]] = {}
        fused_scores: dict[str, float] = {}

        for weight, rows in ((self.dense_weight, dense_rows), (self.lexical_weight, lex_rows)):
            for idx, row in enumerate(rows, start=1):
                cid = row["id"]
                if cid not in by_id:
                    by_id[cid] = dict(row)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + weight / (self.rrf_k + idx)

        ranked_ids = sorted(fused_scores, key=lambda cid: (-fused_scores[cid], cid))[:top_k]
        results = [
            {**by_id[cid], "score": round(fused_scores[cid], 6)}
            for cid in ranked_ids
        ]
        return _with_ranks(results, self.name)


class NeighborExpandedContextPacker:
    """Deduplicate retrieved chunks and optionally expand with adjacent chunks on the same/next page."""

    def __init__(self, chunks: list[dict[str, Any]], budget: int = 2400, include_adjacent: bool = True) -> None:
        self.budget = budget
        self.include_adjacent = include_adjacent
        self._by_id = {c["id"]: c for c in chunks}
        # Index ordered list per source_id to support adjacent chunk lookup across page boundaries.
        self._ordered_by_source: dict[str, list[dict[str, Any]]] = {}
        self._pos_by_id: dict[str, tuple[str, int]] = {}
        for c in sorted(chunks, key=lambda x: (x["source_id"], int(x["page"]), int((x.get("metadata") or {}).get("chunk_index", 0)), x["id"])):
            sid = c["source_id"]
            lst = self._ordered_by_source.setdefault(sid, [])
            self._pos_by_id[c["id"]] = (sid, len(lst))
            lst.append(c)

    def pack(self, results: list[dict[str, Any]], *, max_primary: int = 3) -> tuple[str, list[str]]:
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        primary = results[:max_primary] if max_primary else results

        for c in primary:
            if c["id"] not in seen:
                seen.add(c["id"])
                ordered.append(c)

        if self.include_adjacent:
            # Add immediate neighbor chunks after top-ranked hits (prioritizing top-1/top-2).
            for c in primary[:2]:
                pos = self._pos_by_id.get(c["id"])
                if not pos:
                    continue
                sid, idx = pos
                src_list = self._ordered_by_source[sid]
                for offset in (1, -1):
                    n_idx = idx + offset
                    if 0 <= n_idx < len(src_list):
                        nb = src_list[n_idx]
                        if nb["id"] not in seen:
                            seen.add(nb["id"])
                            ordered.append(nb)

        blocks: list[str] = []
        ids: list[str] = []
        for c in ordered:
            block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
            candidate_text = "\n\n".join([*blocks, block])
            if len(candidate_text) <= self.budget:
                blocks.append(block)
                ids.append(c["id"])
        return "\n\n".join(blocks), ids
