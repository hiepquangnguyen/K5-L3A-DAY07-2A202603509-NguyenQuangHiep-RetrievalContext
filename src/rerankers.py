"""LLM-based reranker, query rewriter, and OpenAI usage/cost accounting helpers."""
from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from retrievers import RetrieverProtocol, _with_ranks

CHAT_PRICING_PER_1M_USD: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


def estimate_chat_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts using published per-1M token rates."""
    for prefix, (in_rate, out_rate) in CHAT_PRICING_PER_1M_USD.items():
        if model.startswith(prefix):
            return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate
    return 0.0


def _build_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=key, base_url=url, timeout=45.0)


class OpenAIRerankRetriever:
    """Two-stage retriever: fetch candidate_k from base retriever, then rerank with an OpenAI model."""

    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        *,
        candidate_k: int = 10,
        model: str | None = None,
        client: Any | None = None,
        name: str = "hybrid_rerank",
    ) -> None:
        self.name = name
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.client = client or _build_client()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.rerank_seconds = 0.0

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        candidates = self.base_retriever.search(query, filters=filters, top_k=max(self.candidate_k, top_k))
        if len(candidates) <= 1:
            return _with_ranks(candidates[:top_k], self.name)

        snippets = []
        for idx, c in enumerate(candidates):
            sec = (c.get("metadata") or {}).get("section", "")
            snippets.append(
                f"[{idx}] id={c['id']} | source={c['source_id']} p.{c['page']} | section={sec}\n{c['text'][:700]}"
            )
        prompt = (
            "Hãy xếp hạng các đoạn tài liệu sau theo mức độ trả lời trực tiếp và đầy đủ câu hỏi.\n"
            "Chỉ trả về JSON hợp lệ dạng {\"order\": [chỉ_số_nguyên...]} liệt kê các chỉ số từ phù hợp nhất đến ít nhất.\n"
            f"Câu hỏi: {query}\n\n"
            "Các đoạn ứng viên:\n" + "\n\n".join(snippets)
        )
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Bạn là bộ xếp hạng bằng chứng retrieval chính xác. Luôn trả JSON hợp lệ."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self.rerank_seconds += time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        if usage:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.cost_usd += estimate_chat_cost_usd(self.model, pt, ct)

        raw_content = (resp.choices[0].message.content or "{}").strip()
        parsed = json.loads(raw_content)
        raw_order = parsed.get("order", [])
        seen: set[int] = set()
        reordered: list[dict[str, Any]] = []
        for idx in raw_order:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                row = dict(candidates[idx])
                row["base_score"] = row.get("score")
                row["score"] = round(1.0 / (len(reordered) + 1), 6)
                reordered.append(row)
        for idx, c in enumerate(candidates):
            if idx not in seen:
                row = dict(c)
                row["base_score"] = row.get("score")
                row["score"] = round(1.0 / (len(reordered) + 1), 6)
                reordered.append(row)
        return _with_ranks(reordered[:top_k], self.name)


class OpenAIQueryRewriteRetriever:
    """Expand/rewrite the original query via OpenAI and fuse rankings across original + rewritten queries."""

    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        *,
        candidate_k: int = 8,
        rrf_k: int = 60,
        model: str | None = None,
        client: Any | None = None,
        name: str = "query_rewrite_rrf",
    ) -> None:
        self.name = name
        self.base_retriever = base_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
        self.client = client or _build_client()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.last_rewrites: list[str] = []

    def _rewrite(self, query: str) -> list[str]:
        prompt = (
            "Hãy viết lại câu hỏi sau thành tối đa 2 truy vấn tìm kiếm ngắn gọn giàu từ khóa chuyên ngành quy chế/văn bản "
            "để cải thiện tìm kiếm tài liệu. Trả JSON dạng {\"queries\": [\"...\"]}.\n"
            f"Câu hỏi gốc: {query}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        usage = getattr(resp, "usage", None)
        if usage:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.cost_usd += estimate_chat_cost_usd(self.model, pt, ct)
        parsed = json.loads((resp.choices[0].message.content or "{}").strip())
        rewrites = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()][:2]
        return rewrites

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        rewrites = self._rewrite(query)
        self.last_rewrites = rewrites
        all_queries = [query, *rewrites]

        by_id: dict[str, dict[str, Any]] = {}
        fused: dict[str, float] = {}
        for q_idx, q_str in enumerate(all_queries):
            weight = 1.2 if q_idx == 0 else 0.9
            rows = self.base_retriever.search(q_str, filters=filters, top_k=max(self.candidate_k, top_k))
            for rank_idx, row in enumerate(rows, start=1):
                cid = row["id"]
                if cid not in by_id:
                    by_id[cid] = dict(row)
                fused[cid] = fused.get(cid, 0.0) + weight / (self.rrf_k + rank_idx)

        ranked_ids = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
        results = [{**by_id[cid], "score": round(fused[cid], 6), "rewritten_queries": rewrites} for cid in ranked_ids]
        return _with_ranks(results, self.name)
