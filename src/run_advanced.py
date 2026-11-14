"""CLI runner for Advanced Multimodal Agentic RAG on FPT_BCTN_2025 (232 pages)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from advanced_agent import run_advanced_agent  # noqa: E402
from rerankers import OpenAIRerankRetriever  # noqa: E402
from retrievers import Bm25Retriever, DenseQdrantRetriever, HybridRrfRetriever  # noqa: E402


def summarize_advanced_traces(
    traces: list[dict[str, Any]],
    *,
    embed_cost_usd: float,
    embed_tokens: int,
    rerank_cost_usd: float,
    rerank_tokens: int,
    total_seconds: float,
    split: str,
    chat_model: str,
    vision_model: str,
    embed_model: str,
) -> dict[str, Any]:
    latencies = [float(t.get("elapsed_ms") or 0.0) for t in traces]
    tool_counts: Counter[str] = Counter()
    searches = []
    statuses = {"answered": 0, "abstain": 0, "error": 0}
    for t in traces:
        st = t.get("status", "error")
        statuses[st] = statuses.get(st, 0) + 1
        s_count = 0
        for e in t.get("events", []):
            if e.get("event") == "search":
                s_count += 1
                tool_counts[str(e.get("tool") or "search")] += 1
        searches.append(s_count)

    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))] if ordered else 0.0
    agent_tokens = sum(int(t.get("llm_tokens") or 0) for t in traces)
    agent_cost = sum(float(t.get("llm_cost_usd") or 0.0) for t in traces)
    return {
        "split": split,
        "n": len(traces),
        "statuses": statuses,
        "avg_tool_calls": round(sum(searches) / len(searches), 2) if searches else 0.0,
        "tool_breakdown": dict(tool_counts),
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_ms": round(p95, 2),
        "total_seconds": round(total_seconds, 2),
        "chat_model": chat_model,
        "vision_model": vision_model,
        "embed_model": embed_model,
        "agent_llm_tokens": agent_tokens,
        "agent_cost_usd": round(agent_cost, 6),
        "embed_tokens": embed_tokens,
        "embed_cost_usd": round(embed_cost_usd, 6),
        "rerank_tokens": rerank_tokens,
        "rerank_cost_usd": round(rerank_cost_usd, 6),
        "total_estimated_cost_usd": round(agent_cost + embed_cost_usd + rerank_cost_usd, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Advanced Multimodal Agent on dev/test split.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "starter" / "data" / "advanced" / "cases.json")
    parser.add_argument("--chunks", type=Path, default=ROOT / "starter" / "data" / "advanced" / "chunks.json")
    parser.add_argument("--qdrant-dir", type=Path, default=ROOT / "starter" / "data" / "advanced" / "qdrant_db")
    parser.add_argument("--collection", default=os.getenv("ADVANCED_COLLECTION_NAME", "fpt_bctn_2025"))
    parser.add_argument("--pdf", type=Path, default=ROOT / "starter" / "data" / "advanced" / "raw" / "FPT_BCTN_2025.pdf")
    parser.add_argument("--pages-dir", type=Path, default=ROOT / "starter" / "data" / "advanced" / "pages")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "starter" / "results" / "advanced")
    parser.add_argument("--chat-model", default=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--vision-model", default=os.getenv("ADVANCED_VISION_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--embed-model", default=os.getenv("ADVANCED_EMBEDDING_MODEL", "text-embedding-3-large"))
    parser.add_argument("--use-reranker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--budget", type=int, default=8000)
    parser.add_argument("--max-turns", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raw_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if isinstance(raw_cases, dict):
        cases = [{**c, "split": c.get("split") or args.split} for c in (raw_cases.get(args.split) or [])]
    else:
        cases = [{**c, "split": c.get("split") or args.split} for c in raw_cases if c.get("split") == args.split]
    if args.limit:
        cases = cases[: args.limit]
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))

    bm25 = Bm25Retriever(chunks, name="bm25_advanced")
    dense = DenseQdrantRetriever(
        qdrant_path=args.qdrant_dir,
        collection_name=args.collection,
        strategy_filter=None,
        embed_model=args.embed_model,
        name="dense_qdrant_large",
    )
    try:
        hybrid = HybridRrfRetriever(dense, bm25, candidate_k=15, rrf_k=60, name="hybrid_rrf_advanced")
        reranker = (
            OpenAIRerankRetriever(hybrid, candidate_k=12, model=args.chat_model, name="hybrid_rerank_advanced")
            if args.use_reranker
            else None
        )
        retriever = reranker if reranker is not None else hybrid

        traces: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        for c in cases:
            tr = run_advanced_agent(
                c,
                hybrid_retriever=retriever,
                lexical_retriever=bm25,
                chunks=chunks,
                pdf_path=args.pdf,
                pages_dir=args.pages_dir,
                chat_model=args.chat_model,
                vision_model=args.vision_model,
                max_turns=args.max_turns,
                top_k=5,
                budget=args.budget,
            )
            traces.append(tr)
            tools = [e.get("tool") or e.get("event") for e in tr.get("events", [])]
            print(
                f"[adv/{args.split}] {c['id']} ({c.get('category', '')}) -> {tr['status']} "
                f"cits={len(tr.get('citations', []))} tools={tools} ({tr['elapsed_ms']}ms)",
                flush=True,
            )
        total_seconds = time.perf_counter() - t0
        out_dir = args.output_dir / args.split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

        stats = summarize_advanced_traces(
            traces,
            embed_cost_usd=dense.total_cost_usd,
            embed_tokens=dense.total_tokens,
            rerank_cost_usd=reranker.cost_usd if reranker else 0.0,
            rerank_tokens=(reranker.prompt_tokens + reranker.completion_tokens) if reranker else 0,
            total_seconds=total_seconds,
            split=args.split,
            chat_model=args.chat_model,
            vision_model=args.vision_model,
            embed_model=args.embed_model,
        )
        (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved Advanced ({args.split}) summary:\n{json.dumps(stats, ensure_ascii=False, indent=2)}")
    finally:
        dense.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
