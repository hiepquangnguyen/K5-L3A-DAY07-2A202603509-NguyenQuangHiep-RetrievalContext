"""CLI runner for Day 7 Basic Agent evaluations (Baseline vs Chosen Config)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from basic_agent import run_baseline_starter_agent, run_multitool_agent  # noqa: E402
from rerankers import OpenAIRerankRetriever  # noqa: E402
from retrievers import (  # noqa: E402
    Bm25Retriever,
    DenseQdrantRetriever,
    HybridRrfRetriever,
)

DEFAULT_DAY6_QDRANT = (
    ROOT.parent / "K5-L3A-Day06-NguyenQuangHiep-2A202603509-DataIngestion" / "starter" / "qdrant_db"
).resolve()


def summarize_traces(traces: list[dict[str, Any]], extra_cost_usd: float = 0.0, total_seconds: float = 0.0) -> dict[str, Any]:
    latencies = [float(t.get("elapsed_ms") or 0.0) for t in traces]
    searches = [sum(1 for e in t.get("events", []) if e.get("event") == "search") for t in traces]
    statuses = {"answered": 0, "abstain": 0, "error": 0}
    for t in traces:
        st = t.get("status", "error")
        statuses[st] = statuses.get(st, 0) + 1
    llm_tokens = sum(int(t.get("llm_tokens") or 0) for t in traces)
    agent_cost = sum(float(t.get("llm_cost_usd") or 0.0) for t in traces)
    p95 = 0.0
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    return {
        "n": len(traces),
        "statuses": statuses,
        "avg_searches": round(sum(searches) / len(searches), 2) if searches else 0.0,
        "median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_ms": round(p95, 2),
        "total_seconds": round(total_seconds, 2),
        "agent_llm_tokens": llm_tokens,
        "agent_cost_usd": round(agent_cost, 6),
        "retrieval_extra_cost_usd": round(extra_cost_usd, 6),
        "total_estimated_cost_usd": round(agent_cost + extra_cost_usd, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Baseline and/or Chosen Agentic RAG on basic benchmark split.")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--mode", choices=("baseline", "chosen", "both"), default="both")
    parser.add_argument("--cases", type=Path, default=ROOT / "starter" / "data" / "basic" / "cases.json")
    parser.add_argument("--chunks", type=Path, default=ROOT / "starter" / "data" / "basic" / "chunks.json")
    default_qdrant = os.getenv("DAY6_QDRANT_PATH", str(DEFAULT_DAY6_QDRANT))
    parser.add_argument("--qdrant", type=Path, default=Path(default_qdrant))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "starter" / "results" / "basic_agent")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke testing.")
    args = parser.parse_args()

    qdrant_path = args.qdrant if args.qdrant.is_absolute() else (ROOT / args.qdrant).resolve()
    all_cases = json.loads(args.cases.read_text(encoding="utf-8"))
    split_cases = [c for c in all_cases if c.get("split") == args.split]
    if args.limit:
        split_cases = split_cases[: args.limit]
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))

    bm25 = Bm25Retriever(chunks)
    dense = DenseQdrantRetriever(
        qdrant_path=qdrant_path,
        collection_name=os.getenv("DAY6_COLLECTION_NAME", "day06_moet_chunks"),
        strategy_filter=os.getenv("DAY6_STRATEGY", "C"),
        embed_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    try:
        modes = ["baseline", "chosen"] if args.mode == "both" else [args.mode]
        for mode in modes:
            traces: list[dict[str, Any]] = []
            reranker: OpenAIRerankRetriever | None = None
            before_embed_cost = dense.total_cost_usd
            before_embed_tokens = dense.total_tokens
            t0 = time.perf_counter()

            if mode == "baseline":
                for case in split_cases:
                    trace = run_baseline_starter_agent(case, dense, max_searches=4, top_k=3, budget=800)
                    traces.append(trace)
                    print(
                        f"[baseline/{args.split}] {case['id']} -> {trace['status']} "
                        f"cits={len(trace.get('citations', []))} ({trace['elapsed_ms']}ms)",
                        flush=True,
                    )
            else:
                hybrid = HybridRrfRetriever(dense, bm25, candidate_k=10, rrf_k=60)
                reranker = OpenAIRerankRetriever(hybrid, candidate_k=10, name="hybrid_rerank")
                for case in split_cases:
                    trace = run_multitool_agent(
                        case,
                        hybrid_retriever=reranker,
                        lexical_retriever=bm25,
                        chunks=chunks,
                        max_turns=5,
                        top_k=3,
                        budget=2400,
                    )
                    traces.append(trace)
                    print(
                        f"[chosen/{args.split}] {case['id']} -> {trace['status']} "
                        f"cits={len(trace.get('citations', []))} ({trace['elapsed_ms']}ms)",
                        flush=True,
                    )

            total_seconds = time.perf_counter() - t0
            embed_cost_delta = dense.total_cost_usd - before_embed_cost
            embed_tokens_delta = dense.total_tokens - before_embed_tokens
            rerank_cost = reranker.cost_usd if reranker else 0.0
            rerank_tokens = (reranker.prompt_tokens + reranker.completion_tokens) if reranker else 0

            out_dir = args.output_dir / args.split / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

            stats = summarize_traces(traces, extra_cost_usd=embed_cost_delta + rerank_cost, total_seconds=total_seconds)
            stats.update(
                {
                    "split": args.split,
                    "mode": mode,
                    "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
                    "embed_tokens": embed_tokens_delta,
                    "embed_cost_usd": round(embed_cost_delta, 6),
                    "rerank_tokens": rerank_tokens,
                    "rerank_cost_usd": round(rerank_cost, 6),
                }
            )
            (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved {mode} ({args.split}) summary:\n{json.dumps(stats, ensure_ascii=False, indent=2)}")
    finally:
        dense.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
