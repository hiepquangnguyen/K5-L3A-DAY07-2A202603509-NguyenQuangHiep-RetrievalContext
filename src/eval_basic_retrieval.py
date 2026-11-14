"""Benchmark multiple retrieval & context packing configs on a specified split (dev or test)."""
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
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))

from baseline import hit, pack_context, validate  # noqa: E402
from rerankers import OpenAIQueryRewriteRetriever, OpenAIRerankRetriever  # noqa: E402
from retrievers import (  # noqa: E402
    Bm25Retriever,
    DenseQdrantRetriever,
    FilteredTfidfRetriever,
    HybridRrfRetriever,
    NeighborExpandedContextPacker,
)


def _all_gold_in_context(context: str, results_in_ctx: list[dict[str, Any]], gold: list[dict[str, Any]], multi: bool) -> bool:
    """Check evidence coverage in packed context (all clauses required when multi=True)."""
    if not multi:
        return any(hit(c, gold) for c in results_in_ctx)
    return all(any(hit(c, [g]) for c in results_in_ctx) for g in gold)


def _all_gold_in_topk(results: list[dict[str, Any]], gold: list[dict[str, Any]], multi: bool) -> bool:
    if not multi:
        return any(hit(c, gold) for c in results)
    return all(any(hit(c, [g]) for c in results) for g in gold)


def run_config(
    config_name: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    packer: NeighborExpandedContextPacker | None = None,
    budget: int = 800,
    top_k: int = 3,
    dense_ref: DenseQdrantRetriever | None = None,
    llm_ref: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    t_start = time.perf_counter()
    dense_tokens_before = dense_ref.total_tokens if dense_ref else 0
    dense_cost_before = dense_ref.total_cost_usd if dense_ref else 0.0
    llm_pt_before = getattr(llm_ref, "prompt_tokens", 0) if llm_ref else 0
    llm_ct_before = getattr(llm_ref, "completion_tokens", 0) if llm_ref else 0
    llm_cost_before = getattr(llm_ref, "cost_usd", 0.0) if llm_ref else 0.0

    traces: list[dict[str, Any]] = []
    for q in questions:
        q_filter = q.get("filter")
        multi = bool(q.get("multi_evidence", False))
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=q_filter, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if packer is not None:
            context, ids = packer.pack(results, max_primary=top_k)
            # Lookup packed chunks (including expanded neighbors) from packer map.
            packed_chunks = [packer._by_id[cid] for cid in ids if cid in packer._by_id]
        else:
            context, ids = pack_context(results, budget=budget)
            by_id = {c["id"]: c for c in results}
            packed_chunks = [by_id[cid] for cid in ids if cid in by_id]

        hit1 = bool(results and hit(results[0], q["gold"]))
        hit3_any = any(hit(c, q["gold"]) for c in results[:3])
        hit3_strict = _all_gold_in_topk(results[:3], q["gold"], multi)
        ctx_hit = _all_gold_in_context(context, packed_chunks, q["gold"], multi)

        trace_row: dict[str, Any] = {
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "filter": q_filter,
            "multi_evidence": multi,
            "config": config_name,
            "top_k": top_k,
            "budget_chars": packer.budget if packer else budget,
            "results": [
                {
                    "id": c["id"],
                    "source_id": c["source_id"],
                    "page": c["page"],
                    "score": c.get("score"),
                    "rank": c.get("rank"),
                    "hit": hit(c, q["gold"]),
                }
                for c in results
            ],
            "context_ids": ids,
            "context_chars": len(context),
            "elapsed_ms": round(elapsed_ms, 2),
            "hit1": hit1,
            "hit3": hit3_any,
            "hit3_strict": hit3_strict,
            "context_hit": ctx_hit,
        }
        if hasattr(retriever, "last_rewrites"):
            trace_row["rewritten_queries"] = list(retriever.last_rewrites)
        traces.append(trace_row)

    total_seconds = time.perf_counter() - t_start
    n = len(traces)
    dense_tokens = (dense_ref.total_tokens - dense_tokens_before) if dense_ref else 0
    dense_cost = (dense_ref.total_cost_usd - dense_cost_before) if dense_ref else 0.0
    llm_tokens = (
        (getattr(llm_ref, "prompt_tokens", 0) - llm_pt_before)
        + (getattr(llm_ref, "completion_tokens", 0) - llm_ct_before)
        if llm_ref
        else 0
    )
    llm_cost = (getattr(llm_ref, "cost_usd", 0.0) - llm_cost_before) if llm_ref else 0.0

    summary = {
        "config": config_name,
        "split": questions[0]["split"] if questions else None,
        "n": n,
        "hit1": sum(t["hit1"] for t in traces),
        "hit1_rate": round(sum(t["hit1"] for t in traces) / n, 4),
        "hit3": sum(t["hit3"] for t in traces),
        "hit3_rate": round(sum(t["hit3"] for t in traces) / n, 4),
        "hit3_strict": sum(t["hit3_strict"] for t in traces),
        "context_hit": sum(t["context_hit"] for t in traces),
        "context_hit_rate": round(sum(t["context_hit"] for t in traces) / n, 4),
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
        "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2),
        "total_seconds": round(total_seconds, 3),
        "embed_tokens": dense_tokens,
        "llm_tokens": llm_tokens,
        "estimated_cost_usd": round(dense_cost + llm_cost, 6),
    }
    return summary, traces


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run Day 7 retrieval benchmark suite on dev or test.")
    parser.add_argument("--chunks", type=Path, default=ROOT / "starter" / "data" / "basic" / "chunks.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "starter" / "data" / "basic" / "questions.json")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--configs", nargs="*", default=None, help="Optional subset of config names to run.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "starter" / "results" / "basic_retrieval")
    args = parser.parse_args()

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    all_questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [q for q in all_questions if q.get("split") == args.split]
    validate(chunks, selected)

    qdrant_path_str = os.getenv(
        "DAY6_QDRANT_PATH",
        str((ROOT.parent / "K5-L3A-DAY06-NguyenQuangHiep-2A202603509-DataIngestion/starter/qdrant_db").resolve()),
    )
    qdrant_path = Path(qdrant_path_str)
    if not qdrant_path.is_absolute():
        qdrant_path = (ROOT / qdrant_path).resolve()

    tfidf = FilteredTfidfRetriever(chunks, name="tfidf")
    bm25 = Bm25Retriever(chunks, name="bm25")
    dense = DenseQdrantRetriever(
        qdrant_path=qdrant_path,
        collection_name=os.getenv("DAY6_COLLECTION_NAME", "day06_moet_chunks"),
        strategy_filter=os.getenv("DAY6_STRATEGY", "C"),
        embed_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        name="dense_qdrant",
    )
    hybrid = HybridRrfRetriever(dense, bm25, candidate_k=10, rrf_k=60, name="hybrid_rrf")
    reranker = OpenAIRerankRetriever(hybrid, candidate_k=10, name="hybrid_rerank")
    rewriter = OpenAIQueryRewriteRetriever(hybrid, candidate_k=8, name="query_rewrite_rrf")
    packer_2400 = NeighborExpandedContextPacker(chunks, budget=2400, include_adjacent=True)

    suite = [
        ("tfidf_baseline_800", tfidf, None, 800, None, None),
        ("bm25_800", bm25, None, 800, None, None),
        ("dense_qdrant_800", dense, None, 800, dense, None),
        ("hybrid_rrf_800", hybrid, None, 800, dense, None),
        ("hybrid_rrf_pack2400", hybrid, packer_2400, 2400, dense, None),
        ("hybrid_rerank_pack2400", reranker, packer_2400, 2400, dense, reranker),
        ("query_rewrite_pack2400", rewriter, packer_2400, 2400, dense, rewriter),
    ]
    if args.configs:
        wanted = set(args.configs)
        suite = [row for row in suite if row[0] in wanted]

    out_root = args.output_dir / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    try:
        for cfg_name, ret, pkr, bdg, d_ref, l_ref in suite:
            summary, traces = run_config(
                cfg_name,
                ret,
                selected,
                packer=pkr,
                budget=bdg,
                top_k=3,
                dense_ref=d_ref,
                llm_ref=l_ref,
            )
            summaries.append(summary)
            cfg_dir = out_root / cfg_name
            cfg_dir.mkdir(parents=True, exist_ok=True)
            (cfg_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (cfg_dir / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        dense.close()

    (out_root / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
