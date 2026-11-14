"""Offline unit tests for OpenAI single-tool and multi-tool basic agents (mocked LLM + local retriever)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))

from basic_agent import (  # noqa: E402
    OpenAISingleToolController,
    run_baseline_starter_agent,
    run_multitool_agent,
)
from retrievers import Bm25Retriever, FilteredTfidfRetriever  # noqa: E402


CHUNKS = [
    {
        "id": "doc:p01:c000",
        "text": "Quy định chung về chuẩn ngoại ngữ tiến sĩ: chứng chỉ IELTS đạt từ 5.5 trở lên.",
        "source_id": "tt18-2021",
        "source": "TT 18",
        "page": 1,
        "metadata": {"cap_dao_tao": "tien_si", "section": "Điều 5"},
    },
    {
        "id": "doc:p01:c001",
        "text": "Trong thời gian tối đa 03 tháng kể từ khi đạt phản biện độc lập, cơ sở đào tạo thành lập Hội đồng.",
        "source_id": "tt18-2021",
        "source": "TT 18",
        "page": 1,
        "metadata": {"cap_dao_tao": "tien_si", "section": "Điều 16"},
    },
    {
        "id": "doc:p01:c002",
        "text": "Cha, mẹ, vợ hoặc chồng, con, anh, chị, em ruột của nghiên cứu sinh không tham gia Hội đồng.",
        "source_id": "tt18-2021",
        "source": "TT 18",
        "page": 1,
        "metadata": {"cap_dao_tao": "tien_si", "section": "Điều 16"},
    },
]


def _make_tool_response(name: str, args: dict, *, pt: int = 100, ct: int = 30, call_id: str = "tc_1"):
    tc = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )
    msg = SimpleNamespace(content="", tool_calls=[tc])
    usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


class FakeChatClient:
    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("No more fake responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class BasicAgentTests(unittest.TestCase):
    def test_single_tool_controller_search_and_answer(self):
        client = FakeChatClient(
            [
                _make_tool_response("search", {"query": "IELTS tiến sĩ"}),
                _make_tool_response(
                    "answer",
                    {
                        "abstain": False,
                        "answer": "Yêu cầu IELTS từ 5.5 trở lên [doc:p01:c000].",
                        "citations": ["doc:p01:c000"],
                    },
                ),
            ]
        )
        retriever = FilteredTfidfRetriever(CHUNKS)
        case = {"id": "q1", "split": "dev", "question": "IELTS tối thiểu là bao nhiêu?"}
        trace = run_baseline_starter_agent(case, retriever, client=client, budget=800)
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["doc:p01:c000"])
        self.assertGreater(trace["llm_tokens"], 0)
        self.assertEqual(len(client.calls), 2)

    def test_single_tool_controller_abstain_and_error(self):
        abstain_client = FakeChatClient(
            [
                _make_tool_response("search", {"query": "học bổng"}),
                _make_tool_response("answer", {"abstain": True, "answer": "", "citations": []}),
            ]
        )
        retriever = FilteredTfidfRetriever(CHUNKS)
        trace = run_baseline_starter_agent(
            {"id": "u1", "split": "dev", "question": "Mức học bổng là bao nhiêu?"},
            retriever,
            client=abstain_client,
        )
        self.assertEqual(trace["status"], "abstain")
        self.assertEqual(trace["citations"], [])

        err_client = FakeChatClient([RuntimeError("API 503")])
        err_trace = run_baseline_starter_agent(
            {"id": "e1", "split": "dev", "question": "Hỏi gì đó"},
            retriever,
            client=err_client,
        )
        self.assertEqual(err_trace["status"], "error")
        self.assertIn("API 503", err_trace["error"])

    def test_multitool_agent_reads_adjacent_and_cites_both(self):
        client = FakeChatClient(
            [
                _make_tool_response("search_hybrid", {"query": "thời hạn thành lập Hội đồng", "cap_dao_tao": "tien_si", "top_k": 1}),
                _make_tool_response("read_adjacent_chunks", {"chunk_id": "doc:p01:c001"}),
                _make_tool_response(
                    "submit_final_answer",
                    {
                        "abstain": False,
                        "answer": "Tối đa 03 tháng [doc:p01:c001]; cha mẹ vợ chồng con anh chị em ruột không tham gia [doc:p01:c002].",
                        "citations": ["doc:p01:c001", "doc:p01:c002"],
                    },
                ),
            ]
        )
        bm25 = Bm25Retriever(CHUNKS)
        case = {"id": "q2", "split": "dev", "question": "Thời hạn lập Hội đồng và ai không được tham gia?"}
        trace = run_multitool_agent(
            case,
            hybrid_retriever=bm25,
            lexical_retriever=bm25,
            chunks=CHUNKS,
            client=client,
            budget=2400,
        )
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["doc:p01:c001", "doc:p01:c002"])
        search_events = [e for e in trace["events"] if e["event"] == "search"]
        self.assertEqual([e["tool"] for e in search_events], ["search_hybrid", "read_adjacent_chunks"])

    def test_multitool_agent_recovers_inline_citations_and_retries_invalid(self):
        # First submit contains a fake chunk id -> agent sends feedback tool message -> second submit succeeds with comma-separated inline bracket.
        client = FakeChatClient(
            [
                _make_tool_response("search_exact_keywords", {"query": "IELTS 5.5"}),
                _make_tool_response(
                    "submit_final_answer",
                    {
                        "abstain": False,
                        "answer": "IELTS từ 5.5 [fake:chunk].",
                        "citations": ["fake:chunk"],
                    },
                ),
                _make_tool_response(
                    "submit_final_answer",
                    {
                        "abstain": False,
                        "answer": "IELTS từ 5.5 trở lên [doc:p01:c000].",
                        "citations": [],
                    },
                ),
            ]
        )
        bm25 = Bm25Retriever(CHUNKS)
        trace = run_multitool_agent(
            {"id": "q3", "split": "dev", "question": "IELTS?"},
            hybrid_retriever=bm25,
            lexical_retriever=bm25,
            chunks=CHUNKS,
            client=client,
            budget=2400,
        )
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["doc:p01:c000"])
        self.assertTrue(any(e["event"] == "invalid_or_missing_citations" for e in trace["events"]))


if __name__ == "__main__":
    unittest.main()
