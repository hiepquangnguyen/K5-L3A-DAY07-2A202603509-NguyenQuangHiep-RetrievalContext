"""Offline unit tests for Advanced Multimodal Agent (mocked OpenAI + FitZ page PNGs)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advanced_agent import run_advanced_agent  # noqa: E402
from build_advanced_benchmark import get_advanced_cases, validate_advanced_benchmark  # noqa: E402
from retrievers import Bm25Retriever  # noqa: E402
from run_advanced import summarize_advanced_traces  # noqa: E402


def _make_tool_resp(name: str, args: dict, *, pt: int = 120, ct: int = 40):
    tc = SimpleNamespace(
        id=f"tc_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tc]))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


def _make_text_resp(text: str, *, pt: int = 300, ct: int = 60):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )


class FakeAdvancedOpenAI:
    def __init__(self, responses: list[object]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AdvancedAgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        self.pdf_path = self.tmp_dir / "sample.pdf"
        self.pages_dir = self.tmp_dir / "pages"
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=400, height=300)
        page.insert_text((40, 60), "Doanh thu 2025: 70.113 ty VND")
        doc.save(str(self.pdf_path))
        doc.close()
        self.chunks = [
            {
                "id": "adv:fpt2025:p001:v000",
                "text": "[Trang 1 - Vision | Kết quả 2025]\nDoanh thu 2025 đạt 70.113 tỷ VNĐ.",
                "source_id": "fpt-bctn-2025",
                "source": "FPT 2025",
                "page": 1,
                "metadata": {"modality": "multimodal", "section": "Kết quả 2025"},
            },
            {
                "id": "adv:fpt2025:p002:t000",
                "text": "[Trang 2 - Text | Ghi chú]\nKhông có thông tin cổ tức tiền mặt năm 2030.",
                "source_id": "fpt-bctn-2025",
                "source": "FPT 2025",
                "page": 2,
                "metadata": {"modality": "text", "section": "Ghi chú"},
            },
        ]
        self.bm25 = Bm25Retriever(self.chunks)

    def tearDown(self):
        self.tmp.cleanup()

    def test_advanced_agent_inspects_image_and_cites_verified_chunk(self):
        # Order of OpenAI calls:
        # 1. Agent decides search_multimodal_hybrid
        # 2. Agent decides inspect_page_image (which triggers 1 Vision chat completion!)
        # 3. Vision response inside inspect_page_image
        # 4. Agent decides submit_final_answer
        client = FakeAdvancedOpenAI(
            [
                _make_tool_resp("search_multimodal_hybrid", {"query": "doanh thu 2025"}),
                _make_tool_resp("inspect_page_image", {"page": 1, "focus_question": "Doanh thu 2025?", "region": "top"}),
                _make_text_resp("Trên biểu đồ trang 1, tổng doanh thu năm 2025 là 70.113 tỷ VNĐ."),
                _make_tool_resp(
                    "submit_final_answer",
                    {
                        "abstain": False,
                        "answer": "Tổng doanh thu năm 2025 đạt 70.113 tỷ VNĐ [adv:fpt2025:p001:v000].",
                        "citations": ["adv:fpt2025:p001:v000"],
                    },
                ),
            ]
        )
        trace = run_advanced_agent(
            {"id": "adv_t1", "split": "dev", "question": "Doanh thu năm 2025 là bao nhiêu?"},
            hybrid_retriever=self.bm25,
            lexical_retriever=self.bm25,
            chunks=self.chunks,
            pdf_path=self.pdf_path,
            pages_dir=self.pages_dir,
            client=client,
        )
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["adv:fpt2025:p001:v000"])
        self.assertTrue(any(e.get("tool") == "inspect_page_image" and "70.113" in e.get("visual_observation", "") for e in trace["events"]))

    def test_advanced_agent_abstains_cleanly(self):
        client = FakeAdvancedOpenAI(
            [
                _make_tool_resp("search_exact_keywords", {"query": "cổ tức 2030"}),
                _make_tool_resp("submit_final_answer", {"abstain": True, "answer": "", "citations": []}),
            ]
        )
        trace = run_advanced_agent(
            {"id": "adv_u1", "split": "dev", "question": "Cổ tức năm 2030 là bao nhiêu?"},
            hybrid_retriever=self.bm25,
            lexical_retriever=self.bm25,
            chunks=self.chunks,
            pdf_path=self.pdf_path,
            pages_dir=self.pages_dir,
            client=client,
        )
        self.assertEqual(trace["status"], "abstain")
        self.assertEqual(trace["citations"], [])

    def test_advanced_agent_rejects_evicted_inline_chunk_tag_and_recovers(self):
        from advanced_agent import _extract_invalid_inline_citations

        self.assertEqual(
            _extract_invalid_inline_citations(
                "Số liệu [2025] [12:30] hợp lệ [adv:fpt2025:p001:v000] và lạc [adv:fpt2025:p118:t002].",
                {"adv:fpt2025:p001:v000"},
            ),
            ["adv:fpt2025:p118:t002"],
        )

        client = FakeAdvancedOpenAI(
            [
                _make_tool_resp("read_page_full", {"page": 1}),
                _make_tool_resp(
                    "submit_final_answer",
                    {
                        "abstain": False,
                        "answer": "Doanh thu 2025 đạt 70.113 tỷ VNĐ [adv:fpt2025:p001:v000] [adv:fpt2025:p999:t999].",
                        "citations": ["adv:fpt2025:p001:v000"],
                    },
                ),
                _make_tool_resp(
                    "submit_final_answer",
                    {
                        "abstain": False,
                        "answer": "Doanh thu 2025 đạt 70.113 tỷ VNĐ [adv:fpt2025:p001:v000].",
                        "citations": ["adv:fpt2025:p001:v000"],
                    },
                ),
            ]
        )
        trace = run_advanced_agent(
            {"id": "adv_t_inline", "split": "dev", "question": "Doanh thu năm 2025 là bao nhiêu?"},
            hybrid_retriever=self.bm25,
            lexical_retriever=self.bm25,
            chunks=self.chunks,
            pdf_path=self.pdf_path,
            pages_dir=self.pages_dir,
            client=client,
        )
        self.assertEqual(trace["status"], "answered")
        self.assertEqual(trace["citations"], ["adv:fpt2025:p001:v000"])
        self.assertNotIn("adv:fpt2025:p999:t999", trace["answer"])
        self.assertTrue(any(e.get("event") == "invalid_or_missing_citations" for e in trace["events"]))

    def test_advanced_benchmark_disjoint_and_quotas(self):
        payload = get_advanced_cases()
        summary = validate_advanced_benchmark(payload)
        self.assertEqual(summary["page_overlap"], [])
        self.assertGreaterEqual(summary["dev_answerable"], 15)
        self.assertGreaterEqual(summary["dev_unanswerable"], 3)
        self.assertGreaterEqual(summary["test_total"], 30)

        broken = {
            "dev": list(payload["dev"]),
            "test": [{**payload["test"][0], "source_pages": [payload["dev"][0]["source_pages"][0]]}] + payload["test"][1:],
        }
        with self.assertRaises(ValueError):
            validate_advanced_benchmark(broken)

    def test_summarize_advanced_traces(self):
        traces = [
            {
                "query_id": "q1",
                "status": "answered",
                "elapsed_ms": 1000.0,
                "llm_tokens": 500,
                "llm_cost_usd": 0.001,
                "events": [{"event": "search", "tool": "search_multimodal_hybrid"}, {"event": "search", "tool": "inspect_page_image"}],
            },
            {
                "query_id": "u1",
                "status": "abstain",
                "elapsed_ms": 500.0,
                "llm_tokens": 200,
                "llm_cost_usd": 0.0005,
                "events": [{"event": "search", "tool": "search_exact_keywords"}, {"event": "abstain"}],
            },
        ]
        stats = summarize_advanced_traces(
            traces,
            embed_cost_usd=0.0002,
            embed_tokens=100,
            rerank_cost_usd=0.0003,
            rerank_tokens=150,
            total_seconds=1.5,
            split="dev",
            chat_model="gpt-4.1-mini",
            vision_model="gpt-4.1-mini",
            embed_model="text-embedding-3-large",
        )
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["statuses"]["answered"], 1)
        self.assertEqual(stats["statuses"]["abstain"], 1)
        self.assertEqual(stats["avg_tool_calls"], 1.5)
        self.assertAlmostEqual(stats["total_estimated_cost_usd"], 0.002, places=6)


if __name__ == "__main__":
    unittest.main()
