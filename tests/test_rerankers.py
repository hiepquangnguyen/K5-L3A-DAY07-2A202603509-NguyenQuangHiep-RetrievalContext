"""Offline unit tests for OpenAI reranker/rewriter wrapper logic with mocked client."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rerankers import OpenAIRerankRetriever, estimate_chat_cost_usd  # noqa: E402


class FakeBaseRetriever:
    name = "fake_base"

    def search(self, query: str, *, filters=None, top_k: int = 3):
        return [
            {"id": "c0", "text": "Ít liên quan", "source_id": "d", "source": "d.pdf", "page": 1, "metadata": {}, "score": 0.9},
            {"id": "c1", "text": "Đúng đáp án cần tìm", "source_id": "d", "source": "d.pdf", "page": 2, "metadata": {}, "score": 0.5},
        ][:top_k]


class FakeOpenAI:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"order": [1, 0, 99]})))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
                )


class RerankerTests(unittest.TestCase):
    def test_rerank_reorders_safely_and_tracks_cost(self):
        rr = OpenAIRerankRetriever(FakeBaseRetriever(), candidate_k=5, model="gpt-4.1-mini", client=FakeOpenAI())
        res = rr.search("câu hỏi", top_k=2)
        self.assertEqual([r["id"] for r in res], ["c1", "c0"])
        self.assertEqual(rr.prompt_tokens, 100)
        self.assertEqual(rr.completion_tokens, 20)
        self.assertAlmostEqual(estimate_chat_cost_usd("gpt-4.1-mini", 1_000_000, 1_000_000), 2.0)


if __name__ == "__main__":
    unittest.main()
