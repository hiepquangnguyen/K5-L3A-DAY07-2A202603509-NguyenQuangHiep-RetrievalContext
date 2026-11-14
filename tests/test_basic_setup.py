"""Offline unit tests for Day 7 Step 1 helpers (exporting Day 6 schema and benchmark validation)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

from baseline import hit, validate  # noqa: E402
from export_day6 import export_questions  # noqa: E402


class Step1SetupTests(unittest.TestCase):
    def test_export_questions_detects_missing_gold_match(self):
        chunks = [
            {
                "id": "para:doc:p01:c000",
                "text": "Quy định chung về học phần.",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 1,
                "metadata": {},
            }
        ]
        raw_q = [
            {
                "id": "q1",
                "split": "dev",
                "question": "Quy định gì?",
                "expected_answer": "Quy định chung",
                "gold_source_id": "doc",
                "gold_page": 1,
                "evidence_quote": "Quy định chung",
            },
            {
                "id": "q2",
                "split": "dev",
                "question": "Thiếu?",
                "expected_answer": "Không khớp",
                "gold_source_id": "doc",
                "gold_page": 1,
                "evidence_quote": "Đoạn không tồn tại",
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            qpath = Path(td) / "q.json"
            qpath.write_text(json.dumps(raw_q, ensure_ascii=False), encoding="utf-8")
            converted, issues = export_questions(qpath, chunks)
        self.assertEqual(len(converted), 2)
        self.assertEqual([i["id"] for i in issues], ["q2"])
        self.assertTrue(hit(chunks[0], converted[0]["gold"]))
        validate(chunks, [converted[0]])


if __name__ == "__main__":
    unittest.main()
