"""Offline unit tests for Day 7 retrievers, RRF fusion, metadata filters, and context packing."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrievers import (  # noqa: E402
    Bm25Retriever,
    FilteredTfidfRetriever,
    HybridRrfRetriever,
    NeighborExpandedContextPacker,
)


def _chunk(cid: str, text: str, source_id: str = "doc1", page: int = 1, cap: str = "dai_hoc", idx: int = 0) -> dict:
    return {
        "id": cid,
        "text": text,
        "source_id": source_id,
        "source": f"{source_id}.pdf",
        "page": page,
        "metadata": {"cap_dao_tao": cap, "chunk_index": idx},
    }


class FakeDense:
    name = "fake_dense"

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def search(self, query: str, *, filters=None, top_k: int = 3) -> list[dict]:
        return self.rows[:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            _chunk("c1", "Quy chế đại học: sinh viên học trực tuyến tối đa 30% khối lượng.", "tt08", 7, "dai_hoc", 0),
            _chunk("c2", "Quy chế tiến sĩ: thời gian đào tạo tối đa 06 năm.", "tt18", 3, "tien_si", 0),
            _chunk("c3", "Phụ lục Bảng điểm chuẩn ngoại ngữ bậc tiến sĩ IELTS 5.5.", "tt18", 22, "tien_si", 0),
            _chunk("c4", "Tiếp theo trang 11: thang điểm 10 xếp loại Xuất sắc 9-10.", "tt08", 11, "dai_hoc", 0),
            _chunk("c5", "Trang 10: mở đầu xếp loại học lực theo thang điểm 4.", "tt08", 10, "dai_hoc", 0),
        ]

    def test_bm25_and_tfidf_respect_metadata_filters(self) -> None:
        bm25 = Bm25Retriever(self.chunks)
        tfidf = FilteredTfidfRetriever(self.chunks)
        res_bm25 = bm25.search("tối đa", filters={"cap_dao_tao": "tien_si"}, top_k=3)
        res_tfidf = tfidf.search("tối đa", filters={"cap_dao_tao": "tien_si"}, top_k=3)
        self.assertEqual([r["id"] for r in res_bm25], ["c2"])
        self.assertEqual([r["id"] for r in res_tfidf], ["c2"])

    def test_hybrid_rrf_combines_dense_and_lexical_without_duplicates(self) -> None:
        bm25 = Bm25Retriever(self.chunks)
        dense = FakeDense([self.chunks[1], self.chunks[0]])
        hybrid = HybridRrfRetriever(dense, bm25, candidate_k=3)
        res = hybrid.search("tiến sĩ tối đa 06 năm", top_k=3)
        ids = [r["id"] for r in res]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "c2")

    def test_neighbor_packer_captures_adjacent_cross_page_chunk(self) -> None:
        packer = NeighborExpandedContextPacker(self.chunks, budget=2400, include_adjacent=True)
        context, ids = packer.pack([self.chunks[4]], max_primary=1)  # c5 on page 10
        self.assertIn("c5", ids)
        self.assertIn("c4", ids)  # c4 on page 11 should be appended as neighbor
        self.assertLessEqual(len(context), 2400)


if __name__ == "__main__":
    unittest.main()
