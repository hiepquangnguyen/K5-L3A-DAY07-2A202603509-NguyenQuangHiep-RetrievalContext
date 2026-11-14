"""Unit tests for Advanced Challenge multimodal ingestion helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advanced_ingest import (  # noqa: E402
    build_multimodal_chunks,
    clean_pdf_text,
    split_text_windows,
)


class AdvancedIngestTests(unittest.TestCase):
    def test_clean_pdf_text_and_window_overlap(self):
        raw = "  Dòng   một  \n\n\n\n  Dòng hai   rất dài " + ("abc " * 300)
        cleaned = clean_pdf_text(raw)
        self.assertNotIn("   ", cleaned)
        windows = split_text_windows(cleaned, max_chars=300, overlap=60)
        self.assertGreater(len(windows), 2)
        self.assertTrue(all(len(w) <= 300 for w in windows))

    def test_build_multimodal_chunks_links_page_and_image_path(self):
        pages = [
            {
                "page": 10,
                "image_path": "starter/data/advanced/pages/p010.png",
                "abs_image_path": "p010.png",
                "raw_text": "CON SỐ NỔI BẬT\nTổng doanh thu 70.113 tỷ VNĐ",
                "num_images": 0,
                "num_drawings": 23,
            }
        ]
        vision = {
            10: {
                "page": 10,
                "page_title": "Con số nổi bật 2025",
                "content_types": ["chart", "text"],
                "visual_summary": "4 biểu đồ cột tăng trưởng từ 2021 đến 2025",
                "charts_and_tables": "| Năm | Tổng doanh thu |\n|---|---|\n| 2025 | 70.113 |",
            }
        }
        chunks = build_multimodal_chunks(pages, vision)
        modalities = [c["metadata"]["modality"] for c in chunks]
        self.assertEqual(modalities, ["text", "multimodal"])
        for c in chunks:
            self.assertEqual(c["page"], 10)
            self.assertEqual(c["metadata"]["image_path"], "starter/data/advanced/pages/p010.png")
        self.assertIn("70.113", chunks[1]["text"])


if __name__ == "__main__":
    unittest.main()
