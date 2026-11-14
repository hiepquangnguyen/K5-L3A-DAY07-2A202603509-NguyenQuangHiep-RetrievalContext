import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo_server import DemoAppContext, make_handler  # noqa: E402


class DemoServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ctx = DemoAppContext(root=ROOT)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.ctx))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.ctx.close()

    def _get(self, path: str) -> tuple[int, str, bytes]:
        with urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
            return resp.status, resp.headers.get("Content-Type", ""), resp.read()

    def test_serves_frontend_and_blocks_traversal(self) -> None:
        status, ctype, html = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn("Day 7 Agentic RAG", html.decode("utf-8"))

        status_js, ctype_js, _ = self._get("/assets/app.js")
        self.assertEqual(status_js, 200)
        self.assertIn("javascript", ctype_js)

        with self.assertRaises(HTTPError) as exc:
            urlopen(f"http://127.0.0.1:{self.port}/assets/../demo_server.py")
        self.assertEqual(exc.exception.code, 404)

    def test_overview_and_chunk_query(self) -> None:
        status, _, raw = self._get("/api/overview")
        self.assertEqual(status, 200)
        data = json.loads(raw.decode("utf-8"))
        self.assertIn("presets", data)
        self.assertGreaterEqual(len(data["presets"]), 6)
        self.assertEqual(data["basic"]["chunk_count"], 210)
        self.assertEqual(data["advanced"]["chunk_count"], 1243)

        _, _, c_raw = self._get("/api/chunks?kb=advanced&page=169&limit=10")
        c_data = json.loads(c_raw.decode("utf-8"))
        self.assertGreaterEqual(c_data["total"], 1)
        self.assertTrue(all(item["page"] == 169 for item in c_data["items"]))

    def test_serves_page_png_and_validates_bounds(self) -> None:
        status, ctype, body = self._get("/api/advanced/page/1")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "image/png")
        self.assertTrue(body.startswith(b"\x89PNG"))

        with self.assertRaises(HTTPError) as exc:
            urlopen(f"http://127.0.0.1:{self.port}/api/advanced/page/999")
        self.assertEqual(exc.exception.code, 400)

    def test_ask_endpoint_validation_and_dispatch(self) -> None:
        empty_req = Request(
            f"http://127.0.0.1:{self.port}/api/ask",
            data=json.dumps({"question": "   ", "kb": "advanced"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as exc:
            urlopen(empty_req)
        self.assertEqual(exc.exception.code, 400)

        fake_resp = {
            "kb": "advanced",
            "mode": "advanced",
            "primary": {
                "query_id": "live_test",
                "status": "answered",
                "answer": "Doanh thu 70.113 tỷ [adv:fpt2025:p010:v000].",
                "citations": ["adv:fpt2025:p010:v000"],
                "context_ids": ["adv:fpt2025:p010:v000"],
                "events": [{"event": "answer"}],
                "elapsed_ms": 12.3,
                "chunk_details": {},
                "pages": [10],
            },
        }
        with patch.object(self.ctx, "run_live_query", return_value=fake_resp) as mock_run:
            req = Request(
                f"http://127.0.0.1:{self.port}/api/ask",
                data=json.dumps({"question": "Doanh thu 2025?", "kb": "advanced"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                payload = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(payload["primary"]["status"], "answered")
                mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
