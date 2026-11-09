"""离线测试：用本地 HTTP 服务模拟各家 API，检查请求格式、回复解析、自动降级重试，以及评审 / 汇总流程。

    python -m unittest discover -s tests -v
不联网、不需要 API key。
"""
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm_feedback"))
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"

import aggregate_judges as AG  # noqa: E402
import check_faithfulness as CF  # noqa: E402
import make_negative_control as NC  # noqa: E402
import run_judges as RJ  # noqa: E402
from llm_clients import make_client, parse_json_reply  # noqa: E402

REPLY = '```json\n{"ok": true}\n```'


class Handler(BaseHTTPRequestHandler):
    log = []
    reject_temperature = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Handler.log.append(dict(path=self.path, headers={k.lower(): v for k, v in self.headers.items()}, body=body))
        if self.path.endswith("/chat/completions") and Handler.reject_temperature and "temperature" in body:
            return self._send(400, {"error": {"message": "Unsupported parameter: 'temperature'"}})
        if self.path.endswith("/v1/messages"):
            out = {"model": "claude-test-returned", "content": [{"type": "text", "text": REPLY}],
                   "usage": {"input_tokens": 3, "output_tokens": 4}, "stop_reason": "end_turn"}
        elif self.path.endswith("/chat/completions"):
            out = {"model": "gpt-test-returned", "choices": [{"message": {"content": REPLY}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        elif ":generateContent" in self.path:
            out = {"modelVersion": "gemini-test-returned",
                   "candidates": [{"content": {"parts": [{"text": REPLY}]}, "finishReason": "STOP"}]}
        else:
            return self._send(404, {})
        self._send(200, out)

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class TestClients(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Handler)
        cls.url = f"http://127.0.0.1:{cls.srv.server_port}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        os.environ.update(ANTHROPIC_API_KEY="ka", OPENAI_API_KEY="ko", GEMINI_API_KEY="kg", X_KEY="kx")

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        Handler.log.clear()
        Handler.reject_temperature = False

    def test_anthropic(self):
        c = make_client(dict(provider="anthropic", model="m", base_url=self.url, temperature=0))
        r = c.chat("SYS", "USER", max_tokens=99)
        q = Handler.log[-1]
        self.assertEqual(q["path"], "/v1/messages")
        self.assertEqual(q["headers"]["x-api-key"], "ka")
        self.assertIn("anthropic-version", q["headers"])
        self.assertEqual((q["body"]["system"], q["body"]["max_tokens"], q["body"]["temperature"]), ("SYS", 99, 0))
        self.assertEqual(r.model, "claude-test-returned")
        self.assertEqual(parse_json_reply(r.text), {"ok": True})

    def test_openai_and_temperature_fallback(self):
        Handler.reject_temperature = True
        c = make_client(dict(provider="openai", model="m", base_url=self.url + "/v1", temperature=0, json_mode=True))
        r = c.chat("SYS", "USER")
        first, second = Handler.log[-2]["body"], Handler.log[-1]["body"]
        self.assertIn("temperature", first)
        self.assertNotIn("temperature", second)                     # 自动去掉不支持的参数后重试
        self.assertEqual(second["response_format"], {"type": "json_object"})
        self.assertIn("max_completion_tokens", second)
        self.assertEqual(Handler.log[-1]["headers"]["authorization"], "Bearer ko")
        self.assertEqual(r.model, "gpt-test-returned")

    def test_openai_compatible(self):
        c = make_client(dict(provider="openai_compatible", model="m", base_url=self.url, api_key_env="X_KEY"))
        c.chat("SYS", "USER")
        q = Handler.log[-1]
        self.assertEqual(q["path"], "/chat/completions")
        self.assertEqual(q["headers"]["authorization"], "Bearer kx")
        self.assertIn("max_tokens", q["body"])

    def test_gemini(self):
        c = make_client(dict(provider="gemini", model="gm", base_url=self.url, temperature=0, json_mode=True))
        r = c.chat("SYS", "USER")
        q = Handler.log[-1]
        self.assertEqual(q["path"], "/models/gm:generateContent")
        self.assertEqual(q["headers"]["x-goog-api-key"], "kg")
        self.assertEqual(q["body"]["systemInstruction"]["parts"][0]["text"], "SYS")
        self.assertEqual(q["body"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(r.model, "gemini-test-returned")

    def test_missing_key(self):
        os.environ.pop("NOPE", None)
        c = make_client(dict(provider="openai_compatible", model="m", base_url=self.url, api_key_env="NOPE",
                             max_retries=0))
        with self.assertRaises(Exception):
            c.chat("s", "u")


class TestJudgePipeline(unittest.TestCase):
    def test_negative_control_and_mock_panel(self):
        lf = ROOT / "llm_feedback"
        inp = json.loads((lf / "inputs.json").read_text(encoding="utf-8"))
        ex = json.loads((lf / "outputs_example_chat.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(len(v) for v in CF.check(inp, ex).values()), 0)
        seeded, seeds = NC.seed(inp, ex)
        res = CF.check(inp, seeded)
        found = {c for v in res.values() for c, _, _ in v}
        self.assertEqual(found, {"C1", "C2", "C3", "C4", "C5", "C6"})   # 字面错误规则都能发现
        with tempfile.TemporaryDirectory() as d:
            nc = Path(d) / "nc"
            nc.mkdir()
            (nc / "outputs_seeded.json").write_text(json.dumps(seeded, ensure_ascii=False), encoding="utf-8")
            (nc / "seeds.json").write_text(json.dumps(dict(source=str(lf / "outputs_example_chat.json"), seeds=seeds),
                                                      ensure_ascii=False), encoding="utf-8")
            out = Path(d) / "j"
            RJ.main(["--mock", "--target", f"main={lf / 'outputs_example_chat.json'}",
                     "--negative_control", str(nc), "--out", str(out), "--workers", "2"])
            with open(out / "negative_control_detection.csv", encoding="utf-8-sig") as f:
                det = list(__import__("csv").DictReader(f))
            self.assertEqual(len(det), 10)
            self.assertTrue(all(r["panel_majority"] == "✓" for r in det))
            self.assertEqual(sum(r["rule_check"] == "✓" for r in det if r["kind"] == "semantic"), 0)
            self.assertTrue((out / "judge_report.md").exists())

    def test_agreement_stats(self):
        self.assertAlmostEqual(AG.fleiss_kappa([[3, 0], [0, 3], [3, 0], [0, 3]]), 1.0)
        self.assertAlmostEqual(AG.gwet_ac1([[3, 0], [0, 3]]), 1.0)
        self.assertAlmostEqual(AG.cohen_kappa([True, False, True, False], [True, False, True, False]), 1.0)


if __name__ == "__main__":
    unittest.main()
