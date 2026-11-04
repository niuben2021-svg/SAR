"""Local synthetic protocol checks, not a real GLM service evaluation."""

import http.server
import json
import subprocess
import sys
import threading
import unittest
from pathlib import Path


class Handler(http.server.BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"data": [{"id": "fixture-model"}]}).encode())

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer fixture-private-key"
        if "tools" in payload and self.mode == "ok":
            message = {
                "tool_calls": [{"function": {"name": "report_probe", "arguments": '{"ok":true}'}}]
            }
        else:
            message = {"content": "pong"}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": message}]}).encode())


class TestProbe(unittest.TestCase):
    def invoke(self, mode, tools):
        import os

        Handler.mode = mode
        with http.server.HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("check_glm_api.py")),
                    "--base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--model",
                    "fixture-model",
                ]
                if tools:
                    command.append("--check-tools")
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "GLM_API_KEY": "fixture-private-key"},
                )
            finally:
                server.shutdown()
                thread.join()
        self.assertNotIn("fixture-private-key", result.stdout + result.stderr)
        return result

    def test_chat(self):
        r = self.invoke("ok", False)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_tools(self):
        r = self.invoke("ok", True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("not an end-to-end Agent test", r.stdout)

    def test_plain_text_does_not_pass_tools(self):
        r = self.invoke("no-tools", True)
        self.assertEqual(r.returncode, 1, r.stdout)


if __name__ == "__main__":
    unittest.main()
