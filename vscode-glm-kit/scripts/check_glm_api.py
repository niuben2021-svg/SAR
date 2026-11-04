"""Probe an existing OpenAI-compatible endpoint; do not execute returned model tools."""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://node:8000/v1")
    parser.add_argument("--model", help="exact served model ID, not a marketing/display name")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--check-tools", action="store_true")
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        parser.error("use an http(s) base URL without embedded credentials/query/fragment")
    key = (
        ""
        if args.no_auth
        else os.environ.get("GLM_API_KEY") or getpass.getpass("API Key (hidden): ")
    )

    def request(path, payload=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not echo server response headers/body that may contain credential details.
            raise RuntimeError("HTTP " + str(exc.code) + " on " + path) from None

    try:
        ids = [row["id"] for row in request("/models").get("data", [])]
        print("Served model IDs:", json.dumps(ids, ensure_ascii=False))
        if not args.model:
            print("Run again with --model EXACT_ID to test chat and optional tools.")
            return 0
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        print("Model listing unavailable:", type(exc).__name__)
        if not args.model:
            print("Ask the endpoint administrator for the exact model ID and pass --model.")
            return 2
    try:
        result = request(
            "/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "Reply with the single word pong."}],
                "max_tokens": 128,
                "stream": False,
            },
        )
        message = result["choices"][0]["message"]
        if not message.get("content"):
            raise ValueError("no text answer returned")
        print("Chat completion: PASS")
        if args.check_tools:
            result = request(
                "/chat/completions",
                {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Call report_probe with ok=true. Do not return prose.",
                        }
                    ],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "report_probe",
                                "description": "Harmless connection probe. No external action.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"],
                                },
                            },
                        }
                    ],
                    "tool_choice": {"type": "function", "function": {"name": "report_probe"}},
                    "max_tokens": 256,
                    "stream": False,
                },
            )
            calls = result["choices"][0]["message"].get("tool_calls", [])
            good = any(
                c.get("function", {}).get("name") == "report_probe"
                and json.loads(c["function"]["arguments"]).get("ok") is True
                for c in calls
            )
            if not good:
                raise ValueError("expected structured tool call was not returned")
            print("Structured forced tool call: PASS (not executed; not an end-to-end Agent test)")
        return 0
    except (RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        print(
            "Probe FAILED:",
            type(exc).__name__,
            str(exc).replace(key, "[REDACTED]") if key else str(exc),
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
