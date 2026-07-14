#!/usr/bin/env python3
"""Small local reward-function server for PPO API rewards."""

from __future__ import annotations

import argparse
import ast
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
CODE_START_RE = re.compile(r"(?m)^(?:from\s+\S+\s+import\s+\S+|import\s+\S+|def\s+\w+|class\s+\w+)")
REFUSAL_RE = re.compile(r"\b(?:sorry|cannot|can't|unable|as an ai|i do not)\b", re.IGNORECASE)


def extract_code(text: str) -> str:
    text = str(text or "").strip()
    fenced = FENCED_CODE_RE.findall(text)
    if fenced:
        return fenced[-1].strip()

    match = CODE_START_RE.search(text)
    if match:
        return text[match.start() :].strip()

    return text


def score_response(text: str) -> float:
    text = str(text or "").strip()
    code = extract_code(text)
    score = 0.0

    if not text:
        return -2.0

    if FENCED_CODE_RE.search(text):
        score += 0.2
    if REFUSAL_RE.search(text):
        score -= 1.0

    try:
        ast.parse(code)
    except SyntaxError:
        score -= 1.0
    else:
        score += 1.0

    if re.search(r"(?m)^\s*(def|class)\s+\w+", code):
        score += 0.4
    if re.search(r"(?m)^\s*return\b", code):
        score += 0.2
    if len(code.split()) < 8:
        score -= 0.4
    if len(code) > 4096:
        score -= 0.4

    return max(-2.0, min(2.0, score))


class RewardHandler(BaseHTTPRequestHandler):
    server_version = "CodeRewardServer/1.0"

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = payload.get("messages") or []
            scores = [score_response(message) for message in messages]
            body = json.dumps({"scores": scores}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc), "scores": []}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), RewardHandler)
    print(f"reward_server=http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
