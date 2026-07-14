#!/usr/bin/env python3
"""HTTP reward API for LLaMA-Factory PPO.

LLaMA-Factory sends decoded prompt+response strings and expects
{"scores": [float, ...]}. Extra details are included for debugging and ignored
by the trainer.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from a2full_ppo_reward import score_message


class RewardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        enable_prompt_tests: bool,
        test_timeout: float,
        memory_mb: int,
        log_jsonl: Path | None,
    ) -> None:
        super().__init__(server_address, RewardHandler)
        self.enable_prompt_tests = enable_prompt_tests
        self.test_timeout = test_timeout
        self.memory_mb = memory_mb
        self.log_jsonl = log_jsonl
        if self.log_jsonl is not None:
            self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)


class RewardHandler(BaseHTTPRequestHandler):
    server: RewardServer
    server_version = "A2FullPPOReward/1.0"

    def do_GET(self) -> None:  # noqa: N802
        self._send_json({"ok": True, "server": self.server_version})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json()
            messages = payload.get("messages") or []
            if not isinstance(messages, list):
                raise ValueError("payload.messages must be a list")

            results = [
                score_message(
                    str(message),
                    enable_prompt_tests=self.server.enable_prompt_tests,
                    test_timeout=self.server.test_timeout,
                    memory_mb=self.server.memory_mb,
                )
                for message in messages
            ]
            scores = [result.score for result in results]
            details = [result.to_dict() for result in results]
            self._log_batch(details)
            self._send_json({"scores": scores, "details": details})
        except Exception as exc:
            self._send_json({"error": str(exc), "scores": []}, status=500)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_batch(self, details: list[dict[str, Any]]) -> None:
        path = self.server.log_jsonl
        if path is None:
            return
        now = time.time()
        with path.open("a", encoding="utf-8") as f:
            for detail in details:
                row = {"time": now, **detail}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_message(self, fmt: str, *args: object) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--enable-prompt-tests", action="store_true")
    parser.add_argument("--test-timeout", type=float, default=1.5)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = RewardServer(
        (args.host, args.port),
        enable_prompt_tests=args.enable_prompt_tests,
        test_timeout=args.test_timeout,
        memory_mb=args.memory_mb,
        log_jsonl=args.log_jsonl,
    )
    print(
        "reward_server=http://{}:{} prompt_tests={} timeout={}s".format(
            args.host, args.port, args.enable_prompt_tests, args.test_timeout
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
