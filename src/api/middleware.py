"""ASGI 미들웨어 — 요청·응답 로그: 클라이언트가 보낸 본문을 서버 로그에 남긴다.

순수 ASGI 라 receive 를 감싸 본문을 복사만 한다 — 라우트 쪽 파싱·검증에 영향 없다.
거절·422·500 을 포함한 모든 결과에 대해 남으므로 "무엇을 보냈길래" 를 로그로 추적할 수 있다.
"""

import json
import time

from log import get_logger

log = get_logger(__name__)

SKIP_PATHS = frozenset({"/healthz", "/readyz", "/docs", "/redoc", "/openapi.json"})


def _fmt_body(raw: bytes) -> str:
    """본문 → 로그 문자열. JSON 이면 들여써서 여러 줄로, 아니면 그대로."""
    text = raw.decode("utf-8", errors="replace")
    try:
        return "\n" + json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except ValueError:
        return text


class AccessLogMiddleware:
    """`← METHOD PATH 요청본문` / `→ METHOD PATH status (ms) 응답본문` 두 줄을 남긴다."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope["path"] in SKIP_PATHS:
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        path = scope["path"]
        query = scope.get("query_string", b"").decode("latin-1")
        target = f"{path}?{query}" if query else path
        req_chunks: list[bytes] = []
        res_chunks: list[bytes] = []
        logged_in = logged_out = False
        status = None
        started = time.monotonic()

        def log_in() -> None:
            nonlocal logged_in
            if logged_in:
                return
            logged_in = True
            body = _fmt_body(b"".join(req_chunks))
            log.info("← %s %s %s", method, target, body or "-")

        def log_out() -> None:
            nonlocal logged_out
            if logged_out:
                return
            logged_out = True
            elapsed = (time.monotonic() - started) * 1000
            body = _fmt_body(b"".join(res_chunks))
            log.info("→ %s %s %s (%.0fms) %s", method, target, status, elapsed, body or "-")

        async def recv():
            msg = await receive()
            if msg["type"] == "http.request":
                req_chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    log_in()
            return msg

        async def snd(msg) -> None:
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
                log_in()
            elif msg["type"] == "http.response.body":
                res_chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    log_out()   # 응답이 다 나간 시점 — 뒤따르는 BackgroundTasks 소요를 섞지 않게
            await send(msg)

        try:
            await self.app(scope, recv, snd)
        finally:
            log_in()
            log_out()
