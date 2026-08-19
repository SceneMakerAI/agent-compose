"""vLLM chat 호출 공통 — thinking 모드 대응 (bench4 compose/llm.py 이식, async 전환).

thinking 출력이 content 에 <think>…</think> 로 인라인되는 서버 구성 대비:
파서(줄 형식 키: 값)가 사고 과정 문장을 오파싱하지 않게 본문에서 제거한다.
LLM 출력은 항상 줄 형식 — JSON 강제는 금지 (bench4 실측 확정: 줄 형식이 안정).
"""

import re
import time

import httpx

from config import Settings
from log import get_logger

log = get_logger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# 응답 상한 — thinking 은 사고+본문을 같이 소모하므로 크게 (bench4 운영값)
MAX_TOKENS = 512
MAX_TOKENS_THINK = 32768


class ChatLLM:
    """
    Summary:
        chat 호출 객체 — openai 호환 /v1/chat/completions (lifespan 공유 전제).
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 엔드포인트·모델·thinking 정책을 받는다."""
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url, timeout=settings.llm_timeout)
        self._model = settings.llm_model
        self._thinking_enabled = settings.llm_thinking

    async def chat(self, system: str, user: str, thinking: bool = False,
                   trace=None, name: str = "") -> str:
        """
        Summary:
            시스템+유저 프롬프트로 1회 판정. thinking 은 호출자가 명시
            (그래프 3콜 전부 True — .env llm_thinking=0 으로 일괄 비활성 가능).
        Args:
            trace: Trace | None — 주면 프롬프트·응답·thinking 전문을 남긴다.
            name (str): 트레이스에 찍을 콜 이름 (plan·bounds·verify…).
        Returns:
            str: <think> 제거된 본문. 본문이 비면 thinking 끄고 1회 재시도.
        """
        use_think = thinking and self._thinking_enabled
        started = time.monotonic()
        log.debug("LLM 요청 system(%d자):\n%s", len(system), system)
        log.debug("LLM 요청 user(%d자):\n%s", len(user), user)
        resp = await self._client.post("/chat/completions", json={
            "model": self._model, "temperature": 0,
            "max_tokens": MAX_TOKENS_THINK if use_think else MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "chat_template_kwargs": {"enable_thinking": use_think},
        })
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        # 사고 필드명은 서버 구성에 따라 다르다 — 이 vLLM 은 'reasoning' (실측)
        think = msg.get("reasoning") or msg.get("reasoning_content")
        if think:
            log.debug("thinking(%d자): %s…", len(think), think[:200])
        text = _THINK.sub("", msg.get("content") or "").strip()
        if trace is not None:
            trace.llm(name or "chat", system, user, text, thinking=think,
                      elapsed=time.monotonic() - started)
        if not text and use_think:
            log.warning("thinking 이 토큰을 소진해 본문 없음 — thinking 끄고 재시도")
            return await self.chat(system, user, thinking=False, trace=trace,
                                   name=f"{name}:retry" if name else "retry")
        return text

    async def ready(self) -> bool:
        """모델 서빙 여부 (readyz 프로브) — 짧은 타임아웃, 예외는 False."""
        try:
            resp = await self._client.get("/models", timeout=3.0)
            return resp.is_success
        except httpx.HTTPError as e:
            log.warning("chat LLM 프로브 실패: %s", e)
            return False

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await self._client.aclose()
