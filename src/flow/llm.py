"""vLLM chat 호출 공통 — thinking 모드 대응 (bench4 compose/llm.py 이식, async 전환).

thinking 출력이 content 에 <think>…</think> 로 인라인되는 서버 구성 대비:
파서(줄 형식 키: 값)가 사고 과정 문장을 오파싱하지 않게 본문에서 제거한다.
LLM 출력은 항상 줄 형식 — JSON 강제는 금지 (bench4 실측 확정: 줄 형식이 안정).
"""

import asyncio
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
# thinking 상한(토큰)은 쓰지 않는다 — 사고 도중에 잘려 결론 직전에 끊길 수 있다.
# 대신 **시간 가드**를 둔다: 정상 콜의 최대가 166초인데(v200 comp16 실측) 교착 콜은
# 804초를 쓰고도 빈 응답을 냈고, thinking 을 끈 재시도가 1초 만에 같은 답을 냈다.
# 즉 폭주는 답에 다가가는 과정이 아니라 교착이다. 정상 범위를 한참 넘긴 뒤에만
# 발동하므로 토큰 상한과 달리 정상 콜의 사고를 자르지 않는다.
#
# 전역으로 thinking 을 끄는 선택지는 실측으로 기각했다 — 같은 13콜을 끄고 돌리자
# 11건이 "시작 유지"에서 "시작 이동"으로 바뀌었다. 사고 없이는 "현재 시작이 이미
# 투구 지점이면 유지"라는 판단을 못 하고 후보를 아무거나 고른다.
THINK_TIMEOUT_SEC = 240.0


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
        self._gate = asyncio.Semaphore(settings.llm_concurrency)

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
        # 세마포어는 **전송만** 감싼다. 메서드 전체를 감싸면 아래 thinking 재시도가
        # 자기 자신을 재귀 호출하면서 이미 쥔 허가 위에 또 허가를 기다려 교착한다.
        try:
            async with self._gate:
                resp = await asyncio.wait_for(self._post(system, user, use_think),
                                              THINK_TIMEOUT_SEC if use_think else None)
        except TimeoutError:
            # 취소하면 vLLM 도 그 시퀀스를 접어 GPU 를 놓는다 — 교착 1건이 배치 전체를
            # 붙잡는 걸 여기서 끊는다.
            log.warning("thinking 교착(%s, %.0fs 초과) — thinking 끄고 재시도",
                        name or "chat", THINK_TIMEOUT_SEC)
            return await self.chat(system, user, thinking=False, trace=trace,
                                   name=f"{name}:timeout" if name else "timeout")
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return await self._finish(msg, system, user, started, use_think, trace, name)

    async def _post(self, system: str, user: str, use_think: bool):
        """전송 1회 — wait_for 가 취소할 수 있게 분리한다."""
        return await self._client.post("/chat/completions", json={
                "model": self._model, "temperature": 0,
                "max_tokens": MAX_TOKENS_THINK if use_think else MAX_TOKENS,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "chat_template_kwargs": {"enable_thinking": use_think},
        })

    async def _finish(self, msg: dict, system: str, user: str, started: float,
                      use_think: bool, trace, name: str) -> str:
        """응답 1건 처분 — 사고 제거·트레이스·본문 없을 때 폴백 판정."""
        # 사고 필드명은 서버 구성에 따라 다르다 — 이 vLLM 은 'reasoning' (실측)
        think = msg.get("reasoning") or msg.get("reasoning_content")
        if think:
            log.debug("thinking(%d자): %s…", len(think), think[:200])
        text = _THINK.sub("", msg.get("content") or "").strip()
        if trace is not None:
            trace.llm(name or "chat", system, user, text, thinking=think,
                      elapsed=time.monotonic() - started)
        if not text and use_think:
            # 콜 이름을 남긴다 — 어느 클립이 소진했는지 알아야 그 프롬프트를 고칠 수 있다.
            log.warning("thinking 이 토큰을 소진해 본문 없음(%s) — thinking 끄고 재시도",
                        name or "chat")
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
