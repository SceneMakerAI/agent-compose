"""chat LLM 클라이언트 — openai 호환 /v1/chat/completions (vLLM qwen).

역할은 전송뿐이다 — 프롬프트의 의미(질의 해석·선곡)는 domains/* 가 갖는다.
lifespan 1식 공유 전제 (호출마다 생성 금지). 동시 전송은 세마포어로 서버
상한(--max-num-seqs)에 맞춘다.

thinking 은 콜마다 호출자가 정하고(기본 끔 — 현재 select_clips 만 켠다), .env 의
LLM_THINKING=0 으로 전 콜 일괄 비활성할 수 있다. LLM_REASONING_EFFORT 를 주면
thinking 콜은 enable_thinking 토글 대신 reasoning_effort 로 사고량을 지정한다. 서버가 사고 과정을 content 에
<think>…</think> 로 인라인하는 구성 대비, 본문에서 항상 제거한다 (줄 형식
파서가 사고 문장을 오파싱하지 않게).
"""

import asyncio
import re
import time

import httpx

from config import Settings
from log import get_logger

log = get_logger(__name__)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# 응답 상한 — thinking 은 사고+본문을 같이 소모하므로 크게
MAX_TOKENS = 512
MAX_TOKENS_THINK = 32768
# thinking 시간 가드 — 토큰 상한이 아니라 시간으로 끊는다 (실측: 교착 콜은 수백 초를
# 쓰고도 빈 응답을 냈고, thinking 끈 재시도가 1초 만에 같은 답을 냈다 —
# 폭주는 답에 다가가는 과정이 아니라 교착이다). 걸리면 thinking 을 끄고 1회 재시도한다.
# ⚠ 전송 타임아웃(Settings.llm_timeout)은 이 값보다 커야 한다 — 같거나 작으면 가드가
# 울기 전에 httpx 가 끊는다 (__init__ 이 경고로 드러낸다).
THINK_TIMEOUT_SEC = 480.0


class ChatLLM:
    """
    Summary:
        chat 호출 객체 — 시스템+유저 프롬프트 1회 판정 (lifespan 공유 전제).
    """

    def __init__(self, settings: Settings) -> None:
        """설정에서 엔드포인트·모델·동시성 상한·thinking 전역 스위치를 받는다."""
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url, timeout=settings.llm_timeout)
        self._model = settings.llm_model
        self._gate = asyncio.Semaphore(settings.llm_concurrency)
        self._thinking_enabled = settings.llm_thinking
        self._reasoning_effort = settings.llm_reasoning_effort
        # /tokenize 는 /v1 밖 루트 경로다 — base_url 에서 /v1 을 걷어내 절대 URL 로 만든다
        self._tokenize_url = settings.llm_base_url.removesuffix("/v1") + "/tokenize"
        if settings.llm_timeout <= THINK_TIMEOUT_SEC:
            log.warning("LLM_TIMEOUT(%.0fs) 이 thinking 가드(%.0fs) 이하 — 가드가 울기 전에 "
                        "전송이 끊긴다. .env 의 LLM_TIMEOUT 을 %.0fs 초과로 올릴 것",
                        settings.llm_timeout, THINK_TIMEOUT_SEC, THINK_TIMEOUT_SEC)

    async def count_tokens(self, system: str, user: str) -> int:
        """
        Summary:
            이 프롬프트를 보내면 몇 토큰인지 — 서버 /tokenize 로 정확히 센다.
        Description:
            - messages 형태로 물어 chat 템플릿(system/user 래핑)까지 적용된
              최종 전송분과 1:1 인 값이다 (추정이 아니다).
            - 용도: 긴 인벤토리의 맵-리듀스 분기 근거·프롬프트 크기 관측.
        Returns:
            int: 토큰 수.
        Raises:
            httpx.HTTPError: 접속 불가 등 — 폴백 판단은 호출자 몫.
        """
        resp = await self._client.post(self._tokenize_url, json={
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "add_generation_prompt": True,
        })
        resp.raise_for_status()
        return resp.json()["count"]

    async def chat(self, system: str, user: str, thinking: bool = False,
                   trace=None, name: str = "") -> str:
        """
        Summary:
            시스템+유저 프롬프트로 1회 판정 — <think> 제거된 본문을 돌려준다.
        Args:
            system (str): 시스템 프롬프트.
            user (str): 유저 프롬프트.
            thinking (bool): 사고 모드 — 판정이 추론 사슬을 요구하는 콜만 켠다.
            trace: InferTraceLog | None — 주면 프롬프트·응답 전문을 남긴다.
                모든 LLM 콜이 이 메서드를 지나므로 여기서 기록하면 누락이 없다.
            name (str): 트레이스에 찍을 노드 이름 (예: 'parse_query').
        Returns:
            str: 응답 본문 (앞뒤 공백 제거). thinking 콜이 교착(시간 가드)하거나
                본문 없이 끝나면 thinking 을 끄고 1회 재시도한 결과다.
        Raises:
            httpx.HTTPError: 접속 불가·타임아웃·4xx/5xx — 폴백 판단은 호출자 몫.
                실패도 트레이스에 '(오류) …' 로 남긴다.
        """
        use_think = thinking and self._thinking_enabled
        started = time.monotonic()
        try:
            # 세마포어는 전송만 감싼다 — 아래 재시도가 자기 자신을 재귀 호출할 때
            # 이미 쥔 허가 위에 또 허가를 기다려 교착하는 것을 막는다.
            async with self._gate:
                resp = await asyncio.wait_for(
                    self._post(system, user, use_think),
                    THINK_TIMEOUT_SEC if use_think else None)
            resp.raise_for_status()
        except (TimeoutError, httpx.TimeoutException) as e:
            if not use_think:
                # thinking 없이도 시간을 넘겼다 = 진짜 장애
                if trace is not None:
                    trace.llm(name, system, user, f"(오류) {type(e).__name__}: {e}",
                              elapsed=time.monotonic() - started)
                raise
            # thinking 교착 — 취소하면 vLLM 도 그 시퀀스를 접어 GPU 를 놓는다
            log.warning("thinking 교착(%s, %s) — thinking 끄고 재시도",
                        name or "chat", type(e).__name__)
            if trace is not None:
                trace.llm(name, system, user, f"(thinking 교착 {type(e).__name__} — 끄고 재시도)",
                          elapsed=time.monotonic() - started)
            return await self.chat(system, user, thinking=False, trace=trace,
                                   name=f"{name}:timeout" if name else "timeout")
        except Exception as e:
            # 실패 콜도 트레이스에 남긴다 — "왜 폴백했나"를 파일에서 볼 수 있게
            if trace is not None:
                trace.llm(name, system, user, f"(오류) {type(e).__name__}: {e}",
                          elapsed=time.monotonic() - started)
            raise
        message = resp.json()["choices"][0]["message"]
        text = _THINK.sub("", message.get("content") or "").strip()

        # thinking 이 토큰을 소진해 본문 없이 끝난 콜 — thinking 끄고 1회 재시도
        if not text and use_think:
            log.warning("thinking 본문 없음(%s) — thinking 끄고 재시도", name or "chat")
            if trace is not None:
                trace.llm(name, system, user, "(thinking 본문 없음 — 끄고 재시도)",
                          elapsed=time.monotonic() - started)
            return await self.chat(system, user, thinking=False, trace=trace,
                                   name=f"{name}:retry" if name else "retry")

        if trace is not None:
            trace.llm(name, system, user, text, elapsed=time.monotonic() - started)
        return text

    async def _post(self, system: str, user: str, use_think: bool):
        """전송 1회 — wait_for 가 취소할 수 있게 분리한다."""
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": MAX_TOKENS_THINK if use_think else MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if use_think and self._reasoning_effort:
            # effort 모드 — 사고량은 서버가 effort 로 정한다. enable_thinking 토글과
            # 섞어 보내지 않는다 (템플릿이 어느 쪽을 따를지 서버 구성에 좌우되므로).
            payload["reasoning_effort"] = self._reasoning_effort
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": use_think}
        return await self._client.post("/chat/completions", json=payload)

    async def ready(self) -> bool:
        """모델 서빙 여부 (프로브) — 짧은 타임아웃, 예외는 False."""
        try:
            resp = await self._client.get("/models", timeout=3.0)
            return resp.is_success
        except httpx.HTTPError as e:
            log.warning("chat LLM 프로브 실패: %s", e)
            return False

    async def aclose(self) -> None:
        """클라이언트 반납 (앱 종료 시)."""
        await self._client.aclose()
