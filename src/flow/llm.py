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
# 선택지가 좁은 판단(bounds·verify)용 상한. 전량이 동시에 출발해도 **가장 오래 생각하는
# 한 콜이 배치 전체를 붙잡는다** — v201 comp9 실측: bounds 28콜 중앙값 2,781자인데
# 한 콜이 59,501자·422초를 썼고, 그 1건이 노드 소요를 2분 30초에서 7분 3초로 늘렸다.
# 6144 토큰 ≈ 19,000자(실측 3.1자/토큰)로 상위 4건(11~15k자)까지는 그대로 통과하고
# 튄 1건만 잘린다. 잘려서 본문이 비면 thinking 을 끄고 재시도하므로 답은 나온다.
MAX_TOKENS_PICK = 6144


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
                   trace=None, name: str = "", think_max: int | None = None) -> str:
        """
        Summary:
            시스템+유저 프롬프트로 1회 판정. thinking 은 호출자가 명시
            (그래프 3콜 전부 True — .env llm_thinking=0 으로 일괄 비활성 가능).
        Args:
            trace: Trace | None — 주면 프롬프트·응답·thinking 전문을 남긴다.
            name (str): 트레이스에 찍을 콜 이름 (plan·bounds·verify…).
            think_max (int|None): thinking 응답 상한 — 생략하면 MAX_TOKENS_THINK.
                팬아웃 노드는 MAX_TOKENS_PICK 을 줘 폭주 1건이 배치를 붙잡는 걸 막는다.
        Returns:
            str: <think> 제거된 본문. 본문이 비면 thinking 끄고 1회 재시도.
        """
        use_think = thinking and self._thinking_enabled
        started = time.monotonic()
        log.debug("LLM 요청 system(%d자):\n%s", len(system), system)
        log.debug("LLM 요청 user(%d자):\n%s", len(user), user)
        # 세마포어는 **전송만** 감싼다. 메서드 전체를 감싸면 아래 thinking 재시도가
        # 자기 자신을 재귀 호출하면서 이미 쥔 허가 위에 또 허가를 기다려 교착한다.
        async with self._gate:
            resp = await self._client.post("/chat/completions", json={
                "model": self._model, "temperature": 0,
                "max_tokens": (think_max or MAX_TOKENS_THINK) if use_think else MAX_TOKENS,
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
            # 상한을 같이 남긴다 — 상한을 낮춘 뒤 이 경고가 잦아지면 값이 너무 빡빡하다는
            # 신호다(품질 저하가 재시도로 조용히 흡수되므로 로그 말고는 드러나지 않는다).
            log.warning("thinking 이 토큰을 소진해 본문 없음(%s, 상한 %d) — thinking 끄고 재시도",
                        name or "chat", think_max or MAX_TOKENS_THINK)
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
