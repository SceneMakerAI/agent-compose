"""LLM 요청 트레이스 — 프롬프트·응답 전문을 사람이 읽는 파일로 남긴다.

왜 로그가 아니라 파일인가: 프롬프트는 콜당 수천 자라 운영 로그에 흘리면 로그를 덮는다.
"이 질의에 왜 이 필터/선곡이 나왔나"는 파일에서 해당 콜 블록을 찾아 읽는 게 맞다.

파일 규약: {trace_dir}/{v_id}_{comp_id}/{v_id}_{comp_id}_{node_name}.md — 편성 실행
1건이 디렉터리 1개, 그 안에 노드별 파일. 같은 노드가 여러 콜을 내면(맵-리듀스 청크 등)
그 파일에 append 된다. comp_id 가 실행마다 신규 발급이라 실행 간 덮어쓰기가 없다 —
재실행 대조는 디렉터리끼리 비교한다.

콜 완료 즉시 append 한다 — 실행이 중간에 죽어도 그때까지는 남는다.
끄면(TRACE_DIR 빈 값) 모든 메서드가 no-op — 수집 자체가 없어 운영 부담이 없다.
기록 실패는 본 처리를 죽이지 않는다 (관측용이지 산출물이 아니다).
"""

from pathlib import Path

from log import get_logger

log = get_logger(__name__)

# src/infer/ 의 조부모 = 레포 루트. 상대 TRACE_DIR 은 여기 기준으로 푼다 —
# 서비스 CWD 에 좌우되면 트레이스가 어디에 떨어졌는지 찾을 수 없다.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class InferTraceLog:
    """
    Summary:
        편성 실행 1건 분량의 LLM 콜 기록기 — {v_id}_{comp_id}/ 디렉터리 아래
        {v_id}_{comp_id}_{node_name}.md 에 콜 블록을 append 한다.
    Description:
        - 실행(comp_id)마다 전용 디렉터리라 이전 실행 기록을 건드리지 않는다.
        - 비활성(root 빈 값·초기화 실패)이면 모든 메서드가 즉시 반환한다.
        - system 프롬프트는 노드 파일당 처음 한 번만(바뀌면 그때만 다시) 기록한다 —
          같은 노드의 반복 콜에서 파일이 콜 수만큼 붓지 않게.
    """

    def __init__(self, root: str | None, v_id: int, comp_id: int) -> None:
        """
        Summary:
            실행 전용 기록 디렉터리 {root}/{v_id}_{comp_id}/ 를 준비한다 — root 가 비면 비활성.
        Args:
            root (str | None): 트레이스 기준 디렉터리 (상대 경로는 레포 루트 기준).
            v_id (int): 대상 영상 id — 디렉터리·파일명 접두어가 된다.
            comp_id (int): 편성 id (v_id 안 시퀀스) — v_id 와 함께 실행을 특정한다.
        """
        self._dir: Path | None = None
        self._prefix = f"{v_id}_{comp_id}"
        self._system: dict[str, str] = {}     # 노드별 마지막 기록 system — 반복 기록 방지
        self._started: set[str] = set()       # 헤더를 쓴 노드 파일
        if not root:
            return
        base = Path(root) if Path(root).is_absolute() else _REPO_ROOT / root
        run_dir = base / self._prefix
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            self._dir = run_dir
            log.info("LLM 트레이스 기록 시작: %s", run_dir)
        except OSError as e:
            log.warning("LLM 트레이스 초기화 실패(끔): %s", e)
            self._dir = None

    @property
    def on(self) -> bool:
        return self._dir is not None

    def llm(self, node: str, system: str, user: str, response: str,
            elapsed: float | None = None) -> None:
        """
        Summary:
            LLM 콜 1건을 append 한다 — 프롬프트·응답 전문 (잘라 남기면 재현이 안 된다).
        Args:
            node (str): 노드 이름 (예: 'parse_query') — 파일명이 된다.
            system (str): 시스템 프롬프트.
            user (str): 유저 프롬프트.
            response (str): 응답 본문 (실패 건은 호출부가 '(오류) …' 를 넣는다).
            elapsed (float | None): 소요 초.
        """
        if not self.on:
            return
        node = node or "chat"
        # 파일은 '[' 앞 노드명 기준 — 맵-리듀스 콜(select_clips[1]…)이 한 파일에 모인다
        file_node = node.split("[")[0]
        parts: list[str] = []

        # 파일 헤더 — 노드 파일당 처음 한 번
        if file_node not in self._started:
            parts.append(f"# {self._prefix} — {file_node}")
            self._started.add(file_node)

        # system — 처음 한 번, 바뀌면 그때만 다시
        if system != self._system.get(file_node):
            parts += [f"\n## system — {file_node}", "", system.rstrip()]
            self._system[file_node] = system

        # 콜 블록 — user 와 응답
        meta = f" — {elapsed:.1f}초" if elapsed is not None else ""
        parts += [
            f"\n## {node}{meta}",
            "",
            "[user]",
            user.rstrip(),
            "",
            "[응답]",
            (response or "").rstrip(),
            "",
        ]

        self._append(file_node, parts)

    def note(self, node: str, title: str, body: str) -> None:
        """
        Summary:
            LLM 콜이 아닌 노드 산출(검색 결과 등)을 append 한다 — 같은 파일 규약.
        Args:
            node (str): 노드 이름 (예: 'retrieve_evidence') — 파일명이 된다.
            title (str): 블록 제목 (예: '검색어 1 — "…"').
            body (str): 블록 본문 (여러 줄 텍스트).
        """
        if not self.on:
            return
        parts: list[str] = []
        if node not in self._started:
            parts.append(f"# {self._prefix} — {node}")
            self._started.add(node)
        parts += [f"\n## {title}", "", body.rstrip(), ""]
        self._append(node, parts)

    def _append(self, node: str, parts: list[str]) -> None:
        """노드 파일에 블록을 append 한다 — 실패는 무시(관측용)."""
        path = self._dir / f"{self._prefix}_{node}.md"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write("\n".join(parts))
        except OSError as e:
            log.warning("LLM 트레이스 기록 실패(무시): %s", e)


__all__ = ["InferTraceLog"]
