"""증거 repository — Milvus 야구 증거 컬렉션(sm_sport_baseball) 읽기 전용.

컬렉션은 agent-vision ingest 단계의 산출물이다 — 쓰기·스키마 정본은 agent-vision
(vector/store.py)가 쥐고, 여기는 소비만 한다. 어느 필드를 어떻게 읽는지(야구
메타 축·이닝 정렬)는 이 모듈 소유 — 연결 통로는 vector.client 가 내어준다
(rdb: pool ↔ 도메인 repo 와 같은 분리).
"""

from config import Settings
from log import get_logger
from vector.client import VectorClient

log = get_logger(__name__)

# 한 번에 읽는 최대 행 수 (Milvus query 상한 16384) — 경기 1건 증거는 수천 행 수준.
_QUERY_LIMIT = 16384

# 메타 어휘 축 — parse_query 가 LLM 에게 "가진 필터 항목"으로 제시할 필드들.
# labels·board_tags 는 콤마 연결 문자열이라 조회 후 분해한다.
_META_FIELDS = ["labels", "board_tags", "inning", "home_team", "away_team", "score_delta"]

# 검색 히트에 실어 오는 필드 — scene_no(구간 귀속)와 사람이 읽을 재료.
_SEARCH_FIELDS = ["kind", "start_sec", "end_sec", "scene_no", "text",
                  "labels", "board_tags", "inning"]

# 증거 종류 — agent-vision 색인 스키마 정의 전체 (stt=해설 / shot=화면 캡션 / etc=하단 자막).
# 종류별로 따로 검색한다 — 한 검색에 섞으면 개수 많은 종류(shot·etc)가 상위를
# 독식해 해설(stt)이 밀려난다.
KINDS = ("stt", "shot", "etc")

# 질의 임베딩 instruction — Qwen3-Embedding 비대칭 검색 권장 (질의에만 프리픽스,
# 문서는 원문 그대로 색인돼 있다). 문구는 야구 도메인 소유.
QUERY_INSTRUCT = ("Instruct: 야구 중계의 해설 대사와 장면 설명에서 질의와 관련된 "
                  "구간을 찾는다\nQuery: ")


def _split_csv(text: str | None) -> list[str]:
    """콤마 연결 문자열 → 낱개 목록. NULL·빈 칸·공백 조각은 버린다."""
    if not text:
        return []
    items = []
    for piece in text.split(","):
        piece = piece.strip()
        if piece:
            items.append(piece)
    return items


def _inning_key(inning: str) -> tuple[int, int]:
    """이닝 정렬 키 — '10회초' 가 '1회초' 뒤에 오게 (사전순 금지). 형식 밖은 뒤로."""
    try:
        no, half = inning.split("회")
        return int(no), 0 if half == "초" else 1
    except ValueError:
        return 99, 9


def dedup_hits(hits: list[dict]) -> list[dict]:
    """여러 검색어의 히트 병합 — 같은 증거는 유사도 높은 쪽만 남긴다.

    키는 (kind, scene_no, 본문) — 시각은 넣지 않는다. etc 는 같은 자막이 프레임마다
    반복 색인돼 시각만 다른 동일 증거가 되는데(실측), 같은 구간 안에서 같은
    내용이면 하나로 충분하다. 다른 구간의 같은 정형구 캡션은 각자 남는다(귀속이 다르다).
    """
    best: dict[tuple, dict] = {}
    for hit in hits:
        key = (hit.get("kind"), hit.get("scene_no"), hit.get("text", ""))
        if key not in best or hit["distance"] > best[key]["distance"]:
            best[key] = hit
    merged = list(best.values())
    merged.sort(key=lambda h: -h["distance"])
    return merged


def group_by_scene(hits: list[dict], snippets_max: int = 2) -> tuple[list[dict], int]:
    """
    Summary:
        검색 히트 → 구간(scene_no) 그룹. (히트 수 → 최고 유사도) 순으로 정렬해 돌려준다.
    Args:
        hits (list[dict]): dedup 된 히트들 (distance 내림차순).
        snippets_max (int): 그룹당 대표 스니펫 수.
    Returns:
        tuple: (그룹 목록, orphan 히트 수). 그룹은 {scene_no, hits, sim, by_kind, snippets}.
            orphan(scene_no=-1)은 어느 구간에도 안 겹친 증거 — 수만 센다.
    """
    groups: dict[int, dict] = {}
    orphan = 0
    for hit in hits:
        scene_no = hit["scene_no"]
        if scene_no < 0:
            orphan += 1
            continue

        if scene_no not in groups:
            groups[scene_no] = {"scene_no": scene_no, "hits": 0, "sim": 0.0,
                                "by_kind": {}, "snippets": []}
        group = groups[scene_no]

        group["hits"] += 1
        group["sim"] = max(group["sim"], round(float(hit["distance"]), 3))

        kind = hit.get("kind") or "?"
        group["by_kind"][kind] = group["by_kind"].get(kind, 0) + 1

        if len(group["snippets"]) < snippets_max:
            group["snippets"].append(
                f"[{kind} {hit.get('start_sec', 0):.0f}~{hit.get('end_sec', 0):.0f}s] "
                f"{hit.get('text', '')}")

    ordered = sorted(groups.values(), key=lambda g: (-g["hits"], -g["sim"]))
    return ordered, orphan


class EvidenceRepo:
    """야구 증거 컬렉션 조회 전담 (읽기 전용 — 쓰기는 agent-vision 소유)."""

    def __init__(self, vector: VectorClient, settings: Settings) -> None:
        """VectorClient(연결 래퍼)와 설정(컬렉션명·top_k)을 주입받는다."""
        self._vector = vector
        self._col = settings.milvus_collection
        self._top_k = settings.vector_top_k

    async def fetch_texts(self, v_id: int) -> list[dict]:
        """
        Summary:
            증거 원문 전량 — 클립 범위의 "클립 내용" 렌더 재료 (벡터 제외라 가볍다).
        Returns:
            list[dict]: {kind, start_sec, end_sec, scene_no, text} 시간순.
        Description:
            - orphan(scene_no=-1)도 포함한다 — 클립 내용은 시간 겹침으로 귀속하므로
              구간 미귀속 증거도 클립 범위에 들면 내용이다.
        """
        rows = await self._vector.query(self._col, f"v_id == {v_id}",
                                        ["kind", "start_sec", "end_sec", "scene_no", "text"],
                                        _QUERY_LIMIT)
        rows.sort(key=lambda r: r.get("start_sec", 0))
        return rows

    async def search(self, query_vec: list[float], v_id: int,
                     kind: str | None = None) -> list[dict]:
        """
        Summary:
            질의 벡터로 v_id 범위 검색 — 상위 top_k 히트. kind 를 주면 그 종류만.
        Returns:
            list[dict]: {distance, kind, start_sec, end_sec, scene_no, text, …}.
        Description:
            - orphan(scene_no=-1 — 어느 구간에도 안 겹친 증거)은 검색에서 제외한다.
              구간에 귀속할 수 없어 편성에 못 쓰는데 top_k 자리만 차지한다.
            - top_k 는 넉넉히(50) 받는다 — etc 등 프레임 반복 색인 탓에 같은
              구간·같은 내용 히트가 섞여 오고, 그건 dedup_hits 가 걸러낸다.
        """
        filter_expr = f"v_id == {v_id} and scene_no >= 0"
        if kind is not None:
            filter_expr += f' and kind == "{kind}"'
        return await self._vector.search(self._col, query_vec, filter_expr,
                                         _SEARCH_FIELDS, self._top_k)

    async def meta_vocab(self, v_id: int) -> dict:
        """
        Summary:
            v_id 증거의 메타 어휘 — 필드별 실존 값 집합 (질의 해석 프롬프트 재료).
        Args:
            v_id (int): 대상 영상 id.
        Returns:
            dict: {labels, board_tags, innings, teams: list[str],
                score_delta_max: int}. 이 경기에 실제로 존재하는 값만 담는다 —
                LLM 은 이 중에서만 필터를 고르므로 없는 값을 지어낼 수 없다.
                teams 는 home/away 사실 값의 합집합 (이 경기의 두 팀).
        """
        rows = await self._vector.query(self._col, f"v_id == {v_id}",
                                        _META_FIELDS, _QUERY_LIMIT)
        labels: set[str] = set()
        board_tags: set[str] = set()
        innings: set[str] = set()
        teams: set[str] = set()
        delta_max = 0
        for row in rows:
            # labels — 콤마 연결 문자열을 낱개로 분해해 담는다 (예: "안타,적시타")
            for label in _split_csv(row.get("labels")):
                labels.add(label)

            # board_tags — labels 와 같은 콤마 연결 형식
            for tag in _split_csv(row.get("board_tags")):
                board_tags.add(tag)

            # inning — 단일값, 빈 문자열(orphan 증거)은 건너뛴다
            inning = row.get("inning")
            if inning:
                innings.add(inning)

            # 팀 — home/away 사실 값의 합집합 (이 경기의 두 팀)
            home = row.get("home_team")
            if home:
                teams.add(home)
            away = row.get("away_team")
            if away:
                teams.add(away)

            # score_delta — 값 목록이 아니라 최대치만 필요하다 (필터 상한 안내용)
            delta = int(row.get("score_delta") or 0)
            delta_max = max(delta_max, delta)
        vocab = {
            "labels": sorted(labels),
            "board_tags": sorted(board_tags),
            "innings": sorted(innings, key=_inning_key),
            "teams": sorted(teams),
            "score_delta_max": delta_max,
        }
        log.info("메타 어휘: v_id=%s 증거 %d행 — labels %d·board_tags %d·innings %d·teams %s",
                 v_id, len(rows), len(vocab["labels"]), len(vocab["board_tags"]),
                 len(vocab["innings"]), vocab["teams"])
        return vocab
