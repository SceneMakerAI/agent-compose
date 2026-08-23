"""증거 수집·색인 — vision3 발행 직후 호출되는 ingest 의 본체.

bench4 tools/index_evidence.py 의 서비스 이식. 증거 세 종을 한 컬렉션에 kind 로 넣는다:
- shot: t_segment.summary (scene 단계 caption, 시각 증거)
- stt : t_dialogue — 인접 발화를 청크로 병합 (실측: 발화 단위 그대로면 "네"·"칠구"
  같은 0.1~1s 파편이 코사인 상위를 점령해 검색이 무너진다)
- etc : t_frame_baseball_board_detail(kind=ETC) 하단 자막 OCR — 매치업·선수 기록. 인물 질의를
  caption(이름 환각)·STT(전사 깨짐)가 아니라 자막 사실로 잡는 재료. 런 병합 후 색인.
소속 장면(scene_id)은 시간 겹침 최대치로 귀속 — seg_id 조인 금지 관례. 겹침 없으면 -1
(발행 누락·장면 밖 콜 회수도 목적의 일부라 버리지 않는다).

2026-08-23 상류 개편 이후 orphan 비율이 오른다 — 발행본이 대표 투구가 있는 사건만
담고 구간도 좁아졌기 때문이다. STT 는 여운 귀속(STT_TRAIL_ATTACH_MAX_SEC)이 그중
직전 플레이 해설을 계속 회수한다.

t_scene_baseball.description('검색용 서술', 임베딩 대상)은 아직 안 쓴다 — 상류가
전 행 NULL 로 둔 상태다(실측 v200~203). 채워지면 kind='scene' 한 종을 늘리는 것으로
끝난다: text·s·e·scene_id 를 그대로 쓰므로 컬렉션 스키마는 안 바뀐다.
"""

from db.repos import SourceRepo
from log import get_logger
from vector.embedder import Embedder
from vector.store import VectorStore

log = get_logger(__name__)

STT_CHUNK_GAP_SEC = 2.0    # 이 이하 간격의 인접 발화는 한 청크로 병합
STT_CHUNK_MAX_CHARS = 300  # 청크 상한 (넘으면 새 청크)
STT_MIN_CHARS = 6          # 병합 후에도 이보다 짧은 청크는 버림 (추임새 파편)
# 여운 귀속(STT 전용): 장면과 안 겹치는 청크가 직전 장면 끝 뒤 이 초 이내에 시작하면
# 직전 장면에 귀속 — 여운 해설은 직전 플레이 이야기다 (실측: "멋진 다이빙 캐치" 콜이
# 장면 끝 +25s 라 orphan 처리돼 해당 장면이 후보에서 빠졌다). shot 은 제외 —
# 다음 타석 준비 화면 오귀속 위험.
STT_TRAIL_ATTACH_MAX_SEC = 30

ETC_MIN_CHARS = 5          # 너무 짧은 자막(로고 조각 등) 제외
ETC_GAP_SEC = 3            # 같은 자막의 프레임 간격이 이 이하면 한 런 (OCR 깜빡임 허용)


def chunk_stt(utts: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """인접 발화 병합 — 간격 STT_CHUNK_GAP_SEC 이하·상한 자수까지 한 청크 (순수 함수)."""
    chunks: list[list] = []
    for s, e, text in utts:
        if not text:
            continue
        if (chunks and s - chunks[-1][1] <= STT_CHUNK_GAP_SEC
                and len(chunks[-1][2]) + len(text) + 1 <= STT_CHUNK_MAX_CHARS):
            chunks[-1][1] = e
            chunks[-1][2] += " " + text
        else:
            chunks.append([s, e, text])
    return [(s, e, t) for s, e, t in chunks if len(t) >= STT_MIN_CHARS]


def merge_etc(rows: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """프레임 단위 ETC 자막 → 같은 텍스트의 연속 [s, e) 런으로 병합 (순수 함수).

    병합은 완전 일치만 — OCR 변형까지 뭉치면 매치업 교체 순간을 잃는다.
    """
    runs: list[list] = []
    for sec, txt in rows:
        if len(txt) < ETC_MIN_CHARS:
            continue
        if runs and runs[-1][2] == txt and sec - runs[-1][1] <= ETC_GAP_SEC:
            runs[-1][1] = sec
        else:
            runs.append([sec, sec, txt])
    return [(s, e + 1, t) for s, e, t in runs]


def owner_of(scenes: list[dict], s: float, e: float, kind: str) -> dict | None:
    """증거 [s,e) 의 소속 장면 — 겹침 최대치, STT 만 여운 귀속 폴백 (순수 함수)."""
    best, ov = None, 0.0
    for r in scenes:
        o = min(e, r["e"]) - max(s, r["s"])
        if o > ov:
            best, ov = r, o
    if best is not None:
        return best
    if kind != "stt":
        return None
    prev = max((r for r in scenes if r["e"] <= s), key=lambda r: r["e"], default=None)
    if prev is not None and s - prev["e"] <= STT_TRAIL_ATTACH_MAX_SEC:
        return prev
    return None


def _trunc_bytes(text: str, limit: int) -> str:
    """UTF-8 바이트 기준 절단 — Milvus VARCHAR max_length 는 바이트 수다
    (실측 v200: 문자 기준 [:1024] 절단본이 1,041바이트로 insert 거부, code=1100)."""
    b = text.encode("utf-8")
    if len(b) <= limit:
        return text
    return b[:limit].decode("utf-8", errors="ignore")


def build_rows(v_id: int, scenes: list[dict],
               evidence: list[tuple[str, float, float, str, str]]) -> list[dict]:
    """(kind, s, e, shot_type, text) 증거 → 색인 행 (벡터 제외, 순수 함수).

    귀속 장면의 메타는 **색인 시점 사본**이다 — 상류가 재발행되면 어긋나므로
    발행 훅이 재색인을 부른다. 절단 한도는 store 스키마의 max_length 와 짝이고,
    Milvus 의 max_length 는 **바이트** 기준이라 _trunc_bytes 를 쓴다.
    """
    out = []
    for kind, s, e, shot_type, text in evidence:
        text = (text or "").strip()
        if not text:
            continue
        sc = owner_of(scenes, s, e, kind)
        out.append({
            "v_id": v_id, "kind": kind, "s": s, "e": e,
            "scene_id": sc["scene_id"] if sc else -1,
            "shot_type": shot_type,
            "tags": _trunc_bytes(",".join(sc["tags"]), 128) if sc else "",
            "labels": _trunc_bytes(",".join(sc["label_list"]), 128) if sc else "",
            "board_tags": _trunc_bytes(",".join(sc["board_tags"]), 256) if sc else "",
            "game_context": (sc["game_context"] or "") if sc else "",
            "score_delta": sc["score_delta"] if sc else 0,
            "inning": _trunc_bytes(sc["inning"] or "", 16) if sc else "",
            "text": _trunc_bytes(text, 1024),
        })
    return out


async def ingest(v_id: int, repo: SourceRepo, embedder: Embedder,
                 store: VectorStore) -> dict:
    """
    Summary:
        v_id 의 증거를 수집·임베딩해 Milvus 색인을 교체한다 (delete-insert 멱등).
    Args:
        v_id (int): 대상 영상 id.
        repo/embedder/store: lifespan 공유 자원.
    Returns:
        dict: {v_id, rows, mapped, orphan, by_kind} 색인 요약.
    Raises:
        ValueError: 발행본(t_scene_baseball) 이 없으면 — 색인 전제 미충족 (조용한 성공 금지).
    """
    scenes = await repo.fetch_scenes(v_id)
    if not scenes:
        raise ValueError(
            f"t_scene_baseball 이 비어 있음 — vision3 scene 선행 필요 (v_id={v_id})")

    evidence: list[tuple[str, float, float, str, str]] = []
    for r in await repo.fetch_shots(v_id):
        evidence.append(("shot", r["s"], r["e"], r["shot_type"] or "", r["summary"]))
    for s, e, text in chunk_stt(await repo.fetch_utterances(v_id)):
        evidence.append(("stt", s, e, "", text))
    for s, e, text in merge_etc(await repo.fetch_etc_rows(v_id)):
        evidence.append(("etc", float(s), float(e), "", text))

    rows = build_rows(v_id, scenes, evidence)
    vecs = await embedder.embed_docs([r["text"] for r in rows])
    for r, v in zip(rows, vecs):
        r["vector"] = v

    n = await store.replace(v_id, rows)
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    summary = {
        "v_id": v_id, "rows": n,
        "mapped": sum(1 for r in rows if r["scene_id"] >= 0),
        "orphan": sum(1 for r in rows if r["scene_id"] < 0),
        "by_kind": by_kind,
    }
    log.info("ingest 완료: %s", summary)
    return summary
