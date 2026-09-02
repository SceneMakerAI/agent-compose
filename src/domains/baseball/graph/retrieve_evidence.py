"""retrieve_evidence 노드 — 검색어(중계 문장형)를 임베딩해 증거를 벡터 검색한다.

종류(stt·shot·etc)별로 따로 검색한다 — 한 검색에 섞으면 개수 많은 종류(shot·etc)가
상위를 독식해 해설(stt)이 밀려난다. 임베딩은 검색어당 1회 재사용.
검색어가 '없음'(구체적 행위·상황 묘사가 없는 범용 질의)이면 **벡터 검색을 생략**한다 —
'하이라이트' 같은 추상 메타 단어로 검색하면 매칭이 무너진다 (질의 원문 폴백 금지).
검색 실패는 전파하지 않고 증거 없이 진행한다 — 필터 축이 남아 있어 편성은 가능하다.
"""

from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.evidences import (
    KINDS,
    QUERY_INSTRUCT,
    EvidenceRepo,
    dedup_hits,
    group_by_scene,
)
from infer.embedder import Embedder
from log import get_logger

log = get_logger(__name__)


def make_node(embedder: Embedder, evidence_repo: EvidenceRepo):
    """자원 주입 팩토리 — build.py 가 호출한다."""

    async def retrieve_evidence(st: ComposeState) -> dict:
        """검색어마다 종류별 검색 → 구간(scene_no) 귀속 그룹으로 정리한다."""
        spec = st.get("spec") or {}
        phrases = spec.get("phrases") or []
        trace = st.get("trace")

        # 검색어 없음 = 구체적 묘사가 없는 범용 질의 — 벡터 검색 생략 (원문 폴백 금지)
        if not phrases:
            log.info("retrieve_evidence 생략: 검색어 없음 (필터 축만으로 진행)")
            if trace is not None:
                trace.note("retrieve_evidence", "검색 생략",
                           "검색어 없음 — 추상 질의는 벡터 검색을 하지 않는다 (필터 축만으로 진행)")
            return {"evidence": [], "evidence_orphan": 0}

        hits: list[dict] = []
        try:
            for i, phrase in enumerate(phrases, 1):
                query_vec = await embedder.embed_query(QUERY_INSTRUCT, phrase)

                # 종류별 검색 — 트레이스에 kind 섹션으로 나눠 남긴다
                lines: list[str] = []
                for kind in KINDS:
                    kind_hits = await evidence_repo.search(query_vec, st["v_id"], kind)
                    # 같은 구간의 **동일한 내용**은 1건으로 접는다 — etc 는 프레임
                    # 반복 자막이라 group_size=2 가 같은 텍스트 2건이 되기 쉽다.
                    # 내용이 다르면 2건 다 남는다 (구간당 상한은 grouping search 몫).
                    kind_hits = dedup_hits(kind_hits)
                    hits += kind_hits

                    lines.append(f"[{kind} 상위 {len(kind_hits)}건]")
                    for h in kind_hits:
                        lines.append(
                            f"- {h['distance']:.3f} scene {h['scene_no']:>3} "
                            f"{h.get('start_sec', 0):.0f}~{h.get('end_sec', 0):.0f}s "
                            f"{h.get('text', '')}")
                    lines.append("")

                if trace is not None:
                    trace.note("retrieve_evidence", f'검색어 {i} — "{phrase}"',
                               "\n".join(lines) or "(히트 없음)")
        except Exception as e:               # noqa: BLE001 — 보조 단계, 죽이지 않는다
            log.warning("retrieve_evidence 실패(증거 없이 진행): %s", e)
            if trace is not None:
                trace.note("retrieve_evidence", "검색 실패",
                           f"(오류) {type(e).__name__}: {e} — 증거 없이 진행")
            return {"evidence": [], "evidence_orphan": 0}

        merged = dedup_hits(hits)
        evidence, orphan = group_by_scene(merged)
        log.info("retrieve_evidence: 검색어 %d건 × 종류 %d → 히트 %d → 후보 구간 %s (orphan %d)",
                 len(phrases), len(KINDS), len(hits),
                 [g["scene_no"] for g in evidence[:10]], orphan)

        # 구간 귀속 결과 요약 — 최종적으로 select 가 보게 될 후보 순서
        if trace is not None:
            lines = []
            for g in evidence:
                lines.append(f"- scene {g['scene_no']:>3}: hits {g['hits']} · "
                             f"sim {g['sim']:.3f} · {g['by_kind']}")
                for snippet in g["snippets"]:
                    lines.append(f"    {snippet}")
            lines.append(f"- orphan(구간 미귀속) {orphan}건")
            trace.note("retrieve_evidence", "구간 귀속 결과 (히트수→유사도 순)",
                       "\n".join(lines))
        # 히트 원본도 상태에 — select_clips 가 클립 내용 줄에 [질의 유사] 를 표기하는 근거
        return {"evidence": evidence, "evidence_orphan": orphan, "evidence_hits": merged}

    return retrieve_evidence
