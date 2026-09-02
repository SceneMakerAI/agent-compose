"""cate_id → 도메인 플로우 분기표 — 정확 일치만, 미등록은 None(미지원).

새 도메인 추가 = 도메인 모듈에 진입 함수 작성 + 아래 표에 한 줄.
"""

from domains.baseball import flow as baseball

# cate_id → 플로우 진입 함수
# (v_id, comp_id, query, budget_sec, db, llm, embedder, vector, settings, on_node?) -> ComposeState
DOMAIN_FLOWS = {
    5100: baseball.run,   # 야구
}


def resolve(cate_id: int | None):
    """
    Summary:
        cate_id 로 도메인 플로우 진입 함수를 찾는다 — 정확 일치만.
    Args:
        cate_id (int | None): t_video.cate_id.
    Returns:
        Callable | None: 진입 함수. 미등록이면 None —
            호출자는 명시적 unsupported 처리를 해야 한다(조용한 스킵 금지).
    """
    if cate_id is None:
        return None
    return DOMAIN_FLOWS.get(cate_id)
