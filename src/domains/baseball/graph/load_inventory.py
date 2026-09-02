"""load_inventory 노드 — 편성 인벤토리 로드 (t_scene_baseball 불변 스냅샷 1회)."""

from domains.baseball.graph.state import ComposeState
from domains.baseball.repo.scenes import SceneRepo
from log import get_logger

log = get_logger(__name__)


def make_node(scene_repo: SceneRepo):
    """자원 주입 팩토리 — build.py 가 호출한다."""

    async def load_inventory(st: ComposeState) -> dict:
        """인벤토리 로드 — 구간 전량 (Scene frozen — 노드는 수정 불가)."""
        scenes = await scene_repo.fetch(st["v_id"])
        if not scenes:
            # 발행 선행 미충족 — 조용한 빈 편성이 아니라 오류로 드러낸다
            raise ValueError(
                f"t_scene_baseball 가 비어 있음 — agent-vision 분석 선행 필요 (v_id={st['v_id']})")
        return {"scenes": scenes}

    return load_inventory
