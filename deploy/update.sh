#!/usr/bin/env bash
# agent-compose 자체 업데이트 — GitHub origin/main 최신으로 갱신 후 서비스 재기동.
# (worker-prep-vision deploy/update.sh 계승 — sudo 이원화만 다름: 이 서비스는
#  agent 계정 소유라 git/uv 는 sudo -u agent, systemctl 은 sudo 로 돈다)
#
# 사용(sm-api-01, ec2-user 등 sudo 가능 계정에서):
#   deploy/update.sh            # 변경 없으면 아무것도 안 하고 종료
#   deploy/update.sh --force    # 변경 없어도 sync + 재기동 강제
#
# 전제:
#   - 배포 디렉토리가 GitHub 를 origin(ssh 별칭 + read-only deploy key) 으로 둔 git clone, 소유자 agent
#     (최초 1회 부트스트랩은 CLAUDE.md 참조)
#   - .env 는 gitignore(미추적)라 reset --hard 에도 보존된다
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE=agent-compose.service
UNIT_SRC="$APP_DIR/deploy/agent-compose.service"
UNIT_DST="/etc/systemd/system/$SERVICE"
BRANCH=main
RUN_AS=agent

# 소유 계정으로 실행 (deploy key 는 agent 의 ~/.ssh 에 있다)
as_owner() { sudo -u "$RUN_AS" -H "$@"; }
GIT=(as_owner git -C "$APP_DIR")

"${GIT[@]}" fetch origin "$BRANCH"
LOCAL=$("${GIT[@]}" rev-parse HEAD)
REMOTE=$("${GIT[@]}" rev-parse "origin/$BRANCH")
if [[ "$LOCAL" == "$REMOTE" && "${1:-}" != "--force" ]]; then
    echo "이미 최신입니다($("${GIT[@]}" rev-parse --short HEAD)) — 종료 (강제하려면 --force)"
    exit 0
fi

echo "업데이트: $("${GIT[@]}" rev-parse --short HEAD) → $("${GIT[@]}" rev-parse --short "origin/$BRANCH")"
# 서버 로컬 수정을 버리고 원격 main 을 정본으로 강제 일치(.env 등 미추적 파일은 보존)
"${GIT[@]}" reset --hard "origin/$BRANCH"

# 의존성 — uv.lock 그대로 재현(잠금 갱신은 개발 머신 몫)
(cd "$APP_DIR" && as_owner uv sync --frozen)

# systemd 유닛이 저장소 버전과 다르면 갱신(멱등)
if ! cmp -s "$UNIT_SRC" "$UNIT_DST" 2>/dev/null; then
    sudo install -m644 "$UNIT_SRC" "$UNIT_DST"
    sudo systemctl daemon-reload
    echo "systemd 유닛 갱신됨"
fi

sudo systemctl restart "$SERVICE"

# 헬스 확인 — readyz(DB+embed+Milvus)까지 최대 30초 대기
PORT=$(grep -E '^APP_PORT=' "$APP_DIR/.env" | cut -d= -f2 | awk '{print $1}')
PORT=${PORT:-8084}
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null; then
        echo "배포 완료: $("${GIT[@]}" rev-parse --short HEAD) / readyz OK (port ${PORT})"
        exit 0
    fi
    sleep 1
done

# readyz 는 embed(GPU vLLM)·Milvus 원격 프로브 포함 — GPU 야간 자동 중지 시간대엔
# 배포가 정상이어도 여기서 실패한다. 프로세스 기동 여부로 원인을 갈라 안내.
echo "경고: readyz 무응답 — GPU 중지 시간대면 프로브 실패가 정상일 수 있음." >&2
echo "  systemctl status ${SERVICE} / journalctl -u ${SERVICE} 로 기동 여부 확인" >&2
exit 1
