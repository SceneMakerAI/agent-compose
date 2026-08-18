# agent-compose

질의 기반 하이라이트 편성 API. 사용자 질의 1건을 LangGraph flow에 태워
클립 편성(`t_compose`/`t_compose_clip`)을 만들고, 그 재료가 되는 장면 증거를
Milvus(`sm_scene_evidence`)에 색인한다.

```
질의 → retrieve(벡터 검색) → plan(LLM 선곡) → cutrank(컷·예산 채움)
     → [backfill(충원)] → endfix(LLM 끝 보정) → verify(LLM 검수 소견) → 저장
```

- LLM은 **제안**만(선곡·끝 보정·소견), **처분은 결정적 코드**가 한다
  (선곡 검산, 끝 검증기, verify 기각권 없음).
- 색인 증거는 3종: `shot`(t_segment 캡션) / `stt`(t_dialogue 청크) /
  `etc`(하단 자막 OCR). 장면 귀속은 시간 겹침 기준.

## 실행

```bash
uv sync && cp .env.example .env   # 값 채우기 (LLM·embed·Milvus·DB — 기본값 없음, 누락 시 부팅 실패)
PYTHONPATH=src uv run python src/run.py   # .env 의 APP_HOST/APP_PORT (기본 8084)
uv run pytest tests/ -q
```

배포는 `deploy/update.sh` 참조 (origin/main 기준 sync + systemd 재기동).

## API

베이스: `http://<host>:8084` (아래 예시는 localhost). 모든 오류 응답은
`{"detail": {"code": ..., "message"/기타}}` 형태.

### 색인 (ingest) — 편성 전 선행 (vision3 발행 훅이 자동 호출)

#### `POST /api/v1/ingest` → 202

증거 수집 → 임베딩 → Milvus 해당 v_id **통째 교체** (delete-insert 멱등 —
재호출 안전. 재발행 후에는 반드시 재ingest).

```bash
curl -X POST localhost:8084/api/v1/ingest \
  -H 'Content-Type: application/json' -d '{"v_id": 202}'
# → {"v_id": 202, "status": "accepted"}
```

- 전제: `t_scene_baseball(source='board')` 발행본 존재 — 없으면 백그라운드 실패(로그 기록).
- 같은 v_id 진행 중이면 **409** `ALREADY_RUNNING`.
- 소요: 수십 초 (임베딩 배치).

#### `GET /api/v1/ingest[?v_id=]` — 색인 현황

```bash
curl 'localhost:8084/api/v1/ingest?v_id=202'
# → {"collection": "sm_scene_evidence", "exists": true, "v_id": 202, "rows": 2095, "running": []}
```

`rows > 0` 이고 `running` 에 없으면 색인 완료. `rows: 0` 이어도 compose 는
동작하지만 벡터 증거 없이 보드 사실(태그·라벨)만으로 편성된다.

### 편성 (compose)

#### `POST /api/v1/compose` → 202

```bash
curl -X POST localhost:8084/api/v1/compose \
  -H 'Content-Type: application/json' \
  -d '{"v_id": 202, "query": "득점 장면 위주 하이라이트", "budget": 180}'
# → {"job_id": "a1b2c3d4e5f6", "status": "running"}
```

- `budget`(초)은 선택 — 주면 질의 해석보다 우선. 생략 시 질의에서 해석 (기본 180).
- `render`(기본 **false**): true 면 편성 ok 후 worker-render 까지 이어 호출하는 원샷 —
  잡 결과에 `render` 필드({status, output_path} 또는 skipped/error 사유)가 추가된다.
  empty·이닝 결손이면 렌더를 생략하고 사유를 남기며, 렌더 실패가 편성 성공을 뒤집지 않는다.
  `bumper`(기본 true)는 render=true 일 때만 의미.
- plan 노드가 thinking 이라 **총 1~3분** — 그래서 202 + 잡 폴링 패턴.
- 질의는 어휘(태그·라벨)로 번역돼 선곡된다. "이닝별 하이라이트"는 이닝 커버리지,
  "득점 장면"은 적시타·역전·동점 등 득점 라벨 위주로 해석된다.

#### `GET /api/v1/compose/{job_id}` — 잡 폴링 (2~5초 간격 권장)

진행 중 — `progress` 에 완료 노드가 순서대로 쌓인다:

```json
{"status": "running", "v_id": 202, "query": "...", "progress": ["retrieve", "plan"]}
```

완료 (`status`: `ok` | `empty` | `error`):

```json
{
  "status": "ok", "v_id": 202, "query": "...",
  "comp_id": 6,
  "spec": {"mode": "collection", "targets": ["적시타", "역전"], "view": "전체",
           "budget": 180, "picked": [25, 36, 38], "reason": "..."},
  "clips": [
    {"scene_id": 38, "h_id": 3, "start": 8789, "end": 8809,
     "label": "안타", "labels": "역전,적시타", "inning": "6회 말",
     "score_before": "3-3", "score_after": "3-5", "cut_mode": "레시피+대사꼬리"}
  ],
  "duration": 96,
  "suspicions": [[41, "verify 의심 사유 (클립은 유지됨)"]],
  "endfix_moved": ["장면38 8805→8809"],
  "orphans": [{"s": 1810.0, "text": "장면 밖 증거 — 발행 누락 의심"}]
}
```

- `clips` 가 결과 본체 — `start`/`end` 는 초(int), 시간순.
- `status: "empty"` = 조건에 맞는 장면 없음 (`clips: []`). 억지로 채우지 않는다.
- `duration` 은 endfix 연장 때문에 budget 을 다소 넘을 수 있다.
- **404** `JOB_NOT_FOUND`: 잡은 인메모리라 프로세스 재시작 시 소실 —
  결과는 `comp_id` 로 재조회한다.

#### `GET /api/v1/compose?comp_id=` — 저장된 편성 재조회 (영구)

```bash
curl 'localhost:8084/api/v1/compose?comp_id=6'
```

`t_compose` 헤더 + `clips` 배열. 없으면 404 `COMPOSE_NOT_FOUND`.

### 렌더 (render) — 편성을 mp4 로

#### `POST /api/v1/render` — 동기 (렌더 완주까지 대기)

저장된 편성(comp_id)을 worker-render 에 넘겨 하이라이트 mp4 를 만든다.
클립을 이닝 키(`3회 초` → `3_top`)로 그룹핑해 전달하며, 원본은
worker-prep-stt 산출 고정 파일명(`source.mp4`)을 쓴다.

```bash
curl -X POST localhost:8084/api/v1/render \
  -H 'Content-Type: application/json' -d '{"comp_id": 5, "bumper": true}'
# → {"comp_id": 5, "v_id": 202, "status": "done",
#    "output_path": "/mnt/nvme/vod/202/c_5.mp4", "error": ""}
```

- `bumper`(기본 true): 이닝 그룹 사이 범퍼 삽입.
- 상태 저장 없음 — 요청-응답으로 끝. 실패는 응답으로 드러난다.
- 사전 차단 (worker 호출 전 이쪽에서 거른다):
  - **409** `COMPOSE_NOT_RENDERABLE` — status=empty 또는 클립 0건 (빈 렌더 금지)
  - **422** `COMPOSE_INVALID_INNING` — 이닝 없는 클립 존재 (발행 데이터 결함 신호)
  - **404** `COMPOSE_NOT_FOUND`
- **502** `RENDER_FAILED` — worker-render 접속 불가·타임아웃·오류 응답.

### 헬스

| 경로 | 용도 |
|------|------|
| `GET /healthz` | 라이브니스 — 프로세스 생존만 |
| `GET /readyz` | 레디니스 — DB·embed·Milvus 3종 프로브, 하나라도 실패면 503 |

```bash
curl localhost:8084/readyz
# → {"status": "ready", "db": "ok", "embed": "ok", "milvus": "ok"}
```

GPU(embed) 야간 자동 중지 시간대엔 readyz 실패가 정상이다.

## 전형적인 수동 시나리오 (재발행 후)

```bash
V=202; H=localhost:8084

# 1) 재색인 (발행본 갱신 반영)
curl -X POST $H/api/v1/ingest -H 'Content-Type: application/json' -d "{\"v_id\": $V}"
watch -n 3 "curl -s '$H/api/v1/ingest?v_id=$V'"     # rows 확인, running 비면 완료

# 2) 편성
JOB=$(curl -s -X POST $H/api/v1/compose -H 'Content-Type: application/json' \
  -d "{\"v_id\": $V, \"query\": \"득점 장면 위주 하이라이트\"}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')

# 3) 폴링
watch -n 3 "curl -s $H/api/v1/compose/$JOB | python3 -m json.tool"
```

## 문서

- `CLAUDE.md` — 코딩 컨벤션·배포·진행 상태
- `.aidoc/design.md` — 설계 근거, bench4 대비 수정 결함 목록
- `.aidoc/compose-flow.md` — flow 상세와 Phase 4 검토 실측
- `.aidoc/vector-collection.md` — Milvus 컬렉션 스키마
