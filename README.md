# agent-compose

질의 기반 하이라이트 편성 API. 사용자 질의 1건을 LangGraph flow에 태워
클립 편성(`t_compose`/`t_compose_clip`)을 만들고, 그 재료가 되는 장면 증거를
Milvus(`sm_scene_evidence`)에 색인한다.

```
질의 → rephrase_query(질의 재작성) → retrieve_evidence(벡터 검색)
     → select_clips(LLM 선곡) → refine_end_bound(LLM 끝 확정, 클립당 1콜)
     → refine_start_bound(LLM 시작 확정, 앵커 없는 클립만) → finish → 저장
```

- LLM은 **제안**만(선곡·경계), **처분은 결정적 코드**가 한다 — 선곡은 실존 장면으로
  검산하고, 경계는 제시한 후보 밖의 초를 기각한다.
- **선곡이 곧 편성이다** (2026-08-20): 채점·0점 제외·예산 절단·필수 회수를 폐기했다.
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
  -d '{"v_id": 202, "query": "득점 장면 위주 하이라이트"}'
# → {"job_id": "a1b2c3d4e5f6", "status": "running"}
```

- `render`(기본 **false**): true 면 편성 ok 후 worker-render 까지 이어 호출하는 원샷 —
  잡 결과에 `render` 필드({status, output_path} 또는 skipped/error 사유)가 추가된다.
  원샷은 **완주까지 기다린다**(단독 렌더는 접수만) — 잡 하나로 편성·렌더가 함께 끝난다.
  렌더 성공 시 `t_compose.render_datetime` 도 함께 기록된다.
  empty·이닝 결손이면 렌더를 생략하고 사유를 남기며, 렌더 실패가 편성 성공을 뒤집지 않는다.
  `bumper`(기본 true)는 render=true 일 때만 의미.
- select_clips 가 thinking 이라 **총 1~3분** — 그래서 202 + 잡 폴링 패턴.
- 질의는 어휘(태그·라벨)로 번역돼 선곡된다. "이닝별 하이라이트"는 이닝 커버리지,
  "득점 장면"은 적시타·역전·동점 등 득점 라벨 위주로 해석된다.

#### `GET /api/v1/compose/{job_id}` — 잡 폴링 (2~5초 간격 권장)

진행 중 — `progress` 에 완료 노드 이름이 순서대로 쌓인다. **노드명은 바뀔 수 있으니**
화면에는 매핑한 문구를 쓰고, 미매핑 값은 원문 노출 대신 무시하는 쪽이 안전하다
(2026-08-20 개편 실측: 두 UI 모두 옛 이름 표를 들고 있어 진행 표시가 죽어 있었다):

```json
{"status": "running", "v_id": 202, "query": "...",
 "progress": ["rephrase_query", "retrieve_evidence", "select_clips"]}
```

완료 (`status`: `ok` | `empty` | `error`):

```json
{
  "status": "ok", "v_id": 202, "query": "...",
  "comp_id": 6,
  "spec": {"mode": "compose", "targets": ["적시타", "역전"], "view": "전체",
           "picked": [25, 36, 38]},
  "clips": [
    {"scene_id": 38, "h_id": 3, "start": 8789, "end": 8809,
     "label": "안타", "labels": "역전,적시타", "inning": "6회 말",
     "score_before": "3-3", "score_after": "3-5", "cut_mode": "레시피+대사꼬리"}
  ],
  "duration": 96,
  "end_moved": ["장면38 끝 8805.0→8809.0"],
  "start_moved": ["장면5 시작 657.0→646.0"],
  "orphans": [{"s": 1810.0, "text": "장면 밖 증거 — 발행 누락 의심"}]
}
```

- `clips` 가 결과 본체 — `start`/`end` 는 초(int), 시간순.
- `status: "empty"` = 조건에 맞는 장면 없음 (`clips: []`). 억지로 채우지 않는다.
- 길이 제한이 없다 — 선곡된 장면이 그대로 편성된다(예산 절단 폐기).
- **404** `JOB_NOT_FOUND`: 잡은 인메모리라 프로세스 재시작 시 소실 —
  결과는 `comp_id` 로 재조회한다.

#### `GET /api/v1/compose?comp_id=` — 저장된 편성 재조회 (영구)

```bash
curl 'localhost:8084/api/v1/compose?comp_id=6'
```

`t_compose` 헤더 + `clips` 배열. 없으면 404 `COMPOSE_NOT_FOUND`.

### 렌더 (render) — 편성을 mp4 로

#### `POST /api/v1/render` — 202 접수 (완료는 폴링)

저장된 편성(comp_id)을 worker-render 에 넘겨 하이라이트 mp4 를 만든다.
클립을 이닝 키(`3회 초` → `3_top`)로 그룹핑해 전달하며, 원본은
worker-prep-stt 산출 고정 파일명(`source.mp4`)을 쓴다.
렌더가 GPU 수 분이라 접수만 하고 202 를 돌려준다 — 완료는 서버가 백그라운드로
워커에 물어 확정하고, 호출자는 아래 GET 으로 확인한다.

```bash
curl -X POST localhost:8084/api/v1/render \
  -H 'Content-Type: application/json' -d '{"comp_id": 5, "bumper": true}'
# → 202 {"comp_id": 5, "v_id": 202, "status": "accepted",
#        "output_path": "/mnt/nvme/vod/202/202_5.mp4"}   ← 예정 경로 (파일은 아직 없음)
```

- `bumper`(기본 true): 이닝 그룹 사이 범퍼 삽입 (워커 필드는 `bumper_yn`, 워커 기본값 false).
- `force`(기본 false): 이미 렌더된 편성을 다시 렌더 (범퍼 변경 등 운영용).
- 렌더 상태는 `t_compose.render_status` 한 컬럼으로 읽는다 (`t_code.result` 와 같은 규약 —
  **0 이 '끝난 것'**):

  | 값 | 뜻 | 뷰어 |
  |----|----|------|
  | `NULL` | 렌더 요청된 적 없음 | 렌더 버튼 노출 |
  | `1` | 진행중 (접수~완료 전) | "만드는 중" · 요청 차단 |
  | `0` | 성공 | 영상 준비됨 (`render_datetime` 도 채워진다) |
  | `-1` | 실패 | 재렌더 가능 (실패가 편성을 잠그지 않는다) |

  접수 시 `bumper_yn`(실제 사용값)도 함께 기록한다 — 입력이 아니라 기록이라 기본값이 없다
  (기본 1 이면 미렌더 편성도 "범퍼 켜고 렌더됨"으로 읽힌다).
  진행중(1)은 서비스 재기동에도 DB 에 남으므로, 감시가 끊긴 채 남은 값은 **다음 요청이나
  조회가 워커에 물어 정정한다**(고아 복구). 워커 응답이 없으면 정정하지 않고 막는 쪽을 택한다.
- 사전 차단 (worker 호출 전 이쪽에서 거른다):
  - **409** `COMPOSE_NOT_RENDERABLE` — status=empty 또는 클립 0건 (빈 렌더 금지)
  - **409** `COMPOSE_ALREADY_RENDERED` — 이미 렌더됨 (`rendered_at` 동봉). 사용자 잘못이
    아니라 경쟁 상황이므로 에러가 아닌 안내로 처리하고 재생을 권한다. 재렌더는 `force`.
  - **409** `RENDER_IN_PROGRESS` — 같은 편성이 렌더 중 (완료 전엔 `render_datetime` 이 NULL 이라
    이 검사가 중복을 막는다)
  - **422** `COMPOSE_INVALID_INNING` — 이닝 없는 클립 존재 (발행 데이터 결함 신호)
  - **404** `COMPOSE_NOT_FOUND`
- **502** `RENDER_FAILED` — 접수 자체가 실패 (worker-render 접속 불가·오류 응답).

#### `GET /api/v1/render/{comp_id}` — 렌더 상태

```bash
curl localhost:8084/api/v1/render/5
# → {"comp_id": 5, "v_id": 202, "status": "running", "output_path": "...", "error": ""}
```

`status`: `done`(완료) · `running` · `accepted`(큐 대기) · `error` ·
`not_requested`(렌더 요청된 적 없음) · `unknown`(워커 조회 불가 — GPU 중지 등, `error` 에 사유).

완료 기록이 이미 있으면 워커를 거치지 않고 DB 로 답한다(`rendered_at` 동봉).
기록이 없는데 워커가 `done` 이면 **이 조회가 그 자리에서 기록을 보정한다** — 배포·재기동으로
감시가 끊겨도 조회 한 번이면 되살아난다. 보정까지 실패하면 `stamped: false` 가 함께 온다
(영상은 있으나 기록 실패 — `status_code` 4960).

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
