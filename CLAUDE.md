# agent-compose — 질의 기반 하이라이트 편성 API

`poc/poc-search-bench4`(데모 검증 완료)의 서비스 승격. 질의 → LangGraph flow → 클립 편성
(`t_compose`/`t_compose_clip`) + Milvus 증거 색인(`sm_scene_evidence`)을 소유한다.
설계 근거·flow 맵·이식 시 수정할 결함 목록은 `.aidoc/design.md`.

## 코딩 컨벤션 (agent-vision3 계승 — 항상 지킬 것)

- **`src/` 가 import 루트** (`PYTHONPATH=src`, 비패키지 앱). `from vector...`·`from db...`.
- **설정은 전부 `config.Settings`(.env) — 하드코딩 금지.** 도메인 상수(청크 규칙·랭크
  가중·컷 레시피)는 config 가 아니라 소유 모듈에 실측 근거 주석과 함께.
- **docstring 한국어** Summary/Args/Returns/Description. ruff line-length 100.
- **fail-open 금지** — Milvus·embed 접속 불가는 /readyz 로 드러낸다 (bench4 와 다른 점).
- 클라이언트(DB 풀·Embedder·VectorStore)는 lifespan 1식 공유 — 호출마다 생성 금지.
- **재설계 제약(bench4 실측 — 반증 없이 못 되돌림)**: verify 기각권 없음 / LLM 출력
  JSON 금지(줄 형식) / STT 원문만·대사 조회는 장면 전체 구간 / seg_id 조인 금지 /
  마지막 채움은 결정적 계산.
- 테스트: 네트워크·DB 없는 순수 함수 단위 테스트 (tests/, pytest asyncio_mode=auto).

## 실행

```bash
uv sync && cp .env.example .env             # 값 채우기
PYTHONPATH=src uv run python src/run.py     # .env 의 APP_HOST/APP_PORT (8084)
uv run pytest tests/ -q
```

## 배포 (sm-api-01 — worker-prep-vision 방식)

서버 디렉토리 `/usr/service/source/scenemaker/agent/agent-compose` 가 GitHub
`SceneMakerAI/agent-compose`(**private** — 서버는 agent 계정의 read-only **deploy key**
+ ssh 별칭 `github.com-agent-compose` 으로 fetch)의 clone (소유자 agent).
**정본은 GitHub main** —
로컬 수정은 commit+push 후 서버에서:

```bash
deploy/update.sh            # origin/main 신규 커밋 있으면 sync+재기동, 없으면 no-op
deploy/update.sh --force    # 강제 재기동
```

- `.env` 는 미추적이라 `reset --hard` 에도 보존 (서버 값: Milvus·DB 사설, LLM/embed GA).
- systemd 유닛(`agent-compose.service`)도 `deploy/` 가 정본 — 달라지면 자동 설치.
- readyz 게이트는 embed(GPU)·Milvus 원격 프로브 포함 — **GPU 야간 자동 중지 시간대엔
  실패가 정상** (배포 실패 아님, 스크립트 경고 문구 참조).

## API

- `POST /api/v1/ingest {v_id}` → 202. 증거 수집→임베딩→Milvus v_id 교체 (멱등).
  vision3 발행 직후 호출되는 진입점. 동일 v_id 동시 요청은 409.
- `GET /api/v1/ingest[?v_id=]` — 색인 현황.
- `/healthz`, `/readyz`(DB+embed+Milvus 3종 프로브).

## 상태 (2026-08-17)

- **Phase 1 완료**: 골격(config/log/db/vector/api) + ingest API.
  v200~203 원격(sm-db-01) Milvus 실색인 — 총 7,203행, 검색 스모크 통과
  ("다이빙 캐치 호수비" → 실측 문제 케이스였던 장면60이 STT 콜+캡션으로 1·2위).
  단위 테스트 7건. 로컬 미러는 레거시 삭제 후 신규 스키마 재생성(빈 상태).
- 실측 함정 (코드에 근거 주석 있음): Milvus VARCHAR max_length 는 **바이트** 기준
  (v200 색인 실패 실측 — `_trunc_bytes` 로 UTF-8 경계 절단).
- v202·203 은 shot(캡션) 증거 0 — 새 구조 scene 미실행 상태라 t_segment 캡션 부재.
  scene 실행 후 재ingest 필요 (데이터 상태, 버그 아님).
- **Phase 2 완료 (2026-08-18)**: compose flow(LangGraph) 이식 — `src/flow/`
  (vocab·rank·cut·llm·prompts·plan·state·graph). 결함 4건 수정: A1(endfix 를
  route/backfill 뒤로 재배선 — 충원 클립도 끝보정), A2(verify 사유 "장면 N: …"
  클립별 파싱), A3(backfill total 갱신), B4(Inventory 불변 스냅샷 + 행 복사).
  그래프는 lifespan 1회 컴파일. 어휘는 복제 + 회귀 테스트로 vision3·bench4 와
  동기 고정 (tests/test_flow_deterministic.py — rank·cut 은 bench4 원본 모듈과
  전행 등가 대조). API: POST /api/v1/compose(202+job 폴링)·GET job/comp_id.
  실 e2e(v201 "다이빙 캐치 수비 모음"): retrieve→plan(선곡 3)→backfill(충원 9클립)
  →endfix(끝 이동 2 — 충원 클립 포함 = A1 실증)→verify ok→t_compose comp_id=2.
  테스트 16건.
- **Phase 3 완료 (2026-08-18)**: sm-api-01 배포 — systemd `agent-compose.service`
  (포트 8084, .env: LLM/embed=GA 166.117.29.29:8002/8003, Milvus·DB=사설 192.168.0.5).
  vision3 발행 훅 연결: publish 완료 → POST /api/v1/ingest (fire-and-forget,
  실패해도 발행 유효). 체인 실검증: v201 발행 → 훅 접수 → 21초 뒤 자동 색인 2,095행.
- **Phase 4 검토 완료 (2026-08-18)** — compose-flow.md §6 에 실측 근거:
  질의 재작성 루프 **보류**(6종 프로브 — "후보 부실" 트리거 신호 부재, 오탐 위험 > 이득) ·
  verify 재배치 **보류**(plan thinking 지배라 절감 <3%) · 진행 스트리밍 **적용**
  (checkpointer 없이 — astream 노드 완료를 job `progress` 필드로 노출, sm-api-01 배포됨).
  검증 e2e: v200 "홈런 모음" budget 60 → 홈런 1클립 59초, FULL_CLIP_TAGS 통째 컷+대사꼬리.
- **노드 개편 (2026-08-20)** — 위 Phase 기록의 노드명(retrieve·plan·cutrank·backfill·
  endfix·verify)은 **그 시점 이름**이다. 현재 그래프는 동사_목적어로 통일됐다:
  `rephrase_query → retrieve_evidence → select_clips → refine_end_bound
  → refine_start_bound → finish` (분기 `retry_select`·
  `end_empty`). 최신 흐름·노드표는 `.aidoc/compose-flow.md` §1.
  구현(LLM 사용 여부·임계값)은 이름에 넣지 않는다 — 단계가 LLM↔규칙 사이를 오간다.
  **노드명은 `GET /api/v1/compose/{job_id}` 의 `progress` 배열로 밖에 나간다.**
  바꿀 때 두 UI 의 매핑표를 같이 고쳐야 한다 (ui-sbs-viwer `lib/server/compose-agent.ts`
  NODE_LABEL · ui-workspace `components/compose/ComposeRequestForm.tsx` NODES).
  안 고치면 진행 표시가 조용히 죽는다 — 실제로 두 곳 다 bench4 세대 이름으로
  방치돼 체크가 영영 안 켜지고 내부 노드명이 화면에 노출되고 있었다.
  t_video 상태 코드(4020·4030·4040)는 **국면**이라 값 불변이되, 매핑은 단조여야 한다
  (`_NODE_CODE` — 구 표는 verify 뒤 select 가 4030 으로 되돌아갔다).
