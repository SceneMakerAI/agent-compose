# Phase 2 — compose flow 설계 (v0.1, 2026-08-18)

bench4 LangGraph flow 의 서비스 이식 설계. 결함 A1~A3·B4 수정을 포함하되,
**판정 규칙·컷 레시피·프롬프트는 원문 이식**(등가 검증 가능 단위 유지)이 원칙.

## 1. 그래프 (bench4 대비 재배선)

```
START → retrieve → plan★ → cutrank → route ─┬→ backfill ─┐
                    ▲                        ├────────────┼→ endfix★ → verify★ → END
                    │                        │            │
                    └── feedback ←───────────┤ (0건, 최대 1회)
                                             └→ empty → END
```

**bench4 와 다른 점 — endfix 를 route/backfill 뒤로 이동 (A1 수정).**
구그래프(`cutrank → endfix → route → backfill`)에선 backfill 충원 클립이 LLM 끝보정을
못 받아 경로별 품질이 불균일했다. 신그래프는 **발행될 클립 집합이 확정된 뒤** endfix 가
한 번 돌므로 전 클립이 같은 보정을 받는다. LLM 콜 수는 동일(경로당 endfix 1회).
의도된 의미 변화 1건: route 의 미달 판정(총길이 < 예산×0.7)이 bench4 는 endfix
**연장 후** 총길이를 봤지만 신그래프는 **연장 전**을 본다 — endfix 는 연장만 하므로
경계 케이스에서 bench4 보다 backfill 이 한 번 더 발동할 수 있다 (수정의 수용된 부수효과).
검증: 2026-08-18 실 e2e(v201 수비 질의)에서 backfill 충원 클립(장면 70)의 끝 이동
확인 — bench4 구조에선 불가능한 동작.

| 노드 | LLM | 역할 (bench4 원문 이식 + 수정 표기) |
|---|---|---|
| retrieve | — | 질의 임베딩(Embedder) → VectorStore 검색 → scene_id 그룹핑(히트수→유사도) 상위 8 + orphan 분리. **fail-open 폐기** — Milvus 예외는 전파(플로우 실패) |
| plan | ★ thinking | 인벤토리+벡터 후보(+feedback) → 선곡 spec. 실존 scene_id 검산 통과분만. budget 인자는 질의 해석을 덮어씀 |
| cutrank | — | rank 정렬 → 샷 레시피 컷 → greedy 예산 채움(picked/spare/total). **행 복사 후 수정 (B4)** |
| route | — | 0건→feedback(1회)·소진→empty / 예산 70% 미달→backfill / 그 외→endfix |
| backfill | — | plan 태그 범위 안 결정적 충원. **state.total 갱신 (A3)** |
| feedback | — | 0건 전용 재선곡 사유 → plan 재진입 (비결정 재선곡은 폐기 유지 — bench4 확정) |
| endfix | ★ | 전 채택 클립의 끝 근처 STT 발화 제시(검증기 수용 가능분만) → 제안 검증 후 처분 |
| verify | ★ | 3층 증거 검수 — 기각권 없음(실측 확정). **의심 사유를 클립별 파싱 (A2)**: 응답 줄 형식 `클립N: 사유` 를 N별로 매핑, 매핑 실패 줄은 공통 소견으로 강등 |
| empty | — | picked=[], status=empty |

LLM 콜: happy path 3회(plan+endfix+verify), 재선곡 시 4회, empty 2회. + 임베딩 1회.

## 2. State 불변 규약 (B4 수정 — 서비스 동시성의 전제)

```python
class ComposeState(TypedDict, total=False):
    # 요청 입력 (불변)
    v_id: int; query: str; budget: int | None
    # 요청 시점 스냅샷 (불변 — 노드는 절대 수정하지 않는다)
    inv: Inventory          # scenes·segs·utts 튜플화 (아래)
    # 흐름 상태 (노드가 "교체"만 한다 — in-place 수정 금지)
    evidence: list[dict]; evidence_orphan: list[dict]
    feedback: str; attempt: int
    spec: dict
    picked: list[dict]; spare: list[dict]; total: int
    endfix_applied: int
    suspicions: list[tuple[int, str]]
    status: str             # ok | empty
```

- **Inventory 는 요청 시작에 1회 fetch 후 튜플/불변 dict 로 굳힌다.** bench4 는
  scenes dict 를 클로저로 들고 cutrank·endfix 가 in-place 수정했다 — 요청마다 새로
  만들어 우연히 안전했을 뿐, 캐시·동시 요청 도입 즉시 깨진다. 신설계 규칙:
  **노드가 만드는 모든 행은 복사본**(`{**scene}`), 수정은 자기 복사본에만.
- 그래프는 **lifespan 에서 1회 컴파일**(상태 없는 순수 배선) — bench4 는 요청마다
  컴파일했다. 자원(Embedder·VectorStore·LLM·DB)은 app.state 주입, State 엔 안 넣는다
  (직렬화 불가물 배제 — 향후 checkpointer 도입 여지).
- reducer 없는 전체 교체 의미 유지 + `stream(stream_mode="updates")` 노드 순서 로깅 계승.

## 3. 모듈 배치

```
src/flow/
  state.py        ComposeState + Inventory (불변 규약 docstring)
  graph.py        build_graph(resources) — 배선·컴파일 (lifespan 1회)
  llm.py          chat 호출 (줄 형식 강제·thinking 선별·빈 본문 1회 재시도 — bench4 이식)
  prompts.py      plan/endfix/verify 프롬프트 (원문 이식 — byte 등가 검증 대상)
  rank.py cut.py  순수 계산 (원문 이식 + 행 복사)
  nodes.py        노드 함수들 (retrieve~verify)
  vocab.py        태그·라벨 어휘 + CUT_RECIPE·LABEL_EXTRA_SHOTS
src/db/compose_repo.py   t_compose(+clip) 저장 + 조회
src/api/compose.py       라우트
```

**어휘 단일 원천 (드리프트 해소)**: vision3 `sports/baseball/vocab.py` 를 **복제**하되,
모노레포 상대 경로로 원본을 로드해 **등가를 회귀 테스트로 고정**한다
(`tests/test_vocab_sync.py` — PLAY_TAGS 이름·속성 대조). 공유 패키지 추출은 배포 체계가
둘로 갈라진 현 구조(vision3=sm-api-01, compose=동일 호스트 예정)에서 이득보다 비용이 커
보류. CUT_RECIPE 키는 vocab 의 태그명에서 파생시켜 존재하지 않는 태그 키를 부팅 시 검증.

## 4. API 계약

```
POST /api/v1/compose {v_id, query, budget?}   → 202 {job_id}
GET  /api/v1/compose/{job_id}                 → {status: running|ok|empty|error,
                                                 comp_id?, clips?, report?}
GET  /api/v1/compose?comp_id=N                → 저장된 편성 재조회 (t_compose_clip)
```

- 비동기 202 패턴 — plan(thinking)이 1~3분. job 은 프로세스 내 레지스트리(단일 워커
  전제, ingest 와 동일). 완료 시 t_compose/t_compose_clip 저장(요청마다 새 comp_id —
  이력 보존)·EDL JSON 은 응답에 포함(파일 산출은 슬러그 충돌 문제로 폐기, DB가 원본).
- 같은 v_id 동시 편성은 허용 (읽기 전용 + 새 comp_id — ingest 와 달리 충돌 없음).

## 5. 등가 검증 계획 (이식 게이트)

1. **프롬프트 byte 등가**: plan/endfix/verify 프롬프트 렌더 결과를 bench4 와 같은 입력
   으로 대조 (v201 스냅샷 고정) — 재배선·불변화가 프롬프트에 영향 없음을 증명.
2. **결정 경로 등가**: rank 순서·cut 구간·_assemble 채움·backfill 충원을 bench4 모듈
   직접 호출과 대조 (LLM 무관 — 전행 일치 요구).
3. ~~e2e 대조 (bench4 CLI 기준본과 클립 목록 비교)~~ — **범위 제외 (2026-08-18)**:
   기준본이 재현 불가가 됐다 — 당일 DB 재발행(v200~203)·로컬 Milvus 비움·컬렉션
   sm_db 이전으로, bench4 CLI(.env localhost)는 fail-open 무벡터 모드로 돌아
   비교 가능한 입력이 아니다. 게이트 1(byte 등가)+2(결정층 전행 등가)+그래프 경로
   스텁 고정+실 e2e 1회로 대체 — 이 체인이 LLM 밖 전 구간을 덮는다.
4. A1 수정 효과 확인: backfill 경로 질의에서 충원 클립에 endfix 적용 로그 확인
   → **완료** (실 e2e 장면 70 끝 이동 실증).

## 6. Phase 4 검토 결과 (2026-08-18 실측)

- **retrieve 질의 재작성 루프 — 보류.** 6종 질의 프로브 실측: "후보 부실" 트리거 신호가
  없다 — 존재하지 않는 개념("끝내기", top sim 0.613)이 실재 은어("낫아웃", 0.563)보다
  유사도가 높고, top-20 구조상 그룹은 항상 5~8개 나와 0건 트리거도 발동 불가.
  어휘 번역(더블플레이→병살)은 plan 이 주 경로라 벡터 부실이 결과를 해치지 않는 구조.
  재작성은 무관 질의에서 무관 후보를 강화할 오탐 위험 > 이득. 재검토 트리거:
  실사용 질의 로그에서 은어 미스가 관측될 때.
- **verify 재배치 — 보류.** verify 는 non-thinking ~3-5초, 전체 시간은 plan thinking
  (80~180초)이 지배 — 비동기화 절감 <3% 에 job 상태 복잡도만 증가. 소견은 job 응답
  suspicions 필드로 충분.
- **진행 스트리밍 — 적용 (checkpointer 없이).** astream(updates) 노드 완료를 job 의
  progress 필드로 노출 — 폴링 중 현재 단계가 보인다. checkpointer 는 미도입:
  1~3분 잡은 실패 시 재실행이 더 싸다.
- 잔여 후보: consolidate 득점 계열·서사 배치 고도화 (bench4 §10 안건 6).
