# Phase 2 — compose flow 설계 (v0.1, 2026-08-18)

bench4 LangGraph flow 의 서비스 이식 설계. 결함 A1~A3·B4 수정을 포함하되,
**판정 규칙·컷 레시피·프롬프트는 원문 이식**(등가 검증 가능 단위 유지)이 원칙.

## 1. 그래프 (2026-08-20 현재)

```
START → rephrase_query → retrieve_evidence → select_clips★ → set_bounds
                              ▲                                   │
                              │                    ┌──────────────┴──────────────┐
                              └── retry_select ←───┤ (0건, 최대 MAX_REPLAN 회)   │
                                       end_empty ←─┘ (소진)          refine_bounds★
                                          │                                │
                                          ▼                          score_match★
                                         END                               │
                                                                    drop_unmatched
                                                                           │
                                                                     fill_budget → END
```

★ = LLM 콜. happy path 는 4노드에서 콜이 나가고, refine_bounds·score_match 는
**클립당 1콜을 동시에** 보낸다 (묶어 보내면 전송이 직렬이라 GPU 가 논다 — 실측 v201
24클립 10분 26초 동안 서버 Running 1·KV 3%).

노드 이름은 전부 **동사_목적어**이고 그 노드가 만들어 내는 것을 가리킨다. 구현(LLM
사용 여부·임계값)은 이름에 넣지 않는다 — 이 파이프라인은 단계가 LLM↔규칙 사이를
오간 전례가 있다(구 `drop0` 은 임계값이 바뀌면 거짓말이 되는 이름이었고, 구 `bounds`
의 시작 판정은 `set_bounds` 규칙으로 내려왔다).

| 노드 | LLM | 역할 |
|---|---|---|
| rephrase_query | ★ | 질의를 중계의 언어로 다시 쓴다 + 메타 필터 힌트. 실패는 원 질의 폴백 (실측: 추상 질의 최고 유사도 0.58 vs 구체 질의 0.66~0.78) |
| retrieve_evidence | — | 검색어별 임베딩 → VectorStore → scene_id 그룹핑 상위 8 + orphan. **fail-open 폐기** — Milvus 예외는 전파 |
| select_clips | ★ thinking | 인벤토리+벡터 후보(+feedback) → 선곡 spec. 실존 scene_id 검산 통과분만. budget 인자가 질의 해석을 덮어씀 |
| set_bounds | — | 샷 레시피로 구간 확정 + **필수 장면 회수**(선곡이 놓친 득점·역전 등). FULL_CLIP_TAGS 는 끝만 통째, 시작은 앵커 |
| retry_select | — | 0건 전용 재선곡 사유 → select_clips 재진입 |
| refine_bounds | ★ 클립당 | **앵커 없는 클립만** 경계 후보를 제시하고 고르게 한다. 시작 후보는 앞으로만 (되돌리는 방향은 전부 앞 플레이의 투구 — 5경기 29건 전수) |
| score_match | ★ 클립당 | 질의 일치도 0~3 + 완결성. **기각권 없음** — 점수만 매긴다 |
| drop_unmatched | — | 0점 제외 (필수 장면은 예외 — 사실이 소견을 이긴다) |
| fill_budget | — | 순서(일치도→이닝 분산→중요도)를 정하고 예산만큼 담는다. **절단이 마지막**이라 총 길이가 정확하다 |
| end_empty | — | picked=[], status=empty |

**절단이 마지막인 이유**: 예전에는 경계 확정 전에 잘랐고 그 뒤 끝 보정이 끝을 늘려
예산 보장이 무효가 됐다 (실측: 900초 요청에 947~1018초).

> 이 문서의 §3 이하와 `redesign.md`·`audit.md` 는 **bench4 이식 시점 기록**이라
> 구 노드명(retrieve·plan·cutrank·route·backfill·endfix·verify)이 그대로 남아 있다.
> 그때 무엇을 보고 무슨 판단을 했는지가 근거라 소급 수정하지 않는다.

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
