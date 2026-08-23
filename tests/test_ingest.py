"""ingest 순수 함수 검증 — STT 청크·ETC 런 병합·장면 귀속 (DB·네트워크 없음).

상수 근거는 bench4 실측 계승: 발화 단위 색인 실패(파편이 코사인 상위 점령),
여운 귀속(+25s 콜 orphan 실측), ETC 완전 일치 병합(변형까지 뭉치면 교체 순간 소실).
"""

from vector.ingest import build_rows, chunk_stt, merge_etc, owner_of


def scene(scene_id, s, e, **kw):
    """발행본 행 흉내 — 키는 SourceRepo.fetch_scenes 산출과 같아야 한다."""
    return {"scene_id": scene_id, "s": float(s), "e": float(e),
            "tags": kw.get("tags", ["안타"]), "label_list": kw.get("labels", []),
            "board_tags": kw.get("board_tags", ["1루"]),
            "game_context": kw.get("game_context"),
            "score_delta": kw.get("delta", 0), "inning": kw.get("inning", "1회초")}


# ── chunk_stt ──────────────────────────────────────────

def test_chunk_stt_merges_adjacent_and_drops_fragments():
    utts = [(1.0, 2.0, "홈런입니다"), (3.5, 4.0, "대단해요"),   # 간격 1.5 ≤ 2 → 병합
            (10.0, 10.3, "네"),                                  # 병합 불가 + 6자 미만 → 버림
            (20.0, 21.0, "다이빙 캐치")]
    chunks = chunk_stt(utts)
    assert chunks == [(1.0, 4.0, "홈런입니다 대단해요"), (20.0, 21.0, "다이빙 캐치")]


def test_chunk_stt_respects_max_chars():
    long = "가" * 295
    utts = [(1.0, 2.0, long), (3.0, 4.0, "이어지는 말")]   # 295+1+6 > 300 → 새 청크
    chunks = chunk_stt(utts)
    assert len(chunks) == 2 and chunks[1][2] == "이어지는 말"


# ── merge_etc ──────────────────────────────────────────

def test_merge_etc_exact_match_runs_only():
    rows = [(100, "P 김진욱 / 1 김현준"), (102, "P 김진욱 / 1 김현준"),  # 간격 2 ≤ 3 → 런
            (106, "P 김진욱 / 1 구자욱"),                                # 텍스트 다름 → 새 런
            (200, "LG"),                                                 # 5자 미만 → 제외
            ]
    runs = merge_etc(rows)
    assert runs == [(100, 103, "P 김진욱 / 1 김현준"), (106, 107, "P 김진욱 / 1 구자욱")]


# ── owner_of (장면 귀속) ───────────────────────────────

SCENES = [scene(1, 100, 120), scene(2, 200, 230)]


def test_owner_max_overlap():
    assert owner_of(SCENES, 110, 125, "shot")["scene_id"] == 1   # 겹침 10 vs 0
    assert owner_of(SCENES, 118, 210, "shot")["scene_id"] == 2   # 겹침 2 vs 10


def test_owner_stt_trail_attach_within_30s():
    # 장면 1 끝(120) 뒤 +25s 시작 — STT 만 직전 장면 귀속 (여운 해설)
    assert owner_of(SCENES, 145, 150, "stt")["scene_id"] == 1
    assert owner_of(SCENES, 145, 150, "shot") is None            # shot 은 폴백 금지
    assert owner_of(SCENES, 160, 165, "stt") is None             # +40s — 상한 초과


# ── build_rows ─────────────────────────────────────────

def test_build_rows_orphan_kept_and_fields():
    ev = [("shot", 105.0, 112.0, "투구", "투수가 던진다"),
          ("stt", 500.0, 505.0, "", "장면 밖 해설")]              # 어느 장면과도 무관
    rows = build_rows(7, SCENES, ev)
    assert rows[0]["scene_id"] == 1 and rows[0]["shot_type"] == "투구"
    assert rows[1]["scene_id"] == -1                             # orphan 도 버리지 않는다
    # orphan 은 귀속 메타가 전부 빈 값 — 남의 장면 사실이 새면 검색 필터가 오작동한다
    assert rows[1]["tags"] == rows[1]["labels"] == ""
    assert rows[1]["board_tags"] == rows[1]["game_context"] == ""
    assert all(r["v_id"] == 7 for r in rows)


def test_build_rows_carries_four_meta_axes():
    """네 축(행위·파생·판세·전광판 사실)이 각자 칸으로 실린다 — 2026-08-23 상류 개편.

    labels 한 칸에 섞여 오는 해석을 repo 가 둘로 가르고(vocab.split_labels),
    전광판 사실은 별도 컬럼에서 온다. 색인이 이걸 하나로 뭉치면 '역전'·'아웃' 같은
    메타 필터가 어느 축인지 모른 채 걸린다.
    """
    sc = scene(1, 100, 130, tags=["안타"], labels=["적시타"],
               board_tags=["1루", "주자득점", "1점"], game_context="역전")
    row = build_rows(7, [sc], [("stt", 105.0, 110.0, "", "역전 적시타")])[0]
    assert row["tags"] == "안타" and row["labels"] == "적시타"
    assert row["board_tags"] == "1루,주자득점,1점" and row["game_context"] == "역전"


# ── _trunc_bytes (Milvus VARCHAR 바이트 한도) ──────────

def test_trunc_bytes_korean_boundary():
    from vector.ingest import _trunc_bytes
    t = "가" * 400                                  # 1,200바이트
    out = _trunc_bytes(t, 1024)
    assert len(out.encode("utf-8")) <= 1024
    assert out == "가" * 341                        # 1023바이트 — 문자 경계 안전
    assert _trunc_bytes("짧다", 1024) == "짧다"      # 한도 이내는 원문
