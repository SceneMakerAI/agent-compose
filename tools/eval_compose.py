"""편성 품질 채점 — 재설계 판정용 단일 잣대.

flow 를 바꿀 때마다 "좋아졌나" 를 말이 아니라 수치로 답하기 위한 도구.
어떤 구조로 바꾸든 이 지표가 같은 질의·같은 예산에서 나아지지 않으면 채택하지 않는다.

지표 (전부 발행본 t_scene_baseball 을 정답지로 삼는 결정적 계산 — LLM 무관):
- 득점 포함률: 득점이 난 장면(score_delta>0) 중 편성에 들어온 비율.
  누락은 그대로 "놓친 하이라이트" 다. 야구 하이라이트의 최소 요건.
- 라벨 포함률: 역전·동점·끝내기·경기 종료 등 결정적 라벨 장면의 포함률.
- 이닝 커버리지: 발행 장면이 있는 이닝 중 편성이 덮은 비율.
  단, **득점 0 이닝은 분모에서 뺀 값도 함께 낸다** — 무득점 이닝을 억지로 채우는 것은
  품질이 아니다 (v201 실측: 빠진 7이닝 전부 득점 0 = 정당한 제외).
- 예산 준수: |결과 - 예산| / 예산. 초과·미달 모두 벌점.
- 중복: 같은 scene_id 가 두 번 들어갔는가 (0 이어야 한다).

**해석 주의**: 득점 포함률·이닝 커버리지는 "경기 전체 하이라이트" 류 질의에서만 의미가
있다. `"홈런 모음"` 처럼 좁은 질의는 홈런 아닌 득점을 빼는 게 정답이므로 낮은 포함률이
정상이다 — 그런 질의는 예산 준수·중복만 본다. 재설계 A/B 는 **같은 질의·같은 예산**끼리만
비교할 것.

사용:
    PYTHONPATH=src uv run --with pymysql python tools/eval_compose.py 11 12 13
    PYTHONPATH=src uv run --with pymysql python tools/eval_compose.py --v-id 201  # 최신 편성
"""

import argparse
import os
import sys

import pymysql

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DECISIVE_LABELS = ("역전", "동점", "끝내기", "경기 종료")


def _conn():
    """.env 의 접속 정보로 연결 — 접속값 하드코딩 금지 (레포 관례)."""
    env = {}
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return pymysql.connect(
        host=env["DB_IP"], port=int(env.get("DB_PORT", 3306)),
        user=env["DB_USER"], password=env["DB_PW"], db=env["DB_NAME"],
        charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor,
    )


def score(cur, comp_id: int) -> dict | None:
    cur.execute("SELECT * FROM t_compose WHERE comp_id=%s", (comp_id,))
    head = cur.fetchone()
    if not head:
        return None
    v_id = head["v_id"]

    cur.execute(
        "SELECT scene_id, inning, scene_type, labels FROM t_compose_clip WHERE comp_id=%s",
        (comp_id,))
    clips = cur.fetchall()
    picked = [c["scene_id"] for c in clips]

    cur.execute(
        "SELECT scene_id, inning, score_delta, IFNULL(labels,'') labels "
        "FROM t_scene_baseball WHERE v_id=%s AND source='board'", (v_id,))
    published = cur.fetchall()
    if not published:
        return None
    chosen = set(picked)

    scoring = [r for r in published if r["score_delta"] > 0]
    decisive = [r for r in published
                if any(lab in r["labels"] for lab in DECISIVE_LABELS)]
    innings_all = {r["inning"] for r in published if r["inning"]}
    innings_scoring = {r["inning"] for r in scoring if r["inning"]}
    innings_hit = {c["inning"] for c in clips if c["inning"]}

    def frac(hit, total):
        return (len(hit) / len(total)) if total else 1.0

    budget = head["budget_sec"] or 0
    return {
        "comp_id": comp_id, "v_id": v_id, "query": head["query"],
        "mode": head["mode"], "budget": budget,
        "duration": head["duration"], "clips": len(clips),
        "budget_dev": (head["duration"] - budget) / budget if budget else 0.0,
        "scoring_hit": len([r for r in scoring if r["scene_id"] in chosen]),
        "scoring_total": len(scoring),
        "scoring_miss": [r["scene_id"] for r in scoring if r["scene_id"] not in chosen],
        "decisive_hit": len([r for r in decisive if r["scene_id"] in chosen]),
        "decisive_total": len(decisive),
        "decisive_miss": [(r["scene_id"], r["labels"]) for r in decisive
                          if r["scene_id"] not in chosen],
        "inning_cov": frac(innings_hit & innings_all, innings_all),
        "inning_cov_scoring": frac(innings_hit & innings_scoring, innings_scoring),
        "dups": len(picked) - len(chosen),
        "avg_clip": (head["duration"] / len(clips)) if clips else 0,
    }


def report(rows: list[dict]) -> None:
    for m in rows:
        print(f"\n=== comp {m['comp_id']} · v{m['v_id']} · {m['mode']} "
              f"· \"{m['query'][:40]}\"")
        print(f"  득점 포함    {m['scoring_hit']}/{m['scoring_total']}"
              f" ({m['scoring_hit'] / max(m['scoring_total'], 1):.0%})"
              + (f"  누락 scene {m['scoring_miss']}" if m["scoring_miss"] else ""))
        print(f"  결정 라벨    {m['decisive_hit']}/{m['decisive_total']}"
              + (f"  누락 {m['decisive_miss']}" if m["decisive_miss"] else ""))
        print(f"  이닝 커버    전체 {m['inning_cov']:.0%}"
              f" · 득점이닝 {m['inning_cov_scoring']:.0%}")
        print(f"  예산         {m['duration']}s / {m['budget']}s"
              f" ({m['budget_dev']:+.1%})")
        print(f"  클립         {m['clips']}개 · 평균 {m['avg_clip']:.0f}s"
              + (f"  ⚠ 중복 {m['dups']}건" if m["dups"] else ""))

    if len(rows) > 1:
        n = len(rows)
        print(f"\n=== 합계 ({n}건) ===")
        print(f"  득점 포함률   {sum(m['scoring_hit'] for m in rows)}"
              f"/{sum(m['scoring_total'] for m in rows)}")
        print(f"  결정 라벨     {sum(m['decisive_hit'] for m in rows)}"
              f"/{sum(m['decisive_total'] for m in rows)}")
        print(f"  이닝 커버(득점) 평균 {sum(m['inning_cov_scoring'] for m in rows) / n:.0%}")
        print(f"  예산 편차 평균 {sum(abs(m['budget_dev']) for m in rows) / n:+.1%}")
        print(f"  중복 합계     {sum(m['dups'] for m in rows)}건")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("comp_ids", nargs="*", type=int)
    ap.add_argument("--v-id", type=int, action="append",
                    help="해당 v_id 의 최신 편성을 채점 (반복 지정 가능)")
    args = ap.parse_args()

    conn = _conn()
    try:
        with conn.cursor() as cur:
            ids = list(args.comp_ids)
            for v in args.v_id or []:
                cur.execute(
                    "SELECT comp_id FROM t_compose WHERE v_id=%s "
                    "ORDER BY comp_id DESC LIMIT 1", (v,))
                row = cur.fetchone()
                if row:
                    ids.append(row["comp_id"])
                else:
                    print(f"v{v}: 편성 없음", file=sys.stderr)
            if not ids:
                cur.execute("SELECT comp_id FROM t_compose ORDER BY comp_id")
                ids = [r["comp_id"] for r in cur.fetchall()]
            rows = [m for cid in ids if (m := score(cur, cid))]
    finally:
        conn.close()

    if not rows:
        print("채점할 편성이 없다 (t_compose 비었거나 발행본 부재)", file=sys.stderr)
        sys.exit(1)
    report(rows)


if __name__ == "__main__":
    main()
