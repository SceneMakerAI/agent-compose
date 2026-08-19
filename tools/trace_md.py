"""기존 트레이스 JSON → 읽기용 Markdown 변환 (배포 전 실행분 구제용).

서비스는 이제 .json 과 .md 를 같이 떨구지만, 그 전에 쌓인 트레이스는 JSON 뿐이다.
같은 렌더러를 쓰므로 새로 남는 것과 형식이 어긋나지 않는다.

사용:
    PYTHONPATH=src uv run python tools/trace_md.py <파일|디렉토리>…
"""

import json
import sys
from pathlib import Path
from trace import render_md


def convert(path: Path) -> None:
    """JSON 1건 → 같은 이름 .md. 이미 있으면 덮어쓴다(렌더러가 정본)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    out = path.with_suffix(".md")
    out.write_text(render_md(data), encoding="utf-8")
    print(f"{path.name} → {out.name} ({out.stat().st_size:,}바이트)")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for arg in argv:
        p = Path(arg)
        targets = sorted(p.glob("*.json")) if p.is_dir() else [p]
        if not targets:
            print(f"대상 없음: {arg}", file=sys.stderr)
        for t in targets:
            convert(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
