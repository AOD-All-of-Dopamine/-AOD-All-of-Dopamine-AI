"""기준 목록 도구(`steam_baseline` · `webnovel_baseline`)의 공통부.

두 도구는 같은 규칙으로 돌아야 한다 — 같은 머신이면 **완전 일치**, 다른 머신이면 점수가
같은 자리의 순서 차이만 허용(`--allow-ties`). 규칙이 두 파일에서 갈라지지 않게 여기 한 곳에
적는다. 사례 목록·실행 방법·meta 의 `what`/`corpus` 만 도구마다 다르다.
"""
from __future__ import annotations
import argparse, json, platform as _pf, subprocess, sys
from datetime import datetime, timezone

from aod_serving.tools.compare import pages_equal


def parse_args(doc: str, argv=None):
    """공통 CLI. 인자 조합이 틀리면 **None** 을 돌려준다 — 호출부는 그때 2 로 끝낸다."""
    ap = argparse.ArgumentParser(description=doc)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out"); g.add_argument("--check"); g.add_argument("--bench", action="store_true")
    ap.add_argument("--allow-ties", action="store_true")
    ap.add_argument("--cases", help="사례 id 에 이 문자열이 든 것만 돌린다(개발용 빠른 반복). "
                                    "--check 도 고른 사례만 비교한다 — 최종 확인은 필터 없이.")
    a = ap.parse_args(argv)
    if a.cases and (a.out or a.bench):
        print("--cases 는 --check 에서만 쓴다 — 기준 목록(--out)은 전체로만 만들고 "
              "--bench 는 고정 사례를 잰다", file=sys.stderr)
        return None
    return a


def code_sha(root) -> str:
    """기준 목록에 적을 커밋. dev 이미지에는 git 이 없다 — 그때는 빈 값으로 두고 나중에 손으로 채운다."""
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def meta(what: str, corpus: str, sha: str) -> dict:
    """기준 목록의 머리말. 다른 머신에서 비교할 때 무엇이 달랐는지 알아보기 위한 기록이다."""
    import numpy, pandas as pd
    return {"what": what,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "machine": "팀장 PC · Docker python:3.12-slim (평가 서버 아님)", "cpu": _pf.processor() or _pf.machine(),
            "numpy": numpy.__version__, "pandas": pd.__version__, "corpus": corpus, "code_sha": sha or "unknown"}


def save(path: str, doc: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)


def diff(want: dict, got: dict, allow_ties: bool) -> list[str]:
    problems = []
    for cid, wp in want.items():
        gp = got[cid]
        if len(wp) != len(gp):   # zip 이 조용히 잘라내지 않게 먼저 본다
            problems.append(f"{cid} 쪽 수 다름: 기준 {len(wp)} · 결과 {len(gp)}")
        for i, (w, g) in enumerate(zip(wp, gp)):
            if w["ids"] == g["ids"] and w["scores"] == g["scores"]:
                continue
            if allow_ties and pages_equal(w["ids"], g["ids"], w["scores"], allow_ties=True):
                print(f"  동점 순서 차이(허용) {cid} p{i + 1}", file=sys.stderr); continue
            problems.append(f"{cid} p{i + 1}\n    want {w['ids']}\n    got  {g['ids']}")
    return problems


def check(path: str, got: dict, cases: str | None, allow_ties: bool) -> int:
    """기준 파일과 지금 결과를 비교하고 종료 코드를 돌려준다."""
    want_doc = json.load(open(path, encoding="utf-8"))
    want = {c["id"]: c["pages"] for c in want_doc["cases"]}
    if cases:   # 고른 사례만 비교한다 — 나머지를 "사례 없음"으로 세지 않는다
        want = {k: v for k, v in want.items() if cases in k}
    missing = set(want) - set(got)
    problems = [f"사례 없음: {m}" for m in missing] + diff({k: v for k, v in want.items() if k in got}, got, allow_ties)
    print(f"사례 {len(want)} · 어긋남 {len(problems)}")
    for p in problems[:10]:
        print(" ", p)
    return 1 if problems else 0
