"""랭킹 대비 코퍼스 피복률 — **크롤이 뒤처졌는지 알려주는 경보**.

랭킹은 "세상이 지금 보는 작품 목록"이다. 거기 있는데 우리 코퍼스에 없으면
그 작품은 **어떤 시드를 넣어도 추천될 수 없다.** 임베딩도 랭커도 손댈 것이 없고,
크롤을 다시 돌려야 하는 문제다.

이 검사는 시계열이 필요 없다. 오늘 스냅샷 한 장이면 바로 답이 나온다
(랭킹의 고유 가치인 변화율·급상승은 시간축이 쌓여야 쓸 수 있고, 그건 별개다).

**누락이 전부 크롤 지연은 아니다.** 코퍼스마다 수집 정책이 있고, 정책상 제외된 것을
경보로 올리면 늑대 소년이 된다. 그래서 아래 `POLICY` 로 "원래 안 받는 것"을 먼저 걷어낸다.

    TMDB 는 vote_count 29 미만을 애초에 안 받는다(코퍼스 실측 최솟값 29).
    그런데 trending 에는 3표짜리 신작이 그대로 올라온다 — 누락작 vote_count 중앙이 16 이다.
    이건 지연이 아니라 **설계대로** 빠진 것이다.

실측 (2026-09-13):
    웹소설 top100 중 43개 누락 (1위 포함) · Steam most_played 11개 · 웹툰 요일당 6~14개

    python rankings/coverage.py              # 사람이 읽는 표
    python rankings/coverage.py --quiet      # 경보만 (크론용)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE.parent
DATA = HERE / "data"

#: 플랫폼 → (스냅샷, 코퍼스, id 열, 랭킹 id 변환)
#: 웹소설만 저장소가 다르다 (aod-webnovel).
SOURCES = {
    "steam": (DATA / "steam.jsonl",
              R / "steam/artifacts/tags_full/dataset.parquet", "steam_appid", int),
    "tmdb": (DATA / "tmdb.jsonl",
             R / "tmdb/artifacts/tmdb_v1/dataset.parquet", "item_id", str),
    "webtoon": (DATA / "webtoon.jsonl",
                R / "webtoon/artifacts/wt_v1/dataset.parquet", "item_id", int),
    "webnovel": (DATA / "webnovel.jsonl",
                 Path("/home/ubuntu/aod-webnovel/recommendation/webnovel"
                      "/artifacts/wn_v6/dataset.parquet"), "item_id", int),
}

#: 이 아래로 떨어지면 경보. 랭킹 상위가 코퍼스에 없다는 건 크롤 지연이다.
DEFAULT_FLOOR = 0.90

#: 플랫폼 → 랭킹 항목이 **원래 코퍼스 대상이 아닌** 조건. 피복률 분모에서 뺀다.
#: 없는 정보로는 판단하지 않는다 — 랭킹에 그 필드가 없으면 대상으로 친다.
POLICY = {
    # TMDB 크롤은 vote_count 하한이 있다(코퍼스 실측 최솟값 29).
    "tmdb": lambda i: i.get("vote_count") is not None and i["vote_count"] < 29,
    # 웹툰 성인물은 수집 단계에서 401 로 빠지고 후처리에서도 제외한다.
    "webtoon": lambda i: bool(i.get("adult")),
}


def latest_by_source(path: Path) -> dict:
    """소스별 **가장 최근** 스냅샷. 한 파일에 여러 소스가 섞여 있다(steam 은 2종)."""
    out = {}
    if not path.exists():
        return out
    for line in path.open():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[o.get("source", "?")] = o
    return out


def check(platform: str, floor: float):
    snap_path, corpus_path, col, cast = SOURCES[platform]
    if not corpus_path.exists():
        return [(platform, "?", None, 0, 0, f"코퍼스 없음: {corpus_path}", None, 0)]
    have = set(pd.read_parquet(corpus_path, columns=[col])[col])
    rows = []
    for source, snap in latest_by_source(snap_path).items():
        raw = snap.get("items") or []
        if not raw:
            continue
        skip = POLICY.get(platform, lambda i: False)
        items = [i for i in raw if not skip(i)]
        excluded = len(raw) - len(items)
        if not items:
            continue
        miss = [(i.get("rank"), cast(i["id"]), i.get("name"))
                for i in items if cast(i["id"]) not in have]
        n = len({cast(i["id"]) for i in items})
        cov = (n - len({m[1] for m in miss})) / n
        rows.append((platform, source, snap.get("date"), n, len(miss),
                     sorted(miss)[:5], None if cov >= floor else "경보", excluded))
    return rows


def main():
    ap = argparse.ArgumentParser(description="랭킹 대비 코퍼스 피복률")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--quiet", action="store_true", help="경보가 있을 때만 출력")
    a = ap.parse_args()

    alarms = []
    lines = []
    for p in SOURCES:
        for plat, source, date, n, nmiss, sample, alarm, excluded in check(p, a.floor):
            if isinstance(sample, str):        # 코퍼스 없음
                lines.append(f"  {plat:9} {sample}")
                continue
            cov = (n - nmiss) / n * 100
            mark = "⚠" if alarm else " "
            ex = f" (정책 제외 {excluded})" if excluded else ""
            lines.append(f"{mark} {plat:9} {source:14} {date}  대상 {n:4}개 중 누락 "
                         f"{nmiss:3}  피복 {cov:5.1f}%{ex}")
            if nmiss:
                top = ", ".join(f"{r}위 {nm or i}" for r, i, nm in sample)
                lines.append(f"    누락 상위: {top}")
            if alarm:
                alarms.append(f"{plat}/{source} {cov:.1f}%")

    if alarms:
        print(f"⚠ 크롤 지연 경보 — {' · '.join(alarms)}  (문턱 {a.floor*100:.0f}%)")
    elif a.quiet:
        return 0
    print("\n".join(lines))
    return 1 if alarms else 0


if __name__ == "__main__":
    sys.exit(main())
