"""P0 — 네 플랫폼의 **오늘 인기 순위** 스냅샷을 하루 한 번 jsonl 에 append 한다.

용도는 **판정이 아니라 진단**이다 (PLAN_BEHAVIORAL_EVAL.md §3-5):
  · 코퍼스 신선도  — 지금 인기 top-N 중 우리 코퍼스에 없는 비율
  · 도달 순위     — 인기작이 그 장르 시드에 대해 몇 위에 나오는가
  · 시의성 편향   — 추천 목록의 출시연도 vs 코퍼스
인기 순위를 P@k 문턱에 넣지 않는다. X-23 에서 인기 top-50 은 대조군(R@50 0.040)이었다.

레코드 한 줄 = 한 플랫폼의 하루치. {"date","platform","source","items":[{rank, id, name, ...}]}
같은 날 두 번 돌리면 두 줄이 쌓인다 — 마지막 줄이 유효.

    python rankings/collect.py            # 전부
    python rankings/collect.py --only steam
"""
from __future__ import annotations
import argparse, json, re, sys, time, datetime as dt
from pathlib import Path
import urllib.request, urllib.parse

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def _get(url, headers=None, timeout=20):
    h = {"User-Agent": UA}; h.update(headers or {})
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
        return r.read()


def _json(url, headers=None):
    return json.loads(_get(url, headers))


def _env(path: Path, keys):
    if not path.exists(): return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"): continue
        k, v = line.split("=", 1)
        if k.strip() in keys: return v.strip().strip('"').strip("'")
    return None


# ── Steam ────────────────────────────────────────────────────────────────────
def steam():
    d = _json("https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/")["response"]
    items = [dict(rank=r["rank"], id=int(r["appid"]), peak_in_game=r.get("peak_in_game"),
                  last_week_rank=r.get("last_week_rank")) for r in d.get("ranks", [])]
    yield "most_played", items
    try:  # 상점 톱셀러 — 있으면 같이 (구조가 바뀌면 조용히 건너뛴다)
        f = _json("https://store.steampowered.com/api/featuredcategories?cc=kr&l=koreana")
        ts = f.get("top_sellers", {}).get("items", [])
        yield "top_sellers", [dict(rank=i + 1, id=int(x["id"]), name=x.get("name")) for i, x in enumerate(ts)]
    except Exception as e:
        print(f"  steam top_sellers 건너뜀: {type(e).__name__}", file=sys.stderr)


# ── TMDB ─────────────────────────────────────────────────────────────────────
def tmdb():
    key = _env(HERE.parent / "tmdb" / ".env", {"TMDB_API_KEY", "API_KEY", "TMDB_KEY"})
    if not key: raise RuntimeError("tmdb/.env 에 키가 없다")
    for src, path in (("trending_movie_day", "trending/movie/day"), ("trending_tv_day", "trending/tv/day"),
                      ("popular_movie", "movie/popular"), ("popular_tv", "tv/popular")):
        items = []
        for page in range(1, 6):                       # 5페이지 = 100편
            d = _json(f"https://api.themoviedb.org/3/{path}?api_key={key}&language=ko-KR&page={page}")
            for x in d.get("results", []):
                m = "tv" if ("tv" in path or x.get("media_type") == "tv") else "movie"
                items.append(dict(rank=len(items) + 1, id=f"{m}_{x['id']}",
                                  name=x.get("title") or x.get("name"), vote_count=x.get("vote_count"),
                                  popularity=x.get("popularity")))
            time.sleep(0.25)
        yield src, items


# ── 네이버 웹툰 ───────────────────────────────────────────────────────────────
def webtoon():
    h = {"Referer": "https://comic.naver.com/"}
    # order=user = 플랫폼 자체의 인기순(요일별). 요일 7개를 각각 한 소스로 남긴다.
    for w in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
        d = _json(f"https://comic.naver.com/api/webtoon/titlelist/weekday?week={w}&order=user", h)
        items = [dict(rank=i + 1, id=int(x["titleId"]), name=x.get("titleName"),
                      star=x.get("starScore"), adult=bool(x.get("adult", False)))
                 for i, x in enumerate(d.get("titleList", []))]
        yield f"weekday_{w}", items
        time.sleep(0.3)


# ── 네이버 시리즈 웹소설 ─────────────────────────────────────────────────────
def webnovel():
    seen, items = set(), []
    for page in range(1, 6):
        html = _get(f"https://series.naver.com/novel/top100List.series?page={page}").decode("utf-8", "ignore")
        ids = [int(x) for x in re.findall(r"productNo=(\d+)", html)]
        new = [i for i in dict.fromkeys(ids) if i not in seen]
        if not new: break
        for i in new:
            seen.add(i); items.append(dict(rank=len(items) + 1, id=i))
        time.sleep(0.4)
    yield "top100", items


COLLECTORS = {"steam": steam, "tmdb": tmdb, "webtoon": webtoon, "webnovel": webnovel}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat(); ts = dt.datetime.now().isoformat(timespec="seconds")
    ok = fail = 0
    for plat, fn in COLLECTORS.items():
        if a.only and plat != a.only: continue
        try:
            with (OUT / f"{plat}.jsonl").open("a", encoding="utf-8") as f:
                for src, items in fn():
                    f.write(json.dumps(dict(date=today, ts=ts, platform=plat, source=src,
                                            n=len(items), items=items), ensure_ascii=False) + "\n")
                    print(f"  {plat:<8} {src:<20} {len(items):>4}개", flush=True); ok += 1
        except Exception as e:
            fail += 1; print(f"  {plat:<8} 실패: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print(f"{today} · 소스 {ok}개 저장 · 실패 {fail}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
