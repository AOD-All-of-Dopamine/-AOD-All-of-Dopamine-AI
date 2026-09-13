"""네이버 웹툰 공식 연재/완결 작품을 크롤링해 `webtoons.jsonl` 을 만든다.

`webnovel/src/crawl_naverseries.py` 의 이어받기·적응형 레이트 구조를 따르되, 네이버 웹툰은
**공개 JSON API** 라 HTML 파싱이 필요 없다.

    목록  : /api/webtoon/titlelist/weekday?week={mon..sun}   (연재 중, 요일별)
            /api/webtoon/titlelist/finished?page=N           (완결, 페이지)
    상세  : /api/article/list/info?titleId=N                 (줄거리·장르태그·관심수·연령)
    회차수: /api/article/list?titleId=N&page=1               (`totalCount` 한 번에)

실측(2026-09-04): 연재 752 · 완결 3,116 → **고유 3,868편**. 도전만화·베스트도전은 제외한다
(정식 연재가 아니고 품질 분포가 완전히 다르다 — 코퍼스를 섞으면 랭킹이 무너진다).

    python -m src.crawl_naverwebtoon --out data/webtoons.jsonl [--limit N] [--rps 2.0]
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import urllib.request, urllib.error

BASE = "https://comic.naver.com"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")
HEADERS = {"User-Agent": UA, "Referer": BASE + "/", "Accept": "application/json"}
WEEK = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DOW = {"MONDAY": "월", "TUESDAY": "화", "WEDNESDAY": "수", "THURSDAY": "목",
       "FRIDAY": "금", "SATURDAY": "토", "SUNDAY": "일"}


def _get(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def enumerate_titles(rps: float) -> dict[int, dict]:
    """연재(요일별) + 완결(페이지) 목록. 목록 단계의 필드도 함께 보관한다
    (`starScore`·`adult` 는 목록에만 있다)."""
    out: dict[int, dict] = {}
    for w in WEEK:
        d = _get(f"{BASE}/api/webtoon/titlelist/weekday?week={w}&order=user")
        for t in d.get("titleList", []):
            out.setdefault(int(t["titleId"]), {})["list"] = t
        print(f"  연재 {w}: {len(d.get('titleList', []))}", flush=True)
        time.sleep(1.0 / rps)
    page = 1
    while True:
        d = _get(f"{BASE}/api/webtoon/titlelist/finished?page={page}&order=UPDATE")
        tl = d.get("titleList", [])
        if not tl:
            break
        for t in tl:
            out.setdefault(int(t["titleId"]), {})["list"] = t
        page += 1
        time.sleep(1.0 / rps)
    print(f"  완결 페이지 {page - 1} · 고유 {len(out)}편", flush=True)
    return out


def fetch_detail(tid: int, meta: dict) -> dict:
    info = _get(f"{BASE}/api/article/list/info?titleId={tid}")
    eps = _get(f"{BASE}/api/article/list?titleId={tid}&page=1&sort=DESC")
    lst = meta.get("list", {})
    tags = [t.get("tagName") for t in info.get("curationTagList", []) if t.get("tagName")]
    genres = [t.get("tagName") for t in info.get("curationTagList", [])
              if str(t.get("curationType", "")).startswith("GENRE_")]
    artists = info.get("communityArtists") or []
    age = info.get("age") or {}
    return dict(
        item_id=tid,
        name=info.get("titleName") or lst.get("titleName"),
        synopsis=(info.get("synopsis") or "").strip(),
        genres=genres,
        tags=tags,                                   # 장르 + 큐레이션 태그(명작 등)
        author=lst.get("author") or " / ".join(a.get("name", "") for a in artists),
        artists=[a.get("name") for a in artists if a.get("name")],
        novel_origin=[a.get("name") for a in (lst.get("novelOriginAuthors") or [])],
        favorite_count=int(info.get("favoriteCount") or 0),
        star_score=float(lst.get("starScore") or 0.0),
        episode_count=int(eps.get("totalCount") or 0),
        age_rating=age.get("description") or "",
        age_type=age.get("type") or "",
        adult=bool(lst.get("adult", False)),
        finished=bool(info.get("finished", lst.get("finish", False))),
        rest=bool(info.get("rest", False)),
        weekdays=[DOW.get(d, d) for d in (info.get("publishDayOfWeekList") or [])],
        publish_desc=info.get("publishDescription") or "",
        level=info.get("webtoonLevelCode") or "",
        url=f"{BASE}/webtoon/list?titleId={tid}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/webtoons.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rps", type=float, default=2.0)
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            line = line.strip()
            if line:
                try: done.add(int(json.loads(line)["item_id"]))
                except Exception: pass
        print(f"이어받기: 이미 {len(done)}편", flush=True)

    print("목록 열거", flush=True)
    titles = enumerate_titles(a.rps)
    todo = [t for t in titles if t not in done]
    if a.limit: todo = todo[:a.limit]
    print(f"받을 작품 {len(todo)}편 (rps {a.rps})", flush=True)

    delay = 1.0 / a.rps; ok = fail = 0; skipped: list[int] = []
    with out.open("a", encoding="utf-8") as f:
        for i, tid in enumerate(todo, 1):
            try:
                rec = fetch_detail(tid, titles[tid])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                ok += 1; delay = max(1.0 / a.rps, delay * 0.9)          # 성공하면 조금 빨리
            except urllib.error.HTTPError as e:
                fail += 1
                if e.code in (401, 403, 404):
                    # **영구 조건이다** — 성인 웹툰은 로그인이 있어야 상세가 열린다.
                    # 레이트 제한이 아니므로 물러서지 않는다. (초기 판에서 이걸 백오프로
                    # 잘못 다뤄, 성인물이 몰려 있는 구간에서 지연이 5초까지 올라 처리량이 무너졌다.)
                    skipped.append(tid)
                else:
                    delay = min(5.0, delay * 2.0)                        # 진짜로 막힌 경우만
                    print(f"  HTTP {e.code} titleId={tid} → delay {delay:.2f}s", flush=True)
            except Exception as e:
                fail += 1
                print(f"  실패 titleId={tid}: {type(e).__name__}", flush=True)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} · 성공 {ok} 실패 {fail}", flush=True)
            time.sleep(delay)
    print(f"완료: 성공 {ok} · 실패 {fail} (그 중 접근불가 {len(skipped)}) → {out}", flush=True)
    if skipped:
        out.with_suffix(".skipped.json").write_text(json.dumps(sorted(skipped)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
