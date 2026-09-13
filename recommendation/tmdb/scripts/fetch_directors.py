"""감독 수집 — **기존 enriched 파일은 건드리지 않는다.**

왜 별도 파일인가: `movie_enriched.jsonl` 은 이미 임베딩까지 간 산출물의 입력이다.
거기에 열을 덧붙이려고 다시 돌리면 줄거리·키워드까지 재수집되고, 그 사이 TMDB 가
바꾼 값이 섞여 들어와 **기존 코퍼스와 새 코퍼스를 비교할 수 없게 된다.**
감독만 따로 받아 `item_id` 로 붙인다.

영화와 드라마는 감독의 정의가 다르다:

    movie  /movie/{id}/credits  →  crew 에서 job == "Director"
    tv     /tv/{id}             →  created_by (크리에이터)

드라마 crew 에는 Director 가 **없다** (실측: 브레이킹 배드 crew job 목록에 Director 부재).
회차마다 감독이 달라서 시즌 단위로는 의미가 없기 때문이다. 드라마에서 작품의
색을 정하는 사람은 쇼러너 = `created_by` 다 (브레이킹 배드 → 빈스 길리건).

**이름이 아니라 person id 로 맞춘다.** 이름은 언어 설정에 따라 '크리스토퍼 놀란' /
'Christopher Nolan' 으로 갈리고 동명이인도 있다. id 는 하나다. 이름은 사람이 읽기
위해서만 같이 저장한다.

    python scripts/fetch_directors.py movie
    python scripts/fetch_directors.py tv

중단되면 다시 돌리면 된다 — 이미 받은 id 는 건너뛴다.
"""
import json, os, sys, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
KEY = os.environ.get("TMDB_API_KEY") or os.environ["TMDB_KEY"]

# 실측(2026-09-13): 순차 1건 5.9s · 32워커 3.8건/s · 96워커 6.1건/s(중 23건 타임아웃).
# **병목이 TMDB 레이트리밋이 아니라 이쪽 네트워크 지연이다** — 워커를 늘려도 거의 안 오른다.
# 8월 enrich 가 16건/s 였던 것과 다르다. 24 는 타임아웃이 안 나는 선에서 고른 값.
WORKERS = 24
_lock = threading.Lock()
_done = 0
_t0 = time.time()


def get(path, **q):
    """enrich_tmdb.py 의 재시도 규약을 그대로 쓴다 (429 는 물러서고, 404 는 없는 것)."""
    q["api_key"] = KEY
    url = f"https://api.themoviedb.org/3{path}?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            if attempt == 4:
                return None
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            if attempt == 4:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch(args):
    media, i = args
    if media == "movie":
        d = get(f"/movie/{i}/credits", language="ko-KR")
        people = [p for p in (d or {}).get("crew", []) if p.get("job") == "Director"]
    else:
        d = get(f"/tv/{i}", language="ko-KR")
        people = list((d or {}).get("created_by") or [])
    if d is None:
        return None
    seen, out = set(), []
    for p in people:                       # 공동 감독은 순서를 지켜 중복만 제거
        pid = p.get("id")
        if pid is None or pid in seen:
            continue
        seen.add(pid)
        out.append({"id": pid, "name": p.get("name")})
    return {"id": i, "media": media, "directors": out}


def main():
    media = sys.argv[1] if len(sys.argv) > 1 else "movie"
    assert media in ("movie", "tv")
    src = DATA / f"{media}_enriched.jsonl"
    dst = DATA / f"{media}_directors.jsonl"

    ids = [json.loads(l)["id"] for l in src.open()]
    have = set()
    if dst.exists():
        for l in dst.open():
            try:
                have.add(json.loads(l)["id"])
            except json.JSONDecodeError:
                pass
    todo = [i for i in ids if i not in have]
    print(f"[{media}] 목록 {len(ids):,} · 완료 {len(have):,} · 남음 {len(todo):,} · 워커 {WORKERS}",
          flush=True)

    global _done
    with dst.open("a") as f, ThreadPoolExecutor(WORKERS) as ex:
        for r in ex.map(fetch, ((media, i) for i in todo)):
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()   # 4시간짜리라 진행을 눈으로 봐야 한다
            with _lock:
                _done += 1
                if _done % 200 == 0:
                    rate = _done / (time.time() - _t0)
                    left = (len(todo) - _done) / max(rate, 1e-9) / 60
                    print(f"[{media}] {_done:,}/{len(todo):,}  {rate:.1f}건/s  "
                          f"남은 시간 {left:.0f}분", flush=True)
                    f.flush()
    print(f"[{media}] 완료 → {dst}", flush=True)


if __name__ == "__main__":
    main()
