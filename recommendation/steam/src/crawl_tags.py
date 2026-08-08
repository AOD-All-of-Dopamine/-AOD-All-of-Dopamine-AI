# src/crawl_tags.py
"""Steam 사용자 태그 수집 — 표현 문제의 핵심 신호.

**왜 필요한가**

판정 580쌍 중 실패(0~1점) 235건의 원인을 세면 표현 실패가 77% 다:

    GENRE_ONLY     89 (38%)   장르만 맞고 경험이 다르다
    IRRELEVANT     71 (30%)   완전히 틀렸다
    KEYWORD_MATCH  21 ( 9%)   'Salt' / 'Salem' / 'Metro' 처럼 단어만 겹친다

원인은 임베딩이 보는 것이 마케팅 문구 한 문단 + 거친 장르뿐이라는 데 있다.
`genres` 는 '액션'(코퍼스의 42%) '인디'(72%) 같은 값이라 변별력이 거의 없다:

    Salt and Sanctuary  현재 genres → 액션, 인디, RPG
                        사용자 태그 → Souls-like, Metroidvania, Dark Fantasy, 2D,
                                      Difficult, Platformer, Atmospheric, ...
    Hollow Knight       사용자 태그 → Metroidvania, Platformer, Souls-like, Difficult, ...

태그로는 두 게임이 Metroidvania/Souls-like/Difficult 3개를 공유한다. 장르로는 액션·인디뿐이다.

**왜 공식 Web API 인가**

같은 태그를 세 경로로 얻을 수 있는데 이게 가장 낫다:

    IStoreBrowseService/GetItems  공식 · 배치 200개 · 0.3초 · **태그별 투표 수(weight) 포함**
    상점 페이지 InitAppTagModal   공식이지만 HTML 파싱, 연령 게이트 쿠키 필요, 1건씩
    SteamSpy API                  비공식 서드파티, 1건씩

weight 가 있어야 "이 게임은 Souls-like 표가 1,756 인데 Co-op 표는 176" 을 구분할 수 있다.
전부 같은 무게로 넣으면 꼬리 태그가 노이즈가 된다.

    python -m src.crawl_tags --out data/steam_tags.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, artifact_dir

ITEMS_URL = "https://api.steampowered.com/IStoreBrowseService/GetItems/v1/"
TAGLIST_URL = "https://api.steampowered.com/IStoreService/GetTagList/v1/"
UA = {"User-Agent": "Mozilla/5.0"}

BATCH = 200          # 500 은 HTTP 400. 200 이 실측 상한이다.
TAG_COUNT = 20       # 상점이 보여주는 것과 같은 깊이
SLEEP = 0.2
MAX_RETRY = 6        # 429 는 건너뛰면 안 된다 — 그 배치의 200개가 통째로 구멍이 된다


def load_key(env_path: Path | None = None) -> str:
    env = env_path or PROJECT_ROOT / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("STEAM_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{env} 에 STEAM_API_KEY 가 없습니다.")


def _get(url: str, timeout: int = 40) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def fetch_tag_dictionary(key: str, language: str = "english") -> dict[int, str]:
    """태그 id → 이름. 446개짜리 사전이라 한 번만 받으면 된다."""
    r = _get(f"{TAGLIST_URL}?key={key}&language={urllib.parse.quote(language)}")
    return {int(t["tagid"]): t["name"] for t in r["response"]["tags"]}


def fetch_batch(key: str, appids: list[int], language: str = "english") -> list[dict]:
    req = {
        "ids": [{"appid": int(a)} for a in appids],
        "context": {"language": language, "country_code": "US"},
        "data_request": {"include_tag_count": TAG_COUNT},
    }
    url = f"{ITEMS_URL}?key={key}&input_json={urllib.parse.quote(json.dumps(req))}"
    return _get(url)["response"].get("store_items", [])


def already_done(path: Path) -> set[int]:
    """이어받기. 배치 단위라 중간에 끊겨도 받은 것은 남는다."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(int(json.loads(line)["appid"]))
                except Exception:
                    continue
    return done


def fetch_with_backoff(key: str, chunk: list[int], sleep: float) -> tuple[list[dict], float, int]:
    """429 를 만나면 대기를 두 배로 늘려 재시도한다.

    처음 구현은 실패한 배치를 그냥 건너뛰었는데, 실측에서 29개 배치가 429 로 날아가
    12,852개 중 5,800개가 조용히 빠졌다. 배치 하나가 200개짜리라 구멍이 크다.
    """
    delay = sleep
    for attempt in range(MAX_RETRY):
        try:
            return fetch_batch(key, chunk), delay, attempt
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == MAX_RETRY - 1:
                raise
            delay = min(delay * 2 if delay else 1.0, 30.0)
            time.sleep(delay)
        except Exception:
            if attempt == MAX_RETRY - 1:
                raise
            time.sleep(min((attempt + 1) * 2.0, 30.0))
    raise RuntimeError("unreachable")


def crawl(appids: list[int], key: str, out: Path, tag_names: dict[int, str],
          batch: int = BATCH, sleep: float = SLEEP) -> dict:
    done = already_done(out)
    todo = [a for a in appids if a not in done]
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"requested": len(appids), "skipped": len(appids) - len(todo),
             "written": 0, "with_tags": 0, "errors": 0, "retries": 0, "failed_appids": []}
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                items, sleep, retries = fetch_with_backoff(key, chunk, sleep)
                stats["retries"] += retries
            except Exception as e:
                stats["errors"] += 1
                stats["failed_appids"].extend(chunk)
                print(f"  배치 {i // batch} 포기: {type(e).__name__} {e}")
                continue
            got = {int(it["appid"]) for it in items if "appid" in it}
            for it in items:
                if "appid" not in it:
                    continue
                tags = it.get("tags") or []
                row = {
                    "appid": int(it["appid"]),
                    "name": it.get("name", ""),
                    "tags": [{"name": tag_names.get(int(t["tagid"]), str(t["tagid"])),
                              "weight": int(t.get("weight", 0))} for t in tags],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats["with_tags"] += bool(tags)
            # 성공한 배치는 조금씩 속도를 되돌린다 (429 후 영원히 느려지지 않게)
            sleep = max(SLEEP, sleep * 0.9)
            # 응답에 안 온 appid 도 기록해야 이어받기가 무한 재시도하지 않는다
            for a in chunk:
                if a not in got:
                    f.write(json.dumps({"appid": int(a), "name": "", "tags": []}) + "\n")
                    stats["written"] += 1
            f.flush()
            if (i // batch) % 50 == 0:
                el = time.time() - t0
                pct = (i + len(chunk)) / max(len(todo), 1)
                print(f"  {i + len(chunk):>7,}/{len(todo):,} ({pct:5.1%})  "
                      f"{el:5.0f}초 경과, 남은 예상 {el / max(pct, 1e-9) - el:5.0f}초")
            time.sleep(sleep)
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["failed_count"] = len(stats.pop("failed_appids"))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/steam_tags.jsonl")
    ap.add_argument("--artifacts", default="artifacts/full_v1")
    ap.add_argument("--min-reviews", type=int, default=0,
                    help="0 이면 코퍼스 전체. 배치가 빨라서 전체를 받아도 몇 분이다.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ds = pd.read_parquet(artifact_dir(args.artifacts) / "dataset.parquet")
    if args.min_reviews:
        rev = ds["recommendations_total"].astype("float").fillna(0)
        ds = ds[rev >= args.min_reviews]
    appids = [int(a) for a in ds["steam_appid"]]
    if args.limit:
        appids = appids[:args.limit]

    key = load_key()
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    print(f"태그 사전 수집 중...")
    tag_names = fetch_tag_dictionary(key)
    print(f"  태그 {len(tag_names)}개")
    print(f"대상 {len(appids):,}개 → {out}")
    stats = crawl(appids, key, out, tag_names)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
