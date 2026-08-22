"""Steam 리뷰 요약 크롤 — D-39.

**왜.** dataset 의 `recommendations_total` 은 못 쓴다. 173,691개 중 151,799개(87.4%)가
결측이고, 값이 있는 것도 틀렸다 — Warframe 2,881(실제 674,726) · Neverwinter 295(실제
39,644). Steam appdetails 의 `recommendations.total` 은 Valve 가 일관되게 채우지 않는다.

`store.steampowered.com/appreviews/{appid}` 의 `query_summary` 는 실측을 준다:
`total_reviews` · `total_positive` · `total_negative` · `review_score_desc`.
**긍정 비율은 인기도와 별개의 축이라** "무명이지만 좋은 것"을 "무명이고 나쁜 것"과
가를 수 있다 — 롱테일 취향 프로필에 필요한 신호다.

미출시작(coming_soon)은 건너뛴다. 어차피 후처리가 걸러낸다.
체크포인트를 남겨 중단 후 이어받는다.
"""
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import artifact_dir  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "reviews"
URL = ("https://store.steampowered.com/appreviews/{a}?json=1&num_per_page=0"
       "&language=all&purchase_type=all&filter=summary")


def fetch(appid: int, tries: int = 3) -> dict | None:
    for i in range(tries):
        try:
            with urllib.request.urlopen(URL.format(a=appid), timeout=20) as r:
                q = json.load(r).get("query_summary") or {}
            return dict(steam_appid=appid,
                        total_reviews=int(q.get("total_reviews") or 0),
                        total_positive=int(q.get("total_positive") or 0),
                        total_negative=int(q.get("total_negative") or 0),
                        review_score=int(q.get("review_score") or 0))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (i + 1)); continue
            return dict(steam_appid=appid, total_reviews=-1, total_positive=0,
                        total_negative=0, review_score=0)
        except Exception:
            time.sleep(1 + i)
    return dict(steam_appid=appid, total_reviews=-1, total_positive=0,
                total_negative=0, review_score=0)


def main(workers: int = 12, chunk: int = 4000) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = pd.read_parquet(artifact_dir() / "dataset.parquet")
    todo = ds.loc[~ds["coming_soon"].astype(bool), "steam_appid"].astype(int).tolist()
    # 실패(-1)는 done 으로 치지 않는다 — 12워커 동시성에서 오는 일시 스로틀링이라
    # 재시도하면 대부분 복구된다. 실측: 실패 표본 5/5 재시도 성공.
    done: set[int] = set()
    for f in sorted(OUT.glob("part-*.parquet")):
        d = pd.read_parquet(f)
        done |= set(d.loc[d["total_reviews"] >= 0, "steam_appid"].astype(int))
    todo = [a for a in todo if a not in done]
    print(f"대상 {len(todo):,} · 이미 받음 {len(done):,}", flush=True)
    part = len(list(OUT.glob("part-*.parquet")))
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            rows = [r for r in ex.map(fetch, batch) if r]
        pd.DataFrame(rows).to_parquet(OUT / f"part-{part:04d}.parquet", index=False)
        part += 1
        el = time.time() - t0
        print(f"  {i + len(batch):,}/{len(todo):,}  {el:.0f}s  "
              f"({len(batch)/el:.1f}/s)  실패 {sum(1 for r in rows if r['total_reviews'] < 0)}",
              flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
