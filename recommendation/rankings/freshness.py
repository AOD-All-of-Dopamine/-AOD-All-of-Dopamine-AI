"""P0 진단 ① 코퍼스 신선도 — 오늘 인기 top-N 중 우리 코퍼스에 **없는** 비율.
추천 품질이 아니라 **크롤 주기** 지표다. 판정 문턱에 넣지 않는다."""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; R = HERE.parent
def last(plat):
    rows = [json.loads(l) for l in (HERE / "data" / f"{plat}.jsonl").open(encoding="utf-8") if l.strip()]
    day = max(r["date"] for r in rows)
    return day, {r["source"]: r["items"] for r in rows if r["date"] == day}
CORP = {
  "steam":    set(pd.read_parquet(R/"steam/artifacts/tags_full/dataset.parquet", columns=["steam_appid"])["steam_appid"].astype(int)),
  "tmdb":     set(pd.read_parquet(R/"tmdb/artifacts/tmdb_v1/dataset.parquet", columns=["item_id"])["item_id"]),
  "webtoon":  set(pd.read_parquet(R/"webtoon/artifacts/wt_v1/dataset.parquet", columns=["item_id"])["item_id"].astype(int)),
  "webnovel": set(pd.read_parquet("/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4/dataset.parquet", columns=["item_id"])["item_id"].astype(int)),
}
CRAWL = {"steam": "2026-08", "tmdb": "2026-08", "webtoon": "2026-09-04", "webnovel": "2026-07"}
print(f"{'플랫폼':<9}{'소스':<20}{'top-N':>6}{'코퍼스에 없음':>10}   크롤 시점")
for plat in CORP:
    day, srcs = last(plat)
    for src, items in srcs.items():
        miss = sum(1 for x in items if x["id"] not in CORP[plat])
        flag = "  ← 주의" if items and miss / len(items) >= 0.2 else ""
        print(f"{plat:<9}{src:<20}{len(items):>6}{miss:>6} ({miss/max(len(items),1)*100:4.0f}%)   {CRAWL[plat]}{flag}")
