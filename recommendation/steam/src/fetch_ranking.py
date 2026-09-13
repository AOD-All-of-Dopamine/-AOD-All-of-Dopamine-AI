import re
from pathlib import Path

import pandas as pd
import requests

STEAM_SEARCH_URL = (
    "https://store.steampowered.com/search/results/"
    "?filter=topsellers&category1=998&l=korean&cc=KR&start=0&count=100"
)


def fetch_top_sellers() -> pd.DataFrame:
    s = requests.Session()
    s.get("https://store.steampowered.com/", headers={"User-Agent": "Mozilla/5.0"})
    r = s.get(STEAM_SEARCH_URL, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    appids = re.findall(r'data-ds-appid="(\d+)"', r.text)
    names = re.findall(r'<span class="title">([^<]+)</span>', r.text)
    records = []
    for i, (aid, name) in enumerate(zip(appids, names)):
        records.append({
            "steam_appid": int(aid),
            "steam_rank": i + 1,
            "steam_rank_name": name,
        })
    return pd.DataFrame(records)


def main():
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    df = fetch_top_sellers()
    df.to_parquet(out / "steam_ranking.parquet", index=False)
    print(f"wrote {out / 'steam_ranking.parquet'} rows={len(df)}")
    print(f"  Top-5: {df['steam_rank_name'].head(5).tolist()}")


if __name__ == "__main__":
    main()
