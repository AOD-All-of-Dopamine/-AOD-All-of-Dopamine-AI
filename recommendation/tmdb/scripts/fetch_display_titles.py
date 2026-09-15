"""카드 표시명 수집 — 한글도 라틴도 없는 제목(데바나가리·텔루구·한자 등)만 영어 제목을 받는다.

왜: 수집이 `language=ko-KR` 이라 한국어 번역이 없으면 TMDB 가 원제로 떨어진다.
V-1 이 원어 줄거리 노출은 정책으로 허용했지만, 한국어 사용자가 **제목조차 못 읽는 카드**
(`तेरी मेरी कहानी`)는 그 의도가 아니다. 추천(랭킹·임베딩)은 건드리지 않는다 — 표시만 바꾼다.

    python scripts/fetch_display_titles.py          → data/display_titles.jsonl (재개 가능)
    python scripts/fetch_display_titles.py build    → artifacts/tmdb_v1/display_names.parquet
"""
import json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_directors import get, WORKERS, DATA, ROOT  # noqa: E402  같은 재시도 규약·키 로딩

READABLE = re.compile(r"[가-힣A-Za-z]")
DST = DATA / "display_titles.jsonl"
ART = ROOT / "artifacts" / "tmdb_v1"


def targets() -> pd.DataFrame:
    ds = pd.read_parquet(ART / "dataset.parquet")
    return ds[~ds["name"].fillna("").map(lambda s: bool(READABLE.search(s)))][["item_id", "tmdb_id", "media", "name"]]


def fetch(row):
    item_id, tmdb_id, media = row
    d = get(f"/{media}/{int(tmdb_id)}", language="en-US")
    if d is None:
        return None
    return {"item_id": item_id, "en": d.get("title") if media == "movie" else d.get("name"),
            "original": d.get("original_title") if media == "movie" else d.get("original_name")}


def collect():
    t = targets()
    have = set()
    if DST.exists():
        for l in DST.open():
            try: have.add(json.loads(l)["item_id"])
            except json.JSONDecodeError: pass
    todo = [r for r in t[["item_id", "tmdb_id", "media"]].itertuples(index=False, name=None) if r[0] not in have]
    print(f"대상 {len(t):,} · 완료 {len(have):,} · 남음 {len(todo):,} · 워커 {WORKERS}", flush=True)
    done, t0, lock = 0, time.time(), threading.Lock()
    with DST.open("a") as f, ThreadPoolExecutor(WORKERS) as ex:
        for r in ex.map(fetch, todo):
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            with lock:
                done += 1
                if done % 200 == 0:
                    print(f"  {done:,}/{len(todo):,} {done / (time.time() - t0):.1f}건/s", flush=True)
    print(f"완료 → {DST}", flush=True)


def build():
    t = targets().set_index("item_id")
    got = {}
    for l in DST.open():
        r = json.loads(l); en = (r.get("en") or "").strip()
        if en and READABLE.search(en): got[r["item_id"]] = en
    out = pd.DataFrame({"item_id": list(got), "display_name": list(got.values())})
    out.to_parquet(ART / "display_names.parquet", index=False)
    print(f"표시명 {len(out):,}/{len(t):,} (영어 제목이 없거나 역시 못 읽는 것 {len(t) - len(out):,}) → {ART / 'display_names.parquet'}")


if __name__ == "__main__":
    build() if sys.argv[1:] == ["build"] else collect()
