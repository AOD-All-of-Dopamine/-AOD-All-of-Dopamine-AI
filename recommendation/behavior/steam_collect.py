"""P2 — Steam 공동보유 본 수집. 이어받기 가능, 일일 한도 아래로.

  게임 선택 (기계적): 리뷰 ≥100 인 코퍼스 게임을 리뷰 수 4분위로 나눠 각 250편 무작위(seed 2609).
                     인기 편중을 피하려는 것 — 롱테일 취향의 행동도 필요하다 (PLAN §3-1).
  1단계 리뷰:  게임당 최근 100건 → steamid · 플레이타임          → data/reviews.jsonl
  2단계 보유:  고유 steamid 마다 GetOwnedGames                   → data/owned.jsonl  (비공개는 game_count 0 으로 기록)
둘 다 append 이고 이미 처리한 키는 건너뛴다. 죽어도 다시 돌리면 이어진다.

    python behavior/steam_collect.py --games 1000 --reviews 100 --rps 1.2
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from steam_pilot import reviews, owned, _env, R, OUT


def pick_games(n: int, seed: int = 2609) -> list[int]:
    d = pd.read_parquet(R / "steam/artifacts/tags_full/dataset.parquet", columns=["steam_appid", "recommendations_total"])
    d = d[d["recommendations_total"].fillna(0) >= 100].copy()
    d["q"] = pd.qcut(d["recommendations_total"].rank(method="first"), 4, labels=False)
    rng = random.Random(seed); out = []
    for q in range(4):
        pool = d[d["q"] == q]["steam_appid"].astype(int).tolist(); rng.shuffle(pool); out += pool[:n // 4]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1000); ap.add_argument("--reviews", type=int, default=100)
    ap.add_argument("--rps", type=float, default=1.2); ap.add_argument("--daily-cap", type=int, default=80000)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    key = _env(R / "steam" / ".env", {"STEAM_API_KEY", "STEAM_KEY", "API_KEY"})
    games = pick_games(a.games)
    json.dump(games, (OUT / "games.json").open("w"))

    RV, OW = OUT / "reviews.jsonl", OUT / "owned.jsonl"
    done_g = {json.loads(l)["appid"] for l in RV.open()} if RV.exists() else set()
    print(f"게임 {len(games)} (이미 {len(done_g)}) · 리뷰 {a.reviews}/게임", flush=True)
    with RV.open("a", encoding="utf-8") as f:
        for i, g in enumerate(games, 1):
            if g in done_g: continue
            try:
                rv = reviews(g, a.reviews)
            except Exception as e:
                print(f"  리뷰 실패 {g}: {type(e).__name__}", file=sys.stderr, flush=True); time.sleep(5); continue
            for r in rv: f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0: print(f"  리뷰 {i}/{len(games)}", flush=True)
            time.sleep(0.3)

    ids = list(dict.fromkeys(json.loads(l)["steamid"] for l in RV.open() if json.loads(l).get("steamid")))
    done_u = {json.loads(l)["steamid"] for l in OW.open()} if OW.exists() else set()
    todo = [s for s in ids if s not in done_u]
    print(f"고유 steamid {len(ids):,} · 남은 프로브 {len(todo):,}", flush=True)
    calls = 0; day = time.time(); pub = 0
    with OW.open("a", encoding="utf-8") as f:
        for j, sid in enumerate(todo, 1):
            if calls >= a.daily_cap:
                wait = 86400 - (time.time() - day)
                print(f"  일일 한도 {a.daily_cap} 도달 → {wait/3600:.1f}h 대기", flush=True)
                time.sleep(max(wait, 0)); calls = 0; day = time.time()
            try:
                n, gl = owned(key, sid); calls += 1
            except Exception as e:
                print(f"  프로브 실패 {sid}: {type(e).__name__}", file=sys.stderr, flush=True); time.sleep(10); continue
            if n > 0: pub += 1
            f.write(json.dumps(dict(steamid=sid, game_count=n, games=gl)) + "\n")
            if j % 500 == 0: f.flush(); print(f"  프로브 {j:,}/{len(todo):,} · 공개 {pub:,}", flush=True)
            time.sleep(1.0 / a.rps)
    print(f"완료 · 공개 프로필 {pub:,}", flush=True)


if __name__ == "__main__":
    main()
