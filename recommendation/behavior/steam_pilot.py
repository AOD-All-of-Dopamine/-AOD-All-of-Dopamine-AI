"""P1 — Steam 공동보유 파일럿. 본 수집이 몇 시간짜리인지 재기 위한 것.

    게임 N개 → 리뷰 API(키 불필요) → 리뷰어 steamid + 플레이타임
             → GetOwnedGames(우리 키) → 유저별 보유 게임 + 게임별 플레이타임

재는 것: 게임당 고유 steamid 수 · **공개 프로필 비율** · 유저당 보유 수 분포 · 처리량.
게임 선택은 기계적: Steam p1 프로필의 시드(저리뷰·롱테일·대작이 섞여 있다)에서 앞 N개.

    python behavior/steam_pilot.py --games 20 --reviews 100 --probe 300
결과: behavior/data/pilot_reviews.jsonl · pilot_owned.jsonl · pilot_summary.json
"""
from __future__ import annotations
import argparse, json, time, sys, statistics as st
from pathlib import Path
import urllib.request, urllib.parse
import pandas as pd

HERE = Path(__file__).resolve().parent; R = HERE.parent; OUT = HERE / "data"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"


def _env(path, keys):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in keys: return v.strip().strip('"').strip("'")


def _json(url, timeout=20):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
        return json.load(r)


def reviews(appid: int, n: int):
    """최근 리뷰 n개. cursor 로 100개씩. 리뷰어 steamid·플레이타임을 남긴다."""
    got, cursor = [], "*"
    while len(got) < n:
        q = urllib.parse.urlencode(dict(json=1, num_per_page=100, filter="recent", language="all",
                                        purchase_type="all", cursor=cursor))
        d = _json(f"https://store.steampowered.com/appreviews/{appid}?{q}")
        rv = d.get("reviews", [])
        if not rv: break
        for r in rv:
            a = r.get("author", {})
            got.append(dict(appid=appid, steamid=a.get("steamid"), playtime=a.get("playtime_forever"),
                            owned=a.get("num_games_owned"), voted_up=r.get("voted_up")))
        cursor = d.get("cursor") or ""
        if not cursor: break
        time.sleep(0.5)
    return got[:n]


def owned(key: str, steamid: str):
    d = _json(f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/?key={key}&steamid={steamid}"
              f"&include_played_free_games=1")["response"]
    return d.get("game_count", 0), [(g["appid"], g.get("playtime_forever", 0)) for g in d.get("games", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20); ap.add_argument("--reviews", type=int, default=100)
    ap.add_argument("--probe", type=int, default=300); ap.add_argument("--rps", type=float, default=2.0)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    key = _env(R / "steam" / ".env", {"STEAM_API_KEY", "STEAM_KEY", "API_KEY"})
    if not key: sys.exit("steam/.env 에 키가 없다")

    prof = pd.read_parquet(R / "steam/artifacts/p1/profiles.parquet")
    games = list(dict.fromkeys(int(x) for s in prof["liked_appids"] for x in s))[:a.games]
    names = pd.read_parquet(R / "steam/artifacts/tags_full/dataset.parquet",
                            columns=["steam_appid", "name"]).set_index("steam_appid")["name"]
    print(f"게임 {len(games)}개 · 리뷰 {a.reviews}/게임 · 프로브 {a.probe}명", flush=True)

    t0 = time.time(); rows = []
    with (OUT / "pilot_reviews.jsonl").open("w", encoding="utf-8") as f:
        for i, g in enumerate(games, 1):
            rv = reviews(g, a.reviews); rows += rv
            for r in rv: f.write(json.dumps(r) + "\n")
            print(f"  [{i:>2}/{len(games)}] {str(names.get(g, g))[:28]:<28} 리뷰 {len(rv):>3} · 고유 steamid {len({r['steamid'] for r in rv})}", flush=True)
    t_rev = time.time() - t0
    ids = list(dict.fromkeys(r["steamid"] for r in rows if r.get("steamid")))
    print(f"리뷰 단계: {len(rows)}건 · 고유 steamid {len(ids)} · {t_rev:.0f}s", flush=True)

    t1 = time.time(); pub = 0; counts = []; probe = ids[:a.probe]
    with (OUT / "pilot_owned.jsonl").open("w", encoding="utf-8") as f:
        for j, sid in enumerate(probe, 1):
            try:
                n, gl = owned(key, sid)
            except Exception as e:
                n, gl = -1, []; print(f"  {sid} {type(e).__name__}", file=sys.stderr)
            if n > 0: pub += 1; counts.append(n)
            f.write(json.dumps(dict(steamid=sid, game_count=n, games=gl)) + "\n")
            if j % 50 == 0: print(f"  프로브 {j}/{len(probe)} · 공개 {pub}", flush=True)
            time.sleep(1.0 / a.rps)
    t_own = time.time() - t1
    played = [sum(1 for _, p in json.loads(l)["games"] if p >= 60) for l in (OUT / "pilot_owned.jsonl").open() if json.loads(l)["game_count"] > 0]
    summ = dict(games=len(games), reviews=len(rows), unique_steamids=len(ids), probed=len(probe),
                public=pub, public_rate=round(pub / max(len(probe), 1), 3),
                owned_median=st.median(counts) if counts else None, owned_p90=(sorted(counts)[int(len(counts) * .9)] if counts else None),
                played1h_median=st.median(played) if played else None,
                sec_per_game_reviews=round(t_rev / len(games), 1), sec_per_profile=round(t_own / max(len(probe), 1), 2))
    json.dump(summ, (OUT / "pilot_summary.json").open("w"), indent=1)
    print("\n=== 파일럿 요약 ===")
    for k, v in summ.items(): print(f"  {k:<22} {v}")
    # 본 수집 추정: 게임 G개 × 리뷰 R개
    for G, Rn in ((1000, 100), (3000, 100), (3000, 300)):
        uid = G * Rn * (len(ids) / max(len(rows), 1))     # 중복 제거 비율 적용
        h = (G * summ["sec_per_game_reviews"] * (Rn / a.reviews) + uid * summ["sec_per_profile"]) / 3600
        print(f"  본 수집 추정  게임 {G:,} × 리뷰 {Rn}: steamid ≈ {uid:,.0f} · 약 {h:.1f}시간 · 공개 프로필 ≈ {uid * summ['public_rate']:,.0f}명")


if __name__ == "__main__":
    main()
