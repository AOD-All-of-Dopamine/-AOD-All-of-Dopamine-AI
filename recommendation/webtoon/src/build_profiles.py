"""웹툰 평가 프로필 — **전부 기계 생성**한다.

세 플랫폼은 "직접 선정 + 규칙 생성"을 섞었는데, 웹툰은 **규칙만** 쓴다. 이유가 있다:
- X-24: 내가 지어낸 조합이 val 집합의 가장 큰 결함이었다. 공개 근거가 없다.
- X-25: **응집된(중복) 시드일수록 P@50 이 높게 나온다**(중복 3개 0.906 vs 다양 3개 0.834).
  내가 "같은 결"이라고 골라 묶으면 그 자체로 점수가 부풀려진다.
그래서 시드 선택에 내 판단을 넣지 않는다. 대신 **축을 설계**한다.

축 (전부 고정 RNG, 재현 가능):
  rule_{장르}_{대역}  장르 × 관심 수 대역(high/mid/low) 안에서 무작위 3편
                      → 인기 구간별로 품질이 다른지 본다 (Steam D-38 이 겪은 문제)
  coh_{장르}          한 장르 안에서 무작위 5편           (응집 — 시드가 서로 가깝다)
  mix_{a}_{b}         서로 다른 두 장르에서 각각 무작위 2~3편 (혼합 — 시드가 멀다)
  k{n}_{장르}         시드 개수 축 1 / 3 / 5 / 10          (X-25 가 −0.103 을 본 축)

`coh` vs `mix` 는 X-25 의 다양성 발견이 웹툰에서도 재현되는지 보는 대조쌍이다.
**판정에 쓰기 전에 사전등록한다.** 여기서는 프로필만 만든다.

    PYTHONPATH=. python -m src.build_profiles
"""
from __future__ import annotations
import json, random
import pandas as pd
from src.config import artifact_dir, PROJECT_ROOT

SEED = 2609
BANDS = [("high", 0.90, 1.01), ("mid", 0.40, 0.90), ("low", 0.0, 0.40)]


def main():
    d = pd.read_parquet(artifact_dir(None) / "dataset.parquet").reset_index(drop=True)
    d["g0"] = d["genres"].apply(lambda x: (list(x)[0] if x is not None and len(x) else "미상"))
    d["pct"] = d["favorite_count"].rank(pct=True)
    genres = [g for g, n in d["g0"].value_counts().items() if n >= 60 and g != "미상"]
    rng = random.Random(SEED)
    P = []

    def pick(pool, n):
        pool = list(pool)
        return [int(x) for x in rng.sample(pool, min(n, len(pool)))]

    # 1) 장르 × 인기 대역 (3시드)
    for g in genres:
        for band, lo, hi in BANDS:
            pool = d[(d["g0"] == g) & (d["pct"] >= lo) & (d["pct"] < hi)]["item_id"]
            if len(pool) < 3: continue
            P.append(dict(pid=f"rule_{g}_{band}", axis="rule", genre=g, band=band,
                          seeds=pick(pool, 3)))
    # 2) 응집 (한 장르 5시드)
    for g in genres:
        pool = d[d["g0"] == g]["item_id"]
        if len(pool) < 5: continue
        P.append(dict(pid=f"coh_{g}", axis="coh", genre=g, seeds=pick(pool, 5)))
    # 3) 혼합 (두 장르에서 각각 2~3시드)
    for i in range(len(genres)):
        for j in range(i + 1, len(genres)):
            a, b = genres[i], genres[j]
            pa = d[d["g0"] == a]["item_id"]; pb = d[d["g0"] == b]["item_id"]
            if len(pa) < 3 or len(pb) < 2: continue
            P.append(dict(pid=f"mix_{a}_{b}", axis="mix", genre=f"{a}+{b}",
                          seeds=pick(pa, 3) + pick(pb, 2)))
    # 혼합은 조합이 많으니 고정 RNG 로 12개만
    mix = [p for p in P if p["axis"] == "mix"]
    keep = set(id(x) for x in rng.sample(mix, min(12, len(mix))))
    P = [p for p in P if p["axis"] != "mix" or id(p) in keep]
    # 4) 시드 개수 축 — 같은 장르에서 1 / 3 / 5 / 10
    for g in genres[:4]:
        pool = list(d[d["g0"] == g]["item_id"])
        base = pick(pool, 10)
        for n in (1, 3, 5, 10):
            P.append(dict(pid=f"k{n}_{g}", axis="count", genre=g, k=n, seeds=base[:n]))

    names = dict(zip(d["item_id"].astype(int), d["name"]))
    for p in P:
        p["seed_names"] = [names[int(s)] for s in p["seeds"]]
    out = PROJECT_ROOT / "eval" / "profiles.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(P, out.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    import collections
    print(f"프로필 {len(P)}개 → {out}")
    print("  축별:", dict(collections.Counter(p["axis"] for p in P)))
    print("  장르:", len(genres), genres)


if __name__ == "__main__":
    main()
