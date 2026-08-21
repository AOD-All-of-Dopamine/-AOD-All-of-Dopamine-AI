"""여러 변형이 가져온 미채점 후보를 한 번에 모아 눈가림 청크로 낸다.

**변형 비교는 미채점이 0 이 되기 전에는 하지 않는다.** 미채점을 집계에서 빼면
새 후보를 많이 가져오는 변형일수록 기저선과 겹치는 부분만으로 평가되어 편향된다.
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval_harness import load_bank, variant_recs, P1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)      # JSON 배열
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", default="d"); ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    prof = pd.read_parquet(P1 / "profiles.parquet")
    if a.split != "all": prof = prof[prof["split"] == a.split]
    bank = load_bank(); need = {}
    comp = None
    for var in json.loads(a.variants):
        recs, comp = variant_recs(var, a.k, prof)   # 변형마다 파라미터가 달라 재생성된다
        ds = comp[3].dataset
        for pid, df in recs.items():
            seeds = list(prof[prof.profile_id == pid].iloc[0].seed_rows)
            for row in df["row"].head(a.k):
                key = (pid, str(int(row)))
                if key in bank or key in need: continue
                need[key] = dict(pid=pid, row=int(row),
                    seeds=[dict(name=str(ds.loc[s,"name"]), ov=str(ds.loc[s,"overview"])[:300]) for s in seeds],
                    cname=str(ds.loc[row,"name"]), cov=str(ds.loc[row,"overview"])[:420],
                    cgen=", ".join(map(str, ds.loc[row,"genres"])), cyr=str(ds.loc[row,"date"])[:4],
                    cmedia=str(ds.loc[row,"media"]))
    need = list(need.values())
    rng = random.Random(a.seed); rng.shuffle(need)
    for i, r in enumerate(need): r["id"] = f"{a.tag}{i:04d}"
    (P1 / f"{a.tag}_key.json").write_text(json.dumps(
        [{k: r[k] for k in ("id","pid","row")} for r in need], ensure_ascii=False), encoding="utf-8")
    outdir = P1 / "chunks"; outdir.mkdir(exist_ok=True)
    for i in range(0, len(need), a.chunk):
        buf = []
        for r in need[i:i+a.chunk]:
            buf.append(f"## {r['id']}\n**좋아한 작품**")
            for s in r["seeds"]: buf.append(f"  · {s['name']} — {s['ov']}")
            buf.append(f"\n**추천 후보** ({r['cmedia']} · {r['cyr']} · {r['cgen']}) {r['cname']}\n{r['cov']}\n")
        (outdir / f"{a.tag}{i//a.chunk+1}.txt").write_text("\n".join(buf), encoding="utf-8")
    print(f"변형 {len(json.loads(a.variants))}개 · 신규 채점 필요 {len(need)}쌍 "
          f"→ {a.tag}1..{(len(need)+a.chunk-1)//a.chunk}.txt")

if __name__ == "__main__": main()
