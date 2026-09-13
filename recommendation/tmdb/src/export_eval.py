"""채점할 (프로필, 후보) 쌍을 뽑아 눈가림 청크로 낸다."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.eval_harness import load_bank, variant_recs, P1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="{}"); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="all"); ap.add_argument("--tag", default="v")
    ap.add_argument("--chunk", type=int, default=30); ap.add_argument("--seed", type=int, default=20260909)
    a = ap.parse_args()
    var = json.loads(a.variant)
    prof = pd.read_parquet(P1 / "profiles.parquet")
    if a.split != "all": prof = prof[prof["split"] == a.split]
    recs, comp = variant_recs(var, a.k, prof)
    ds = comp[3].dataset
    bank = load_bank()
    need = []
    for pid, df in recs.items():
        seeds = list(prof[prof.profile_id == pid].iloc[0].seed_rows)
        for row in df["row"].head(a.k):
            key = (pid, str(int(row)))
            if key in bank: continue
            need.append(dict(pid=pid, row=int(row),
                             seeds=[dict(name=str(ds.loc[s,"name"]), ov=str(ds.loc[s,"overview"])[:420]) for s in seeds],
                             cname=str(ds.loc[row,"name"]), cov=str(ds.loc[row,"overview"])[:700],
                             cgen=", ".join(map(str, ds.loc[row,"genres"])), cyr=str(ds.loc[row,"date"])[:4],
                             cmedia=str(ds.loc[row,"media"])))
    rng = random.Random(a.seed); rng.shuffle(need)
    for i, r in enumerate(need): r["id"] = f"{a.tag}{i:04d}"
    key = [{k: r[k] for k in ("id", "pid", "row")} for r in need]
    (P1 / f"{a.tag}_key.json").write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    outdir = P1 / "chunks"; outdir.mkdir(exist_ok=True)
    for i in range(0, len(need), a.chunk):
        buf = []
        for r in need[i:i+a.chunk]:
            buf.append(f"## {r['id']}\n**좋아한 작품**")
            for s in r["seeds"]: buf.append(f"  · {s['name']} — {s['ov']}")
            buf.append(f"\n**추천 후보** ({r['cmedia']} · {r['cyr']} · {r['cgen']}) {r['cname']}\n{r['cov']}\n")
        (outdir / f"{a.tag}{i//a.chunk+1}.txt").write_text("\n".join(buf), encoding="utf-8")
    print(f"프로필 {len(recs)} · k={a.k} · 신규 채점 필요 {len(need)}쌍 "
          f"(기존 은행 {len(bank)}건 재사용) → {outdir}/{a.tag}1..{(len(need)+a.chunk-1)//a.chunk}.txt")

if __name__ == "__main__": main()
