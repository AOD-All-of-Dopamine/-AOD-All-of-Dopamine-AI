import sys, os, json
plat, pids = sys.argv[1], json.loads(sys.argv[2])
import numpy as np, pandas as pd
if plat == "steam":
    R = "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam"
    os.environ.setdefault("AOD_ARTIFACTS", "artifacts/tags_full"); sys.path.insert(0, R); os.chdir(R)
    from src.s_eval import variant_recs, load_bank, load_profiles
    prof = load_profiles(); prof = prof[prof.profile_id.isin(pids)]; bank = load_bank()
    recs = variant_recs({}, 50, prof)
    out = {}
    for _, p in prof.iterrows():
        pid = p["profile_id"]; ids = [int(a) for a in recs[pid]["steam_appid"].head(50)]
        out[pid] = dict(items=ids, n_seeds=int(p["n_seeds"]), coh=float(p["seed_cohesion"]),
                        grades={str(a): bank.get((pid, a)) for a in ids})
elif plat == "tmdb":
    R = "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/tmdb"; sys.path.insert(0, R); os.chdir(R)
    from src.eval_harness import variant_recs, load_bank
    from src.config import PRODUCTION
    prof = pd.read_parquet("artifacts/p1/profiles.parquet"); prof = prof[prof.profile_id.isin(pids)]; bank = load_bank()
    recs, _ = variant_recs(dict(PRODUCTION), 50, prof)
    out = {}
    for r in prof.itertuples(index=False):
        ids = [int(x) for x in recs[r.profile_id]["row"].head(50)]
        out[r.profile_id] = dict(items=ids, n_seeds=int(r.n_seeds), coh=float(r.seed_cohesion),
                                 grades={str(a): bank.get((r.profile_id, str(a))) for a in ids})
else:
    R = "/home/ubuntu/aod-webnovel/recommendation/webnovel"
    os.environ.setdefault("AOD_ARTIFACTS", "artifacts/wn_v4"); sys.path.insert(0, R); os.chdir(R)
    from src.wn_eval import Engine, load_bank, load_profiles, variant_recs
    eng = Engine(); prof = load_profiles(); prof = prof[prof.profile_id.isin(pids)]; bank = load_bank()
    recs = variant_recs({}, 50, prof, eng)
    out = {}
    for _, p in prof.iterrows():
        pid = p["profile_id"]; ids = [int(x) for x in recs[pid]["item_id"][:50]]
        out[pid] = dict(items=ids, n_seeds=int(p["n_seeds"]), coh=float(p["seed_cohesion"]),
                        grades={str(a): bank.get((pid, str(a))) for a in ids})
print(json.dumps(out))
