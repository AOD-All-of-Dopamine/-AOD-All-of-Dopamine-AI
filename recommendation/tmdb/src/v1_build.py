"""V-1: require_korean True/False 두 팔. 다른 축은 PRODUCTION 고정. seed 11. 은행 재사용 없음."""
import json, os, random, sys, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI"
           "/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")

from src.personalized_retrieve import build_components, recommend
from src.config import PRODUCTION

P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
FIX = {k: v for k, v in PRODUCTION.items() if k != "strategy"}

lists, meta_by_arm = {}, {}
for arm, ko in (("k_on", True), ("k_off", False)):
    comp = build_components(**{**FIX, "require_korean": ko})
    loader, ret, agg, ranker = comp
    ds = ranker.dataset
    meta_by_arm[arm] = (ds, ret)
    for r in P.itertuples(index=False):
        rows = [int(x) for x in r.seed_rows]
        out = recommend(rows, components=comp, strategy=PRODUCTION["strategy"], top_n=50)
        lists[f"{arm}|{r.profile_id}"] = [str(x) for x in out["item_id"]]
json.dump(lists, open(ROOT / "eval/v1_lists.json", "w"))

ds, ret = meta_by_arm["k_off"]
emb = np.asarray(ret.embeddings, dtype=np.float32)
row_of = dict(zip(ds["item_id"], ds["row"]))
has_ko = dict(zip(ds["item_id"],
                  ds["overview"].fillna("").str.contains(r"[가-힣]")))
name_of = dict(zip(ds["item_id"], ds["name"]))
vote_of = dict(zip(ds["item_id"], ds["vote_count"]))


def ils(ids):
    M = emb[[row_of[i] for i in ids[:50]]]
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    iu = np.triu_indices(len(M), 1)
    return float(S[iu].mean())


for arm in ("k_on", "k_off"):
    v = [ils(lists[f"{arm}|{r.profile_id}"]) for r in P.itertuples(index=False)]
    ov = np.mean([len(set(lists[f"{arm}|{r.profile_id}"]) & set(lists[f"k_on|{r.profile_id}"])) / 50
                  for r in P.itertuples(index=False)])
    uniq = len({i for r in P.itertuples(index=False) for i in lists[f"{arm}|{r.profile_id}"]})
    ko = np.mean([has_ko.get(i, False)
                  for r in P.itertuples(index=False) for i in lists[f"{arm}|{r.profile_id}"][:50]])
    print(f"  {arm:6} ILS@50 {np.mean(v):.3f} · k_on 과 겹침 {ov*100:.0f}% · "
          f"고유 {uniq:,} · 한국어 줄거리 비율 {ko*100:.0f}%")

# 채점 시트
kw_of = dict(zip(ds["item_id"], ds["keywords"]))
g_of = dict(zip(ds["item_id"], ds["genres"]))
ov_of = dict(zip(ds["item_id"], ds["overview"]))


def _lst(v):
    """parquet 의 list 열은 numpy 배열로 온다 — `or []` 가 모호성 오류를 낸다."""
    if v is None:
        return []
    return list(v) if hasattr(v, "__len__") else []


def card(i):
    g = _lst(g_of.get(i))
    kw = _lst(kw_of.get(i))[:6]
    syn = str(ov_of.get(i) or "")[:80].replace("\n", " ")
    return dict(name=str(name_of.get(i, i)),
                meta=f"{'영화' if str(i).startswith('movie') else '드라마'} · "
                     f"{', '.join(map(str, g)) or '-'} · {', '.join(map(str, kw)) or '-'} | {syn}")


SN = {r.profile_id: ", ".join(str(ds.iloc[int(x)]["name"]) for x in r.seed_rows)
      for r in P.itertuples(index=False)}

rng = random.Random(11)
rows = []
for key, L in sorted(lists.items()):
    arm, pid = key.split("|")
    head = [(x, i + 1) for i, x in enumerate(L[:10])]
    tail = [(x, i + 11) for i, x in enumerate(L[10:50])]
    pick = ([(x, p, "top10") for x, p in rng.sample(head, min(5, len(head)))] +
            [(x, p, "tail") for x, p in rng.sample(tail, min(5, len(tail)))])
    for x, p, part in pick:
        rows.append(dict(arm=arm, pid=pid, item=x, pos=p, part=part))

need, seen = [], set()
for r in rows:
    k = (r["pid"], r["item"])
    if k not in seen:
        seen.add(k)
        need.append(k)
rng.shuffle(need)
todo = [dict(id=f"v{i:04d}", pid=k[0], item=k[1], seed=SN[k[0]],
             cand=card(k[1])["name"], meta=card(k[1])["meta"])
        for i, k in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(ROOT / "eval/v1_key.json", "w"), ensure_ascii=False)

nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"v1_b{b}.json").write_text(json.dumps(
        [{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]],
        ensure_ascii=False))
by = collections.Counter(r["arm"] for r in rows)
print("슬롯:", dict(by))
print(f"신규 채점 {len(todo)} · 배치 {nb}")
