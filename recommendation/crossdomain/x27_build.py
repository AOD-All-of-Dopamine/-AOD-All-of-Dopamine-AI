"""X-27 빌드: TMDB next_page 5페이지(seen 누적) · 52프로필 전부 → 1·5페이지 전수 눈가림 시트. 사전등록 eval/x27_preregister.md."""
import json, os, random, sys
from pathlib import Path
import pandas as pd
R = Path("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"); OUT = R / "crossdomain"
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
os.chdir(R / "tmdb"); sys.path.insert(0, "."); os.environ.setdefault("AOD_ARTIFACTS", "artifacts/tmdb_v1")
from src.personalized_retrieve import build_components, next_page
from src.config import PRODUCTION
comps = build_components(**{k: v for k, v in PRODUCTION.items() if k != "strategy"})
ds = comps[3].dataset
P = pd.read_parquet("artifacts/p1/profiles.parquet")
name_of = dict(zip(ds["item_id"], ds["name"])); kw_of = dict(zip(ds["item_id"], ds["keywords"]))
g_of = dict(zip(ds["item_id"], ds["genres"])); ov_of = dict(zip(ds["item_id"], ds["overview"]))
_lst = lambda v: list(v) if v is not None and hasattr(v, "__len__") else []
def card(i):   # V-4 · X-26 시트와 같은 형식
    g = _lst(g_of.get(i)); kw = _lst(kw_of.get(i))[:6]; syn = str(ov_of.get(i) or "")[:80].replace("\n", " ")
    return dict(name=str(name_of.get(i, i)), meta=f"{'영화' if str(i).startswith('movie') else '드라마'} · {', '.join(map(str, g)) or '-'} · {', '.join(map(str, kw)) or '-'} | {syn}")
V4 = json.load(open("eval/v4_lists.json"))
pages, SN, p1_same = {}, {}, 0
for r in P.itertuples(index=False):
    seeds = [int(x) for x in r.seed_rows]; seen = set(); pg = []
    for _ in range(5):
        df = next_page(seeds, seen_rows=seen, page_size=10, components=comps)
        seen |= {int(x) for x in df["row"]}; pg.append([str(x) for x in df["item_id"]])
    pages[r.profile_id] = pg; p1_same += pg[0] == V4[f"h4|{r.profile_id}"][:10]
    SN[r.profile_id] = ", ".join(str(ds.iloc[int(x)]["name"]) for x in r.seed_rows)
print(f"[tmdb] 프로필 {len(P)} · 1페이지 == V-4 h4 top-10 {p1_same}/{len(P)}")
assert p1_same == len(P), "1페이지가 현행 서빙 top-10 이 아니다"
json.dump(pages, open(OUT / "x27_tmdb_pages.json", "w"), ensure_ascii=False)
rows = [dict(pid=pid, item=i, page=n, pos=(n - 1) * 10 + j + 1) for pid, pg in pages.items() for n in (1, 5) for j, i in enumerate(pg[n - 1])]
assert len({(r["pid"], r["item"]) for r in rows}) == len(rows), "페이지 간 중복이 있다"
rng = random.Random(27); need = [(r["pid"], r["item"]) for r in rows]; rng.shuffle(need)
todo = [dict(id=f"zx{n:04d}", pid=p, item=i, seed=SN[p], cand=card(i)["name"], meta=card(i)["meta"]) for n, (p, i) in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(OUT / "x27_tmdb_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"x27_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
(OUT / "x27" / "tmdb").mkdir(parents=True, exist_ok=True)
print(f"[tmdb] 슬롯 {len(rows)} · 신규 채점 {len(todo)} · 배치 {nb}")
