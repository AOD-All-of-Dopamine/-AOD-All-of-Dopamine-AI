"""X-26 빌드: TMDB·웹툰 next_page 5페이지(seen 누적) → 페이지 1·3·5 전수 눈가림 시트. 사전등록 eval/x26_preregister.md.

    python crossdomain/x26_build.py tmdb
    python crossdomain/x26_build.py webtoon
플랫폼마다 `src` 패키지가 달라 한 프로세스에 하나씩 돈다.
"""
import json, os, random, sys
from pathlib import Path
import pandas as pd
R = Path("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
OUT = R / "crossdomain"
SCR = Path("/tmp/claude-501/-home-ubuntu--AOD-All-of-Dopamine-AI/23bee083-2d26-402a-a82f-838bff2e4213/scratchpad")
plat = sys.argv[1]; GRADED = (1, 3, 5)
pages, SN, card = {}, {}, None

if plat == "tmdb":
    os.chdir(R / "tmdb"); sys.path.insert(0, "."); os.environ.setdefault("AOD_ARTIFACTS", "artifacts/tmdb_v1")
    from src.personalized_retrieve import build_components, next_page
    from src.config import PRODUCTION
    comps = build_components(**{k: v for k, v in PRODUCTION.items() if k != "strategy"})
    ds = comps[3].dataset
    P = pd.read_parquet("artifacts/p1/profiles.parquet").iloc[::4]
    name_of = dict(zip(ds["item_id"], ds["name"])); kw_of = dict(zip(ds["item_id"], ds["keywords"]))
    g_of = dict(zip(ds["item_id"], ds["genres"])); ov_of = dict(zip(ds["item_id"], ds["overview"]))
    _lst = lambda v: list(v) if v is not None and hasattr(v, "__len__") else []
    def card(i):   # V-4 시트와 같은 형식
        g = _lst(g_of.get(i)); kw = _lst(kw_of.get(i))[:6]; syn = str(ov_of.get(i) or "")[:80].replace("\n", " ")
        return dict(name=str(name_of.get(i, i)), meta=f"{'영화' if str(i).startswith('movie') else '드라마'} · {', '.join(map(str, g)) or '-'} · {', '.join(map(str, kw)) or '-'} | {syn}")
    V4 = json.load(open("eval/v4_lists.json")); p1_same = 0
    for r in P.itertuples(index=False):
        seeds = [int(x) for x in r.seed_rows]; seen = set(); pg = []
        for _ in range(5):
            df = next_page(seeds, seen_rows=seen, page_size=10, components=comps)
            seen |= {int(x) for x in df["row"]}; pg.append([str(x) for x in df["item_id"]])
        pages[r.profile_id] = pg; p1_same += pg[0] == V4[f"h4|{r.profile_id}"][:10]
        SN[r.profile_id] = ", ".join(str(ds.iloc[int(x)]["name"]) for x in r.seed_rows)
    print(f"[tmdb] 프로필 {len(P)} · 1페이지 == V-4 h4 top-10 {p1_same}/{len(P)}")
else:
    os.chdir(R / "webtoon"); sys.path.insert(0, "."); os.environ["AOD_WT_ARTIFACTS"] = str(R / "webtoon/artifacts/wt_v1")
    from src.personalized_retrieve import Engine
    eng = Engine(); P = json.load(open("eval/profiles.json"))[::5]
    dsi = eng.ds.set_index("item_id")
    def card(i):   # T-9 시트와 같은 형식
        r = dsi.loc[int(i)]
        t = [x for x in (list(r["tags"]) if r["tags"] is not None else []) if not str(x).startswith("완결")][:6]
        g = list(r["genres"]) if r["genres"] is not None else []
        return dict(name=str(r["name"]), meta=f"웹툰 · {', '.join(map(str,g)) or '-'} · {', '.join(map(str,t)) or '-'} | {int(r.get('episode_count',0))}화")
    T9 = json.load(open("eval/t9_lists.json")); p1_same = 0
    for p in P:
        seen = []; pg = []
        for _ in range(5):
            ids = [int(x) for x in eng.next_page(p["seeds"], k=10, seen=seen)["item_id"]]
            seen += ids; pg.append(ids)
        pages[p["pid"]] = pg; p1_same += pg[0] == T9[f"g2|{p['pid']}"][:10]
        SN[p["pid"]] = ", ".join(p["seed_names"])
    print(f"[webtoon] 프로필 {len(P)} · 1페이지 == T-9 g2 top-10 {p1_same}/{len(P)}")

json.dump(pages, open(OUT / f"x26_{plat}_pages.json", "w"), ensure_ascii=False)
rows = [dict(pid=pid, item=i, page=n, pos=(n - 1) * 10 + j + 1) for pid, pg in pages.items() for n in GRADED for j, i in enumerate(pg[n - 1])]
assert len({(r["pid"], r["item"]) for r in rows}) == len(rows), "페이지 간 중복이 있다(seen 누적 실패)"
rng = random.Random(26); need = [(r["pid"], r["item"]) for r in rows]; rng.shuffle(need)
tag = "zt" if plat == "tmdb" else "zw"
todo = [dict(id=f"{tag}{n:04d}", pid=p, item=i, seed=SN[p], cand=card(i)["name"], meta=card(i)["meta"]) for n, (p, i) in enumerate(need)]
json.dump(dict(rows=rows, todo=todo), open(OUT / f"x26_{plat}_key.json", "w"), ensure_ascii=False)
nb = (len(todo) + 59) // 60
for b in range(nb):
    (SCR / f"x26_{plat}_b{b}.json").write_text(json.dumps([{k: r[k] for k in ("id", "seed", "cand", "meta")} for r in todo[b*60:(b+1)*60]], ensure_ascii=False))
(OUT / "x26" / plat).mkdir(parents=True, exist_ok=True)
print(f"[{plat}] 슬롯 {len(rows)} · 신규 채점 {len(todo)} · 배치 {nb}")
