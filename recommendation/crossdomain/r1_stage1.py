"""R-1 1단계: dev 60 + val 32 의 Steam·TMDB top-100 을 만들어 저장하고 종료 (모델 없음, ~2GB)."""
import json, os, sys
os.chdir("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation"); sys.path.insert(0,"crossdomain")
import xseed; xseed.load_all()
DEV=set(json.load(open("crossdomain/mixed_x11.json"))["M6"])
dev=[e for e in json.load(open("crossdomain/profiles_random100.json")) if e["pid"] in DEV]
SI=json.load(open("crossdomain/seed_index.json"))["seeds"]
val=json.load(open("crossdomain/personas_v2.json"))
out={}
for e in dev:
    seeds={p:SI[p][e[p]] for p in ("steam","tmdb","wn") if e.get(p)}
    for pl in ("steam","tmdb"):
        if seeds.get(pl): out[f"dev|{pl}|{e['pid']}"]=dict(seeds=seeds[pl], top100=xseed.RANK[pl](seeds,"M6","a",100))
    out[f"dev|wn|{e['pid']}"]=dict(seeds=seeds.get("wn",[]), top100=xseed.RANK["wn"](seeds,"M6","a",50) if seeds.get("wn") else [])
    print("dev",e["pid"],flush=True)
for e in val:
    for pl in ("steam","tmdb"):
        if e["seeds"].get(pl): out[f"val|{pl}|{e['pid']}"]=dict(seeds=e["seeds"][pl], top100=xseed.RANK[pl](e["seeds"],"M6","a",100))
    if e["seeds"].get("wn"): out[f"val|wn|{e['pid']}"]=dict(seeds=e["seeds"]["wn"], top100=xseed.RANK["wn"](e["seeds"],"M6","a",50))
    print("val",e["pid"],flush=True)
json.dump(out,open("crossdomain/r1_top100.json","w"))
print("완료", len(out))
