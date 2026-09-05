"""T-1 집계 — 사전등록 2f06edaa… 문턱대로만. 미채점이 남으면 비교하지 않는다."""
import json, glob, collections, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
K = json.load(open(ROOT / "eval/t1_key.json")); rows, todo = K["rows"], K["todo"]
key = {t["id"]: t for t in todo}
votes = collections.defaultdict(list)
for f in sorted(glob.glob(str(ROOT / "eval/t1/[ABC]_*.json"))):
    for i, g in json.load(open(f)).items():
        if i in key: votes[i].append(int(g) >= 2)
maj, unan = {}, 0
for i, v in votes.items():
    if len(v) >= 3: maj[i] = sum(v) >= 2; unan += (len(set(v)) == 1)
print(f"신규 판정 {len(maj)}/{len(todo)} · 만장일치 {unan/max(len(maj),1):.2f}")
pair = {(key[i]["pid"], key[i]["item"]): m for i, m in maj.items()}
miss = 0
for r in rows:
    k = (r["pid"], r["item"])
    if k in pair: r["maj"] = pair[k]
    else: miss += 1
print(f"미채점 슬롯 {miss}")
if miss: sys.exit("미채점이 남아 비교하지 않는다 (사전등록)")
json.dump(dict(rows=rows), open(ROOT / "eval/t1_graded.json", "w"), ensure_ascii=False)
P = lambda rs: float(np.mean([r["maj"] for r in rs])) if rs else float("nan")
V = {}
print("\n### 팔별 (프로필 67, 보정 0)")
for a in ("rep_v1", "rep_v2"):
    s = [r for r in rows if r["arm"] == a]; h = [r for r in s if r["part"] == "top10"]; t = [r for r in s if r["part"] == "tail"]
    V[a] = P(s); print(f"  {a}: 1-10 {P(h):.3f} · 11-50 {P(t):.3f} · 합산 {P(s):.4f} (n={len(s)})")
d = V["rep_v2"] - V["rep_v1"]
print(f"\nΔ(rep_v2 − rep_v1) = {d:+.4f}")
print("판정:", "(A) rep_v2 채택" if d >= 0.03 else ("(C) rep_v1 채택" if d <= -0.03 else "(B) 동률 → rep_v1 채택"))
prof = {p["pid"]: p for p in json.load(open(ROOT / "eval/profiles.json"))}
print("\n### 축별 합산 (판정 아님)")
for ax in ("rule", "coh", "mix", "count"):
    line = f"  {ax:<6}"
    for a in ("rep_v1", "rep_v2"):
        s = [r for r in rows if r["arm"] == a and prof[r["pid"]]["axis"] == ax]; line += f"  {a} {P(s):.3f}"
    print(line)
print("\n### rule 인기 대역별")
for band in ("high", "mid", "low"):
    line = f"  {band:<5}"
    for a in ("rep_v1", "rep_v2"):
        s = [r for r in rows if r["arm"] == a and prof[r["pid"]].get("band") == band]; line += f"  {a} {P(s):.3f}"
    print(line)
# 프로필별 승패
w = l = e = 0
for pid in prof:
    a = P([r for r in rows if r["pid"] == pid and r["arm"] == "rep_v2"]); b = P([r for r in rows if r["pid"] == pid and r["arm"] == "rep_v1"])
    if a - b > 0.02: w += 1
    elif b - a > 0.02: l += 1
    else: e += 1
print(f"\n프로필 분해: rep_v2 이김 {w} · rep_v1 이김 {l} · 비슷 {e}")
