"""H-CR 검정 — 사전등록 eval/hcr_preregister.md 의 판정 규칙 그대로. 채점 추가 없음.

X = 독립 라운드(V-1 k_off · T-6 b0 · W-3 r0) 기준선 프로필 적합률 (3역할 다수결)
Y = 현 라운드(V-2 · T-7 · W-4) 역할 B 적합률 w=0.10 − w=0
평균으로의 회귀를 피하려고 X 와 Y 는 서로 다른 슬롯 표본에서 온다.
"""
import json, glob
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

R = Path("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
W = Path("/home/ubuntu/aod-webnovel/recommendation/webnovel")

PLAT = [
    ("TMDB",   R / "tmdb/eval/v1_graded.json",    "k_off", R / "tmdb/eval/v2_key.json",    R / "tmdb/eval/v2",    "d0", "d10"),
    ("웹툰",   R / "webtoon/eval/t6_graded.json", "b0",    R / "webtoon/eval/t7_key.json", R / "webtoon/eval/t7", "c0", "c10"),
    ("웹소설", W / "eval/w3_graded.json",         "r0",    W / "eval/w4_key.json",         W / "eval/w4",         "a0", "a10"),
]


def perm_p(x, y, n=5000, seed=13):
    rng = np.random.default_rng(seed)
    r0 = spearmanr(x, y).correlation
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n)])
    return float(np.mean(null <= r0))          # 단측: ρ 가 이만큼 음일 확률


rhos, allx, ally = [], [], []
for name, g_old, arm_old, k_new, d_new, a0, a1 in PLAT:
    old = [r for r in json.load(open(g_old))["rows"] if r["arm"] == arm_old and "maj" in r]
    X = {}
    for r in old:
        X.setdefault(r["pid"], []).append(bool(r["maj"]))
    K = json.load(open(k_new)); key = {t["id"]: t for t in K["todo"]}; target = set(K["target"])
    B = {}
    for f in glob.glob(str(d_new / "B_*.json")):
        for i, g in json.load(open(f)).items():
            B[(key[i]["pid"], key[i]["item"])] = int(g)
    Y = {}
    for p in target:
        f0 = [B[(p, r["item"])] >= 2 for r in K["rows"] if r["pid"] == p and r["arm"] == a0]
        f1 = [B[(p, r["item"])] >= 2 for r in K["rows"] if r["pid"] == p and r["arm"] == a1]
        if f0 and f1 and p in X:
            Y[p] = np.mean(f1) - np.mean(f0)
    ps = sorted(Y)
    x = np.array([np.mean(X[p]) for p in ps]); y = np.array([Y[p] for p in ps])
    rho = spearmanr(x, y).correlation
    rhos.append(rho); allx += list(x); ally += list(y)
    print(f"{name}: 프로필 {len(ps)} · X 평균 {x.mean():.3f} (sd {x.std():.3f}) · Y 평균 {y.mean():+.3f} (sd {y.std():.3f}) · "
          f"ρ = {rho:+.3f} · 순열 p(단측) {perm_p(x, y):.3f}")
    lo, hi = x <= np.median(x), x > np.median(x)
    print(f"        X 하위 절반 Y 평균 {y[lo].mean():+.3f} · 상위 절반 {y[hi].mean():+.3f}")

neg = sum(1 for r in rhos if r < 0); mean_rho = float(np.mean(rhos))
print(f"\n음의 ρ 플랫폼 {neg}/3 · 평균 ρ {mean_rho:+.3f}")
print(f"(부수, 판정 아님 — 순환적) 풀링 ρ {spearmanr(allx, ally).correlation:+.3f}")
if neg >= 2 and mean_rho <= -0.15:
    print("판정: 지지 — 헤드룸이 작은 프로필일수록 창작자 보너스 이득이 작다")
elif mean_rho >= 0:
    print("판정: 반증 — 프로필 단위에서 헤드룸과 이득의 음의 관계가 보이지 않는다")
else:
    print("판정: 결론 없음 — 방향은 음이나 문턱 미달 (검정력 부족으로 기록, 반대 증거 아님)")
