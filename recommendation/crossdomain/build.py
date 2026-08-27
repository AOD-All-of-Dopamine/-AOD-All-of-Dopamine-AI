"""30개 크로스 프로필에 대해 세 플랫폼 랭커 top-50 을 만들고 M0/M1/M2 를 적용한다.

**채점하지 않는다.** 각 슬롯의 **플랫폼 내 등급**만 기존 은행에서 붙인다 —
"이 항목이 그 플랫폼 프로필에는 맞았나"는 이미 알려진 값이다. 크로스 적합성은 모른다.
플랫폼별로 별도 프로세스에서 돌린다 (`src` 패키지 충돌).
"""
import sys, os, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from profiles import PROFILES
from mix import RULES

PY = {"steam": "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/.venv/bin/python",
      "tmdb":  "/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/.venv/bin/python",
      "wn":    "/home/ubuntu/aod-webnovel/recommendation/webnovel/.venv/bin/python"}

def fetch(plat, pids):
    out = subprocess.run([PY[plat], os.path.join(HERE, "_one.py"), plat, json.dumps(sorted(set(pids)))],
                         capture_output=True, text=True)
    if out.returncode: raise SystemExit(f"{plat} 실패:\n{out.stderr[-1500:]}")
    return json.loads(out.stdout)

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    need = {p: [pr[p] for pr in PROFILES if pr[p]] for p in ("steam", "tmdb", "wn")}
    data = {p: fetch(p, need[p]) for p in need}       # {plat: {pid: {items, grades, n_seeds, coh}}}
    result = {}
    for pr in PROFILES:
        lists, seeds, coh, grades = {}, {}, {}, {}
        for p in ("steam", "tmdb", "wn"):
            if not pr[p]: continue
            d = data[p][pr[p]]
            lists[p] = d["items"]; seeds[p] = d["n_seeds"]; coh[p] = d["coh"]; grades[p] = d["grades"]
        result[pr["pid"]] = {}
        for name, rule in RULES.items():
            mixed = rule(lists, seeds, coh, k=K)
            result[pr["pid"]][name] = [dict(plat=p, item=it, rank=r, in_platform_grade=grades[p].get(str(it)))
                                       for p, it, r in mixed]
    json.dump(result, open(os.path.join(HERE, f"mixed_k{K}.json"), "w"), ensure_ascii=False, indent=1)
    # 요약
    import collections
    for name in RULES:
        lens = [len(result[pid][name]) for pid in result]
        cov = collections.Counter()
        ing = []
        for pid in result:
            for s in result[pid][name]:
                cov[s["plat"]] += 1
                if s["in_platform_grade"] is not None: ing.append(s["in_platform_grade"] >= 2)
        print(f"{name}: 평균 길이 {sum(lens)/len(lens):.1f} · 슬롯 {dict(cov)} · "
              f"플랫폼 내 적합률(기존 은행) {sum(ing)/len(ing):.3f} (채점됨 {len(ing)}/{sum(cov.values())})")
