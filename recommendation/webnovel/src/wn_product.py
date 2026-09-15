"""확정 조건에서 제품 지표를 낸다. 미채점이 0 인지 반드시 확인한다."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wn_eval import Engine, load_bank, load_profiles, score, variant_recs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--variant", default="{}")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    eng, profs, bank = Engine(), load_profiles(a.split), load_bank()
    v = json.loads(a.variant)
    recs = variant_recs(v, a.k, profs, eng)
    s = score(recs, bank, a.k)

    print(f"변형 {a.variant} · split={a.split} · k={a.k}")
    print(f"  프로필 {len(profs)} · 채점 {s['graded']} · 미채점 {s['ungraded']}")
    print(f"  **적합률 {s['fit']:.4f}**  · 평균 등급 {s['mean_grade']:.3f} "
          f"· 0.8 미만 {sum(1 for x in s['per'].values() if x[0] < 0.8)}개")
    if a.quiet:
        return

    axis: dict[str, list] = {}
    for pid, (f, _, _) in s["per"].items():
        axis.setdefault(pid.split("_")[0], []).append(f)
    print("\n  축별:")
    for kx in sorted(axis):
        print(f"    {kx:10s} n={len(axis[kx]):2d} 적합 {np.mean(axis[kx]):.3f}")

    coh = dict(zip(profs.profile_id, profs.seed_cohesion))
    ns = dict(zip(profs.profile_id, profs.n_seeds))
    print("\n  하위 8개:")
    for pid, (f, _, _) in sorted(s["per"].items(), key=lambda x: x[1][0])[:8]:
        print(f"    {pid:24s} {f:.2f} (시드 {ns[pid]}, 응집 {coh[pid]:.2f})")


if __name__ == "__main__":
    main()
