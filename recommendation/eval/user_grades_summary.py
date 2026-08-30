"""사람 채점(user_grades.jsonl, v2) 요약.
항목 키 = (rater, platform, 시드/프로필, item_id). 같은 키의 마지막 줄만 쓴다.
출력: 채점자별 n · 적합률(볼 의향 ≥2) · 이미 봄 비율 · 이유 분포.
  python eval/user_grades_summary.py [경로]
"""
import json, sys, collections
path = sys.argv[1] if len(sys.argv) > 1 else "eval/user_grades.jsonl"
last = {}
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    if r.get("v", 1) != 2: continue
    ctx = r.get("pid") or "|".join(str(s["id"]) for s in r.get("seeds", []))
    last[(r.get("rater"), r["platform"], ctx, str(r["item_id"]))] = r
by = collections.defaultdict(list)
for k, r in last.items(): by[(k[0], k[1])].append(r)
print(f"{'채점자':<10}{'플랫폼':<9}{'n':>4}{'적합률':>7}{'이미봄':>7}  이유")
for (rater, plat), rs in sorted(by.items()):
    n = len(rs); fit = sum(r["grade"] >= 2 for r in rs) / n
    seen = sum(r.get("known") == "seen" for r in rs) / n
    why = collections.Counter(r.get("why") for r in rs if r.get("why"))
    print(f"{str(rater):<10}{plat:<9}{n:>4}{fit:>7.2f}{seen:>7.2f}  {dict(why.most_common(4))}")
print(f"\n총 {len(last)} 항목 (v2)")
