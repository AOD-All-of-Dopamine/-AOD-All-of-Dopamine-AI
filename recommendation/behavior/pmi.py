"""공동보유 PMI 정답 집합 — P2 산출물.

행동 데이터(Steam 보유·플레이타임)에서 **아이템-아이템 관련성**을 만든다.
LLM 을 쓰지 않는다. 이것이 P3 판정의 정답이 된다.

왜 PMI 인가: 원시 공동보유 수는 인기작을 그대로 베낀다 —
"이 게임을 산 사람은 CS2 도 샀다"는 어느 시드에서나 참이라 정보가 없다.
PMI 는 **우연히 겹칠 양보다 얼마나 더 겹치는가**를 재므로 그 편향을 나눠 없앤다.

    PMI(a,b) = log( P(a,b) / (P(a)·P(b)) )

두 가지 하한을 둔다:
  · **플레이타임 하한**: 산 것과 한 것은 다르다. 기본 60분 — 파일럿에서
    보유 중앙 38.5개 중 1시간 이상은 20.5개(53%)였다. 라이브러리에 묻힌
    번들 게임을 취향 신호로 세지 않는다.
  · **최소 공동보유**: 표본이 작으면 PMI 가 폭발한다(1명만 겹쳐도 큰 값).
    기본 20명 미만은 버린다.
"""
from __future__ import annotations
import argparse, json, math, collections
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def load_users(path: Path, min_minutes: int, min_lib: int, max_lib: int):
    """steamid → 보유 게임 집합. 플레이타임 하한과 라이브러리 크기 창을 적용한다."""
    users = []
    seen = set()
    for line in path.open():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = o.get("steamid")
        if sid in seen:
            continue
        seen.add(sid)
        games = {int(a) for a, m in (o.get("games") or []) if (m or 0) >= min_minutes}
        # 라이브러리가 너무 작으면 신호가 없고, 너무 크면(수집가) 아무 쌍이나 만든다.
        if min_lib <= len(games) <= max_lib:
            users.append(games)
    return users


def build(users, min_co: int, top_per_item: int):
    """공동보유 쌍 → 아이템별 PMI 이웃.

    **쌍을 세기 전에 희소 아이템을 쳐낸다.** `co[(a,b)] <= cnt[a]` 이므로
    `cnt[a] < min_co` 인 아이템이 낀 쌍은 **반드시** 문턱에서 탈락한다.
    세어 본 뒤 버리는 것과 결과가 완전히 같은데 쌍 공간은 훨씬 작아진다
    (Steam 은 롱테일이 길어 대부분의 아이템이 여기서 빠진다).

    이 가지치기가 없으면 메모리가 터진다 — 라이브러리 1,000개짜리 사용자 한 명이
    50만 쌍을 만들고 그 쌍이 전부 딕셔너리 키로 남는다 (실측 OOM, 종료코드 137).
    """
    n = len(users)
    cnt = collections.Counter()
    for g in users:
        cnt.update(g)
    keep = {a for a, c in cnt.items() if c >= min_co}
    co = collections.Counter()
    for g in users:
        items = sorted(g & keep)
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                co[(a, b)] += 1
    out = collections.defaultdict(list)
    for (a, b), c in co.items():
        if c < min_co:
            continue
        pmi = math.log((c / n) / ((cnt[a] / n) * (cnt[b] / n)))
        npmi = pmi / (-math.log(c / n))          # [-1, 1] 로 정규화
        out[a].append((b, npmi, c))
        out[b].append((a, npmi, c))
    for a in out:
        out[a].sort(key=lambda x: -x[1])
        del out[a][top_per_item:]
    return out, cnt, n


def main():
    ap = argparse.ArgumentParser(description="공동보유 PMI 정답 집합")
    ap.add_argument("--owned", default=str(DATA / "owned.jsonl"))
    ap.add_argument("--min-minutes", type=int, default=60)
    ap.add_argument("--min-lib", type=int, default=5)
    # 수집가 계정은 아무 쌍이나 만든다. 1,000 이면 한 명이 50만 쌍을 낳는다.
    ap.add_argument("--max-lib", type=int, default=500)
    ap.add_argument("--min-co", type=int, default=20)
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default=str(DATA / "pmi.json"))
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 통계만 낸다")
    a = ap.parse_args()

    users = load_users(Path(a.owned), a.min_minutes, a.min_lib, a.max_lib)
    if not users:
        raise SystemExit("조건을 통과한 사용자가 없다")
    pmi, cnt, n = build(users, a.min_co, a.top)
    sizes = sorted(len(g) for g in users)
    print(f"사용자 {n:,} · 라이브러리 중앙 {sizes[len(sizes)//2]} · p90 {sizes[int(len(sizes)*0.9)]}")
    kept = sum(1 for c in cnt.values() if c >= a.min_co)
    print(f"아이템 {len(cnt):,} (쌍 세기 대상 {kept:,} · {kept/len(cnt)*100:.1f}%) · "
          f"PMI 이웃 보유 {len(pmi):,} · 쌍 {sum(len(v) for v in pmi.values())//2:,}")
    if pmi:
        deg = sorted(len(v) for v in pmi.values())
        print(f"아이템당 이웃 중앙 {deg[len(deg)//2]} · 최대 {deg[-1]}")
    if a.dry_run:
        print("[시험 실행] 저장하지 않았다")
        return
    Path(a.out).write_text(json.dumps(
        {str(k): [[int(b), round(s, 4), c] for b, s, c in v] for k, v in pmi.items()}))
    print(f"저장 {a.out}")


if __name__ == "__main__":
    main()
