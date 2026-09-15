"""여러 변형의 미채점 쌍을 한 번에 눈가림 청크로 뽑는다.

변형별로 따로 뽑으면 겹치는 후보를 두 번 채점하게 되고, 무엇보다
**미채점을 남긴 채 비교하는 사고**가 난다. 합집합을 한 번에 뽑아 0 으로 만든다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.wn_eval import Engine, P1, load_bank, load_profiles, variant_recs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--variants", required=True, help="JSON 배열")
    a = ap.parse_args()

    eng = Engine()
    profs = load_profiles(a.split)
    bank = load_bank()
    variants = json.loads(a.variants)

    seen, need = set(), []
    for v in variants:
        for pid, df in variant_recs(v, a.k, profs, eng).items():
            seeds = profs.loc[profs.profile_id == pid, "seed_ids"].iloc[0]
            for _, r in df.iterrows():
                key = (pid, str(r["item_id"]))
                if key in bank or key in seen:
                    continue
                seen.add(key)
                need.append({"pid": pid, "item_id": int(r["item_id"]),
                             "name": r["name"], "seeds": list(seeds)})

    ds = eng.ds.set_index("item_id")
    chunks, key = [], []
    for n, it in enumerate(need):
        gid = f"{a.tag}{n:04d}"
        key.append({"id": gid, "pid": it["pid"], "item_id": it["item_id"]})
        seed_txt = "\n".join(
            f"  · {ds.loc[s,'name']} — {str(ds.loc[s,'genres'])[:20]}" for s in it["seeds"][:6])
        row = ds.loc[it["item_id"]]
        chunks.append(
            f"## {gid}\n**좋아한 작품**\n{seed_txt}\n\n"
            f"**추천 후보** ({str(row['genres'])[:24]} · {int(row['episode_count'])}화) {it['name']}\n"
            f"{str(row['synopsis'])[:400]}\n")

    (P1 / f"{a.tag}_key.json").write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    cd = P1 / "chunks"
    cd.mkdir(exist_ok=True)
    files = 0
    for i in range(0, len(chunks), a.chunk):
        files += 1
        (cd / f"{a.tag}{files}.txt").write_text("\n".join(chunks[i:i + a.chunk]), encoding="utf-8")
    print(f"변형 {len(variants)}개 · 신규 채점 필요 {len(need)}쌍 → {a.tag}1..{files}.txt")


if __name__ == "__main__":
    main()
