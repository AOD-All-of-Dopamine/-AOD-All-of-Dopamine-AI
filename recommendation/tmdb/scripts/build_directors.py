"""감독 jsonl → `artifacts/<dir>/directors.parquet`.

**dataset.parquet 에 열을 더하지 않는다.** 그 파일은 임베딩(corpus_index.parquet)과
행 순서로 묶여 있어서, 다시 만들면 임베딩까지 같이 검증해야 한다. 감독은 임베딩에
들어가지 않는 **랭킹 전용 신호**라 옆에 따로 둔다. 없으면 랭커가 축을 끈다.

키는 사람 id(정수)다. 이름은 진단용으로만 같이 싣는다 — 매칭에는 안 쓴다.

    python scripts/build_directors.py
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import artifact_dir

DATA = ROOT / "data"


def main():
    d = artifact_dir(None)
    have = set(pd.read_parquet(d / "dataset.parquet", columns=["item_id"])["item_id"])
    rows = []
    for media in ("movie", "tv"):
        p = DATA / f"{media}_directors.jsonl"
        if not p.exists():
            print(f"없음: {p}")
            continue
        for line in p.open():
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = f"{o['media']}_{o['id']}"
            if iid not in have:            # 코퍼스 필터(줄거리 20자)에서 빠진 것
                continue
            rows.append({"item_id": iid,
                         "directors": [p_["id"] for p_ in o["directors"]],
                         "director_names": [p_["name"] for p_ in o["directors"]]})
    df = pd.DataFrame(rows).drop_duplicates("item_id").reset_index(drop=True)
    out = d / "directors.parquet"
    df.to_parquet(out, index=False)
    n_any = int((df["directors"].str.len() > 0).sum())
    print(f"{len(df):,}건 저장 → {out}")
    print(f"  코퍼스 {len(have):,} 중 피복 {len(df)/len(have)*100:.1f}% · "
          f"감독 1명 이상 {n_any:,} ({n_any/max(len(df),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
