# src/text_builder.py
"""임베딩에 넣을 `semantic_text` 를 만든다.

**텍스트 자체는 걱정했던 것보다 두껍다.** 49건 실측: 줄거리 중앙값 482자, p25 346자,
30자 미만 0건 — Steam 의 `short_description`(300자 내외)보다 오히려 길다.
(초기 조사에서 "중앙값 53자"로 본 것은 측정 오류였다. 정규식이 첫 `</div>` 에서 끊겨
중첩 마크업 안의 본문을 잘라먹었다. bs4 로 제대로 뽑으면 위 수치가 나온다.)

**대신 장르가 얇다** — 작품당 1개 coarse 라벨(현판/판타지/무협/로판…)뿐이다.
Steam 이 300자 설명 + 다중 장르 + 게임플레이 모드를 갖고도 "분위기로 정의되는 장르는
무너진다"는 표현 한계를 기록했으므로, 여기서도 변별력은 반드시 측정하고 넘어가야 한다.

그래서 표현을 1급 실험 축으로 둔다. `--out DIR` 로 표현별 아티팩트를 분리하고
`src.diagnose_representation` 으로 매번 변별력을 측정한다. rep 를 늘리기 전에 반드시 측정한다 —
측정 없이 텍스트만 늘리면 뭐가 효과였는지 알 수 없다.
"""
from pathlib import Path

import pandas as pd

from src.config import ensure_artifacts_dir

# 관심 수·평점·댓글 수는 넣지 않는다. Steam 이 판매 순위를 뺀 것과 같은 이유다 —
# 인기도는 의미 신호가 아니라 랭킹 신호이고, 랭커가 이미 interest_percentile 로 다룬다.


def build_semantic_text(
    name: str,
    genres: list[str],
    synopsis: str,
    author: str = "",
    publisher: str = "",
    with_meta: bool = False,
) -> str:
    """rep_v1 = 제목 + 장르 + 줄거리.

    **Steam 과 달리 제목을 넣는다.** Steam 은 이름을 뺐지만(고유명사라 의미가 옅다),
    웹소설 제목은 `"퇴사 후 아포칼립스로 출근합니다"` · `"마탑 천재가 물리학을 숨김"` 처럼
    그 자체가 로그라인이다. 줄거리가 카피 문구뿐인 꼬리 작품(실측 22자 사례)에서는
    제목이 사실상 유일한 의미 신호가 된다.

    `with_meta=True` 는 rep_v2 — 작가/출판사를 덧붙인다. 같은 작가의 작품군이 실제
    취향 신호이기 때문인데, 고유명사가 임베딩을 지배할 위험도 있어 기본은 끈다.
    변별력 측정으로 판단한다.
    """
    parts = [f"제목: {name}"]
    if genres:
        parts.append(f"장르: {', '.join(genres)}")
    if synopsis:
        parts.append(f"줄거리: {synopsis}")
    if with_meta:
        if author:
            parts.append(f"작가: {author}")
        if publisher:
            parts.append(f"출판사: {publisher}")
    return "\n".join(parts)


def add_semantic_text(df: pd.DataFrame, with_meta: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["semantic_text"] = [
        build_semantic_text(n, g, s, a, p, with_meta=with_meta)
        for n, g, s, a, p in zip(
            df["name"], df["genres"], df["synopsis"],
            df.get("author", pd.Series([""] * len(df))),
            df.get("publisher", pd.Series([""] * len(df))),
        )
    ]
    return df


def main():
    """`--out DIR` 로 새 표현을 별도 디렉터리에 만든다.

    안전장치는 Steam 판과 같다: 이미 임베딩이 있는 디렉터리를 덮어쓰려 하면 막는다.
    semantic_text 를 바꿔놓고 corpus_embeddings.npy 는 옛 텍스트로 만든 것을 그대로 두면
    두 파일이 조용히 어긋나서 이후 모든 실험 비교가 무의미해진다.
    """
    import sys

    src = ensure_artifacts_dir()
    out = src
    for i, a in enumerate(sys.argv):
        if a == "--out" and i + 1 < len(sys.argv):
            out = Path(sys.argv[i + 1])
    out.mkdir(parents=True, exist_ok=True)
    with_meta = "--with-meta" in sys.argv

    if (out / "corpus_embeddings.npy").exists() and "--force" not in sys.argv:
        raise SystemExit(
            f"{out} 에 이미 corpus_embeddings.npy 가 있습니다.\n"
            f"  semantic_text 만 바꾸면 임베딩과 어긋나 실험 비교가 무의미해집니다.\n"
            f"  새 표현은 --out 으로 새 디렉터리에 만드세요 (예: --out artifacts/rep_v2).\n"
            f"  정말 덮어쓰려면 --force 를 붙이고, 반드시 재임베딩까지 다시 하세요."
        )

    df = pd.read_parquet(src / "dataset.parquet")
    before = df["semantic_text"].str.len().median() if "semantic_text" in df else None
    df = add_semantic_text(df, with_meta=with_meta)
    print(f"rows={len(df)}  with_meta={with_meta}  →  {out / 'dataset.parquet'}")
    after = df["semantic_text"].str.len()
    if before:
        print(f"semantic_text 길이 중앙값: {before:.0f}자 → {after.median():.0f}자")
    else:
        print(f"semantic_text 길이: 중앙값 {after.median():.0f}자 "
              f"(p25 {after.quantile(.25):.0f} / p75 {after.quantile(.75):.0f})")
    df.to_parquet(out / "dataset.parquet", index=False)
    print("--- 예시 ---")
    print(df.iloc[0]["semantic_text"])


if __name__ == "__main__":
    main()
