# src/text_builder.py
from pathlib import Path

import pandas as pd

from src.config import ensure_artifacts_dir

# Steam categories 는 게임플레이 정보와 플랫폼 기능이 섞여 있다.
# 플랫폼 기능은 거의 모든 게임에 붙어서 정보량이 0 인데 semantic_text 의 17%(중앙값)를
# 차지한다. 측정: 이 문구들만으로 만든 벡터가 전 코퍼스와 평균 0.519 유사도를 낸다
# (무관한 문장은 0.229). 즉 "임의의 두 게임이 52% 비슷하다"는 결과가 게임 내용이 아니라
# 이 공통 문구 때문에 생긴다. 빼야 변별 구간이 넓어진다.
#
# 출현율(19,476개 중): 가족 공유 99.2% / 싱글 플레이어 95.6% / Steam 도전 과제 57.2%
#                     Steam 트레이딩 카드 38.8% / Steam Cloud 33.6% / 컨트롤러 완벽 지원 27.3%
PLATFORM_CATEGORY_PREFIXES = (
    "Steam ",           # 도전 과제 · 트레이딩 카드 · 순위표 · 워크샵 · Cloud · Turn Notifications
    "Remote Play",      # 휴대전화 · 태블릿 · TV · Together
    "컨트롤러",           # 완벽 지원 · 일부 지원
    "추적되는 컨트롤러",
    "Valve 안티",
)
PLATFORM_CATEGORY_EXACT = frozenset({
    "가족 공유", "가족 공유 라이브러리", "통계", "수시 저장", "인앱 구매",
    "다운로드 가능한 콘텐츠", "게임 내 구매", "부분 컨트롤러 지원",
    "captions available", "자막 사용 가능", "commentary available",
    "색상 대체", "음량 개별 조절", "마우스 전용 옵션", "퀵타임 이벤트 없이 플레이 가능",
    "카메라 움직임 조정", "채팅 텍스트 음성 변환", "사용자 지정 볼륨 조절",
    "DUALSHOCK 컨트롤러 지원", "DualSense 컨트롤러 지원",
    "HDR 사용 가능", "SteamVR 컬렉션",
})

# 남길 것 — 이건 진짜 게임플레이 정보다.
#   싱글 플레이어 / 멀티플레이어 / 협동 / 온라인 협동 / PvP / 온라인 PvP /
#   공유 및 분할 화면 / 대규모 멀티플레이어 / VR 전용 / VR 지원 ...
# 특히 'VR 전용'(9.3%)은 P03 에서 VR 전용 게임이 4위로 올라온 것을 걸러낼 신호다.


def is_platform_noise(category: str) -> bool:
    c = category.strip()
    return c in PLATFORM_CATEGORY_EXACT or c.startswith(PLATFORM_CATEGORY_PREFIXES)


def clean_categories(categories) -> list[str]:
    """플랫폼 기능을 걷어내고 게임플레이 모드만 남긴다. 중복도 제거한다."""
    out, seen = [], set()
    for c in categories if categories is not None else []:
        c = str(c).strip()
        if not c or c in seen or is_platform_noise(c):
            continue
        seen.add(c)
        out.append(c)
    return out


def build_semantic_text(
    short_description: str,
    genres: list[str],
    categories: list[str] | None = None,
) -> str:
    """임베딩에 넣을 의미 텍스트.

    판매 순위(`steam_rank`)는 일부러 넣지 않는다. 인기도는 의미 신호가 아니라 랭킹
    신호이고, R1 이 이미 `recommendations_percentile` 로 랭킹 단계에서 다룬다.
    """
    text = f"Description: {short_description}"
    if len(genres):
        text += f"\nGenres: {', '.join(genres)}"
    modes = clean_categories(categories)
    if modes:
        text += f"\nModes: {', '.join(modes)}"
    return text


def add_semantic_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_cat = "categories" in df.columns
    df["semantic_text"] = [
        build_semantic_text(d, g, categories=c if has_cat else None)
        for d, g, c in zip(
            df["short_description"],
            df["genres"],
            df["categories"] if has_cat else [None] * len(df),
        )
    ]
    return df


def main():
    """`--out DIR` 로 새 표현을 별도 디렉터리에 만든다.

    안전장치: 이미 임베딩이 있는 디렉터리를 덮어쓰려 하면 막는다. semantic_text 를
    바꿔놓고 corpus_embeddings.npy 는 옛 텍스트로 만든 것을 그대로 두면, 두 파일이
    조용히 어긋나서 이후 모든 실험 비교가 무의미해진다.
    """
    import sys

    src = ensure_artifacts_dir()
    out = src
    for i, a in enumerate(sys.argv):
        if a == "--out" and i + 1 < len(sys.argv):
            out = Path(sys.argv[i + 1])
    out.mkdir(parents=True, exist_ok=True)

    if (out / "corpus_embeddings.npy").exists() and "--force" not in sys.argv:
        raise SystemExit(
            f"{out} 에 이미 corpus_embeddings.npy 가 있습니다.\n"
            f"  semantic_text 만 바꾸면 임베딩과 어긋나 실험 비교가 무의미해집니다.\n"
            f"  새 표현은 --out 으로 새 디렉터리에 만드세요 (예: --out artifacts/rep_v3).\n"
            f"  정말 덮어쓰려면 --force 를 붙이고, 반드시 재임베딩까지 다시 하세요."
        )

    df = pd.read_parquet(src / "dataset.parquet")
    before = df["semantic_text"].str.len().median() if "semantic_text" in df else None
    df = add_semantic_text(df)
    print(f"rows={len(df)}  →  {out / 'dataset.parquet'}")
    if before:
        after = df["semantic_text"].str.len().median()
        print(f"semantic_text 길이 중앙값: {before:.0f}자 → {after:.0f}자 ({(after / before - 1) * 100:+.0f}%)")
    df.to_parquet(out / "dataset.parquet", index=False)
    print(df.iloc[0]["semantic_text"])


if __name__ == "__main__":
    main()
