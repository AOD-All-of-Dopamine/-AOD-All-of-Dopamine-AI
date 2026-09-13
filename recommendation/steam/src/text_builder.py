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


MAX_TAGS = 15


def build_semantic_text(
    short_description: str,
    genres: list[str],
    categories: list[str] | None = None,
    tags: list[str] | None = None,
) -> str:
    """임베딩에 넣을 의미 텍스트.

    판매 순위(`steam_rank`)는 일부러 넣지 않는다. 인기도는 의미 신호가 아니라 랭킹
    신호이고, R1 이 이미 `recommendations_percentile` 로 랭킹 단계에서 다룬다.

    **`Tags` 가 왜 Description 바로 뒤인가** — 판정 580쌍 중 실패 235건의 77%가 표현
    실패였다(GENRE_ONLY 38% · IRRELEVANT 30% · KEYWORD_MATCH 9%). 원인은 `genres` 의
    변별력이 없다는 것이다 — '액션'은 코퍼스의 42%, '인디'는 72%가 가진 값이다.

        Salt and Sanctuary  genres → 액션, 인디, RPG
                            tags   → Souls-like, Metroidvania, Dark Fantasy, 2D, Difficult
        Hollow Knight       tags   → Metroidvania, Platformer, Souls-like, Difficult

    태그로는 두 게임이 3개를 공유하고 장르로는 '액션·인디'뿐이다.

    `tags` 는 **투표 수 내림차순**으로 들어와야 한다. 상위 15개만 쓰는데, 꼬리로 갈수록
    'Singleplayer'·'Indie' 같은 일반 태그라 신호보다 희석이 크다. 실측 토큰 길이 p99 가
    277 이고 태그 15개가 ~45 토큰이라 `max_seq_length=512` 안에 들어간다.

    ────────────────────────────────────────────────────────────────────────
    2026-08-15 — **설명문은 잡음이 아니라 안정기다. 빼지 말 것.**

    가설이었다: NecroDancer 의 최근접 이웃이 리듬 게임이 아니라 던전 크롤러인 것은
    설명문이 던전 어휘에 절여져 있어서다("리듬 기반 **던전 크롤링** … 랜덤 생성
    **던전** … 스켈레톤, 좀비, 드래곤과 싸우며"). 그렇다면 쿼리 쪽만 정제해도
    재임베딩 없이 고칠 수 있다.

    검증: 시드 13개(리듬 2 · 코지 2 · 스포츠 3 · MMO 3 · JRPG 3)를 4가지 변형으로
    다시 인코딩해 기존 코퍼스에 질의했다 — 원본 / Modes 제거 / 설명문 제거 /
    태그+장르만. **가설은 틀렸다.**

        NecroDancer   원본       Crawl · Rift of the NecroDancer · Friendship Dungeon
                      Modes제거  Rift 가 1위로 (미세 개선)
                      설명문제거 Legend of Dungeon · Crawlers and Brawlers · Dungeon Souls
                                 → 리듬이 **완전히 사라진다**
        TOEM          원본       TOEM 2 · Snap Quest · Hidden Folks   (전부 정답)
                      설명문제거 Christmas Wonderland 2/16 · 틀린그림찾기 → 양산형으로 붕괴
        FIFA 22       원본       EA FC 27/26/25 · eFootball          (전부 정답)
                      설명문제거 축구 온라인: 볼3D · SHOOTER LEAGUE - ROBOT GOAL

    태그·장르만 남기면 **모든 시드에서** 유명작이 사라지고 같은 태그를 단 무명작이
    올라온다. 태그 문자열은 게임을 식별하지 않는다 — 수천 개가 같은 15개를 공유한다.
    설명문만이 개별 게임을 구별하는 유일한 자유 텍스트다.

    남은 개선 여지는 `Modes:` 줄뿐이고(효과는 미세), 그건 `PLATFORM_CATEGORY_EXACT`
    누락 — '캡션 이용 가능' · '자막 옵션' · '난이도 조정' · '키보드 전용 옵션' ·
    '스테레오 사운드' · '레벨 에디터 포함' 이 안 걸러진다. 고쳐도 **재임베딩(약 35시간)
    후에만** 반영된다.
    ────────────────────────────────────────────────────────────────────────
    """
    text = f"Description: {short_description}"
    if tags:
        text += f"\nTags: {', '.join(tags[:MAX_TAGS])}"
    if len(genres):
        text += f"\nGenres: {', '.join(genres)}"
    modes = clean_categories(categories)
    if modes:
        text += f"\nModes: {', '.join(modes)}"
    return text


def add_semantic_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_cat = "categories" in df.columns
    has_tags = "tags" in df.columns
    df["semantic_text"] = [
        build_semantic_text(d, g, categories=c, tags=t)
        for d, g, c, t in zip(
            df["short_description"],
            df["genres"],
            df["categories"] if has_cat else [None] * len(df),
            df["tags"] if has_tags else [None] * len(df),
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
