from aod_ai.config import Settings
from aod_ai.models import ReviewSource

VANE_SYSTEM_INSTRUCTIONS = (
    "독자 반응과 재미 요소 위주로 실제 리뷰·후기의 원본 표현을 최대한 보존해 수집하라. "
    "합성 요약을 만들지 말고, 독자들이 재미있다고 느낀 지점의 원문 스니펫을 그대로 전달하라."
)

# 스펙 §4②의 {도메인} 라벨. 새 도메인은 여기에 한 줄만 추가하면 확장(쿼리 문자열 수정 불필요).
_DOMAIN_LABEL = {
    "WEBNOVEL": "웹소설",
    "WEBTOON": "웹툰",
    "MOVIE": "영화",
    "TV": "드라마",
    "GAME": "게임",
}


def collect_reviews(vane, target) -> list[ReviewSource]:
    label = _DOMAIN_LABEL[target.domain]
    query = f'"{target.master_title}" {label} 리뷰 후기 재미'
    raw = vane.search(
        query=query,
        sources=["web", "discussions"],
        system_instructions=VANE_SYSTEM_INSTRUCTIONS,
    )
    max_sources = Settings().vane_max_sources
    seen: set[str] = set()
    out: list[ReviewSource] = []
    for src in raw:
        if src.url in seen:
            continue
        seen.add(src.url)
        out.append(src)
        if len(out) >= max_sources:
            break
    return out
