"""라우터 핵심 — 탭 분기 · 엔진 동시 호출 · 부분 응답 · 전체 탭 혼합 (REC_TAB_DESIGN §4-2·§8-4)."""
from __future__ import annotations
import asyncio, logging
from typing import Awaitable, Callable

from aod_serving.common.models import (MAX_K, EngineRequest, EngineResponse, RouterItem, RouterRequest, RouterResponse,
                                       RouterVersions)
from aod_serving.router.mixing import ALL_TAB_PLATFORMS, mix_all

log = logging.getLogger("aod.router")

#: 전체 탭에서 플랫폼마다 받는 개수 — M6 이 평가된 깊이(§8-4)
ALL_TAB_ENGINE_K = 50
#: 탭 → [(플랫폼, media)]
TAB_PLAN: dict[str, list[tuple[str, str | None]]] = {
    "all": [(p, None) for p in ALL_TAB_PLATFORMS],
    "game": [("steam", None)], "movie": [("tmdb", "movie")], "tv": [("tmdb", "tv")],
    "webtoon": [("webtoon", None)], "webnovel": [("webnovel", None)],
}
EngineCall = Callable[[str, EngineRequest], Awaitable[EngineResponse]]


class EnginesUnavailable(Exception):
    """부른 엔진이 전부 실패했다 — 백엔드는 대체 목록으로 간다."""
    def __init__(self, partial: list[str]):
        super().__init__(f"engines unavailable: {partial}"); self.partial = partial


async def recommend(req: RouterRequest, call: EngineCall, *, router_sha: str) -> RouterResponse:
    plan = TAB_PLAN[req.tab]
    want = req.k + req.buffer
    engine_k = ALL_TAB_ENGINE_K if req.tab == "all" else min(want, MAX_K)
    called = [(p, m) for p, m in plan if req.seeds.get(p)]
    exhausted = {p: True for p, _ in plan if not req.seeds.get(p)}

    async def one(p: str, media: str | None):
        try:
            # EngineRequest 생성도 try 안에서 — RouterRequest 가 이미 같은 한도를 검사하지만(422),
            # 여기서도 실패하면 그 플랫폼만 partial 로 빠지게 방어적 이중화를 둔다(요청 전체 500 방지).
            r = EngineRequest(k=engine_k, seeds=req.seeds[p], disliked=req.disliked.get(p, []),
                              excluded=req.excluded.get(p, []), seen=req.seen.get(p, []), media=media)
            return p, await call(p, r)
        except Exception as e:                       # noqa: BLE001 — 어떤 실패든 그 플랫폼만 뺀다
            log.warning("엔진 %s 실패: %s: %s", p, type(e).__name__, e)
            return p, None

    results = dict(await asyncio.gather(*(one(p, m) for p, m in called)))
    partial = [p for p, r in results.items() if r is None]
    ok = {p: r for p, r in results.items() if r is not None}
    if called and not ok:
        raise EnginesUnavailable(partial)

    exhausted |= {p: r.exhausted for p, r in ok.items()}
    dropped = {p: r.dropped_seeds for p, r in ok.items() if r.dropped_seeds}
    if req.tab == "all":
        # 엔진이 보고한 실제 사용 시드 수를 그대로 믿는다 — 원본 요청 문자열을 라우터가 다시 세면
        # 어댑터의 파싱 후 중복 제거(예: "7"·"007" 이 같은 코퍼스 키)와 어긋난다(M6 쿼터가 틀어진다).
        n_seeds = {p: r.used_seeds for p, r in ok.items()}
        lists = {p: r.items for p, r in ok.items()}
        # M6 의 플랫폼 쿼터는 k 의 함수다 — k+buffer 로 섞고 앞 k 개를 자르면 평가된 M6@k 와 다른 목록이 된다
        # (시드 조합 3개 모두 상위 20 이 달랐다, REC_TAB_DESIGN 부록 D A5). 그래서 **앞 k 개는 M6@k 그대로** 두고,
        # 버퍼(백엔드가 DB 에 없는·성인 작품을 뺀 자리를 채우는 여분)는 M6@(k+buffer) 순서에서 아직 안 나온 것으로 잇는다.
        head = mix_all(lists, n_seeds, k=req.k)
        used = {(p, i.key) for p, i in head}
        tail = [(p, i) for p, i in mix_all(lists, n_seeds, k=want) if (p, i.key) not in used]
        ordered = head + tail[: want - len(head)]
    else:
        ordered = [(p, i) for p, r in ok.items() for i in r.items][:want]

    items = [RouterItem(platform=p, key=i.key, rank=n, dominant_seed=i.dominant_seed, score=i.score,
                        factor_schema=ok[p].factor_schema) for n, (p, i) in enumerate(ordered)]
    return RouterResponse(items=items, exhausted=exhausted, dropped_seeds=dropped, partial=partial,
                          versions=RouterVersions(router=router_sha, engines={p: r.version for p, r in ok.items()}))
