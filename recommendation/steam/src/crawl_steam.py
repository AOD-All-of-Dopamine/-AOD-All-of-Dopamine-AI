# src/crawl_steam.py
"""Steam 전체 게임을 크롤링해 `steam_games.jsonl` 을 만든다.

백엔드의 Java 크롤러(`-AOD-All-of-Dopamine-crawler`)는 Postgres 에 적재하는데, 이 파이프라인은
jsonl 을 읽는다. 같은 API·같은 필드를 쓰되 출력만 jsonl 로 맞춘 얇은 크롤러다.
필드 목록은 백엔드의 `rules/game/steam.yml` 매핑을 따랐다.

**이어받기(resume)가 기본이다.** 17만 건은 수십 시간이 걸려 한 번에 끝나지 않는다.
매 건을 즉시 append 하고, 재시작하면 이미 받은 appid 를 건너뛴다.

    python -m src.crawl_steam --out data/steam_games.jsonl [--limit N] [--rps 1.5] [--reviews]

`--reviews` 는 appreviews 요약(total_positive/negative/reviews)을 함께 받는다. 요청이 2배가
되지만 지금 쓰는 recommendations.total 보다 나은 품질 신호다.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from src.config import PROJECT_ROOT

APPLIST_URL = "https://api.steampowered.com/IStoreService/GetAppList/v1/"
DETAILS_URL = "https://store.steampowered.com/api/appdetails"
REVIEWS_URL = "https://store.steampowered.com/appreviews/{appid}"

# steam.yml 매핑이 쓰는 필드 + 기존 파이프라인(data_loader)이 쓰는 필드
KEEP_FIELDS = (
    "type", "name", "steam_appid", "short_description", "detailed_description",
    "release_date", "developers", "publishers", "platforms", "genres", "categories",
    "metacritic", "recommendations", "price_overview", "is_free", "required_age",
    "header_image",
)


def load_api_key() -> str:
    if key := os.environ.get("STEAM_API_KEY"):
        return key
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("STEAM_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("STEAM_API_KEY 가 없습니다 (.env 또는 환경변수)")


def fetch_all_game_ids(key: str) -> list[int]:
    """IStoreService/GetAppList — 게임만, 페이지네이션."""
    ids, last = [], None
    while True:
        params = {
            "key": key, "include_games": "true", "include_dlc": "false",
            "include_software": "false", "include_videos": "false",
            "include_hardware": "false", "max_results": 50000,
        }
        if last:
            params["last_appid"] = last
        r = requests.get(APPLIST_URL, params=params, timeout=60)
        r.raise_for_status()
        resp = r.json().get("response", {})
        ids.extend(a["appid"] for a in resp.get("apps", []))
        if not resp.get("have_more_results"):
            break
        last = resp.get("last_appid")
    return ids


def already_done(path: Path) -> set[int]:
    """이어받기 — 이미 저장한 appid. 깨진 줄은 건너뛴다."""
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                aid = json.loads(line).get("steam_appid")
                if isinstance(aid, int):
                    done.add(aid)
            except json.JSONDecodeError:
                continue
    return done


def fetch_detail(appid: int, lang: str = "korean") -> dict | None:
    r = requests.get(DETAILS_URL, params={"appids": appid, "l": lang}, timeout=30)
    if r.status_code == 429:
        raise RateLimited()
    r.raise_for_status()
    node = r.json().get(str(appid)) or {}
    if not node.get("success"):
        return None
    data = node.get("data") or {}
    return {k: data[k] for k in KEEP_FIELDS if k in data}


def fetch_review_summary(appid: int) -> dict | None:
    r = requests.get(
        REVIEWS_URL.format(appid=appid),
        params={"json": 1, "language": "all", "purchase_type": "all", "num_per_page": 0},
        timeout=30,
    )
    if r.status_code == 429:
        raise RateLimited()
    if r.status_code != 200:
        return None
    s = r.json().get("query_summary") or {}
    s.pop("num_reviews", None)  # num_per_page=0 이라 항상 0인 노이즈
    return s or None


class RateLimited(Exception):
    pass


class AdaptiveRate:
    """429 를 맞으면 느려지고, 잘 되면 조금씩 빨라진다.

    Steam appdetails 의 실제 한계는 문서화돼 있지 않고 시간대·IP 에 따라 달라진다.
    실측(2026-07): 1.19 req/s 로 168요청 만에 429. 고정 속도로 돌리면 계속 429 를 맞고
    백오프에 시간을 버리므로, 실제 한계를 스스로 찾아가게 한다.
    """

    def __init__(self, target_rps: float, min_rps: float = 0.3):
        self.delay = 1.0 / target_rps
        self.min_delay = 1.0 / target_rps
        self.max_delay = 1.0 / min_rps
        self.ok_streak = 0
        self.n_429 = 0

    def on_success(self) -> None:
        self.ok_streak += 1
        # 200회 연속 성공하면 5% 빨라진다 (한계가 풀렸을 수 있으므로)
        if self.ok_streak >= 200 and self.delay > self.min_delay:
            self.delay = max(self.min_delay, self.delay * 0.95)
            self.ok_streak = 0

    def on_429(self) -> float:
        """반환값: 이번에 쉴 시간(초)."""
        self.n_429 += 1
        self.ok_streak = 0
        self.delay = min(self.max_delay, self.delay * 1.5)
        return min(120.0, 15.0 * min(self.n_429, 4))  # 15/30/45/60초 대기

    @property
    def rps(self) -> float:
        return 1.0 / self.delay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/steam_games.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (테스트용)")
    ap.add_argument("--rps", type=float, default=1.5, help="초당 요청 수 목표")
    ap.add_argument("--reviews", action="store_true", help="리뷰 요약도 수집 (요청 2배)")
    ap.add_argument("--lang", default="korean")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    key = load_api_key()
    print("게임 앱 목록 수집 중...", flush=True)
    ids = fetch_all_game_ids(key)
    done = already_done(out)
    todo = [i for i in ids if i not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"전체 {len(ids):,} | 이미 받음 {len(done):,} | 이번에 받을 것 {len(todo):,}", flush=True)
    if not todo:
        return

    rate = AdaptiveRate(args.rps)
    t0 = time.time()
    saved = skipped = failed = 0
    n = 0

    with open(out, "a", encoding="utf-8") as f:
        queue = list(todo)
        while queue:
            appid = queue.pop(0)
            n += 1
            try:
                detail = fetch_detail(appid, args.lang)
                if detail is None:
                    skipped += 1
                else:
                    if args.reviews:
                        time.sleep(rate.delay)
                        try:
                            if rs := fetch_review_summary(appid):
                                detail["review_summary"] = rs
                        except RateLimited:
                            pass  # 리뷰는 있으면 좋은 것 — 없어도 진행
                        except Exception:
                            pass
                    detail.setdefault("steam_appid", appid)
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    saved += 1
                rate.on_success()
            except RateLimited:
                wait = rate.on_429()
                print(f"  429 (#{rate.n_429}) — {wait:.0f}초 대기, 속도 {rate.rps:.2f} req/s 로 조정 "
                      f"| 진행 {n:,}/{len(todo):,}", flush=True)
                f.flush()
                time.sleep(wait)
                queue.insert(0, appid)  # 실패한 건 다시 시도
                n -= 1
                continue
            except Exception as e:
                failed += 1
                if failed <= 20:
                    print(f"  실패 appid={appid}: {type(e).__name__}", flush=True)

            if n % 500 == 0:
                el = time.time() - t0
                obs = n / el
                eta = (len(todo) - n) / obs / 3600 if obs else 0
                f.flush()
                print(f"  {n:,}/{len(todo):,}  저장 {saved:,} 제외 {skipped:,} 실패 {failed:,} "
                      f"429 {rate.n_429}  | 실측 {obs:.2f} req/s | 남은 {eta:.1f}h", flush=True)
            time.sleep(rate.delay)

    el = time.time() - t0
    print(f"\n완료: 저장 {saved:,} / 제외(비게임·미공개) {skipped:,} / 실패 {failed:,} / 429 {rate.n_429}")
    print(f"소요 {el / 3600:.2f}시간 → {out}")


if __name__ == "__main__":
    main()
