"""서빙 가능 목록 — 백엔드 카탈로그에 **있는** 작품만 후보로 삼는다 (spec3 §10).

왜 필요한가. 코퍼스(Steam 173,691 · TMDB 59,780 · 웹소설 29,494 · 웹툰 3,687)는 백엔드
카탈로그보다 훨씬 크다. 로컬 개발 DB 실측으로 코퍼스 중 DB 에 있는 작품은 0.2~2.7% 라,
후보 50개를 받아도 카드로 만들 수 있는 것이 0~1개다. 재임베딩으로 코퍼스=카탈로그가 되기
전까지 엔진이 **백엔드에 있는 것만** 후보로 삼게 한다. TMDB 의 `media` 탭 필터와 같은
성격(제품이 정한 후보 범위)이고 **랭킹 공식은 건드리지 않는다.**

설정(엔진 환경변수)
  · `CATALOG_KEYS_URL`  — `GET` 해서 `text/plain`, 한 줄에 외부 키 하나 (백엔드
    `GET /api/recommendations/catalog-keys?platform=<p>`)
  · `CATALOG_KEYS_FILE` — 같은 형식의 파일 (테스트·로컬용). URL 이 있으면 URL 이 이긴다.
  · 둘 다 없으면 **기능 꺼짐** — 지금까지와 동작이 완전히 같다.
  · `CATALOG_REFRESH_S` — 갱신 주기(초, 기본 600)

실패 의미론
  · 받기 제한 시간 5초. 실패하면 **직전 목록을 그대로 유지**하고 경고를 남긴다.
  · 한 번도 못 받았으면 **필터 없이(전체 코퍼스) 서빙**하고 갱신 실패마다 경고를 남긴다 —
    백엔드가 없다고 추천이 멈추지는 않는다.
  · 성공적으로 받은 **빈 목록(0줄)은 유효한 목록**으로 받아들인다. "아직 못 받음"과
    "정말로 서빙할 게 없음"을 구별해야 하기 때문이다 — 빈 목록을 무시하면 카탈로그를 비운
    운영자가 그 사실을 영영 모른다. 대신 ERROR 로 크게 남긴다(엔진이 빈 목록을 돌려주고
    라우터에서 `exhausted` 가 된다).

스레드 규칙 — **이 모듈은 pandas·pyarrow 를 절대 건드리지 않는다.** 엔진의 Arrow 접촉
스레드는 전용 계산 스레드 하나뿐이어야 한다(`engine/app.py` 머리말의 mimalloc SIGSEGV).
여기서 하는 일은 HTTP/파일 읽기와 파이썬 문자열·집합 연산뿐이고, 어댑터의 `set_catalog`
도 `load()` 가 계산 스레드에서 떠 둔 numpy 배열·파이썬 집합만 쓴다(numpy 는 스레드와
무관하다 — 위험한 것은 Arrow 할당자다).
"""
from __future__ import annotations
import logging, os, threading, time, urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

log = logging.getLogger("aod.engine.catalog")

DEFAULT_REFRESH_S = 600.0
#: 받기 제한 시간(초). 백엔드가 느리거나 죽어 있어도 기동이 여기서 매달리지 않는다.
FETCH_TIMEOUT_S = 5.0
#: 본문 상한. 키 17만개라도 2MB 수준이라 넉넉하다 — 무한정 읽어 메모리를 먹지 않게 한다.
MAX_BYTES = 32 * 2 ** 20

#: 기능이 꺼졌을 때의 `/health.catalog`.
DISABLED: dict = {"enabled": False, "size": None, "matched": None,
                  "loaded_at": None, "source": None, "last_error": None, "blocking_all": False}


@dataclass(frozen=True)
class CatalogSource:
    kind: str                    # "url" | "file"
    target: str
    refresh_s: float = DEFAULT_REFRESH_S
    timeout_s: float = FETCH_TIMEOUT_S


def source_from_env(env: dict | None = None) -> CatalogSource | None:
    """환경변수 → 출처. 둘 다 없으면 `None`(기능 꺼짐)."""
    env = os.environ if env is None else env
    url = (env.get("CATALOG_KEYS_URL") or "").strip()
    path = (env.get("CATALOG_KEYS_FILE") or "").strip()
    if not url and not path:
        return None
    raw = (env.get("CATALOG_REFRESH_S") or "").strip()
    try:
        refresh = float(raw) if raw else DEFAULT_REFRESH_S
    except ValueError:
        log.warning("CATALOG_REFRESH_S=%r 를 숫자로 못 읽었다 — 기본값 %.0fs 를 쓴다", raw, DEFAULT_REFRESH_S)
        refresh = DEFAULT_REFRESH_S
    if refresh <= 0:
        refresh = DEFAULT_REFRESH_S
    return CatalogSource("url", url, refresh) if url else CatalogSource("file", path, refresh)


def parse_keys(text: str) -> list[str]:
    """한 줄에 키 하나. 빈 줄·CRLF·앞뒤 공백을 없애고 중복을 지우되 **순서는 지킨다**(순수 함수)."""
    return list(dict.fromkeys(k for line in text.splitlines() if (k := line.strip())))


def fetch_text(src: CatalogSource) -> str:
    """목록 원문. HTTP 상태가 200 이 아니면 `urlopen` 이 예외를 낸다(호출부가 경고로 삼킨다).

    **`text/plain` 이 아니면 거절한다.** 주소를 잘못 잡아 엉뚱한 서비스의 HTML 이 200 으로
    돌아오면, 그 본문이 "키 목록"으로 파싱돼 코퍼스와 하나도 안 겹치고 → 전부 차단이 된다.
    목록이 비어 있는 것(=의도)과 응답이 목록이 아닌 것(=사고)은 다르게 다뤄야 한다.
    """
    if src.kind == "file":
        return Path(src.target).read_text(encoding="utf-8")
    with urllib.request.urlopen(src.target, timeout=src.timeout_s) as r:   # noqa: S310 — 운영자가 준 주소다
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        body = r.read(MAX_BYTES + 1)
    if ctype and ctype != "text/plain":
        raise ValueError(f"Content-Type 이 {ctype!r} 다 — text/plain 인 키 목록이 아니다: {src.target}")
    if len(body) > MAX_BYTES:
        raise ValueError(f"목록이 너무 크다(>{MAX_BYTES}바이트) — 주소를 확인하라: {src.target}")
    return body.decode("utf-8")


class CatalogLoader:
    """목록을 받아 어댑터에 꽂는다. 기동 1회는 계산 스레드가, 이후 갱신은 타이머 스레드가 부른다.

    타이머 스레드가 하는 일은 `fetch` → `parse_keys` → `adapter.set_catalog` 셋뿐이다 —
    pandas·pyarrow 를 만지는 경로가 없다(모듈 머리말).
    """

    def __init__(self, source: CatalogSource, *, fetcher: Callable[[CatalogSource], str] = fetch_text,
                 now: Callable[[], float] = time.time):
        self.source, self._fetch, self._now = source, fetcher, now
        self._adapter = None
        self._lock = threading.Lock()          # `_state` 딕셔너리만 지킨다(목록 자체는 어댑터가 통째로 교체)
        self._state: dict = {"size": None, "matched": None, "loaded_at": None, "last_error": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def bind(self, adapter) -> None:
        """적재가 끝난 어댑터를 붙인다. 이게 없으면 `refresh_once` 는 실패로 기록된다."""
        self._adapter = adapter

    # ── 갱신 ─────────────────────────────────────────────────────────
    def refresh_once(self) -> bool:
        """한 번 받아 적용한다. **예외를 밖으로 내보내지 않는다** — 실패하면 직전 목록을 유지한다."""
        try:
            if self._adapter is None:
                raise RuntimeError("어댑터가 아직 붙지 않았다")
            size, matched = self._adapter.set_catalog(parse_keys(self._fetch(self.source)))
        except Exception as e:                  # noqa: BLE001 — 갱신 실패가 서빙을 멈추면 안 된다
            with self._lock:
                self._state["last_error"] = f"{type(e).__name__}: {e}"
                never = self._state["loaded_at"] is None
            log.warning("서빙 가능 목록 갱신 실패 (%s %s) — %s: %s · %s", self.source.kind, self.source.target,
                        type(e).__name__, e,
                        "한 번도 받은 적이 없다 → 필터 없이(전체 코퍼스) 서빙한다" if never else "직전 목록을 유지한다")
            return False
        with self._lock:
            self._state = {"size": size, "matched": matched,
                           "loaded_at": datetime.fromtimestamp(self._now(), timezone.utc).isoformat(timespec="seconds"),
                           "last_error": None}
        if size == 0:
            log.error("서빙 가능 목록이 비어 있다(0줄) — **유효한 목록으로 받아들인다**: 이 엔진은 아무것도 "
                      "추천하지 않는다(라우터에서 exhausted). 백엔드 카탈로그나 %s 를 확인하라", self.source.target)
        elif matched == 0:
            log.error("서빙 가능 목록 %d개가 코퍼스와 **하나도** 안 겹친다 — 키 형식이나 코퍼스 버전이 어긋났을 "
                      "가능성이 크다. 목록대로 전부 차단한다: %s", size, self.source.target)
        else:
            log.info("서빙 가능 목록 적용 — 키 %d개 중 코퍼스에 있는 것 %d개 (%s)", size, matched, self.source.kind)
        return True

    def start(self) -> None:
        """주기 갱신 타이머를 띄운다(데몬). 이미 떠 있으면 아무것도 하지 않는다."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="catalog-refresh", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.source.refresh_s):
            self.refresh_once()

    def stop(self) -> None:
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=2)

    # ── /health ──────────────────────────────────────────────────────
    def health(self) -> dict:
        with self._lock:
            s = dict(self._state)
        # 목록을 받은 적이 있고(size != None) 그게 0줄이거나 코퍼스와 하나도 안 겹치면 —
        # 이 엔진은 지금 **아무것도** 추천하지 않는다(전 요청이 exhausted). 운영자가 /health 만
        # 보고도 알아채야 하는 상태라 계산해서 얹어 둔다(README §6-1 — 이 필드에 알림을 건다).
        blocking_all = s["size"] is not None and (s["size"] == 0 or s["matched"] == 0)
        return {"enabled": True, "size": s["size"], "matched": s["matched"],
                "loaded_at": s["loaded_at"], "source": self.source.kind, "last_error": s["last_error"],
                "blocking_all": blocking_all}
