"""서빙 가능 목록 — 파싱 · 갱신 실패 의미론 · 원자적 교체 · 스레드 규칙 · /health (spec3 §10).

전부 hermetic 이다(아티팩트 없음): HTTP 는 임시 포트에 띄운 `http.server`, 파일은 tmp_path,
플랫폼 호출은 `_bind` 로 갈아 끼운 가짜 함수.
"""
from __future__ import annotations
import http.server, logging, threading, time
import numpy as np, pandas as pd, pytest

from aod_serving.engine.adapters.steam import SteamAdapter
from aod_serving.engine.adapters.tmdb import TmdbAdapter
from aod_serving.engine.adapters.webnovel import WebnovelAdapter
from aod_serving.engine.adapters.webtoon import WebtoonAdapter
from aod_serving.engine.catalog import (DISABLED, CatalogLoader, CatalogSource, fetch_text, parse_keys,
                                        source_from_env)
from aod_serving.engine.overrides import EffectiveConfig


def cfg(production=None, postprocess=None):
    return EffectiveConfig(production=production or {}, postprocess=postprocess or {}, hash="h" * 12,
                           approved=False, verdict=None)


class Spy:
    def __init__(self, frame=None): self.frame, self.args, self.kw, self.calls = frame, None, None, 0
    def __call__(self, *args, **kw):
        self.args, self.kw, self.calls = args, kw, self.calls + 1
        return self.frame if self.frame is not None else pd.DataFrame({"item_id": [], "seed_similarity": [], "final_score": []})


def steam(known=(1, 2, 3, 4, 5), keys=None):
    a = SteamAdapter("/a", cfg({"strategy": "top2_mean", "rec_boost": 0.03}))
    spy = Spy(pd.DataFrame({"steam_appid": [3], "seed_similarity": [0.5], "final_score": [0.6], "dominant_seed": [1]}))
    a._bind(spy, "C", known=set(known), first=min(known),
            corpus_keys=np.asarray(keys if keys is not None else sorted(known), dtype=np.int64))
    return a, spy


def webnovel(known=(1, 2, 3, 4, 5)):
    a = WebnovelAdapter("/a", cfg({"strategy": "top2_mean", "pop_boost": 0.0, "min_interest_count": None,
                                   "drop_excluded_series": True}))
    spy = Spy(pd.DataFrame({"item_id": [3], "seed_similarity": [0.5], "final_score": [0.6], "dominant_seed": [1]}))
    a._bind(spy, "C", known=set(known), first=min(known), episodes={}, corpus_keys=np.asarray(sorted(known), dtype=np.int64))
    return a, spy


def webtoon(known=(1, 2, 3, 4, 5)):
    prod = {"strategy": "top2_mean", "pop_boost": 0.0, "hub_lambda": 0.0, "star_boost": 0.0, "tag_w": 0.0,
            "creator_w": 0.0, "tag_drop_genre": False, "dislike_w": 0.0, "dislike_floor": 0.61}
    a = WebtoonAdapter("/a", cfg(prod, {"series_max": 1, "artist_max": 2, "drop_adult": True}))
    spy = Spy(pd.DataFrame({"item_id": [3], "seed_similarity": [0.5], "final_score": [0.6], "dominant_seed": [1]}))
    a._bind(spy, known=set(known), first=min(known))
    return a, spy


def tmdb(item_ids=("movie_1", "movie_2", "tv_3", "tv_9")):
    a = TmdbAdapter("/a", cfg({"strategy": "mean"}, {}))
    spy = Spy(pd.DataFrame({"row": [3], "item_id": ["tv_9"], "seed_similarity": [0.5], "final_score": [0.6],
                            "dominant_seed": [0]}))
    a._bind(spy, "C", item_ids=list(item_ids))
    return a, spy


# ── 파싱 ────────────────────────────────────────────────────────────────

def test_parse_keys_handles_blank_lines_crlf_whitespace_and_duplicates():
    assert parse_keys("730\r\n\r\n  570  \n730\n\n10\n") == ["730", "570", "10"]
    assert parse_keys("") == [] and parse_keys("\n \n\r\n") == []


def test_unknown_keys_are_ignored_and_counted_separately():
    a, _ = steam(known=(1, 2, 3))
    assert a.set_catalog(["1", "3", "999", "abc", ""]) == (5, 2)     # 받은 키 5 · 코퍼스에 있는 것 2
    assert a._catalog.size == 5 and a._catalog.matched == 2


def test_duplicate_keys_are_counted_once():
    a, _ = steam(known=(1, 2, 3))
    assert a.set_catalog(["1", "1", "01", "2"]) == (3, 2)            # "1"·"01" 은 같은 코퍼스 키다


def test_tmdb_parses_movie_and_tv_prefixed_keys():
    a, _ = tmdb()
    assert a.set_catalog(["movie_1", "tv_9", "movie_404", "9"]) == (4, 2)
    assert list(a._catalog.restrict) == [True, False, False, True]   # 행 0(movie_1) · 3(tv_9) 만 서빙 가능


# ── 제외 구조 ────────────────────────────────────────────────────────────

def test_row_mask_platforms_block_everything_outside_the_catalog():
    for make in (steam, webnovel):
        a, _ = make(known=(1, 2, 3, 4, 5))
        a.set_catalog(["2", "4"])
        assert list(a._catalog.restrict) == [True, False, True, False, True]   # True = 후보에서 뺀다


def test_row_mask_follows_corpus_row_order_not_sorted_keys():
    """`blocked_rows` 는 `full_corpus_frame()` 의 **행 순서**에 맞아야 한다 — 키를 정렬한 순서가 아니다."""
    a, _ = steam(known=(10, 20, 30), keys=[30, 10, 20])
    a.set_catalog(["10"])
    assert list(a._catalog.restrict) == [True, False, True]


def test_webtoon_blocks_with_a_frozenset_of_ids():
    a, _ = webtoon(known=(1, 2, 3))
    a.set_catalog(["2"])
    assert a._catalog.restrict == frozenset({1, 3}) and isinstance(a._catalog.restrict, frozenset)


def test_restrict_structures_are_read_only():
    for make in (steam, webnovel, tmdb):
        a, _ = make()
        a.set_catalog(["2"] if make is not tmdb else ["movie_1"])
        r = a._catalog.restrict
        with pytest.raises(ValueError):
            r[0] = True                       # 불변 — 요청 스레드와 갱신 스레드가 같이 봐도 안전하다


def test_empty_catalog_blocks_everything_and_is_a_valid_catalog():
    a, _ = steam(known=(1, 2, 3))
    assert a.set_catalog([]) == (0, 0)
    assert a._catalog is not None and list(a._catalog.restrict) == [True, True, True]


def test_set_catalog_none_turns_the_feature_off():
    a, _ = steam()
    a.set_catalog(["1"]); assert a._catalog is not None
    assert a.set_catalog(None) == (0, 0) and a._catalog is None


# ── 어댑터 → 플랫폼 호출 ─────────────────────────────────────────────────

def test_platform_call_has_no_extra_argument_while_the_feature_is_off():
    """꺼져 있으면 평가 경로 호출 모양이 예전 그대로여야 한다(결과 불변의 전제)."""
    a, spy = steam(); a.recommend(k=1, seeds=["1"])
    assert "blocked_rows" not in spy.kw
    a, spy = tmdb(); a.recommend(k=1, seeds=["movie_1"])
    assert "servable_rows" not in spy.kw
    a, spy = webtoon(); a.recommend(k=1, seeds=["1"])
    assert spy.kw["seen"] == []


def test_blocked_rows_reach_the_platform_call_when_the_catalog_is_on():
    a, spy = steam(known=(1, 2, 3)); a.set_catalog(["2"])
    a.recommend(k=1, seeds=["1"])
    assert list(spy.kw["blocked_rows"]) == [True, False, True]
    a, spy = webnovel(known=(1, 2, 3)); a.set_catalog(["2"])
    a.recommend(k=1, seeds=["1"])
    assert list(spy.kw["blocked_rows"]) == [True, False, True]
    a, spy = tmdb(); a.set_catalog(["tv_9"])
    a.recommend(k=1, seeds=["movie_1"])
    assert list(spy.kw["servable_rows"]) == [False, False, False, True]


def test_webtoon_merges_blocked_ids_into_seen_without_platform_changes():
    a, spy = webtoon(known=(1, 2, 3, 4)); a.set_catalog(["2", "4"])
    a.recommend(k=1, seeds=["1"], seen=["3"])
    assert spy.kw["seen"] == [1, 3]           # 목록 밖(1·3) + 이미 본 것(3) — 시드 1 은 제외돼도 시드로는 쓰인다


def test_seeds_are_not_filtered_by_the_catalog():
    """목록에 없는 작품도 유효한 시드다 — 좋아요를 목록 문제로 무효화하지 않는다."""
    a, spy = steam(known=(1, 2, 3)); a.set_catalog(["2"])
    r = a.recommend(k=1, seeds=["1", "3"])
    assert spy.args == ([1, 3],) and r.dropped_seeds == []


def test_one_request_sees_one_catalog_even_if_it_is_swapped_mid_call():
    """요청은 목록 속성을 한 번만 읽는다 — 계산 도중 갱신돼도 섞이지 않는다."""
    a, spy = steam(known=(1, 2, 3)); a.set_catalog(["2"])
    first = a._catalog.restrict

    def swap(*args, **kw):
        a.set_catalog(["3"])                  # 플랫폼 호출 도중에 갱신이 들어온다
        assert list(kw["blocked_rows"]) == list(first)
        return pd.DataFrame({"steam_appid": [], "seed_similarity": [], "final_score": []})

    a._fn = swap
    a.recommend(k=1, seeds=["1"])
    assert list(a._catalog.restrict) == [True, True, False]   # 다음 요청부터 새 목록


def test_catalog_swap_is_atomic_no_half_built_state_is_visible(monkeypatch):
    a, _ = steam(known=(1, 2, 3)); a.set_catalog(["1"])
    old = a._catalog
    seen_mid = []
    real = type(a)._restrict_from

    def slow(self, allowed):
        seen_mid.append(self._catalog)        # 구조를 짓는 도중에 보이는 것
        return real(self, allowed)

    monkeypatch.setattr(type(a), "_restrict_from", slow)
    a.set_catalog(["2"])
    assert seen_mid == [old] and a._catalog is not old      # 짓는 동안은 계속 예전 목록


# ── 출처 설정 ────────────────────────────────────────────────────────────

def test_source_from_env_off_by_default():
    assert source_from_env({}) is None and source_from_env({"CATALOG_KEYS_URL": "", "CATALOG_KEYS_FILE": " "}) is None


def test_source_from_env_url_wins_and_refresh_defaults():
    s = source_from_env({"CATALOG_KEYS_URL": "http://x/k", "CATALOG_KEYS_FILE": "/tmp/k.txt"})
    assert (s.kind, s.target, s.refresh_s, s.timeout_s) == ("url", "http://x/k", 600.0, 5.0)
    assert source_from_env({"CATALOG_KEYS_FILE": "/tmp/k.txt"}).kind == "file"
    assert source_from_env({"CATALOG_KEYS_URL": "http://x/k", "CATALOG_REFRESH_S": "30"}).refresh_s == 30.0


@pytest.mark.parametrize("bad", ["", "abc", "0", "-5"])
def test_bad_refresh_falls_back_to_the_default(bad):
    assert source_from_env({"CATALOG_KEYS_URL": "http://x/k", "CATALOG_REFRESH_S": bad}).refresh_s == 600.0


def test_file_source_reads_the_same_format(tmp_path):
    p = tmp_path / "keys.txt"; p.write_text("1\r\n\n2\n", encoding="utf-8")
    assert parse_keys(fetch_text(CatalogSource("file", str(p)))) == ["1", "2"]


# ── HTTP ────────────────────────────────────────────────────────────────

class _Serve(http.server.BaseHTTPRequestHandler):
    body, status, ctype = b"1\n2\n", 200, "text/plain; charset=utf-8"

    def do_GET(self):                                     # noqa: N802 — http.server 규약
        cls = type(self)
        self.send_response(cls.status)
        self.send_header("Content-Type", cls.ctype)
        self.send_header("Content-Length", str(len(cls.body)))
        self.end_headers(); self.wfile.write(cls.body)

    def log_message(self, *a): pass


@pytest.fixture()
def http_keys():
    """임시 포트에 목록 서버 하나. `set(body=…, status=…, ctype=…)` 로 응답을 바꾼다."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Serve)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    class H:
        url = f"http://127.0.0.1:{srv.server_address[1]}/api/recommendations/catalog-keys?platform=steam"
        @staticmethod
        def set(body=None, status=None, ctype=None):
            if body is not None: _Serve.body = body
            if status is not None: _Serve.status = status
            if ctype is not None: _Serve.ctype = ctype
    H.set(body=b"1\n2\n", status=200, ctype="text/plain; charset=utf-8")
    yield H
    srv.shutdown(); srv.server_close()


def test_http_source_loads_keys_end_to_end(http_keys):
    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    assert ld.refresh_once() is True
    assert ld.health()["size"] == 2 and ld.health()["matched"] == 2 and ld.health()["source"] == "url"
    assert list(a._catalog.restrict) == [False, False, True]


def test_http_404_never_loaded_serves_unfiltered_and_records_last_error(http_keys, caplog):
    http_keys.set(status=404)
    a, spy = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    with caplog.at_level(logging.WARNING, logger="aod.engine.catalog"):
        assert ld.refresh_once() is False
    h = ld.health()
    assert h["enabled"] is True and h["loaded_at"] is None and h["size"] is None and "404" in h["last_error"]
    a.recommend(k=1, seeds=["1"])
    assert "blocked_rows" not in spy.kw                      # 필터 없이 서빙한다
    assert any("한 번도 받은 적이 없다" in r.getMessage() for r in caplog.records)


def test_refresh_failure_keeps_the_previous_catalog(http_keys, caplog):
    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    ld.refresh_once()
    good, loaded_at = a._catalog, ld.health()["loaded_at"]
    http_keys.set(status=500)
    with caplog.at_level(logging.WARNING, logger="aod.engine.catalog"):
        assert ld.refresh_once() is False
    assert a._catalog is good                                 # 직전 목록 그대로
    h = ld.health()
    assert h["loaded_at"] == loaded_at and h["size"] == 2 and "500" in h["last_error"]
    assert any("직전 목록을 유지한다" in r.getMessage() for r in caplog.records)


def test_a_successful_refresh_clears_the_previous_error(http_keys):
    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    http_keys.set(status=500); ld.refresh_once()
    http_keys.set(status=200, body=b"3\n"); assert ld.refresh_once() is True
    assert ld.health()["last_error"] is None and list(a._catalog.restrict) == [True, True, False]


def test_empty_response_is_a_valid_catalog_and_is_logged_loudly(http_keys, caplog):
    http_keys.set(body=b"\n\n")
    a, spy = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    with caplog.at_level(logging.INFO, logger="aod.engine.catalog"):
        assert ld.refresh_once() is True
    assert ld.health()["size"] == 0 and ld.health()["last_error"] is None
    a.recommend(k=1, seeds=["1"]); assert list(spy.kw["blocked_rows"]) == [True, True, True]
    loud = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(loud) == 1 and "비어 있다" in loud[0].getMessage()


def test_html_response_is_refused_instead_of_blocking_everything(http_keys):
    """주소를 잘못 잡아 엉뚱한 서비스가 200 + HTML 을 주면 그 본문이 "키"로 파싱돼 전부 차단이 된다.
    목록이 **비어 있는 것**(의도)과 응답이 **목록이 아닌 것**(사고)을 가른다."""
    http_keys.set(body=b"<html><body>404</body></html>", ctype="text/html; charset=utf-8")
    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    assert ld.refresh_once() is False and "text/html" in ld.health()["last_error"]
    assert a._catalog is None                       # 목록이 안 걸린다 → 필터 없이 서빙


def test_zero_overlap_with_the_corpus_is_applied_but_logged_loudly(http_keys, caplog):
    http_keys.set(body=b"999\n888\n")
    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url)); ld.bind(a)
    with caplog.at_level(logging.INFO, logger="aod.engine.catalog"):
        assert ld.refresh_once() is True
    assert (ld.health()["size"], ld.health()["matched"]) == (2, 0)
    loud = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(loud) == 1 and "하나도" in loud[0].getMessage()


def test_loader_without_an_adapter_fails_softly():
    ld = CatalogLoader(CatalogSource("file", "/nowhere/keys.txt"))
    assert ld.refresh_once() is False and "어댑터" in ld.health()["last_error"]


def test_timer_thread_refreshes_and_stops():
    calls = []

    def fetch(src):
        calls.append(1); return "2\n"

    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("file", "x", refresh_s=0.02), fetcher=fetch); ld.bind(a)
    ld.start(); ld.start()                       # 두 번 불러도 스레드는 하나
    for _ in range(200):
        if len(calls) >= 2: break
        time.sleep(0.01)
    ld.stop()
    assert len(calls) >= 2
    n = len(calls); time.sleep(0.08)
    assert len(calls) == n                       # 멈춘 뒤에는 더 안 돈다


# ── 스레드 규칙: 갱신 경로는 pandas·pyarrow 를 건드리지 않는다 ───────────────────

def test_refresh_only_calls_fetch_and_set_catalog_on_the_adapter():
    """타이머 스레드가 어댑터에서 부르는 것은 `set_catalog` 하나뿐이다 — 설계로 보장한다."""
    touched = []

    class Recorder:
        def __init__(self, inner): object.__setattr__(self, "_inner", inner)
        def __getattr__(self, name):
            touched.append(name); return getattr(object.__getattribute__(self, "_inner"), name)

    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("file", "x"), fetcher=lambda s: "1\n"); ld.bind(Recorder(a))
    assert ld.refresh_once() is True and touched == ["set_catalog"]


@pytest.mark.parametrize("make,keys", [(steam, ["2"]), (webnovel, ["2"]), (webtoon, ["2"]), (tmdb, ["movie_1"])])
def test_set_catalog_builds_no_pandas_objects(monkeypatch, make, keys):
    """`set_catalog` 은 numpy·파이썬만 쓴다. pandas 객체를 하나라도 만들면 여기서 터진다 —
    Arrow 를 만지는 스레드를 계산 스레드 하나로 묶어 둔 계약(app.py 머리말)이 갱신 스레드 때문에
    깨지지 않게 하는 방어다(numpy 는 스레드와 무관해 안전하다)."""
    a, _ = make()                       # 어댑터·가짜 프레임은 감시 전에 만든다(적재는 계산 스레드의 몫)
    for name in ("Series", "DataFrame", "Index", "array"):
        monkeypatch.setattr(pd, name, lambda *a, **k: pytest.fail(f"set_catalog 이 pandas.{name} 를 만들었다"))
    a.set_catalog(keys)
    assert a._catalog is not None


def test_corpus_keys_snapshot_is_plain_numpy():
    a, _ = steam()
    assert type(a._corpus_keys) is np.ndarray and a._corpus_keys.dtype == np.int64


# ── 도구·테스트가 쓰는 코퍼스 키 목록 ────────────────────────────────────────

def test_corpus_keys_lists_external_keys():
    assert steam(known=(3, 1, 2))[0].corpus_keys() == ["1", "2", "3"]
    assert tmdb()[0].corpus_keys() == ["movie_1", "movie_2", "tv_3", "tv_9"]


# ── /health ─────────────────────────────────────────────────────────────

def test_health_shape_when_disabled_and_when_enabled(http_keys):
    assert DISABLED == {"enabled": False, "size": None, "matched": None, "loaded_at": None,
                        "source": None, "last_error": None}
    a, _ = steam(known=(1, 2, 3))
    ld = CatalogLoader(CatalogSource("url", http_keys.url), now=lambda: 1_700_000_000.0); ld.bind(a)
    ld.refresh_once()
    assert ld.health() == {"enabled": True, "size": 2, "matched": 2,
                           "loaded_at": "2023-11-14T22:13:20+00:00", "source": "url", "last_error": None}
    assert set(ld.health()) == set(DISABLED)          # 켜짐/꺼짐이 같은 모양이어야 한다
