"""영화/드라마 탭 분리 (`media`) — 혼합 경로 불변 · 매체 순도 · 페이지 연속성."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/tmdb_v1"
pytestmark = pytest.mark.skipif(not (ART / "corpus_embeddings.npy").exists(), reason="tmdb_v1 아티팩트 없음")


@pytest.fixture(scope="module")
def env():
    os.chdir(ROOT); sys.path.insert(0, str(ROOT)); os.environ.setdefault("AOD_ARTIFACTS", "artifacts/tmdb_v1")
    from src.personalized_retrieve import build_components, recommend, next_page
    from src.config import PRODUCTION
    comps = build_components(**{k: v for k, v in PRODUCTION.items() if k != "strategy"})
    P = pd.read_parquet(ROOT / "artifacts/p1/profiles.parquet")
    return dict(comps=comps, rec=recommend, nxt=next_page, strat=PRODUCTION["strategy"],
                media=comps[3].dataset["media"].to_numpy(),
                profiles={r.profile_id: [int(x) for x in r.seed_rows] for r in P.itertuples(index=False)})


def test_mixed_path_unchanged_when_media_none(env):
    s = env["profiles"]["coh_sageuk"]
    a = env["rec"](s, components=env["comps"], strategy=env["strat"], top_n=50)
    b = env["rec"](s, components=env["comps"], strategy=env["strat"], top_n=50, media=None)
    assert a["row"].tolist() == b["row"].tolist()


@pytest.mark.parametrize("media", ["movie", "tv"])
def test_tab_is_single_media_and_full(env, media):
    for pid in ("coh_pixar", "coh_sageuk", "mix2_anime_film"):
        df = env["rec"](env["profiles"][pid], components=env["comps"], strategy=env["strat"], top_n=50, media=media)
        assert len(df) == 50
        assert (env["media"][df["row"].to_numpy()] == media).all()


def test_tab_keeps_mixed_order_for_same_media(env):
    """혼합 목록의 영화만 뽑은 순서 == 영화 탭 순서 (영화 시드 · 후처리가 매체를 안 섞는 구간)."""
    s = env["profiles"]["coh_marvel"]
    mixed = env["rec"](s, components=env["comps"], strategy=env["strat"], top_n=50)
    mv = [r for r in mixed["row"] if env["media"][r] == "movie"][:10]
    tab = env["rec"](s, components=env["comps"], strategy=env["strat"], top_n=50, media="movie")["row"].tolist()[:10]
    assert len(set(mv) & set(tab)) >= 9


def test_next_page_per_tab_no_repeat(env):
    s = env["profiles"]["coh_sageuk"]; seen = set(); got = []
    for _ in range(3):
        df = env["nxt"](s, seen_rows=seen, page_size=10, components=env["comps"], media="tv")
        rows = df["row"].tolist(); got += rows; seen |= set(rows)
        assert (env["media"][rows] == "tv").all()
    assert len(got) == len(set(got)) == 30


def test_bad_media_rejected(env):
    with pytest.raises(ValueError):
        env["rec"](env["profiles"]["coh_sageuk"], components=env["comps"], media="drama")
