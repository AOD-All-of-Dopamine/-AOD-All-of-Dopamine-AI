"""싫어요(T-11) 계약 — 제외 · 시리즈 제외 · 무싫어요 불변 · 임계 아래 무변화 · 표적 감소."""
import json, os, sys
from pathlib import Path
import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts/wt_v1/corpus_embeddings.npy").exists(), reason="wt_v1 아티팩트 없음")


@pytest.fixture(scope="module")
def env():
    os.chdir(ROOT); sys.path.insert(0, str(ROOT)); os.environ.setdefault("AOD_WT_ARTIFACTS", str(ROOT / "artifacts/wt_v1"))
    from src.personalized_retrieve import Engine
    from src.postprocess import series_key
    eng = Engine()
    P = json.load(open(ROOT / "eval/profiles.json"))
    return dict(eng=eng, P=P, key=series_key, name=dict(zip(eng.ds["item_id"].astype(int), eng.ds["name"])))


def ids(df):
    return [int(x) for x in df["item_id"]]


def test_no_dislike_equals_graded_t10_n0(env):
    T10 = json.load(open(ROOT / "eval/t10_pages.json"))
    for p in env["P"][:8]:
        a = ids(env["eng"].next_page(p["seeds"], k=10, seen=[]))
        b = ids(env["eng"].next_page(p["seeds"], k=10, seen=[], disliked_ids=[], dislike_w=2.0))
        assert a == b == T10[f"n0|{p['pid']}"][0]


@pytest.mark.parametrize("w", [0.0, 2.0])
def test_disliked_and_its_series_never_appear(env, w):
    eng = env["eng"]
    for p in env["P"][:8]:
        d = ids(eng.next_page(p["seeds"], k=10))[0]
        out = ids(eng.next_page(p["seeds"], k=50, disliked_ids=[d], dislike_w=w))
        assert d not in out
        assert env["key"](env["name"][d]) not in {env["key"](env["name"][i]) for i in out}


def test_floor_above_all_similarity_is_no_op(env):
    """감점은 임계 이상에서만 걸린다 — 임계 1.01 이면 w 와 무관하게 w=0 과 같은 목록."""
    eng = env["eng"]
    for p in env["P"][:8]:
        d = ids(eng.next_page(p["seeds"], k=10))[0]
        base = ids(eng.next_page(p["seeds"], k=20, disliked_ids=[d], dislike_w=0.0))
        high = ids(eng.next_page(p["seeds"], k=20, disliked_ids=[d], dislike_w=5.0, dislike_floor=1.01))
        assert base == high


def test_target_dislike_pushes_neighbors_down(env):
    eng = env["eng"]; before = after = 0
    for p in env["P"][:10]:
        p1 = ids(eng.next_page(p["seeds"], k=10)); d = p1[0]
        s = eng.emb @ eng.emb[eng.row[d]]
        nb = {int(eng.ds["item_id"].iloc[t]) for t in np.argsort(-s)[1:51]}
        before += len(nb & set(ids(eng.next_page(p["seeds"], k=20, seen=p1, disliked_ids=[d], dislike_w=0.0))))
        after += len(nb & set(ids(eng.next_page(p["seeds"], k=20, seen=p1, disliked_ids=[d], dislike_w=3.0))))
    assert after < before
