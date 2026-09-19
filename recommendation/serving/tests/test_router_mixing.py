import pytest
from aod_serving.common.models import EngineItem, Score
from aod_serving.router.mixing import load_m6, mix_all


def items(prefix, n, episodes=None):
    return [EngineItem(key=f"{prefix}{i}", rank=i, score=Score(final=1 - i / 100, sim=0.5),
                       episode_count=(episodes[i] if episodes else None)) for i in range(n)]


def test_m6_is_the_evaluated_function_from_crossdomain():
    m6 = load_m6()
    assert m6.__module__.endswith("mix") and m6.__name__ == "M6"


def test_load_m6_raises_when_mix_path_is_missing(monkeypatch, tmp_path):
    """기동 시 M6 를 먼저 적재하는 것(I3, `router/app.py`)이 실패로 이어지려면 `load_m6()` 자체가
    없는 파일에서 예외를 내야 한다 — 요청까지 미루면 `/health` 는 준비됐다는데 전체 탭마다 500 이 난다."""
    monkeypatch.setenv("MIX_PATH", str(tmp_path / "missing_mix.py"))
    load_m6.cache_clear()
    try:
        with pytest.raises(Exception):
            load_m6()
    finally:
        load_m6.cache_clear()


def test_round_robin_by_seed_count_order():
    out = mix_all({"steam": items("s", 50), "tmdb": items("t", 50), "webnovel": items("w", 50, [30] * 50)},
                  {"steam": 3, "tmdb": 5, "webnovel": 2}, k=6)
    assert [(p, i.key) for p, i in out] == [("tmdb", "t0"), ("steam", "s0"), ("webnovel", "w0"),
                                            ("tmdb", "t1"), ("steam", "s1"), ("webnovel", "w1")]


def test_single_seed_platform_gets_half_quota():
    out = mix_all({"steam": items("s", 50), "tmdb": items("t", 50)}, {"steam": 1, "tmdb": 4}, k=9)
    assert sum(p == "steam" for p, _ in out) == 3 and sum(p == "tmdb" for p, _ in out) == 6


def test_webnovel_under_20_episodes_is_filtered_and_missing_counts_as_zero():
    eps = [5, 30, None, 100]
    out = mix_all({"webnovel": items("w", 4, eps)}, {"webnovel": 2}, k=10)
    assert [i.key for _, i in out] == ["w1", "w3"]


def test_exhausted_platform_leaves_slots_empty():
    out = mix_all({"steam": items("s", 1), "tmdb": items("t", 50)}, {"steam": 3, "tmdb": 3}, k=10)
    assert len(out) == 6            # steam 5칸 중 1칸만 찬다 — 채우지 않는다(D-23)


def test_no_seeds_means_empty():
    assert mix_all({}, {}, k=10) == []
