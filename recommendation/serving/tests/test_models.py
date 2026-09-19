import pytest
from pydantic import ValidationError
from aod_serving.common.models import (EngineRequest, EngineResponse, EngineItem, Score, VersionInfo, RouterRequest,
                                       MAX_SEEDS, MAX_EXCLUSIONS)


def test_engine_request_defaults_and_limits():
    r = EngineRequest(k=50, seeds=["730"])
    assert (r.disliked, r.excluded, r.seen, r.media) == ([], [], [], None)
    with pytest.raises(ValidationError): EngineRequest(k=0, seeds=[])
    with pytest.raises(ValidationError): EngineRequest(k=101, seeds=[])
    with pytest.raises(ValidationError): EngineRequest(k=10, seeds=[str(i) for i in range(51)])
    with pytest.raises(ValidationError): EngineRequest(k=10, seeds=[], seen=[str(i) for i in range(5001)])
    with pytest.raises(ValidationError): EngineRequest(k=10, seeds=[], media="anime")


def test_combined_exclusion_limit_counts_all_three_lists():
    big = [str(i) for i in range(2000)]
    with pytest.raises(ValidationError):
        EngineRequest(k=10, seeds=[], disliked=big, excluded=big, seen=big)


def test_keys_must_be_strings_not_numbers():
    with pytest.raises(ValidationError): EngineRequest(k=10, seeds=[730])


def test_response_serializes_camel_case():
    resp = EngineResponse(platform="steam", exhausted=False, dropped_seeds=["9"], factor_schema="steam.v1",
                          version=VersionInfo(sha="abc", config="def", corpus="tags_full"), took_ms=12,
                          items=[EngineItem(key="240", rank=0, dominant_seed="730",
                                            score=Score(final=1.19, sim=0.58, factors={"quality": 1.0}))])
    d = resp.model_dump(by_alias=True)
    assert d["droppedSeeds"] == ["9"] and d["factorSchema"] == "steam.v1" and d["tookMs"] == 12
    assert d["items"][0]["dominantSeed"] == "730" and d["items"][0]["episodeCount"] is None


def test_router_request_shape():
    r = RouterRequest.model_validate({"tab": "all", "k": 20, "buffer": 10,
                                      "seeds": {"steam": ["730"], "tmdb": ["movie_603"]}})
    assert r.seeds["tmdb"] == ["movie_603"] and r.seen == {} and r.buffer == 10
    with pytest.raises(ValidationError): RouterRequest.model_validate({"tab": "anime", "seeds": {}})
    with pytest.raises(ValidationError): RouterRequest.model_validate({"tab": "all", "seeds": {"netflix": ["1"]}})


def test_router_request_seed_limit_per_platform():
    ok = RouterRequest.model_validate({"tab": "game", "seeds": {"steam": [str(i) for i in range(MAX_SEEDS)]}})
    assert len(ok.seeds["steam"]) == MAX_SEEDS
    with pytest.raises(ValidationError):
        RouterRequest.model_validate({"tab": "game", "seeds": {"steam": [str(i) for i in range(MAX_SEEDS + 1)]}})


def test_router_request_seed_limit_is_per_platform_not_combined():
    # 두 플랫폼 각각 한도(50)까지 — 합쳐서 100 이어도 OK
    RouterRequest.model_validate({"tab": "all", "seeds": {"steam": [str(i) for i in range(MAX_SEEDS)],
                                                          "tmdb": [str(i) for i in range(MAX_SEEDS)]}})


def test_router_request_seed_limit_error_names_platform_and_limit():
    with pytest.raises(ValidationError) as ex:
        RouterRequest.model_validate({"tab": "game", "seeds": {"steam": [str(i) for i in range(MAX_SEEDS + 1)]}})
    msg = str(ex.value)
    assert "steam" in msg and str(MAX_SEEDS) in msg


def test_router_request_combined_exclusion_limit_per_platform():
    ok = RouterRequest.model_validate({"tab": "game", "seeds": {"steam": ["1"]},
                                       "disliked": {"steam": [str(i) for i in range(MAX_EXCLUSIONS)]}})
    assert len(ok.disliked["steam"]) == MAX_EXCLUSIONS
    with pytest.raises(ValidationError):
        RouterRequest.model_validate({"tab": "game", "seeds": {"steam": ["1"]},
                                      "disliked": {"steam": [str(i) for i in range(MAX_EXCLUSIONS + 1)]}})


def test_router_request_combined_exclusion_limit_counts_all_three_lists():
    big = [str(i) for i in range(2000)]
    with pytest.raises(ValidationError):
        RouterRequest.model_validate({"tab": "game", "seeds": {"steam": ["1"]}, "disliked": {"steam": big},
                                      "excluded": {"steam": big}, "seen": {"steam": big}})


def test_router_request_exclusion_limit_error_names_platform_and_limit():
    with pytest.raises(ValidationError) as ex:
        RouterRequest.model_validate({"tab": "game", "seeds": {"steam": ["1"]},
                                      "disliked": {"steam": [str(i) for i in range(MAX_EXCLUSIONS + 1)]}})
    msg = str(ex.value)
    assert "steam" in msg and str(MAX_EXCLUSIONS) in msg
