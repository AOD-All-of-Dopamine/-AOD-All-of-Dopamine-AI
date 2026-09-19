import json, pytest
from aod_serving.engine.overrides import ConfigError, EffectiveConfig, resolve_config, ALLOWED

DEFAULTS = {"strategy": "top2_mean", "pop_boost": 0.0, "min_interest_count": None, "drop_excluded_series": True,
            "rating_boost": 0.0}     # 마지막 키는 허용 목록 밖 — 결과에 들어가면 안 된다


def cfg(d, body):
    (d / "config.json").write_text(json.dumps(body), encoding="utf-8")


def resolve(d, **kw):
    return resolve_config("webnovel", d, defaults=DEFAULTS, post_defaults={}, corpus_version=kw.pop("corpus_version", "wn_v6"),
                          mode=kw.pop("mode", "dev"))


def test_no_config_file_means_production_defaults(tmp_path):
    c = resolve(tmp_path)
    assert c.production == {"strategy": "top2_mean", "pop_boost": 0.0, "min_interest_count": None, "drop_excluded_series": True}
    assert c.postprocess == {} and len(c.hash) == 12


def test_override_changes_value_and_hash(tmp_path):
    base = resolve(tmp_path).hash
    cfg(tmp_path, {"production": {"pop_boost": 0.03}})
    c = resolve(tmp_path)
    assert c.production["pop_boost"] == 0.03 and c.hash != base


def test_hash_is_stable_across_key_order(tmp_path):
    cfg(tmp_path, {"production": {"pop_boost": 0.03, "strategy": "mean"}}); a = resolve(tmp_path).hash
    cfg(tmp_path, {"production": {"strategy": "mean", "pop_boost": 0.03}}); assert resolve(tmp_path).hash == a


@pytest.mark.parametrize("body, msg", [
    ({"production": {"pop_bost": 0.03}}, "pop_bost"),                 # 오타
    ({"production": {"rating_boost": 0.1}}, "rating_boost"),          # 허용 목록 밖
    ({"production": {"pop_boost": "0.03"}}, "pop_boost"),             # 타입
    ({"production": {"drop_excluded_series": 1}}, "drop_excluded_series"),   # bool 자리에 int
    ({"postprocess": {"series_max": 2}}, "series_max"),               # 웹소설엔 postprocess 키가 없다
    ({"prod": {}}, "prod"),                                           # 최상위 키
])
def test_bad_config_is_refused(tmp_path, body, msg):
    cfg(tmp_path, body)
    with pytest.raises(ConfigError, match=msg): resolve(tmp_path)


def test_int_is_accepted_where_float_is_expected(tmp_path):
    cfg(tmp_path, {"production": {"pop_boost": 0}})
    assert resolve(tmp_path).production["pop_boost"] == 0


def test_null_allowed_only_for_nullable_keys(tmp_path):
    cfg(tmp_path, {"production": {"min_interest_count": None}}); resolve(tmp_path)
    cfg(tmp_path, {"production": {"pop_boost": None}})
    with pytest.raises(ConfigError): resolve(tmp_path)


def test_prod_mode_refuses_unapproved_new_corpus(tmp_path):
    with pytest.raises(ConfigError, match="approved"): resolve(tmp_path, corpus_version="wn_v7", mode="prod")
    cfg(tmp_path, {"production": {}, "approved": True})
    with pytest.raises(ConfigError, match="verdict"): resolve(tmp_path, corpus_version="wn_v7", mode="prod")
    cfg(tmp_path, {"production": {}, "approved": True, "verdict": {"id": "W-9", "preregister_md5": "abc"}})
    assert resolve(tmp_path, corpus_version="wn_v7", mode="prod").approved is True


def test_prod_mode_accepts_baseline_corpus_without_config(tmp_path):
    assert resolve(tmp_path, corpus_version="wn_v6", mode="prod").production["pop_boost"] == 0.0


def test_dev_mode_accepts_unapproved_new_corpus(tmp_path):
    assert resolve(tmp_path, corpus_version="wn_v7", mode="dev").approved is False


def test_allowed_tables_cover_four_platforms():
    assert set(ALLOWED) == {"steam", "tmdb", "webtoon", "webnovel"}
