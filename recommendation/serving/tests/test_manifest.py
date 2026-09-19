import json
import pytest
from aod_serving.tools import make_manifest
from aod_serving.tools.make_manifest import _production as _real_production   # autouse 픽스처가 make_manifest._production
                                                                                # 이름 자체를 바꿔치기하므로, 진짜 구현을
                                                                                # 직접 테스트하려면 함수 객체를 따로 쥐고 있어야 한다


@pytest.fixture(autouse=True)
def _no_platform_import(monkeypatch):
    # 단위 테스트에서는 플랫폼 코드를 올리지 않는다(프로세스가 그 플랫폼 전용이 돼 버린다) — 실물 경로는 Step 5 가 본다
    monkeypatch.setattr(make_manifest, "_production", lambda platform, d: {})


def test_cli_writes_manifest_and_validates(corpus, monkeypatch, capsys):
    rc = make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"])
    m = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert rc == 0 and m["rows"] == 6 and m["corpus_version"] == "wn_test" and set(m["files"]) == {
        "corpus_embeddings.npy", "corpus_index.parquet", "dataset.parquet"}
    assert "검증 통과" in capsys.readouterr().out


def test_cli_fails_when_contract_is_broken(corpus, capsys):
    (corpus / "corpus_index.parquet").unlink()
    assert make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"]) == 1


def test_corpus_version_defaults_to_folder_name(corpus):
    assert make_manifest.main(["--platform", "webnovel", "--dir", str(corpus)]) == 0
    assert json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))["corpus_version"] == "wn_test"


# ── 이미 있는 manifest.json 을 덮어쓰지 않는다(§Step12) ────────────────────────────

def test_rerun_without_force_validates_against_existing_manifest_and_does_not_write(corpus, capsys):
    assert make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"]) == 0
    before = (corpus / "manifest.json").read_bytes()
    rc = make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"])
    out = capsys.readouterr().out
    assert rc == 0 and "기존 manifest 와 일치 — 검증 통과" in out
    assert (corpus / "manifest.json").read_bytes() == before          # 다시 쓰지 않았다


def test_rerun_without_force_on_mismatch_fails_mentions_force_and_does_not_overwrite(corpus, capsys):
    assert make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"]) == 0
    before = (corpus / "manifest.json").read_bytes()
    # 코퍼스 버전이 바뀌었다(예: gitignore 된 .npy 가 다른 빌드로 갈아치워짐) — 기존 manifest 와 안 맞는다
    rc = make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test_v2"])
    out = capsys.readouterr().out
    assert rc == 1 and "--force" in out
    assert (corpus / "manifest.json").read_bytes() == before          # 나쁜 상태에 기준을 맞춰 덮어쓰지 않았다


def test_force_regenerates_manifest(corpus, capsys):
    assert make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"]) == 0
    rc = make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test_v2", "--force"])
    assert rc == 0
    assert json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))["corpus_version"] == "wn_test_v2"


# ── config.json 이 깨졌으면(ConfigError) CLI 가 실패해야 한다 — _production 이 삼키면 안 된다 ──

def test_production_lets_config_error_propagate(corpus, monkeypatch):
    from aod_serving.engine.overrides import ConfigError

    def boom(platform, d):
        raise ConfigError("config.json production.pop_bost 는 허용된 키가 아니다")

    monkeypatch.setattr("aod_serving.engine.overrides.effective_config_for", boom)
    with pytest.raises(ConfigError):
        _real_production("webnovel", corpus)


def test_production_swallows_only_import_error(corpus, monkeypatch):
    def boom(platform, d):
        raise ModuleNotFoundError("src")           # 플랫폼 코드를 못 불러온다(라우터 이미지·테스트 환경)

    monkeypatch.setattr("aod_serving.engine.overrides.effective_config_for", boom)
    assert _real_production("webnovel", corpus) == {}


def test_main_fails_with_clear_message_when_config_is_invalid(corpus, monkeypatch, capsys):
    from aod_serving.engine.overrides import ConfigError

    def boom(platform, d):
        raise ConfigError("config.json production.pop_bost 는 허용된 키가 아니다")

    monkeypatch.setattr(make_manifest, "_production", boom)      # _no_platform_import 의 패치를 이 테스트에서만 덮는다
    rc = make_manifest.main(["--platform", "webnovel", "--dir", str(corpus), "--corpus-version", "wn_test"])
    assert rc == 1 and "pop_bost" in capsys.readouterr().out
