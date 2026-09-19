import json
import pytest
from aod_serving.tools import make_manifest


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
