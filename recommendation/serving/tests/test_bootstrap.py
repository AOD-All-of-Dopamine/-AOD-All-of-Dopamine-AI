import os, sys
import pytest
from aod_serving.engine import bootstrap


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    cwd, path = os.getcwd(), list(sys.path)
    monkeypatch.setattr(bootstrap, "_entered", None)
    yield
    os.chdir(cwd); sys.path[:] = path


def make_platform(tmp_path, name):
    root = tmp_path / name; (root / "src").mkdir(parents=True); (root / "src" / "__init__.py").write_text("")
    return root


def test_enter_sets_cwd_path_and_artifact_env(tmp_path, monkeypatch):
    monkeypatch.delenv("AOD_ARTIFACTS", raising=False)
    root = make_platform(tmp_path, "steam"); art = tmp_path / "art"; art.mkdir()
    bootstrap.enter_platform("steam", platform_root=root, artifacts=art)
    assert os.getcwd() == str(root) and sys.path[0] == str(root)
    assert os.environ["AOD_ARTIFACTS"] == str(art)


def test_webtoon_uses_its_own_env_name(tmp_path, monkeypatch):
    monkeypatch.delenv("AOD_WT_ARTIFACTS", raising=False)
    root = make_platform(tmp_path, "webtoon"); art = tmp_path / "art"; art.mkdir()
    bootstrap.enter_platform("webtoon", platform_root=root, artifacts=art)
    assert os.environ["AOD_WT_ARTIFACTS"] == str(art)


def test_second_platform_in_same_process_is_refused(tmp_path):
    a, b = make_platform(tmp_path, "steam"), make_platform(tmp_path, "tmdb")
    bootstrap.enter_platform("steam", platform_root=a, artifacts=tmp_path)
    bootstrap.enter_platform("steam", platform_root=a, artifacts=tmp_path)      # 같은 플랫폼 재진입은 허용
    with pytest.raises(RuntimeError, match="steam"):
        bootstrap.enter_platform("tmdb", platform_root=b, artifacts=tmp_path)


def test_unknown_platform():
    with pytest.raises(ValueError):
        bootstrap.enter_platform("netflix")
