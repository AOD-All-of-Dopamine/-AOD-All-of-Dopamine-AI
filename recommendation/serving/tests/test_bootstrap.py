import os, sys
import pytest
from aod_serving.engine import bootstrap


_ARTIFACT_ENV_NAMES = ("AOD_ARTIFACTS", "AOD_WT_ARTIFACTS")


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    cwd, path = os.getcwd(), list(sys.path)
    monkeypatch.setattr(bootstrap, "_entered", None)
    # `enter_platform` 은 os.environ 을 monkeypatch 를 거치지 않고 직접 쓴다. 테스트가 시작하기 전에
    # 그 변수가 아예 없었으면(흔한 경우) `monkeypatch.delenv(..., raising=False)` 는 "원래 없었다"는
    # 사실을 기록하지 못해 — enter_platform 이 심어 놓은 값이 다음 테스트로 샌다(삭제된 tmp_path 를
    # 가리키는 채로). 여기서 직접 저장·복원해 확실히 지운다.
    saved = {name: os.environ.get(name) for name in _ARTIFACT_ENV_NAMES}
    yield
    os.chdir(cwd); sys.path[:] = path
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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


def test_artifact_env_vars_do_not_leak_into_later_tests():
    """앞의 두 테스트가 심은 AOD_ARTIFACTS·AOD_WT_ARTIFACTS 가 원래 없었다면(흔한 경우) 여기서는
    없어야 한다 — `delenv(raising=False)` 는 "원래 없었다"를 기록하지 못해, `enter_platform` 이
    os.environ 을 직접 심은 값이 다음 테스트로(삭제된 tmp_path 를 가리킨 채) 새는 걸 막지 못했었다."""
    assert "AOD_ARTIFACTS" not in os.environ
    assert "AOD_WT_ARTIFACTS" not in os.environ


def test_second_platform_in_same_process_is_refused(tmp_path):
    a, b = make_platform(tmp_path, "steam"), make_platform(tmp_path, "tmdb")
    bootstrap.enter_platform("steam", platform_root=a, artifacts=tmp_path)
    bootstrap.enter_platform("steam", platform_root=a, artifacts=tmp_path)      # 같은 플랫폼 재진입은 허용
    with pytest.raises(RuntimeError, match="steam"):
        bootstrap.enter_platform("tmdb", platform_root=b, artifacts=tmp_path)


def test_unknown_platform():
    with pytest.raises(ValueError):
        bootstrap.enter_platform("netflix")
