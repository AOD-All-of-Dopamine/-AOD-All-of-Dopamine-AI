"""동일성 테스트 — 아티팩트가 있는 환경에서만 돈다. 플랫폼마다 별도 프로세스(최상위 `src` 충돌)."""
import json, os, subprocess, sys
from pathlib import Path
import pytest

REC = Path(os.environ.get("AOD_REC_ROOT") or Path(__file__).resolve().parents[2])
CORPORA = {"steam": "tags_full", "tmdb": "tmdb_v1", "webtoon": "wt_v1", "webnovel": "wn_v6"}


def _identity(platform: str, *args: str) -> dict:
    d = REC / platform / "artifacts" / CORPORA[platform]
    if not (d / "corpus_embeddings.npy").exists() or not (d / "manifest.json").exists():
        pytest.skip(f"{platform} 아티팩트(또는 manifest.json) 없음")
    p = subprocess.run([sys.executable, "-m", "aod_serving.tools.identity", "--platform", platform, *args],
                       capture_output=True, text=True, encoding="utf-8", env=os.environ.copy())
    assert p.returncode == 0, p.stderr[-3000:] + p.stdout[-500:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.mark.artifacts
@pytest.mark.parametrize("platform", list(CORPORA))
def test_catalog_restricts_candidates_on_real_corpora(platform):
    """서빙 가능 목록(spec3 §10) — 목록 안에서만 추천 · 쪽 중복 없음 · 목록=전체면 OFF 와 동일 ·
    목록 밖 시드도 동작. 세부 판정은 `tools/identity.py:catalog_check`."""
    summary = _identity(platform, "--catalog-check")
    assert summary["catalog"]["mismatch"] == 0 and summary["catalog"]["compared"] >= 11


@pytest.mark.artifacts
@pytest.mark.parametrize("platform", list(CORPORA))
def test_service_results_equal_evaluation_path(platform):
    summary = _identity(platform)
    assert summary["l1"]["mismatch"] == 0 and summary["l2"]["mismatch"] == 0
    assert summary["l2"]["compared"] >= {"steam": 270, "tmdb": 260, "webtoon": 335, "webnovel": 156}[platform]
    # tmdb 는 L1 에 media(영화/드라마 탭) 케이스가 더 붙는다 — 프로필 5개 × 2 media 만큼 늘어난다
    assert summary["l1"]["compared"] >= {"steam": 5, "tmdb": 15, "webtoon": 5, "webnovel": 5}[platform]
