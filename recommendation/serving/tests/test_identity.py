"""동일성 테스트 — 아티팩트가 있는 환경에서만 돈다. 플랫폼마다 별도 프로세스(최상위 `src` 충돌)."""
import json, os, subprocess, sys
from pathlib import Path
import pytest

REC = Path(os.environ.get("AOD_REC_ROOT") or Path(__file__).resolve().parents[2])
CORPORA = {"steam": "tags_full", "tmdb": "tmdb_v1", "webtoon": "wt_v1", "webnovel": "wn_v6"}


@pytest.mark.artifacts
@pytest.mark.parametrize("platform", list(CORPORA))
def test_service_results_equal_evaluation_path(platform):
    d = REC / platform / "artifacts" / CORPORA[platform]
    if not (d / "corpus_embeddings.npy").exists() or not (d / "manifest.json").exists():
        pytest.skip(f"{platform} 아티팩트(또는 manifest.json) 없음")
    p = subprocess.run([sys.executable, "-m", "aod_serving.tools.identity", "--platform", platform],
                       capture_output=True, text=True, encoding="utf-8", env=os.environ.copy())
    assert p.returncode == 0, p.stderr[-3000:] + p.stdout[-500:]
    summary = json.loads(p.stdout.strip().splitlines()[-1])
    assert summary["l1"]["mismatch"] == 0 and summary["l2"]["mismatch"] == 0
    assert summary["l2"]["compared"] >= {"steam": 270, "tmdb": 260, "webtoon": 335, "webnovel": 156}[platform]
