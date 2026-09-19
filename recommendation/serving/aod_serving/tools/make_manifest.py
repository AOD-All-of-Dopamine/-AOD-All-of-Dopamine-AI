"""코퍼스 폴더에 manifest.json 을 만들고 계약 검증까지 돌린다.

    python -m aod_serving.tools.make_manifest --platform steam --dir /artifacts/tags_full

검증은 그 플랫폼의 확정값(PRODUCTION) 기준이다 — config.json 이 있으면 그 설정으로 본다.
게시 규칙: 검증을 통과한 폴더만 서빙 경로에 둔다(REC_TAB_DESIGN §8-6).
"""
from __future__ import annotations
import argparse
from pathlib import Path

from aod_serving.engine.contract import ArtifactError, load_schema, validate_artifacts
from aod_serving.engine.manifest import write_manifest


def _production(platform: str, d: Path) -> dict:
    """확정값 + config.json. 플랫폼 코드를 import 할 수 없으면(테스트·라우터 환경) 빈 설정 — 컬럼 검사만 한다."""
    try:
        from aod_serving.engine.overrides import effective_config_for
        return effective_config_for(platform, d).production
    except Exception:
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True); ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--corpus-version"); ap.add_argument("--text-builder-version", default="unknown")
    ap.add_argument("--source", default="external-crawl")
    a = ap.parse_args(argv)
    a.dir = a.dir.resolve()            # 플랫폼 진입이 chdir 하므로 먼저 절대 경로로
    version = a.corpus_version or a.dir.name
    try:
        schema = load_schema(a.platform)
        missing = [f for f in schema["required_files"] if not (a.dir / f).exists()]
        if missing:
            raise ArtifactError(f"필수 파일 없음: {missing}")
        out = write_manifest(a.dir, schema, corpus_version=version, text_builder_version=a.text_builder_version, source=a.source)
        info = validate_artifacts(a.dir, schema, corpus_version=version, production=_production(a.platform, a.dir))
    except ArtifactError as e:
        print(f"실패: {e}"); return 1
    print(f"{out} · 검증 통과 — {info['rows']:,}행 × {info['dim']}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
