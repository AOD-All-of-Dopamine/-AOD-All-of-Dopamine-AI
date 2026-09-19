"""코퍼스 폴더에 manifest.json 을 만들고 계약 검증까지 돌린다.

    python -m aod_serving.tools.make_manifest --platform steam --dir /artifacts/tags_full [--force]

검증은 그 플랫폼의 확정값(PRODUCTION) 기준이다 — config.json 이 있으면 그 설정으로 본다.
게시 규칙: 검증을 통과한 폴더만 서빙 경로에 둔다(REC_TAB_DESIGN §8-6).

manifest.json 이 이미 있으면(추적된 세 기준 manifest 처럼) `--force` 없이는 **덮어쓰지 않는다** —
그 파일의 sha256 값이 gitignore 된 corpus_embeddings.npy 의 무결성 기준이다. 대신 그 기존
manifest 로 지금 폴더를 검증만 한다: 맞으면 통과, 틀리면(예: .npy 가 깨졌다) 실패로 알리고
그 나쁜 파일에 기준을 맞춰 다시 쓰지 않는다. 정말 다시 만들고 싶으면 `--force`.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from aod_serving.engine.contract import ArtifactError, load_schema, validate_artifacts
from aod_serving.engine.manifest import build_manifest, write_manifest_dict
from aod_serving.engine.overrides import ConfigError


def _production(platform: str, d: Path) -> dict:
    """확정값 + config.json. 플랫폼 코드를 import 할 수 없으면(테스트·라우터 환경) 빈 설정 — 컬럼 검사만 한다.

    config.json 이 실제로 깨졌으면(ConfigError) 여기서 삼키지 않는다 — 게시 검사가 엔진의 기동 시
    검증보다 느슨해지면 안 되기 때문이다(반쯤 맞는 설정으로 게시 통과 → 엔진은 기동 실패).
    """
    from aod_serving.engine.overrides import effective_config_for
    try:
        return effective_config_for(platform, d).production
    except (ImportError, ModuleNotFoundError):
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", required=True); ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--corpus-version"); ap.add_argument("--text-builder-version", default="unknown")
    ap.add_argument("--source", default="external-crawl")
    ap.add_argument("--force", action="store_true", help="manifest.json 이 이미 있어도 다시 만든다")
    a = ap.parse_args(argv)
    a.dir = a.dir.resolve()            # 플랫폼 진입이 chdir 하므로 먼저 절대 경로로
    version = a.corpus_version or a.dir.name
    try:
        schema = load_schema(a.platform)
        missing = [f for f in schema["required_files"] if not (a.dir / f).exists()]
        if missing:
            raise ArtifactError(f"필수 파일 없음: {missing}")
        production = _production(a.platform, a.dir)
        manifest_path = a.dir / "manifest.json"
        if manifest_path.exists() and not a.force:
            try:
                info = validate_artifacts(a.dir, schema, corpus_version=version, production=production)
            except ArtifactError as e:
                raise ArtifactError(f"{e} — 다시 만들려면 --force") from e
            print(f"기존 manifest 와 일치 — 검증 통과 · {info['rows']:,}행 × {info['dim']}")
            return 0
        # 새로 만들거나(--force) 처음 만든다 — 디스크에 쓰기 전에 먼저 메모리 위에서 검증한다.
        manifest = build_manifest(a.dir, schema, corpus_version=version, text_builder_version=a.text_builder_version,
                                  source=a.source)
        info = validate_artifacts(a.dir, schema, corpus_version=version, production=production, manifest=manifest)
        out = write_manifest_dict(a.dir, manifest)
    except (ArtifactError, ConfigError) as e:
        print(f"실패: {e}"); return 1
    print(f"{out} · 검증 통과 — {info['rows']:,}행 × {info['dim']}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
