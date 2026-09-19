#!/usr/bin/env bash
# 서빙 개발 도우미 — 전부 Docker dev 이미지 안에서 돈다.
#   build            dev 이미지 빌드
#   test [인자…]     레포를 읽기 전용으로 마운트하고 pytest (운영 컨테이너도 루트가 읽기 전용이다)
#   run <명령…>      레포를 읽기·쓰기로 마운트하고 임의 명령 (기준 목록·manifest 생성용)
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC="$(cd "$HERE/../.." && (pwd -W 2>/dev/null || pwd))"   # Git Bash 에서는 C:/… 형태가 필요하다
IMAGE="${AOD_REC_DEV_IMAGE:-aod-rec-dev:latest}"
cmd="${1:-help}"; [ $# -gt 0 ] && shift
case "$cmd" in
  build) docker build -f "$REC/serving/Dockerfile" --target dev -t "$IMAGE" "$REC" ;;
  test)  docker run --rm -e PYTHONIOENCODING=utf-8 -v "$REC:/rec:ro" --tmpfs /tmp "$IMAGE" pytest -p no:cacheprovider "$@" ;;
  run)   docker run --rm -e PYTHONIOENCODING=utf-8 -v "$REC:/rec" "$IMAGE" "$@" ;;
  *)     sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
