#!/usr/bin/env bash
# 서빙 개발 도우미 — 전부 Docker dev 이미지 안에서 돈다.
#   build            dev 이미지 빌드
#   test [인자…]     레포를 읽기 전용으로 마운트하고 pytest (운영 컨테이너도 루트가 읽기 전용이다)
#   run <명령…>      레포를 읽기·쓰기로 마운트하고 임의 명령 (기준 목록·manifest 생성용)
#   up               compose 5개 서비스(엔진 4 + 라우터) 빌드 후 기동
#   down             compose 스택 내리기
#   net <명령…>      compose 망 안에서 dev 이미지로 명령 실행(e2e·loadgate) — 레포를 /rec 에 읽기 전용으로 마운트
#   세그폴트 회귀: scripts/stress_engine.sh <플랫폼> <run 수> <요청 수> · scripts/repro_segv.py
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC="$(cd "$HERE/../.." && (pwd -W 2>/dev/null || pwd))"   # Git Bash 에서는 C:/… 형태가 필요하다
IMAGE="${AOD_REC_DEV_IMAGE:-aod-rec-dev:latest}"
cmd="${1:-help}"; [ $# -gt 0 ] && shift
case "$cmd" in
  build) docker build -f "$REC/serving/Dockerfile" --target dev -t "$IMAGE" "$REC" ;;
  test)  docker run --rm -e PYTHONIOENCODING=utf-8 -v "$REC:/rec:ro" --tmpfs /tmp "$IMAGE" pytest -p no:cacheprovider "$@" ;;
  run)   docker run --rm -i -e PYTHONIOENCODING=utf-8 -v "$REC:/rec" "$IMAGE" "$@" ;;
  up)    GIT_SHA="$(git -C "$REC" rev-parse --short HEAD 2>/dev/null || echo dev)" docker compose -p aod-rec -f "$REC/serving/compose.yaml" -f "$REC/serving/compose.local.yaml" up -d --build "$@" ;;   # $REC (pwd -W 형태) 를 쓴다 — $HERE(POSIX 경로)는 MSYS_NO_PATHCONV=1 에서 네이티브 git.exe 가 못 읽는다
  down)  docker compose -p aod-rec -f "$REC/serving/compose.yaml" -f "$REC/serving/compose.local.yaml" down "$@" ;;
  net)   docker run --rm -i -e PYTHONIOENCODING=utf-8 --network aod-rec_aod-rec -v "$REC:/rec:ro" --tmpfs /tmp "$IMAGE" "$@" ;;   # compose 망 안에서 dev 이미지로 명령 실행(e2e·loadgate)
  *)     sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
