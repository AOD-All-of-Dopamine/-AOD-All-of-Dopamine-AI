#!/usr/bin/env bash
# 추천 서빙을 리눅스 VM 한 대에 올린다 (README §5 · REC_TAB_DESIGN §8-6).
#
#   [개발 PC에서]  scripts/deploy.sh push <user@vm>   아티팩트 4개 코퍼스를 VM 의 ARTIFACTS_BASE 로 보낸다(rsync)
#   [VM에서]      scripts/deploy.sh all              점검 → 이미지 빌드 → 아티팩트 검증 → 기동 → 확인 (처음 한 번)
#
#   check    도커·메모리·CPU·디스크·아티팩트·BIND_IP 점검 (아무것도 바꾸지 않는다)
#   build    엔진 4장 + 라우터 1장 이미지 빌드 (태그 = git 커밋)
#   verify   아티팩트 무결성 — 커밋된 manifest.json 의 sha256 과 대조 (컨테이너 안에서, 읽기 전용)
#   up       compose 기동 후 라우터가 응답할 때까지 기다린다 (Steam 첫 기동은 최대 5분)
#   smoke    라우터 /health 와 추천 한 건
#   ps · logs [서비스] · down
#
# 설정: serving/deploy.env (없으면 deploy.env.example 을 복사해 채운다). 파일에 없는 값은 아래 기본값을 쓴다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVING="$(cd "$HERE/.." && pwd)"
REC="$(cd "$SERVING/.." && pwd)"                # 빌드 컨텍스트 = recommendation/
ENV_FILE="${AOD_DEPLOY_ENV:-$SERVING/deploy.env}"
PLATFORMS=(steam tmdb webtoon webnovel)

if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
GIT_SHA="${GIT_SHA:-$(git -C "$REC" rev-parse --short HEAD 2>/dev/null || echo dev)}"
export REGISTRY="${REGISTRY:-local}"
export ENGINE_TAG="${ENGINE_TAG:-$GIT_SHA}"
export ROUTER_TAG="${ROUTER_TAG:-$GIT_SHA}"
export ARTIFACTS_BASE="${ARTIFACTS_BASE:-/srv/aod-artifacts}"
export BIND_IP="${BIND_IP:-127.0.0.1}"
export SERVING_MODE="${SERVING_MODE:-prod}"
export STEAM_CORPUS="${STEAM_CORPUS:-tags_full}" TMDB_CORPUS="${TMDB_CORPUS:-tmdb_v1}"
export WEBTOON_CORPUS="${WEBTOON_CORPUS:-wt_v1}" WEBNOVEL_CORPUS="${WEBNOVEL_CORPUS:-wn_v6}"
export GIT_SHA

corpus_of() {
  case "$1" in
    steam) echo "$STEAM_CORPUS" ;; tmdb) echo "$TMDB_CORPUS" ;;
    webtoon) echo "$WEBTOON_CORPUS" ;; webnovel) echo "$WEBNOVEL_CORPUS" ;;
  esac
}
engine_image() { echo "$REGISTRY/aod-rec-engine:$1-$ENGINE_TAG"; }
router_url() { echo "http://$BIND_IP:8080"; }
compose() { docker compose -p aod-rec -f "$SERVING/compose.yaml" "$@"; }
say() { printf '\n== %s\n' "$*"; }
fail() { printf '✗ %s\n' "$*" >&2; exit 1; }

cmd_push() {
  local target="${1:-}"
  [ -n "$target" ] || fail "사용법: scripts/deploy.sh push <user@vm> — 개발 PC 에서 돌린다"
  say "아티팩트 → $target:$ARTIFACTS_BASE (약 1.2GB — steam 764M · tmdb 281M · webnovel 143M · webtoon 17M, 두 번째부터는 바뀐 것만 간다)"
  ssh "$target" "sudo mkdir -p '$ARTIFACTS_BASE' && sudo chown \"\$(id -u):\$(id -g)\" '$ARTIFACTS_BASE'"
  for p in "${PLATFORMS[@]}"; do
    local cv; cv="$(corpus_of "$p")"
    local src="$REC/$p/artifacts/$cv/"
    [ -d "$src" ] || fail "$src 가 없다"
    ssh "$target" "mkdir -p '$ARTIFACTS_BASE/$p/$cv'"
    rsync -avP --delete "$src" "$target:$ARTIFACTS_BASE/$p/$cv/"
  done
  # 컨테이너는 uid 10001 로 읽기 전용 마운트한다 — 모두에게 읽기 권한이 있어야 한다(README §5-1)
  ssh "$target" "chmod -R a+rX '$ARTIFACTS_BASE'"
  echo "✓ 전송 끝. VM 에서: scripts/deploy.sh all"
}

cmd_check() {
  local bad=0
  say "도구"
  docker --version || { echo "✗ docker 가 없다"; bad=1; }
  docker compose version || { echo "✗ docker compose 플러그인이 없다"; bad=1; }
  docker info >/dev/null 2>&1 || { echo "✗ docker 데몬에 접근할 수 없다 (sudo usermod -aG docker \$USER 후 재로그인)"; bad=1; }

  say "자원 (실측 필요량: 메모리 약 3.3GB · 권장 8GB · vCPU 4 이상)"
  local mem_gb cpus free_gb
  mem_gb=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
  cpus=$(nproc); free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  echo "아키텍처 $(uname -m) · 메모리 ${mem_gb}GB · vCPU ${cpus} · 디스크 여유 ${free_gb}GB"
  awk "BEGIN{exit !($mem_gb < 6)}" && { echo "✗ 메모리 6GB 미만 — 엔진 4개를 함께 못 띄운다"; bad=1; }
  awk "BEGIN{exit !($mem_gb >= 6 && $mem_gb < 7.5)}" && echo "! 메모리 8GB 미만 — 뜨지만 Steam 캐시가 밀리면 지연이 튄다"
  [ "$cpus" -lt 4 ] && echo "! vCPU 4 미만 — 동시 요청에서 스로틀링"
  [ "$free_gb" -lt 15 ] && { echo "✗ 디스크 여유 15GB 미만 (이미지 + 아티팩트)"; bad=1; }

  say "아티팩트 ($ARTIFACTS_BASE)"
  for p in "${PLATFORMS[@]}"; do
    local d; d="$ARTIFACTS_BASE/$p/$(corpus_of "$p")"
    local missing=()
    for f in manifest.json corpus_embeddings.npy corpus_index.parquet dataset.parquet; do
      [ -f "$d/$f" ] || missing+=("$f")
    done
    if [ ${#missing[@]} -gt 0 ]; then echo "✗ $d — 없음: ${missing[*]}"; bad=1; continue; fi
    local unreadable; unreadable=$(find "$d" ! -perm -o+r | wc -l)
    [ "$unreadable" -gt 0 ] && { echo "✗ $d — 남이 못 읽는 파일 ${unreadable}개 (chmod -R a+rX $ARTIFACTS_BASE)"; bad=1; }
    echo "✓ $p $(du -shL "$d" | cut -f1)"
  done

  say "바인딩 주소 (BIND_IP=$BIND_IP)"
  if [ "$BIND_IP" = "127.0.0.1" ]; then
    echo "! 127.0.0.1 — 이 VM 안에서만 부를 수 있다. 백엔드가 부르려면 deploy.env 에 사설 IP 를 넣는다"
  elif ! ip -o addr 2>/dev/null | grep -q " $BIND_IP/"; then
    echo "✗ $BIND_IP 는 이 VM 의 주소가 아니다 ($(hostname -I 2>/dev/null))"; bad=1
  else
    echo "✓ $BIND_IP:8080 에만 연다 — 라우터에는 인증이 없다. 방화벽·보안 그룹에서 백엔드 API 서버만 허용할 것"
  fi
  [ "$bad" -eq 0 ] || fail "점검 실패 — 위 ✗ 를 고친 뒤 다시"
  echo; echo "✓ 점검 통과"
}

cmd_build() {
  say "이미지 빌드 (태그 $ENGINE_TAG · 빌드 컨텍스트 $REC)"
  for p in "${PLATFORMS[@]}"; do
    docker build -f "$SERVING/Dockerfile" --target engine --build-arg PLATFORM="$p" \
      --build-arg GIT_SHA="$GIT_SHA" -t "$(engine_image "$p")" "$REC"
  done
  docker build -f "$SERVING/Dockerfile" --target router --build-arg GIT_SHA="$GIT_SHA" \
    -t "$REGISTRY/aod-rec-router:$ROUTER_TAG" "$REC"
  echo "✓ 5장 빌드"
}

cmd_verify() {
  say "아티팩트 무결성 — 커밋된 manifest 의 sha256 과 대조 (새로 쓰지 않는다)"
  for p in "${PLATFORMS[@]}"; do
    local cv; cv="$(corpus_of "$p")"
    docker run --rm -v "$ARTIFACTS_BASE/$p:/artifacts:ro" "$(engine_image "$p")" \
      python -m aod_serving.tools.make_manifest --platform "$p" --dir "/artifacts/$cv" \
      || fail "$p 검증 실패 — 전송 중 깨졌을 수 있다. push 를 다시 하고 verify"
  done
  echo "✓ 4개 코퍼스 검증 통과"
}

cmd_up() {
  say "기동 (라우터는 엔진 4개가 모두 healthy 가 된 뒤 뜬다)"
  compose up -d
  local deadline=$((SECONDS + 900))
  until curl -fsS -o /dev/null "$(router_url)/health" 2>/dev/null; do
    [ $SECONDS -lt $deadline ] || { compose ps; fail "15분 안에 라우터가 응답하지 않았다 — scripts/deploy.sh logs rec-steam"; }
    printf '.'; sleep 10
  done
  echo; compose ps
  echo "✓ 라우터 응답 $(router_url)"
}

cmd_smoke() {
  say "라우터 /health"
  curl -fsS "$(router_url)/health"; echo
  say "추천 한 건 (게임 탭 · 시드 Dead by Daylight 381210)"
  local body
  body=$(curl -fsS -X POST "$(router_url)/v1/recommend" -H 'Content-Type: application/json' \
    -d '{"tab":"game","k":5,"buffer":0,"seeds":{"steam":["381210"]}}') || fail "추천 요청 실패"
  if command -v python3 >/dev/null; then
    printf '%s' "$body" | python3 -c 'import json,sys; r=json.load(sys.stdin); print("✓ 항목", len(r["items"]), "개 ·", [i["key"] for i in r["items"]], "· partial", r["partial"])'
  else
    echo "$body"
  fi
}

cmd="${1:-help}"; [ $# -gt 0 ] && shift
case "$cmd" in
  push)   cmd_push "$@" ;;
  check)  cmd_check ;;
  build)  cmd_build ;;
  verify) cmd_verify ;;
  up)     cmd_up ;;
  smoke)  cmd_smoke ;;
  all)    cmd_check; cmd_build; cmd_verify; cmd_up; cmd_smoke
          echo; echo "✓ 끝. 백엔드에 REC_ROUTER_BASE_URL=$(router_url) 를 넣으면 추천이 이 서버로 온다" ;;
  ps)     compose ps ;;
  logs)   compose logs --tail=200 -f "$@" ;;
  down)   compose down ;;
  *)      sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
