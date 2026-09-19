#!/usr/bin/env bash
# 엔진 세그폴트 스트레스 하네스 — 컨테이너를 RUNS 번 새로 띄우고 매번 REQS 개 요청을 쏜다.
#
#   scripts/stress_engine.sh <platform> <runs> <reqs> [docker 추가 인자…]
#   scripts/stress_engine.sh webtoon 10 12
#   scripts/stress_engine.sh webtoon 10 12 -e ARROW_DEFAULT_MEMORY_POOL=mimalloc   # 할당자만 되돌려 보기
#
# 이 순서가 중요하다 — 과거 크래시는 아래 세 자리에서 났다:
#   1) 기동 직후 **첫** 요청            (적재 스레드가 막 죽고 그 TLS 슬롯이 재활용된다)
#   2) 동시 요청 묶음                   (anyio 워커가 여러 개 새로 생긴다)
#   3) 유휴 10초 초과 뒤의 요청         (anyio 워커가 죽고 **새 OS 스레드**가 뜬다)
set -uo pipefail
export MSYS_NO_PATHCONV=1

PLATFORM="${1:?platform}"; RUNS="${2:-5}"; REQS="${3:-10}"
shift 3 2>/dev/null || shift $# ; EXTRA=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC="$(cd "$HERE/../.." && (pwd -W 2>/dev/null || pwd))"
IMAGE="${AOD_REC_DEV_IMAGE:-aod-rec-dev:latest}"
NAME="aod-stress-$PLATFORM"
PORT="${STRESS_PORT:-18000}"
OUT="${STRESS_OUT:-/tmp/aod-stress}"; mkdir -p "$OUT"

docker rm -f "$NAME" >/dev/null 2>&1 || true
mapfile -t SEEDS < <(docker run --rm -v "$REC:/rec:ro" --tmpfs /tmp "$IMAGE" \
                       python scripts/engine_seeds.py "$PLATFORM" 12)
[ "${#SEEDS[@]}" -gt 0 ] || { echo "시드를 못 읽었다"; exit 2; }
echo "시드 ${#SEEDS[@]}개: ${SEEDS[*]:0:4} …"

post() {   # $1 = seed  → HTTP 코드
  curl -s -m 120 -o "$OUT/last_body.json" -w '%{http_code}' -X POST "localhost:$PORT/engine/recommend" \
    -H 'Content-Type: application/json' -d "{\"k\":10,\"seeds\":[\"$1\"]}" 2>/dev/null || echo 000
}
alive() { [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; }

crashes=0; total=0; bad=0
for run in $(seq 1 "$RUNS"); do
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" -e PLATFORM="$PLATFORM" -e PYTHONFAULTHANDLER=1 "${EXTRA[@]+"${EXTRA[@]}"}" \
    -p "$PORT:8000" -v "$REC:/rec:ro" --tmpfs /tmp "$IMAGE" python -m aod_serving.engine >/dev/null

  ready=0
  for _ in $(seq 1 240); do
    [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "localhost:$PORT/health" 2>/dev/null)" = "200" ] && { ready=1; break; }
    alive || break
    sleep 1
  done
  if [ "$ready" != 1 ]; then
    echo "run $run: 준비 실패 (exit=$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null))"
    docker logs "$NAME" > "$OUT/notready_run$run.log" 2>&1; crashes=$((crashes+1)); continue
  fi

  fail=""
  for r in $(seq 1 "$REQS"); do
    seed="${SEEDS[$(( (r - 1) % ${#SEEDS[@]} ))]}"
    if [ "$r" = 3 ]; then                       # (2) 동시 요청 묶음
      for s in "${SEEDS[@]:0:4}"; do post "$s" > "$OUT/c_$s.code" & done; wait
      for s in "${SEEDS[@]:0:4}"; do
        total=$((total+1)); c="$(cat "$OUT/c_$s.code")"; [ "$c" = 200 ] || { bad=$((bad+1)); echo "  동시요청 http=$c"; }
      done
    elif [ "$r" = 5 ]; then                     # (3) anyio 워커 유휴 타임아웃(10s) 넘기기
      sleep 12
      code="$(post "$seed")"; total=$((total+1)); [ "$code" = 200 ] || { bad=$((bad+1)); echo "  유휴후 http=$code"; }
    else
      code="$(post "$seed")"; total=$((total+1)); [ "$code" = 200 ] || { bad=$((bad+1)); echo "  req$r http=$code"; }
    fi
    if ! alive; then
      ec="$(docker inspect -f '{{.State.ExitCode}}' "$NAME")"
      echo "run $run req $r: 💥 CRASH exit=$ec"
      docker logs "$NAME" > "$OUT/crash_run${run}_req${r}.log" 2>&1
      crashes=$((crashes+1)); fail=1; break
    fi
  done
  [ -z "$fail" ] && echo "run $run: ok ($REQS 요청, arrow_pool=$(curl -s "localhost:$PORT/health" | grep -o '"arrow_pool":"[^"]*"'))"
done
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "──────────────────────────────────────────────"
echo "결과: 크래시 $crashes / $RUNS run · 요청 $total 개 중 비정상 응답 $bad 개"
echo "      platform=$PLATFORM extra=${EXTRA[*]:-없음}"
[ "$crashes" = 0 ] && [ "$bad" = 0 ]
