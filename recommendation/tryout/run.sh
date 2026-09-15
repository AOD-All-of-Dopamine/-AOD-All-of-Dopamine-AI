#!/usr/bin/env bash
# 시험대 전체를 띄운다. 이미 떠 있으면 내리고 다시 띄운다.
#   ./run.sh          — 전부
#   ./run.sh steam    — 하나만 (게이트웨이는 항상 같이)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
AOD="$HERE/.."
PY_AOD="$AOD/steam/.venv/bin/python"
PY_WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/.venv/bin/python"
LOG="${TMPDIR:-/tmp}/aod-tryout"; mkdir -p "$LOG"

declare -A PORT=([steam]=8011 [tmdb]=8012 [webnovel]=8013 [webtoon]=8014)
declare -A PYB=([steam]=$PY_AOD [tmdb]=$PY_AOD [webnovel]=$PY_WN [webtoon]=$PY_AOD)
WANT=("${@:-steam tmdb webnovel webtoon}"); WANT=(${WANT[@]})

for p in "${WANT[@]}"; do
  pkill -f "backend.py --platform $p" 2>/dev/null
  # 포트가 풀릴 때까지 기다린다. 안 기다리면 새 프로세스가 EADDRINUSE 로 죽고
  # 옛 프로세스가 계속 서빙해서 "재시작했는데 그대로"가 된다.
  for _ in $(seq 1 20); do curl -s -m 1 -o /dev/null "http://127.0.0.1:${PORT[$p]}/health" || break; sleep 0.5; done
  setsid nohup "${PYB[$p]}" "$HERE/backend.py" --platform "$p" --port "${PORT[$p]}" \
    > "$LOG/$p.log" 2>&1 < /dev/null &
  echo "  $p 시작 (:${PORT[$p]}) · 로그 $LOG/$p.log"
done

pkill -f "gateway.py --port" 2>/dev/null
setsid nohup python3 "$HERE/gateway.py" --port 8000 > "$LOG/gateway.log" 2>&1 < /dev/null &
echo "  게이트웨이 시작 (:8000)"

echo -n "  준비되는 중"
for i in $(seq 1 120); do
  sleep 2; echo -n "."
  ok=1
  for p in "${WANT[@]}"; do
    curl -s -m 3 -o /dev/null "http://127.0.0.1:${PORT[$p]}/health" || ok=0
  done
  [ "$ok" = 1 ] && { echo; echo "  준비 완료 → http://localhost:8000"; exit 0; }
done
echo; echo "  일부 백엔드가 안 뜹니다. 로그를 보세요: $LOG/"; exit 1
