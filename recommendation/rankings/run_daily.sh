#!/bin/sh
# 랭킹 스냅샷 — **매시 정각에 돌고, 그날 치가 다 쌓이면 바로 빠진다.**
# crontab: 0 * * * * /home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/rankings/run_daily.sh
#
# 왜 하루 한 번이 아니라 매시인가 (2026-09-12):
#   9/5~9/12 여드레 동안 `0 9 * * *` 가 매일 돌았지만 **32회 전부 실패**했다.
#   전부 같은 오류 — `Temporary failure in name resolution`. 성공 0회.
#   같은 스크립트를 낮에 손으로 돌리면 네 소스 다 정상이다. 즉 코드가 아니라
#   **9시라는 시각에 네트워크가 없다**(호스트가 자는 중으로 추정).
#   짧은 백오프 재시도는 같은 1분 안에 또 실패하니 소용없다. 그날 안에
#   네트워크가 살아나는 시각을 잡으려면 시간 간격으로 다시 시도해야 한다.
#
# 이미 오늘 치가 있으면 즉시 종료하므로, 성공한 날의 나머지 23번은 API 를 건드리지 않는다.
set -u
ROOT=/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation
cd "$ROOT" || exit 1
PY="$ROOT/steam/.venv/bin/python"
LOG="$ROOT/rankings/data/collect.log"
TODAY=$(date +%Y-%m-%d)

# 네 소스 전부 오늘 날짜가 있으면 할 일 없음 — 부분 실패한 소스만 다시 받는다.
MISSING=""
for p in steam tmdb webtoon webnovel; do
    f="$ROOT/rankings/data/$p.jsonl"
    if [ ! -f "$f" ] || ! tail -1 "$f" | grep -q "\"date\": *\"$TODAY\""; then
        MISSING="$MISSING $p"
    fi
done
[ -z "$MISSING" ] && exit 0

for p in $MISSING; do
    "$PY" rankings/collect.py --only "$p" >> "$LOG" 2>&1
done

# 수집 직후 코퍼스 피복률을 잰다 — 랭킹에 있는데 우리한테 없으면 그 작품은
# 어떤 시드로도 추천될 수 없다. 크롤 지연 경보이지 랭커 문제가 아니다.
"$PY" rankings/coverage.py --quiet >> "$LOG" 2>&1
