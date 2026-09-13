#!/bin/sh
# CPU 상한 래퍼 — 무거운 작업(임베딩·크롤·수집)을 이걸로 감싼다.
#
#   scripts/cpu_capped.sh <명령> [인자...]
#
# 이 컨테이너는 cgroup 에서 이미 **5코어**로 묶여 있다 (`/sys/fs/cgroup/cpu.max` = 500000 100000).
# 그 5코어를 한 프로세스가 다 쓰면 체감상 "CPU 를 다 먹는" 상태가 된다.
# 여기서는 **4코어(할당량의 80%, 여유 1코어)**로 묶고 우선순위를 낮춘다.
#   · taskset  : 실제로 쓸 수 있는 코어를 4개로 제한 (BLAS/torch 가 무시할 수 없는 하드 제한)
#   · OMP/MKL  : 스레드 수도 같이 맞춘다 (안 맞추면 4코어 위에서 12스레드가 다투며 더 느려진다)
#   · nice 10  : 대화형 작업이 항상 먼저 가도록
#
# cgroup cpu.max 를 직접 고치는 방법은 root 가 필요해서 쓰지 않는다.
CORES="${AOD_CPU_CORES:-0-3}"
N="$(echo "$CORES" | awk -F- '{print $2-$1+1}')"
export OMP_NUM_THREADS="$N" MKL_NUM_THREADS="$N" OPENBLAS_NUM_THREADS="$N" \
       NUMEXPR_NUM_THREADS="$N" TOKENIZERS_PARALLELISM=false
exec nice -n 10 taskset -c "$CORES" "$@"
