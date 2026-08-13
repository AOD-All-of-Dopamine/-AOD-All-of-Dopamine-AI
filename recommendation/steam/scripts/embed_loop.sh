#!/usr/bin/env bash
# 임베딩을 여러 프로세스로 나눠 돌린다.
#
# cgroup 4GiB 한도에서 한 프로세스로 17만 건을 돌리면 반드시 죽는다(실측 6회).
# RSS 는 2.81GB 인데 OOM 이 나는데, cgroup 은 페이지 캐시까지 세기 때문이다 —
# 모델 safetensors 2.4GB 를 읽은 캐시가 한도를 함께 먹는다. 프로세스 안에서
# 이 캐시를 확실히 놓아주기는 어렵다.
#
# 그래서 원인을 파고드는 대신 구조로 우회한다: N건마다 프로세스를 끝내고 새로 띄운다.
# 종료 시 커널이 전부 회수하므로 누적이 사라진다. 진행 상황은 _embeddings.raw 파일
# 크기에 남아 있어 다음 프로세스가 이어받는다.
#
# 모델 로드는 약 30초. 5,000건(약 1시간)마다 재시작하면 오버헤드는 1% 미만이다.
#
#   scripts/embed_loop.sh artifacts/full_v1 [MAX_ITEMS] [추가 인자...]
#   scripts/embed_loop.sh artifacts/tags_v1 5000 --append
set -uo pipefail
cd "$(dirname "$0")/.."

ART="${1:-artifacts/full_v1}"
STEP="${2:-5000}"
shift 2 2>/dev/null || shift $# 
EXTRA=("$@")          # --append 등을 그대로 넘긴다
PY=".venv/bin/python"
export AOD_CONFIG="${AOD_CONFIG:-configs/full_v1.yaml}"

rows() { [ -f "$ART/_embeddings.raw" ] && stat -c%s "$ART/_embeddings.raw" || echo 0; }

echo "임베딩 루프 시작: $ART (프로세스당 ${STEP}건)"
stuck=0
prev=$(rows)
for attempt in $(seq 1 500); do
    $PY -u -m src.embed_qwen --in "$ART" --out "$ART" --max-items "$STEP" "${EXTRA[@]}"
    rc=$?
    if [ -f "$ART/corpus_embeddings.npy" ] && [ ! -f "$ART/_embeddings.raw" ]; then
        echo "완료 (프로세스 ${attempt}회)"
        exit 0
    fi
    cur=$(rows)
    if [ "$cur" -le "$prev" ]; then
        stuck=$((stuck + 1))
        echo "  진행 없음 (${stuck}/3) — 종료코드 ${rc}"
        # 한 청크도 못 넘기면 무한 재시작이 된다. 3회 연속이면 멈춘다.
        if [ $stuck -ge 3 ]; then
            echo "3회 연속 진행이 없습니다. 청크 크기(CHUNK)나 메모리를 확인하세요."
            exit 1
        fi
    else
        stuck=0
        echo "  프로세스 ${attempt}: $((cur / 4096))건 완료 (종료코드 ${rc})"
    fi
    prev=$cur
    sleep 3
done
echo "500회를 넘겼습니다 — 진행이 멈춘 것은 아닌지 확인하세요"
exit 1
