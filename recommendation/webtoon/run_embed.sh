#!/bin/sh
# 웹툰 코퍼스 임베딩 — **장시간 작업이다** (CPU 전용, GPU 없음).
# 실측 토큰 길이: 중앙 147 · p90 237 · p99 361 · 최대 572 → 상한 384 면 0.8% 만 잘린다.
# 배치 4 로 낮춘 것은 메모리 때문이 아니라 cgroup 5코어에서 긴 배치가 오래 붙잡히기 때문이다.
set -e
cd /home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/webtoon
export PYTHONPATH=. OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
../steam/.venv/bin/python -m src.embed_qwen --rep v2 --out artifacts/wt_v1  --batch 4 --max-seq 384
echo "=== rep_v2 완료 ==="
../steam/.venv/bin/python -m src.embed_qwen --rep v1 --out artifacts/rep_v1 --batch 4 --max-seq 384
echo "=== rep_v1 완료 · 둘 다 끝 ==="
