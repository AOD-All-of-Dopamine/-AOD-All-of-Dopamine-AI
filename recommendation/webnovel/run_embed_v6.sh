#!/bin/sh
# wn_v6 — 전체 크롤(29,494편) 임베딩. CPU 전용, 약 28시간. 백엔드·리랭커와 같이 띄우지 않는다.
cd /home/ubuntu/aod-webnovel/recommendation/webnovel
export PYTHONPATH=. AOD_ARTIFACTS=artifacts/wn_v6 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam/.venv/bin/python -m src.embed_qwen --in artifacts/wn_v6 --out artifacts/wn_v6
echo "=== wn_v6 임베딩 완료 ==="
