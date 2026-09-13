"""TMDB 코퍼스 임베딩 — Steam 파이프라인을 그대로 재사용한다.

모델·차원·정규화·max_seq_length 를 Steam 과 동일하게 맞춘다.
교차 도메인 유사도를 비교하려면 같은 벡터 공간이어야 하므로 선택의 여지가 없다.

메모리 규율도 Steam 것을 그대로 따른다. 처음엔 df['semantic_text'].tolist() 를
통째로 넘겼다가 cgroup 6GiB 한도에서 OOM 으로 죽었다(커널 SIGKILL 이라
encode_with_backoff 의 배치 반감 백오프는 잡을 예외조차 없다). Steam 은 TMDB 의 3배인
17만 건을 같은 한도에서 통과시켰는데, 차이는 배치 크기가 아니라 아래 네 가지였다:
  1. 텍스트를 임시 parquet 으로 내리고 청크로 흘려 읽는다 (stream_texts)
  2. 임베딩을 누적하지 않고 청크마다 원시 파일에 append 한다 (encode_to_shard)
  3. 모델 로드 후 safetensors 페이지 캐시 2.4GB 를 커널에 반환한다
     (drop_model_page_cache) — cgroup v2 는 페이지 캐시도 한도에 넣는다
  4. .npy 변환 전에 모델을 내린다

중단해도 샤드 파일 크기가 곧 진행 상황이라 같은 명령으로 재실행하면 이어간다.

산출물은 Steam/웹소설과 같은 레이아웃으로 둔다:
  corpus_embeddings.npy · corpus_index.parquet · dataset.parquet · qwen_run_config.json
"""
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TMDB = Path(__file__).resolve().parents[1]
STEAM = Path('/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation/steam')
# TMDB 와 Steam 둘 다 `src` 패키지를 갖고 있다. sys.path 로 섞으면 먼저 import 된
# 쪽이 `src` 를 선점해 다른 쪽 모듈을 찾지 못한다. TMDB 것은 경로로 직접 로드한다.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('tmdb_text_builder', TMDB / 'src' / 'text_builder.py')
_tb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tb)
build_dataset = _tb.build_dataset

sys.path.insert(0, str(STEAM))
os.chdir(STEAM)                                        # steam 모듈이 상대경로를 쓴다
from src.config import load_config                     # noqa: E402
from src.embed_qwen import (                           # noqa: E402
    _cgroup_gb, _shard_rows, drop_model_page_cache,
    encode_to_shard, resolve_runtime, shard_to_npy,
)

OUT = TMDB / 'artifacts' / 'tmdb_v1'
OUT.mkdir(parents=True, exist_ok=True)
TEXTS = OUT / '_texts.parquet'
SHARD = OUT / '_embeddings.raw'


def main():
    cfg = load_config()

    # --- 1. 데이터셋을 만들어 전부 디스크로 내린다 ---
    if not TEXTS.exists():
        df = build_dataset([TMDB / 'data' / 'movie_enriched.jsonl',
                            TMDB / 'data' / 'tv_enriched.jsonl'])
        print(f"코퍼스 {len(df):,}건  "
              f"(영화 {(df['media']=='movie').sum():,} · 드라마 {(df['media']=='tv').sum():,})")
        print(f"  한국어 {(df['lang']=='ko').sum():,} · 영어보완 {(df['lang']=='en').sum():,}")
        print(f"  키워드 0개 {(df['n_keywords']==0).sum():,} ({100*(df['n_keywords']==0).mean():.1f}%)")
        print(f"  semantic_text 길이 중앙 {df['semantic_text'].str.len().median():.0f}자")
        df.to_parquet(OUT / 'dataset.parquet', index=False)
        pd.DataFrame({'embedding_row': range(len(df)),
                      'item_id': df['item_id'].to_numpy(),
                      'name': df['name'].to_numpy()}
                     ).to_parquet(OUT / 'corpus_index.parquet', index=False)
        # 임베딩 순서는 이 파일이 유일한 근거다 — dataset.parquet 과 행 순서가 같아야 한다
        df[['semantic_text']].to_parquet(TEXTS, index=False)
        del df
        gc.collect()

    n_total = len(pd.read_parquet(OUT / 'corpus_index.parquet', columns=['embedding_row']))

    # --- 2. 모델을 올리고 페이지 캐시를 반환한다 ---
    model, device, batch = resolve_runtime(cfg)
    freed = drop_model_page_cache(model)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"\n임베딩 {n_total:,}건 (device={device} batch={batch} dim={dim}) | "
          f"cgroup {_cgroup_gb():.2f}GB (페이지 캐시 {freed:.2f}GB 반환)", flush=True)

    # --- 3. 청크마다 원시 파일에 append (중단 시 이어받기) ---
    encode_to_shard(model, TEXTS, n_total, batch, OUT, dim)
    done = _shard_rows(SHARD, dim)
    if done < n_total:
        print(f"부분 완료: {done:,}/{n_total:,} — 같은 명령으로 재실행하세요")
        return

    # --- 4. 모델을 내려야 .npy 변환에 쓸 메모리가 생긴다 ---
    del model
    gc.collect()
    total = shard_to_npy(SHARD, OUT / 'corpus_embeddings.npy', dim)
    SHARD.unlink(missing_ok=True)
    TEXTS.unlink(missing_ok=True)

    json.dump({'model': cfg['embedding']['model_name'],
               'normalize_embeddings': cfg['embedding']['normalize'],
               'max_seq_length': cfg['runtime']['max_seq_length'],
               'device': device, 'batch_size': batch,
               'corpus_rows': int(total), 'embedding_dim': int(dim),
               'source': 'TMDB discover vote_count>=30 (movie+tv), ko 우선/en 보완',
               'semantic_text': '줄거리 + 키워드(top15) + 장르 — Steam 관례, 제목 제외'},
              open(OUT / 'qwen_run_config.json', 'w'), ensure_ascii=False, indent=2)
    print(f"저장 {OUT}  ({total:,}, {dim})")


if __name__ == '__main__':
    main()
