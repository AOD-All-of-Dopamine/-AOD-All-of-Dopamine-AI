# -*- coding: utf-8 -*-
# 유저 임베딩 생성 API

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

BASE = r"csv 데이터\clean"
OUT_USER = os.path.join(BASE, "user_embeddings.csv")

def load_json_to_df(json_data: Dict) -> pd.DataFrame:
    """JSON 데이터를 DataFrame으로 변환"""
    if json_data is None:
        return None
    try:
        if isinstance(json_data, list):
            return pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            return pd.DataFrame([json_data])
        else:
            return None
    except Exception as e:
        print(f"⚠️ JSON→DataFrame 변환 실패: {e}")
        return None

def load_content_genres_tall(rawg_df: pd.DataFrame) -> pd.DataFrame:
    """
    content_raw_genres.csv 가
      - 세로형: content_id, source, raw_genre
      - 또는 가로형: content_id, source, raw_genre_1~3
    둘 중 어떤 포맷이든 받아서
    항상 (content_id, raw_genre) 세로형으로 반환.
    """
    g = rawg_df.copy()
    g.columns = [c.strip() for c in g.columns]

    if "content_id" not in g.columns:
        raise RuntimeError("content_raw_genres에 content_id 컬럼이 없습니다.")

    # 이미 raw_genre 단일 컬럼이 있으면 그대로 사용
    if "raw_genre" in g.columns:
        g["content_id"] = pd.to_numeric(g["content_id"], errors="coerce").astype("Int64")
        g = g[g["content_id"].notna()].copy()
        g["content_id"] = g["content_id"].astype(int)
        g["raw_genre"]  = g["raw_genre"].astype(str).str.strip()
        g = g[g["raw_genre"] != ""]
        return g[["content_id", "raw_genre"]].drop_duplicates().reset_index(drop=True)

    # 가로형: raw_genre_1~3 을 세로형으로 펴기
    genre_cols = [c for c in ["raw_genre_1", "raw_genre_2", "raw_genre_3"] if c in g.columns]
    if not genre_cols:
        raise RuntimeError("content_raw_genres에 raw_genre 또는 raw_genre_1/2/3 컬럼이 필요합니다.")

    g["content_id"] = pd.to_numeric(g["content_id"], errors="coerce").astype("Int64")
    g = g[g["content_id"].notna()].copy()
    g["content_id"] = g["content_id"].astype(int)

    tall = g.melt(
        id_vars=["content_id"],
        value_vars=genre_cols,
        value_name="raw_genre"
    )
    tall["raw_genre"] = tall["raw_genre"].astype(str).str.strip()
    tall = tall[tall["raw_genre"].notna() & (tall["raw_genre"] != "")]
    tall = tall[["content_id", "raw_genre"]].drop_duplicates()

    print(f"✅ content_raw_genres wide → tall 변환: rows={len(tall)}")
    return tall.reset_index(drop=True)

def build_user_embeddings(
    item_embeddings_df: pd.DataFrame,
    raw_genres_df: pd.DataFrame,
    user_preferred_genres_df: pd.DataFrame,
    contents_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    유저 임베딩 생성
    
    Args:
        item_embeddings_df: 아이템 임베딩 (content_id, emb_0, emb_1, ...)
        raw_genres_df: 원본 장르 데이터 (content_id, source, raw_genre_1~3 또는 content_id, raw_genre)
        user_preferred_genres_df: 유저 선호 장르 (user_id, genre, [username])
        contents_df: 콘텐츠 데이터 (선택사항)
    
    Returns:
        pd.DataFrame: 유저 임베딩 (user_id, username, emb_0, emb_1, ...)
    """
    
    # 1) 아이템 임베딩 검증 및 정규화
    emb = item_embeddings_df.copy()
    
    if "content_id" not in emb.columns:
        raise RuntimeError("아이템 임베딩에 content_id 컬럼이 필요합니다.")
    
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("아이템 임베딩에 emb_로 시작하는 임베딩 컬럼이 없습니다.")
    
    emb["content_id"] = pd.to_numeric(emb["content_id"], errors="coerce").astype("Int64")
    emb = emb[emb["content_id"].notna()].copy()
    emb["content_id"] = emb["content_id"].astype(int)
    
    for c in emb_cols:
        emb[c] = pd.to_numeric(emb[c], errors="coerce")
    emb.dropna(subset=emb_cols, inplace=True)
    
    # L2 normalize
    M = emb[emb_cols].to_numpy(np.float32)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    emb.loc[:, emb_cols] = M
    
    print(f"✅ 아이템 임베딩 로드: items={len(emb)}, dim={len(emb_cols)}")
    
    ids = emb["content_id"].to_numpy(int)
    
    # 2) 장르 센트로이드 계산
    rawg_tall = load_content_genres_tall(raw_genres_df)
    
    if rawg_tall is None or rawg_tall.empty:
        raise RuntimeError("content_raw_genres에 유효한 장르가 없습니다.")
    
    rawg = rawg_tall.copy()
    rawg["g"] = rawg["raw_genre"].astype(str).str.strip().str.lower()
    
    idx_map = {cid: i for i, cid in enumerate(ids)}
    genre_vecs = {}
    for g, grp in rawg.groupby("g"):
        idxs = [idx_map[c] for c in grp["content_id"].tolist() if c in idx_map]
        if not idxs:
            continue
        v = M[idxs].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        genre_vecs[g] = v
    
    if not genre_vecs:
        raise RuntimeError("장르별로 매칭되는 아이템 임베딩이 없습니다.")
    
    print(f"✅ 장르 센트로이드 수: {len(genre_vecs)}")
    
    # 3) 유저별 선호 장르 처리
    ugen = user_preferred_genres_df.copy()
    ugen.columns = [c.strip() for c in ugen.columns]
    
    # 유저 ID 컬럼 추론
    col_user_id = None
    for cand in ["user_id", "id", "preference_id"]:
        if cand in ugen.columns:
            col_user_id = cand
            break
    if col_user_id is None:
        raise RuntimeError("user_preferred_genres에 user_id / id / preference_id 컬럼이 없습니다.")
    
    if "genre" not in ugen.columns:
        raise RuntimeError("user_preferred_genres에 genre 컬럼이 필요합니다.")
    
    has_username = "username" in ugen.columns
    
    keep_cols = [col_user_id, "genre"]
    if has_username:
        keep_cols.append("username")
    
    ugen = ugen[keep_cols].dropna(subset=[col_user_id, "genre"]).copy()
    ugen[col_user_id] = pd.to_numeric(ugen[col_user_id], errors="coerce").astype("Int64")
    ugen = ugen[ugen[col_user_id].notna()].copy()
    ugen[col_user_id] = ugen[col_user_id].astype(int)
    ugen["g"] = ugen["genre"].astype(str).str.strip().str.lower()
    
    # 4) 유저 임베딩 생성
    rows = []
    skipped = 0
    for uid, grp in ugen.groupby(col_user_id):
        genres = sorted(set([g for g in grp["g"].tolist() if g in genre_vecs]))
        if not genres:
            skipped += 1
            continue
        
        g_vecs = [genre_vecs[g] for g in genres]
        v_user = np.mean(np.stack(g_vecs, axis=0), axis=0)
        v_user = v_user / (np.linalg.norm(v_user) + 1e-12)
        
        uname = ""
        if has_username:
            uname = str(grp["username"].iloc[0])
        
        rows.append([int(uid), uname] + list(map(float, v_user)))
    
    if not rows:
        raise RuntimeError("유저 임베딩 생성 결과가 비었습니다.")
    
    dim = len(emb_cols)
    out = pd.DataFrame(rows, columns=["user_id", "username"] + [f"emb_{i}" for i in range(dim)])
    out = out.sort_values("user_id").reset_index(drop=True)
    
    print(f"✅ 유저 임베딩 생성: users={len(out)}, skipped={skipped}")
    return out
