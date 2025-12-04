# -*- coding: utf-8 -*-
# 유저 임베딩 생성 (유저 선호 장르 centroid만 사용)

import os
import numpy as np
import pandas as pd
from typing import Optional, List

BASE = r"csv 데이터\clean"
ITEM_EMB_1 = os.path.join(BASE, "item_embeddings_torch.csv")
ITEM_EMB_2 = os.path.join(BASE, "item_embeddings.csv")
CONTENTS   = os.path.join(BASE, "contents.csv")
RAW_GENRES = os.path.join(BASE, "content_raw_genres.csv")  # 지금: content_id, source, raw_genre_1~3
UP_GENRES  = os.path.join(BASE, "user_preferred_genres.csv")
OUT_USER   = os.path.join(BASE, "user_embeddings.csv")


def read_csv_retry(path, encodings=("utf-8-sig","utf-8","cp949","euc-kr","latin1"), **kw) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    last = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except Exception as e:
            last = e
    raise last


def load_item_embeddings() -> pd.DataFrame:
    # 명시적 선택 (DataFrame에 or 금지)
    df1 = read_csv_retry(ITEM_EMB_1)
    if df1 is not None and not df1.empty:
        df = df1
        src = os.path.basename(ITEM_EMB_1)
    else:
        df2 = read_csv_retry(ITEM_EMB_2)
        if df2 is not None and not df2.empty:
            df = df2
            src = os.path.basename(ITEM_EMB_2)
        else:
            raise RuntimeError("아이템 임베딩 CSV가 없습니다: item_embeddings_torch.csv 또는 item_embeddings.csv")

    if "content_id" not in df.columns:
        raise RuntimeError("아이템 임베딩 CSV에 content_id 컬럼이 필요합니다.")

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("아이템 임베딩 CSV에 emb_로 시작하는 임베딩 컬럼이 없습니다.")

    df["content_id"] = pd.to_numeric(df["content_id"], errors="coerce").astype("Int64")
    df = df[df["content_id"].notna()].copy()
    df["content_id"] = df["content_id"].astype(int)

    for c in emb_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=emb_cols, inplace=True)

    # L2 normalize
    M = df[emb_cols].to_numpy(np.float32)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    df.loc[:, emb_cols] = M

    print(f"✅ 아이템 임베딩 로드: {src} (items={len(df)}, dim={len(emb_cols)})")
    return df[["content_id"] + emb_cols]


def load_content_genres_tall() -> pd.DataFrame:
    """
    content_raw_genres.csv 가
      - 세로형: content_id, source, raw_genre
      - 또는 가로형: content_id, source, raw_genre_1~3
    둘 중 어떤 포맷이든 받아서
    항상 (content_id, raw_genre) 세로형으로 반환.
    """
    g = read_csv_retry(RAW_GENRES)
    if g is None or g.empty:
        raise RuntimeError("content_raw_genres.csv 가 비어있거나 없습니다.")

    g = g.copy()
    g.columns = [c.strip() for c in g.columns]

    if "content_id" not in g.columns:
        raise RuntimeError("content_raw_genres.csv 에 content_id 컬럼이 없습니다.")

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
        raise RuntimeError("content_raw_genres.csv 에 raw_genre 또는 raw_genre_1/2/3 컬럼이 필요합니다.")

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


def main():
    # 1) 아이템 임베딩 + contents 로드
    emb = load_item_embeddings()
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    M = emb[emb_cols].to_numpy(np.float32)
    ids = emb["content_id"].to_numpy(int)

    contents = read_csv_retry(CONTENTS)
    if contents is None or contents.empty:
        raise RuntimeError("contents.csv 필요.")
    contents = contents[["content_id", "domain"]].copy()
    contents["content_id"] = pd.to_numeric(contents["content_id"], errors="coerce").astype("Int64")
    contents = contents[contents["content_id"].notna()].copy()
    contents["content_id"] = contents["content_id"].astype(int)
    emb_meta = emb.merge(contents, on="content_id", how="left")

    # (옵션) 도메인 정보 출력만
    dom_counts = {}
    for dom, grp in emb_meta.dropna(subset=["domain"]).groupby("domain"):
        dom_counts[str(dom)] = len(grp)
    if dom_counts:
        print(f"ℹ️ 도메인별 아이템 수 (참고용): { {k: dom_counts[k] for k in sorted(dom_counts)} }")

    # 3) 장르 센트로이드 계산 (content_raw_genres.csv + item 임베딩)
    if not os.path.exists(RAW_GENRES):
        raise RuntimeError("content_raw_genres.csv 없음 → 장르 센트로이드 계산 불가")

    rawg_tall = load_content_genres_tall()  # 항상 content_id, raw_genre 세로형으로 반환

    if rawg_tall is None or rawg_tall.empty:
        raise RuntimeError("content_raw_genres.csv 에 유효한 장르가 없습니다. → 장르 센트로이드 계산 불가")

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
        raise RuntimeError("장르별로 매칭되는 아이템 임베딩이 없습니다. genre_vecs 비어있음.")

    print(f"✅ 장르 센트로이드 수: {len(genre_vecs)}")

    # 4) user_preferred_genres.csv 로부터 유저별 선호 장르 가져오기
    ugen = read_csv_retry(UP_GENRES)
    if ugen is None or ugen.empty:
        raise RuntimeError("user_preferred_genres.csv 필요.")

    # 어떤 컬럼이 유저 식별자인지 추론 (user_id / id / preference_id 중 하나)
    col_user_id = None
    for cand in ["user_id", "id", "preference_id"]:
        if cand in ugen.columns:
            col_user_id = cand
            break
    if col_user_id is None:
        raise RuntimeError("user_preferred_genres.csv 에 user_id / id / preference_id 컬럼이 없습니다. 유저 식별 컬럼 하나 필요.")

    if "genre" not in ugen.columns:
        raise RuntimeError("user_preferred_genres.csv 에 genre 컬럼이 필요합니다.")

    has_username = "username" in ugen.columns

    keep_cols = [col_user_id, "genre"]
    if has_username:
        keep_cols.append("username")

    ugen = ugen[keep_cols].dropna(subset=[col_user_id, "genre"]).copy()
    ugen[col_user_id] = pd.to_numeric(ugen[col_user_id], errors="coerce").astype("Int64")
    ugen = ugen[ugen[col_user_id].notna()].copy()
    ugen[col_user_id] = ugen[col_user_id].astype(int)
    ugen["g"] = ugen["genre"].astype(str).str.strip().str.lower()

    # 5) 유저별로 선호 장르 → 장르 벡터 평균 → 유저 임베딩
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
        raise RuntimeError("유저 임베딩 생성 결과가 비었습니다. user_preferred_genres.csv / 장르 센트로이드 내용을 확인하세요.")

    dim = len(emb_cols)
    out = pd.DataFrame(rows, columns=["user_id", "username"] + [f"emb_{i}" for i in range(dim)])
    out.sort_values("user_id").to_csv(OUT_USER, index=False, encoding="utf-8-sig")
    print(f"✅ user_embeddings.csv 저장: {OUT_USER} (users={len(out)}, skipped_users_without_valid_genres={skipped})")


if __name__ == "__main__":
    main()
