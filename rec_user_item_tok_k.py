# -*- coding: utf-8 -*-

#유저/아이템 임베딩만으로 Top-K 추천 (contents.csv에서 이름/도메인까지 붙이기)
#입력:
#  clean/user_embeddings.csv                  (user_id, username, emb_0..)
#  clean/item_embeddings_torch.csv 또는 item_embeddings.csv (content_id, emb_0..)
#  clean/contents.csv                         (content_id, master_title, domain)
#  clean/user_preferences.csv                 (선택) id, username  ← 추가로 사용
#출력:
#  clean/recommendations_topK.csv             (user_id, username, rank, content_id, master_title, domain, score)

import os
import re
import numpy as np
import pandas as pd
from typing import Optional

BASE = r"csv 데이터\clean"
USER_EMB   = os.path.join(BASE, "user_embeddings.csv")
ITEM_EMB_1 = os.path.join(BASE, "item_embeddings_torch.csv")
ITEM_EMB_2 = os.path.join(BASE, "item_embeddings.csv")
CONTENTS   = os.path.join(BASE, "contents.csv")
UPREF      = os.path.join(BASE, "user_preferences.csv")   # ★ 추가
OUT_RECS   = os.path.join(BASE, "recommendations_topK.csv")

TOPK = 5
EXCLUDE_DUP_CONTENTS = True   # content_id 중복행이 있으면 첫 번째만 사용
ASSUME_NORMALIZED = True      # 임베딩이 이미 L2 정규화되어 있다면 True, 아니면 False로

def read_csv_retry(path, encodings=("utf-8-sig","utf-8","cp949","euc-kr","latin1"), **kw) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    last=None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except Exception as e:
            last=e
    raise last

def l2norm(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

def _coerce_content_id_column(df: pd.DataFrame) -> pd.Series:
    """여러 이름/형태로 저장된 content_id를 최대한 복구."""
    df = df.copy()
    orig_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    name_map = {c.lower(): c for c in df.columns}
    for key in ["content_id", "contentid", "content id", "id", "unnamed: 0"]:
        if key in name_map:
            cand = name_map[key]
            s = df[cand]
            if s.dtype == object:
                s = s.astype(str).str.strip()
            s = pd.to_numeric(s, errors="coerce").astype("Int64")
            if s.isna().all():
                continue
            return s
    raise RuntimeError(f"content_id 열을 찾지 못했습니다. 원본 컬럼: {orig_cols}")

def load_user_embeddings() -> pd.DataFrame:
    u = read_csv_retry(USER_EMB)
    if u is None or u.empty:
        raise RuntimeError("user_embeddings.csv 필요.")
    u = u.copy()
    u.columns = [c.strip() for c in u.columns]

    # user_id 정리
    if "user_id" not in u.columns:
        if "id" in u.columns:
            u = u.rename(columns={"id": "user_id"})
        else:
            raise RuntimeError("user_embeddings.csv에 user_id 컬럼 필요.")
    u["user_id"] = pd.to_numeric(u["user_id"], errors="coerce").astype("Int64")
    u = u[u["user_id"].notna()].copy()
    u["user_id"] = u["user_id"].astype(int)

    # 임베딩 컬럼
    emb_cols = [c for c in u.columns if c.lower().startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("user_embeddings.csv에 emb_로 시작하는 임베딩 컬럼이 없습니다.")
    for c in emb_cols:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    u.dropna(subset=emb_cols, inplace=True)
    if not ASSUME_NORMALIZED:
        u.loc[:, emb_cols] = l2norm(u[emb_cols].to_numpy(np.float32))

    # username 컬럼 찾기 (대소문자 무시)
    lower_cols = [x.lower() for x in u.columns]
    if "username" in lower_cols:
        uname_col = u.columns[lower_cols.index("username")]
        cols = ["user_id", uname_col] + emb_cols
        out = u[cols].copy()
        if uname_col != "username":
            out = out.rename(columns={uname_col: "username"})
    else:
        cols = ["user_id"] + emb_cols
        out = u[cols].copy()
        out["username"] = ""

    # 🔹 1단계: user_preferences.csv에서 username 보강
    pref = read_csv_retry(UPREF)
    if pref is not None and not pref.empty:
        pref = pref.copy()
        pref.columns = [c.strip() for c in pref.columns]
        lower = {c.lower(): c for c in pref.columns}

        id_col    = lower.get("id") or lower.get("user_id")
        uname_col = lower.get("username") or lower.get("user_name")

        if id_col is not None and uname_col is not None:
            pref = pref[[id_col, uname_col]].dropna(subset=[id_col])
            pref[id_col] = pd.to_numeric(pref[id_col], errors="coerce").astype("Int64")
            pref = pref[pref[id_col].notna()].copy()
            pref[id_col] = pref[id_col].astype(int)
            pref = pref.rename(columns={id_col: "user_id", uname_col: "username_pref"})

            out = out.merge(pref, on="user_id", how="left")
            # user_embeddings의 username이 비어있으면 preferences의 username_pref로 채우기
            out["username"] = out["username"].astype(str)
            out["username_pref"] = out["username_pref"].astype(str)

            # embeddings 쪽 username이 없거나 "nan"/"None" 같은 값이면 pref 것으로 덮어쓰기
            mask_missing = out["username"].str.strip().eq("") | \
                           out["username"].str.strip().str.lower().isin(["nan", "none"])
            out.loc[mask_missing, "username"] = out.loc[mask_missing, "username_pref"]
            out = out.drop(columns=["username_pref"])

    # 🔹 2단계: 최종적으로 남은 이상값 정리
    out["username"] = out["username"].replace({np.nan: ""})
    out["username"] = out["username"].astype(str)
    bad = out["username"].str.strip().str.lower().isin(["nan", "none"])
    out.loc[bad, "username"] = ""

    return out[["user_id", "username"] + emb_cols]

def load_item_embeddings() -> pd.DataFrame:
    i1 = read_csv_retry(ITEM_EMB_1)
    if i1 is not None and not i1.empty:
        df = i1; src = os.path.basename(ITEM_EMB_1)
    else:
        i2 = read_csv_retry(ITEM_EMB_2)
        if i2 is None or i2.empty:
            raise RuntimeError("item_embeddings CSV 필요: item_embeddings_torch.csv 또는 item_embeddings.csv")
        df = i2; src = os.path.basename(ITEM_EMB_2)

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["content_id"] = _coerce_content_id_column(df)

    emb_cols = [c for c in df.columns if c.lower().startswith("emb_")]
    if not emb_cols:
        alt = [c for c in df.columns if re.search(r"\d+$", c)]
        raise RuntimeError(f"임베딩 컬럼(emb_*) 없음. 확인 필요. 후보: {alt}")

    for c in emb_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=emb_cols + ["content_id"]).copy()
    df["content_id"] = df["content_id"].astype(int)

    if EXCLUDE_DUP_CONTENTS:
        df = df.drop_duplicates(subset=["content_id"], keep="first").reset_index(drop=True)

    if not ASSUME_NORMALIZED:
        df.loc[:, emb_cols] = l2norm(df[emb_cols].to_numpy(np.float32))

    print(f"✅ 아이템 임베딩 로드: {src} (items={len(df)}, dim={len(emb_cols)})")
    return df[["content_id"] + emb_cols]

def load_contents_meta() -> Optional[pd.DataFrame]:
    """
    contents.csv에서 content_id, master_title, domain만 뽑아서 리턴.
    content_id는 위의 _coerce_content_id_column 로 통일.
    """
    c = read_csv_retry(CONTENTS)
    if c is None or c.empty:
        print("⚠️ contents.csv를 찾지 못해 이름/도메인 매핑 없이 저장합니다.")
        return None

    c = c.copy()
    c.columns = [col.strip() for col in c.columns]

    # content_id 강제 추출
    c["content_id"] = _coerce_content_id_column(c)
    c = c[c["content_id"].notna()].copy()
    c["content_id"] = c["content_id"].astype(int)

    # master_title / domain 컬럼 이름(대소문자 무시) 찾기
    lower_map = {col.lower(): col for col in c.columns}
    title_col = lower_map.get("master_title")
    domain_col = lower_map.get("domain")

    if title_col is None:
        raise RuntimeError("contents.csv에서 master_title 컬럼을 찾지 못했습니다.")
    if domain_col is None:
        raise RuntimeError("contents.csv에서 domain 컬럼을 찾지 못했습니다.")

    meta = c[["content_id", title_col, domain_col]].copy()
    # 컬럼 이름 통일
    if title_col != "master_title":
        meta = meta.rename(columns={title_col: "master_title"})
    if domain_col != "domain":
        meta = meta.rename(columns={domain_col: "domain"})

    meta = meta.drop_duplicates(subset=["content_id"], keep="first").reset_index(drop=True)
    print(f"✅ contents 메타 로드: {len(meta)} rows (content_id, master_title, domain)")
    return meta

def main():
    u = load_user_embeddings()
    i = load_item_embeddings()
    meta = load_contents_meta()  # 이름/도메인 매핑

    ucols = [c for c in u.columns if c.lower().startswith("emb_")]
    icols = [c for c in i.columns if c.lower().startswith("emb_")]

    U = u[ucols].to_numpy(np.float32)
    I = i[icols].to_numpy(np.float32)

    # 혹시 ASSUME_NORMALIZED=True인데 실제로 정규화 안되어 있으면, 코사인 = 내적이 아닐 수 있으니 보정
    U = l2norm(U)
    I = l2norm(I)

    item_ids = i["content_id"].to_numpy(int)
    user_ids = u["user_id"].to_numpy(int)
    usernames = u["username"].astype(str).to_numpy() if "username" in u.columns else np.array([""]*len(u))

    rec_rows = []
    for r in range(U.shape[0]):
        sims = (I @ U[r:r+1].T).reshape(-1)   # 코사인(정규화 가정)
        k = min(TOPK, len(sims))
        if k == 0:
            continue
        part = np.argpartition(-sims, k-1)[:k]
        order = part[np.argsort(-sims[part])]
        top_ids = item_ids[order]
        top_scs = sims[order]
        uid = int(user_ids[r]); uname = usernames[r] if r < len(usernames) else ""
        for rank, (cid, sc) in enumerate(zip(top_ids, top_scs), start=1):
            rec_rows.append([uid, uname, rank, int(cid), float(sc)])

    rec = pd.DataFrame(rec_rows, columns=["user_id","username","rank","content_id","score"])

    # contents 메타와 merge 해서 master_title, domain 추가
    if meta is not None:
        rec = rec.merge(meta, on="content_id", how="left")
        cols = ["user_id", "username", "rank", "content_id", "master_title", "domain", "score"]
        cols = [c for c in cols if c in rec.columns]
        rec = rec[cols]

    rec.sort_values(["user_id","rank"]).to_csv(OUT_RECS, index=False, encoding="utf-8-sig")
    print(f"✅ recommendations_topK.csv 저장: {OUT_RECS} (rows={len(rec)})")

if __name__ == "__main__":
    main()
