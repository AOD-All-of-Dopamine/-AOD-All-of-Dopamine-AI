# -*- coding: utf-8 -*-

#도메인별 Top-N 추천 (유저/아이템 임베딩 + contents.csv)
#입력:
#  clean/user_embeddings.csv                  (user_id, username, emb_0..)
#  clean/item_embeddings_torch.csv 또는 item_embeddings.csv (content_id, emb_0..)
#  clean/contents.csv                         (content_id, domain, title/name/...)
#출력:
#  clean/recommendations_per_domain.csv       (user_id, username, rank, content_id, content_title, score, domain)

import os, re
import numpy as np
import pandas as pd
from typing import Optional, Dict

# ===== 경로 =====
BASE = r"csv 데이터\clean"
USER_EMB   = os.path.join(BASE, "user_embeddings.csv")
ITEM_EMB_1 = os.path.join(BASE, "item_embeddings_torch.csv")
ITEM_EMB_2 = os.path.join(BASE, "item_embeddings.csv")
CONTENTS   = os.path.join(BASE, "contents.csv")
OUT_CSV    = os.path.join(BASE, "recommendations_per_domain.csv")

# ===== 설정 =====
DOMAIN_QUOTA: Dict[str, int] = {"AV": 3, "GAME": 3, "WEBNOVEL": 3}   # 도메인별 개수
FILL_WITH_GLOBAL = True    # 도메인 후보 부족 시 글로벌 Top으로 보충
ASSUME_NORMALIZED = True   # 임베딩이 이미 L2 정규화라면 True
EXCLUDE_DUP_CONTENTS = True

# ===== 유틸 =====
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
    df = df.copy()
    orig_cols = list(df.columns)
    df.columns = [c.strip() for c in df.columns]
    name_map = {c.lower(): c for c in df.columns}
    for key in ["content_id", "contentid", "content id", "id", "unnamed: 0"]:
        if key in name_map:
            s = df[name_map[key]]
            if s.dtype == object:
                s = s.astype(str).str.strip()
            s = pd.to_numeric(s, errors="coerce").astype("Int64")
            if not s.isna().all():
                return s
    raise RuntimeError(f"content_id 열을 찾지 못했습니다. 원본 컬럼: {orig_cols}")

def load_user_embeddings() -> pd.DataFrame:
    u = read_csv_retry(USER_EMB)
    if u is None or u.empty:
        raise RuntimeError("user_embeddings.csv 필요.")
    u = u.copy()
    u.columns = [c.strip() for c in u.columns]
    if "user_id" not in u.columns:
        if "id" in u.columns:
            u = u.rename(columns={"id": "user_id"})
        else:
            raise RuntimeError("user_embeddings.csv에 user_id 컬럼 필요.")
    u["user_id"] = pd.to_numeric(u["user_id"], errors="coerce").astype("Int64")
    u = u[u["user_id"].notna()].copy()
    u["user_id"] = u["user_id"].astype(int)

    emb_cols = [c for c in u.columns if c.lower().startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("user_embeddings.csv에 emb_로 시작하는 임베딩 컬럼이 없습니다.")
    for c in emb_cols:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    u.dropna(subset=emb_cols, inplace=True)
    if not ASSUME_NORMALIZED:
        u.loc[:, emb_cols] = l2norm(u[emb_cols].to_numpy(np.float32))

    # username 컬럼 유연 처리
    uname_col = None
    for c in u.columns:
        if c.lower() == "username":
            uname_col = c; break
    if uname_col is None:
        u["username"] = ""
        uname_col = "username"

    return u[["user_id", uname_col] + emb_cols].rename(columns={uname_col: "username"})

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

# 제목 컬럼 자동 탐지: title / name / content_name / ko_title / ko_name 등
def detect_title_column(df: pd.DataFrame) -> Optional[str]:
    cand_list = ["title", "name", "content_name", "ko_title", "ko_name", "master_title"]
    cols_lower = {c.lower(): c for c in df.columns}
    for c in cand_list:
        if c in cols_lower:
            return cols_lower[c]
    return None


def main():
    # --- 데이터 로드 ---
    u = load_user_embeddings()
    i = load_item_embeddings()

    contents = read_csv_retry(CONTENTS)
    if contents is None or contents.empty or {"content_id","domain"} - set(contents.columns):
        raise RuntimeError("contents.csv 필요 (content_id, domain).")
    contents = contents.copy()
    contents.columns = [c.strip() for c in contents.columns]

    # content_id 정리
    contents["content_id"] = pd.to_numeric(contents["content_id"], errors="coerce").astype("Int64")
    contents = contents[contents["content_id"].notna()].copy()
    contents["content_id"] = contents["content_id"].astype(int)

    # ★ 제목 컬럼 찾기
    title_col = detect_title_column(contents)
    if title_col is None:
        raise RuntimeError("contents.csv에서 제목 컬럼(title/name/...)을 찾지 못했습니다. (타이틀 없는 콘텐츠 제외를 위해 제목 컬럼이 필요합니다.)")
    else:
        print(f"✅ 제목 컬럼 발견: {title_col}")

    # ★ 제목이 비어있는 콘텐츠 제외
    contents[title_col] = contents[title_col].astype(str).str.strip()
    contents = contents[contents[title_col] != ""].copy()

    # ★ 제목 있는 콘텐츠의 content_id만 유효
    valid_ids = set(contents["content_id"].unique())

    # ★ 아이템 임베딩도 valid_ids만 남기기 → 타이틀 없는 콘텐츠는 추천 후보에서 제거
    i = i[i["content_id"].isin(valid_ids)].reset_index(drop=True)

    # 도메인 맵은 필터링된 contents 기준
    dom_map = contents.set_index("content_id")["domain"].to_dict()

    ucols = [c for c in u.columns if c.lower().startswith("emb_")]
    icols = [c for c in i.columns if c.lower().startswith("emb_")]

    U = u[ucols].to_numpy(np.float32); U = l2norm(U)   # 안전하게 한 번 더 정규화
    I = i[icols].to_numpy(np.float32); I = l2norm(I)
    item_ids = i["content_id"].to_numpy(int)

    rec_rows = []
    for r in range(U.shape[0]):
        uvec = U[r:r+1, :]                   # [1, D]
        sims = (I @ uvec.T).reshape(-1)      # [N,]

        # 1) 도메인별 quota 채우기
        picked_mask = np.zeros_like(sims, dtype=bool)
        per_domain_picks = []
        for dom, need in DOMAIN_QUOTA.items():
            if need <= 0: 
                continue
            # 해당 도메인 아이템 마스크
            dom_mask = np.array([dom_map.get(int(cid)) == dom for cid in item_ids], dtype=bool)
            if not dom_mask.any():
                continue
            # 아직 안 뽑힌 + 해당 도메인
            cand_mask = dom_mask & (~picked_mask)
            if not cand_mask.any():
                continue
            cand_idx = np.where(cand_mask)[0]
            cand_scores = sims[cand_idx]

            k = min(need, len(cand_idx))
            part = np.argpartition(-cand_scores, k-1)[:k]
            order = part[np.argsort(-cand_scores[part])]
            chosen = cand_idx[order]

            picked_mask[chosen] = True
            for j in chosen:
                per_domain_picks.append((int(item_ids[j]), float(sims[j]), dom))

        # 2) 부족분 글로벌 Top으로 보충(옵션)
        total_need = sum(max(0, n) for n in DOMAIN_QUOTA.values())
        if FILL_WITH_GLOBAL and picked_mask.sum() < total_need:
            remain = total_need - picked_mask.sum()
            left_idx = np.where(~picked_mask)[0]
            if len(left_idx) > 0:
                left_scores = sims[left_idx]
                k2 = min(remain, len(left_idx))
                part = np.argpartition(-left_scores, k2-1)[:k2]
                order = part[np.argsort(-left_scores[part])]

                chosen = left_idx[order]
                for j in chosen:
                    c = int(item_ids[j])
                    per_domain_picks.append((c, float(sims[j]), dom_map.get(c, "")))
                    picked_mask[j] = True

        # 3) rank 매기고 저장
        per_domain_picks.sort(key=lambda x: -x[1])
        uid = int(u.loc[r, "user_id"]); uname = str(u.loc[r, "username"])
        for rank, (cid, sc, dom) in enumerate(per_domain_picks, start=1):
            rec_rows.append([uid, uname, rank, cid, sc, dom])

    rec = pd.DataFrame(rec_rows, columns=["user_id","username","rank","content_id","score","domain"])

    # 제목 join 해서 content_title 컬럼 추가 (여기서 contents는 이미 "제목 있는 것만"이라, 빈 타이틀 없음)
    name_df = contents[["content_id", title_col]].drop_duplicates("content_id")
    rec = rec.merge(name_df, on="content_id", how="left")
    rec = rec.rename(columns={title_col: "content_title"})

    # 저장 컬럼 순서
    rec = rec.sort_values(["user_id","rank"])
    rec_cols = ["user_id","username","rank","content_id","content_title","score","domain"]
    rec[rec_cols].to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ recommendations_per_domain.csv 저장: {OUT_CSV} (rows={len(rec)})")

    # 콘솔 요약
    try:
        for uid in sorted(rec["user_id"].unique()):
            sub = rec[rec["user_id"]==uid].sort_values("rank")
            print(f"\n[User {uid}] 총 {len(sub)}개")
            for _, row in sub.iterrows():
                title = ""
                if isinstance(row["content_title"], str) and row["content_title"]:
                    title = f"  name={row['content_title']}"
                print(f"  #{int(row['rank']):2d}  cid={int(row['content_id'])}  dom={row['domain']}  sim={row['score']:.4f}{title}")
    except Exception:
        pass

if __name__ == "__main__":
    main()
