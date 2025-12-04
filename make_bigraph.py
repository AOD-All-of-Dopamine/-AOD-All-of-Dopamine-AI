# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from typing import Optional, List

BASE = r"csv 데이터\clean"
PATH_CONTENTS = os.path.join(BASE, "contents.csv")
PATH_AV       = os.path.join(BASE, "av_contents.csv")
PATH_GAME     = os.path.join(BASE, "game_contents.csv")
PATH_WEBNOVEL = os.path.join(BASE, "webnovel_contents.csv")  # 있으면 사용

PATH_RAW_ITEM = os.path.join(BASE, "raw_item.csv")           # raw_id, genres_str

OUT_NODES     = os.path.join(BASE, "graph_nodes.csv")              # 콘텐츠 노드
OUT_RAWGEN    = os.path.join(BASE, "content_raw_genres.csv")       # (최종) content_id, source, raw_genre_1~3
OUT_META      = os.path.join(BASE, "meta_nodes.csv")               # 메타노드 목록
OUT_EDGES_BI  = os.path.join(BASE, "graph_edges_bipartite.csv")    # 콘텐츠→메타 엣지

# 엣지 가중치(필요 시 조정)
DOMAIN_EDGE_WEIGHT = 1.0
GENRE_EDGE_WEIGHT  = 1.0

ENCODINGS = ["utf-8-sig","utf-8","cp949","euc-kr","latin1"]
def read_csv_retry(path, **kwargs):
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last = e
    raise last

def safe_read(path):
    if not os.path.exists(path):
        print(f"⚠️ 없음: {path}")
        return None
    try:
        return read_csv_retry(path)
    except Exception as e:
        print(f"⚠️ 읽기 실패: {path} -> {e}")
        return None

def add_prefix_except_key(df, prefix, key="content_id"):
    if df is None: return None
    return df.rename(columns={c: (c if c==key else f"{prefix}{c}") for c in df.columns})

def norm_for_id(s: str) -> str:
    """
    메타노드 ID용 간단 정규화 (공백/구두점 제거, 소문자)
    예: 'Science Fiction' -> 'science_fiction'
    """
    if s is None: return ""
    s = str(s).strip().lower()
    # 한/영/숫자 외는 공백 치환
    s = re.sub(r"[^\w\s가-힣]", " ", s)
    s = "_".join(s.split())
    return s

def parse_genres(genres_str: str) -> List[str]:
    """
    '드라마|범죄|미스터리' → ['드라마','범죄','미스터리']
    공백/콤마/슬래시/· 같은 것도 구분자로 처리
    """
    if not isinstance(genres_str, str):
        return []
    genres_str = genres_str.strip()
    if not genres_str:
        return []
    parts = re.split(r"[|,/·;]", genres_str)
    parts = [g.strip() for g in parts if g.strip()]
    return parts

# 1) 콘텐츠 노드 생성(그대로 유지)
def build_nodes():
    contents = safe_read(PATH_CONTENTS)
    if contents is None or contents.empty:
        raise RuntimeError("❌ contents.csv 비어있음/로드 실패")
    if "content_id" not in contents.columns or "domain" not in contents.columns:
        raise RuntimeError("❌ contents.csv에 content_id/domain 컬럼 필요")

    av   = add_prefix_except_key(safe_read(PATH_AV), "av_")
    game = add_prefix_except_key(safe_read(PATH_GAME), "game_")
    web  = add_prefix_except_key(safe_read(PATH_WEBNOVEL), "webnovel_")

    def typed_left_merge(left, right, key="content_id"):
        if right is None or right.empty:
            return left.copy()
        L = left.copy(); R = right.copy()
        if key in L.columns: L[key] = pd.to_numeric(L[key], errors="coerce").astype("Int64")
        if key in R.columns: R[key] = pd.to_numeric(R[key], errors="coerce").astype("Int64")
        return L.merge(R, on=key, how="left")

    parts = []
    doms = contents["domain"].dropna().unique().tolist()
    if "AV" in doms:
        sub = contents[contents["domain"]=="AV"].copy()
        sub = typed_left_merge(sub, av); parts.append(sub)
    if "GAME" in doms:
        sub = contents[contents["domain"]=="GAME"].copy()
        sub = typed_left_merge(sub, game); parts.append(sub)
    if "WEBNOVEL" in doms:
        sub = contents[contents["domain"]=="WEBNOVEL"].copy()
        sub = typed_left_merge(sub, web); parts.append(sub)

    known = {"AV","GAME","WEBNOVEL"}
    nodes = pd.concat(parts, ignore_index=True) if parts else contents[contents["domain"].isin(known)].copy()
    nodes = nodes.drop_duplicates(subset=["content_id"], keep="first")

    # 보기 좋게 정렬
    first = [c for c in ["content_id","domain","master_title","original_title","release_year",
                         "poster_image_url","created_at","updated_at","synopsis"] if c in nodes.columns]
    nodes = nodes[first + [c for c in nodes.columns if c not in first]]

    nodes.to_csv(OUT_NODES, index=False, encoding="utf-8-sig")
    print(f"✅ graph_nodes 저장: {OUT_NODES} (rows={len(nodes)})")
    return nodes

# 2) 장르 표 생성
#   2-1) 우선 예전처럼 세로형(content_id, source, raw_genre)을 만들고
#   2-2) raw_item.csv의 genres_str를 이용해 최대 3개 장르로 가로형(raw_genre_1~3)으로 변환
def build_raw_genres(nodes):
    valid_ids = set(pd.to_numeric(nodes["content_id"], errors="coerce").dropna().astype(int))
    rows = []

    # ----- AV: TMDB 장르 (genres.tmdb_genres.N.name) -----
    av = safe_read(PATH_AV)
    if av is not None and not av.empty:
        av = av.copy()
        av.columns = [c.strip() for c in av.columns]
        name_cols = [c for c in av.columns if c.startswith("genres.tmdb_genres.") and c.endswith(".name")]
        if name_cols and "content_id" in av.columns:
            tmp = av.melt(id_vars=["content_id"], value_vars=name_cols, value_name="raw_genre").dropna(subset=["raw_genre"])
            tmp["raw_genre"] = tmp["raw_genre"].astype(str).str.strip()
            tmp = tmp[(tmp["raw_genre"]!="") & (tmp["content_id"].isin(valid_ids))]
            tmp["source"] = "tmdb"
            rows.append(tmp[["content_id","source","raw_genre"]])

    # ----- GAME: Steam 장르 (genres_str: "A;B;C") -----
    game = safe_read(PATH_GAME)
    if game is not None and not game.empty:
        game = game.copy()
        game.columns = [c.strip() for c in game.columns]
        if "genres_str" in game.columns and "content_id" in game.columns:
            g2 = game.dropna(subset=["genres_str"]).copy()
            g2["raw_genre"] = g2["genres_str"].astype(str).str.split(r"\s*;\s*")
            g2 = g2.explode("raw_genre").dropna(subset=["raw_genre"])
            g2["raw_genre"] = g2["raw_genre"].astype(str).str.strip()
            g2 = g2[(g2["raw_genre"]!="") & (g2["content_id"].isin(valid_ids))]
            g2["source"] = "steam"
            rows.append(g2[["content_id","source","raw_genre"]])

    # ----- WEBNOVEL: 장르 컬럼이 있다면 분해 (예: 'genres'가 "판타지;로맨스") -----
    web = safe_read(PATH_WEBNOVEL)
    if web is not None and not web.empty:
        web = web.copy()
        web.columns = [c.strip() for c in web.columns]
        cand_cols = [c for c in ["genres","genre","genre_str"] if c in web.columns]
        if cand_cols:
            col = cand_cols[0]
            w2 = web.dropna(subset=[col]).copy()
            w2["raw_genre"] = w2[col].astype(str).str.split(r"\s*[;|/,]\s*")
            w2 = w2.explode("raw_genre").dropna(subset=["raw_genre"])
            w2["raw_genre"] = w2["raw_genre"].astype(str).str.strip()
            if "content_id" in w2.columns:
                w2 = w2[w2["raw_genre"]!=""]
                w2 = w2[w2["content_id"].isin(valid_ids)]
                w2["source"] = "webnovel"
                rows.append(w2[["content_id","source","raw_genre"]])

    # ----- 2-1) tall: content_id, source, raw_genre -----
    if rows:
        cg = pd.concat(rows, ignore_index=True)
        cg["content_id"] = pd.to_numeric(cg["content_id"], errors="coerce").astype("Int64")
        cg = cg[cg["content_id"].notna()]
        cg["content_id"] = cg["content_id"].astype(int)
        cg["raw_genre"] = cg["raw_genre"].astype(str).str.strip()
        cg = cg[cg["raw_genre"]!=""]
        cg = cg.drop_duplicates().reset_index(drop=True)
    else:
        cg = pd.DataFrame(columns=["content_id","source","raw_genre"])

    # ----- 2-2) raw_item.csv 의 genres_str 를 이용해 raw_genre_1~3 생성 -----
    ri = safe_read(PATH_RAW_ITEM)
    if ri is not None and not ri.empty:
        ri = ri.copy()
        ri.columns = [c.strip() for c in ri.columns]

        needed_ri = ["raw_id", "genres_str"]
        miss_ri = [c for c in needed_ri if c not in ri.columns]
        if miss_ri:
            print("⚠️ raw_item.csv 에 다음 컬럼이 없어 genres_str를 사용할 수 없습니다:", ", ".join(miss_ri))
            ri = None
        else:
            ri["raw_id"] = pd.to_numeric(ri["raw_id"], errors="coerce").astype("Int64")
            ri = ri[ri["raw_id"].notna()].copy()
            ri["raw_id"] = ri["raw_id"].astype(int)
    else:
        ri = None

    if ri is not None and not cg.empty:
        merged = cg.merge(
            ri[["raw_id", "genres_str"]],
            left_on="content_id",
            right_on="raw_id",
            how="left"
        )
    else:
        merged = cg.copy()
        merged["genres_str"] = ""

    # raw_genre_1,2,3 생성
    new_rows = []
    for _, row in merged.iterrows():
        cid    = int(row["content_id"])
        source = str(row["source"]).strip()
        base_genre = str(row["raw_genre"]).strip()

        genres_from_str = parse_genres(row.get("genres_str", ""))

        genres = list(genres_from_str)
        # 기존 raw_genre가 있고 리스트에 없으면 맨 앞에 추가
        if base_genre and base_genre not in genres:
            genres.insert(0, base_genre)

        g1 = genres[0] if len(genres) > 0 else ""
        g2 = genres[1] if len(genres) > 1 else ""
        g3 = genres[2] if len(genres) > 2 else ""

        new_rows.append((cid, source, g1, g2, g3))

    wide = pd.DataFrame(new_rows, columns=["content_id","source","raw_genre_1","raw_genre_2","raw_genre_3"])
    wide = wide.drop_duplicates(subset=["content_id","source","raw_genre_1","raw_genre_2","raw_genre_3"])

    wide.to_csv(OUT_RAWGEN, index=False, encoding="utf-8-sig")
    print(f"✅ content_raw_genres (wide) 저장: {OUT_RAWGEN} (rows={len(wide)})")
    return wide

# wide(raw_genre_1~3) → tall(raw_genre) 변환 (그래프 내부용)
def to_tall_raw_genres(raw_genres: pd.DataFrame) -> pd.DataFrame:
    df = raw_genres.copy()
    df.columns = [c.strip() for c in df.columns]

    # 이미 raw_genre 단일 컬럼이면 그대로 사용
    if "raw_genre" in df.columns and not any(
        c in df.columns for c in ["raw_genre_1","raw_genre_2","raw_genre_3"]
    ):
        df["content_id"] = pd.to_numeric(df["content_id"], errors="coerce").astype("Int64")
        df = df[df["content_id"].notna()].copy()
        df["content_id"] = df["content_id"].astype(int)
        df["raw_genre"] = df["raw_genre"].astype(str).str.strip()
        df = df[df["raw_genre"]!=""]
        return df[["content_id","source","raw_genre"]].drop_duplicates().reset_index(drop=True)

    # 가로형 raw_genre_1~3 → 세로형
    genre_cols = [c for c in ["raw_genre_1","raw_genre_2","raw_genre_3"] if c in df.columns]
    if not genre_cols:
        raise RuntimeError("raw_genres에 raw_genre 또는 raw_genre_1/2/3 컬럼이 없습니다.")

    df["content_id"] = pd.to_numeric(df["content_id"], errors="coerce").astype("Int64")
    df = df[df["content_id"].notna()].copy()
    df["content_id"] = df["content_id"].astype(int)

    tall = df.melt(
        id_vars=["content_id","source"],
        value_vars=genre_cols,
        value_name="raw_genre"
    )
    tall["raw_genre"] = tall["raw_genre"].astype(str).str.strip()
    tall = tall[tall["raw_genre"].notna() & (tall["raw_genre"]!="")]
    tall = tall[["content_id","source","raw_genre"]].drop_duplicates()

    print(f"✅ raw_genre_1~3 → 세로형 변환: rows={len(tall)}")
    return tall.reset_index(drop=True)

# 3) 메타노드 + 이분 엣지 생성
def build_bipartite(nodes, raw_genres):
    # 먼저 wide(raw_genre_1~3)를 tall(raw_genre)로 변환
    tall = to_tall_raw_genres(raw_genres) if raw_genres is not None and not raw_genres.empty else None

    # 메타노드: 도메인
    domain_nodes = sorted(nodes["domain"].dropna().unique().tolist())

    meta_rows = []
    for d in domain_nodes:
        meta_rows.append({"meta_id": f"DOM:{d}", "meta_type": "domain", "label": d, "source": ""})

    # 메타노드: 장르 (네임스페이스 적용)
    if tall is not None and not tall.empty:
        rg = tall.copy()
        rg["raw_genre_norm"] = rg["raw_genre"].astype(str).str.strip()
        rg["raw_genre_norm"] = rg["raw_genre_norm"].replace({"": None})
        rg = rg.dropna(subset=["raw_genre_norm"])
        uniq = rg[["source","raw_genre_norm"]].drop_duplicates()
        for src, g in uniq.itertuples(index=False):
            meta_id = f"GEN:{src}:{norm_for_id(g)}" if g else None
            if meta_id:
                meta_rows.append({"meta_id": meta_id, "meta_type": "genre", "label": g, "source": src})

    df_meta = pd.DataFrame(meta_rows, columns=["meta_id","meta_type","label","source"]).drop_duplicates()
    df_meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")
    print(f"✅ meta_nodes 저장: {OUT_META} (rows={len(df_meta)})")

    # 이분 엣지: 콘텐츠 → 도메인
    edges = []
    sub = nodes[["content_id","domain"]].dropna()
    for cid, dom in sub.itertuples(index=False):
        edges.append({
            "src_content_id": int(cid),
            "dst_meta_id":    f"DOM:{dom}",
            "edge_type":      "content-domain",
            "weight":         DOMAIN_EDGE_WEIGHT
        })

    # 이분 엣지: 콘텐츠 → 장르 (한 콘텐츠가 여러 장르와 연결됨)
    if tall is not None and not tall.empty:
        for cid, src, g in tall[["content_id","source","raw_genre"]].itertuples(index=False):
            g_norm = norm_for_id(g)
            if not g_norm:
                continue
            meta_id = f"GEN:{src}:{g_norm}"
            edges.append({
                "src_content_id": int(cid),
                "dst_meta_id":    meta_id,
                "edge_type":      f"content-genre-{src}",
                "weight":         GENRE_EDGE_WEIGHT
            })

    df_edges = pd.DataFrame(edges, columns=["src_content_id","dst_meta_id","edge_type","weight"]).drop_duplicates()
    df_edges.to_csv(OUT_EDGES_BI, index=False, encoding="utf-8-sig")
    print(f"✅ graph_edges_bipartite 저장: {OUT_EDGES_BI} (edges={len(df_edges)})")
    return df_meta, df_edges

def main():
    nodes = build_nodes()
    rawg  = build_raw_genres(nodes)  # 여기서 content_raw_genres.csv (raw_genre_1~3) 생성
    build_bipartite(nodes, rawg)    # 여기서 한 콘텐츠가 여러 장르 노드와 엣지로 연결됨

if __name__ == "__main__":
    main()
