# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
from collections import defaultdict

BASE = r"csv 데이터\clean"
PATH_NODES   = os.path.join(BASE, "graph_nodes.csv")
PATH_RAWGEN  = os.path.join(BASE, "content_raw_genres.csv")
OUT_EDGES    = os.path.join(BASE, "graph_edges_item_item.csv")

# ========= 하이퍼파라미터(조절 포인트) =========
DOMAIN_BASE_WEIGHT    = 1.0   # 같은 도메인 공유 기여치
GENRE_BASE_WEIGHT     = 1.0   # 같은 '도메인별 원본장르' 공유 기여치
ALPHA                 = 0.7   # 허브 완화 지수(0=완화 없음, 1=강한 완화). 보통 0.5~0.8
USE_IDF               = True  # 메타노드 정보량 가중치(log(1+N/deg)) 사용
TOPK_PER_ITEM         = 100   # 아이템당 상위 K 이웃만 유지 (희소화)
MAX_MEMBERS_PER_META  = 2000  # 메타노드 멤버가 너무 클 때 샘플링 상한 (메모리/시간 보호)

ENCODINGS = ["utf-8-sig","utf-8","cp949","euc-kr","latin1"]

def read_csv_retry(path, **kwargs):
    last = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last = e
    raise last

def to_tall_raw_genres(rawg: pd.DataFrame) -> pd.DataFrame:
    """
    content_raw_genres.csv가
      - (A) 세로형: content_id, source, raw_genre
      - (B) 가로형: content_id, source, raw_genre_1~3
    둘 중 어느 포맷이든 받을 수 있게 하고,
    최종적으로 항상 세로형 (content_id, source, raw_genre) 로 변환.
    """
    df = rawg.copy()
    df.columns = [c.strip() for c in df.columns]

    # (A) 이미 raw_genre 단일 컬럼이 있으면 그대로 정리해서 반환
    if "raw_genre" in df.columns:
        if "content_id" not in df.columns or "source" not in df.columns:
            raise RuntimeError("raw_genres에 content_id/source/raw_genre 컬럼이 모두 필요합니다.")
        df["content_id"] = pd.to_numeric(df["content_id"], errors="coerce").astype("Int64")
        df = df[df["content_id"].notna()].copy()
        df["content_id"] = df["content_id"].astype(int)
        df["raw_genre"] = df["raw_genre"].astype(str).str.strip()
        df = df[df["raw_genre"] != ""]
        return df[["content_id", "source", "raw_genre"]].drop_duplicates().reset_index(drop=True)

    # (B) 가로형(raw_genre_1~3)을 세로형으로 변환
    genre_cols = [c for c in ["raw_genre_1", "raw_genre_2", "raw_genre_3"] if c in df.columns]
    if not genre_cols:
        raise RuntimeError("content_raw_genres.csv에 raw_genre 또는 raw_genre_1/2/3 컬럼이 없습니다.")

    if "content_id" not in df.columns or "source" not in df.columns:
        raise RuntimeError("content_raw_genres.csv에 content_id/source 컬럼이 필요합니다.")

    df["content_id"] = pd.to_numeric(df["content_id"], errors="coerce").astype("Int64")
    df = df[df["content_id"].notna()].copy()
    df["content_id"] = df["content_id"].astype(int)

    tall = df.melt(
        id_vars=["content_id", "source"],
        value_vars=genre_cols,
        value_name="raw_genre"
    )
    tall["raw_genre"] = tall["raw_genre"].astype(str).str.strip()
    tall = tall[tall["raw_genre"].notna() & (tall["raw_genre"] != "")]
    tall = tall[["content_id", "source", "raw_genre"]].drop_duplicates()

    print(f"✅ raw_genre_1~3 → 세로형 변환: rows={len(tall)}")
    return tall.reset_index(drop=True)

def main():
    # 1) 노드 로드
    nodes = read_csv_retry(PATH_NODES)
    if nodes is None or nodes.empty:
        raise RuntimeError("graph_nodes.csv 비어있음/로드 실패")
    if "content_id" not in nodes.columns or "domain" not in nodes.columns:
        raise RuntimeError("graph_nodes.csv에 content_id/domain 컬럼 필요")

    nodes = nodes.dropna(subset=["content_id", "domain"]).copy()
    nodes["content_id"] = pd.to_numeric(nodes["content_id"], errors="coerce").astype("Int64")
    nodes = nodes[nodes["content_id"].notna()]
    nodes["content_id"] = nodes["content_id"].astype(int)

    # 도메인 맵
    domain_of = dict(nodes[["content_id", "domain"]].values)
    all_items = set(domain_of.keys())
    N_items   = max(1, len(all_items))

    # 2) content_raw_genres 로드 (wide → tall 변환 포함)
    if os.path.exists(PATH_RAWGEN):
        rawg_raw = read_csv_retry(PATH_RAWGEN)
        if rawg_raw is None or rawg_raw.empty:
            rawg = pd.DataFrame(columns=["content_id", "source", "raw_genre"])
        else:
            rawg = to_tall_raw_genres(rawg_raw)
    else:
        print("⚠️ content_raw_genres.csv 없음 → 장르 기반 기여 없이 도메인만 사용")
        rawg = pd.DataFrame(columns=["content_id", "source", "raw_genre"])

    # 3) 메타노드 멤버 구성
    members_of_meta = defaultdict(list)

    # (a) 도메인 메타노드: DOM:<domain>
    for cid, dom in domain_of.items():
        members_of_meta[f"DOM:{dom}"].append(cid)

    # (b) 도메인별 장르 메타노드: GEN:<source>:<raw_genre>@<domain>
    if not rawg.empty:
        rawg = rawg.dropna(subset=["content_id", "source", "raw_genre"]).copy()
        rawg["content_id"] = pd.to_numeric(rawg["content_id"], errors="coerce").astype("Int64")
        rawg = rawg[rawg["content_id"].notna()]
        rawg["content_id"] = rawg["content_id"].astype(int)

        for cid, src, g in rawg[["content_id", "source", "raw_genre"]].drop_duplicates().itertuples(index=False):
            dom = domain_of.get(cid)
            if not dom:
                continue
            meta_id = f"GEN:{src}:{g}@{dom}"
            members_of_meta[meta_id].append(cid)

    # 4) 메타노드 공동소속 기반 아이템↔아이템 가중치 계산
    def meta_contrib(meta_id, deg):
        # deg = 메타노드 멤버 수
        base = DOMAIN_BASE_WEIGHT if meta_id.startswith("DOM:") else GENRE_BASE_WEIGHT
        if deg <= 1:
            return 0.0
        # 허브 완화: 1 / (deg^ALPHA)
        w = base / (deg ** ALPHA)
        # 정보량(IDF) 보정: log(1 + N / deg)
        if USE_IDF:
            w *= math.log(1.0 + (N_items / float(deg)))
        return w

    edges = defaultdict(float)

    for meta_id, members in members_of_meta.items():
        members = list(set(members))
        deg = len(members)
        if deg <= 1:
            continue

        # 너무 큰 메타노드(예: 특정 도메인 전체)가 폭발하지 않게 샘플링/상한
        if deg > MAX_MEMBERS_PER_META:
            members = members[:MAX_MEMBERS_PER_META]
            deg = len(members)

        contrib = meta_contrib(meta_id, deg)
        if contrib == 0.0:
            continue

        # 근린 제한: 각 멤버마다 앞쪽 K개와만 연결 (전쌍 O(n^2) 방지)
        K = min(TOPK_PER_ITEM, deg - 1)
        members.sort()
        for i, a in enumerate(members):
            for b in members[i+1 : i+1+K]:
                # 서로 도메인이 다르면 스킵 (안전장치)
                if domain_of.get(a) != domain_of.get(b):
                    continue
                if a > b:
                    a, b = b, a
                edges[(a, b)] += contrib

    # 5) 아이템당 TOPK 이웃만 유지 (양방향 기준 최대 가중치 유지)
    nbrs = defaultdict(list)
    for (a, b), w in edges.items():
        nbrs[a].append((b, w))
        nbrs[b].append((a, w))

    pruned = {}
    for a, lst in nbrs.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        for b, w in lst[:TOPK_PER_ITEM]:
            key = (a, b) if a < b else (b, a)
            if key not in pruned or pruned[key] < w:
                pruned[key] = w

    df_edges = pd.DataFrame(
        [(a, b, w) for (a, b), w in pruned.items()],
        columns=["src_content_id", "dst_content_id", "weight"]
    )
    df_edges.to_csv(OUT_EDGES, index=False, encoding="utf-8-sig")
    print(f"✅ 저장: {OUT_EDGES} (edges={len(df_edges):,}, items={len(all_items):,})")

if __name__ == "__main__":
    main()
