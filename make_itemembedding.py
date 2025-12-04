# learn_item_embeddings.py
# -*- coding: utf-8 -*-

#아이템-아이템 가중 그래프 -> Node2Vec 스타일 임베딩 (Pure PyTorch, gensim/scipy 불필요)
#입력 : C:\Users\LG\Desktop\2025-2\A.O.D\db_export\clean\graph_edges_item_item.csv
#출력 : C:\Users\LG\Desktop\2025-2\A.O.D\db_export\clean\item_embeddings_torch.csv
#사용 : python learn_item_embeddings.py
#필요 : pip install torch pandas numpy
import os, math, random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# ===== 경로 =====
BASE = r"csv 데이터\clean"
EDGES_PATH = os.path.join(BASE, "graph_edges_item_item.csv")
OUT_EMB_CSV = os.path.join(BASE, "item_embeddings_torch.csv")

# ===== 하이퍼파라미터 =====
DIM            = 64       # 임베딩 차원
WALK_LENGTH    = 40       # 랜덤워크 길이
NUM_WALKS      = 10       # 노드당 워크 수
P_RETURN       = 1.0      # node2vec p (되돌아가기 선호 p↓)
Q_INOUT        = 1.0      # node2vec q (원거리 탐색 선호 q↑)
WINDOW         = 5        # Skip-gram 윈도우
NEGATIVE_K     = 5        # 네거티브 샘플 수
EPOCHS         = 3
BATCH_SIZE     = 8192
LR             = 0.025
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- 그래프 로드 & 가중 룰렛 준비 ----------
def load_graph(edges_csv):
    df = pd.read_csv(edges_csv, encoding="utf-8-sig")
    need = {"src_content_id","dst_content_id","weight"}
    if not need.issubset(df.columns):
        raise ValueError(f"{edges_csv}에 {need} 컬럼이 필요합니다.")

    adj = defaultdict(list)   # node -> [(nbr, w), ...]
    nodes = set()
    for a, b, w in df[["src_content_id","dst_content_id","weight"]].itertuples(index=False):
        if a == b: 
            continue
        w = float(w) if pd.notna(w) else 1.0
        adj[a].append((b, w))
        adj[b].append((a, w))
        nodes.add(a); nodes.add(b)

    # 누적확률(룰렛휠) 사전
    cumsums = {}
    for u, lst in adj.items():
        if not lst:
            continue
        nbrs, ws = zip(*lst)
        ws = np.maximum(np.array(ws, dtype=float), 1e-12)
        csum = np.cumsum(ws); cumsums[u] = (np.array(nbrs, dtype=int), csum / csum[-1])
    return adj, cumsums, sorted(nodes)

def weighted_choice(nbrs, cprobs):
    r = random.random()
    lo, hi = 0, len(cprobs)-1
    while lo < hi:
        mid = (lo + hi) // 2
        if cprobs[mid] < r: lo = mid + 1
        else: hi = mid
    return int(nbrs[lo])

# ---------- Node2Vec 가중 랜덤워크 ----------
def node2vec_walks(adj, cumsums, nodes, walk_length, num_walks, p, q):
    walks = []
    base_nodes = list(nodes)
    for _ in range(num_walks):
        random.shuffle(base_nodes)
        for start in base_nodes:
            if start not in cumsums:
                continue
            walk = [start]
            if walk_length == 1:
                walks.append(walk); continue

            # 첫 스텝: 가중치 비례 선택
            nbrs, cprobs = cumsums[start]
            if len(nbrs) == 0:
                walks.append(walk); continue
            curr = weighted_choice(nbrs, cprobs)
            walk.append(curr); prev = start

            for _ in range(2, walk_length):
                cand = adj.get(curr, [])
                if not cand: break
                cand_nodes = [n for n,_ in cand]
                cand_w = []
                prev_nbrs = {n for n,_ in adj.get(prev, [])}
                for nxt, w in cand:
                    # p/q 바이어스(근사)
                    if nxt == prev: bias = 1.0 / p
                    elif nxt in prev_nbrs: bias = 1.0
                    else: bias = 1.0 / q
                    cand_w.append(max(w,1e-12) * bias)
                cw = np.array(cand_w, dtype=float)
                cs = np.cumsum(cw); cs /= cs[-1]
                idx = np.searchsorted(cs, random.random())
                nxt = cand_nodes[min(idx, len(cand_nodes)-1)]
                walk.append(nxt); prev, curr = curr, nxt
            walks.append(walk)
    return walks

# ---------- Skip-gram(네거티브 샘플링) ----------
def generate_pairs(walks, window):
    pairs = []
    for walk in walks:
        L = len(walk)
        for i in range(L):
            c = walk[i]
            l = max(0, i-window); r = min(L, i+window+1)
            for j in range(l, r):
                if j == i: continue
                pairs.append((c, walk[j]))
    return pairs

class SkipGramNS(torch.nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_emb  = torch.nn.Embedding(vocab_size, dim)
        self.out_emb = torch.nn.Embedding(vocab_size, dim)
        torch.nn.init.uniform_(self.in_emb.weight,  -0.5/dim, 0.5/dim)
        torch.nn.init.uniform_(self.out_emb.weight, -0.5/dim, 0.5/dim)

    def forward(self, center, pos, neg):
        # center:[B], pos:[B], neg:[B,K]
        v  = self.in_emb(center)           # [B, D]
        u  = self.out_emb(pos)             # [B, D]
        uv = (v * u).sum(dim=1)            # [B]
        pos_loss = torch.nn.functional.logsigmoid(uv).mean()

        neg_u = self.out_emb(neg)          # [B, K, D]
        neg_uv = torch.bmm(neg_u, v.unsqueeze(2)).squeeze(2)  # [B, K]
        neg_loss = torch.nn.functional.logsigmoid(-neg_uv).mean()
        return -(pos_loss + neg_loss)      # minimize

def train_skipgram_ns(pairs, id2idx, epochs=3, batch_size=8192, dim=64, neg_k=5, lr=0.025):
    # 노드 인덱싱
    vocab = sorted(id2idx.keys())
    vocab_size = len(vocab)

    # 빈도 기반 네거티브 샘플링 분포(0.75 승)
    counts = defaultdict(int)
    for a,b in pairs:
        counts[a] += 1; counts[b] += 1
    idx_counts = np.zeros(vocab_size, dtype=np.float64)
    for nid, cnt in counts.items():
        idx_counts[id2idx[nid]] = cnt
    prob = idx_counts ** 0.75
    prob = prob / prob.sum()
    alias_table = np.cumsum(prob)

    def sample_neg(B, K):
        # 누적분포 기반 벡터화 샘플링
        r = np.random.rand(B, K)
        idx = np.searchsorted(alias_table, r, side="right")
        return torch.from_numpy(idx.astype(np.int64))

    # 학습 데이터 텐서화(인덱스 변환)
    centers = torch.tensor([id2idx[a] for a,_ in pairs], dtype=torch.long)
    contexts= torch.tensor([id2idx[b] for _,b in pairs], dtype=torch.long)

    ds_size = len(pairs)
    model = SkipGramNS(vocab_size, dim).to(DEVICE)
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        perm = torch.randperm(ds_size)
        centers = centers[perm]; contexts = contexts[perm]
        total_loss = 0.0; steps = 0

        for i in range(0, ds_size, batch_size):
            c_batch = centers[i:i+batch_size].to(DEVICE)
            p_batch = contexts[i:i+batch_size].to(DEVICE)
            B = c_batch.size(0)
            n_batch = sample_neg(B, neg_k).to(DEVICE)

            loss = model(c_batch, p_batch, n_batch)
            optim.zero_grad(); loss.backward(); optim.step()

            total_loss += loss.item(); steps += 1

        avg = total_loss / max(1, steps)
        print(f"[Epoch {epoch}/{epochs}] loss={avg:.4f}")

    with torch.no_grad():
        emb = model.in_emb.weight.detach().cpu().numpy()
    return emb  # shape [vocab_size, dim]

def main():
    print("📥 그래프 로딩...")
    adj, cumsums, nodes = load_graph(EDGES_PATH)
    print(f"nodes={len(nodes):,}, edges(undirected)≈{sum(len(v) for v in adj.values())//2:,}")

    print("🚶 랜덤워크 생성...")
    walks = node2vec_walks(
        adj, cumsums, nodes,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        p=P_RETURN, q=Q_INOUT
    )
    avg_len = np.mean([len(w) for w in walks]) if walks else 0
    print(f"walks={len(walks):,}, avg_len≈{avg_len:.1f}")

    print("🧩 학습 쌍 생성(Skip-gram window)...")
    pairs = generate_pairs(walks, WINDOW)
    print(f"pairs={len(pairs):,}")

    # 노드 id → 연속 index 매핑
    id2idx = {nid:i for i, nid in enumerate(sorted(nodes))}
    print("🧠 Skip-gram(NS) 학습 시작...")
    emb = train_skipgram_ns(
        pairs, id2idx,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        dim=DIM, neg_k=NEGATIVE_K, lr=LR
    )

    # CSV 저장
    idx2id = {i:nid for nid,i in id2idx.items()}
    rows = [[idx2id[i]] + list(map(float, emb[i])) for i in range(len(idx2id))]
    cols = ["content_id"] + [f"emb_{i}" for i in range(DIM)]
    out = pd.DataFrame(rows, columns=cols).sort_values("content_id")
    out.to_csv(OUT_EMB_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 임베딩 저장: {OUT_EMB_CSV} (nodes={len(out)}, dim={DIM})")

if __name__ == "__main__":
    main()
