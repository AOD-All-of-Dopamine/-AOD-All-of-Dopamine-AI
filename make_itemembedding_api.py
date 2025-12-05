"""
아이템 임베딩 생성 API 모듈 (Node2Vec + Skip-gram)
Graph edges → 아이템 임베딩 (64차원)
"""

import pandas as pd
import numpy as np
global torch

# torch는 무겁기 때문에 요청 시 지연 로드(lazy import)합니다.
torch = None
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ====== 설정 ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[ItemEmbedding] Using device: {DEVICE}")

# 하이퍼파라미터 (기본값)
DIM = 64
WALK_LENGTH = 40
NUM_WALKS = 10
P_RETURN = 1.0
Q_INOUT = 1.0
WINDOW = 5
NEGATIVE_K = 5
EPOCHS = 3
BATCH_SIZE = 8192
LR = 0.025


# ====== 그래프 로딩 ======
def load_graph(edges_df):
    """
    DataFrame edges를 인접 리스트로 변환
    edges_df: (src_content_id, dst_content_id, weight) 또는 최소 3개 컬럼
    
    반환: (adj, cumsums, sorted_nodes)
    - adj: {node: [(neighbor, weight), ...]}
    - cumsums: 누적합(가중치 기반 랜덤 선택용)
    - sorted_nodes: 정렬된 노드 리스트
    """
    # 컬럼명 결정 (유연한 처리)
    cols = edges_df.columns.tolist()
    src_col = next((c for c in ['src_content_id', 'from_content_id', 'source'] if c in cols), cols[0])
    dst_col = next((c for c in ['dst_content_id', 'to_content_id', 'target'] if c in cols), cols[1])
    weight_col = next((c for c in ['weight', 'score', 'similarity'] if c in cols), cols[2])
    
    adj = defaultdict(list)
    cumsums = {}
    
    for _, row in edges_df.iterrows():
        src, dst, w = row[src_col], row[dst_col], row[weight_col]
        # 그래프 가중치를 float로 변환
        w = float(w) if not pd.isna(w) else 1.0
        
        # 무방향 그래프로 처리
        adj[src].append((dst, w))
        adj[dst].append((src, w))
    
    # 가중치 누적합 계산 (별칭 샘플링용)
    for node in adj:
        neighbors, weights = zip(*adj[node])
        total = sum(weights)
        cumsum = np.cumsum([w/total for w in weights])
        cumsums[node] = (neighbors, cumsum)
    
    sorted_nodes = sorted(adj.keys())
    return adj, cumsums, sorted_nodes


# ====== Node2Vec 랜덤 워크 ======
def node2vec_walks(adj, cumsums, nodes, walk_length=40, num_walks=10, p=1.0, q=1.0):
    """
    Node2Vec 방식의 가중치 기반 랜덤 워크 생성
    p: return 파라미터 (작을수록 현재 노드로 돌아감)
    q: in-out 파라미터 (작을수록 탐색적)
    """
    walks = []
    
    # Transition 확률 미리 계산 (p, q 반영)
    trans = {}
    for src in adj:
        trans[src] = {}
        for dst, w in adj[src]:
            trans[src][dst] = w
        
        # p-q 보정 (2-hop 정보 활용)
        # 간단 버전: p, q 사용하지 않음 (속도 최적화)
        # 정확한 Node2Vec는 이전 노드 정보 필요
    
    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            curr = node
            
            for _ in range(walk_length - 1):
                if curr not in adj or not adj[curr]:
                    break
                
                neighbors, cumsum = cumsums[curr]
                # 가중치 기반 샘플링
                r = np.random.rand()
                idx = np.searchsorted(cumsum, r)
                idx = min(idx, len(neighbors) - 1)
                next_node = neighbors[idx]
                
                walk.append(next_node)
                curr = next_node
            
            walks.append(walk)
    
    return walks


# ====== Skip-gram 학습 쌍 생성 ======
def generate_pairs(walks, window=5):
    """
    랜덤 워크에서 Skip-gram 학습 쌍 추출
    (center_word, context_word)
    """
    pairs = []
    for walk in walks:
        for i, word in enumerate(walk):
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    pairs.append((word, walk[j]))
    return pairs


# ====== Skip-gram 모델 정의 ======
class SkipGramNS:
    """
    지연 로드된 torch를 사용하는 SkipGramNS 래퍼.
    실제 torch가 로드되지 않은 경우 예외를 발생시킵니다.
    """
    def __init__(self, vocab_size, dim):
        global torch
        if torch is None:
            try:
                import torch as _torch
                torch = _torch
            except Exception:
                raise RuntimeError('PyTorch가 설치되어 있지 않거나 로드할 수 없습니다. 아이템 임베딩 API를 사용하려면 torch를 설치하세요.')
        class _Model(torch.nn.Module):
            def __init__(self, vocab_size, dim):
                super().__init__()
                self.in_emb = torch.nn.Embedding(vocab_size, dim)
                self.out_emb = torch.nn.Embedding(vocab_size, dim)
                torch.nn.init.xavier_uniform_(self.in_emb.weight)
                torch.nn.init.xavier_uniform_(self.out_emb.weight)
            def forward(self, center, pos, neg):
                c_emb = self.in_emb(center)
                p_emb = self.out_emb(pos)
                n_emb = self.out_emb(neg)
                pos_score = (c_emb * p_emb).sum(dim=1)
                pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)
                neg_score = (c_emb.unsqueeze(2) * n_emb).sum(dim=1)
                neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8)
                loss = pos_loss.mean() + neg_loss.mean()
                return loss
        self._model = _Model(vocab_size, dim)

    def to(self, device):
        self._model.to(device)
        return self

    def parameters(self):
        return self._model.parameters()

    @property
    def in_emb(self):
        return self._model.in_emb

    def __call__(self, center, pos, neg):
        return self._model(center, pos, neg)
    """
    Skip-gram with Negative Sampling
    입력: center word index
    출력: embedding
    """
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_emb = torch.nn.Embedding(vocab_size, dim)
        self.out_emb = torch.nn.Embedding(vocab_size, dim)
        # Xavier 초기화
        torch.nn.init.xavier_uniform_(self.in_emb.weight)
        torch.nn.init.xavier_uniform_(self.out_emb.weight)
    
    def forward(self, center, pos, neg):
        """
        center: [B] 중심 단어 인덱스
        pos: [B] 긍정 컨텍스트 인덱스
        neg: [B, K] 부정 샘플 인덱스
        """
        c_emb = self.in_emb(center)  # [B, D]
        p_emb = self.out_emb(pos)    # [B, D]
        n_emb = self.out_emb(neg)    # [B, K, D]
        
        # 긍정 점수 (최대화)
        pos_score = (c_emb * p_emb).sum(dim=1)  # [B]
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)
        
        # 부정 점수 (최소화)
        neg_score = (c_emb.unsqueeze(2) * n_emb).sum(dim=1)  # [B, K]
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8)
        
        loss = pos_loss.mean() + neg_loss.mean()
        return loss


# ====== Skip-gram 학습 ======
def train_skipgram_ns(pairs, id2idx, epochs=3, batch_size=8192, 
                      dim=64, neg_k=5, lr=0.025):
    """
    Skip-gram with Negative Sampling 학습
    pairs: [(center, context), ...] 리스트
    id2idx: {노드_id: 연속_index} 딕셔너리
    
    반환: 임베딩 배열 [vocab_size, dim]
    """
    vocab_size = len(id2idx)
    
    # 빈도 기반 네거티브 샘플링 분포 (0.75 승)
    counts = defaultdict(int)
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1
    
    idx_counts = np.zeros(vocab_size, dtype=np.float64)
    for nid, cnt in counts.items():
        idx_counts[id2idx[nid]] = cnt
    
    prob = idx_counts ** 0.75
    prob = prob / prob.sum()
    alias_table = np.cumsum(prob)
    
    def sample_neg(B, K):
        r = np.random.rand(B, K)
        idx = np.searchsorted(alias_table, r, side="right")
        return torch.from_numpy(idx.astype(np.int64))
    
    # 학습 데이터 텐서화
    centers = torch.tensor([id2idx[a] for a, _ in pairs], dtype=torch.long)
    contexts = torch.tensor([id2idx[b] for _, b in pairs], dtype=torch.long)
    
    ds_size = len(pairs)
    # 지연 로드: 함수 내부에서 torch를 임포트
    
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
        except Exception:
            raise RuntimeError('PyTorch가 설치되어 있지 않습니다. 아이템 임베딩 API를 사용하려면 torch를 설치하세요.')

    model = SkipGramNS(vocab_size, dim).to(DEVICE)
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(ds_size)
        centers = centers[perm]
        contexts = contexts[perm]
        total_loss = 0.0
        steps = 0
        
        for i in range(0, ds_size, batch_size):
            c_batch = centers[i:i+batch_size].to(DEVICE)
            p_batch = contexts[i:i+batch_size].to(DEVICE)
            B = c_batch.size(0)
            n_batch = sample_neg(B, neg_k).to(DEVICE)
            
            loss = model(c_batch, p_batch, n_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            steps += 1
        
        avg = total_loss / max(1, steps)
        print(f"[Epoch {epoch}/{epochs}] loss={avg:.4f}")
    
    # 학습된 임베딩 추출
    with torch.no_grad():
        emb = model.in_emb.weight.detach().cpu().numpy()
    
    return emb


# ====== 메인 함수 ======
def build_item_embeddings(edges_df, dim=DIM, walk_length=WALK_LENGTH, 
                          num_walks=NUM_WALKS, p=P_RETURN, q=Q_INOUT,
                          window=WINDOW, neg_k=NEGATIVE_K, epochs=EPOCHS,
                          batch_size=BATCH_SIZE, lr=LR):
    """
    아이템 그래프에서 임베딩 생성 (Node2Vec + Skip-gram)
    
    입력:
    - edges_df: DataFrame (src_content_id, dst_content_id, weight)
    
    출력:
    - DataFrame (content_id, emb_0, emb_1, ..., emb_{dim-1})
    """
    
    print("[ItemEmbedding] 그래프 로딩...")
    adj, cumsums, nodes = load_graph(edges_df)
    edge_count = sum(len(v) for v in adj.values()) // 2
    print(f"[ItemEmbedding] nodes={len(nodes):,}, edges≈{edge_count:,}")
    
    print("[ItemEmbedding] 랜덤워크 생성...")
    walks = node2vec_walks(adj, cumsums, nodes, 
                           walk_length=walk_length, 
                           num_walks=num_walks, 
                           p=p, q=q)
    avg_len = np.mean([len(w) for w in walks]) if walks else 0
    print(f"[ItemEmbedding] walks={len(walks):,}, avg_len≈{avg_len:.1f}")
    
    print("[ItemEmbedding] Skip-gram 학습 쌍 생성...")
    pairs = generate_pairs(walks, window=window)
    print(f"[ItemEmbedding] pairs={len(pairs):,}")
    
    # 노드 id → 연속 index 매핑
    id2idx = {nid: i for i, nid in enumerate(sorted(nodes))}
    
    print("[ItemEmbedding] Skip-gram(NS) 학습 시작...")
    emb = train_skipgram_ns(pairs, id2idx, epochs=epochs, 
                            batch_size=batch_size, dim=dim, 
                            neg_k=neg_k, lr=lr)
    
    # L2 정규화
    print("[ItemEmbedding] L2 정규화...")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = emb / norms
    
    # DataFrame으로 변환
    idx2id = {i: nid for nid, i in id2idx.items()}
    rows = [[idx2id[i]] + list(map(float, emb[i])) for i in range(len(idx2id))]
    cols = ["content_id"] + [f"emb_{i}" for i in range(dim)]
    
    df_out = pd.DataFrame(rows, columns=cols).sort_values("content_id").reset_index(drop=True)
    print(f"[ItemEmbedding] ✅ 임베딩 생성 완료: {len(df_out)} nodes, {dim}차원")
    
    return df_out


if __name__ == "__main__":
    # 테스트용 (API 서버에서는 호출되지 않음)
    pass
