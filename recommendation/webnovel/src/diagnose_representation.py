# src/diagnose_representation.py
"""표현(representation)이 변별력을 갖는지 측정한다. **사람 판정에 들어가기 전의 게이트다.**

이 도메인의 리스크는 **장르가 얇다**는 것이다 — 작품당 1개 coarse 라벨뿐이다
(줄거리는 중앙값 110자로 충분한 편이다). 임베딩이 "전부 비슷비슷한" 공간을 만들면
랭킹·후처리를 아무리 손봐도 소용이 없다. 그래서 표현을 바꿀 때마다 여기를 먼저 통과시킨다.

**절대 유사도로 판정하지 않는다.** 정규화된 임베딩의 코사인 절대값은 모델·언어마다
기준점이 달라서 그 자체로는 의미가 없다. 실제로 봐야 하는 것은 두 가지다:

  ① 변별력 — 같은 장르와 다른 장르가 벌어지는가 (gap), 이웃이 실제로 같은 장르인가
     (purity, 단 무작위 기준선 Σp² 대비 lift 로 봐야 한다)
  ② 스프레드 — 순위를 매길 여지가 있는가 (std)

`--control` 은 웹소설과 무관한 한국어 텍스트를 같은 형식으로 임베딩해 **바닥값**을
실측한다. 코퍼스 평균이 높아도 바닥값과 충분히 벌어져 있으면 그것은 "전부 비슷함"이
아니라 "같은 도메인이라 응집된 것"이다. 자세한 근거는 아래 게이트 상수 주석 참고.

    python -m src.diagnose_representation [--artifacts DIR] [--sample N] [--control]
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.config import artifact_dir, load_config

# 합격선.
#
# **절대 유사도 임계는 쓰지 않는다.** 처음에는 Steam 수치를 그대로 가져와
# `무작위 쌍 평균 < 0.45` 를 걸었는데, 실측해 보니 잘못된 게이트였다:
#
#   무관한 한국어 텍스트끼리        0.223   ← 모델의 바닥값 (Steam 실측 0.229 와 동일)
#   무관한 텍스트 ↔ 웹소설 코퍼스   0.276
#   웹소설 ↔ 웹소설                0.471
#
# 즉 0.471 은 모델·언어 오프셋이 아니라 **같은 도메인이라서 생기는 실제 응집성**이다
# (바닥값 대비 +0.195). Steam 의 0.519 가 문제였던 이유는 값이 높아서가 아니라
# 정보량이 0인 플랫폼 상용구가 원인이었기 때문이고, 여기엔 그런 상용구가 없다.
# 절대값이 아니라 **변별력(장르 구분)과 스프레드(순위를 매길 여지)** 로 판정한다.
GATE_MIN_GENRE_GAP = 0.05     # 같은 장르 - 다른 장르 유사도 차
GATE_MIN_GENRE_PURITY = 0.5   # Top-10 이웃 중 같은 장르 비율
GATE_MIN_GENRE_LIFT = 0.15    # 그 순도가 무작위 기준선(Σp²)보다 이만큼은 높아야 한다
GATE_MIN_SPREAD = 0.03        # 쌍 유사도 표준편차. 0 에 가까우면 순위를 매길 수 없다
GATE_MIN_CONTROL_SEP = 0.10   # 코퍼스 평균 − 무관 텍스트 바닥값 (--control 일 때만)

# 바닥값 측정용 대조군. 웹소설과 무관한 한국어 텍스트를 semantic_text 와 같은 형식으로 쓴다
# (형식이 다르면 형식 차이가 유사도 차이로 잡혀서 바닥값이 과소평가된다).
CONTROL_TEXTS = [
    "제목: 국세청 연말정산 간소화 서비스 개통\n장르: 뉴스\n줄거리: 국세청은 15일부터 연말정산 "
    "간소화 서비스를 개통한다고 밝혔다. 근로자는 홈택스에서 소득·세액공제 자료를 조회할 수 있다.",
    "제목: 김치찌개 끓이는 법\n장르: 요리\n줄거리: 돼지고기를 볶다가 잘 익은 김치를 넣고 함께 "
    "볶는다. 물을 붓고 두부와 대파를 넣어 20분간 끓인다. 고춧가루로 간을 맞춘다.",
    "제목: 민법 제750조 불법행위의 내용\n장르: 법률\n줄거리: 고의 또는 과실로 인한 위법행위로 "
    "타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다.",
    "제목: PostgreSQL 인덱스 튜닝 가이드\n장르: 기술문서\n줄거리: B-tree 인덱스는 등호와 범위 "
    "조건에 적합하다. 카디널리티가 낮은 컬럼에는 부분 인덱스를 고려한다.",
    "제목: 2026년 최저임금 심의 결과\n장르: 뉴스\n줄거리: 최저임금위원회는 내년도 최저임금을 "
    "시간당 원으로 의결했다. 경영계와 노동계는 각각 유감을 표명했다.",
    "제목: 등산 배낭 꾸리는 요령\n장르: 생활\n줄거리: 무거운 짐은 등판 가까이 위쪽에 배치한다. "
    "침낭은 맨 아래에 넣고 방수 커버를 준비한다.",
]


def primary_genre(genres) -> str:
    """작품당 장르가 사실상 1개라 첫 번째를 대표 장르로 쓴다."""
    if isinstance(genres, (list, np.ndarray)) and len(genres):
        return str(genres[0])
    return ""


def random_pair_similarity(emb: np.ndarray, rng: np.random.Generator, n: int = 20000) -> np.ndarray:
    a = rng.integers(0, len(emb), n)
    b = rng.integers(0, len(emb), n)
    mask = a != b
    return np.einsum("ij,ij->i", emb[a[mask]], emb[b[mask]])


def genre_gap(emb: np.ndarray, genres: pd.Series, rng: np.random.Generator, n: int = 20000):
    """같은 장르 쌍과 다른 장르 쌍의 평균 유사도 차이.

    이 차이가 0 에 가까우면 임베딩이 장르조차 구분하지 못하는 것이고, 그러면
    '취향'은 당연히 못 잡는다. 랭킹으로 고칠 수 있는 문제가 아니다.
    """
    a = rng.integers(0, len(emb), n)
    b = rng.integers(0, len(emb), n)
    ok = (a != b) & (genres.values[a] != "") & (genres.values[b] != "")
    a, b = a[ok], b[ok]
    sims = np.einsum("ij,ij->i", emb[a], emb[b])
    same = genres.values[a] == genres.values[b]
    if same.sum() == 0 or (~same).sum() == 0:
        return None, None, None
    return float(sims[same].mean()), float(sims[~same].mean()), int(same.sum())


def genre_base_rate(genres: pd.Series) -> float:
    """무작위로 이웃을 골랐을 때 장르가 같을 확률 = Σ p(g)².

    **순도(purity)는 이 값과 비교해야 의미가 있다.** 코퍼스의 76%가 판타지면
    아무 의미 없는 임베딩도 순도 0.76 을 낸다. 실제로 봐야 하는 것은 순도 자체가 아니라
    이 기준선 대비 얼마나 올렸는가(lift)다.
    """
    labeled = genres[genres != ""]
    if labeled.empty:
        return 0.0
    p = labeled.value_counts(normalize=True).values
    return float((p ** 2).sum())


def neighbor_genre_purity(emb: np.ndarray, genres: pd.Series, rng: np.random.Generator,
                          sample: int = 500, k: int = 10):
    """무작위 표본의 Top-k 이웃 중 같은 장르 비율.

    실제 추천이 하는 일에 가장 가까운 측정이다. 유사도 평균이 좋아 보여도
    이게 낮으면 추천 결과는 장르조차 안 맞는 목록이 된다.
    """
    idx = rng.choice(len(emb), size=min(sample, len(emb)), replace=False)
    idx = np.array([i for i in idx if genres.values[i] != ""])
    if len(idx) == 0:
        return None
    purity = []
    for start in range(0, len(idx), 128):  # 메모리 때문에 청크로 곱한다
        chunk = idx[start:start + 128]
        sims = emb[chunk] @ emb.T
        sims[np.arange(len(chunk)), chunk] = -np.inf  # 자기 자신 제외
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for r, i in enumerate(chunk):
            neigh = genres.values[top[r]]
            purity.append(float((neigh == genres.values[i]).mean()))
    return float(np.mean(purity))


def measure_control_floor(emb: np.ndarray, cfg: dict) -> dict:
    """무관한 텍스트의 유사도 바닥값을 실측한다. 모델을 새로 올리므로 느리다(--control).

    이게 있어야 코퍼스 평균이 "전부 비슷함"인지 "같은 도메인이라 응집된 것"인지 구분된다.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["embedding"]["model_name"], device="cpu")
    model.max_seq_length = cfg["runtime"]["max_seq_length"]
    ce = model.encode(
        CONTROL_TEXTS,
        batch_size=cfg["runtime"]["cpu_batch_size"],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    iu = np.triu_indices(len(ce), 1)
    return {
        "control_to_control": float((ce @ ce.T)[iu].mean()),
        "control_to_corpus": float((ce @ emb.T).mean()),
        "n_control_texts": len(CONTROL_TEXTS),
    }


def diagnose(artifacts=None, sample: int = 20000, seed: int = 42, control: bool = False) -> dict:
    d = artifact_dir(artifacts)
    emb = np.asarray(np.load(d / "corpus_embeddings.npy", mmap_mode="r"), dtype=np.float32)
    index = pd.read_parquet(d / "corpus_index.parquet")
    dataset = pd.read_parquet(d / "dataset.parquet")

    # corpus_index 순서가 임베딩 행 순서다. dataset 은 그 부분집합일 수 있다(단계적 임베딩).
    meta = index.merge(dataset, on="item_id", how="left", suffixes=("", "_ds"))
    genres = meta["genres"].map(primary_genre)

    rng = np.random.default_rng(seed)
    pair_sims = random_pair_similarity(emb, rng, sample)
    same_g, diff_g, n_same = genre_gap(emb, genres, rng, sample)
    purity = neighbor_genre_purity(emb, genres, rng)
    base = genre_base_rate(genres)
    lift = (purity - base) if purity is not None else None

    syn_len = dataset["synopsis"].str.len() if "synopsis" in dataset else pd.Series(dtype=int)
    txt_len = meta["semantic_text"].str.len() if "semantic_text" in meta else pd.Series(dtype=int)

    report = {
        "artifacts": str(d),
        "corpus_size": int(len(emb)),
        "embedding_dim": int(emb.shape[1]),
        "random_pair_similarity": {
            "mean": float(pair_sims.mean()),
            "p50": float(np.percentile(pair_sims, 50)),
            "p95": float(np.percentile(pair_sims, 95)),
            "std": float(pair_sims.std()),
        },
        "genre_similarity": {
            "same_genre_mean": same_g,
            "diff_genre_mean": diff_g,
            "gap": (same_g - diff_g) if same_g is not None else None,
            "same_genre_pairs": n_same,
        },
        "neighbor_genre_purity@10": purity,
        # 순도는 기준선 대비로 읽어야 한다. 코퍼스가 한 장르로 쏠려 있으면 순도가 저절로 높다.
        "genre_base_rate": base,
        "genre_purity_lift": lift,
        "genre_label_count": int((genres != "").sum()),
        "distinct_genres": int(genres[genres != ""].nunique()),
        "semantic_text_chars": {
            "median": float(txt_len.median()) if len(txt_len) else None,
            "p25": float(txt_len.quantile(0.25)) if len(txt_len) else None,
        },
        "synopsis_chars": {
            "median": float(syn_len.median()) if len(syn_len) else None,
            "under_30_ratio": float((syn_len < 30).mean()) if len(syn_len) else None,
        },
    }

    gates = {
        "genre_gap > %.2f" % GATE_MIN_GENRE_GAP:
            (report["genre_similarity"]["gap"] or 0) > GATE_MIN_GENRE_GAP,
        "genre_purity@10 > %.2f" % GATE_MIN_GENRE_PURITY:
            (purity or 0) > GATE_MIN_GENRE_PURITY,
        # 쏠린 코퍼스에서 순도 단독은 무의미하다. 기준선을 실제로 넘겼는지 따로 본다.
        "genre_purity_lift > %.2f" % GATE_MIN_GENRE_LIFT:
            (lift or 0) > GATE_MIN_GENRE_LIFT,
        # 스프레드가 없으면 유사도로 순위를 매길 수 없다
        "pair_similarity_std > %.2f" % GATE_MIN_SPREAD:
            report["random_pair_similarity"]["std"] > GATE_MIN_SPREAD,
    }

    if control:
        cfg = load_config()
        ctl = measure_control_floor(emb, cfg)
        sep = report["random_pair_similarity"]["mean"] - ctl["control_to_control"]
        ctl["separation_from_floor"] = sep
        report["control"] = ctl
        gates["corpus_mean − control_floor > %.2f" % GATE_MIN_CONTROL_SEP] = sep > GATE_MIN_CONTROL_SEP

    report["gates"] = gates
    report["passed"] = all(gates.values())
    return report


def main():
    ap = argparse.ArgumentParser(description="표현 변별력 진단 (판정 전 게이트)")
    ap.add_argument("--artifacts", default=None)
    ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--control", action="store_true",
                    help="무관 텍스트 바닥값을 실측한다 (모델을 새로 올려서 느리다)")
    args = ap.parse_args()

    rep = diagnose(args.artifacts, args.sample, args.seed, control=args.control)
    out = artifact_dir(args.artifacts) / "representation_diagnosis.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    r = rep["random_pair_similarity"]
    g = rep["genre_similarity"]
    print(f"코퍼스 {rep['corpus_size']:,}개 · {rep['embedding_dim']}차원  ({rep['artifacts']})")
    print(f"\n무작위 쌍 유사도   평균 {r['mean']:.3f}  중앙 {r['p50']:.3f}  p95 {r['p95']:.3f}  표준편차 {r['std']:.3f}")
    if "control" in rep:
        c = rep["control"]
        print(f"  대조군(무관 텍스트)  끼리 {c['control_to_control']:.3f} · 코퍼스와 {c['control_to_corpus']:.3f}"
              f"  → 바닥값 대비 +{c['separation_from_floor']:.3f}")
    else:
        print(f"  (실측 바닥값: 무관 한국어 텍스트끼리 0.223 — --control 로 재측정)")
    if g["gap"] is not None:
        print(f"장르 유사도        같은 장르 {g['same_genre_mean']:.3f}  vs  다른 장르 {g['diff_genre_mean']:.3f}"
              f"   차이 {g['gap']:+.3f}")
    print(f"Top-10 이웃 장르 일치율  {rep['neighbor_genre_purity@10']:.3f}"
          f"   (무작위 기준선 {rep['genre_base_rate']:.3f} → lift {rep['genre_purity_lift']:+.3f})")
    print(f"  장르 라벨 {rep['genre_label_count']:,}건 · 서로 다른 장르 {rep['distinct_genres']}종")
    t, s = rep["semantic_text_chars"], rep["synopsis_chars"]
    print(f"semantic_text 중앙값 {t['median']:.0f}자  |  시놉시스 중앙값 {s['median']:.0f}자 "
          f"(30자 미만 {s['under_30_ratio'] * 100:.0f}%)")

    print("\n--- 게이트 ---")
    for name, ok in rep["gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n=> {'통과 — 판정 단계로 진행' if rep['passed'] else '불합격 — 표현을 바꿔야 한다 (README §5 rep_v2/v3)'}")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
