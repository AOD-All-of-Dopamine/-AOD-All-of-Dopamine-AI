"""웹소설 평가용 프로필 52개.

Steam·TMDB 와 같은 레시피다: 직접 선정 28개 + 규칙 생성 24개.
기준도 같다 — 52프로필 × k=50 · 적합률 0.85.

**시드는 `item_id` 로 저장한다.** 이름이 아니다(D-20). 웹소설은 중복 이름이
77개 / 154행(2.2%)으로 TMDB(8.8%)보다 적지만, 이름 키는 같은 제목의 다른
작품을 조용히 섞는다. `item_id` 는 네이버 시리즈의 productNo 로 유일하다.

축 설계 의도:
  coh_*      응집형 3시드 — 같은 결의 작품 셋
  mix2_*     2축 혼합 — 서로 다른 장르/결
  single/two/five/seven/ten/twenty  시드 개수 축
  rule_*     규칙 생성 — 장르 × 관심 수 대역. 롱테일·저관심·틈새를 덮는다
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import artifact_dir  # noqa: E402

# ── 직접 선정 28개 ───────────────────────────────────────────────────────────
HAND: dict[str, list[str]] = {
    # 응집형 (3시드)
    "coh_hunter":        ["게임 속 바바리안으로 살아남기", "내가 키운 S급들", "튜토리얼이 너무 어렵다"],
    "coh_regression":    ["회귀의 전설", "8클래스 마법사의 회귀", "권왕환생"],
    "coh_murim":         ["절대 검감(絶對 劍感)", "태존비록(怠尊秘錄)", "검신재림(劍神再臨)"],
    "coh_murim_modern":  ["천재 무림 트레이너", "무한 레벨업 in 무림", "현질무신"],
    "coh_romfan_revenge":["전남편의 미친개를 길들였다", "구원, 그 잔혹함에 대하여", "내 남편의 정부에게"],
    "coh_romfan_divorce":["그 결혼, 이번 생엔 제가 할게요", "지루하신 공작님과 강철의 이혼녀", "함부로 길들이지 마시오!"],
    "coh_romance_adult": ["19년지기의 하룻밤", "아찔함과 야릇함 사이", "스위트룸에서"],
    "coh_bl_marriage":   ["결혼과 부부생활", "목표는 안전이혼인데요", "메리지 어게인"],
    "coh_modern_success":["영업 천재가 되었다", "재벌가 복덩이", "재벌집 기둥서방이 되었다."],
    "coh_talent":        ["두 번 사는 프로듀서", "음악천재를 위하여", "다 잘하는 히어로"],
    # 2축 혼합
    "mix2_fantasy_murim":["템빨", "절대 검감(絶對 劍感)"],
    "mix2_romfan_bl":    ["바스티안", "결혼과 부부생활"],
    "mix2_apocalypse":   ["멸망한 세계의 취사병", "멸망 이후의 세계"],
    "mix2_mystery":      ["복수 법률사무소", "오뉘탑: 퇴마사건일지"],
    "mix2_lightnovel":   ["이 멋진 세계에 축복을! 1~15권", "마왕학원의 부적합자 ~사상 최강의 마왕인 시조, 전생해서 자손들의 학교에 다니다~"],
    "mix2_villainess":   ["악녀의 시집살이는 즐겁다", "여성향 게임의 파멸 플래그밖에 없는 악역 영애로 환생해버렸다…"],
    # 시드 개수 축
    "single_barbarian":  ["게임 속 바바리안으로 살아남기"],
    "single_sword":      ["절대 검감(絶對 劍感)"],
    "single_bastian":    ["바스티안"],
    "two_gate":          ["내가 키운 S급들", "플레이어가 과거를 숨김"],
    "two_murim":         ["권왕환생", "약혼녀가 소천마였다"],
    "two_romance":       ["나쁜 착각", "결혼의 목적"],
    "five_toptier":      ["게임 속 바바리안으로 살아남기", "내가 키운 S급들", "절대 검감(絶對 劍感)",
                          "신화급 귀속 아이템을 손에 넣었다", "바스티안"],
    "five_romfan":       ["바스티안", "무기의 여왕", "전남편의 미친개를 길들였다",
                          "최선을 다해 후회하세요", "발칙한 요녀를 원하신다면"],
    "five_hunter":       ["튜토리얼이 너무 어렵다", "지구식 구원자 전형", "규격 외 혈통 천재",
                          "나 혼자 특성빨로 무한 성장", "성좌들이 내 제자"],
    "seven_fantasy":     ["템빨", "더 해머", "홍등가의 소드마스터", "망나니 1왕자가 되었다",
                          "마왕으로 살아남는 법", "8클래스 마법사의 회귀", "규격 외 혈통 천재"],
    "ten_mixed":         ["회귀의 전설", "템빨", "바스티안", "절대 검감(絶對 劍感)", "나쁜 착각",
                          "결혼과 부부생활", "복수 법률사무소", "그녀는 마왕?!",
                          "멸망 이후의 세계", "영업 천재가 되었다"],
    "twenty_library":    ["신화급 귀속 아이템을 손에 넣었다", "회귀의 전설", "히든 특성 13개 들고 시작한다",
                          "멸망한 세계의 취사병", "두 번 사는 프로듀서", "게임 속 바바리안으로 살아남기",
                          "내가 키운 S급들", "템빨", "성좌들이 내 제자", "더 해머",
                          "바스티안", "무기의 여왕", "악녀의 시집살이는 즐겁다",
                          "절대 검감(絶對 劍感)", "수라왕", "선역",
                          "19년지기의 하룻밤", "결혼의 목적", "결혼과 부부생활", "복수 법률사무소"],
}

# ── 규칙 생성 24개: (이름, 장르, 관심 수 하한, 상한, 시드 수) ────────────────
# 관심 수 대역이 축이다. `none`(=0) 은 1,759건이고 크롤 누락이 아니라 진짜 무명작이다
# (화수 중앙 6 · 댓글 중앙 0 · 평점 미보고 61%). 저관심 취향이 실제로 서빙되는지 본다.
RULE: list[tuple[str, str, int, int, int]] = [
    ("rule_hp_high",     "현판",     1_000_000, 10**12, 3),
    ("rule_hp_mid",      "현판",        10_000,  999_999, 3),
    ("rule_hp_low",      "현판",             1,    9_999, 3),
    ("rule_hp_none",     "현판",             0,        0, 3),
    ("rule_fant_high",   "판타지",   1_000_000, 10**12, 3),
    ("rule_fant_mid",    "판타지",      10_000,  999_999, 3),
    ("rule_fant_low",    "판타지",           1,    9_999, 3),
    ("rule_fant_none",   "판타지",           0,        0, 3),
    ("rule_rf_high",     "로판",     1_000_000, 10**12, 3),
    ("rule_rf_mid",      "로판",        10_000,  999_999, 3),
    ("rule_rf_low",      "로판",             1,    9_999, 3),
    ("rule_rf_none",     "로판",             0,        0, 3),
    ("rule_mu_high",     "무협",     1_000_000, 10**12, 3),
    ("rule_mu_mid",      "무협",        10_000,  999_999, 3),
    ("rule_mu_low",      "무협",             1,    9_999, 3),
    ("rule_mu_none",     "무협",             0,        0, 3),
    ("rule_rom_high",    "로맨스",   1_000_000, 10**12, 3),
    ("rule_rom_mid",     "로맨스",      10_000,  999_999, 3),
    ("rule_rom_low",     "로맨스",           0,    9_999, 3),
    ("rule_bl_high",     "BL",          50_000, 10**12, 3),
    ("rule_bl_mid",      "BL",           1_000,   49_999, 3),
    ("rule_bl_low",      "BL",               0,      999, 3),
    ("rule_ln",          "라이트노벨",        0, 10**12, 3),
    ("rule_myst",        "미스터리",          0, 10**12, 3),
]

SEED = 20260821


def _resolve(names: list[str], d: pd.DataFrame) -> list[int]:
    """제목 → item_id. 못 찾거나 여러 개면 큰 소리로 죽는다.

    조용히 건너뛰면 시드 수가 프로필마다 달라져 시드 개수 축이 오염된다.
    """
    out = []
    for nm in names:
        hit = d.index[d["name"] == nm].tolist()
        if not hit:
            raise SystemExit(f"코퍼스에 없는 제목: {nm!r}")
        if len(hit) > 1:
            # 이름 충돌 — 관심 수가 가장 큰 쪽을 쓰되 반드시 알린다
            print(f"  ⚠ 이름 충돌 {nm!r}: {len(hit)}건 → 관심 수 최대 선택", file=sys.stderr)
            hit = [d.loc[hit, "interest_count"].idxmax()]
        out.append(int(d.loc[hit[0], "item_id"]))
    return out


def _rule_seeds(d: pd.DataFrame, emb: np.ndarray, genre: str,
                lo: int, hi: int, n: int, rng: np.random.Generator) -> list[int]:
    """대역 안에서 앵커 하나를 뽑고, 그 이웃 n-1 개를 붙인다.

    무작위 n 개를 뽑으면 응집도가 바닥이라 "취향"이 되지 않는다. 앵커-이웃
    방식이면 실제 사람이 좋아할 법한 묶음이 된다.
    """
    pool = d[(d["g"] == genre) & (d["interest_count"] >= lo) & (d["interest_count"] <= hi)]
    if len(pool) < n:
        raise SystemExit(f"{genre} [{lo},{hi}] 풀이 {len(pool)}건 — 시드 {n}개를 못 만든다")
    anchor = pool.index[rng.integers(len(pool))]
    rows = pool["embedding_row"].to_numpy()
    sims = emb[rows] @ emb[int(d.loc[anchor, "embedding_row"])]
    order = np.argsort(-sims)
    chosen = [int(d.loc[pool.index[i], "item_id"]) for i in order[:n]]
    return chosen


def main() -> None:
    art = artifact_dir()
    d = pd.read_parquet(art / "dataset.parquet")
    ix = pd.read_parquet(art / "corpus_index.parquet")
    emb = np.asarray(np.load(art / "corpus_embeddings.npy"), dtype=np.float32)
    d = d.merge(ix[["item_id", "embedding_row"]], on="item_id", how="left")
    d["g"] = d["genres"].astype(str).str.strip("[]'\"")
    id_to_row = dict(zip(d["item_id"], d["embedding_row"]))

    rng = np.random.default_rng(SEED)
    rows = []
    for name, titles in HAND.items():
        rows.append({"profile_id": name, "seed_ids": _resolve(titles, d), "source": "hand"})
    for name, genre, lo, hi, n in RULE:
        rows.append({"profile_id": name,
                     "seed_ids": _rule_seeds(d, emb, genre, lo, hi, n, rng),
                     "source": "rule"})

    out = pd.DataFrame(rows)
    out["n_seeds"] = out["seed_ids"].apply(len)

    # 응집도 = 시드 임베딩 쌍별 코사인 평균. 시드 1개면 1.0.
    def cohesion(ids: list[int]) -> float:
        if len(ids) < 2:
            return 1.0
        v = emb[[id_to_row[i] for i in ids]]
        g = v @ v.T
        iu = np.triu_indices(len(ids), k=1)
        return float(g[iu].mean())

    out["seed_cohesion"] = out["seed_ids"].apply(cohesion).round(4)

    # dev/val 분할: 축이 양쪽에 고루 들어가도록 프로필 순서대로 번갈아 자른다.
    # 무작위 분할은 twenty 같은 1개짜리 축을 한쪽으로 몰아 val 을 못 쓰게 만든다.
    out = out.reset_index(drop=True)
    out["profile_order"] = out.index
    out["split"] = ["val" if (i % 5 in (0, 3)) and i < 50 else "dev" for i in out.index]
    # dev 32 / val 20 을 맞춘다
    while (out["split"] == "val").sum() > 20:
        i = out.index[out["split"] == "val"][-1]
        out.loc[i, "split"] = "dev"
    while (out["split"] == "val").sum() < 20:
        i = out.index[out["split"] == "dev"][-1]
        out.loc[i, "split"] = "val"

    p1 = art.parent / "p1"
    p1.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p1 / "profiles.parquet", index=False)

    print(f"프로필 {len(out)}개 → {p1/'profiles.parquet'}")
    print(f"  dev {(out.split=='dev').sum()} / val {(out.split=='val').sum()}")
    print(f"  시드 수 {out.n_seeds.min()}~{out.n_seeds.max()} · 응집도 "
          f"{out.seed_cohesion.min():.2f}~{out.seed_cohesion.max():.2f}")
    print(out[["profile_id", "n_seeds", "seed_cohesion", "split"]].to_string(index=False))


if __name__ == "__main__":
    main()
