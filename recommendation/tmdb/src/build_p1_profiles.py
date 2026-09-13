"""TMDB 평가 프로필 52개 — Steam 의 p1/profiles.parquet 과 같은 구조·같은 축.

Steam 은 52프로필 × k=50 · 적합률 0.85 를 기준으로 삼는다. 같은 잣대를 쓰기 위해
프로필 수(52) · 분할(dev 30 / val 22) · 크기 축(1~20) · 유형 축을 맞춘다.

축 구성이 Steam 과 다른 점 — 도메인이 다르므로 의도적으로 바꾼 것:
  · Steam 의 `lowrev_`(저리뷰)  →  TMDB 의 `lowvote_`(저투표)
    한국 영화가 코퍼스에서 106~363표로 실제 롱테일을 이룬다. 인위적 축이 아니다.
  · Steam 의 `publisher` 축 없음 — TMDB 에 배급사 필드가 없다.
  · TV 축 추가 — Steam 에 없는 매체 구분(movie 50,301 / tv 9,479)이 있다.

손으로 고른 것과 규칙으로 고른 것을 나눈다:
  · 손: 실제로 사람이 가질 법한 취향 묶음. 제목이 코퍼스에 있는지 전수 확인했다.
  · 규칙: 저투표·롱테일 축. 손으로 고르면 내가 아는 것만 뽑혀 편향된다.
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

T = Path(__file__).resolve().parents[1] / "artifacts" / "tmdb_v1"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "p1"
RNG = 20260908

# ── 손으로 고른 프로필 ────────────────────────────────────────────────────
HAND = {
 # 응집 — 한 축이 뚜렷한 취향
 "coh_marvel":        ["아이언맨","캡틴 아메리카: 시빌 워","토르: 라그나로크"],
 "coh_nolan":         ["인터스텔라","인셉션","다크 나이트"],
 "coh_pixar":         ["코코","월•E","토이 스토리"],
 "coh_ghibli":        ["센과 치히로의 행방불명","이웃집 토토로","하울의 움직이는 성"],
 "coh_slasher":       ["샤이닝","할로윈","스크림"],
 "coh_crime_classic": ["대부","좋은 친구들","스카페이스"],
 "coh_romcom":        ["노팅 힐","러브 액츄얼리","500일의 썸머"],
 "coh_zombie":        ["워킹 데드","좀비랜드","28일 후"],
 "coh_prestige_tv":   ["왕좌의 게임","브레이킹 배드","종이의 집"],
 "coh_sageuk":        ["명량","광해, 왕이 된 남자","관상"],          # 한국 사극 · 저투표
 # 혼합 — 두 축이 섞인 취향
 "mix2_arthouse":     ["위플래쉬","버드맨","문라이트"],
 "mix2_elevated_horror":["겟 아웃","유전","미드소마"],
 "mix2_kthriller":    ["살인의 추억","곡성","올드보이"],
 "mix2_anime_film":   ["너의 이름은.","날씨의 아이","목소리의 형태"],
 "mix2_romance_time": ["이터널 선샤인","어바웃 타임","비포 선라이즈"],
 "mix2_kaction":      ["도둑들","베테랑","극한직업"],                # 한국 오락 · 저투표
 # 크기 축 — 작은 쪽
 "single_interstellar":["인터스텔라"],
 "single_parasite":   ["기생충"],
 "two_classic":       ["쇼생크 탈출","포레스트 검프"],
 "two_ghibli":        ["센과 치히로의 행방불명","이웃집 토토로"],
 "two_kthriller":     ["곡성","부산행"],
 # 크기 축 — 큰 쪽
 "five_acclaimed":    ["쇼생크 탈출","펄프 픽션","포레스트 검프","다크 나이트","대부"],
 "five_horror":       ["샤이닝","엑소시스트","할로윈","스크림","28일 후"],
 "five_animation":    ["코코","업","인사이드 아웃","토이 스토리","라이온 킹"],
 "seven_tv":          ["왕좌의 게임","브레이킹 배드","기묘한 이야기","오징어 게임","종이의 집","워킹 데드","더 보이즈"],
 "seven_korean":      ["기생충","괴물","살인의 추억","올드보이","부산행","곡성","암살"],
 "ten_scifi":         ["인터스텔라","인셉션","매트릭스","아바타","어벤져스","아이언맨","가디언즈 오브 갤럭시",
                       "어벤져스: 인피니티 워","데드풀","해리 포터와 마법사의 돌"],
 "twenty_library":    ["인터스텔라","인셉션","다크 나이트","쇼생크 탈출","펄프 픽션","포레스트 검프","코코","업",
                       "토이 스토리","라이온 킹","타이타닉","라라랜드","이터널 선샤인","샤이닝","겟 아웃",
                       "기생충","올드보이","왕좌의 게임","브레이킹 배드","너의 이름은."],
}

# ── 규칙으로 고르는 축 (저투표 · 롱테일 · 니치) ───────────────────────────
# (이름, 장르, 투표수 구간, 개수)  — 구간은 코퍼스 분위에서 미리 정한다.
RULE = [
 ("lowvote_horror",   "공포",        (60, 400), 3), ("lowvote_romance",  "로맨스",      (60, 400), 3),
 ("lowvote_drama",    "드라마",      (60, 400), 3), ("lowvote_animation","애니메이션",  (60, 400), 3),
 ("lowvote_action",   "액션",        (60, 400), 3), ("lowvote_comedy",   "코미디",      (60, 400), 3),
 ("longtail_thriller","스릴러",      (30,  90), 3), ("longtail_family",  "가족",        (30,  90), 3),
 ("longtail_music",   "음악",        (30,  90), 3), ("longtail_war",     "전쟁",        (30,  90), 3),
 ("longtail_western", "서부",        (30, 200), 3), ("longtail_history", "역사",        (30,  90), 3),
 ("niche_doc",        "다큐멘터리",  (100, 4000), 4), ("niche_mystery",   "미스터리",   (150, 1200), 4),
 ("niche_fantasy",    "판타지",      (150, 1200), 4), ("niche_tv_drama",  "드라마",     (200, 1500), 4),
 ("five_lowvote_mix", None,          (60, 300), 5), ("five_tv_lowvote", None,          (80, 600), 5),
 ("seven_longtail",   None,          (30, 120), 7), ("ten_lowvote_broad",None,         (60, 400), 10),
 ("ten_midvote_mix",  None,          (400, 3000), 10), ("two_doc",       "다큐멘터리", (200, 2000), 2),
 ("two_western",      "서부",        (100, 3000), 2), ("single_lowvote", "공포",       (60, 200), 1),
]

def main():
    d = pd.read_parquet(T / "dataset.parquet")
    idx = pd.read_parquet(T / "corpus_index.parquet").sort_values("embedding_row")
    d = d.set_index("item_id").loc[idx["item_id"].to_numpy()].reset_index()
    d["row"] = np.arange(len(d))
    ko = d["overview"].fillna("").str.contains(r"[가-힣]")
    pool = d[ko].copy()                      # 서빙 가능 풀
    E = np.asarray(np.load(T / "corpus_embeddings.npy", mmap_mode="r"), dtype=np.float32)

    # 이름 → 행. 동명이인은 표수가 가장 많은 것 (D-20: 이름이 아니라 행으로 저장한다)
    by_name = {}
    for r in pool.itertuples(index=False):
        cur = by_name.get(str(r.name))
        if cur is None or r.vote_count > cur[1]: by_name[str(r.name)] = (int(r.row), int(r.vote_count))

    rows, miss = [], []
    for pid, titles in HAND.items():
        rr = []
        for t in titles:
            if t in by_name: rr.append(by_name[t][0])
            else: miss.append((pid, t))
        if rr: rows.append((pid, rr, "hand"))

    rng = np.random.default_rng(RNG)
    used = set()
    for pid, genre, (lo, hi), n in RULE:
        sub = pool[(pool["vote_count"] >= lo) & (pool["vote_count"] <= hi)]
        if genre: sub = sub[sub["genres"].apply(lambda v, g=genre: g in list(v))]
        if pid.startswith(("five_tv", "seven_tv")) or "tv" in pid: sub = sub[sub["media"] == "tv"]
        sub = sub[~sub["row"].isin(used)]
        if len(sub) < n: miss.append((pid, f"풀 부족 {len(sub)}<{n}")); continue
        # 응집된 취향을 만들려면 무작위 n 개가 아니라 **한 앵커의 이웃**을 뽑아야 한다.
        anchor = sub.sample(1, random_state=int(rng.integers(1 << 31))).iloc[0]
        av = E[int(anchor["row"])]
        sim = E[sub["row"].to_numpy()] @ av
        pick = sub.iloc[np.argsort(-sim)[:n]]["row"].astype(int).tolist()
        used |= set(pick)
        rows.append((pid, pick, "rule"))

    if miss:
        print("[누락]", *[f"  {a} · {b}" for a, b in miss], sep="\n")

    # 응집도 = 시드 임베딩들의 평균 쌍별 코사인 (Steam 과 같은 정의)
    recs = []
    for pid, rr, src in rows:
        V = E[rr]
        if len(rr) >= 2:
            S = V @ V.T; n = len(rr)
            coh = float((S.sum() - np.trace(S)) / (n * (n - 1)))
        else:
            coh = np.nan
        recs.append(dict(profile_id=pid, seed_rows=rr, n_seeds=len(rr),
                         source=src, seed_cohesion=coh))
    prof = pd.DataFrame(recs)
    # dev / val 분할 — 축 이름 접두사가 양쪽에 고루 가도록 순번으로 나눈다
    prof["prefix"] = prof["profile_id"].str.split("_").str[0]
    prof = prof.sort_values(["prefix", "profile_id"]).reset_index(drop=True)
    prof["split"] = ["dev" if i % 5 < 3 else "val" for i in range(len(prof))]
    prof["profile_order"] = np.arange(len(prof))
    prof = prof.drop(columns=["prefix"])
    OUT.mkdir(parents=True, exist_ok=True)
    prof.to_parquet(OUT / "profiles.parquet", index=False)
    print(f"\n프로필 {len(prof)}개 저장 → {OUT/'profiles.parquet'}")
    print(f"  분할 {prof['split'].value_counts().to_dict()} · 출처 {prof['source'].value_counts().to_dict()}")
    print(f"  크기 분포 {prof['n_seeds'].value_counts().sort_index().to_dict()}")
    print(f"  응집도 {prof['seed_cohesion'].min():.2f} ~ {prof['seed_cohesion'].max():.2f} "
          f"(중앙 {prof['seed_cohesion'].median():.2f})")

if __name__ == "__main__":
    main()
