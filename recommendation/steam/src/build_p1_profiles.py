# src/build_p1_profiles.py
"""평가 프로필 정의 — 판정 414쌍이 이 파일 위에 서 있다.

**절대 어기면 안 되는 불변식**

판정은 `(profile_id, candidate_appid)` 로만 키가 잡히는데 **관련성은 시드에 의존한다.**
따라서 기존 프로필의 `liked` 를 바꾸면 조인은 성공하는데 의미가 달라진다 —
가장 위험한 종류의 오염이다. `profile_id` 를 바꾸면 조인이 조용히 실패해 미판정이 된다.

  · 이미 판정된 프로필의 `liked` · `profile_id` 를 바꾸지 않는다.
  · `split` 은 라벨이라 바꿔도 판정은 안전하다. 단 **dev → val 이동은 금지** —
    튜닝에 쓴 프로필은 홀드아웃이 될 수 없다. val → dev 한 방향만 허용한다.

**왜 split 탐색을 없앴나**

예전 구현은 rng 로 섞어가며 통과하는 split 을 찾았다(`max_retries=100`). 그 방식은
(a) 이미 튜닝에 쓴 dev 프로필을 val 로 보낼 수 있고 (b) 판정 이력과 무관하게 재배치한다.
게다가 검증 기준이 `max_shared > 2` 라 **3개 중 2개 공유가 통과**했다. 실측 결과:

    val 8개 중 4개가 dev 프로필과 시드를 2개씩 공유
    val(누수 4개) P@10 0.800  vs  val(깨끗 4개) P@10 0.650  (dev 0.750)

"홀드아웃이 버텼다"는 결론이 그 누수분이 만든 것이었다. 지금은 split 을 정의에 명시하고
검증만 한다 — `max_shared >= 2` 면 실패한다.

**니치 축**

기존 34개 시드가 전부 코퍼스 상위 0.22%(최저 TF2 47,572 리뷰)라 니치 취향이 하나도 없었다.
그래서 "시드 인기도에 따라 인기도 부스트를 조절한다"는 실험이 모든 프로필에서 같은 값이라
검증 불가로 끝났다. `niche_*` 6개는 **리뷰 1,000~20,000 구간에서만** 뽑았고,
시드 개수도 1·3·5·10 으로 섞어 고정 3개 가정을 깬다.
"""
import hashlib

import numpy as np
import pandas as pd

DEV, VAL = "dev", "val"

# 기존 20개 — liked 는 절대 수정 금지 (판정 414쌍이 여기에 매여 있다).
# split 만 조정했다: 아래 4개가 dev 프로필과 시드를 2개씩 공유해 홀드아웃 자격이 없어
# val → dev 로 옮겼다 (mix_multi_indie / coh_openworld_survival / mix_indie_multi / coh_vehicle_sim).
LEGACY_PROFILES = [
    # --- 선언상 coherent ---
    {"profile_id": "coh_fps",                "liked": [730, 578080, 359550],    "declared": "coherent", "split": DEV},
    {"profile_id": "coh_survival_craft",     "liked": [108600, 219740, 242760], "declared": "coherent", "split": VAL},
    {"profile_id": "coh_openworld_survival", "liked": [264710, 305620, 275850], "declared": "coherent", "split": DEV},   # ← 누수
    {"profile_id": "coh_strategy",           "liked": [289070, 268500, 281990], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_grand_strategy",     "liked": [394360, 236850, 8930],   "declared": "coherent", "split": VAL},
    {"profile_id": "coh_arpg",               "liked": [374320, 292030, 489830], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_indie_platformer",   "liked": [367520, 588650, 504230], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_cozy",               "liked": [413150, 433340, 648800], "declared": "coherent", "split": DEV},
    {"profile_id": "coh_vehicle_sim",        "liked": [227300, 284160, 244210], "declared": "coherent", "split": DEV},   # ← 누수
    {"profile_id": "coh_classic_multi",      "liked": [4000, 440, 550],         "declared": "coherent", "split": DEV},
    # --- 선언상 mixed ---
    {"profile_id": "mix_fps_cozy",           "liked": [730, 578080, 413150],    "declared": "mixed", "split": DEV},
    {"profile_id": "mix_survival_strategy",  "liked": [242760, 251570, 289070], "declared": "mixed", "split": VAL},
    {"profile_id": "mix_rpg_racing",         "liked": [292030, 489830, 284160], "declared": "mixed", "split": DEV},
    {"profile_id": "mix_indie_multi",        "liked": [367520, 588650, 4000],   "declared": "mixed", "split": DEV},      # ← 누수
    {"profile_id": "mix_openworld_cozy",     "liked": [275850, 264710, 433340], "declared": "mixed", "split": DEV},
    {"profile_id": "mix_vehicle_fps",        "liked": [227300, 244210, 550],    "declared": "mixed", "split": DEV},
    {"profile_id": "mix_arpg_survival",      "liked": [374320, 377160, 242760], "declared": "mixed", "split": DEV},
    {"profile_id": "mix_grand_casual",       "liked": [8930, 236850, 477160],   "declared": "mixed", "split": VAL},
    {"profile_id": "mix_multi_indie",        "liked": [440, 550, 105600],       "declared": "mixed", "split": DEV},      # ← 누수
    {"profile_id": "mix_cozy_fps",           "liked": [648800, 413150, 730],    "declared": "mixed", "split": DEV},
]

# 니치 축 — 시드는 전부 리뷰 1,000~20,000. 시드 개수도 1·3·5·10 으로 섞는다.
NICHE_PROFILES = [
    # **공급 제약 프로필 (2026-08-12 정정)** — 처음엔 "코퍼스에 38개뿐이라 구조적으로 불가능"
    # 이라고 적었는데 **그건 리뷰 하한 안에서의 이야기였다.** 리뷰 300+ 풀(12,852개)에서는
    # Programming+Automation 후보가 38개지만, 전체 코퍼스(173,691)를 열면 P@50 이
    # 0.70 → 0.88 로 오른다. 한계를 만든 것은 카탈로그가 아니라 우리가 건 하한이었다.
    #
    # 좁게 진단한 것을 남겨 둔다 — "공급이 없다"고 결론짓기 전에 **어느 풀에서 없는지**를
    # 먼저 물어야 한다는 사례다. 프로필 자체는 손대지 않았다.
    {"profile_id": "niche_puzzle_solo", "split": VAL, "declared": "single",
     "liked": [558990]},                                                    # Opus Magnum 5,856
    {"profile_id": "niche_soulslike_solo", "split": VAL, "declared": "single",
     "liked": [283640]},                                                    # Salt and Sanctuary 19,987
    {"profile_id": "niche_tactics", "split": DEV, "declared": "coherent",
     "liked": [590380, 1102190, 287980]},                                   # Into the Breach / Monster Train / Mini Metro
    {"profile_id": "niche_cozy_casual", "split": DEV, "declared": "coherent",
     "liked": [1055540, 1046030, 557600]},                                  # A Short Hike / ISLANDERS / Gorogoa
    {"profile_id": "niche_sim", "split": VAL, "declared": "mixed",
     "liked": [1161580, 365450, 1677770, 1435790, 2162800]},                # Shipbreaker / Hacknet / Golden Idol / 방탈출 / shapez 2
    {"profile_id": "niche_roguelite", "split": DEV, "declared": "coherent",
     "liked": [894020, 330020, 692890, 418530, 242680,
               512900, 788100, 1123770, 1253920, 1740720]},                 # 10개
]

# 저리뷰 축 — 코퍼스의 70% 를 차지하는데 시드가 하나도 없던 구간.
#
# `niche_*` 를 만들 때 잡은 밴드(1,000~20,000)는 **구 코퍼스 기준**이었다. 전체 코퍼스로
# 넓히고 나서 다시 재보니 그것도 상위권이다:
#
#     리뷰 구간      코퍼스(21,883)  시드
#     101~299          41.3%          0     ← 완전 공백
#     300~2만          55.0%         23
#     2만+              3.7%         34     ← 코퍼스의 4% 에 시드의 60%
#
# 발견 제품의 대상 사용자는 "리뷰 200개짜리 인디를 좋아하는 사람"인데 그 취향을 한 번도
# 재본 적이 없다. 전체 코퍼스(173,691) 임베딩이 여는 구간이 정확히 여기다.
#
# 시드는 **태그로 축이 뚜렷한 것**만 골랐다. 무명 게임은 내가 사전 지식이 없으므로,
# 축이 모호하면 판정이 "설명끼리 닮았나"로 퇴화해 임베딩과 순환한다.
LOWREV_PROFILES = [
    {"profile_id": "lowrev_metroidvania", "split": DEV, "declared": "coherent",
     "liked": [1549750, 2316580, 253840]},      # Vomitoreum 897 / Tales of Kenzera 892 / Shantae 1,973
    {"profile_id": "lowrev_deckbuilder", "split": DEV, "declared": "coherent",
     "liked": [1016730, 1996430, 3332600]},     # Deck of Ashes 868 / Dicefolk 862 / Cubic Cosmos 892
    {"profile_id": "lowrev_towerdefense", "split": VAL, "declared": "coherent",
     "liked": [115120, 345090, 2257010]},       # Iron Brigade 899 / Ancient Planet 865 / Creeper World IXE 850
    {"profile_id": "lowrev_detective", "split": VAL, "declared": "coherent",
     "liked": [46480, 319870, 762830]},         # Still Life 886 / Jenny LeClue 1,238 / Telling Lies 1,239
    {"profile_id": "lowrev_cozy_narrative", "split": VAL, "declared": "mixed",
     "liked": [1272840, 1688580, 1425350]},     # Dordogne 891 / A YEAR OF SPRINGS 896 / Botany Manor 999
]

# 은퇴 — 정의는 남기되 평가에서 뺀다.
#
# `mix_*` 10개는 전부 `coh_*` 의 시드 3개 중 1개만 바꿔 만들어졌다. 그래서 측정이 겹친다.
# 상위 30칸 실측(2026-08-11, 태그 전체 코퍼스):
#
#     coh_fps ~ mix_fps_cozy              20/30 겹침
#     coh_cozy ~ mix_cozy_fps             20/30
#     coh_indie_platformer ~ mix_indie_multi  20/30
#     mix_fps_cozy ~ mix_cozy_fps         18/30   ← 둘 다 CS2+Stardew, 같은 프로필 두 번
#     ...
#     30칸이 8개 이상 겹치는 쌍 20개 — **20개 전부가 mix_* 를 끼고 있다.**
#     coh_/niche_/lowrev_ 21개끼리는 8개 이상 겹치는 쌍이 하나도 없다.
#
# 결과적으로 프로필 31개가 독립 측정 22개였고, 페어드 부트스트랩 CI 가 21% 좁게 나왔다
# (태그 효과 [+0.100,+0.206] → 덩어리 보정 [+0.113,+0.241]).
#
# **지우지 않고 은퇴시킨다.** liked 를 고치면 판정의 의미가 조용히 바뀌고, 정의를 지우면
# 과거 결과를 재현할 수 없다. 판정은 풀에 그대로 남아 있다 — 다시 쓰려면 여기서 빼면 된다.
RETIRED = {
    "mix_fps_cozy", "mix_survival_strategy", "mix_rpg_racing", "mix_indie_multi",
    "mix_openworld_cozy", "mix_vehicle_fps", "mix_arpg_survival", "mix_grand_casual",
    "mix_multi_indie", "mix_cozy_fps",
}

# 혼합 취향 v2 — 기존 시드 72개와 **한 개도 겹치지 않는** 게임으로 새로 짰다.
# 혼합 축 자체는 버릴 수 없다(대부분의 사람은 취향이 하나가 아니다). 버린 것은
# "coh_ 를 한 칸 흔든다"는 만드는 방식이고, 대신 실재할 법한 인물로 세웠다.
MIX_V2_PROFILES = [
    {"profile_id": "mix2_soulslike_narrative", "split": DEV, "declared": "mixed",
     "liked": [1245620, 632470, 753640]},        # ELDEN RING / Disco Elysium / Outer Wilds
    {"profile_id": "mix2_coop_horror", "split": DEV, "declared": "mixed",
     "liked": [1966720, 739630, 548430]},        # Lethal Company / Phasmophobia / Deep Rock Galactic
    {"profile_id": "mix2_colony_cozy", "split": VAL, "declared": "mixed",
     "liked": [294100, 457140, 1135690]},        # RimWorld / Oxygen Not Included / Unpacking
    {"profile_id": "mix2_arcade_action", "split": DEV, "declared": "mixed",
     "liked": [1551360, 1817230, 1364780]},      # Forza Horizon 5 / Hi-Fi RUSH / Street Fighter 6
    {"profile_id": "mix2_modern_roguelite", "split": DEV, "declared": "mixed",
     "liked": [2379780, 1145360, 1794680]},      # Balatro / Hades / Vampire Survivors
    {"profile_id": "mix2_builder_sim", "split": VAL, "declared": "mixed",
     "liked": [255710, 427520, 1248130]},        # Cities: Skylines / Factorio / Farming Simulator 22
    {"profile_id": "mix2_crpg_sandbox", "split": DEV, "declared": "mixed",
     "liked": [1086940, 261550, 233860]},        # Baldur's Gate 3 / Mount & Blade II / Kenshi
    {"profile_id": "mix2_survival_farm", "split": VAL, "declared": "mixed",
     "liked": [892970, 962130, 666140]},         # Valheim / Grounded / My Time at Portia
    {"profile_id": "mix2_party_narrative", "split": DEV, "declared": "mixed",
     "liked": [1426210, 728880, 501300]},        # It Takes Two / Overcooked! 2 / Edith Finch
    {"profile_id": "mix2_puzzle_survival", "split": DEV, "declared": "mixed",
     "liked": [257510, 252490, 1092790]},        # The Talos Principle / Rust / Inscryption
]

# 진짜 롱테일 — 시드 리뷰 100~300.
#
# `lowrev_*` 는 이름과 달리 롱테일이 아니었다(시드 850~1,973 = Steam 기준 중견 히트).
# 코퍼스 173,691 중 리뷰가 알려진 것은 21,892개뿐이고 그 중 9,040개가 100~300 구간인데,
# 이 구간을 시드로 가진 프로필이 하나도 없었다. **품질 하한을 열지 말지는 결국 이 구간의
# 추천이 쓸 만한지에 달려 있는데, 물어볼 대상 자체가 없었다.**
#
# `lowrev_*` 와 축을 일부러 맞췄다 — 같은 취향을 인기도 밴드만 바꿔 세 단계로 볼 수 있다
# (예: 메트로배니아 = coh_indie_platformer 13만~50만 · lowrev 850~1,973 · longtail 289~295).
LONGTAIL_PROFILES = [
    {"profile_id": "longtail_metroidvania", "split": DEV, "declared": "coherent",
     "liked": [922050, 1522930, 1550760]},       # DOOMBLADE 289 / Transiruby 291 / Blast Brigade 295
    {"profile_id": "longtail_deckbuilder", "split": VAL, "declared": "coherent",
     "liked": [2071430, 1716940, 2427450]},      # Roots of Yggdrasil 298 / Ancient Gods 275 / Flick Shot Rogues 288
    {"profile_id": "longtail_detective", "split": VAL, "declared": "coherent",
     "liked": [736810, 513890, 1201550]},        # The Raven 295 / The Frostrune 296 / Mad Experiments 298
    {"profile_id": "longtail_puzzle_platformer", "split": DEV, "declared": "coherent",
     "liked": [218740, 408650, 1803140]},        # Pid 296 / ChromaGun 294 / Deer & Boy 297
]

#: 2026-08-14 프로필 감사에서 드러난 공백을 메우는 8개.
#:
#: 감사 결과 커버리지·인기도·연식은 적절했으나 **좋아요 개수 분포가 현실을 못 덮었다**:
#: 35개 중 31개가 3시드라, 실사용자 여정(1 → 2 → 3 → … → 수십)에서 상시 측정되는
#: 지점이 사실상 하나뿐이었다. 축으로는 MMO·스포츠·JRPG 가 개발 셋에 아예 없었고
#: (홀드아웃에만 있었는데 1회 소진돼 회귀 감시가 불가능했다).
#:
#: TWO_SEED_PROFILES — **확인된 품질 저점이자 전원 경유지.** 전이 평가에서 2시드 구간이
#:   0.80~0.90 으로 저점이었다: 단일시드 희귀태그 필터는 꺼지는데 top2_mean 은 아직
#:   "2개 전부의 평균"이라 선택의 여지가 없다. 인기도 3밴드로 나눠 비례 하한의
#:   켜짐/꺼짐 분기도 함께 덮는다(대작 112k → 하한 112 · 중견 23k · 저리뷰 8.9k → 꺼짐).
#: GAP_AXIS_PROFILES — 홀드아웃 시드와 **완전 무겹침**으로 재구성한다. MMO(0.61)·
#:   스포츠(0.72)는 홀드아웃에서 실패한 축인데, 앞으로 고쳐도 잴 프로필이 없었다.
#: BIGLIB_PROFILES — 6시드(활성)·20시드(파워). 20시드는 인터리빙의 구조적 한계를
#:   직접 친다: 버킷 20개인데 페이지는 10칸이라 bucket_offset 회전이 실제로 도는지.
TWO_SEED_PROFILES = [
    # 대작 밴드(중앙 112k) — 시드 비례 하한이 켜지는 쪽
    {"profile_id": "two_bigaction", "liked": [1593500, 1151640], "declared": "coherent", "split": DEV},
    # 중견 밴드(중앙 23k) — 리듬 축도 함께 보강(개발 셋 시드 1개뿐이었다)
    {"profile_id": "two_rhythm_arcade", "liked": [247080, 531510], "declared": "coherent", "split": VAL},
    # 저리뷰 밴드(중앙 8.9k) — 하한이 꺼지는 쪽
    {"profile_id": "two_cozy_puzzle", "liked": [1307580, 3035120], "declared": "coherent", "split": DEV},
]

GAP_AXIS_PROFILES = [
    # 홀드아웃 ho_mmo(ESO/New World/OSRS) 와 무겹침
    {"profile_id": "coh_mmo", "liked": [1284210, 2429640, 761890], "declared": "coherent", "split": VAL},
    # 홀드아웃 ho_sports(EA FC25/NBA2K25/FM26) 와 무겹침
    {"profile_id": "coh_sports", "liked": [1506830, 3230400, 2385530], "declared": "coherent", "split": DEV},
    # 홀드아웃 ho_jrpg(P4G/DQXI/Tales of Arise) 와 무겹침
    {"profile_id": "coh_jrpg", "liked": [2161700, 921570, 251150], "declared": "coherent", "split": VAL},
]

BIGLIB_PROFILES = [
    # 활성 사용자 6시드 — 3인칭 액션 어드벤처 대작 취향
    {"profile_id": "six_action_adv", "liked": [1174180, 1091500, 1817070, 203160, 1332010, 1259420],
     "declared": "coherent", "split": DEV},
    # 파워 유저 20시드 — 현실적인 라이브러리는 넓다. 인터리빙 버킷 20 vs 페이지 10 검증용.
    {"profile_id": "twenty_broad",
     "liked": [105600, 322330, 400, 391540, 40800, 387290, 632360, 250900, 646570, 49520,
               22380, 582010, 381210, 286160, 200510, 570940, 346110, 594650, 1085660, 377160],
     "declared": "mixed", "split": VAL},
]

ALL_PROFILES = [p for p in LEGACY_PROFILES if p["profile_id"] not in RETIRED] \
    + NICHE_PROFILES + LOWREV_PROFILES + MIX_V2_PROFILES + LONGTAIL_PROFILES \
    + TWO_SEED_PROFILES + GAP_AXIS_PROFILES + BIGLIB_PROFILES

NICHE_REVIEW_BAND = (1000, 20000)
LOWREV_REVIEW_BAND = (100, 2000)
LONGTAIL_REVIEW_BAND = (100, 300)


# ------------------------------------------------------------------ 검증

def seeds_fingerprint(profiles: list[dict]) -> str:
    """`(profile_id, liked)` 의 지문. 기존 프로필이 조용히 바뀌었는지 잡는다."""
    h = hashlib.sha256()
    for p in sorted(profiles, key=lambda x: x["profile_id"]):
        h.update(f"{p['profile_id']}:{','.join(map(str, p['liked']))}|".encode())
    return h.hexdigest()[:12]


def validate_split(profiles: list[dict], max_shared: int = 1) -> dict:
    """dev/val 이 진짜로 분리돼 있는지 검증한다. 탐색하지 않고 주어진 것을 본다.

    `max_shared=1` — 시드 3개 중 2개를 공유하면 사실상 같은 프로필이다.
    예전 기준(`> 2` 일 때만 실패)은 그것을 통과시켰고, 그래서 홀드아웃이 오염됐다.
    """
    dev = [p for p in profiles if p["split"] == DEV]
    val = [p for p in profiles if p["split"] == VAL]
    if not dev or not val:
        raise ValueError(f"split 이 한쪽으로 쏠렸습니다: dev {len(dev)} / val {len(val)}")

    ids = [p["profile_id"] for p in profiles]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"profile_id 중복: {dup}")

    worst, leaks = 0, []
    for v in val:
        sv = set(v["liked"])
        for d in dev:
            shared = len(sv & set(d["liked"]))
            worst = max(worst, shared)
            if shared > max_shared:
                leaks.append(f"{v['profile_id']} ~ {d['profile_id']} ({shared}개 공유)")
    if leaks:
        raise ValueError(
            f"dev–val 시드 누수 {len(leaks)}건 (허용 {max_shared}개 이하):\n  "
            + "\n  ".join(leaks)
        )

    dev_seeds = {a for p in dev for a in p["liked"]}
    return {
        "dev": len(dev), "val": len(val),
        "max_shared_dev_val": worst,
        "val_seeds": len({a for p in val for a in p["liked"]}),
        "val_seeds_also_in_dev": len({a for p in val for a in p["liked"]} & dev_seeds),
        "unique_seeds": len({a for p in profiles for a in p["liked"]}),
        "seed_slots": sum(len(p["liked"]) for p in profiles),
    }


def validate_independence(profiles: list[dict], max_shared: int = 1) -> dict:
    """**어떤 두 프로필도** 시드를 `max_shared` 개 넘게 공유하지 않는지 본다.

    `validate_split` 은 dev–val 만 봤다. 그래서 dev 안에서 서로 2개씩 공유하는 쌍이
    통과했고, 그게 `mix_*` 10개였다 — 페어드 부트스트랩은 프로필을 독립 표본으로 다루는데
    실제로는 31개가 22개였다. CI 가 21% 좁게 나온 원인이다.

    시드를 공유하지 않아도 출력이 겹칠 수는 있으므로 이것은 필요조건일 뿐이다.
    실측으로는 시드 공유 0~1개인 쌍들의 상위 30칸 겹침이 전부 8개 미만이었다.
    """
    dup = []
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            shared = len(set(a["liked"]) & set(b["liked"]))
            if shared > max_shared:
                dup.append(f"{a['profile_id']} ~ {b['profile_id']} ({shared}개 공유)")
    if dup:
        raise ValueError(
            f"시드를 {max_shared}개 넘게 공유하는 프로필 쌍 {len(dup)}건 — 독립 측정이 아닙니다:\n  "
            + "\n  ".join(dup)
        )
    return {"profiles": len(profiles), "max_shared_any_pair": max_shared}


def validate_niche(profiles: list[dict], dataset: pd.DataFrame) -> dict:
    """니치 프로필의 시드가 정말 니치 구간인지 확인한다.

    이것을 안 재면 "니치 축을 추가했다"고 적어놓고 실제로는 또 대작만 들어 있을 수 있다.
    """
    rec = dataset.set_index("steam_appid")["recommendations_total"].astype("float")
    lo, hi = NICHE_REVIEW_BAND
    bad = []
    for p in profiles:
        if not p["profile_id"].startswith("niche_"):
            continue
        for a in p["liked"]:
            r = rec.get(a)
            if pd.isna(r) or not (lo <= r <= hi):
                bad.append(f"{p['profile_id']}/{a} 리뷰 {r}")
    if bad:
        raise ValueError(f"니치 구간({lo:,}~{hi:,}) 밖의 시드:\n  " + "\n  ".join(bad))

    for prefix, (lo2, hi2) in (("lowrev_", LOWREV_REVIEW_BAND),
                               ("longtail_", LONGTAIL_REVIEW_BAND)):
        bad2 = []
        for p in profiles:
            if not p["profile_id"].startswith(prefix):
                continue
            for a in p["liked"]:
                r = rec.get(a)
                if pd.isna(r) or not (lo2 <= r <= hi2):
                    bad2.append(f"{p['profile_id']}/{a} 리뷰 {r}")
        if bad2:
            raise ValueError(f"{prefix} 구간({lo2:,}~{hi2:,}) 밖의 시드:\n  " + "\n  ".join(bad2))

    all_seeds = [a for p in profiles for a in p["liked"]]
    revs = rec.reindex(all_seeds).fillna(0)
    pct = rec.fillna(0).rank(pct=True).reindex(all_seeds)
    return {
        "seed_reviews_min": int(revs.min()), "seed_reviews_max": int(revs.max()),
        "seed_percentile_min": round(float(pct.min()), 4),
        "seed_percentile_max": round(float(pct.max()), 4),
    }


# ------------------------------------------------------- 라벨을 측정으로 대체

def compute_seed_cohesion(profiles: list[dict], artifacts: str | None = None) -> dict[str, float]:
    """시드 쌍 코사인의 평균. 시드 1개면 NaN (쌍이 없다).

    손으로 붙인 `coherent`/`mixed` 라벨이 데이터로 확인되지 않았기 때문에 넣는다 —
    실측하면 coherent 평균 0.581 / mixed 0.528 로 구간이 완전히 겹쳤고
    `coh_arpg`(0.478) 가 `mix_multi_indie`(0.643) 보다 덜 일관적이었다.
    """
    from src.personalization.seed_loader import SeedLoader

    loader = SeedLoader(artifacts)
    out = {}
    for p in profiles:
        liked = p["liked"]
        if len(liked) < 2:
            out[p["profile_id"]] = float("nan")
            continue
        vecs = list(loader.load(liked).values())
        sims = [float(np.dot(a, b)) for i, a in enumerate(vecs) for b in vecs[i + 1:]]
        out[p["profile_id"]] = float(np.mean(sims))
    return out


def derive_profile_type(cohesion: dict[str, float]) -> dict[str, str]:
    """실측 응집도의 중앙값으로 coherent/mixed 를 가른다. 시드 1개는 `single`."""
    vals = [v for v in cohesion.values() if not np.isnan(v)]
    med = float(np.median(vals)) if vals else 0.0
    return {
        pid: "single" if np.isnan(v) else ("coherent" if v >= med else "mixed")
        for pid, v in cohesion.items()
    }


# ------------------------------------------------------------------ 빌드

def build_profiles_df(artifacts: str | None = None, dataset: pd.DataFrame | None = None,
                      profiles: list[dict] | None = None) -> pd.DataFrame:
    profiles = profiles or ALL_PROFILES
    stats = validate_split(profiles)
    stats |= validate_independence(profiles)
    if dataset is not None:
        stats |= validate_niche(profiles, dataset)

    cohesion = compute_seed_cohesion(profiles, artifacts)
    derived = derive_profile_type(cohesion)

    df = pd.DataFrame([{
        "profile_id": p["profile_id"],
        "liked_appids": list(p["liked"]),
        "n_seeds": len(p["liked"]),
        "profile_type": derived[p["profile_id"]],
        "profile_type_declared": p["declared"],
        "seed_cohesion": cohesion[p["profile_id"]],
        "split": p["split"],
        "profile_order": i,
    } for i, p in enumerate(profiles)])
    df.attrs["stats"] = stats
    return df


def main():
    import argparse
    import json

    from src.config import ARTIFACTS_DIR, artifact_dir

    # 기본 artifacts 는 s1_v2(트렌드 트랙)라 개인화 코퍼스가 아니다.
    # 니치 구간 검증과 응집도 계산은 전체 코퍼스 임베딩이 필요하다.
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts/full_v1")
    args = ap.parse_args()

    art = artifact_dir(args.artifacts)
    ds_path = art / "dataset.parquet"
    dataset = pd.read_parquet(ds_path) if ds_path.exists() else None

    df = build_profiles_df(artifacts=args.artifacts, dataset=dataset)
    p1_dir = ARTIFACTS_DIR.parent / "p1"
    p1_dir.mkdir(parents=True, exist_ok=True)
    path = p1_dir / "profiles.parquet"
    df.to_parquet(path, index=False)

    print(f"저장: {path}  ({len(df)}행)")
    print(json.dumps(df.attrs["stats"], ensure_ascii=False, indent=2))
    print(f"시드 지문: {seeds_fingerprint(ALL_PROFILES)}  (기존 프로필이 바뀌면 달라진다)")
    flipped = df[(df.profile_type != df.profile_type_declared)]
    print(f"\n선언 라벨과 실측이 다른 프로필 {len(flipped)}/{len(df)}:")
    for _, r in flipped.iterrows():
        print(f"  {r.profile_id:24s} {r.profile_type_declared} → {r.profile_type} "
              f"(응집도 {r.seed_cohesion:.3f})")
    print()
    for _, r in df.iterrows():
        print(f"  {r.profile_id:24s} [{r.split}] seeds={r.n_seeds:2d} "
              f"cohesion={r.seed_cohesion:.3f} {r.profile_type}")


if __name__ == "__main__":
    main()


#: 코퍼스에 적합 후보가 목표 커버리지보다 적어, 시스템 개선으로는 0.8 을 넘길 수 없는 프로필.
#: 평가에서 빼지 않는다 — 계속 측정하되 "모든 프로필 0.8 이상" 판정에서만 제외한다.
#: 2026-08-12 하한 개방으로 해소됨 — niche_puzzle_solo 는 P@50 0.70 → 0.88 이 됐다.
#: 비워 두되 개념은 남긴다: 앞으로 어떤 프로필이 여기 들어오려면 **전체 코퍼스 기준**으로
#: 공급이 모자란다는 것을 보여야 한다. 하한 안에서 모자란 것은 하한 문제지 공급 문제가 아니다.
SUPPLY_LIMITED: set[str] = set()

#: 홀드아웃 프로필 — **1회성 일반화 검증 전용.** ALL_PROFILES(개발 셋)에 절대 넣지 않는다.
#:
#: 존재 이유: 35프로필 위에서 레버 7개를 채택하고 하이퍼파라미터(λ=0.35, 계수 0.001)까지
#: 같은 데이터로 골랐다. "전 프로필 ≥0.8"이 최적화 목표 자체였으므로, 그 수치는 낙관
#: 편향을 포함할 수 있다. 이 프로필들은 어떤 레버 결정에도 쓰인 적이 없는 상태에서
#: **딱 한 번** 평가해 정직한 일반화 추정치를 얻는다.
#:
#: 규율: 결과가 나빠도 여기에 맞춰 튜닝하지 않는다. 튜닝에 쓰는 순간 개발 셋이 되므로,
#: 그때는 새 홀드아웃을 만들어야 한다. 축은 개발 셋의 공백을 채운다:
#: JRPG · 솔로 호러 · MMO · 스포츠 · 리듬 · 모순 취향 · 대형 라이브러리(시드 12개).
#:
#: ho_contradictory 판정 기준: 시드 간 합의가 정의상 없으므로, 최소 한 시드의 취향에
#: 강하게 부합하면 타당으로 본다(다른 프로필은 라이브러리 전체 취향 기준).
HOLDOUT_PROFILES = [
    {"profile_id": "ho_jrpg",          "liked": [1113000, 1295510, 740130], "declared": "coherent", "split": "holdout"},  # P4G / DQ XI S / Tales of Arise
    {"profile_id": "ho_horror_solo",   "liked": [238320, 214490, 594330],   "declared": "coherent", "split": "holdout"},  # Outlast / Alien: Isolation / Visage
    {"profile_id": "ho_mmo",           "liked": [306130, 1063730, 1343370], "declared": "coherent", "split": "holdout"},  # ESO / New World / OSRS
    {"profile_id": "ho_sports",        "liked": [2669320, 2878980, 3551340],"declared": "coherent", "split": "holdout"},  # EA FC 25 / NBA 2K25 / FM26
    {"profile_id": "ho_rhythm",        "liked": [774171, 960170, 774181],   "declared": "coherent", "split": "holdout"},  # Muse Dash / DJMAX / Rhythm Doctor
    {"profile_id": "ho_contradictory", "liked": [570, 613100, 782330, 1222670], "declared": "mixed", "split": "holdout"}, # Dota 2 / House Flipper / DOOM Eternal / Sims 4
    {"profile_id": "ho_biglib",        "liked": [620, 220, 205100, 480490, 870780, 337000, 8870, 282140, 383870, 1237970, 412020, 268910], "declared": "mixed", "split": "holdout"},
    # ho_biglib: Portal 2 / HL2 / Dishonored / Prey / Control / Deus Ex MD / BioShock Inf /
    # SOMA / Firewatch / Titanfall 2 / Metro Exodus / Cuphead — 현실 라이브러리답게
    # 이질적인 것(Cuphead) 하나를 일부러 남겼다.
]

#: 동적 시나리오 — **좋아요가 늘어날 때** 추천이 어떻게 바뀌는지 재는 전이 평가.
#:
#: 정적 평가(고정 시드 × k=100)가 못 보는 세 가지를 잰다:
#:   성장(grow):   시드 1 → 2 → 3. 특히 1→2 는 코드 경로 경계다 — 단일시드 희귀태그
#:                 필터가 꺼지고 top2_mean 이 max 에서 실질 mean 으로 바뀐다. 절벽 검출.
#:   전환(pivot):  기존 취향 3개로 2페이지 소비한 뒤 다른 축 시드를 추가. 다음 2페이지에
#:                 새 취향이 (좋은 품질로) 나타나는지. 인터리빙의 존재 이유를 검증한다.
#:   안정(stable): 기존 취향과 일치하는 4번째 시드 추가. top-20 이 불필요하게 요동치는지.
#:
#: 판정 문맥: 각 단계의 시드 집합이 그 시점의 취향이다. pivot 판정은 "기존 취향 또는
#: 새 시드 취향에 부합하면 타당" — 사람이 실제로 두 취향을 다 가진 상태이므로.
#:
#: 2026-08-13 실측 (미판정 0, 판정은 BLIND_DYN + 개발셋 재사용):
#:   성장  cozy 0.95 → 0.80 → 0.95 · det 1.00 → 0.90 → 0.95 (P@20)
#:         **2시드가 저점이다.** 단일시드 희귀태그 필터는 꺼지는데 top2_mean 은 아직
#:         "2개 전부의 평균"이라 두 시드의 퍼지한 중간 지점을 찾는다(Stardew+SlimeRancher
#:         → Space Rangers·Bum Simulator 같은 '기묘한 샌드박스'가 샘). 3점 비중은
#:         단계마다 오른다(50→55→60%, 50→50→75%) — 시드가 늘수록 강한 추천이 늘어난다.
#:   전환  새 시드가 즉시 다음 2페이지의 5~6/20칸을 받고(인터리빙 검증), 그 칸들의 품질이
#:         전부 2점 이상. 전체 페이지 품질 0.90~1.00 (대조군 1.00).
#:   안정  일치하는 4번째 시드 추가 시 top-20 의 9~13/20 유지, 품질 무손실
#:         (0.90→0.85 · 0.95→1.00). 교체분도 고품질이라 churn 은 손실이 아니다.
#:
#: 미평가로 남은 전이: 장기 드리프트(수십 개 좋아요 누적), DISLIKE 반영, 좋아요 취소,
#: 연속 전환(pivot 여러 번).
DYN_SCENARIOS = {
    # 성장: 개발 셋 프로필의 시드를 순서대로 늘린다 (3단계 = 원 프로필과 동일 → 판정 재사용)
    "dyn_grow_cozy":     {"base": [413150, 433340, 648800], "kind": "grow", "map_to": "coh_cozy"},
    "dyn_grow_det":      {"base": [46480, 319870, 762830],  "kind": "grow", "map_to": "lowrev_detective"},
    # 전환: base 로 2페이지(20칸) 소비 후 add 를 좋아요 → 다음 2페이지 평가
    "dyn_pivot_cozy":    {"base": [413150, 433340, 648800], "add": 1245620, "kind": "pivot"},  # 코지 + ELDEN RING
    "dyn_pivot_tactics": {"base": [590380, 1102190, 287980], "add": 739630, "kind": "pivot"},  # 전술 + Phasmophobia
    # 안정: 일치하는 4번째 시드 추가 → 처음부터 top-20 재생성, 전후 비교
    "dyn_stable_arpg":   {"base": [374320, 292030, 489830], "add": 1627720, "kind": "stable", "map_to": "coh_arpg"},   # + P의 거짓
    "dyn_stable_cozy":   {"base": [413150, 433340, 648800], "add": 666140,  "kind": "stable", "map_to": "coh_cozy"},   # + My Time at Portia
}
#: 랭킹 레버로는 못 올리는 프로필 — 원인이 **표현(임베딩)** 에 있다.
#:
#: coh_arpg (DARK SOULS III / The Witcher 3 / Skyrim), P@50 0.76:
#:   Action RPG + Open World 를 모두 가진 리뷰 300+ 후보가 231개인데 상위 50에 13개뿐이다.
#:   명작들의 유사도가 상위 50 구간(0.705~0.621) 아래에 깔려 있다:
#:
#:     Fallout: New Vegas    194위  0.5964   Monster Hunter World  223위  0.5944
#:     Fallout 4             441위  0.5814   God of War            653위  0.5739
#:     Horizon Zero Dawn     699위  0.5726   호그와트 레거시        1,566위  0.5564
#:
#:   대신 Weird RPG 2(776리뷰, 0.66) 같은 무명작이 위에 온다. 즉 임베딩이 "Skyrim ≈ Fallout 4"
#:   를 못 잡는다. 시험하고 기각한 것:
#:     · rec_boost 0.15 → 0.60 / 1.50   유사도 격차(0.02~0.07)를 못 뒤집는다. 0.76 → 0.74
#:     · 합의 태그 필터                  이들은 전부 통과한다(Action RPG 보유). 필터 문제 아님
#:     · 설명 길이 편향 의심             ρ 0.028 (p 1.6e-06). 상관 없음 — 재임베딩 근거 없음
#:
#: 고치려면 표현을 바꿔야 한다(설명·태그 외 신호). 랭킹 레벨에서는 여기가 상한이다.
REPRESENTATION_LIMITED = {"coh_arpg"}
