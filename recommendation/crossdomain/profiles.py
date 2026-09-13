"""크로스 도메인 프로필 30개 — 기존 단일 플랫폼 프로필의 **조합**.

새 시드를 고르지 않는다. 각 구성 프로필의 시드 정의와 플랫폼 내 순위 품질은
이미 검증됐다(은행 17,748). 여기서 새로 묻는 것은 **조합**뿐이다.

"결이 맞는다/다르다"는 내가 정한 것이다. 주관이고, 실데이터가 생기면 대체한다.
`None` 은 그 플랫폼에 시드가 없다는 뜻이다 — 1차 설계에서는 쿼터를 주지 않는다.
"""

PROFILES = [
    # ── x3_coh: 세 플랫폼, 같은 결 ────────────────────────────────────────
    dict(pid="x3_coh_apocalypse",  steam="coh_survival_craft",     tmdb="coh_zombie",         wn="mix2_apocalypse"),
    dict(pid="x3_coh_fps_action",  steam="coh_fps",                tmdb="coh_marvel",         wn="coh_hunter"),
    dict(pid="x3_coh_cozy",        steam="coh_cozy",               tmdb="coh_ghibli",         wn="coh_romfan_divorce"),
    dict(pid="x3_coh_strategy",    steam="coh_grand_strategy",     tmdb="coh_prestige_tv",    wn="coh_modern_success"),
    dict(pid="x3_coh_horror",      steam="mix2_coop_horror",       tmdb="coh_slasher",        wn="mix2_mystery"),
    dict(pid="x3_coh_fantasy",     steam="coh_arpg",               tmdb="coh_nolan",          wn="coh_regression"),
    dict(pid="x3_coh_narrative",   steam="lowrev_cozy_narrative",  tmdb="coh_romcom",         wn="coh_romance_adult"),
    dict(pid="x3_coh_martial",     steam="niche_soulslike_solo",   tmdb="coh_sageuk",         wn="coh_murim"),
    # ── x3_clash: 세 플랫폼, 결이 다름 ─────────────────────────────────────
    dict(pid="x3_clash_1",         steam="coh_fps",                tmdb="coh_romcom",         wn="coh_romfan_revenge"),
    dict(pid="x3_clash_2",         steam="coh_cozy",               tmdb="coh_slasher",        wn="coh_murim"),
    dict(pid="x3_clash_3",         steam="coh_grand_strategy",     tmdb="coh_pixar",          wn="coh_bl_marriage"),
    dict(pid="x3_clash_4",         steam="niche_puzzle_solo",      tmdb="coh_zombie",         wn="coh_talent"),
    dict(pid="x3_clash_5",         steam="coh_vehicle_sim",        tmdb="coh_ghibli",         wn="mix2_villainess"),
    dict(pid="x3_clash_6",         steam="mix2_coop_horror",       tmdb="coh_crime_classic",  wn="coh_modern_success"),
    # ── x2: 두 플랫폼만 (시드 없는 플랫폼은 None) ──────────────────────────
    dict(pid="x2_st_apoc",         steam="coh_openworld_survival", tmdb="coh_zombie",         wn=None),
    dict(pid="x2_st_prestige",     steam="coh_strategy",           tmdb="coh_prestige_tv",    wn=None),
    dict(pid="x2_st_indie",        steam="coh_indie_platformer",   tmdb="mix2_anime_film",    wn=None),
    dict(pid="x2_st_horror",       steam="mix2_coop_horror",       tmdb="mix2_elevated_horror", wn=None),
    dict(pid="x2_tw_romance",      steam=None,                     tmdb="coh_romcom",         wn="coh_romfan_divorce"),
    dict(pid="x2_tw_action",       steam=None,                     tmdb="coh_marvel",         wn="coh_hunter"),
    dict(pid="x2_tw_sageuk",       steam=None,                     tmdb="coh_sageuk",         wn="coh_murim"),
    dict(pid="x2_tw_mystery",      steam=None,                     tmdb="coh_crime_classic",  wn="mix2_mystery"),
    dict(pid="x2_sw_hunter",       steam="coh_arpg",               tmdb=None,                 wn="coh_hunter"),
    dict(pid="x2_sw_cozy",         steam="coh_cozy",               tmdb=None,                 wn="coh_romfan_divorce"),
    dict(pid="x2_sw_success",      steam="mix2_builder_sim",       tmdb=None,                 wn="coh_modern_success"),
    dict(pid="x2_sw_murim",        steam="niche_soulslike_solo",   tmdb=None,                 wn="coh_murim_modern"),
    # ── x3_thin: 한 플랫폼이 D-23 얇은 장르 ────────────────────────────────
    dict(pid="x3_thin_myst",       steam="lowrev_detective",       tmdb="coh_crime_classic",  wn="rule_myst"),
    dict(pid="x3_thin_bl",         steam="coh_cozy",               tmdb="coh_romcom",         wn="rule_bl_mid"),
    dict(pid="x3_thin_myst2",      steam="niche_puzzle_solo",      tmdb="mix2_elevated_horror", wn="mix2_mystery"),
    dict(pid="x3_thin_bl2",        steam="lowrev_cozy_narrative",  tmdb="coh_ghibli",         wn="rule_bl_high"),
    # ── 확장 20 (X-5) — 기존 30 이 안 다룬 축 ──────────────────────────────
    # x3_unbal: 시드 개수 불균형(5:1:1 · 10:3:2 …). 기존 30 은 거의 3:3:3 이라 M1 이 M0 과 갈릴 일이 없었다.
    dict(pid="x3_unbal_steam10",  steam="ten_jrpg",              tmdb="two_ghibli",          wn="single_sword"),
    dict(pid="x3_unbal_steam7",   steam="seven_automation",      tmdb="single_interstellar", wn="two_gate"),
    dict(pid="x3_unbal_tmdb10",   steam="two_cozy_puzzle",       tmdb="ten_scifi",           wn="single_bastian"),
    dict(pid="x3_unbal_tmdb7",    steam="two_bigaction",         tmdb="seven_korean",        wn="two_murim"),
    dict(pid="x3_unbal_wn10",     steam="two_rhythm_arcade",     tmdb="two_classic",         wn="ten_mixed"),
    dict(pid="x3_unbal_wn7",      steam="lowrev_deckbuilder",    tmdb="two_kthriller",       wn="seven_fantasy"),
    # x3_lowpop: 세 플랫폼 다 저인기·롱테일 시드. 통합에서 인기 편향이 어느 플랫폼에서 나오나.
    dict(pid="x3_lowpop_1",       steam="lowrev_metroidvania",   tmdb="lowvote_horror",      wn="rule_hp_low"),
    dict(pid="x3_lowpop_2",       steam="longtail_detective",    tmdb="longtail_thriller",   wn="rule_rf_low"),
    dict(pid="x3_lowpop_3",       steam="lowrev_towerdefense",   tmdb="lowvote_animation",   wn="rule_fant_none"),
    dict(pid="x3_lowpop_4",       steam="longtail_puzzle_platformer", tmdb="longtail_music", wn="rule_rom_low"),
    # x3_korean: 한국 콘텐츠 결(K-스릴러 · 사극 · 무협 · 현판). 임베딩이 언어·문화 결을 잡나.
    dict(pid="x3_korean_1",       steam="coh_mmo",               tmdb="mix2_kthriller",      wn="rule_hp_high"),
    dict(pid="x3_korean_2",       steam="coh_jrpg",              tmdb="seven_korean",        wn="rule_mu_high"),
    dict(pid="x3_korean_3",       steam="mix2_soulslike_narrative", tmdb="two_kthriller",    wn="rule_mu_mid"),
    # x3_lowcoh: 세 플랫폼 다 저응집(<0.5). D-22 — 통합에서 M2 가 처음으로 의미를 가질 조건.
    dict(pid="x3_lowcoh_1",       steam="mix2_party_narrative",  tmdb="mix2_arthouse",       wn="two_murim"),
    dict(pid="x3_lowcoh_2",       steam="mix2_puzzle_survival",  tmdb="twenty_library",      wn="ten_mixed"),
    dict(pid="x3_lowcoh_3",       steam="ten_mixed_library",     tmdb="seven_tv",            wn="five_toptier"),
    # x3_bigfan: 대작만. 세 플랫폼 다 최상위 인기 시드 — 통합이 "대작 무한 반복"이 되나.
    dict(pid="x3_bigfan_1",       steam="twenty_broad",          tmdb="twenty_library",      wn="twenty_library"),
    dict(pid="x3_bigfan_2",       steam="five_coop_shooter",     tmdb="five_acclaimed",      wn="five_hunter"),
    # x2 추가: 웹소설+게임 조합이 12 중 4 뿐이었다. 두 개 더.
    # x2_sw_horror(seven_horror_coop + rule_ln) 은 X-6 에서 뺐다 — 라이트노벨 코퍼스가 178건,
    # 20화 이상 & 관심>0 이 15건뿐이라 시드 자체가 서 있을 곳이 없다(D-23 류). 내가 만든 억지 조합.
    dict(pid="x2_sw_horror2",     steam="seven_horror_coop",     tmdb=None,                  wn="rule_hp_mid"),
    dict(pid="x2_tw_family",      steam=None,                    tmdb="longtail_family",     wn="rule_rom_mid"),
]
assert len(PROFILES) == 50 and len({p["pid"] for p in PROFILES}) == 50
