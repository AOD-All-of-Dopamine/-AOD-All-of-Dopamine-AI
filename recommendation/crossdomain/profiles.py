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
]
assert len(PROFILES) == 30 and len({p["pid"] for p in PROFILES}) == 30
