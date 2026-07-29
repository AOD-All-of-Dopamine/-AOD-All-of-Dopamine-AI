# Claude(Opus 5) 판정 — 표현/랭킹 실험을 거치며 새로 등장한 후보들
# 키: (profile_id, candidate_appid) → (점수 0~3, 실패태그)
# 기준선 127쌍은 artifacts/human_eval/recommendation_review_claude.xlsx 에 있다.

# --- rep_v2 (Categories 플랫폼 문구 제거) 에서 새로 진입한 후보 ---
REP_V2 = {
    ("mix_vehicle_fps", 1250):   (3, ""),                    # Killing Floor — 6인 협동 생존호러 FPS
    ("mix_vehicle_fps", 17500):  (3, ""),                    # Zombie Panic! Source
    ("mix_vehicle_fps", 689030): (2, ""),                    # Infection Rate
    ("mix_vehicle_fps", 281920): (1, "GENRE_ONLY"),          # Splatter — 탑다운 슈터
    ("mix_vehicle_fps", 494220): (0, "IRRELEVANT"),          # Blight of the Immortals — 언데드 전략
    ("coh_classic_multi", 17500):  (3, ""),                  # Zombie Panic! Source
    ("coh_classic_multi", 689030): (2, ""),                  # Infection Rate
    ("coh_classic_multi", 281920): (1, "GENRE_ONLY"),        # Splatter
    ("coh_classic_multi", 494220): (0, "IRRELEVANT"),        # Blight of the Immortals
    ("coh_classic_multi", 657990): (2, ""),                  # Crafting Dead — 멀티 좀비 생존
    ("coh_classic_multi", 298240): (1, "TOO_NICHE"),         # War Trigger 3 — 저품질 무료 FPS
    ("mix_openworld_cozy", 632880): (2, ""),                 # Majesty of Colors — 수중 생명체 탐험
    ("coh_cozy", 673950): (3, ""),                           # Farm Together — Stardew 직결
    ("coh_cozy", 598240): (0, "KEYWORD_MATCH"),              # Stupid Raft Battle Sim — 'Raft' 단어만
    ("coh_cozy", 280790): (2, ""),                           # Creativerse — 샌드박스 크래프팅
    ("mix_rpg_racing", 491280): (2, ""),                     # Drift Horizon Online
    ("mix_rpg_racing", 352170): (1, "TOO_NICHE"),            # MadOut
    ("mix_rpg_racing", 400500): (1, "FRANCHISE_OR_VARIANT"), # MadOut Ice Storm
    ("mix_rpg_racing", 415600): (2, ""),                     # Kart Racing Pro
    ("mix_rpg_racing", 41740):  (2, ""),                     # Cargo! The Quest for Gravity
    ("mix_rpg_racing", 488550): (2, ""),                     # Dream Car Builder
    ("mix_rpg_racing", 582390): (0, "IRRELEVANT"),           # Ski Sport: Jumping VR
    ("mix_fps_cozy", 656240): (1, "GENRE_ONLY"),             # Heat — 야생 생존 샌드박스
    ("coh_indie_platformer", 634700): (1, "TOO_NICHE"),      # PLUTONIUM
    ("coh_indie_platformer", 94400):  (1, "MODE_MISMATCH"),  # Nidhogg — 대전 게임
    ("coh_indie_platformer", 541230): (0, "IRRELEVANT"),     # Alien Shooter TD
    ("mix_cozy_fps", 673950): (3, ""),                       # Farm Together
    ("mix_cozy_fps", 598240): (0, "KEYWORD_MATCH"),          # Stupid Raft Battle Sim
}

# --- 후처리(시드 인터리빙 + 시리즈 상한 + hard filter) 에서 새로 진입한 후보 ---
POSTPROCESS = {
    # P02 Garry's Mod / TF2 / Left 4 Dead 2
    ("coh_classic_multi", 696370): (2, ""),                  # BROKE PROTOCOL — 모드 가능 도시 샌드박스 멀티, GMod 결
    ("coh_classic_multi", 773850): (1, "FRANCHISE_OR_VARIANT"),  # WT2 — War Trigger 변형(설명문 동일). series_key 가 못 잡음
    ("coh_classic_multi", 70):     (2, ""),                  # Half-Life — Valve 계보 명작이나 싱글 클래식
    ("coh_classic_multi", 711570): (0, "IRRELEVANT"),        # Epic Battle Simulator 2 — 대규모 전투 시뮬
    # P03 No Man's Sky / 서브노티카 / Slime Rancher
    ("mix_openworld_cozy", 214730): (2, ""),                 # Space Rangers HD — 우주 오픈월드 RPG
    ("mix_openworld_cozy", 210970): (2, ""),                 # The Witness — 섬 퍼즐 탐험, 분위기 결
    # P06 CS2 / PUBG / Stardew
    ("mix_fps_cozy", 300):    (3, ""),                       # Day of Defeat: Source — Valve 멀티 FPS, CS2 정면
    ("mix_fps_cozy", 70):     (2, ""),                       # Half-Life — FPS 계보지만 경쟁 멀티 아님
    ("mix_fps_cozy", 308600): (1, "TOO_NICHE"),              # Skillshot City — 저품질 PvP 로그라이트
    # P07 Hollow Knight / Dead Cells / Celeste
    ("coh_indie_platformer", 496290): (2, ""),               # Deep Dark Dungeon — 로그라이트 액션, Dead Cells 결
    # P08 Raft / Stardew / CS2
    ("mix_cozy_fps", 300): (3, ""),                          # Day of Defeat: Source — CS2 결
}

ALL = {**REP_V2, **POSTPROCESS}
