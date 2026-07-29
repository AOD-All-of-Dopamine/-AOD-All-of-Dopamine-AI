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

# --- 품질 하한(require_known_reviews) + rec_boost 0.15 의 rank 1~30 에서 새로 진입한 후보 ---
DEPTH_V1 = {
    # P04 coh_cozy: Stardew Valley / Slime Rancher / Raft
    ("coh_cozy", 656240): (2, ""),                     # Heat — 야생 생존+정착
    ("coh_cozy", 573090): (1, "GENRE_ONLY"),           # Stormworks — 구조 차량 설계 시뮬
    ("coh_cozy", 257850): (0, "IRRELEVANT"),           # Hyper Light Drifter — 액션 RPG
    ("coh_cozy", 666140): (3, ""),                     # My Time at Portia — Stardew 직결
    ("coh_cozy", 211820): (2, ""),                     # Starbound — 우주 샌드박스 크래프팅
    ("coh_cozy", 257170): (1, "GENRE_ONLY"),           # Rebuild 3 — 종말 도시 재건 전략
    ("coh_cozy", 361420): (2, ""),                     # ASTRONEER — 행성 탐험+기지
    ("coh_cozy", 252390): (1, "GENRE_ONLY"),           # DwarfCorp — 식민지 건설 전략
    ("coh_cozy", 768200): (2, ""),                     # Smalland — 소인 멀티 생존+길들이기
    ("coh_cozy", 688060): (1, "GENRE_ONLY"),           # Odd Realm — 정착지 건설
    ("coh_cozy", 359320): (0, "IRRELEVANT"),           # Elite Dangerous — 우주 MMO
    ("coh_cozy", 416000): (3, ""),                     # Orange Season — 농장 생활 시뮬
    ("coh_cozy", 252870): (0, "IRRELEVANT"),           # PULSAR — 협동 우주선 시뮬
    ("coh_cozy", 424590): (2, ""),                     # Farm Expert 2017 — 농업 시뮬
    ("coh_cozy", 328220): (3, ""),                     # Wild Season — Stardew 계열 농장 시뮬
    ("coh_cozy", 242760): (2, ""),                     # The Forest — 숲 생존(호러)
    # P05 mix_rpg_racing: Witcher 3 / Skyrim / BeamNG
    ("mix_rpg_racing", 306130): (2, ""),               # Elder Scrolls Online — Skyrim 세계관 MMO
    ("mix_rpg_racing", 668580): (3, ""),               # 아토믹 하트 — 1인칭 액션 RPG
    ("mix_rpg_racing", 384570): (1, "TOO_NICHE"),      # Zanzarah — 구형 마이너 판타지 RPG
    ("mix_rpg_racing", 351100): (1, "GENRE_ONLY"),     # Niffelheim — 2D 생존 액션
    ("mix_rpg_racing", 230290): (1, "GENRE_ONLY"),     # Universe Sandbox — 우주 물리 샌드박스
    ("mix_rpg_racing", 585710): (0, "IRRELEVANT"),     # Blazing Beaks — 로그라이트
    ("mix_rpg_racing", 757310): (3, ""),               # Sable — 사막 오픈월드 탐험 RPG
    ("mix_rpg_racing", 477770): (2, ""),               # Ride 2 — 바이크 레이싱 시뮬
    ("mix_rpg_racing", 683900): (0, "IRRELEVANT"),     # RollerCoaster Tycoon — 경영
    ("mix_rpg_racing", 365960): (3, ""),               # rFactor 2 — 사실적 레이싱 시뮬, BeamNG 정면
    ("mix_rpg_racing", 359220): (2, ""),               # MX vs ATV — 모토크로스
    ("mix_rpg_racing", 665370): (1, "KEYWORD_MATCH"),  # Mutilate-a-Doll 2 — '물리'만 겹침
    ("mix_rpg_racing", 406350): (2, ""),               # KartKraft — 카트 시뮬
    ("mix_rpg_racing", 408740): (0, "IRRELEVANT"),     # Pro Gamer Manager — e스포츠 경영
    ("mix_rpg_racing", 431600): (3, ""),               # Automobilista — 사실적 레이싱 시뮬
    ("mix_rpg_racing", 541380): (1, "TOO_NICHE"),      # PhysDrive — 저품질(rev 137)
    ("mix_rpg_racing", 497180): (3, ""),               # Street Legal Racing — 차량 정비/튜닝, BeamNG 결
    ("mix_rpg_racing", 638000): (0, "IRRELEVANT"),     # When Ski Lifts Go Wrong — 건설 퍼즐
    ("mix_rpg_racing", 299970): (2, ""),               # Project Motor Racing — 모터스포츠 시뮬
    ("mix_rpg_racing", 461430): (2, ""),               # EmergeNYC — 응급차량 운전 시뮬
    # P06 mix_fps_cozy: CS2 / PUBG / Stardew
    ("mix_fps_cozy", 765410): (2, ""),                 # 빈 배틀즈 — 캐주얼 배틀로얄
    ("mix_fps_cozy", 1250):   (2, ""),                 # Killing Floor — 협동 생존 FPS
    ("mix_fps_cozy", 433850): (3, ""),                 # Z1 Battle Royale — PUBG 정면
    ("mix_fps_cozy", 222880): (3, ""),                 # Insurgency — 근접 전술 FPS, CS2 정면
    ("mix_fps_cozy", 298240): (1, "TOO_NICHE"),        # War Trigger 3 — rev 161
    ("mix_fps_cozy", 351230): (1, "TOO_NICHE"),        # Counter Spell — rev 116
    ("mix_fps_cozy", 391460): (1, "TOO_NICHE"),        # WARMODE — rev 329
    ("mix_fps_cozy", 42700):  (3, ""),                 # CoD: Black Ops — 대작 FPS
    ("mix_fps_cozy", 417910): (1, "GENRE_ONLY"),       # Street Warriors — 주먹 난투
    ("mix_fps_cozy", 382310): (2, ""),                 # Eco — 협동 건설(Stardew 축)
    ("mix_fps_cozy", 291550): (1, "GENRE_ONLY"),       # Brawlhalla — 플랫폼 파이터
    ("mix_fps_cozy", 318650): (0, "IRRELEVANT"),       # SunAge — 2D RTS, rev 170
    ("mix_fps_cozy", 671860): (3, ""),                 # BattleBit Remastered — 대규모 FPS
    # P07 coh_indie_platformer: Hollow Knight / Dead Cells / Celeste
    ("coh_indie_platformer", 269650): (3, ""),         # Dex — 2D 사이드스크롤 오픈월드, 메트로배니아 계열
    ("coh_indie_platformer", 740080): (1, "GENRE_ONLY"),   # Deadly Days — 좀비 전략 로그라이트
    ("coh_indie_platformer", 45740):  (0, "IRRELEVANT"),   # Dead Rising 2 — 3D 좀비 오픈월드
    ("coh_indie_platformer", 310510): (0, "IRRELEVANT"),   # Deathtrap — 타워디펜스
    ("coh_indie_platformer", 619820): (2, ""),         # Heroes of Hammerwatch II — 로그라이트 액션
    ("coh_indie_platformer", 542050): (3, ""),         # Forgotton Anne — 손그림 2D 액션 어드벤처
    ("coh_indie_platformer", 27940):  (0, "IRRELEVANT"),   # Dead Horde — 저품질 좀비
    ("coh_indie_platformer", 477160): (1, "MODE_MISMATCH"),# Human Fall Flat — 캐주얼 물리 퍼즐
    ("coh_indie_platformer", 668550): (3, ""),         # 사망여각 8Doors — 메트로베니아 2D 플랫포머
    ("coh_indie_platformer", 408410): (0, "IRRELEVANT"),   # X-Morph: Defense — 타워디펜스
    ("coh_indie_platformer", 568930): (0, "IRRELEVANT"),   # The Land of Pain — 호러 어드벤처
    ("coh_indie_platformer", 512900): (2, ""),         # Streets of Rogue — 로그라이트 액션
    ("coh_indie_platformer", 263980): (3, ""),         # Out There Somewhere — Cave Story/Metroid 영감
    ("coh_indie_platformer", 284460): (2, ""),         # DeadCore — 플랫포머-FPS 스피드런
    ("coh_indie_platformer", 347800): (3, ""),         # Ghost Song — 메트로배니아
    ("coh_indie_platformer", 220440): (1, "GENRE_ONLY"),   # DmC — 3D 스타일리시 액션
}

ALL = {**REP_V2, **POSTPROCESS, **DEPTH_V1}
