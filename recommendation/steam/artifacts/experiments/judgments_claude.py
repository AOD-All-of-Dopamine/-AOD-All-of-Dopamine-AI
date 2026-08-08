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

# --- full_v1 (코퍼스 19,476 → 173,691) 에서 새로 진입한 후보 ---
# 8개 프로필 × Top-10 = 80칸 중 35칸은 구 코퍼스 판정을 재사용하고, 나머지 45칸이 신규다.
FULL_V1 = {
    # P01 mix_vehicle_fps: Euro Truck Simulator 2 / Assetto Corsa / Left 4 Dead 2
    ("mix_vehicle_fps", 1066130): (2, ""),                   # Truck Life — 트럭 심, 장르 정확하나 저가 양산품
    ("mix_vehicle_fps", 3917090): (3, ""),                   # Assetto Corsa Rally — 시드의 공식 신작 랠리 심
    ("mix_vehicle_fps", 2268560): (1, "GENRE_ONLY"),         # Zombie Survival Game Online — 좀비지만 오픈월드 루터슈터
    ("mix_vehicle_fps", 1763830): (1, "TOO_NICHE"),          # My Truck Game — rev 239, 사실상 기술 데모

    # P02 coh_classic_multi: Garry's Mod / Team Fortress 2 / Left 4 Dead 2
    ("coh_classic_multi", 2176320): (0, "IRRELEVANT"),       # Magical Drop VI — 낙하 퍼즐
    ("coh_classic_multi", 414120):  (2, "TOO_NICHE"),        # Modbox — GMod 대응 샌드박스지만 VR 중심 rev 198
    ("coh_classic_multi", 2268560): (1, "GENRE_ONLY"),       # Zombie Survival Game Online
    ("coh_classic_multi", 2027330): (3, ""),                 # GoreBox — 물리 샌드박스, GMod 직결 rev 22k

    # P03 mix_openworld_cozy: No Man's Sky / Subnautica / Slime Rancher
    ("mix_openworld_cozy", 1657630): (3, ""),                # Slime Rancher 2 — 공식 후속작
    ("mix_openworld_cozy", 890720):  (3, ""),                # In Other Waters — 외계 해양 탐사, Subnautica 정서 직결
    ("mix_openworld_cozy", 1552500): (1, "KEYWORD_MATCH"),   # Slimekeep — '슬라임'만 겹치는 로그라이크
    ("mix_openworld_cozy", 1031460): (2, ""),                # Sail Forth — 아늑한 항해 탐험
    ("mix_openworld_cozy", 2800450): (2, ""),                # Planetaries — SF 오픈월드 서바이벌(+TD)
    ("mix_openworld_cozy", 2642840): (0, "IRRELEVANT"),      # Humans are not that against Lizardwomen 2
    ("mix_openworld_cozy", 358920):  (2, ""),                # Star Control I and II — NMS 의 정신적 선조

    # P04 coh_cozy: Stardew Valley / Slime Rancher / Raft
    ("coh_cozy", 1657630): (3, ""),                          # Slime Rancher 2
    ("coh_cozy", 1955340): (1, "KEYWORD_MATCH"),             # Super Raft Boat Together — 이름만 Raft, 로그라이트 슈터
    ("coh_cozy", 1552500): (1, "KEYWORD_MATCH"),             # Slimekeep
    ("coh_cozy", 2659960): (0, "LOW_QUALITY"),               # Let's Go! My Harem Farm — 농사 어휘를 쓴 하렘물
    ("coh_cozy", 1999170): (0, "IRRELEVANT"),                # 虫潮 — 벌레 떼 슈터, 코지의 정반대
    ("coh_cozy", 1203180): (2, ""),                          # Breakwaters — 해양 서바이벌 크래프팅, Raft 대응
    ("coh_cozy", 406870):  (1, "GENRE_ONLY"),                # Eventide: Slavic Fable — 히든오브젝트

    # P05 mix_rpg_racing: The Witcher 3 / Skyrim / BeamNG.drive
    ("mix_rpg_racing", 1763830): (1, "TOO_NICHE"),           # My Truck Game
    ("mix_rpg_racing", 2437570): (1, "TOO_NICHE"),           # Golden Chambers — 서부 오픈월드 RPG, rev 188
    ("mix_rpg_racing", 3933710): (1, "TOO_NICHE"),           # My Car My Life — 드래그 레이싱, rev 153
    ("mix_rpg_racing", 1321230): (0, "IRRELEVANT"),          # Angels of Death Episode.Eddie — 애니 단편
    ("mix_rpg_racing", 1565890): (2, ""),                    # RaceLeague — soft-body damage, BeamNG 특징 일치
    ("mix_rpg_racing", 2067920): (1, "GENRE_ONLY"),          # Rogue : Genesia — 뱀서라이크
    ("mix_rpg_racing", 1214520): (0, "KEYWORD_MATCH"),       # Pro Gymnast Simulator — '물리 시뮬' 어휘만

    # P06 mix_fps_cozy: Counter-Strike 2 / PUBG / Stardew Valley
    ("mix_fps_cozy", 805940):  (1, "LOW_QUALITY"),           # RUSSIA BATTLEGROUNDS — 장르는 맞고 품질이 없다
    ("mix_fps_cozy", 2659960): (0, "LOW_QUALITY"),           # Let's Go! My Harem Farm
    ("mix_fps_cozy", 3028330): (3, ""),                      # Battlefield REDSEC — 무료 배틀로얄, PUBG 직결
    ("mix_fps_cozy", 1245560): (3, ""),                      # Roots of Pacha — Stardew 계열 최상급 대응

    # P07 coh_indie_platformer: Hollow Knight / Dead Cells / Celeste
    ("coh_indie_platformer", 1911610): (3, ""),              # Windblown — Dead Cells 제작사 신작
    ("coh_indie_platformer", 1030300): (3, ""),              # Hollow Knight: Silksong — 공식 후속작 rev 381k
    ("coh_indie_platformer", 2803280): (2, ""),              # Dragon Is Dead — 액션 플랫포머 로그라이트
    ("coh_indie_platformer", 1634860): (2, ""),              # Minishoot' Adventures — 개방형 탐험, 전투는 트윈스틱
    ("coh_indie_platformer", 2157210): (1, "TOO_NICHE"),     # Dead of Darkness — 2D 서바이벌 호러, rev 198
    ("coh_indie_platformer", 1589570): (2, ""),              # Dunjungle — 2D 액션 로그라이트
    ("coh_indie_platformer", 2071280): (3, ""),              # Ravenswatch — 로그라이크 액션 rev 16.6k

    # P08 mix_cozy_fps: Raft / Stardew Valley / Counter-Strike 2
    ("mix_cozy_fps", 1955340): (1, "KEYWORD_MATCH"),         # Super Raft Boat Together
    ("mix_cozy_fps", 2659960): (0, "LOW_QUALITY"),           # Let's Go! My Harem Farm
    ("mix_cozy_fps", 1203180): (2, ""),                      # Breakwaters
    ("mix_cozy_fps", 351230):  (1, "KEYWORD_MATCH"),         # Counter Spell — 'Counter' 만 겹침
    ("mix_cozy_fps", 1245560): (3, ""),                      # Roots of Pacha
}

# --- 구 코퍼스(rep_v2) 보충 판정 ---
# 퍼블리셔 상한이 DEPTH_V1 판정 이후에 들어가면서 구 코퍼스 Top-10 에도 새 항목이 14칸 생겼다.
# 이걸 비워두면 구/신 비교에서 구 쪽만 판정된 항목으로 평균을 내게 되어 비교가 성립하지 않는다.
REP_V2_TOPUP = {
    ("mix_vehicle_fps", 759740): (2, ""),                    # RIDE 3 — 오토바이 레이싱 심, Assetto 축
    ("mix_vehicle_fps", 618140): (1, "GENRE_ONLY"),          # Barro — 캐주얼 미니 레이싱, 시뮬 정반대
    ("mix_vehicle_fps", 561600): (2, ""),                    # MXGP3 — 공식 모토크로스 심
    ("coh_classic_multi", 758990): (2, ""),                  # Ancient Warfare 3 — 샌드박스 + 다중 모드
    ("mix_openworld_cozy", 207490): (0, "IRRELEVANT"),       # Rayman Origins — 2D 플랫포머
    ("mix_openworld_cozy", 392110): (1, "KEYWORD_MATCH"),    # Endless Space 2 — '우주'만 겹치는 4X
    ("mix_openworld_cozy", 432940): (0, "IRRELEVANT"),       # Breakneck — 스피더 회피 액션
    ("mix_openworld_cozy", 280790): (2, ""),                 # Creativerse — 샌드박스 크래프팅
    ("mix_openworld_cozy", 584400): (0, "IRRELEVANT"),       # 소닉 매니아
    ("coh_cozy", 271260): (0, "IRRELEVANT"),                 # Star Control: Origins — 우주 어드벤처
    ("coh_cozy", 207490): (0, "IRRELEVANT"),                 # Rayman Origins
    ("mix_rpg_racing", 637650): (3, ""),                     # FINAL FANTASY XV — 대형 오픈월드 액션 RPG
    ("mix_rpg_racing", 234650): (2, ""),                     # Shadowrun Returns — 턴제 전술 RPG
    ("mix_cozy_fps", 70): (2, ""),                           # Half-Life — CS 의 직계 조상 (P02 판정과 동일)
}

# --- full_v1 + 리뷰 수 하한 300 에서 새로 올라온 후보 ---
MIN_REV_300 = {
    ("coh_classic_multi", 1238820): (2, ""),                 # Battlefield 3 — 팀 기반 멀티 FPS, TF2 축
    ("coh_cozy", 1245560): (3, ""),                          # Roots of Pacha — Stardew 계열 최상급
    ("coh_cozy", 876650):  (2, ""),                          # Aground — 무인도 표류 크래프팅, Raft 대응
    ("coh_cozy", 2642840): (0, "IRRELEVANT"),                # Humans are not that against Lizardwomen 2
    ("coh_indie_platformer", 1662480): (2, ""),              # Nuclear Blaze — Dead Cells 제작자의 2D 플랫포머
    ("mix_cozy_fps", 1250):    (3, ""),                      # Killing Floor — 협동 FPS, CS2 축
    ("mix_cozy_fps", 876650):  (2, ""),                      # Aground
    ("mix_cozy_fps", 1248130): (2, ""),                      # Farming Simulator 22 — 농사지만 톤이 다름
    ("mix_fps_cozy", 895400):  (3, ""),                      # Deadside — 오픈월드 생존 슈터, PUBG 대응
    ("mix_fps_cozy", 1248130): (2, ""),                      # Farming Simulator 22
    ("mix_openworld_cozy", 2354000): (0, "IRRELEVANT"),      # Slackers — 쇼핑카트 파티 레이싱
    ("mix_openworld_cozy", 1933840): (1, "GENRE_ONLY"),      # Moon Mystery — 우주 FPS 어드벤처
    ("mix_rpg_racing", 1764530): (2, ""),                    # Sailwind — 사실적 항해 시뮬 + 오픈월드
    ("mix_rpg_racing", 1020470): (1, "GENRE_ONLY"),          # Evoland — RPG 어휘만
    ("mix_rpg_racing", 2393160): (1, "GENRE_ONLY"),          # Nice Day for Fishing — 낚시 RPG
    ("mix_vehicle_fps", 2051120): (1, "GENRE_ONLY"),         # HOT WHEELS UNLEASHED 2 — 아케이드, 시뮬 정반대
    ("mix_vehicle_fps", 1032700): (2, ""),                   # Bus Driving Sim 22 — ETS2 대응
    # next_page 는 후보 풀이 1,500(=top_n×5)이라 3,000 으로 잰 표와 3칸이 다르게 채워진다
    ("coh_cozy", 1248130):     (2, ""),                      # Farming Simulator 22
    ("mix_vehicle_fps", 55040):  (1, "GENRE_ONLY"),          # Atom Zombie Smasher — 탑다운 좀비 전략
    ("mix_vehicle_fps", 849100): (3, ""),                    # Alaskan Road Truckers — ETS2 직결
}

# --- P05 악화 원인 진단 중 후보로 올라온 것들 ---
# 백분위 계산을 '리뷰 있는 게임' 안에서만 하도록 바꾸거나 퍼블리셔 상한을 4로 올렸을 때 진입한다.
RANKER_DIAG = {
    ("mix_vehicle_fps", 699130):   (3, ""),                  # World War Z — 4인 협동 좀비 슈터, L4D2 직결
    ("coh_classic_multi", 699130): (3, ""),                  # World War Z
    ("mix_openworld_cozy", 1104280): (0, "IRRELEVANT"),      # The Slormancer — 핵앤슬래시 ARPG
    ("mix_openworld_cozy", 2410170): (0, "IRRELEVANT"),      # 더 네임리스 — 게임북 턴제 RPG
    ("mix_openworld_cozy", 2515020): (0, "IRRELEVANT"),      # FINAL FANTASY XVI — 탐험/생존과 무관
    ("coh_cozy", 2354000):     (0, "IRRELEVANT"),            # Slackers — 쇼핑카트 파티 레이싱
    ("mix_rpg_racing", 1295510): (2, ""),                    # 드래곤 퀘스트 XI S — 대작 RPG, 다만 턴제 JRPG
}

# --- 백분위 수정 + 인기도 부스트 상향 시 진입하는 후보 ---
BOOST_SWEEP = {
    ("coh_classic_multi", 444090): (3, ""),                  # Paladins — 5v5 팀 히어로 슈터, TF2 직결
    ("coh_cozy", 782330):  (0, "IRRELEVANT"),                # DOOM Eternal — 코지 프로필에 하드코어 FPS
    ("coh_cozy", 2515020): (0, "IRRELEVANT"),                # FINAL FANTASY XVI
    ("coh_indie_platformer", 2218750): (1, "GENRE_ONLY"),    # Halls of Torment — 탑다운 서바이버류
    ("mix_cozy_fps", 222880): (3, ""),                       # Insurgency — 하드코어 멀티 FPS, CS2 축
    ("mix_cozy_fps", 826630): (0, "IRRELEVANT"),             # Iron Harvest — RTS
    ("mix_fps_cozy", 826630): (0, "IRRELEVANT"),             # Iron Harvest
    ("mix_openworld_cozy", 8500):    (1, "MODE_MISMATCH"),   # 이브 온라인 — 우주지만 하드코어 MMO
    ("mix_openworld_cozy", 1456650): (0, "IRRELEVANT"),      # 파스칼 웨이저 — 소울라이크
    ("mix_openworld_cozy", 3489700): (0, "IRRELEVANT"),      # 스텔라 블레이드
    ("mix_rpg_racing", 973760):  (2, ""),                    # Thronebreaker — 위쳐 공식작, 다만 카드 전투
    ("mix_rpg_racing", 1369670): (2, ""),                    # Motor Town — 오픈월드 드라이빙 시뮬
    ("mix_rpg_racing", 2531310): (2, ""),                    # The Last of Us Part II — 서사 대작, RPG 아님
    ("mix_vehicle_fps", 495420):  (2, ""),                   # State of Decay 2 — 좀비 생존, 협동슈터 아님
    ("mix_vehicle_fps", 805550):  (3, ""),                   # 아세토 코르사 컴페티치오네 — 시드의 공식 후속작
    ("mix_vehicle_fps", 1070580): (1, "GENRE_ONLY"),         # Drift86 — 아케이드 드리프트
    ("mix_vehicle_fps", 1239690): (1, "GENRE_ONLY"),         # Retrowave — 신스웨이브 아케이드
}

# --- val split 8개 프로필 × Top-10 = 80쌍 (전부 신규) ---
# dev 12개만 판정된 채로 min_reviews·인터리빙·rec_boost 를 전부 골랐다. 홀드아웃 최초 측정이다.
VAL_V1 = {
    # coh_grand_strategy: Hearts of Iron IV / Europa Universalis IV / Civilization VI
    ("coh_grand_strategy", 25890):   (1, "FRANCHISE_OR_VARIANT"),  # Hearts of Iron III — 전작
    ("coh_grand_strategy", 203770):  (3, ""),                 # Crusader Kings II — Paradox 대전략 직결
    ("coh_grand_strategy", 758990):  (1, "GENRE_ONLY"),       # Ancient Warfare 3 — 샌드박스 전투 시뮬
    ("coh_grand_strategy", 1840800): (2, ""),                 # Headquarters: WWII — 턴제 WW2
    ("coh_grand_strategy", 1183470): (2, ""),                 # Imperiums: Greek Wars — 턴제 4X
    ("coh_grand_strategy", 508600):  (0, "IRRELEVANT"),       # ROD: Revolt Of Defense — 우주 기지 방어
    ("coh_grand_strategy", 312450):  (2, ""),                 # Order of Battle: WWII — 워게임
    ("coh_grand_strategy", 2340500): (2, ""),                 # 창조 삼국지 — 역사 시뮬 + RTS
    ("coh_grand_strategy", 98200):   (1, "GENRE_ONLY"),       # Frozen Synapse — 소대 전술 퍼즐
    ("coh_grand_strategy", 593030):  (2, ""),                 # Strategic Command WWII — 턴제 대전략

    # coh_openworld_survival: Subnautica / The Long Dark / No Man's Sky
    ("coh_openworld_survival", 890720):  (3, ""),             # In Other Waters — 외계 해양 탐사
    ("coh_openworld_survival", 2336440): (2, ""),             # 침묵의 땅 — 종말 세계 탐험 생존
    ("coh_openworld_survival", 214730):  (2, ""),             # Space Rangers HD — 우주 오픈월드
    ("coh_openworld_survival", 1031460): (2, ""),             # Sail Forth — 항해 탐험
    ("coh_openworld_survival", 219740):  (3, ""),             # Don't Starve — 생존 대표작
    ("coh_openworld_survival", 392110):  (1, "KEYWORD_MATCH"),# ENDLESS Space 2 — '우주'만 겹치는 4X
    ("coh_openworld_survival", 664830):  (0, "IRRELEVANT"),   # Zombotron — 액션 플랫포머
    ("coh_openworld_survival", 257050):  (2, ""),             # Darkout — 외계 절차생성 생존 크래프팅
    ("coh_openworld_survival", 1933840): (1, "GENRE_ONLY"),   # Moon Mystery — 우주 FPS
    ("coh_openworld_survival", 609320):  (2, ""),             # FAR: Lone Sails — 종말 여정

    # coh_survival_craft: Project Zomboid / Don't Starve / The Forest
    ("coh_survival_craft", 2796180): (2, ""),                 # SILENT BREATH — 호러 생존
    ("coh_survival_craft", 897730):  (3, ""),                 # Among Trees — 자연 샌드박스 크래프팅
    ("coh_survival_craft", 273500):  (1, "GENRE_ONLY"),       # Over 9000 Zombies! — 탑다운 아케이드
    ("coh_survival_craft", 440730):  (1, "GENRE_ONLY"),       # Survival Zombies — 좀비 FPS
    ("coh_survival_craft", 2408920): (2, ""),                 # 森林之子 — 야생 생존 채집
    ("coh_survival_craft", 495420):  (3, ""),                 # State of Decay 2 — Zomboid 축 직결
    ("coh_survival_craft", 313120):  (3, ""),                 # Stranded Deep — 무인도 생존
    ("coh_survival_craft", 347940):  (2, ""),                 # Forsaken Isle — 무인도 제작/건설
    ("coh_survival_craft", 685340):  (2, ""),                 # Delivery from the Pain — 좀비 생존 RPG
    ("coh_survival_craft", 877010):  (2, ""),                 # Beyond Contact — SF 생존

    # coh_vehicle_sim: Euro Truck Simulator 2 / BeamNG.drive / Assetto Corsa
    ("coh_vehicle_sim", 1066130): (2, ""),                    # Truck Life
    ("coh_vehicle_sim", 491280):  (2, ""),                    # Drift Horizon Online
    ("coh_vehicle_sim", 3917090): (3, ""),                    # 아세토 코르사 랠리 — 공식 신작
    ("coh_vehicle_sim", 270880):  (3, ""),                    # American Truck Simulator — ETS2 자매작
    ("coh_vehicle_sim", 1764530): (2, ""),                    # Sailwind — 사실적 항해 시뮬
    ("coh_vehicle_sim", 356430):  (1, "GENRE_ONLY"),          # Chris Sawyer's Locomotion — 운송 경영
    ("coh_vehicle_sim", 415600):  (2, ""),                    # Kart Racing Pro
    ("coh_vehicle_sim", 1032700): (2, ""),                    # Bus Driving Sim 22
    ("coh_vehicle_sim", 488550):  (2, ""),                    # Dream Car Builder
    ("coh_vehicle_sim", 849100):  (3, ""),                    # Alaskan Road Truckers

    # mix_grand_casual: Civilization VI / Europa Universalis IV / Human Fall Flat
    ("mix_grand_casual", 203770):  (3, ""),                   # Crusader Kings II
    ("mix_grand_casual", 758990):  (1, "GENRE_ONLY"),         # Ancient Warfare 3
    ("mix_grand_casual", 1608700): (1, "GENRE_ONLY"),         # Soundfall — 리듬 루트슈터
    ("mix_grand_casual", 1183470): (2, ""),                   # Imperiums: Greek Wars
    ("mix_grand_casual", 508600):  (0, "IRRELEVANT"),         # ROD: Revolt Of Defense
    ("mix_grand_casual", 4082750): (2, ""),                   # Log Riders — 2인 협동 물리 플랫폼
    ("mix_grand_casual", 2340500): (2, ""),                   # 창조 삼국지
    ("mix_grand_casual", 98200):   (1, "GENRE_ONLY"),         # Frozen Synapse
    ("mix_grand_casual", 2567870): (3, ""),                   # Chained Together — Human Fall Flat 축 직결
    ("mix_grand_casual", 1768280): (2, ""),                   # Ozymandias — 간소화 4X

    # mix_indie_multi: Hollow Knight / Dead Cells / Garry's Mod
    ("mix_indie_multi", 590830):  (3, ""),                    # s&box — GMod 정신적 후속작
    ("mix_indie_multi", 1911610): (3, ""),                    # Windblown
    ("mix_indie_multi", 1030300): (3, ""),                    # Hollow Knight: Silksong
    ("mix_indie_multi", 696370):  (2, ""),                    # BROKE PROTOCOL
    ("mix_indie_multi", 2803280): (2, ""),                    # Dragon Is Dead
    ("mix_indie_multi", 1634860): (2, ""),                    # Minishoot' Adventures
    ("mix_indie_multi", 2027330): (3, ""),                    # GoreBox
    ("mix_indie_multi", 2071280): (3, ""),                    # Ravenswatch
    ("mix_indie_multi", 1589570): (2, ""),                    # Dunjungle
    ("mix_indie_multi", 1284210): (0, "IRRELEVANT"),          # Guild Wars 2 — MMORPG

    # mix_multi_indie: Team Fortress 2 / Left 4 Dead 2 / Terraria
    ("mix_multi_indie", 655780):  (2, ""),                    # Project 5: Sightseer — 오픈월드 샌드박스
    ("mix_multi_indie", 1250):    (3, ""),                    # Killing Floor — 협동 FPS
    ("mix_multi_indie", 70):      (2, ""),                    # Half-Life
    ("mix_multi_indie", 587520):  (1, "GENRE_ONLY"),          # Dungeons of Sundaria — 던전 크롤
    ("mix_multi_indie", 500):     (1, "FRANCHISE_OR_VARIANT"),# Left 4 Dead — 전작
    ("mix_multi_indie", 1238820): (2, ""),                    # Battlefield 3
    ("mix_multi_indie", 538100):  (2, ""),                    # Feel The Snow — Terraria류 2D 샌드박스
    ("mix_multi_indie", 281920):  (1, "GENRE_ONLY"),          # Splatter — 탑다운 슈터
    ("mix_multi_indie", 17710):   (2, ""),                    # Nuclear Dawn — FPS/RTS 멀티
    ("mix_multi_indie", 257850):  (2, ""),                    # Hyper Light Drifter

    # mix_survival_strategy: The Forest / 7 Days to Die / Civilization VI
    ("mix_survival_strategy", 2796180): (2, ""),              # SILENT BREATH
    ("mix_survival_strategy", 1494140): (0, "KEYWORD_MATCH"), # 세븐데이즈 오리진 — '7일'만 겹침
    ("mix_survival_strategy", 2772750): (3, ""),              # Age of History 3 — 대전략
    ("mix_survival_strategy", 440730):  (1, "GENRE_ONLY"),    # Survival Zombies
    ("mix_survival_strategy", 3946810): (0, "KEYWORD_MATCH"), # 我与你共度的七日 — '7일'만 겹침
    ("mix_survival_strategy", 1295660): (3, ""),              # 문명 VII — 시드의 후속작
    ("mix_survival_strategy", 313120):  (3, ""),              # Stranded Deep
    ("mix_survival_strategy", 1377380): (3, ""),              # Night of the Dead — 7DtD 축 직결
    ("mix_survival_strategy", 877010):  (2, ""),              # Beyond Contact
    ("mix_survival_strategy", 454350):  (0, "IRRELEVANT"),    # Days of War — WW2 멀티 FPS
}

ALL = {**REP_V2, **POSTPROCESS, **DEPTH_V1, **FULL_V1, **REP_V2_TOPUP, **MIN_REV_300,
       **RANKER_DIAG, **BOOST_SWEEP, **VAL_V1}
