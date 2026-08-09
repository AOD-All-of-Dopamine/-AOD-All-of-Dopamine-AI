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

# --- 라운드1 블라인드 판정 (src/blind_judge.py) ---
# 비교할 5개 설정(하한 0/300/1000 · 인터리빙 on/off · rec_boost 0.15/0.03)의 Top-10 **합집합** 193쌍.
# 한 설정만 판정하면 그 설정이 자동으로 이긴다 — 실측으로 기준선 미판정 3칸 vs 대안 55~115칸이었고
# 미판정을 성공으로 세면 순위가 뒤집혔다. 합집합을 채워야 비교가 성립한다.
#
# **판정 근거가 이전과 다르다.** 이 라운드는 설정/순위/전략/프로필 실제 id 와 함께
# **리뷰 수도 가린 채** 채점했다. 리뷰 수는 min_reviews 실험의 조작 변수라 보이면 어느 설정에서
# 왔는지 추론된다. 그래서 TOO_NICHE/LOW_QUALITY 태그는 설명과 이름에서 읽히는 범위로만 붙었다.
BLIND_ROUND1 = {
    ("coh_arpg", 270550): (0, "IRRELEVANT"),  # Yet Another Zombie Defense
    ("coh_arpg", 440540): (1, "GENRE_ONLY"),  # Ara Fell: Enhanced Edition
    ("coh_arpg", 570940): (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS™: REMASTERED
    ("coh_arpg", 1245620): (3, ""),  # ELDEN RING
    ("coh_arpg", 2437570): (1, "TOO_NICHE"),  # Golden Chambers
    ("coh_classic_multi", 55040): (1, "GENRE_ONLY"),  # Atom Zombie Smasher
    ("coh_classic_multi", 1801520): (1, "GENRE_ONLY"),  # Zombieville USA 3D
    ("coh_cozy", 1274490): (2, ""),  # Green Project
    ("coh_cozy", 2075580): (2, ""),  # Under A New Sun
    ("coh_cozy", 2846060): (2, ""),  # Subterror
    ("coh_cozy", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("coh_fps", 308600): (1, "GENRE_ONLY"),  # Skillshot City
    ("coh_fps", 351230): (1, "KEYWORD_MATCH"),  # Counter Spell
    ("coh_fps", 390560): (0, "MODE_MISMATCH"),  # Fantasy Strike
    ("coh_fps", 1121710): (2, ""),  # Total Lockdown
    ("coh_fps", 1284210): (0, "IRRELEVANT"),  # Guild Wars 2®
    ("coh_grand_strategy", 214730): (2, ""),  # Space Rangers HD: A War Apart
    ("coh_grand_strategy", 314980): (3, ""),  # Supreme Ruler Ultimate
    ("coh_grand_strategy", 462940): (3, ""),  # Making History: The Second World War
    ("coh_grand_strategy", 589290): (1, "MODE_MISMATCH"),  # Holdfast: Nations At War
    ("coh_grand_strategy", 965320): (2, ""),  # The Settlers® 7 : History Edition
    ("coh_grand_strategy", 1201700): (3, ""),  # Warnament
    ("coh_grand_strategy", 1611600): (3, ""),  # WARNO
    ("coh_grand_strategy", 1768280): (2, ""),  # Ozymandias: Bronze Age Empire Sim
    ("coh_grand_strategy", 1844380): (2, ""),  # Warhammer Age of Sigmar: Realms of Ruin
    ("coh_grand_strategy", 2154730): (0, "IRRELEVANT"),  # Building Destruction
    ("coh_grand_strategy", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("coh_indie_platformer", 1046400): (3, ""),  # Astalon: 지구의 눈물
    ("coh_indie_platformer", 1200770): (0, "IRRELEVANT"),  # Deathground
    ("coh_indie_platformer", 1375900): (2, ""),  # 혈색 광맥
    ("coh_openworld_survival", 280790): (2, ""),  # Creativerse
    ("coh_openworld_survival", 331870): (2, ""),  # AER Memories of Old
    ("coh_openworld_survival", 358920): (2, ""),  # Star Control I and II
    ("coh_openworld_survival", 366870): (3, ""),  # Narcosis
    ("coh_openworld_survival", 401360): (3, ""),  # The Aquatic Adventure of the Last Human
    ("coh_openworld_survival", 655780): (2, ""),  # Project 5: Sightseer
    ("coh_openworld_survival", 1645630): (1, "GENRE_ONLY"),  # FEROCIOUS
    ("coh_openworld_survival", 2060790): (2, ""),  # 하운티
    ("coh_openworld_survival", 2567870): (0, "IRRELEVANT"),  # Chained Together
    ("coh_openworld_survival", 2658470): (0, "IRRELEVANT"),  # Is this Game Trying to Kill Me?
    ("coh_openworld_survival", 2800450): (2, ""),  # Planetaries
    ("coh_openworld_survival", 2941710): (1, "TOO_NICHE"),  # Project Silverfish
    ("coh_openworld_survival", 3151400): (1, "GENRE_ONLY"),  # Liminal Universe
    ("coh_strategy", 226860): (3, ""),  # Galactic Civilizations III
    ("coh_strategy", 280720): (3, ""),  # Imagine Earth
    ("coh_strategy", 282210): (3, ""),  # Sid Meier's Starships
    ("coh_strategy", 366910): (2, ""),  # The Long Journey Home
    ("coh_strategy", 418910): (1, "GENRE_ONLY"),  # Idle Civilization
    ("coh_strategy", 736820): (3, ""),  # Knights of Honor II: Sovereign
    ("coh_strategy", 835570): (1, "GENRE_ONLY"),  # 컨커러스 블레이드
    ("coh_strategy", 2134770): (2, ""),  # SteamWorld Build
    ("coh_survival_craft", 55040): (1, "GENRE_ONLY"),  # Atom Zombie Smasher
    ("coh_survival_craft", 263920): (1, "LOW_QUALITY"),  # Zombie Grinder
    ("coh_survival_craft", 376210): (2, ""),  # The Isle
    ("coh_survival_craft", 408960): (1, "GENRE_ONLY"),  # Zombasite
    ("coh_survival_craft", 876650): (2, ""),  # Aground
    ("coh_survival_craft", 1137490): (2, ""),  # Invasion 2037
    ("coh_survival_craft", 1274490): (2, ""),  # Green Project
    ("coh_survival_craft", 1326470): (3, ""),  # Sons Of The Forest
    ("coh_survival_craft", 1668310): (1, "GENRE_ONLY"),  # 24 Killers
    ("coh_survival_craft", 2268560): (2, ""),  # Zombie Survival Game Online
    ("coh_vehicle_sim", 41740): (2, ""),  # Cargo! The Quest for Gravity
    ("coh_vehicle_sim", 46370): (2, ""),  # Rig n Roll
    ("coh_vehicle_sim", 232010): (1, "FRANCHISE_OR_VARIANT"),  # Euro Truck Simulator
    ("coh_vehicle_sim", 302060): (2, ""),  # Trucks & Trailers
    ("coh_vehicle_sim", 446020): (3, ""),  # Jalopy
    ("coh_vehicle_sim", 493490): (2, ""),  # City Car Driving
    ("coh_vehicle_sim", 768180): (3, ""),  # Truck Driver
    ("coh_vehicle_sim", 1214520): (0, "KEYWORD_MATCH"),  # Pro Gymnast Simulator
    ("coh_vehicle_sim", 1369670): (3, ""),  # Motor Town: Behind The Wheel
    ("coh_vehicle_sim", 1565890): (2, ""),  # RaceLeague
    ("coh_vehicle_sim", 1763830): (1, "TOO_NICHE"),  # My Truck Game
    ("coh_vehicle_sim", 2303180): (1, "GENRE_ONLY"),  # Resoraki: 레이싱
    ("coh_vehicle_sim", 3933710): (1, "TOO_NICHE"),  # My Car My Life
    ("mix_arpg_survival", 382310): (1, "GENRE_ONLY"),  # Eco
    ("mix_arpg_survival", 924140): (1, "LOW_QUALITY"),  # Hand Simulator: Survival
    ("mix_arpg_survival", 1326470): (3, ""),  # Sons Of The Forest
    ("mix_arpg_survival", 1755080): (2, ""),  # Away From Life
    ("mix_arpg_survival", 3712080): (1, "GENRE_ONLY"),  # Funnel Runners
    ("mix_cozy_fps", 2075580): (2, ""),  # Under A New Sun
    ("mix_cozy_fps", 2846060): (2, ""),  # Subterror
    ("mix_cozy_fps", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("mix_fps_cozy", 1274490): (1, "GENRE_ONLY"),  # Green Project
    ("mix_fps_cozy", 1284210): (0, "IRRELEVANT"),  # Guild Wars 2®
    ("mix_fps_cozy", 1726130): (3, ""),  # Pathless Woods
    ("mix_fps_cozy", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("mix_grand_casual", 42810): (3, ""),  # For The Glory: A Europa Universalis Game
    ("mix_grand_casual", 214730): (2, ""),  # Space Rangers HD: A War Apart
    ("mix_grand_casual", 231330): (0, "IRRELEVANT"),  # Deadfall Adventures
    ("mix_grand_casual", 314980): (2, ""),  # Supreme Ruler Ultimate
    ("mix_grand_casual", 965320): (2, ""),  # The Settlers® 7 : History Edition
    ("mix_grand_casual", 1201700): (3, ""),  # Warnament
    ("mix_grand_casual", 1882580): (1, "GENRE_ONLY"),  # FALL GIRLS
    ("mix_grand_casual", 2154730): (0, "IRRELEVANT"),  # Building Destruction
    ("mix_grand_casual", 2179380): (0, "IRRELEVANT"),  # Sand:box
    ("mix_grand_casual", 3035500): (1, "GENRE_ONLY"),  # Fantasy Map Simulator
    ("mix_grand_casual", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("mix_grand_casual", 4373040): (2, ""),  # But Why?
    ("mix_indie_multi", 94400): (1, "MODE_MISMATCH"),  # Nidhogg
    ("mix_indie_multi", 414120): (2, ""),  # Modbox
    ("mix_indie_multi", 758990): (2, ""),  # Ancient Warfare 3
    ("mix_indie_multi", 1200770): (1, "GENRE_ONLY"),  # Deathground
    ("mix_indie_multi", 1375900): (2, ""),  # 혈색 광맥
    ("mix_indie_multi", 2157210): (1, "TOO_NICHE"),  # Dead of Darkness
    ("mix_multi_indie", 104900): (2, ""),  # ORION: Prelude
    ("mix_multi_indie", 230230): (0, "IRRELEVANT"),  # Divinity: Original Sin (Classic)
    ("mix_multi_indie", 263920): (1, "LOW_QUALITY"),  # Zombie Grinder
    ("mix_multi_indie", 280790): (2, ""),  # Creativerse
    ("mix_multi_indie", 298240): (1, "LOW_QUALITY"),  # War Trigger 3
    ("mix_multi_indie", 610960): (1, "TOO_NICHE"),  # 红石遗迹 - Red Obsidian Remnant
    ("mix_multi_indie", 770100): (0, "IRRELEVANT"),  # One Deck Dungeon
    ("mix_multi_indie", 1815530): (2, ""),  # Dungeon Defenders: Going Rogue
    ("mix_multi_indie", 2176320): (0, "IRRELEVANT"),  # Magical Drop VI
    ("mix_multi_indie", 2268560): (1, "GENRE_ONLY"),  # Zombie Survival Game Online
    ("mix_openworld_cozy", 244770): (2, ""),  # StarMade
    ("mix_openworld_cozy", 655780): (2, ""),  # Project 5: Sightseer
    ("mix_openworld_cozy", 1645630): (1, "GENRE_ONLY"),  # FEROCIOUS
    ("mix_openworld_cozy", 2060790): (2, ""),  # 하운티
    ("mix_rpg_racing", 1020800): (2, ""),  # CAR TUNE: Project
    ("mix_rpg_racing", 3216340): (1, "TOO_NICHE"),  # Tearscape
    ("mix_rpg_racing", 3800340): (0, "KEYWORD_MATCH"),  # ScootX
    ("mix_survival_strategy", 214730): (1, "GENRE_ONLY"),  # Space Rangers HD: A War Apart
    ("mix_survival_strategy", 382310): (2, ""),  # Eco
    ("mix_survival_strategy", 541300): (3, ""),  # Survive the Nights
    ("mix_survival_strategy", 876650): (2, ""),  # Aground
    ("mix_survival_strategy", 924140): (1, "LOW_QUALITY"),  # Hand Simulator: Survival
    ("mix_survival_strategy", 1326470): (3, ""),  # Sons Of The Forest
    ("mix_survival_strategy", 1755080): (2, ""),  # Away From Life
    ("mix_survival_strategy", 3712080): (1, "GENRE_ONLY"),  # Funnel Runners
    ("mix_survival_strategy", 4474950): (2, ""),  # LIFE EFFECT Survival
    ("niche_cozy_casual", 347940): (1, "GENRE_ONLY"),  # Forsaken Isle
    ("niche_cozy_casual", 356250): (3, ""),  # Gathering Sky
    ("niche_cozy_casual", 513720): (0, "IRRELEVANT"),  # Archipelago
    ("niche_cozy_casual", 617670): (1, "GENRE_ONLY"),  # Zup! S
    ("niche_cozy_casual", 688130): (1, "MODE_MISMATCH"),  # Pogostuck: Rage With Your Friends
    ("niche_cozy_casual", 710610): (1, "GENRE_ONLY"),  # Don't Sink
    ("niche_cozy_casual", 931270): (3, ""),  # MicroTown
    ("niche_cozy_casual", 1348920): (2, ""),  # Wind Peaks
    ("niche_cozy_casual", 1722520): (2, ""),  # FIND ALL 2: Middle Ages
    ("niche_cozy_casual", 2093900): (2, ""),  # Island Cities - Jigsaw Puzzle
    ("niche_cozy_casual", 2236070): (2, ""),  # Peaks of Yore
    ("niche_cozy_casual", 2239150): (3, ""),  # Thronefall
    ("niche_cozy_casual", 2880750): (2, ""),  # Leaf's Odyssey
    ("niche_cozy_casual", 3580340): (2, ""),  # Ash & Adam's Existential Treads
    ("niche_puzzle_solo", 230290): (1, "GENRE_ONLY"),  # Universe Sandbox
    ("niche_puzzle_solo", 233450): (2, ""),  # Prison Architect
    ("niche_puzzle_solo", 375820): (3, ""),  # Human Resource Machine
    ("niche_puzzle_solo", 973250): (2, ""),  # Altered
    ("niche_puzzle_solo", 1293540): (2, ""),  # Outerverse
    ("niche_puzzle_solo", 1318690): (3, ""),  # shapez
    ("niche_puzzle_solo", 1388770): (0, "IRRELEVANT"),  # Cruelty Squad
    ("niche_puzzle_solo", 1451720): (1, "GENRE_ONLY"),  # Minesweeper Classy
    ("niche_puzzle_solo", 1536570): (3, ""),  # The Last Alchemist
    ("niche_puzzle_solo", 1596310): (1, "GENRE_ONLY"),  # Crypto Mining Simulator
    ("niche_puzzle_solo", 1876000): (2, ""),  # IFO
    ("niche_puzzle_solo", 2685900): (2, ""),  # Mind Over Magnet
    ("niche_roguelite", 250680): (2, ""),  # BELOW
    ("niche_roguelite", 308600): (1, "GENRE_ONLY"),  # Skillshot City
    ("niche_roguelite", 588650): (3, ""),  # Dead Cells
    ("niche_roguelite", 727510): (1, "TOO_NICHE"),  # Void Memory
    ("niche_roguelite", 787810): (3, ""),  # Rogue Heroes: Ruins of Tasos
    ("niche_roguelite", 994220): (1, "GENRE_ONLY"),  # NEOVERSE
    ("niche_roguelite", 1078420): (0, "IRRELEVANT"),  # Critters for Sale
    ("niche_roguelite", 1343810): (0, "IRRELEVANT"),  # Fallen Hero: Retribution
    ("niche_roguelite", 1356280): (1, "GENRE_ONLY"),  # 냥자의모험
    ("niche_roguelite", 1494810): (3, ""),  # Mortal Sin
    ("niche_roguelite", 2181930): (0, "IRRELEVANT"),  # DR LIVESEY ROM AND DEATH EDITION
    ("niche_roguelite", 2273430): (3, ""),  # BlazBlue Entropy Effect
    ("niche_roguelite", 2687400): (3, ""),  # GODBREAKERS
    ("niche_sim", 244910): (2, ""),  # Homesick
    ("niche_sim", 272600): (3, ""),  # Detective Grimoire
    ("niche_sim", 499910): (2, ""),  # SHINRAI - Broken Beyond Despair
    ("niche_sim", 593200): (2, ""),  # The Adventures of Fei Duanmu 端木斐异闻录
    ("niche_sim", 605230): (3, ""),  # Grey Hack
    ("niche_sim", 641990): (3, ""),  # The Escapists 2
    ("niche_sim", 799600): (3, ""),  # Cosmoteer: Starship Architect & Commande
    ("niche_sim", 965810): (2, ""),  # Kara no Shojo
    ("niche_sim", 1112790): (3, ""),  # Automation Empire
    ("niche_sim", 1546920): (3, ""),  # Overboard!
    ("niche_sim", 1603410): (3, ""),  # Lost Nova
    ("niche_sim", 2277090): (3, ""),  # DAEMON MASQUERADE
    ("niche_sim", 2414110): (3, ""),  # Builderment
    ("niche_sim", 3184990): (3, ""),  # 诡秘推理
    ("niche_sim", 4730180): (3, ""),  # Hacker's Journey
    ("niche_soulslike_solo", 280520): (1, "GENRE_ONLY"),  # Crea
    ("niche_tactics", 949230): (2, ""),  # Cities: Skylines II
    ("niche_tactics", 1176470): (2, ""),  # Terra Invicta 테라 인빅타
    ("niche_tactics", 1465550): (0, "IRRELEVANT"),  # One Lonely Outpost
    ("niche_tactics", 1842690): (2, ""),  # 기차역 스토리 (Station Manager)
    ("niche_tactics", 2272400): (3, ""),  # Station to Station
    ("niche_tactics", 2449450): (2, ""),  # 우주 식민 회사
    ("niche_tactics", 2853590): (2, ""),  # Void War
    ("niche_tactics", 3680900): (2, ""),  # Future War Tactics: SOF vs Alien Invasio
}


# --- 라운드3 블라인드: 태그 임베딩 (src/blind_judge.py) ---
# 비교 4설정(태그없음 / 태그있음 / 태그+트렌드0.05 / 태그+트렌드0.15)의 Top-10 합집합 224쌍.
# 같은 12,844개 게임 위에서 태그 유무만 바꾼다 — 코퍼스 크기 변화가 섞이지 않게.
# 판정자는 태그를 못 본다(시트에 genres 와 설명만 있다). 태그를 보여주면 "태그가 겹치니 좋다"는
# 순환이 생긴다.
BLIND_TAGS = {
    ("coh_arpg", 22330): (1, "FRANCHISE_OR_VARIANT"),  # The Elder Scrolls IV: Oblivion® Game of 
    ("coh_arpg", 335300): (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS™ II: Scholar of the First Sin
    ("coh_arpg", 750130): (2, ""),  # The Sinking City Remastered
    ("coh_arpg", 973760): (2, ""),  # Thronebreaker: The Witcher Tales
    ("coh_arpg", 1620730): (2, ""),  # Hell is Us
    ("coh_arpg", 1771300): (3, ""),  # Kingdom Come: Deliverance II
    ("coh_arpg", 1920490): (3, ""),  # The Outer Worlds: Spacer's Choice Editio
    ("coh_arpg", 3282300): (2, ""),  # Mistfall Hunter
    ("coh_arpg", 3321460): (3, ""),  # 붉은사막
    ("coh_classic_multi", 333930): (3, ""),  # Dirty Bomb®
    ("coh_classic_multi", 383150): (2, ""),  # Dead Island Definitive Edition
    ("coh_classic_multi", 1451480): (3, ""),  # The Greatest Penguin Heist of All Time
    ("coh_classic_multi", 1492070): (3, ""),  # Sker Ritual
    ("coh_classic_multi", 2827230): (2, ""),  # Wild Assault / 兽猎突袭
    ("coh_cozy", 214730): (0, "IRRELEVANT"),  # Space Rangers HD: A War Apart
    ("coh_cozy", 580200): (3, ""),  # Yonder: The Cloud Catcher Chronicles
    ("coh_cozy", 1092590): (0, "IRRELEVANT"),  # 沙雕之路
    ("coh_cozy", 1263240): (0, "IRRELEVANT"),  # Skate Story
    ("coh_cozy", 1963370): (2, ""),  # No One Survived
    ("coh_cozy", 2252680): (3, ""),  # Farlands
    ("coh_cozy", 2340520): (3, ""),  # 세이큐 이야기
    ("coh_cozy", 2418520): (3, ""),  # Farm Together 2
    ("coh_cozy", 2661300): (3, ""),  # Grounded 2
    ("coh_cozy", 3321460): (0, "IRRELEVANT"),  # 붉은사막
    ("coh_fps", 222880): (3, ""),  # Insurgency
    ("coh_fps", 282440): (2, ""),  # Quake Live
    ("coh_fps", 433850): (3, ""),  # Z1 Battle Royale
    ("coh_fps", 729040): (2, ""),  # Borderlands Game of the Year Enhanced
    ("coh_fps", 1962663): (3, ""),  # 콜 오브 듀티®: 워존
    ("coh_fps", 2000950): (3, ""),  # Call of Duty®: Modern Warfare®
    ("coh_grand_strategy", 21970): (3, ""),  # R.U.S.E.™
    ("coh_grand_strategy", 70600): (1, "GENRE_ONLY"),  # Worms Ultimate Mayhem
    ("coh_grand_strategy", 235380): (3, ""),  # Blitzkrieg 3
    ("coh_grand_strategy", 597180): (3, ""),  # Old World 올드 월드
    ("coh_grand_strategy", 603850): (3, ""),  # Age of History II
    ("coh_grand_strategy", 743640): (0, "IRRELEVANT"),  # Achievement Clicker
    ("coh_grand_strategy", 1468720): (1, "GENRE_ONLY"),  # Ultimate Epic Battle Simulator 2
    ("coh_grand_strategy", 1560250): (1, "GENRE_ONLY"),  # Rising Front
    ("coh_grand_strategy", 2168680): (1, "GENRE_ONLY"),  # Nuclear Option
    ("coh_grand_strategy", 3105960): (1, "GENRE_ONLY"),  # Astrobuilder
    ("coh_grand_strategy", 3381680): (3, ""),  # Age of History 2: Definitive Edition
    ("coh_indie_platformer", 40800): (3, ""),  # Super Meat Boy
    ("coh_indie_platformer", 236090): (3, ""),  # Dust: An Elysian Tail
    ("coh_indie_platformer", 916730): (3, ""),  # Gato Roboto
    ("coh_indie_platformer", 2273430): (2, ""),  # BlazBlue Entropy Effect
    ("coh_indie_platformer", 2317640): (2, ""),  # JUMP KING QUEST
    ("coh_indie_platformer", 2665680): (2, ""),  # 바벨탑: 혼돈의 생존자들
    ("coh_openworld_survival", 8500): (1, "MODE_MISMATCH"),  # 이브 온라인
    ("coh_openworld_survival", 211820): (3, ""),  # Starbound
    ("coh_openworld_survival", 274520): (2, ""),  # Darkwood 다크우드
    ("coh_openworld_survival", 313120): (3, ""),  # Stranded Deep
    ("coh_openworld_survival", 359320): (2, ""),  # Elite Dangerous
    ("coh_openworld_survival", 361420): (3, ""),  # ASTRONEER
    ("coh_openworld_survival", 738520): (3, ""),  # Breathedge
    ("coh_openworld_survival", 848450): (3, ""),  # 서브노티카: 빌로우 제로
    ("coh_openworld_survival", 914620): (2, ""),  # Mist Survival
    ("coh_openworld_survival", 1931180): (3, ""),  # Lost Skies
    ("coh_openworld_survival", 3276050): (2, ""),  # SpaceCraft
    ("coh_openworld_survival", 3280350): (2, ""),  # DEATH STRANDING 2: ON THE BEACH
    ("coh_strategy", 1124300): (3, ""),  # 휴먼카인드
    ("coh_strategy", 1295660): (3, ""),  # 시드 마이어의 문명 VII
    ("coh_survival_craft", 322330): (2, ""),  # Don't Starve Together
    ("coh_survival_craft", 346110): (3, ""),  # ARK: Survival Evolved
    ("coh_survival_craft", 360170): (3, ""),  # How to Survive 2
    ("coh_survival_craft", 391730): (3, ""),  # Crashlands
    ("coh_survival_craft", 541300): (3, ""),  # Survive the Nights
    ("coh_survival_craft", 914620): (3, ""),  # Mist Survival
    ("coh_survival_craft", 1766060): (3, ""),  # HumanitZ
    ("coh_survival_craft", 1783560): (2, ""),  # The Last Caretaker
    ("coh_survival_craft", 1963370): (3, ""),  # No One Survived
    ("coh_vehicle_sim", 266410): (3, ""),  # iRacing
    ("coh_vehicle_sim", 287310): (1, "GENRE_ONLY"),  # 리볼트
    ("coh_vehicle_sim", 539720): (1, "GENRE_ONLY"),  # Razortron 2000
    ("coh_vehicle_sim", 635260): (2, ""),  # CarX Drift Racing Online
    ("coh_vehicle_sim", 1351240): (3, ""),  # Taxi Life: A City Driving Simulator
    ("coh_vehicle_sim", 1520370): (3, ""),  # Mon Bazou
    ("coh_vehicle_sim", 1578390): (3, ""),  # My Garage
    ("coh_vehicle_sim", 1849250): (3, ""),  # EA SPORTS™ WRC
    ("coh_vehicle_sim", 3616550): (2, ""),  # Car Dealership Simulator 2
    ("coh_vehicle_sim", 3800340): (0, "KEYWORD_MATCH"),  # ScootX
    ("mix_arpg_survival", 70): (1, "GENRE_ONLY"),  # Half-Life
    ("mix_arpg_survival", 22370): (1, "FRANCHISE_OR_VARIANT"),  # Fallout 3: Game of the Year Edition
    ("mix_arpg_survival", 335300): (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS™ II: Scholar of the First Sin
    ("mix_arpg_survival", 529180): (2, ""),  # Dark and Light
    ("mix_arpg_survival", 637650): (2, ""),  # FINAL FANTASY XV WINDOWS EDITION
    ("mix_arpg_survival", 914620): (2, ""),  # Mist Survival
    ("mix_arpg_survival", 1245620): (3, ""),  # ELDEN RING
    ("mix_arpg_survival", 1783560): (2, ""),  # The Last Caretaker
    ("mix_arpg_survival", 2186680): (2, ""),  # Warhammer 40,000: Rogue Trader
    ("mix_arpg_survival", 3282300): (2, ""),  # Mistfall Hunter
    ("mix_cozy_fps", 666140): (3, ""),  # My Time at Portia
    ("mix_cozy_fps", 768200): (3, ""),  # Smalland: Survive the Wilds
    ("mix_cozy_fps", 1938090): (2, ""),  # 콜 오브 듀티®
    ("mix_cozy_fps", 1963370): (2, ""),  # No One Survived
    ("mix_cozy_fps", 2252680): (3, ""),  # Farlands
    ("mix_cozy_fps", 2340520): (3, ""),  # 세이큐 이야기
    ("mix_cozy_fps", 2418520): (3, ""),  # Farm Together 2
    ("mix_cozy_fps", 2661300): (3, ""),  # Grounded 2
    ("mix_fps_cozy", 42680): (2, ""),  # Call of Duty®: Modern Warfare® 3 (2011)
    ("mix_fps_cozy", 1962663): (3, ""),  # 콜 오브 듀티®: 워존
    ("mix_fps_cozy", 2252680): (3, ""),  # Farlands
    ("mix_fps_cozy", 2340520): (3, ""),  # 세이큐 이야기
    ("mix_fps_cozy", 2418520): (3, ""),  # Farm Together 2
    ("mix_grand_casual", 70600): (1, "GENRE_ONLY"),  # Worms Ultimate Mayhem
    ("mix_grand_casual", 394360): (3, ""),  # Hearts of Iron IV
    ("mix_grand_casual", 597180): (3, ""),  # Old World 올드 월드
    ("mix_grand_casual", 743640): (0, "IRRELEVANT"),  # Achievement Clicker
    ("mix_grand_casual", 1071870): (2, ""),  # Biped
    ("mix_grand_casual", 1468720): (1, "GENRE_ONLY"),  # Ultimate Epic Battle Simulator 2
    ("mix_grand_casual", 1560250): (1, "GENRE_ONLY"),  # Rising Front
    ("mix_grand_casual", 2168680): (1, "GENRE_ONLY"),  # Nuclear Option
    ("mix_grand_casual", 2560240): (2, ""),  # Biped 2
    ("mix_grand_casual", 3105960): (1, "GENRE_ONLY"),  # Astrobuilder
    ("mix_grand_casual", 3263320): (3, ""),  # Carry The Glass
    ("mix_grand_casual", 3381680): (3, ""),  # Age of History 2: Definitive Edition
    ("mix_grand_casual", 3450310): (3, ""),  # Europa Universalis V
    ("mix_indie_multi", 236090): (3, ""),  # Dust: An Elysian Tail
    ("mix_indie_multi", 347800): (3, ""),  # Ghost Song
    ("mix_indie_multi", 552100): (3, ""),  # Brick Rigs
    ("mix_indie_multi", 1451480): (3, ""),  # The Greatest Penguin Heist of All Time
    ("mix_indie_multi", 2218750): (1, "GENRE_ONLY"),  # Halls of Torment
    ("mix_indie_multi", 2273430): (2, ""),  # BlazBlue Entropy Effect
    ("mix_indie_multi", 2665680): (2, ""),  # 바벨탑: 혼돈의 생존자들
    ("mix_multi_indie", 211820): (3, ""),  # Starbound
    ("mix_multi_indie", 333930): (3, ""),  # Dirty Bomb®
    ("mix_multi_indie", 383150): (2, ""),  # Dead Island Definitive Edition
    ("mix_multi_indie", 444090): (3, ""),  # Paladins®
    ("mix_multi_indie", 1169040): (3, ""),  # Necesse: 네세스
    ("mix_multi_indie", 1492070): (3, ""),  # Sker Ritual
    ("mix_multi_indie", 1504570): (2, ""),  # 太荒初境
    ("mix_multi_indie", 2827230): (2, ""),  # Wild Assault / 兽猎突袭
    ("mix_openworld_cozy", 211820): (3, ""),  # Starbound
    ("mix_openworld_cozy", 313120): (2, ""),  # Stranded Deep
    ("mix_openworld_cozy", 359320): (2, ""),  # Elite Dangerous
    ("mix_openworld_cozy", 580200): (3, ""),  # Yonder: The Cloud Catcher Chronicles
    ("mix_openworld_cozy", 674140): (3, ""),  # Bugsnax
    ("mix_openworld_cozy", 738520): (3, ""),  # Breathedge
    ("mix_openworld_cozy", 848450): (3, ""),  # 서브노티카: 빌로우 제로
    ("mix_openworld_cozy", 1092590): (0, "IRRELEVANT"),  # 沙雕之路
    ("mix_openworld_cozy", 1263240): (0, "IRRELEVANT"),  # Skate Story
    ("mix_openworld_cozy", 1931180): (3, ""),  # Lost Skies
    ("mix_openworld_cozy", 3276050): (2, ""),  # SpaceCraft
    ("mix_rpg_racing", 22330): (1, "FRANCHISE_OR_VARIANT"),  # The Elder Scrolls IV: Oblivion® Game of 
    ("mix_rpg_racing", 228280): (2, ""),  # Baldur's Gate: Enhanced Edition
    ("mix_rpg_racing", 635260): (2, ""),  # CarX Drift Racing Online
    ("mix_rpg_racing", 750130): (2, ""),  # The Sinking City Remastered
    ("mix_rpg_racing", 1578390): (3, ""),  # My Garage
    ("mix_rpg_racing", 1620730): (2, ""),  # Hell is Us
    ("mix_rpg_racing", 1771300): (3, ""),  # Kingdom Come: Deliverance II
    ("mix_rpg_racing", 3321460): (3, ""),  # 붉은사막
    ("mix_rpg_racing", 3616550): (2, ""),  # Car Dealership Simulator 2
    ("mix_survival_strategy", 221100): (3, ""),  # DayZ
    ("mix_survival_strategy", 239140): (3, ""),  # Dying Light
    ("mix_survival_strategy", 914620): (3, ""),  # Mist Survival
    ("mix_survival_strategy", 1124300): (3, ""),  # 휴먼카인드
    ("mix_survival_strategy", 1183470): (2, ""),  # Imperiums: Greek Wars
    ("mix_survival_strategy", 1783560): (2, ""),  # The Last Caretaker
    ("mix_survival_strategy", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("mix_vehicle_fps", 383150): (2, ""),  # Dead Island Definitive Edition
    ("mix_vehicle_fps", 1066890): (3, ""),  # Automobilista 2
    ("mix_vehicle_fps", 1369670): (3, ""),  # Motor Town: Behind The Wheel
    ("mix_vehicle_fps", 1492070): (2, ""),  # Sker Ritual
    ("mix_vehicle_fps", 1849250): (3, ""),  # EA SPORTS™ WRC
    ("niche_cozy_casual", 331870): (3, ""),  # AER Memories of Old
    ("niche_cozy_casual", 355630): (2, ""),  # Leo’s Fortune - HD Edition
    ("niche_cozy_casual", 375820): (2, ""),  # Human Resource Machine
    ("niche_cozy_casual", 493200): (3, ""),  # RiME
    ("niche_cozy_casual", 787810): (1, "GENRE_ONLY"),  # Rogue Heroes: Ruins of Tasos
    ("niche_cozy_casual", 1084020): (2, ""),  # TheoTown
    ("niche_cozy_casual", 1148650): (2, ""),  # The Legend of Bum-Bo
    ("niche_cozy_casual", 1730250): (3, ""),  # Pan'orama
    ("niche_cozy_casual", 1740300): (3, ""),  # Smushi Come Home
    ("niche_cozy_casual", 2019810): (3, ""),  # Boxes: Lost Fragments
    ("niche_cozy_casual", 2121980): (2, ""),  # Void Stranger
    ("niche_cozy_casual", 2368930): (3, ""),  # 아일랜더스: 새로운 해안
    ("niche_cozy_casual", 4160210): (3, ""),  # The Artisan of Glimmith
    ("niche_puzzle_solo", 304410): (2, ""),  # Hexcells Infinite
    ("niche_puzzle_solo", 370360): (3, ""),  # TIS-100
    ("niche_puzzle_solo", 574720): (2, ""),  # Little Big Workshop
    ("niche_puzzle_solo", 617670): (1, "GENRE_ONLY"),  # Zup! S
    ("niche_puzzle_solo", 792100): (3, ""),  # 7 Billion Humans
    ("niche_puzzle_solo", 1062160): (3, ""),  # Poly Bridge 2
    ("niche_puzzle_solo", 1444480): (3, ""),  # Turing Complete
    ("niche_puzzle_solo", 1910680): (2, ""),  # Orb of Creation
    ("niche_puzzle_solo", 2162800): (3, ""),  # shapez 2 - Factory
    ("niche_puzzle_solo", 3700980): (1, "GENRE_ONLY"),  # How to Make an Atomic Bomb in Your Garde
    ("niche_puzzle_solo", 3846120): (3, ""),  # MineMogul
    ("niche_roguelite", 753420): (3, ""),  # Dungreed
    ("niche_roguelite", 958520): (3, ""),  # 33 Immortals
    ("niche_roguelite", 1315180): (2, ""),  # Spark in the Dark
    ("niche_roguelite", 1721110): (3, ""),  # Abyssus
    ("niche_roguelite", 1887840): (3, ""),  # Another Crab's Treasure
    ("niche_roguelite", 2071280): (3, ""),  # Ravenswatch
    ("niche_roguelite", 2334730): (3, ""),  # Death Must Die
    ("niche_roguelite", 2351560): (3, ""),  # 아포칼립스 파티
    ("niche_roguelite", 2665680): (2, ""),  # 바벨탑: 혼돈의 생존자들
    ("niche_roguelite", 3228590): (3, ""),  # Deadzone: Rogue
    ("niche_roguelite", 3489700): (1, "GENRE_ONLY"),  # 스텔라 블레이드™
    ("niche_sim", 350640): (3, ""),  # Sherlock Holmes: The Devil's Daughter
    ("niche_sim", 383120): (2, ""),  # Empyrion - Galactic Survival
    ("niche_sim", 574720): (2, ""),  # Little Big Workshop
    ("niche_sim", 1366540): (3, ""),  # Dyson Sphere Program
    ("niche_sim", 1369700): (1, "GENRE_ONLY"),  # Solar Expanse - Space Exploration Manage
    ("niche_sim", 1614550): (3, ""),  # Astro Colony
    ("niche_sim", 1754840): (3, ""),  # Hacker Simulator
    ("niche_sim", 2779120): (3, ""),  # Modulus: Factory Automation
    ("niche_sim", 2797960): (3, ""),  # 은폐된 살인의 진실들 - 하드코어 본격 추리 탐정 게임
    ("niche_sim", 2879840): (3, ""),  # 방 탈출 시뮬레이터2
    ("niche_soulslike_solo", 236090): (3, ""),  # Dust: An Elysian Tail
    ("niche_soulslike_solo", 236430): (3, ""),  # DARK SOULS™ II
    ("niche_soulslike_solo", 385380): (1, "GENRE_ONLY"),  # Planet Centauri
    ("niche_soulslike_solo", 747200): (1, "GENRE_ONLY"),  # Keplerth
    ("niche_soulslike_solo", 1369630): (3, ""),  # ENDER LILIES: Quietus of the Knights
    ("niche_soulslike_solo", 1863430): (2, ""),  # Dragonkin: The Banished
    ("niche_soulslike_solo", 2317640): (1, "GENRE_ONLY"),  # JUMP KING QUEST
    ("niche_soulslike_solo", 3216340): (3, ""),  # Tearscape
    ("niche_soulslike_solo", 3418990): (2, ""),  # 신역 추락
    ("niche_tactics", 496620): (3, ""),  # Monster Slayers
    ("niche_tactics", 637090): (3, ""),  # BATTLETECH
    ("niche_tactics", 673880): (3, ""),  # Warhammer 40,000: Mechanicus
    ("niche_tactics", 861540): (3, ""),  # Dicey Dungeons
    ("niche_tactics", 2432860): (3, ""),  # MENACE 메너스
    ("niche_tactics", 3481020): (3, ""),  # 恶魔牌
    ("niche_tactics", 3709430): (2, ""),  # 마녀: 종말의 여행
}


# --- 라운드4 블라인드: 태그 코퍼스에서 리뷰 하한 스윕 ---
# 코퍼스를 21,883(리뷰 101+)으로 넓히고 하한 0/300/1000 을 비교. 합집합 미판정이 24쌍뿐이었다 —
# 태그가 붙으니 하한을 풀어도 Top-10 이 거의 안 바뀐다는 뜻이다.
BLIND_FLOOR = {
    ("coh_cozy", 674140): (3, ""),  # Bugsnax
    ("coh_grand_strategy", 209670): (1, "GENRE_ONLY"),  # Cortex Command
    ("coh_grand_strategy", 2050680): (1, "GENRE_ONLY"),  # Warbox Sandbox
    ("coh_indie_platformer", 2474430): (3, ""),  # TetherGeist
    ("coh_openworld_survival", 244770): (2, ""),  # StarMade
    ("coh_openworld_survival", 450860): (1, "TOO_NICHE"),  # Andarilho
    ("coh_strategy", 2021880): (3, ""),  # Ara History Untold: Anniversary Edition
    ("coh_survival_craft", 420930): (2, ""),  # CHKN
    ("coh_vehicle_sim", 1089830): (2, ""),  # Monster Energy Supercross - The Official
    ("coh_vehicle_sim", 1114150): (3, ""),  # CarX Street
    ("coh_vehicle_sim", 1364690): (1, "GENRE_ONLY"),  # First Racer
    ("mix_grand_casual", 209670): (1, "GENRE_ONLY"),  # Cortex Command
    ("mix_grand_casual", 2050680): (1, "GENRE_ONLY"),  # Warbox Sandbox
    ("mix_indie_multi", 916730): (3, ""),  # Gato Roboto
    ("mix_openworld_cozy", 314790): (0, "IRRELEVANT"),  # Silence
    ("mix_rpg_racing", 321800): (2, ""),  # Icewind Dale: Enhanced Edition
    ("mix_rpg_racing", 1114150): (3, ""),  # CarX Street
    ("niche_puzzle_solo", 1260520): (3, ""),  # Patrick's Parabox
    ("niche_puzzle_solo", 1577620): (3, ""),  # The Signal State
    ("niche_soulslike_solo", 252030): (3, ""),  # Valdis Story: Abyssal City
    ("niche_soulslike_solo", 1264880): (3, ""),  # Watcher Chronicles
    ("niche_soulslike_solo", 1456650): (3, ""),  # 파스칼 웨이저: 완전판
    ("niche_tactics", 355680): (2, ""),  # Overland
    ("niche_tactics", 981430): (3, ""),  # Gordian Quest
}


# --- 라운드5 블라인드: 저리뷰 축 5개 프로필 (리뷰 850~1,973 시드) ---
# 코퍼스의 70% 를 차지하는데 시드가 하나도 없던 구간. 무명 시드라 시트에 **시드 설명**을
# 함께 넣었다 — 이름만으로는 어떤 게임인지 알 수 없고, 모르는 채로 채점하면
# "설명끼리 닮았나"만 보게 되어 임베딩과 순환한다.
BLIND_LOWREV = {
    ("lowrev_cozy_narrative", 232430): (3, ""),  # Gone Home
    ("lowrev_cozy_narrative", 331870): (2, ""),  # AER Memories of Old
    ("lowrev_cozy_narrative", 520720): (3, ""),  # Dear Esther: Landmark Edition
    ("lowrev_cozy_narrative", 638230): (3, ""),  # Journey
    ("lowrev_cozy_narrative", 695330): (3, ""),  # SEASON: A letter to the future
    ("lowrev_cozy_narrative", 858940): (3, ""),  # Flowers -Le volume sur ete-
    ("lowrev_cozy_narrative", 1281270): (2, ""),  # Fatum Betula
    ("lowrev_cozy_narrative", 1506980): (2, ""),  # 葬花·暗黑桃花源
    ("lowrev_cozy_narrative", 3069120): (2, ""),  # 러브커스: 사랑이 아니면 죽음뿐
    ("lowrev_cozy_narrative", 3528450): (2, ""),  # 종이집
    ("lowrev_deckbuilder", 646570): (3, ""),  # Slay the Spire
    ("lowrev_deckbuilder", 981430): (3, ""),  # Gordian Quest
    ("lowrev_deckbuilder", 1076200): (3, ""),  # Roguebook
    ("lowrev_deckbuilder", 1638390): (3, ""),  # Indies' Lies
    ("lowrev_deckbuilder", 1755830): (3, ""),  # Astrea: Six-Sided Oracles
    ("lowrev_deckbuilder", 1815570): (3, ""),  # Aces & Adventures
    ("lowrev_deckbuilder", 2026820): (3, ""),  # Die in the Dungeon
    ("lowrev_deckbuilder", 2693930): (3, ""),  # 주사위와 마왕의 성
    ("lowrev_deckbuilder", 2842800): (2, ""),  # 尸姬之梦
    ("lowrev_deckbuilder", 2870340): (2, ""),  # Decktamer
    ("lowrev_detective", 284770): (2, ""),  # Enigmatis 2: The Mists of Ravenwood
    ("lowrev_detective", 350640): (3, ""),  # Sherlock Holmes: The Devil's Daughter
    ("lowrev_detective", 368370): (3, ""),  # Her Story
    ("lowrev_detective", 373390): (3, ""),  # Contradiction: Spot The Liar
    ("lowrev_detective", 615770): (3, ""),  # Nancy Drew®: Message in a Haunted Mansio
    ("lowrev_detective", 712730): (2, ""),  # SIMULACRA
    ("lowrev_detective", 1271300): (3, ""),  # Methods: The Detective Competition
    ("lowrev_detective", 1466390): (3, ""),  # Kathy Rain 2: Soothsayer
    ("lowrev_detective", 2450840): (0, "IRRELEVANT"),  # Detective Dotson
    ("lowrev_detective", 2514960): (1, "GENRE_ONLY"),  # Refind Self: 성격 진단 게임
    ("lowrev_metroidvania", 332200): (3, ""),  # Axiom Verge
    ("lowrev_metroidvania", 345820): (3, ""),  # Shantae and the Pirate's Curse
    ("lowrev_metroidvania", 598700): (1, "GENRE_ONLY"),  # The Vagrant
    ("lowrev_metroidvania", 813230): (3, ""),  # ANIMAL WELL
    ("lowrev_metroidvania", 1379870): (1, "GENRE_ONLY"),  # Tribal Hunter
    ("lowrev_metroidvania", 1419160): (3, ""),  # Souldiers
    ("lowrev_metroidvania", 1517970): (3, ""),  # Aeterna Noctis
    ("lowrev_metroidvania", 1522870): (1, "FRANCHISE_OR_VARIANT"),  # Supraland Six Inches Under
    ("lowrev_metroidvania", 1748620): (1, "LOW_QUALITY"),  # FlipWitch - Forbidden Sex Hex
    ("lowrev_metroidvania", 2971610): (3, ""),  # HOLE
    ("lowrev_towerdefense", 23530): (1, "GENRE_ONLY"),  # Earth Defense Force: Insect Armageddon
    ("lowrev_towerdefense", 408410): (3, ""),  # X-Morph: Defense
    ("lowrev_towerdefense", 422900): (3, ""),  # Particle Fleet: Emergence
    ("lowrev_towerdefense", 458710): (3, ""),  # Kingdom Rush Frontiers - Tower Defense
    ("lowrev_towerdefense", 603320): (3, ""),  # Age of Defense
    ("lowrev_towerdefense", 701870): (2, ""),  # Swarm Queen
    ("lowrev_towerdefense", 848480): (2, ""),  # Creeper World 4
    ("lowrev_towerdefense", 1522820): (3, ""),  # Orcs Must Die! 3
    ("lowrev_towerdefense", 1566690): (2, ""),  # Outpost: Infinity Siege
    ("lowrev_towerdefense", 2607060): (3, ""),  # From Glory To Goo
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

# --- dev 미판정 4개 (전략·ARPG·FPS 축 확보) ---
# dev 12개 중 8개만 판정된 채로 min_reviews·인터리빙·rec_boost 를 골랐다. 표본을 늘린다.
DEV_TOPUP_V1 = {
    # coh_fps: Counter-Strike 2 / PUBG / 레인보우식스 시즈
    ("coh_fps", 240):     (1, "FRANCHISE_OR_VARIANT"),       # Counter-Strike: Source — 전작
    ("coh_fps", 805940):  (1, "LOW_QUALITY"),                # RUSSIA BATTLEGROUNDS
    ("coh_fps", 2379390): (3, ""),                           # R6 Extraction — 시즈의 공식 협동 스핀오프
    ("coh_fps", 300):     (3, ""),                           # Day of Defeat: Source
    ("coh_fps", 3028330): (3, ""),                           # Battlefield REDSEC
    ("coh_fps", 1649240): (1, "GENRE_ONLY"),                 # Returnal — 3인칭 로그라이크 슈터
    ("coh_fps", 1250):    (3, ""),                           # Killing Floor
    ("coh_fps", 895400):  (3, ""),                           # Deadside — 오픈월드 생존 슈터
    ("coh_fps", 826630):  (0, "IRRELEVANT"),                 # Iron Harvest — RTS
    ("coh_fps", 17710):   (2, ""),                           # Nuclear Dawn — FPS/RTS 하이브리드

    # coh_strategy: Civilization VI / XCOM 2 / Stellaris
    ("coh_strategy", 3276050): (1, "GENRE_ONLY"),            # SpaceCraft — 우주 샌드박스 크래프팅
    ("coh_strategy", 882100):  (3, ""),                      # XCOM: Chimera Squad — 공식 스핀오프
    ("coh_strategy", 2772750): (3, ""),                      # Age of History 3 — 대전략
    ("coh_strategy", 1614550): (1, "GENRE_ONLY"),            # Astro Colony — 자동화 팩토리
    ("coh_strategy", 760060):  (3, ""),                      # Mutant Year Zero — XCOM식 턴제 전술
    ("coh_strategy", 359320):  (1, "GENRE_ONLY"),            # Elite Dangerous — 우주 비행 시뮬
    ("coh_strategy", 223830):  (3, ""),                      # Xenonauts — XCOM 계열
    ("coh_strategy", 392160):  (2, ""),                      # X4: Foundations — 우주 제국 경영
    ("coh_strategy", 612570):  (2, ""),                      # Fort Triumph — 턴제 + 탐험
    ("coh_strategy", 1369700): (1, "TOO_NICHE"),             # Solar Expanse — rev 1,140

    # coh_arpg: DARK SOULS III / Witcher 3 / Skyrim
    ("coh_arpg", 20900):   (1, "FRANCHISE_OR_VARIANT"),      # The Witcher: Enhanced Edition — 전작
    ("coh_arpg", 814380):  (3, ""),                          # Sekiro — FromSoftware 직결
    ("coh_arpg", 306130):  (2, ""),                          # The Elder Scrolls Online
    ("coh_arpg", 644830):  (2, ""),                          # The Surge 2
    ("coh_arpg", 939850):  (0, "IRRELEVANT"),                # Man of Medan — 시네마틱 호러
    ("coh_arpg", 1321230): (0, "IRRELEVANT"),                # Angels of Death Episode.Eddie
    ("coh_arpg", 2067920): (1, "GENRE_ONLY"),                # Rogue : Genesia
    ("coh_arpg", 274520):  (1, "GENRE_ONLY"),                # Darkwood — 탑다운 서바이벌 호러
    ("coh_arpg", 1020470): (1, "GENRE_ONLY"),                # Evoland
    ("coh_arpg", 2393160): (1, "GENRE_ONLY"),                # Nice Day for Fishing

    # mix_arpg_survival: DARK SOULS III / Fallout 4 / The Forest
    ("mix_arpg_survival", 1716740): (3, ""),                 # Starfield — Bethesda 오픈월드 RPG
    ("mix_arpg_survival", 2796180): (2, ""),                 # SILENT BREATH
    ("mix_arpg_survival", 814380):  (3, ""),                 # Sekiro
    ("mix_arpg_survival", 440730):  (1, "GENRE_ONLY"),       # Survival Zombies
    ("mix_arpg_survival", 939850):  (1, "GENRE_ONLY"),       # Man of Medan — 호러지만 생존 아님
    ("mix_arpg_survival", 313120):  (3, ""),                 # Stranded Deep
    ("mix_arpg_survival", 274520):  (3, ""),                 # Darkwood — 가혹함 + 생존, 이 프로필엔 정확
    ("mix_arpg_survival", 877010):  (2, ""),                 # Beyond Contact
    ("mix_arpg_survival", 570940):  (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS: REMASTERED — 전작
    ("mix_arpg_survival", 876650):  (2, ""),                 # Aground
}

# --- 니치 시드 프로필 6개 (리뷰 1천~2만 구간 시드, 시드 개수 1·3·5·10) ---
NICHE_V1 = {
    # niche_puzzle_solo (1시드): Opus Magnum — Zachtronics 엔지니어링 퍼즐
    ("niche_puzzle_solo", 92800):   (3, ""),                 # SpaceChem — 같은 제작사 같은 장르
    ("niche_puzzle_solo", 300570):  (3, ""),                 # Infinifactory — 같은 제작사
    ("niche_puzzle_solo", 2738230): (2, ""),                 # The House of Tesla — 기계 퍼즐
    ("niche_puzzle_solo", 2331280): (1, "KEYWORD_MATCH"),    # 최고의 대장장이 — '제작'만 겹침
    ("niche_puzzle_solo", 508600):  (0, "IRRELEVANT"),       # ROD: Revolt Of Defense
    ("niche_puzzle_solo", 262410):  (1, "GENRE_ONLY"),       # World of Guns — 분해 시뮬, 설계 아님
    ("niche_puzzle_solo", 4194800): (2, ""),                 # 스타베스터 — 인크리멘탈 최적화
    ("niche_puzzle_solo", 983870):  (2, ""),                 # FOUNDRY — 공장 자동화 최적화
    ("niche_puzzle_solo", 418040):  (1, "GENRE_ONLY"),       # hocus — 착시 퍼즐, 종류가 다름
    ("niche_puzzle_solo", 591380):  (2, ""),                 # Bomb Squad Academy — 논리 퍼즐

    # niche_soulslike_solo (1시드): Salt and Sanctuary — 2D 소울라이크
    ("niche_soulslike_solo", 327860):  (0, "KEYWORD_MATCH"), # Salt — 이름만 겹치는 항해 게임
    ("niche_soulslike_solo", 328760):  (0, "KEYWORD_MATCH"), # SanctuaryRPG — 이름만 겹침
    ("niche_soulslike_solo", 1162130): (0, "IRRELEVANT"),    # Windbound — 난파 생존
    ("niche_soulslike_solo", 2140510): (0, "KEYWORD_MATCH"), # Town of Salem 2 — 'Salem' 유사
    ("niche_soulslike_solo", 1245620): (3, ""),              # ELDEN RING — 소울라이크 정점
    ("niche_soulslike_solo", 1063730): (1, "GENRE_ONLY"),    # New World — MMORPG
    ("niche_soulslike_solo", 1230530): (1, "GENRE_ONLY"),    # Atlas Fallen — 액션 RPG
    ("niche_soulslike_solo", 1072420): (0, "IRRELEVANT"),    # 드래곤 퀘스트 빌더즈 2
    ("niche_soulslike_solo", 250680):  (2, ""),              # BELOW — 가혹한 고독 탐험
    ("niche_soulslike_solo", 2757820): (0, "LOW_QUALITY"),   # Sensual Adventures — 성인물, 필터가 놓침

    # niche_tactics (3시드): Into the Breach / Monster Train / Mini Metro
    ("niche_tactics", 1127500): (3, ""),                     # Mini Motorways — Mini Metro 제작사 후속작
    ("niche_tactics", 2742830): (3, ""),                     # Monster Train 2 — 공식 후속작
    ("niche_tactics", 538030):  (2, ""),                     # Xenonauts 2 — 턴제 전술
    ("niche_tactics", 1124180): (2, ""),                     # 철도관제 시뮬레이터
    ("niche_tactics", 1062810): (3, ""),                     # Inkbound — Monster Train 제작진
    ("niche_tactics", 2097570): (3, ""),                     # StarVaders — 덱빌딩 + 그리드 전술
    ("niche_tactics", 1122120): (2, ""),                     # STATIONflow — 역 동선 관리
    ("niche_tactics", 2870340): (2, ""),                     # Decktamer — 덱빌딩 로그라이크
    ("niche_tactics", 701870):  (1, "GENRE_ONLY"),           # Swarm Queen — 실시간 전략
    ("niche_tactics", 3331320): (0, "KEYWORD_MATCH"),        # Back in Service — 'Metro' 만 겹침

    # niche_cozy_casual (3시드): A Short Hike / ISLANDERS / Gorogoa
    ("niche_cozy_casual", 1957990): (3, ""),                 # Tile Cities — ISLANDERS 직결
    ("niche_cozy_casual", 3047750): (3, ""),                 # Herdling — A Short Hike 정서
    ("niche_cozy_casual", 596590):  (2, ""),                 # Linked — 미니멀 퍼즐
    ("niche_cozy_casual", 3527290): (1, "GENRE_ONLY"),       # PEAK — 등반이지만 하드코어 협동 생존
    ("niche_cozy_casual", 2533960): (3, ""),                 # SUMMERHOUSE — 규칙 없는 아기자기 건축
    ("niche_cozy_casual", 1385100): (1, "GENRE_ONLY"),       # Insurmountable — 등산 로그라이트
    ("niche_cozy_casual", 1004270): (2, ""),                 # My Island — 편안한 섬 생활
    ("niche_cozy_casual", 400740):  (1, "GENRE_ONLY"),       # VERGE — 액션 어드벤처
    ("niche_cozy_casual", 569480):  (2, ""),                 # Kingdoms and Castles — 가벼운 도시건설
    ("niche_cozy_casual", 115800):  (2, ""),                 # Owlboy — 따뜻한 픽셀 탐험

    # niche_sim (5시드): Shipbreaker / Hacknet / Golden Idol / 방탈출 / shapez 2
    ("niche_sim", 3184110): (2, ""),                         # Widget Inc. — 공장 빌더
    ("niche_sim", 1170570): (1, "GENRE_ONLY"),               # The Drifter — 추리 퍼즐 아님
    ("niche_sim", 850450):  (3, ""),                         # Escape First — 방탈출 직결
    ("niche_sim", 1022980): (3, ""),                         # Ostranauts — Shipbreaker 정확 대응
    ("niche_sim", 526740):  (3, ""),                         # hack_me — Hacknet 직결
    ("niche_sim", 427520):  (3, ""),                         # Factorio — shapez 계열 원형
    ("niche_sim", 3036350): (3, ""),                         # A Case of Fraud — Golden Idol 정확 대응
    ("niche_sim", 1812090): (3, ""),                         # Escape Academy
    ("niche_sim", 747910):  (2, ""),                         # Disassembly 3D — 해체 물리
    ("niche_sim", 2980270): (2, ""),                         # HackHub — 해커 시뮬

    # niche_roguelite (10시드): Death's Door / Children of Morta / Roboquest / ... 10개
    ("niche_roguelite", 239350):  (1, "FRANCHISE_OR_VARIANT"),  # Spelunky — Spelunky 2 의 전작
    ("niche_roguelite", 2060790): (1, "TOO_NICHE"),          # 하운티 — rev 358, 로그라이트 아님
    ("niche_roguelite", 3159570): (2, ""),                   # Lynked — 로봇 액션 협동
    ("niche_roguelite", 2847740): (1, "TOO_NICHE"),          # Super Drift Blade — rev 341
    ("niche_roguelite", 655780):  (0, "IRRELEVANT"),         # Project 5: Sightseer — 샌드박스
    ("niche_roguelite", 1078200): (2, ""),                   # Spirits Abyss — 로그라이트 플랫포머
    ("niche_roguelite", 241600):  (1, "FRANCHISE_OR_VARIANT"),  # Rogue Legacy — 전작
    ("niche_roguelite", 1331210): (0, "IRRELEVANT"),         # Wolfstride — 턴제 메카 RPG
    ("niche_roguelite", 257850):  (2, ""),                   # Hyper Light Drifter
    ("niche_roguelite", 2569760): (2, ""),                   # The Mound — 협동 탐험 액션
}

# --- 라운드1 블라인드 판정 (src/blind_judge.py) ---
# 비교할 5개 설정(하한 0/300/1000 · 인터리빙 on/off · rec_boost 0.15/0.03)의 Top-10 **합집합** 193쌍.
# 한 설정만 판정하면 그 설정이 자동으로 이긴다 — 실측으로 기준선 미판정 3칸 vs 대안 55~115칸이었고
# 미판정을 성공으로 세면 순위가 뒤집혔다. 합집합을 채워야 비교가 성립한다.
#
# **판정 근거가 이전과 다르다.** 이 라운드는 설정/순위/전략/프로필 실제 id 와 함께
# **리뷰 수도 가린 채** 채점했다. 리뷰 수는 min_reviews 실험의 조작 변수라 보이면 어느 설정에서
# 왔는지 추론된다. 그래서 TOO_NICHE/LOW_QUALITY 태그는 설명과 이름에서 읽히는 범위로만 붙었다.
BLIND_ROUND1 = {
    ("coh_arpg", 270550): (0, "IRRELEVANT"),  # Yet Another Zombie Defense
    ("coh_arpg", 440540): (1, "GENRE_ONLY"),  # Ara Fell: Enhanced Edition
    ("coh_arpg", 570940): (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS™: REMASTERED
    ("coh_arpg", 1245620): (3, ""),  # ELDEN RING
    ("coh_arpg", 2437570): (1, "TOO_NICHE"),  # Golden Chambers
    ("coh_classic_multi", 55040): (1, "GENRE_ONLY"),  # Atom Zombie Smasher
    ("coh_classic_multi", 1801520): (1, "GENRE_ONLY"),  # Zombieville USA 3D
    ("coh_cozy", 1274490): (2, ""),  # Green Project
    ("coh_cozy", 2075580): (2, ""),  # Under A New Sun
    ("coh_cozy", 2846060): (2, ""),  # Subterror
    ("coh_cozy", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("coh_fps", 308600): (1, "GENRE_ONLY"),  # Skillshot City
    ("coh_fps", 351230): (1, "KEYWORD_MATCH"),  # Counter Spell
    ("coh_fps", 390560): (0, "MODE_MISMATCH"),  # Fantasy Strike
    ("coh_fps", 1121710): (2, ""),  # Total Lockdown
    ("coh_fps", 1284210): (0, "IRRELEVANT"),  # Guild Wars 2®
    ("coh_grand_strategy", 214730): (2, ""),  # Space Rangers HD: A War Apart
    ("coh_grand_strategy", 314980): (3, ""),  # Supreme Ruler Ultimate
    ("coh_grand_strategy", 462940): (3, ""),  # Making History: The Second World War
    ("coh_grand_strategy", 589290): (1, "MODE_MISMATCH"),  # Holdfast: Nations At War
    ("coh_grand_strategy", 965320): (2, ""),  # The Settlers® 7 : History Edition
    ("coh_grand_strategy", 1201700): (3, ""),  # Warnament
    ("coh_grand_strategy", 1611600): (3, ""),  # WARNO
    ("coh_grand_strategy", 1768280): (2, ""),  # Ozymandias: Bronze Age Empire Sim
    ("coh_grand_strategy", 1844380): (2, ""),  # Warhammer Age of Sigmar: Realms of Ruin
    ("coh_grand_strategy", 2154730): (0, "IRRELEVANT"),  # Building Destruction
    ("coh_grand_strategy", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("coh_indie_platformer", 1046400): (3, ""),  # Astalon: 지구의 눈물
    ("coh_indie_platformer", 1200770): (0, "IRRELEVANT"),  # Deathground
    ("coh_indie_platformer", 1375900): (2, ""),  # 혈색 광맥
    ("coh_openworld_survival", 280790): (2, ""),  # Creativerse
    ("coh_openworld_survival", 331870): (2, ""),  # AER Memories of Old
    ("coh_openworld_survival", 358920): (2, ""),  # Star Control I and II
    ("coh_openworld_survival", 366870): (3, ""),  # Narcosis
    ("coh_openworld_survival", 401360): (3, ""),  # The Aquatic Adventure of the Last Human
    ("coh_openworld_survival", 655780): (2, ""),  # Project 5: Sightseer
    ("coh_openworld_survival", 1645630): (1, "GENRE_ONLY"),  # FEROCIOUS
    ("coh_openworld_survival", 2060790): (2, ""),  # 하운티
    ("coh_openworld_survival", 2567870): (0, "IRRELEVANT"),  # Chained Together
    ("coh_openworld_survival", 2658470): (0, "IRRELEVANT"),  # Is this Game Trying to Kill Me?
    ("coh_openworld_survival", 2800450): (2, ""),  # Planetaries
    ("coh_openworld_survival", 2941710): (1, "TOO_NICHE"),  # Project Silverfish
    ("coh_openworld_survival", 3151400): (1, "GENRE_ONLY"),  # Liminal Universe
    ("coh_strategy", 226860): (3, ""),  # Galactic Civilizations III
    ("coh_strategy", 280720): (3, ""),  # Imagine Earth
    ("coh_strategy", 282210): (3, ""),  # Sid Meier's Starships
    ("coh_strategy", 366910): (2, ""),  # The Long Journey Home
    ("coh_strategy", 418910): (1, "GENRE_ONLY"),  # Idle Civilization
    ("coh_strategy", 736820): (3, ""),  # Knights of Honor II: Sovereign
    ("coh_strategy", 835570): (1, "GENRE_ONLY"),  # 컨커러스 블레이드
    ("coh_strategy", 2134770): (2, ""),  # SteamWorld Build
    ("coh_survival_craft", 55040): (1, "GENRE_ONLY"),  # Atom Zombie Smasher
    ("coh_survival_craft", 263920): (1, "LOW_QUALITY"),  # Zombie Grinder
    ("coh_survival_craft", 376210): (2, ""),  # The Isle
    ("coh_survival_craft", 408960): (1, "GENRE_ONLY"),  # Zombasite
    ("coh_survival_craft", 876650): (2, ""),  # Aground
    ("coh_survival_craft", 1137490): (2, ""),  # Invasion 2037
    ("coh_survival_craft", 1274490): (2, ""),  # Green Project
    ("coh_survival_craft", 1326470): (3, ""),  # Sons Of The Forest
    ("coh_survival_craft", 1668310): (1, "GENRE_ONLY"),  # 24 Killers
    ("coh_survival_craft", 2268560): (2, ""),  # Zombie Survival Game Online
    ("coh_vehicle_sim", 41740): (2, ""),  # Cargo! The Quest for Gravity
    ("coh_vehicle_sim", 46370): (2, ""),  # Rig n Roll
    ("coh_vehicle_sim", 232010): (1, "FRANCHISE_OR_VARIANT"),  # Euro Truck Simulator
    ("coh_vehicle_sim", 302060): (2, ""),  # Trucks & Trailers
    ("coh_vehicle_sim", 446020): (3, ""),  # Jalopy
    ("coh_vehicle_sim", 493490): (2, ""),  # City Car Driving
    ("coh_vehicle_sim", 768180): (3, ""),  # Truck Driver
    ("coh_vehicle_sim", 1214520): (0, "KEYWORD_MATCH"),  # Pro Gymnast Simulator
    ("coh_vehicle_sim", 1369670): (3, ""),  # Motor Town: Behind The Wheel
    ("coh_vehicle_sim", 1565890): (2, ""),  # RaceLeague
    ("coh_vehicle_sim", 1763830): (1, "TOO_NICHE"),  # My Truck Game
    ("coh_vehicle_sim", 2303180): (1, "GENRE_ONLY"),  # Resoraki: 레이싱
    ("coh_vehicle_sim", 3933710): (1, "TOO_NICHE"),  # My Car My Life
    ("mix_arpg_survival", 382310): (1, "GENRE_ONLY"),  # Eco
    ("mix_arpg_survival", 924140): (1, "LOW_QUALITY"),  # Hand Simulator: Survival
    ("mix_arpg_survival", 1326470): (3, ""),  # Sons Of The Forest
    ("mix_arpg_survival", 1755080): (2, ""),  # Away From Life
    ("mix_arpg_survival", 3712080): (1, "GENRE_ONLY"),  # Funnel Runners
    ("mix_cozy_fps", 2075580): (2, ""),  # Under A New Sun
    ("mix_cozy_fps", 2846060): (2, ""),  # Subterror
    ("mix_cozy_fps", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("mix_fps_cozy", 1274490): (1, "GENRE_ONLY"),  # Green Project
    ("mix_fps_cozy", 1284210): (0, "IRRELEVANT"),  # Guild Wars 2®
    ("mix_fps_cozy", 1726130): (3, ""),  # Pathless Woods
    ("mix_fps_cozy", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("mix_grand_casual", 42810): (3, ""),  # For The Glory: A Europa Universalis Game
    ("mix_grand_casual", 214730): (2, ""),  # Space Rangers HD: A War Apart
    ("mix_grand_casual", 231330): (0, "IRRELEVANT"),  # Deadfall Adventures
    ("mix_grand_casual", 314980): (2, ""),  # Supreme Ruler Ultimate
    ("mix_grand_casual", 965320): (2, ""),  # The Settlers® 7 : History Edition
    ("mix_grand_casual", 1201700): (3, ""),  # Warnament
    ("mix_grand_casual", 1882580): (1, "GENRE_ONLY"),  # FALL GIRLS
    ("mix_grand_casual", 2154730): (0, "IRRELEVANT"),  # Building Destruction
    ("mix_grand_casual", 2179380): (0, "IRRELEVANT"),  # Sand:box
    ("mix_grand_casual", 3035500): (1, "GENRE_ONLY"),  # Fantasy Map Simulator
    ("mix_grand_casual", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("mix_grand_casual", 4373040): (2, ""),  # But Why?
    ("mix_indie_multi", 94400): (1, "MODE_MISMATCH"),  # Nidhogg
    ("mix_indie_multi", 414120): (2, ""),  # Modbox
    ("mix_indie_multi", 758990): (2, ""),  # Ancient Warfare 3
    ("mix_indie_multi", 1200770): (1, "GENRE_ONLY"),  # Deathground
    ("mix_indie_multi", 1375900): (2, ""),  # 혈색 광맥
    ("mix_indie_multi", 2157210): (1, "TOO_NICHE"),  # Dead of Darkness
    ("mix_multi_indie", 104900): (2, ""),  # ORION: Prelude
    ("mix_multi_indie", 230230): (0, "IRRELEVANT"),  # Divinity: Original Sin (Classic)
    ("mix_multi_indie", 263920): (1, "LOW_QUALITY"),  # Zombie Grinder
    ("mix_multi_indie", 280790): (2, ""),  # Creativerse
    ("mix_multi_indie", 298240): (1, "LOW_QUALITY"),  # War Trigger 3
    ("mix_multi_indie", 610960): (1, "TOO_NICHE"),  # 红石遗迹 - Red Obsidian Remnant
    ("mix_multi_indie", 770100): (0, "IRRELEVANT"),  # One Deck Dungeon
    ("mix_multi_indie", 1815530): (2, ""),  # Dungeon Defenders: Going Rogue
    ("mix_multi_indie", 2176320): (0, "IRRELEVANT"),  # Magical Drop VI
    ("mix_multi_indie", 2268560): (1, "GENRE_ONLY"),  # Zombie Survival Game Online
    ("mix_openworld_cozy", 244770): (2, ""),  # StarMade
    ("mix_openworld_cozy", 655780): (2, ""),  # Project 5: Sightseer
    ("mix_openworld_cozy", 1645630): (1, "GENRE_ONLY"),  # FEROCIOUS
    ("mix_openworld_cozy", 2060790): (2, ""),  # 하운티
    ("mix_rpg_racing", 1020800): (2, ""),  # CAR TUNE: Project
    ("mix_rpg_racing", 3216340): (1, "TOO_NICHE"),  # Tearscape
    ("mix_rpg_racing", 3800340): (0, "KEYWORD_MATCH"),  # ScootX
    ("mix_survival_strategy", 214730): (1, "GENRE_ONLY"),  # Space Rangers HD: A War Apart
    ("mix_survival_strategy", 382310): (2, ""),  # Eco
    ("mix_survival_strategy", 541300): (3, ""),  # Survive the Nights
    ("mix_survival_strategy", 876650): (2, ""),  # Aground
    ("mix_survival_strategy", 924140): (1, "LOW_QUALITY"),  # Hand Simulator: Survival
    ("mix_survival_strategy", 1326470): (3, ""),  # Sons Of The Forest
    ("mix_survival_strategy", 1755080): (2, ""),  # Away From Life
    ("mix_survival_strategy", 3712080): (1, "GENRE_ONLY"),  # Funnel Runners
    ("mix_survival_strategy", 4474950): (2, ""),  # LIFE EFFECT Survival
    ("niche_cozy_casual", 347940): (1, "GENRE_ONLY"),  # Forsaken Isle
    ("niche_cozy_casual", 356250): (3, ""),  # Gathering Sky
    ("niche_cozy_casual", 513720): (0, "IRRELEVANT"),  # Archipelago
    ("niche_cozy_casual", 617670): (1, "GENRE_ONLY"),  # Zup! S
    ("niche_cozy_casual", 688130): (1, "MODE_MISMATCH"),  # Pogostuck: Rage With Your Friends
    ("niche_cozy_casual", 710610): (1, "GENRE_ONLY"),  # Don't Sink
    ("niche_cozy_casual", 931270): (3, ""),  # MicroTown
    ("niche_cozy_casual", 1348920): (2, ""),  # Wind Peaks
    ("niche_cozy_casual", 1722520): (2, ""),  # FIND ALL 2: Middle Ages
    ("niche_cozy_casual", 2093900): (2, ""),  # Island Cities - Jigsaw Puzzle
    ("niche_cozy_casual", 2236070): (2, ""),  # Peaks of Yore
    ("niche_cozy_casual", 2239150): (3, ""),  # Thronefall
    ("niche_cozy_casual", 2880750): (2, ""),  # Leaf's Odyssey
    ("niche_cozy_casual", 3580340): (2, ""),  # Ash & Adam's Existential Treads
    ("niche_puzzle_solo", 230290): (1, "GENRE_ONLY"),  # Universe Sandbox
    ("niche_puzzle_solo", 233450): (2, ""),  # Prison Architect
    ("niche_puzzle_solo", 375820): (3, ""),  # Human Resource Machine
    ("niche_puzzle_solo", 973250): (2, ""),  # Altered
    ("niche_puzzle_solo", 1293540): (2, ""),  # Outerverse
    ("niche_puzzle_solo", 1318690): (3, ""),  # shapez
    ("niche_puzzle_solo", 1388770): (0, "IRRELEVANT"),  # Cruelty Squad
    ("niche_puzzle_solo", 1451720): (1, "GENRE_ONLY"),  # Minesweeper Classy
    ("niche_puzzle_solo", 1536570): (3, ""),  # The Last Alchemist
    ("niche_puzzle_solo", 1596310): (1, "GENRE_ONLY"),  # Crypto Mining Simulator
    ("niche_puzzle_solo", 1876000): (2, ""),  # IFO
    ("niche_puzzle_solo", 2685900): (2, ""),  # Mind Over Magnet
    ("niche_roguelite", 250680): (2, ""),  # BELOW
    ("niche_roguelite", 308600): (1, "GENRE_ONLY"),  # Skillshot City
    ("niche_roguelite", 588650): (3, ""),  # Dead Cells
    ("niche_roguelite", 727510): (1, "TOO_NICHE"),  # Void Memory
    ("niche_roguelite", 787810): (3, ""),  # Rogue Heroes: Ruins of Tasos
    ("niche_roguelite", 994220): (1, "GENRE_ONLY"),  # NEOVERSE
    ("niche_roguelite", 1078420): (0, "IRRELEVANT"),  # Critters for Sale
    ("niche_roguelite", 1343810): (0, "IRRELEVANT"),  # Fallen Hero: Retribution
    ("niche_roguelite", 1356280): (1, "GENRE_ONLY"),  # 냥자의모험
    ("niche_roguelite", 1494810): (3, ""),  # Mortal Sin
    ("niche_roguelite", 2181930): (0, "IRRELEVANT"),  # DR LIVESEY ROM AND DEATH EDITION
    ("niche_roguelite", 2273430): (3, ""),  # BlazBlue Entropy Effect
    ("niche_roguelite", 2687400): (3, ""),  # GODBREAKERS
    ("niche_sim", 244910): (2, ""),  # Homesick
    ("niche_sim", 272600): (3, ""),  # Detective Grimoire
    ("niche_sim", 499910): (2, ""),  # SHINRAI - Broken Beyond Despair
    ("niche_sim", 593200): (2, ""),  # The Adventures of Fei Duanmu 端木斐异闻录
    ("niche_sim", 605230): (3, ""),  # Grey Hack
    ("niche_sim", 641990): (3, ""),  # The Escapists 2
    ("niche_sim", 799600): (3, ""),  # Cosmoteer: Starship Architect & Commande
    ("niche_sim", 965810): (2, ""),  # Kara no Shojo
    ("niche_sim", 1112790): (3, ""),  # Automation Empire
    ("niche_sim", 1546920): (3, ""),  # Overboard!
    ("niche_sim", 1603410): (3, ""),  # Lost Nova
    ("niche_sim", 2277090): (3, ""),  # DAEMON MASQUERADE
    ("niche_sim", 2414110): (3, ""),  # Builderment
    ("niche_sim", 3184990): (3, ""),  # 诡秘推理
    ("niche_sim", 4730180): (3, ""),  # Hacker's Journey
    ("niche_soulslike_solo", 280520): (1, "GENRE_ONLY"),  # Crea
    ("niche_tactics", 949230): (2, ""),  # Cities: Skylines II
    ("niche_tactics", 1176470): (2, ""),  # Terra Invicta 테라 인빅타
    ("niche_tactics", 1465550): (0, "IRRELEVANT"),  # One Lonely Outpost
    ("niche_tactics", 1842690): (2, ""),  # 기차역 스토리 (Station Manager)
    ("niche_tactics", 2272400): (3, ""),  # Station to Station
    ("niche_tactics", 2449450): (2, ""),  # 우주 식민 회사
    ("niche_tactics", 2853590): (2, ""),  # Void War
    ("niche_tactics", 3680900): (2, ""),  # Future War Tactics: SOF vs Alien Invasio
}


# --- 라운드3 블라인드: 태그 임베딩 (src/blind_judge.py) ---
# 비교 4설정(태그없음 / 태그있음 / 태그+트렌드0.05 / 태그+트렌드0.15)의 Top-10 합집합 224쌍.
# 같은 12,844개 게임 위에서 태그 유무만 바꾼다 — 코퍼스 크기 변화가 섞이지 않게.
# 판정자는 태그를 못 본다(시트에 genres 와 설명만 있다). 태그를 보여주면 "태그가 겹치니 좋다"는
# 순환이 생긴다.
BLIND_TAGS = {
    ("coh_arpg", 22330): (1, "FRANCHISE_OR_VARIANT"),  # The Elder Scrolls IV: Oblivion® Game of 
    ("coh_arpg", 335300): (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS™ II: Scholar of the First Sin
    ("coh_arpg", 750130): (2, ""),  # The Sinking City Remastered
    ("coh_arpg", 973760): (2, ""),  # Thronebreaker: The Witcher Tales
    ("coh_arpg", 1620730): (2, ""),  # Hell is Us
    ("coh_arpg", 1771300): (3, ""),  # Kingdom Come: Deliverance II
    ("coh_arpg", 1920490): (3, ""),  # The Outer Worlds: Spacer's Choice Editio
    ("coh_arpg", 3282300): (2, ""),  # Mistfall Hunter
    ("coh_arpg", 3321460): (3, ""),  # 붉은사막
    ("coh_classic_multi", 333930): (3, ""),  # Dirty Bomb®
    ("coh_classic_multi", 383150): (2, ""),  # Dead Island Definitive Edition
    ("coh_classic_multi", 1451480): (3, ""),  # The Greatest Penguin Heist of All Time
    ("coh_classic_multi", 1492070): (3, ""),  # Sker Ritual
    ("coh_classic_multi", 2827230): (2, ""),  # Wild Assault / 兽猎突袭
    ("coh_cozy", 214730): (0, "IRRELEVANT"),  # Space Rangers HD: A War Apart
    ("coh_cozy", 580200): (3, ""),  # Yonder: The Cloud Catcher Chronicles
    ("coh_cozy", 1092590): (0, "IRRELEVANT"),  # 沙雕之路
    ("coh_cozy", 1263240): (0, "IRRELEVANT"),  # Skate Story
    ("coh_cozy", 1963370): (2, ""),  # No One Survived
    ("coh_cozy", 2252680): (3, ""),  # Farlands
    ("coh_cozy", 2340520): (3, ""),  # 세이큐 이야기
    ("coh_cozy", 2418520): (3, ""),  # Farm Together 2
    ("coh_cozy", 2661300): (3, ""),  # Grounded 2
    ("coh_cozy", 3321460): (0, "IRRELEVANT"),  # 붉은사막
    ("coh_fps", 222880): (3, ""),  # Insurgency
    ("coh_fps", 282440): (2, ""),  # Quake Live
    ("coh_fps", 433850): (3, ""),  # Z1 Battle Royale
    ("coh_fps", 729040): (2, ""),  # Borderlands Game of the Year Enhanced
    ("coh_fps", 1962663): (3, ""),  # 콜 오브 듀티®: 워존
    ("coh_fps", 2000950): (3, ""),  # Call of Duty®: Modern Warfare®
    ("coh_grand_strategy", 21970): (3, ""),  # R.U.S.E.™
    ("coh_grand_strategy", 70600): (1, "GENRE_ONLY"),  # Worms Ultimate Mayhem
    ("coh_grand_strategy", 235380): (3, ""),  # Blitzkrieg 3
    ("coh_grand_strategy", 597180): (3, ""),  # Old World 올드 월드
    ("coh_grand_strategy", 603850): (3, ""),  # Age of History II
    ("coh_grand_strategy", 743640): (0, "IRRELEVANT"),  # Achievement Clicker
    ("coh_grand_strategy", 1468720): (1, "GENRE_ONLY"),  # Ultimate Epic Battle Simulator 2
    ("coh_grand_strategy", 1560250): (1, "GENRE_ONLY"),  # Rising Front
    ("coh_grand_strategy", 2168680): (1, "GENRE_ONLY"),  # Nuclear Option
    ("coh_grand_strategy", 3105960): (1, "GENRE_ONLY"),  # Astrobuilder
    ("coh_grand_strategy", 3381680): (3, ""),  # Age of History 2: Definitive Edition
    ("coh_indie_platformer", 40800): (3, ""),  # Super Meat Boy
    ("coh_indie_platformer", 236090): (3, ""),  # Dust: An Elysian Tail
    ("coh_indie_platformer", 916730): (3, ""),  # Gato Roboto
    ("coh_indie_platformer", 2273430): (2, ""),  # BlazBlue Entropy Effect
    ("coh_indie_platformer", 2317640): (2, ""),  # JUMP KING QUEST
    ("coh_indie_platformer", 2665680): (2, ""),  # 바벨탑: 혼돈의 생존자들
    ("coh_openworld_survival", 8500): (1, "MODE_MISMATCH"),  # 이브 온라인
    ("coh_openworld_survival", 211820): (3, ""),  # Starbound
    ("coh_openworld_survival", 274520): (2, ""),  # Darkwood 다크우드
    ("coh_openworld_survival", 313120): (3, ""),  # Stranded Deep
    ("coh_openworld_survival", 359320): (2, ""),  # Elite Dangerous
    ("coh_openworld_survival", 361420): (3, ""),  # ASTRONEER
    ("coh_openworld_survival", 738520): (3, ""),  # Breathedge
    ("coh_openworld_survival", 848450): (3, ""),  # 서브노티카: 빌로우 제로
    ("coh_openworld_survival", 914620): (2, ""),  # Mist Survival
    ("coh_openworld_survival", 1931180): (3, ""),  # Lost Skies
    ("coh_openworld_survival", 3276050): (2, ""),  # SpaceCraft
    ("coh_openworld_survival", 3280350): (2, ""),  # DEATH STRANDING 2: ON THE BEACH
    ("coh_strategy", 1124300): (3, ""),  # 휴먼카인드
    ("coh_strategy", 1295660): (3, ""),  # 시드 마이어의 문명 VII
    ("coh_survival_craft", 322330): (2, ""),  # Don't Starve Together
    ("coh_survival_craft", 346110): (3, ""),  # ARK: Survival Evolved
    ("coh_survival_craft", 360170): (3, ""),  # How to Survive 2
    ("coh_survival_craft", 391730): (3, ""),  # Crashlands
    ("coh_survival_craft", 541300): (3, ""),  # Survive the Nights
    ("coh_survival_craft", 914620): (3, ""),  # Mist Survival
    ("coh_survival_craft", 1766060): (3, ""),  # HumanitZ
    ("coh_survival_craft", 1783560): (2, ""),  # The Last Caretaker
    ("coh_survival_craft", 1963370): (3, ""),  # No One Survived
    ("coh_vehicle_sim", 266410): (3, ""),  # iRacing
    ("coh_vehicle_sim", 287310): (1, "GENRE_ONLY"),  # 리볼트
    ("coh_vehicle_sim", 539720): (1, "GENRE_ONLY"),  # Razortron 2000
    ("coh_vehicle_sim", 635260): (2, ""),  # CarX Drift Racing Online
    ("coh_vehicle_sim", 1351240): (3, ""),  # Taxi Life: A City Driving Simulator
    ("coh_vehicle_sim", 1520370): (3, ""),  # Mon Bazou
    ("coh_vehicle_sim", 1578390): (3, ""),  # My Garage
    ("coh_vehicle_sim", 1849250): (3, ""),  # EA SPORTS™ WRC
    ("coh_vehicle_sim", 3616550): (2, ""),  # Car Dealership Simulator 2
    ("coh_vehicle_sim", 3800340): (0, "KEYWORD_MATCH"),  # ScootX
    ("mix_arpg_survival", 70): (1, "GENRE_ONLY"),  # Half-Life
    ("mix_arpg_survival", 22370): (1, "FRANCHISE_OR_VARIANT"),  # Fallout 3: Game of the Year Edition
    ("mix_arpg_survival", 335300): (1, "FRANCHISE_OR_VARIANT"),  # DARK SOULS™ II: Scholar of the First Sin
    ("mix_arpg_survival", 529180): (2, ""),  # Dark and Light
    ("mix_arpg_survival", 637650): (2, ""),  # FINAL FANTASY XV WINDOWS EDITION
    ("mix_arpg_survival", 914620): (2, ""),  # Mist Survival
    ("mix_arpg_survival", 1245620): (3, ""),  # ELDEN RING
    ("mix_arpg_survival", 1783560): (2, ""),  # The Last Caretaker
    ("mix_arpg_survival", 2186680): (2, ""),  # Warhammer 40,000: Rogue Trader
    ("mix_arpg_survival", 3282300): (2, ""),  # Mistfall Hunter
    ("mix_cozy_fps", 666140): (3, ""),  # My Time at Portia
    ("mix_cozy_fps", 768200): (3, ""),  # Smalland: Survive the Wilds
    ("mix_cozy_fps", 1938090): (2, ""),  # 콜 오브 듀티®
    ("mix_cozy_fps", 1963370): (2, ""),  # No One Survived
    ("mix_cozy_fps", 2252680): (3, ""),  # Farlands
    ("mix_cozy_fps", 2340520): (3, ""),  # 세이큐 이야기
    ("mix_cozy_fps", 2418520): (3, ""),  # Farm Together 2
    ("mix_cozy_fps", 2661300): (3, ""),  # Grounded 2
    ("mix_fps_cozy", 42680): (2, ""),  # Call of Duty®: Modern Warfare® 3 (2011)
    ("mix_fps_cozy", 1962663): (3, ""),  # 콜 오브 듀티®: 워존
    ("mix_fps_cozy", 2252680): (3, ""),  # Farlands
    ("mix_fps_cozy", 2340520): (3, ""),  # 세이큐 이야기
    ("mix_fps_cozy", 2418520): (3, ""),  # Farm Together 2
    ("mix_grand_casual", 70600): (1, "GENRE_ONLY"),  # Worms Ultimate Mayhem
    ("mix_grand_casual", 394360): (3, ""),  # Hearts of Iron IV
    ("mix_grand_casual", 597180): (3, ""),  # Old World 올드 월드
    ("mix_grand_casual", 743640): (0, "IRRELEVANT"),  # Achievement Clicker
    ("mix_grand_casual", 1071870): (2, ""),  # Biped
    ("mix_grand_casual", 1468720): (1, "GENRE_ONLY"),  # Ultimate Epic Battle Simulator 2
    ("mix_grand_casual", 1560250): (1, "GENRE_ONLY"),  # Rising Front
    ("mix_grand_casual", 2168680): (1, "GENRE_ONLY"),  # Nuclear Option
    ("mix_grand_casual", 2560240): (2, ""),  # Biped 2
    ("mix_grand_casual", 3105960): (1, "GENRE_ONLY"),  # Astrobuilder
    ("mix_grand_casual", 3263320): (3, ""),  # Carry The Glass
    ("mix_grand_casual", 3381680): (3, ""),  # Age of History 2: Definitive Edition
    ("mix_grand_casual", 3450310): (3, ""),  # Europa Universalis V
    ("mix_indie_multi", 236090): (3, ""),  # Dust: An Elysian Tail
    ("mix_indie_multi", 347800): (3, ""),  # Ghost Song
    ("mix_indie_multi", 552100): (3, ""),  # Brick Rigs
    ("mix_indie_multi", 1451480): (3, ""),  # The Greatest Penguin Heist of All Time
    ("mix_indie_multi", 2218750): (1, "GENRE_ONLY"),  # Halls of Torment
    ("mix_indie_multi", 2273430): (2, ""),  # BlazBlue Entropy Effect
    ("mix_indie_multi", 2665680): (2, ""),  # 바벨탑: 혼돈의 생존자들
    ("mix_multi_indie", 211820): (3, ""),  # Starbound
    ("mix_multi_indie", 333930): (3, ""),  # Dirty Bomb®
    ("mix_multi_indie", 383150): (2, ""),  # Dead Island Definitive Edition
    ("mix_multi_indie", 444090): (3, ""),  # Paladins®
    ("mix_multi_indie", 1169040): (3, ""),  # Necesse: 네세스
    ("mix_multi_indie", 1492070): (3, ""),  # Sker Ritual
    ("mix_multi_indie", 1504570): (2, ""),  # 太荒初境
    ("mix_multi_indie", 2827230): (2, ""),  # Wild Assault / 兽猎突袭
    ("mix_openworld_cozy", 211820): (3, ""),  # Starbound
    ("mix_openworld_cozy", 313120): (2, ""),  # Stranded Deep
    ("mix_openworld_cozy", 359320): (2, ""),  # Elite Dangerous
    ("mix_openworld_cozy", 580200): (3, ""),  # Yonder: The Cloud Catcher Chronicles
    ("mix_openworld_cozy", 674140): (3, ""),  # Bugsnax
    ("mix_openworld_cozy", 738520): (3, ""),  # Breathedge
    ("mix_openworld_cozy", 848450): (3, ""),  # 서브노티카: 빌로우 제로
    ("mix_openworld_cozy", 1092590): (0, "IRRELEVANT"),  # 沙雕之路
    ("mix_openworld_cozy", 1263240): (0, "IRRELEVANT"),  # Skate Story
    ("mix_openworld_cozy", 1931180): (3, ""),  # Lost Skies
    ("mix_openworld_cozy", 3276050): (2, ""),  # SpaceCraft
    ("mix_rpg_racing", 22330): (1, "FRANCHISE_OR_VARIANT"),  # The Elder Scrolls IV: Oblivion® Game of 
    ("mix_rpg_racing", 228280): (2, ""),  # Baldur's Gate: Enhanced Edition
    ("mix_rpg_racing", 635260): (2, ""),  # CarX Drift Racing Online
    ("mix_rpg_racing", 750130): (2, ""),  # The Sinking City Remastered
    ("mix_rpg_racing", 1578390): (3, ""),  # My Garage
    ("mix_rpg_racing", 1620730): (2, ""),  # Hell is Us
    ("mix_rpg_racing", 1771300): (3, ""),  # Kingdom Come: Deliverance II
    ("mix_rpg_racing", 3321460): (3, ""),  # 붉은사막
    ("mix_rpg_racing", 3616550): (2, ""),  # Car Dealership Simulator 2
    ("mix_survival_strategy", 221100): (3, ""),  # DayZ
    ("mix_survival_strategy", 239140): (3, ""),  # Dying Light
    ("mix_survival_strategy", 914620): (3, ""),  # Mist Survival
    ("mix_survival_strategy", 1124300): (3, ""),  # 휴먼카인드
    ("mix_survival_strategy", 1183470): (2, ""),  # Imperiums: Greek Wars
    ("mix_survival_strategy", 1783560): (2, ""),  # The Last Caretaker
    ("mix_survival_strategy", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("mix_vehicle_fps", 383150): (2, ""),  # Dead Island Definitive Edition
    ("mix_vehicle_fps", 1066890): (3, ""),  # Automobilista 2
    ("mix_vehicle_fps", 1369670): (3, ""),  # Motor Town: Behind The Wheel
    ("mix_vehicle_fps", 1492070): (2, ""),  # Sker Ritual
    ("mix_vehicle_fps", 1849250): (3, ""),  # EA SPORTS™ WRC
    ("niche_cozy_casual", 331870): (3, ""),  # AER Memories of Old
    ("niche_cozy_casual", 355630): (2, ""),  # Leo’s Fortune - HD Edition
    ("niche_cozy_casual", 375820): (2, ""),  # Human Resource Machine
    ("niche_cozy_casual", 493200): (3, ""),  # RiME
    ("niche_cozy_casual", 787810): (1, "GENRE_ONLY"),  # Rogue Heroes: Ruins of Tasos
    ("niche_cozy_casual", 1084020): (2, ""),  # TheoTown
    ("niche_cozy_casual", 1148650): (2, ""),  # The Legend of Bum-Bo
    ("niche_cozy_casual", 1730250): (3, ""),  # Pan'orama
    ("niche_cozy_casual", 1740300): (3, ""),  # Smushi Come Home
    ("niche_cozy_casual", 2019810): (3, ""),  # Boxes: Lost Fragments
    ("niche_cozy_casual", 2121980): (2, ""),  # Void Stranger
    ("niche_cozy_casual", 2368930): (3, ""),  # 아일랜더스: 새로운 해안
    ("niche_cozy_casual", 4160210): (3, ""),  # The Artisan of Glimmith
    ("niche_puzzle_solo", 304410): (2, ""),  # Hexcells Infinite
    ("niche_puzzle_solo", 370360): (3, ""),  # TIS-100
    ("niche_puzzle_solo", 574720): (2, ""),  # Little Big Workshop
    ("niche_puzzle_solo", 617670): (1, "GENRE_ONLY"),  # Zup! S
    ("niche_puzzle_solo", 792100): (3, ""),  # 7 Billion Humans
    ("niche_puzzle_solo", 1062160): (3, ""),  # Poly Bridge 2
    ("niche_puzzle_solo", 1444480): (3, ""),  # Turing Complete
    ("niche_puzzle_solo", 1910680): (2, ""),  # Orb of Creation
    ("niche_puzzle_solo", 2162800): (3, ""),  # shapez 2 - Factory
    ("niche_puzzle_solo", 3700980): (1, "GENRE_ONLY"),  # How to Make an Atomic Bomb in Your Garde
    ("niche_puzzle_solo", 3846120): (3, ""),  # MineMogul
    ("niche_roguelite", 753420): (3, ""),  # Dungreed
    ("niche_roguelite", 958520): (3, ""),  # 33 Immortals
    ("niche_roguelite", 1315180): (2, ""),  # Spark in the Dark
    ("niche_roguelite", 1721110): (3, ""),  # Abyssus
    ("niche_roguelite", 1887840): (3, ""),  # Another Crab's Treasure
    ("niche_roguelite", 2071280): (3, ""),  # Ravenswatch
    ("niche_roguelite", 2334730): (3, ""),  # Death Must Die
    ("niche_roguelite", 2351560): (3, ""),  # 아포칼립스 파티
    ("niche_roguelite", 2665680): (2, ""),  # 바벨탑: 혼돈의 생존자들
    ("niche_roguelite", 3228590): (3, ""),  # Deadzone: Rogue
    ("niche_roguelite", 3489700): (1, "GENRE_ONLY"),  # 스텔라 블레이드™
    ("niche_sim", 350640): (3, ""),  # Sherlock Holmes: The Devil's Daughter
    ("niche_sim", 383120): (2, ""),  # Empyrion - Galactic Survival
    ("niche_sim", 574720): (2, ""),  # Little Big Workshop
    ("niche_sim", 1366540): (3, ""),  # Dyson Sphere Program
    ("niche_sim", 1369700): (1, "GENRE_ONLY"),  # Solar Expanse - Space Exploration Manage
    ("niche_sim", 1614550): (3, ""),  # Astro Colony
    ("niche_sim", 1754840): (3, ""),  # Hacker Simulator
    ("niche_sim", 2779120): (3, ""),  # Modulus: Factory Automation
    ("niche_sim", 2797960): (3, ""),  # 은폐된 살인의 진실들 - 하드코어 본격 추리 탐정 게임
    ("niche_sim", 2879840): (3, ""),  # 방 탈출 시뮬레이터2
    ("niche_soulslike_solo", 236090): (3, ""),  # Dust: An Elysian Tail
    ("niche_soulslike_solo", 236430): (3, ""),  # DARK SOULS™ II
    ("niche_soulslike_solo", 385380): (1, "GENRE_ONLY"),  # Planet Centauri
    ("niche_soulslike_solo", 747200): (1, "GENRE_ONLY"),  # Keplerth
    ("niche_soulslike_solo", 1369630): (3, ""),  # ENDER LILIES: Quietus of the Knights
    ("niche_soulslike_solo", 1863430): (2, ""),  # Dragonkin: The Banished
    ("niche_soulslike_solo", 2317640): (1, "GENRE_ONLY"),  # JUMP KING QUEST
    ("niche_soulslike_solo", 3216340): (3, ""),  # Tearscape
    ("niche_soulslike_solo", 3418990): (2, ""),  # 신역 추락
    ("niche_tactics", 496620): (3, ""),  # Monster Slayers
    ("niche_tactics", 637090): (3, ""),  # BATTLETECH
    ("niche_tactics", 673880): (3, ""),  # Warhammer 40,000: Mechanicus
    ("niche_tactics", 861540): (3, ""),  # Dicey Dungeons
    ("niche_tactics", 2432860): (3, ""),  # MENACE 메너스
    ("niche_tactics", 3481020): (3, ""),  # 恶魔牌
    ("niche_tactics", 3709430): (2, ""),  # 마녀: 종말의 여행
}


# --- 라운드4 블라인드: 태그 코퍼스에서 리뷰 하한 스윕 ---
# 코퍼스를 21,883(리뷰 101+)으로 넓히고 하한 0/300/1000 을 비교. 합집합 미판정이 24쌍뿐이었다 —
# 태그가 붙으니 하한을 풀어도 Top-10 이 거의 안 바뀐다는 뜻이다.
BLIND_FLOOR = {
    ("coh_cozy", 674140): (3, ""),  # Bugsnax
    ("coh_grand_strategy", 209670): (1, "GENRE_ONLY"),  # Cortex Command
    ("coh_grand_strategy", 2050680): (1, "GENRE_ONLY"),  # Warbox Sandbox
    ("coh_indie_platformer", 2474430): (3, ""),  # TetherGeist
    ("coh_openworld_survival", 244770): (2, ""),  # StarMade
    ("coh_openworld_survival", 450860): (1, "TOO_NICHE"),  # Andarilho
    ("coh_strategy", 2021880): (3, ""),  # Ara History Untold: Anniversary Edition
    ("coh_survival_craft", 420930): (2, ""),  # CHKN
    ("coh_vehicle_sim", 1089830): (2, ""),  # Monster Energy Supercross - The Official
    ("coh_vehicle_sim", 1114150): (3, ""),  # CarX Street
    ("coh_vehicle_sim", 1364690): (1, "GENRE_ONLY"),  # First Racer
    ("mix_grand_casual", 209670): (1, "GENRE_ONLY"),  # Cortex Command
    ("mix_grand_casual", 2050680): (1, "GENRE_ONLY"),  # Warbox Sandbox
    ("mix_indie_multi", 916730): (3, ""),  # Gato Roboto
    ("mix_openworld_cozy", 314790): (0, "IRRELEVANT"),  # Silence
    ("mix_rpg_racing", 321800): (2, ""),  # Icewind Dale: Enhanced Edition
    ("mix_rpg_racing", 1114150): (3, ""),  # CarX Street
    ("niche_puzzle_solo", 1260520): (3, ""),  # Patrick's Parabox
    ("niche_puzzle_solo", 1577620): (3, ""),  # The Signal State
    ("niche_soulslike_solo", 252030): (3, ""),  # Valdis Story: Abyssal City
    ("niche_soulslike_solo", 1264880): (3, ""),  # Watcher Chronicles
    ("niche_soulslike_solo", 1456650): (3, ""),  # 파스칼 웨이저: 완전판
    ("niche_tactics", 355680): (2, ""),  # Overland
    ("niche_tactics", 981430): (3, ""),  # Gordian Quest
}


# --- 라운드5 블라인드: 저리뷰 축 5개 프로필 (리뷰 850~1,973 시드) ---
# 코퍼스의 70% 를 차지하는데 시드가 하나도 없던 구간. 무명 시드라 시트에 **시드 설명**을
# 함께 넣었다 — 이름만으로는 어떤 게임인지 알 수 없고, 모르는 채로 채점하면
# "설명끼리 닮았나"만 보게 되어 임베딩과 순환한다.
BLIND_LOWREV = {
    ("lowrev_cozy_narrative", 232430): (3, ""),  # Gone Home
    ("lowrev_cozy_narrative", 331870): (2, ""),  # AER Memories of Old
    ("lowrev_cozy_narrative", 520720): (3, ""),  # Dear Esther: Landmark Edition
    ("lowrev_cozy_narrative", 638230): (3, ""),  # Journey
    ("lowrev_cozy_narrative", 695330): (3, ""),  # SEASON: A letter to the future
    ("lowrev_cozy_narrative", 858940): (3, ""),  # Flowers -Le volume sur ete-
    ("lowrev_cozy_narrative", 1281270): (2, ""),  # Fatum Betula
    ("lowrev_cozy_narrative", 1506980): (2, ""),  # 葬花·暗黑桃花源
    ("lowrev_cozy_narrative", 3069120): (2, ""),  # 러브커스: 사랑이 아니면 죽음뿐
    ("lowrev_cozy_narrative", 3528450): (2, ""),  # 종이집
    ("lowrev_deckbuilder", 646570): (3, ""),  # Slay the Spire
    ("lowrev_deckbuilder", 981430): (3, ""),  # Gordian Quest
    ("lowrev_deckbuilder", 1076200): (3, ""),  # Roguebook
    ("lowrev_deckbuilder", 1638390): (3, ""),  # Indies' Lies
    ("lowrev_deckbuilder", 1755830): (3, ""),  # Astrea: Six-Sided Oracles
    ("lowrev_deckbuilder", 1815570): (3, ""),  # Aces & Adventures
    ("lowrev_deckbuilder", 2026820): (3, ""),  # Die in the Dungeon
    ("lowrev_deckbuilder", 2693930): (3, ""),  # 주사위와 마왕의 성
    ("lowrev_deckbuilder", 2842800): (2, ""),  # 尸姬之梦
    ("lowrev_deckbuilder", 2870340): (2, ""),  # Decktamer
    ("lowrev_detective", 284770): (2, ""),  # Enigmatis 2: The Mists of Ravenwood
    ("lowrev_detective", 350640): (3, ""),  # Sherlock Holmes: The Devil's Daughter
    ("lowrev_detective", 368370): (3, ""),  # Her Story
    ("lowrev_detective", 373390): (3, ""),  # Contradiction: Spot The Liar
    ("lowrev_detective", 615770): (3, ""),  # Nancy Drew®: Message in a Haunted Mansio
    ("lowrev_detective", 712730): (2, ""),  # SIMULACRA
    ("lowrev_detective", 1271300): (3, ""),  # Methods: The Detective Competition
    ("lowrev_detective", 1466390): (3, ""),  # Kathy Rain 2: Soothsayer
    ("lowrev_detective", 2450840): (0, "IRRELEVANT"),  # Detective Dotson
    ("lowrev_detective", 2514960): (1, "GENRE_ONLY"),  # Refind Self: 성격 진단 게임
    ("lowrev_metroidvania", 332200): (3, ""),  # Axiom Verge
    ("lowrev_metroidvania", 345820): (3, ""),  # Shantae and the Pirate's Curse
    ("lowrev_metroidvania", 598700): (1, "GENRE_ONLY"),  # The Vagrant
    ("lowrev_metroidvania", 813230): (3, ""),  # ANIMAL WELL
    ("lowrev_metroidvania", 1379870): (1, "GENRE_ONLY"),  # Tribal Hunter
    ("lowrev_metroidvania", 1419160): (3, ""),  # Souldiers
    ("lowrev_metroidvania", 1517970): (3, ""),  # Aeterna Noctis
    ("lowrev_metroidvania", 1522870): (1, "FRANCHISE_OR_VARIANT"),  # Supraland Six Inches Under
    ("lowrev_metroidvania", 1748620): (1, "LOW_QUALITY"),  # FlipWitch - Forbidden Sex Hex
    ("lowrev_metroidvania", 2971610): (3, ""),  # HOLE
    ("lowrev_towerdefense", 23530): (1, "GENRE_ONLY"),  # Earth Defense Force: Insect Armageddon
    ("lowrev_towerdefense", 408410): (3, ""),  # X-Morph: Defense
    ("lowrev_towerdefense", 422900): (3, ""),  # Particle Fleet: Emergence
    ("lowrev_towerdefense", 458710): (3, ""),  # Kingdom Rush Frontiers - Tower Defense
    ("lowrev_towerdefense", 603320): (3, ""),  # Age of Defense
    ("lowrev_towerdefense", 701870): (2, ""),  # Swarm Queen
    ("lowrev_towerdefense", 848480): (2, ""),  # Creeper World 4
    ("lowrev_towerdefense", 1522820): (3, ""),  # Orcs Must Die! 3
    ("lowrev_towerdefense", 1566690): (2, ""),  # Outpost: Infinity Siege
    ("lowrev_towerdefense", 2607060): (3, ""),  # From Glory To Goo
}


ALL = {**REP_V2, **POSTPROCESS, **DEPTH_V1, **FULL_V1, **REP_V2_TOPUP, **MIN_REV_300,
       **RANKER_DIAG, **BOOST_SWEEP, **VAL_V1, **DEV_TOPUP_V1, **NICHE_V1,
       **BLIND_ROUND1, **BLIND_TAGS, **BLIND_FLOOR, **BLIND_LOWREV}
