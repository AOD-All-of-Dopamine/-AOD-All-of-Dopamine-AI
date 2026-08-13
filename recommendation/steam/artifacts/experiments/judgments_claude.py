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


# --- 라운드6 블라인드: 태그 **없는** 전체 코퍼스(full_v1, 173,691)의 31프로필 Top-10 ---
# 저리뷰 5개 프로필에서 태그 유무가 후보를 크게 갈라 37칸이 새로 생겼다.
# 태그 임베딩이 끝나면 같은 코퍼스 크기·같은 프로필로 태그 유무만 비교하기 위한 기준선이다.
BLIND_NOTAGS31 = {
    ("lowrev_cozy_narrative", 857690): (1, "GENRE_ONLY"),  # Lagoon Lounge : The Poisonous Fountain
    ("lowrev_cozy_narrative", 1372320): (2, ""),  # Cloud Gardens
    ("lowrev_cozy_narrative", 1497360): (3, ""),  # Open Roads
    ("lowrev_cozy_narrative", 1629830): (3, ""),  # Research Story
    ("lowrev_cozy_narrative", 1817940): (1, "GENRE_ONLY"),  # 러브 딜리버리
    ("lowrev_cozy_narrative", 1883090): (2, ""),  # The Symbiant
    ("lowrev_cozy_narrative", 3558600): (2, ""),  # Leafy Corner
    ("lowrev_cozy_narrative", 3910610): (1, "GENRE_ONLY"),  # Growing Things Up - Season 1
    ("lowrev_cozy_narrative", 3924170): (3, ""),  # Eternal Afternoon
    ("lowrev_deckbuilder", 493080): (3, ""),  # Card Quest
    ("lowrev_deckbuilder", 691690): (2, ""),  # Ash of Gods: Redemption
    ("lowrev_deckbuilder", 770100): (2, ""),  # One Deck Dungeon
    ("lowrev_deckbuilder", 804010): (3, ""),  # SteamWorld Quest: Hand of Gilgamech
    ("lowrev_deckbuilder", 1552080): (3, ""),  # Deepest Chamber: Resurrection
    ("lowrev_deckbuilder", 2727210): (3, ""),  # Deathless. The Hero Quest
    ("lowrev_detective", 42210): (3, ""),  # Nancy Drew®: Secrets Can Kill REMASTERED
    ("lowrev_detective", 604450): (0, "IRRELEVANT"),  # Another Adventure
    ("lowrev_detective", 963000): (3, ""),  # Frog Detective 1: The Haunted Island
    ("lowrev_detective", 1241510): (0, "IRRELEVANT"),  # The Test
    ("lowrev_detective", 1677770): (3, ""),  # The Case of the Golden Idol
    ("lowrev_detective", 1723260): (3, ""),  # CaseCracker
    ("lowrev_detective", 1835840): (1, "GENRE_ONLY"),  # Quarantineer
    ("lowrev_detective", 2507620): (0, "IRRELEVANT"),  # The Quintessential Quintuplets - Five Me
    ("lowrev_detective", 2861150): (0, "IRRELEVANT"),  # Working Sakuya
    ("lowrev_metroidvania", 371550): (1, "GENRE_ONLY"),  # A Bastard's Tale
    ("lowrev_metroidvania", 489140): (1, "GENRE_ONLY"),  # Mr. Shifty
    ("lowrev_metroidvania", 1068360): (3, ""),  # Fearmonium
    ("lowrev_metroidvania", 1733250): (3, ""),  # Turbo Kid
    ("lowrev_metroidvania", 2023360): (3, ""),  # The Last Case of Benedict Fox Definitive
    ("lowrev_metroidvania", 3280350): (1, "GENRE_ONLY"),  # DEATH STRANDING 2: ON THE BEACH
    ("lowrev_towerdefense", 4920): (1, "GENRE_ONLY"),  # Natural Selection 2
    ("lowrev_towerdefense", 6810): (1, "GENRE_ONLY"),  # Commandos: Beyond the Call of Duty
    ("lowrev_towerdefense", 48190): (0, "IRRELEVANT"),  # Assassin’s Creed® Brotherhood
    ("lowrev_towerdefense", 332200): (0, "IRRELEVANT"),  # Axiom Verge
    ("lowrev_towerdefense", 1692240): (0, "IRRELEVANT"),  # Fortune's Run
    ("lowrev_towerdefense", 2392280): (3, ""),  # TDS - Tower Defense Strategy
    ("lowrev_towerdefense", 3226530): (3, ""),  # Tower Dominion
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


# --- 라운드6 블라인드: 태그 **없는** 전체 코퍼스(full_v1, 173,691)의 31프로필 Top-10 ---
# 저리뷰 5개 프로필에서 태그 유무가 후보를 크게 갈라 37칸이 새로 생겼다.
# 태그 임베딩이 끝나면 같은 코퍼스 크기·같은 프로필로 태그 유무만 비교하기 위한 기준선이다.
BLIND_NOTAGS31 = {
    ("lowrev_cozy_narrative", 857690): (1, "GENRE_ONLY"),  # Lagoon Lounge : The Poisonous Fountain
    ("lowrev_cozy_narrative", 1372320): (2, ""),  # Cloud Gardens
    ("lowrev_cozy_narrative", 1497360): (3, ""),  # Open Roads
    ("lowrev_cozy_narrative", 1629830): (3, ""),  # Research Story
    ("lowrev_cozy_narrative", 1817940): (1, "GENRE_ONLY"),  # 러브 딜리버리
    ("lowrev_cozy_narrative", 1883090): (2, ""),  # The Symbiant
    ("lowrev_cozy_narrative", 3558600): (2, ""),  # Leafy Corner
    ("lowrev_cozy_narrative", 3910610): (1, "GENRE_ONLY"),  # Growing Things Up - Season 1
    ("lowrev_cozy_narrative", 3924170): (3, ""),  # Eternal Afternoon
    ("lowrev_deckbuilder", 493080): (3, ""),  # Card Quest
    ("lowrev_deckbuilder", 691690): (2, ""),  # Ash of Gods: Redemption
    ("lowrev_deckbuilder", 770100): (2, ""),  # One Deck Dungeon
    ("lowrev_deckbuilder", 804010): (3, ""),  # SteamWorld Quest: Hand of Gilgamech
    ("lowrev_deckbuilder", 1552080): (3, ""),  # Deepest Chamber: Resurrection
    ("lowrev_deckbuilder", 2727210): (3, ""),  # Deathless. The Hero Quest
    ("lowrev_detective", 42210): (3, ""),  # Nancy Drew®: Secrets Can Kill REMASTERED
    ("lowrev_detective", 604450): (0, "IRRELEVANT"),  # Another Adventure
    ("lowrev_detective", 963000): (3, ""),  # Frog Detective 1: The Haunted Island
    ("lowrev_detective", 1241510): (0, "IRRELEVANT"),  # The Test
    ("lowrev_detective", 1677770): (3, ""),  # The Case of the Golden Idol
    ("lowrev_detective", 1723260): (3, ""),  # CaseCracker
    ("lowrev_detective", 1835840): (1, "GENRE_ONLY"),  # Quarantineer
    ("lowrev_detective", 2507620): (0, "IRRELEVANT"),  # The Quintessential Quintuplets - Five Me
    ("lowrev_detective", 2861150): (0, "IRRELEVANT"),  # Working Sakuya
    ("lowrev_metroidvania", 371550): (1, "GENRE_ONLY"),  # A Bastard's Tale
    ("lowrev_metroidvania", 489140): (1, "GENRE_ONLY"),  # Mr. Shifty
    ("lowrev_metroidvania", 1068360): (3, ""),  # Fearmonium
    ("lowrev_metroidvania", 1733250): (3, ""),  # Turbo Kid
    ("lowrev_metroidvania", 2023360): (3, ""),  # The Last Case of Benedict Fox Definitive
    ("lowrev_metroidvania", 3280350): (1, "GENRE_ONLY"),  # DEATH STRANDING 2: ON THE BEACH
    ("lowrev_towerdefense", 4920): (1, "GENRE_ONLY"),  # Natural Selection 2
    ("lowrev_towerdefense", 6810): (1, "GENRE_ONLY"),  # Commandos: Beyond the Call of Duty
    ("lowrev_towerdefense", 48190): (0, "IRRELEVANT"),  # Assassin’s Creed® Brotherhood
    ("lowrev_towerdefense", 332200): (0, "IRRELEVANT"),  # Axiom Verge
    ("lowrev_towerdefense", 1692240): (0, "IRRELEVANT"),  # Fortune's Run
    ("lowrev_towerdefense", 2392280): (3, ""),  # TDS - Tower Defense Strategy
    ("lowrev_towerdefense", 3226530): (3, ""),  # Tower Dominion
}


BLIND_TAGSFULL = {
    ("coh_arpg", 384570): (1, "GENRE_ONLY"),  # Zanzarah: The Hidden Portal
    ("coh_arpg", 1997660): (3, ""),  # GreedFall: The Dying World
    ("coh_arpg", 2516470): (1, "LOW_QUALITY"),  # Weird RPG 2
    ("coh_classic_multi", 224260): (3, ""),  # No More Room in Hell
    ("coh_classic_multi", 278970): (1, "LOW_QUALITY"),  # Digger Online
    ("coh_cozy", 1698220): (0, "IRRELEVANT"),  # Teslagrad 2
    ("coh_fps", 42680): (2, ""),  # Call of Duty®: Modern Warfare® 3 (2011)
    ("coh_fps", 333930): (3, ""),  # Dirty Bomb®
    ("coh_grand_strategy", 42810): (3, ""),  # For The Glory: A Europa Universalis Game
    ("coh_grand_strategy", 46770): (3, ""),  # Making History II: The War of the World
    ("coh_grand_strategy", 280540): (1, "GENRE_ONLY"),  # GhostControl Inc.
    ("coh_grand_strategy", 370540): (3, ""),  # Gary Grigsby's War in the East
    ("coh_grand_strategy", 371190): (3, ""),  # Making History: The Calm and the Storm G
    ("coh_grand_strategy", 1267210): (1, "GENRE_ONLY"),  # Together in Battle
    ("coh_indie_platformer", 320820): (1, "GENRE_ONLY"),  # Toren
    ("coh_indie_platformer", 1720090): (2, ""),  # Aspire: Ina's Tale
    ("coh_indie_platformer", 2014550): (3, ""),  # Voidwrought
    ("coh_openworld_survival", 1304350): (2, ""),  # Urge
    ("coh_openworld_survival", 2075580): (3, ""),  # Under A New Sun
    ("coh_strategy", 1183470): (3, ""),  # Imperiums: Greek Wars
    ("coh_strategy", 1561960): (2, ""),  # Yield! Fall of Rome
    ("coh_survival_craft", 450860): (1, "LOW_QUALITY"),  # Andarilho
    ("coh_survival_craft", 1304350): (3, ""),  # Urge
    ("coh_survival_craft", 2383130): (3, ""),  # Project: Mist
    ("coh_vehicle_sim", 1305080): (2, ""),  # GearBlocks
    ("lowrev_cozy_narrative", 331470): (1, "MODE_MISMATCH"),  # Everlasting Summer
    ("lowrev_cozy_narrative", 363410): (3, ""),  # Always The Same Blue Sky...
    ("lowrev_cozy_narrative", 768450): (3, ""),  # NUTS
    ("lowrev_cozy_narrative", 822760): (1, "MODE_MISMATCH"),  # Fureraba ~Friend to Lover~
    ("lowrev_cozy_narrative", 844670): (3, ""),  # Please Be Happy
    ("lowrev_cozy_narrative", 972160): (2, ""),  # The Witch's Love Diary
    ("lowrev_cozy_narrative", 1497450): (3, ""),  # A Memoir Blue
    ("lowrev_cozy_narrative", 1813860): (3, ""),  # NAIAD
    ("lowrev_cozy_narrative", 2852930): (1, "LOW_QUALITY"),  # Ethel
    ("lowrev_deckbuilder", 536040): (3, ""),  # Cards of Cthulhu
    ("lowrev_deckbuilder", 930780): (3, ""),  # Blood Card
    ("lowrev_deckbuilder", 2140850): (3, ""),  # Looper Tactics
    ("lowrev_deckbuilder", 3035330): (3, ""),  # 주사위 아발론
    ("lowrev_deckbuilder", 3057670): (3, ""),  # Pluto
    ("lowrev_deckbuilder", 3100370): (3, ""),  # Black Jacket
    ("lowrev_deckbuilder", 3542380): (3, ""),  # 룬 다이스
    ("lowrev_detective", 37600): (0, "IRRELEVANT"),  # Windosill
    ("lowrev_detective", 392970): (2, ""),  # New York Mysteries: High Voltage Collect
    ("lowrev_detective", 394680): (1, "LOW_QUALITY"),  # Mountain Crime: Requital
    ("lowrev_detective", 1140290): (3, ""),  # Murder by Numbers
    ("lowrev_detective", 1421790): (2, ""),  # 미로 대탐정
    ("lowrev_detective", 1497360): (2, ""),  # Open Roads
    ("lowrev_detective", 1689870): (2, ""),  # Meridian 157: Chapter 3
    ("lowrev_detective", 3296830): (3, ""),  # CaseCracker3
    ("lowrev_metroidvania", 376730): (1, "GENRE_ONLY"),  # Monument
    ("lowrev_metroidvania", 565860): (0, "LOW_QUALITY"),  # Tomato Way
    ("lowrev_metroidvania", 606960): (1, "GENRE_ONLY"),  # CRIMSON METAL Classic 1999
    ("lowrev_metroidvania", 960190): (2, ""),  # Maytroid. I swear it's a nice game too
    ("lowrev_metroidvania", 1096180): (3, ""),  # Ato
    ("lowrev_metroidvania", 1435970): (1, "GENRE_ONLY"),  # Anomalous
    ("lowrev_metroidvania", 1586800): (1, "GENRE_ONLY"),  # Lil Gator Game
    ("lowrev_metroidvania", 1853760): (3, ""),  # Frogmonster
    ("lowrev_metroidvania", 2004590): (1, "GENRE_ONLY"),  # Conbunn Cardboard
    ("lowrev_metroidvania", 3081830): (2, ""),  # FLARE NUINUI QUEST
    ("lowrev_towerdefense", 402160): (1, "GENRE_ONLY"),  # Star Command Galaxies
    ("lowrev_towerdefense", 531680): (1, "GENRE_ONLY"),  # 8-Bit Invaders!
    ("lowrev_towerdefense", 1035510): (2, ""),  # Ultimate Zombie Defense
    ("lowrev_towerdefense", 1601330): (2, ""),  # Survival Machine
    ("lowrev_towerdefense", 1932260): (3, ""),  # No Creeps Were Harmed TD
    ("lowrev_towerdefense", 2050510): (2, ""),  # 东方王朝-丝路保卫战
    ("mix_arpg_survival", 214730): (1, "GENRE_ONLY"),  # Space Rangers HD: A War Apart
    ("mix_arpg_survival", 1135260): (2, ""),  # The Falconeer: Revolution Remaster
    ("mix_arpg_survival", 1304350): (2, ""),  # Urge
    ("mix_arpg_survival", 2383130): (3, ""),  # Project: Mist
    ("mix_grand_casual", 280540): (2, ""),  # GhostControl Inc.
    ("mix_grand_casual", 1267210): (2, ""),  # Together in Battle
    ("mix_indie_multi", 278970): (1, "LOW_QUALITY"),  # Digger Online
    ("mix_indie_multi", 2014550): (3, ""),  # Voidwrought
    ("mix_multi_indie", 224260): (3, ""),  # No More Room in Hell
    ("mix_multi_indie", 280520): (3, ""),  # Crea
    ("mix_openworld_cozy", 406870): (0, "IRRELEVANT"),  # Eventide: Slavic Fable
    ("mix_openworld_cozy", 1698220): (1, "IRRELEVANT"),  # Teslagrad 2
    ("mix_openworld_cozy", 2075580): (3, ""),  # Under A New Sun
    ("mix_rpg_racing", 1305080): (2, ""),  # GearBlocks
    ("mix_rpg_racing", 2516470): (1, "LOW_QUALITY"),  # Weird RPG 2
    ("mix_survival_strategy", 1304350): (2, ""),  # Urge
    ("mix_survival_strategy", 2383130): (3, ""),  # Project: Mist
    ("mix_vehicle_fps", 224260): (3, ""),  # No More Room in Hell
    ("niche_cozy_casual", 105800): (2, ""),  # PixelJunk™ Eden
    ("niche_cozy_casual", 1008920): (2, ""),  # Photographs
    ("niche_cozy_casual", 2107550): (3, ""),  # Valley Peaks
    ("niche_puzzle_solo", 684270): (3, ""),  # Silicon Zeroes
    ("niche_puzzle_solo", 693700): (3, ""),  # Prime Mover
    ("niche_puzzle_solo", 1257850): (2, ""),  # SOLAS 128
    ("niche_puzzle_solo", 1542810): (3, ""),  # Sunshine Heavy Industries
    ("niche_puzzle_solo", 2262930): (1, "GENRE_ONLY"),  # Bombe
    ("niche_puzzle_solo", 2275490): (3, ""),  # 카이젠: 팩토리 스토리
    ("niche_puzzle_solo", 2870420): (2, ""),  # Mech Builder
    ("niche_roguelite", 1933000): (2, ""),  # Luna Abyss
    ("niche_roguelite", 2002220): (3, ""),  # Keeper's Toll
    ("niche_roguelite", 2443090): (2, ""),  # Deathless: Survivors
    ("niche_roguelite", 2803280): (3, ""),  # Dragon Is Dead
    ("niche_roguelite", 3773590): (3, ""),  # Death by Scrolling
    ("niche_sim", 330460): (3, ""),  # Celestial Command
    ("niche_sim", 1383150): (3, ""),  # Final Factory
    ("niche_soulslike_solo", 540100): (1, "LOW_QUALITY"),  # Imprisoned Light
    ("niche_soulslike_solo", 727510): (3, ""),  # Void Memory
    ("niche_soulslike_solo", 1517970): (3, ""),  # Aeterna Noctis
    ("niche_tactics", 1021070): (2, ""),  # Spaceland: Sci-Fi Indie Tactics
    ("niche_tactics", 1156060): (3, ""),  # Railroads & Catacombs
    ("niche_tactics", 2646720): (3, ""),  # Dead Weight
}


BLIND_K30_P1 = {
    ("coh_arpg", 391660): (1, "LOW_QUALITY"),  # Incitement 3
    ("coh_cozy", 2220690): (0, "KEYWORD_MATCH"),  # Life of Slime
    ("coh_cozy", 3022510): (0, "KEYWORD_MATCH"),  # Slime Scramble
    ("coh_cozy", 3771260): (0, "KEYWORD_MATCH"),  # XP Slime
    ("coh_grand_strategy", 2400080): (0, "IRRELEVANT"),  # Scenarios Of The Ancient Realm
    ("longtail_deckbuilder", 981430): (3, ""),  # Gordian Quest
    ("longtail_deckbuilder", 1076750): (2, ""),  # Dream Engines: Nomad Cities
    ("longtail_deckbuilder", 1142080): (2, ""),  # Pawnbarian
    ("longtail_deckbuilder", 1201830): (3, ""),  # For The Warp
    ("longtail_deckbuilder", 1638390): (3, ""),  # Indies' Lies
    ("longtail_deckbuilder", 1840710): (3, ""),  # These Doomed Isles
    ("longtail_deckbuilder", 1972440): (2, ""),  # Shotgun King: The Final Checkmate
    ("longtail_deckbuilder", 2339040): (3, ""),  # Landnama
    ("longtail_deckbuilder", 2468100): (3, ""),  # Pyrene
    ("longtail_deckbuilder", 2622820): (3, ""),  # Dobbel Dungeon
    ("longtail_deckbuilder", 2870340): (3, ""),  # Decktamer
    ("longtail_deckbuilder", 3415570): (2, ""),  # Die For The Lich
    ("longtail_detective", 11040): (3, ""),  # Sherlock Holmes - Nemesis
    ("longtail_detective", 42210): (3, ""),  # Nancy Drew®: Secrets Can Kill REMASTERED
    ("longtail_detective", 42220): (3, ""),  # Nancy Drew®: The Captive Curse
    ("longtail_detective", 46480): (3, ""),  # Still Life
    ("longtail_detective", 46500): (3, ""),  # Syberia
    ("longtail_detective", 210950): (1, "GENRE_ONLY"),  # Rune Classic
    ("longtail_detective", 233370): (3, ""),  # The Raven - Legacy of a Master Thief
    ("longtail_detective", 284770): (2, ""),  # Enigmatis 2: The Mists of Ravenwood
    ("longtail_detective", 714250): (1, "GENRE_ONLY"),  # Eternity: The Last Unicorn
    ("longtail_detective", 1435790): (3, ""),  # 방 탈출 시뮬레이터
    ("longtail_detective", 1943950): (1, "MODE_MISMATCH"),  # Escape the Backrooms
    ("longtail_detective", 2264140): (3, ""),  # How 2 Escape
    ("longtail_metroidvania", 115800): (2, ""),  # Owlboy
    ("longtail_metroidvania", 367520): (3, ""),  # Hollow Knight
    ("longtail_metroidvania", 791180): (1, "GENRE_ONLY"),  # 1 Screen Platformer
    ("longtail_metroidvania", 946030): (3, ""),  # Axiom Verge 2
    ("longtail_metroidvania", 1096180): (3, ""),  # Ato
    ("longtail_metroidvania", 1252830): (2, ""),  # A Juggler's Tale
    ("longtail_metroidvania", 1368410): (3, ""),  # PSYCRON
    ("longtail_metroidvania", 1374970): (3, ""),  # Moonscars
    ("longtail_metroidvania", 1413660): (3, ""),  # Elderand
    ("longtail_metroidvania", 1419160): (3, ""),  # Souldiers
    ("longtail_metroidvania", 1522870): (3, ""),  # Supraland Six Inches Under
    ("longtail_metroidvania", 1588380): (3, ""),  # Blaster Master Zero 3
    ("longtail_metroidvania", 3288460): (3, ""),  # Rolling Star
    ("longtail_puzzle_platformer", 200190): (3, ""),  # Sideway™ New York
    ("longtail_puzzle_platformer", 215510): (2, ""),  # Rocketbirds: Hardboiled Chicken
    ("longtail_puzzle_platformer", 355150): (3, ""),  # gravilon
    ("longtail_puzzle_platformer", 363670): (3, ""),  # Seeders
    ("longtail_puzzle_platformer", 395160): (3, ""),  # Toby: The Secret Mine
    ("longtail_puzzle_platformer", 396710): (2, ""),  # The Adventure Pals
    ("longtail_puzzle_platformer", 532320): (2, ""),  # de Blob
    ("longtail_puzzle_platformer", 568320): (1, "GENRE_ONLY"),  # Pictopix
    ("longtail_puzzle_platformer", 888630): (2, ""),  # Another Sight - Definitive Edition
    ("longtail_puzzle_platformer", 1158890): (3, ""),  # White Shadows
    ("longtail_puzzle_platformer", 1224020): (2, ""),  # Creepy Tale
    ("longtail_puzzle_platformer", 1740300): (2, ""),  # Smushi Come Home
    ("longtail_puzzle_platformer", 2115300): (3, ""),  # Chroma Key
    ("longtail_puzzle_platformer", 3081690): (3, ""),  # Chromatic Conundrum
    ("lowrev_cozy_narrative", 3283310): (3, ""),  # Kioku: Last Summer
    ("lowrev_cozy_narrative", 3921630): (3, ""),  # Plantiquarian
    ("lowrev_detective", 2659650): (3, ""),  # Case Files: Behind Closed Doors
    ("lowrev_towerdefense", 1549870): (2, ""),  # Human Defense [RTS]
    ("mix2_arcade_action", 310950): (3, ""),  # Street Fighter V
    ("mix2_arcade_action", 744060): (2, ""),  # Groove Coaster
    ("mix2_arcade_action", 767390): (2, ""),  # Dakar 18
    ("mix2_arcade_action", 1114150): (3, ""),  # CarX Street
    ("mix2_arcade_action", 1369670): (2, ""),  # Motor Town: Behind The Wheel
    ("mix2_arcade_action", 1475840): (3, ""),  # Rhythm Sprout: Sick Beats & Bad Sweets
    ("mix2_arcade_action", 1555150): (3, ""),  # Pocket Bravery
    ("mix2_arcade_action", 2625420): (2, ""),  # Drive Beyond Horizons
    ("mix2_arcade_action", 2717010): (2, ""),  # 칼파: 코스믹 심포니(KALPA: Cosmic Symphony)
    ("mix2_arcade_action", 4144680): (3, ""),  # DEAD OR ALIVE 6 Last Round
    ("mix2_builder_sim", 220260): (3, ""),  # Farming Simulator 2013 Titanium Edition
    ("mix2_builder_sim", 313010): (3, ""),  # Cities XXL
    ("mix2_builder_sim", 405710): (2, ""),  # Staxel
    ("mix2_builder_sim", 526870): (3, ""),  # Satisfactory
    ("mix2_builder_sim", 949230): (3, ""),  # Cities: Skylines II
    ("mix2_builder_sim", 983870): (3, ""),  # FOUNDRY
    ("mix2_builder_sim", 1119730): (3, ""),  # Ranch Simulator: Build, Hunt, Farm
    ("mix2_builder_sim", 1489970): (3, ""),  # Highrise City
    ("mix2_builder_sim", 2162800): (3, ""),  # shapez 2 - Factory
    ("mix2_builder_sim", 2220850): (3, ""),  # Factor Y
    ("mix2_builder_sim", 2418520): (2, ""),  # Farm Together 2
    ("mix2_colony_cozy", 352430): (3, ""),  # Farlight Explorers
    ("mix2_colony_cozy", 455980): (3, ""),  # Judgment: Apocalypse Survival Simulation
    ("mix2_colony_cozy", 484900): (3, ""),  # Aven Colony
    ("mix2_colony_cozy", 979110): (3, ""),  # Space Haven
    ("mix2_colony_cozy", 1511460): (2, ""),  # InfraSpace
    ("mix2_colony_cozy", 1638300): (3, ""),  # Solargene
    ("mix2_colony_cozy", 2736840): (3, ""),  # The August Before
    ("mix2_colony_cozy", 2843190): (3, ""),  # Camper Van: Make it Home
    ("mix2_colony_cozy", 3017730): (3, ""),  # Unbox the Room
    ("mix2_colony_cozy", 3105960): (1, "GENRE_ONLY"),  # Astrobuilder
    ("mix2_colony_cozy", 3415240): (3, ""),  # 꼬마 정리수납전문가
    ("mix2_colony_cozy", 3730100): (0, "IRRELEVANT"),  # Whispers from the Star
    ("mix2_coop_horror", 503350): (2, ""),  # White Noise 2
    ("mix2_coop_horror", 1302240): (3, ""),  # Labyrinthine
    ("mix2_coop_horror", 1637320): (2, ""),  # Dome Keeper 돔 키퍼
    ("mix2_coop_horror", 1714900): (2, ""),  # First Dwarf
    ("mix2_coop_horror", 1774770): (3, ""),  # Phantom Hysteria
    ("mix2_coop_horror", 1985420): (3, ""),  # This is a Ghost
    ("mix2_coop_horror", 2309400): (3, ""),  # The Devourer: Hunted Souls
    ("mix2_coop_horror", 2605790): (3, ""),  # Deep Rock Galactic: Rogue Core
    ("mix2_coop_horror", 2780470): (3, ""),  # Bunker Invaders
    ("mix2_coop_horror", 2823080): (3, ""),  # Sketchy's Contract
    ("mix2_coop_horror", 2909110): (3, ""),  # Nuclear Nightmare
    ("mix2_coop_horror", 2960770): (3, ""),  # The Anomaly Project
    ("mix2_coop_horror", 3922920): (3, ""),  # PONE
    ("mix2_crpg_sandbox", 362960): (3, ""),  # Tyranny
    ("mix2_crpg_sandbox", 365360): (3, ""),  # Battle Brothers
    ("mix2_crpg_sandbox", 435150): (3, ""),  # Divinity: Original Sin 2 - Definitive Ed
    ("mix2_crpg_sandbox", 490280): (3, ""),  # Realms of Magic
    ("mix2_crpg_sandbox", 719040): (3, ""),  # Wasteland 3
    ("mix2_crpg_sandbox", 736820): (2, ""),  # Knights of Honor II: Sovereign
    ("mix2_crpg_sandbox", 1096530): (3, ""),  # Solasta: Crown of the Magister
    ("mix2_crpg_sandbox", 1289380): (2, ""),  # Sheltered 2
    ("mix2_crpg_sandbox", 1409830): (2, ""),  # Sons of Valhalla 선즈 오브 발할라
    ("mix2_crpg_sandbox", 1887020): (2, ""),  # Robber Knight
    ("mix2_crpg_sandbox", 2001070): (2, ""),  # Heart of the Machine 하트 오브 더 머신
    ("mix2_crpg_sandbox", 3410340): (2, ""),  # 绝境守护 Dao's Cover
    ("mix2_modern_roguelite", 646570): (3, ""),  # Slay the Spire
    ("mix2_modern_roguelite", 1145350): (3, ""),  # Hades II
    ("mix2_modern_roguelite", 1944570): (3, ""),  # Boneraiser Minions
    ("mix2_modern_roguelite", 1966900): (3, ""),  # 20 Minutes Till Dawn
    ("mix2_modern_roguelite", 2116260): (2, ""),  # Relic Dudes
    ("mix2_modern_roguelite", 2218750): (3, ""),  # Halls of Torment
    ("mix2_modern_roguelite", 2336730): (3, ""),  # Dungeon Survivors
    ("mix2_modern_roguelite", 2388460): (3, ""),  # Pathfinder: Gallowspire Survivors
    ("mix2_modern_roguelite", 2746910): (2, ""),  # 삼국 영웅：혈월의 전설
    ("mix2_modern_roguelite", 2820820): (2, ""),  # 자인언트 슬레이어: 저승의 물결
    ("mix2_modern_roguelite", 3100370): (3, ""),  # Black Jacket
    ("mix2_modern_roguelite", 3617620): (2, ""),  # My Card Is Better Than Your Card!
    ("mix2_modern_roguelite", 4224320): (3, ""),  # 패도리아
    ("mix2_party_narrative", 355970): (3, ""),  # Home is Where One Starts...
    ("mix2_party_narrative", 448510): (3, ""),  # Overcooked
    ("mix2_party_narrative", 592480): (3, ""),  # Knights And Bikes
    ("mix2_party_narrative", 1057750): (3, ""),  # The Suicide of Rachel Foster
    ("mix2_party_narrative", 1073900): (3, ""),  # Brukel
    ("mix2_party_narrative", 1136020): (3, ""),  # Cannibal Cuisine
    ("mix2_party_narrative", 1243830): (3, ""),  # Overcooked! All You Can Eat
    ("mix2_party_narrative", 2001120): (3, ""),  # Split Fiction
    ("mix2_party_narrative", 2455360): (2, ""),  # Cooking Simulator 2: Better Together
    ("mix2_party_narrative", 3263320): (3, ""),  # Carry The Glass
    ("mix2_party_narrative", 3295360): (3, ""),  # 연대 책임
    ("mix2_party_narrative", 3309750): (3, ""),  # As Long As You're Here
    ("mix2_party_narrative", 3453910): (2, ""),  # Happy's Humble Burger Cult
    ("mix2_party_narrative", 3621700): (2, ""),  # Try To Drive
    ("mix2_puzzle_survival", 346110): (3, ""),  # ARK: Survival Evolved
    ("mix2_puzzle_survival", 351290): (1, "LOW_QUALITY"),  # SURVIVAL: Postapocalypse Now
    ("mix2_puzzle_survival", 499520): (3, ""),  # The Turing Test
    ("mix2_puzzle_survival", 625340): (2, ""),  # XERA: Survival
    ("mix2_puzzle_survival", 835960): (3, ""),  # The Talos Principle 2
    ("mix2_puzzle_survival", 930780): (2, ""),  # Blood Card
    ("mix2_puzzle_survival", 1635450): (2, ""),  # Longvinter
    ("mix2_puzzle_survival", 3057670): (2, ""),  # Pluto
    ("mix2_puzzle_survival", 3100370): (2, ""),  # Black Jacket
    ("mix2_puzzle_survival", 3219410): (1, "TOO_NICHE"),  # 트롤리 미니게임
    ("mix2_soulslike_narrative", 241260): (3, ""),  # Sherlock Holmes: Crimes and Punishments
    ("mix2_soulslike_narrative", 275850): (2, ""),  # No Man's Sky
    ("mix2_soulslike_narrative", 374320): (3, ""),  # DARK SOULS™ III
    ("mix2_soulslike_narrative", 1145290): (2, ""),  # Out There: Oceans of Time
    ("mix2_soulslike_narrative", 1160220): (3, ""),  # Paradise Killer
    ("mix2_soulslike_narrative", 1449110): (3, ""),  # The Outer Worlds 2
    ("mix2_soulslike_narrative", 1456650): (3, ""),  # 파스칼 웨이저: 완전판
    ("mix2_soulslike_narrative", 1804470): (2, ""),  # Duel Corp.
    ("mix2_soulslike_narrative", 1933840): (1, "LOW_QUALITY"),  # Moon Mystery
    ("mix2_soulslike_narrative", 2863680): (3, ""),  # ZERO PARADES: For Dead Spies
    ("mix2_soulslike_narrative", 3282300): (2, ""),  # Mistfall Hunter
    ("mix2_soulslike_narrative", 3882670): (1, "LOW_QUALITY"),  # LUNATHREN
    ("mix2_survival_farm", 768200): (3, ""),  # Smalland: Survive the Wilds
    ("mix2_survival_farm", 824000): (3, ""),  # Hokko Life
    ("mix2_survival_farm", 877010): (2, ""),  # Beyond Contact
    ("mix2_survival_farm", 978780): (3, ""),  # STORY OF SEASONS: Friends of Mineral Tow
    ("mix2_survival_farm", 1169040): (3, ""),  # Necesse: 네세스
    ("mix2_survival_farm", 1504570): (2, ""),  # 太荒初境
    ("mix2_survival_farm", 1963370): (2, ""),  # No One Survived
    ("mix2_survival_farm", 2508780): (3, ""),  # 목장이야기 Let's! 바람의 그랜드 바자르
    ("mix2_survival_farm", 2661300): (3, ""),  # Grounded 2
    ("mix2_survival_farm", 2679100): (3, ""),  # Witchspire
    ("niche_cozy_casual", 205020): (3, ""),  # Lumino City
    ("niche_cozy_casual", 207420): (1, "IRRELEVANT"),  # Wizorb
    ("niche_cozy_casual", 646010): (3, ""),  # Golem
    ("niche_cozy_casual", 1285080): (2, ""),  # Gordian Rooms 1: A curious heritage
    ("niche_puzzle_solo", 1560450): (3, ""),  # the Sequence [2]
}


BLIND_K30_P2 = {
    ("coh_arpg", 1630): (1, "IRRELEVANT"),  # Disciples II: Rise of the Elves
    ("coh_arpg", 6300): (2, ""),  # Dreamfall: The Longest Journey
    ("coh_arpg", 1139160): (2, ""),  # Divided Reigns
    ("coh_arpg", 1371980): (3, ""),  # 노 레스트 포 더 위키드
    ("coh_arpg", 1448440): (3, ""),  # Wo Long: Fallen Dynasty
    ("coh_arpg", 1456650): (3, ""),  # 파스칼 웨이저: 완전판
    ("coh_arpg", 2025320): (3, ""),  # Estencel
    ("coh_classic_multi", 63200): (3, ""),  # Monday Night Combat
    ("coh_classic_multi", 355180): (3, ""),  # Codename CURE
    ("coh_classic_multi", 391460): (2, ""),  # WARMODE
    ("coh_classic_multi", 552100): (3, ""),  # Brick Rigs
    ("coh_classic_multi", 654850): (2, ""),  # Masked Forces: Zombie Survival
    ("coh_classic_multi", 665370): (2, ""),  # Mutilate-a-Doll 2
    ("coh_classic_multi", 746140): (2, ""),  # Walking Zombie: Shooter
    ("coh_classic_multi", 1132750): (3, ""),  # Buck Zombies
    ("coh_classic_multi", 1172470): (2, ""),  # Apex 레전드™
    ("coh_classic_multi", 1488560): (1, "IRRELEVANT"),  # Football Simulator
    ("coh_classic_multi", 2996520): (3, ""),  # Dummy Guyz
    ("coh_classic_multi", 3511270): (2, ""),  # StrikeNet
    ("coh_cozy", 786580): (3, ""),  # Gleaner Heights
    ("coh_cozy", 865680): (2, ""),  # The Eternal Cylinder
    ("coh_cozy", 1045430): (2, ""),  # Circadian City
    ("coh_cozy", 1416960): (3, ""),  # Everafter Falls
    ("coh_cozy", 1635450): (2, ""),  # Longvinter
    ("coh_cozy", 2134320): (0, "IRRELEVANT"),  # ENA: Dream BBQ
    ("coh_cozy", 2981850): (1, "KEYWORD_MATCH"),  # 슬라임 공주
    ("coh_cozy", 4499040): (1, "KEYWORD_MATCH"),  # 슬라임 농장
    ("coh_fps", 391460): (2, ""),  # WARMODE
    ("coh_fps", 393380): (3, ""),  # Squad
    ("coh_fps", 444090): (2, ""),  # Paladins®
    ("coh_fps", 849940): (1, "LOW_QUALITY"),  # Freefall Tournament
    ("coh_fps", 1058020): (2, ""),  # STAR WARS™ Battlefront (Classic, 2004)
    ("coh_fps", 1342790): (2, ""),  # GangV | VR & PC Battle Royale
    ("coh_fps", 1548850): (3, ""),  # Six Days in Fallujah
    ("coh_fps", 1657090): (1, "GENRE_ONLY"),  # Mini Royale
    ("coh_fps", 1969870): (3, ""),  # 生死狙击2
    ("coh_fps", 2322660): (2, ""),  # Global Strike
    ("coh_fps", 2364570): (1, "LOW_QUALITY"),  # Multiplayer Shooter FPS
    ("coh_grand_strategy", 7830): (2, ""),  # Men of War™
    ("coh_grand_strategy", 338130): (3, ""),  # Strategy & Tactics: Wargame Collection
    ("coh_grand_strategy", 845050): (2, ""),  # Last Regiment
    ("coh_grand_strategy", 871930): (0, "IRRELEVANT"),  # GLADIABOTS - AI Combat Arena
    ("coh_grand_strategy", 1604450): (0, "IRRELEVANT"),  # AI Battle Royale Generator
    ("coh_grand_strategy", 1957990): (1, "GENRE_ONLY"),  # Tile Cities
    ("coh_indie_platformer", 653060): (3, ""),  # The Siege and the Sandfox
    ("coh_indie_platformer", 1029210): (3, ""),  # 30XX
    ("coh_indie_platformer", 1098080): (3, ""),  # 3000th Duel
    ("coh_indie_platformer", 1772830): (3, ""),  # Rusted Moss
    ("coh_indie_platformer", 3664230): (2, ""),  # Bounce Cat
    ("coh_openworld_survival", 398170): (3, ""),  # Evochron Legacy SE
    ("coh_openworld_survival", 441790): (3, ""),  # Fragmented
    ("coh_openworld_survival", 482300): (2, ""),  # Investigator
    ("coh_openworld_survival", 768200): (2, ""),  # Smalland: Survive the Wilds
    ("coh_openworld_survival", 897730): (3, ""),  # Among Trees
    ("coh_openworld_survival", 1090800): (3, ""),  # Northern Lights
    ("coh_openworld_survival", 3174500): (2, ""),  # 겨울 생존 프로토콜
    ("coh_strategy", 392110): (3, ""),  # ENDLESS Space™ 2
    ("coh_strategy", 538030): (3, ""),  # Xenonauts 2 제노너츠 2
    ("coh_strategy", 839770): (3, ""),  # Phoenix Point
    ("coh_strategy", 874390): (3, ""),  # The Battle of Polytopia
    ("coh_strategy", 1049850): (3, ""),  # UFO2: Extraterrestrials
    ("coh_strategy", 1356810): (3, ""),  # Hexarchy
    ("coh_strategy", 1575940): (3, ""),  # Sins of a Solar Empire II
    ("coh_strategy", 3680900): (2, ""),  # Future War Tactics: SOF vs Alien Invasio
    ("coh_survival_craft", 386590): (2, ""),  # Novus Inceptio
    ("coh_survival_craft", 427950): (2, ""),  # The Withering
    ("coh_survival_craft", 441790): (2, ""),  # Fragmented
    ("coh_survival_craft", 664750): (3, ""),  # Wrongworld
    ("coh_survival_craft", 1270010): (2, ""),  # Gone: Survival
    ("coh_survival_craft", 1394960): (3, ""),  # Winter Survival - 겨울 생존
    ("coh_survival_craft", 1621070): (3, ""),  # DeadPoly
    ("coh_survival_craft", 2067020): (2, ""),  # Polygeddon
    ("coh_survival_craft", 2818210): (2, ""),  # Sustained
    ("coh_survival_craft", 3054440): (3, ""),  # 짧은 눈 / Short Snow
    ("coh_vehicle_sim", 285380): (3, ""),  # On The Road - The Truck Simulator
    ("coh_vehicle_sim", 320540): (1, "MODE_MISMATCH"),  # Coffin Dodgers
    ("coh_vehicle_sim", 995680): (2, ""),  # Truck Mechanic: Dangerous Paths
    ("coh_vehicle_sim", 1020800): (3, ""),  # CAR TUNE: Project
    ("coh_vehicle_sim", 1474930): (3, ""),  # Heavy Cargo - The Truck Simulator
    ("coh_vehicle_sim", 1658480): (3, ""),  # EQDRIVE.IO
    ("coh_vehicle_sim", 2050120): (3, ""),  # Mission: Driver
    ("coh_vehicle_sim", 2269950): (1, "MODE_MISMATCH"),  # The Karters 2: Turbo Charged
    ("coh_vehicle_sim", 2604480): (3, ""),  # City Transport Simulator 2025
    ("coh_vehicle_sim", 2637940): (3, ""),  # Used Cars Simulator
    ("longtail_deckbuilder", 400160): (3, ""),  # Concrete Jungle
    ("longtail_deckbuilder", 493080): (3, ""),  # Card Quest
    ("longtail_deckbuilder", 590380): (2, ""),  # Into the Breach
    ("longtail_deckbuilder", 1075740): (3, ""),  # Banners of Ruin
    ("longtail_deckbuilder", 1373260): (3, ""),  # Obsidian Prince
    ("longtail_deckbuilder", 1676840): (2, ""),  # For The King II
    ("longtail_deckbuilder", 1860600): (2, ""),  # 世界の為の全ての少女
    ("longtail_deckbuilder", 1961600): (2, ""),  # Darfall
    ("longtail_deckbuilder", 2026820): (3, ""),  # Die in the Dungeon
    ("longtail_deckbuilder", 2087460): (3, ""),  # Ruff Ghanor
    ("longtail_deckbuilder", 2091500): (2, ""),  # Warlords Under Siege
    ("longtail_deckbuilder", 2132840): (3, ""),  # Deliverance & Reign
    ("longtail_deckbuilder", 2438260): (3, ""),  # Tiny Kingdom
    ("longtail_deckbuilder", 2655590): (3, ""),  # 카드 왕국
    ("longtail_deckbuilder", 3005620): (3, ""),  # Drop Pockets
    ("longtail_deckbuilder", 3640430): (3, ""),  # Worlds Upon The Wind
    ("longtail_detective", 204060): (2, ""),  # Superbrothers: Sword & Sworcery EP
    ("longtail_detective", 205650): (3, ""),  # The Testament of Sherlock Holmes
    ("longtail_detective", 283880): (3, ""),  # Heroine's Quest: The Herald of Ragnarok
    ("longtail_detective", 314560): (1, "GENRE_ONLY"),  # Kyn
    ("longtail_detective", 316160): (3, ""),  # Broken Sword 4 - the Angel of Death (200
    ("longtail_detective", 326180): (1, "GENRE_ONLY"),  # Sinister City
    ("longtail_detective", 438560): (2, ""),  # Mystery Case Files®: Escape from Ravenhe
    ("longtail_detective", 1125910): (3, ""),  # Crowns and Pawns: Kingdom of Deceit
    ("longtail_detective", 1182310): (3, ""),  # The Excavation of Hob's Barrow
    ("longtail_detective", 1738610): (3, ""),  # First Class Escape: The Train of Thought
    ("longtail_detective", 1812090): (3, ""),  # Escape Academy
    ("longtail_detective", 1858650): (3, ""),  # In Sink: A Co-op Escape Adventure
    ("longtail_detective", 1925320): (3, ""),  # Escape Memoirs: Mansion Heist
    ("longtail_metroidvania", 205230): (2, ""),  # Hell Yeah! Wrath of the Dead Rabbit
    ("longtail_metroidvania", 446810): (2, ""),  # Blossom Tales: The Sleeping King
    ("longtail_metroidvania", 720560): (3, ""),  # Vigil: The Longest Night
    ("longtail_metroidvania", 1072400): (1, "GENRE_ONLY"),  # Outrider Mako
    ("longtail_metroidvania", 1262040): (3, ""),  # Super Mombo Quest
    ("longtail_metroidvania", 1290710): (2, ""),  # Shiro
    ("longtail_metroidvania", 1494110): (2, ""),  # QV(큐브이)
    ("longtail_metroidvania", 1517970): (3, ""),  # Aeterna Noctis
    ("longtail_metroidvania", 1642790): (3, ""),  # Deepening Fire
    ("longtail_metroidvania", 1733250): (3, ""),  # Turbo Kid
    ("longtail_metroidvania", 1781350): (1, "IRRELEVANT"),  # Fireboy & Watergirl: Fairy Tales
    ("longtail_metroidvania", 1794780): (1, "MODE_MISMATCH"),  # SiNiSistar Lite Version
    ("longtail_metroidvania", 1867070): (3, ""),  # Darxanadon
    ("longtail_metroidvania", 2474430): (2, ""),  # TetherGeist
    ("longtail_metroidvania", 3199390): (3, ""),  # INAYAH - Life after Gods
    ("longtail_puzzle_platformer", 65300): (2, ""),  # Dustforce DX
    ("longtail_puzzle_platformer", 205730): (2, ""),  # Insanely Twisted Shadow Planet
    ("longtail_puzzle_platformer", 315430): (3, ""),  # Polarity
    ("longtail_puzzle_platformer", 544330): (3, ""),  # Snake Pass
    ("longtail_puzzle_platformer", 600990): (3, ""),  # The Gardens Between
    ("longtail_puzzle_platformer", 733210): (2, ""),  # Neversong
    ("longtail_puzzle_platformer", 814680): (3, ""),  # Unbound: Worlds Apart
    ("longtail_puzzle_platformer", 860510): (3, ""),  # Little Nightmares II
    ("longtail_puzzle_platformer", 956540): (3, ""),  # Color Phase
    ("longtail_puzzle_platformer", 1436590): (3, ""),  # Phoenotopia: Awakening
    ("longtail_puzzle_platformer", 2078510): (2, ""),  # VIVIDLOPE
    ("longtail_puzzle_platformer", 2982340): (3, ""),  # ChromaGun 2: Dye Hard
    ("longtail_puzzle_platformer", 3317740): (3, ""),  # Chroma
    ("longtail_puzzle_platformer", 4427300): (2, ""),  # Chroma Surge
    ("lowrev_cozy_narrative", 568930): (0, "IRRELEVANT"),  # The Land of Pain
    ("lowrev_cozy_narrative", 952420): (2, ""),  # Shan Gui II: Sweet Osmanthus II
    ("lowrev_cozy_narrative", 1374480): (2, ""),  # 爱人 Lover
    ("lowrev_cozy_narrative", 1589500): (3, ""),  # Fate of Dynasty
    ("lowrev_cozy_narrative", 1833040): (2, ""),  # 晴末置雨
    ("lowrev_cozy_narrative", 2280430): (3, ""),  # The Berlin Apartment
    ("lowrev_cozy_narrative", 2536840): (2, ""),  # GINKA
    ("lowrev_cozy_narrative", 3365030): (1, "GENRE_ONLY"),  # 라이자의 아틀리에 ~어둠의 여왕과 비밀의 은신처~ DX
    ("lowrev_cozy_narrative", 3418570): (2, ""),  # Summer Pockets REFLECTION BLUE
    ("lowrev_cozy_narrative", 3608120): (3, ""),  # In Their Shoes
    ("lowrev_deckbuilder", 1146230): (3, ""),  # DungeonTop
    ("lowrev_deckbuilder", 1156060): (3, ""),  # Railroads & Catacombs
    ("lowrev_deckbuilder", 1990110): (3, ""),  # SpellRogue
    ("lowrev_deckbuilder", 2087460): (3, ""),  # Ruff Ghanor
    ("lowrev_deckbuilder", 2097570): (3, ""),  # StarVaders
    ("lowrev_deckbuilder", 2181950): (3, ""),  # Zero Division
    ("lowrev_deckbuilder", 3621050): (3, ""),  # Doomspire
    ("lowrev_deckbuilder", 3877790): (3, ""),  # 사냥의 밤：소브린 신디케이트
    ("lowrev_detective", 236930): (3, ""),  # Blackwell Epiphany
    ("lowrev_detective", 269650): (1, "IRRELEVANT"),  # Dex
    ("lowrev_detective", 286480): (2, ""),  # Black Mirror III
    ("lowrev_detective", 430020): (1, "GENRE_ONLY"),  # Dance of Death
    ("lowrev_detective", 1546920): (3, ""),  # Overboard!
    ("lowrev_detective", 1781260): (0, "IRRELEVANT"),  # Promenade
    ("lowrev_detective", 2492290): (3, ""),  # 언커버 더 스모킹 건
    ("lowrev_detective", 2875630): (2, ""),  # 迷☆探偵の助手-Remaster-
    ("lowrev_detective", 3218580): (2, ""),  # 감시 아파트
    ("lowrev_detective", 4623310): (2, ""),  # Charm And Clue 2 Collector's Edition
    ("lowrev_metroidvania", 350070): (3, ""),  # Environmental Station Alpha
    ("lowrev_metroidvania", 535480): (3, ""),  # Sundered®: Eldritch Edition
    ("lowrev_metroidvania", 543260): (3, ""),  # Wonder Boy: The Dragon's Trap
    ("lowrev_metroidvania", 655730): (3, ""),  # This Strange Realm Of Mine
    ("lowrev_metroidvania", 731720): (3, ""),  # DOOMED
    ("lowrev_metroidvania", 785880): (2, ""),  # OVERWHELM
    ("lowrev_metroidvania", 791180): (1, "GENRE_ONLY"),  # 1 Screen Platformer
    ("lowrev_metroidvania", 1220150): (3, ""),  # Blue Fire
    ("lowrev_metroidvania", 1290710): (2, ""),  # Shiro
    ("lowrev_metroidvania", 1554600): (2, ""),  # Wife Quest
    ("lowrev_metroidvania", 1634360): (3, ""),  # EMUUROM
    ("lowrev_metroidvania", 1760330): (3, ""),  # Noreya: The Gold Project
    ("lowrev_metroidvania", 2085310): (3, ""),  # Emberbane
    ("lowrev_metroidvania", 3316470): (1, "GENRE_ONLY"),  # DEAD TRASH
    ("lowrev_metroidvania", 3446990): (2, ""),  # FUBUKI ～zero in on Holoearth～ HOLOLIVE A
    ("lowrev_towerdefense", 91200): (3, ""),  # Anomaly: Warzone Earth
    ("lowrev_towerdefense", 209670): (2, ""),  # Cortex Command
    ("lowrev_towerdefense", 214360): (3, ""),  # Tower Wars
    ("lowrev_towerdefense", 235250): (3, ""),  # Super Sanctum TD
    ("lowrev_towerdefense", 485950): (3, ""),  # Rise of the Ancients
    ("lowrev_towerdefense", 541230): (3, ""),  # Alien Shooter TD
    ("lowrev_towerdefense", 923890): (3, ""),  # Yet another tower defence
    ("lowrev_towerdefense", 1113030): (1, "GENRE_ONLY"),  # Stellar Warfare
    ("lowrev_towerdefense", 3162110): (3, ""),  # Space Cycle
    ("mix2_arcade_action", 35140): (1, "IRRELEVANT"),  # Batman: Arkham Asylum Game of the Year E
    ("mix2_arcade_action", 45760): (3, ""),  # Ultra Street Fighter® IV
    ("mix2_arcade_action", 493490): (1, "MODE_MISMATCH"),  # City Car Driving
    ("mix2_arcade_action", 586200): (3, ""),  # Street Fighter 30th Anniversary Collecti
    ("mix2_arcade_action", 658700): (2, ""),  # V-Rally 4
    ("mix2_arcade_action", 938220): (3, ""),  # TAPSONIC BOLD
    ("mix2_arcade_action", 999660): (3, ""),  # SAMURAI SHODOWN NEOGEO COLLECTION
    ("mix2_arcade_action", 1058830): (3, ""),  # Spin Rhythm XD
    ("mix2_arcade_action", 1384160): (3, ""),  # GUILTY GEAR -STRIVE-
    ("mix2_arcade_action", 1465360): (2, ""),  # SnowRunner
    ("mix2_builder_sim", 48240): (3, ""),  # Anno 2070™
    ("mix2_builder_sim", 534370): (3, ""),  # Pure Farming 2018
    ("mix2_builder_sim", 1074180): (3, ""),  # Smart City Plan
    ("mix2_builder_sim", 1112790): (3, ""),  # Automation Empire
    ("mix2_builder_sim", 1137750): (2, ""),  # Farmer's Life
    ("mix2_builder_sim", 1351080): (3, ""),  # Pharaoh™: A New Era
    ("mix2_builder_sim", 1352850): (3, ""),  # Citystate II
    ("mix2_builder_sim", 1509510): (3, ""),  # Settlement Survival
    ("mix2_builder_sim", 1960480): (3, ""),  # AutoForge
    ("mix2_builder_sim", 2182630): (2, ""),  # Mob Factory
    ("mix2_builder_sim", 2206350): (3, ""),  # Farm Manager World
    ("mix2_builder_sim", 2449430): (3, ""),  # Incremental Factory
    ("mix2_builder_sim", 3185000): (3, ""),  # 天际都市物语
    ("mix2_builder_sim", 4834620): (3, ""),  # Fabricatio
    ("mix2_colony_cozy", 252250): (3, ""),  # Maia
    ("mix2_colony_cozy", 382310): (2, ""),  # Eco
    ("mix2_colony_cozy", 688060): (3, ""),  # Odd Realm
    ("mix2_colony_cozy", 711980): (2, ""),  # Starship EVO
    ("mix2_colony_cozy", 870200): (1, "GENRE_ONLY"),  # Juno: New Origins
    ("mix2_colony_cozy", 964130): (3, ""),  # My Colony
    ("mix2_colony_cozy", 1212620): (0, "IRRELEVANT"),  # Pretty Neko
    ("mix2_colony_cozy", 1254320): (3, ""),  # Surviving the Abyss
    ("mix2_colony_cozy", 1876000): (1, "IRRELEVANT"),  # IFO
    ("mix2_colony_cozy", 2020710): (3, ""),  # Havendock
    ("mix2_colony_cozy", 2133570): (3, ""),  # SpaceSlog
    ("mix2_colony_cozy", 2303350): (2, ""),  # Sticky Business
    ("mix2_colony_cozy", 2589500): (3, ""),  # 위스퍼 오브 더 하우스: 잠꼬대마을 이야기
    ("mix2_colony_cozy", 2941660): (2, ""),  # Archean
    ("mix2_colony_cozy", 3084150): (3, ""),  # Packit List
    ("mix2_colony_cozy", 3565100): (1, "KEYWORD_MATCH"),  # Unboxathon
    ("mix2_colony_cozy", 4469180): (3, ""),  # Arcbound
    ("mix2_coop_horror", 552500): (3, ""),  # Warhammer: Vermintide 2
    ("mix2_coop_horror", 632360): (2, ""),  # Risk of Rain 2
    ("mix2_coop_horror", 1237970): (1, "MODE_MISMATCH"),  # Titanfall® 2
    ("mix2_coop_horror", 1450180): (3, ""),  # Haunt Chaser
    ("mix2_coop_horror", 1506620): (3, ""),  # Horror Squad
    ("mix2_coop_horror", 1708460): (3, ""),  # Obsideo
    ("mix2_coop_horror", 1850740): (3, ""),  # Ghost Watchers
    ("mix2_coop_horror", 2400880): (3, ""),  # Haunted Investigation
    ("mix2_coop_horror", 2454890): (3, ""),  # DEATH IN UNISON
    ("mix2_coop_horror", 2541890): (3, ""),  # Abnormality
    ("mix2_coop_horror", 3004140): (3, ""),  # LockYourDoor
    ("mix2_coop_horror", 3176060): (3, ""),  # Emissary Zero
    ("mix2_coop_horror", 3241660): (3, ""),  # R.E.P.O.
    ("mix2_coop_horror", 3433640): (3, ""),  # Mining Company
    ("mix2_coop_horror", 4805600): (3, ""),  # Scavenge Protocol
    ("mix2_crpg_sandbox", 47810): (3, ""),  # Dragon Age: Origins - Ultimate Edition
    ("mix2_crpg_sandbox", 655780): (2, ""),  # Project 5: Sightseer
    ("mix2_crpg_sandbox", 1058710): (2, ""),  # 中世纪君主 Medieval Monarch
    ("mix2_crpg_sandbox", 1170950): (2, ""),  # Mortal Online 2
    ("mix2_crpg_sandbox", 1184370): (3, ""),  # Pathfinder: Wrath of the Righteous - Enh
    ("mix2_crpg_sandbox", 1222690): (3, ""),  # 드래곤 에이지™ 인퀴지션
    ("mix2_crpg_sandbox", 1771300): (3, ""),  # Kingdom Come: Deliverance II
    ("mix2_crpg_sandbox", 1995520): (2, ""),  # Pax Dei
    ("mix2_crpg_sandbox", 2010030): (1, "IRRELEVANT"),  # Denizen
    ("mix2_crpg_sandbox", 3646460): (1, "LOW_QUALITY"),  # Devil Spire Falls
    ("mix2_crpg_sandbox", 4545380): (3, ""),  # Parting Shadows
    ("mix2_modern_roguelite", 656350): (3, ""),  # UnderMine
    ("mix2_modern_roguelite", 930780): (3, ""),  # Blood Card
    ("mix2_modern_roguelite", 1076200): (3, ""),  # Roguebook
    ("mix2_modern_roguelite", 1769830): (3, ""),  # As We Descend: 심연 속으로
    ("mix2_modern_roguelite", 2002220): (3, ""),  # Keeper's Toll
    ("mix2_modern_roguelite", 2003050): (3, ""),  # Perseus: Titan Slayer
    ("mix2_modern_roguelite", 2071280): (3, ""),  # Ravenswatch
    ("mix2_modern_roguelite", 2354600): (2, ""),  # Renfield: Bring Your Own Blood
    ("mix2_modern_roguelite", 2418490): (2, ""),  # Trakonius
    ("mix2_modern_roguelite", 2774490): (2, ""),  # 梦境多人生存挑战
    ("mix2_modern_roguelite", 3032830): (3, ""),  # Katanaut
    ("mix2_modern_roguelite", 3265700): (3, ""),  # Vampire Crawlers: The Turbo Wildcard fro
    ("mix2_modern_roguelite", 4814430): (3, ""),  # Board Poker
    ("mix2_party_narrative", 95400): (3, ""),  # ibb & obb
    ("mix2_party_narrative", 232430): (3, ""),  # Gone Home
    ("mix2_party_narrative", 1206430): (3, ""),  # The Unfinished Swan
    ("mix2_party_narrative", 1319420): (2, ""),  # Chasing Static
    ("mix2_party_narrative", 1509960): (3, ""),  # PICO PARK
    ("mix2_party_narrative", 1801110): (3, ""),  # 다른 겨울의 우리들
    ("mix2_party_narrative", 2560240): (3, ""),  # Biped 2
    ("mix2_party_narrative", 2683150): (2, ""),  # Ale & Tale Tavern
    ("mix2_party_narrative", 2796010): (3, ""),  # Party Club
    ("mix2_party_narrative", 2870920): (2, ""),  # Restaurats
    ("mix2_party_narrative", 2916430): (3, ""),  # Fast Food Simulator
    ("mix2_party_narrative", 3959400): (2, ""),  # The Remains and The Residue
    ("mix2_puzzle_survival", 327090): (2, ""),  # Out of Reach
    ("mix2_puzzle_survival", 385250): (1, "IRRELEVANT"),  # Paint it Back
    ("mix2_puzzle_survival", 768200): (2, ""),  # Smalland: Survive the Wilds
    ("mix2_puzzle_survival", 895400): (3, ""),  # Deadside
    ("mix2_puzzle_survival", 898780): (1, "LOW_QUALITY"),  # Escape Game
    ("mix2_puzzle_survival", 1021070): (1, "IRRELEVANT"),  # Spaceland: Sci-Fi Indie Tactics
    ("mix2_puzzle_survival", 2071500): (3, ""),  # Islands of Insight
    ("mix2_puzzle_survival", 2400510): (3, ""),  # Dungeons & Degenerate Gamblers
    ("mix2_puzzle_survival", 2427410): (2, ""),  # S.T.A.L.K.E.R.: Shadow of Chornobyl 인핸스드
    ("mix2_puzzle_survival", 2717750): (3, ""),  # 토블라 - 신성한 길
    ("mix2_puzzle_survival", 2746910): (2, ""),  # 삼국 영웅：혈월의 전설
    ("mix2_puzzle_survival", 3411020): (2, ""),  # 마을 & 던전
    ("mix2_soulslike_narrative", 1128920): (2, ""),  # EVERSPACE™ 2
    ("mix2_soulslike_narrative", 1135260): (1, "GENRE_ONLY"),  # The Falconeer: Revolution Remaster
    ("mix2_soulslike_narrative", 1225070): (1, "LOW_QUALITY"),  # Family Mysteries: Poisonous Promises
    ("mix2_soulslike_narrative", 1371690): (2, ""),  # GRAVEN
    ("mix2_soulslike_narrative", 1573720): (2, ""),  # 애거서 크리스티 – 에르퀼 푸아로: 첫 번째 사건
    ("mix2_soulslike_narrative", 1771980): (3, ""),  # The Operator
    ("mix2_soulslike_narrative", 1920490): (3, ""),  # The Outer Worlds: Spacer's Choice Editio
    ("mix2_soulslike_narrative", 2526310): (1, "GENRE_ONLY"),  # Out of Sight
    ("mix2_soulslike_narrative", 2536520): (1, "GENRE_ONLY"),  # 디아블로 II 레저렉션 - 지옥불 에디션
    ("mix2_soulslike_narrative", 2797960): (3, ""),  # 은폐된 살인의 진실들 - 하드코어 본격 추리 탐정 게임
    ("mix2_soulslike_narrative", 3216340): (3, ""),  # Tearscape
    ("mix2_survival_farm", 105600): (3, ""),  # Terraria
    ("mix2_survival_farm", 346110): (3, ""),  # ARK: Survival Evolved
    ("mix2_survival_farm", 361420): (3, ""),  # ASTRONEER
    ("mix2_survival_farm", 413150): (3, ""),  # Stardew Valley
    ("mix2_survival_farm", 432010): (3, ""),  # World's Dawn
    ("mix2_survival_farm", 538100): (2, ""),  # Feel The Snow
    ("mix2_survival_farm", 670260): (3, ""),  # Solace Crafting
    ("mix2_survival_farm", 758870): (3, ""),  # Kynseed
    ("mix2_survival_farm", 894940): (3, ""),  # Littlewood
    ("mix2_survival_farm", 897450): (3, ""),  # The Survivalists
    ("mix2_survival_farm", 2868100): (3, ""),  # Shamania
    ("niche_cozy_casual", 226620): (1, "IRRELEVANT"),  # Desktop Dungeons
    ("niche_cozy_casual", 304460): (2, ""),  # Qora
    ("niche_cozy_casual", 388620): (1, "IRRELEVANT"),  # DungeonUp
    ("niche_cozy_casual", 454060): (2, ""),  # Blueprint Tycoon
    ("niche_cozy_casual", 1048920): (1, "IRRELEVANT"),  # The Jolly Gang's Misadventures in Africa
    ("niche_cozy_casual", 1484900): (3, ""),  # Hoa
    ("niche_cozy_casual", 1722100): (3, ""),  # Sky Tale
    ("niche_cozy_casual", 1749430): (3, ""),  # Reefland
    ("niche_cozy_casual", 2324180): (3, ""),  # Tranquil Isle
    ("niche_puzzle_solo", 1686640): (3, ""),  # Platformer::Execute();
    ("niche_puzzle_solo", 1695620): (3, ""),  # CHR$(143)
    ("niche_puzzle_solo", 1955110): (3, ""),  # Star Stuff
    ("niche_puzzle_solo", 2216120): (1, "GENRE_ONLY"),  # Magicube
    ("niche_puzzle_solo", 2216770): (3, ""),  # JOY OF PROGRAMMING - Software Engineerin
    ("niche_puzzle_solo", 2449430): (2, ""),  # Incremental Factory
    ("niche_puzzle_solo", 2915950): (1, "GENRE_ONLY"),  # Mega Mosaic
    ("niche_puzzle_solo", 3320980): (3, ""),  # MoteMancer
    ("niche_roguelite", 238280): (3, ""),  # Legend of Dungeon
    ("niche_roguelite", 320040): (2, ""),  # Moon Hunters
    ("niche_roguelite", 323220): (3, ""),  # Vagante
    ("niche_roguelite", 1092630): (3, ""),  # Rogue Glitch Ultra
    ("niche_roguelite", 1479140): (3, ""),  # AK-xolotl: Together
    ("niche_roguelite", 1761380): (3, ""),  # Otherworld Legends
    ("niche_roguelite", 1954200): (2, ""),  # Kena: Bridge of Spirits
    ("niche_roguelite", 2185780): (2, ""),  # Return to abyss 重返深渊
    ("niche_roguelite", 2369950): (3, ""),  # Blade of the Netherworld
    ("niche_roguelite", 2949580): (2, ""),  # 유은은 구절이다
    ("niche_roguelite", 3206200): (2, ""),  # 마법의 룬스톤
    ("niche_roguelite", 3223160): (2, ""),  # Apocalypse Express
    ("niche_sim", 70100): (3, ""),  # Hacker Evolution
    ("niche_sim", 226820): (2, ""),  # Nancy Drew®: Ghost of Thornton Hall
    ("niche_sim", 615770): (2, ""),  # Nancy Drew®: Message in a Haunted Mansio
    ("niche_sim", 1403740): (3, ""),  # Space Architect
    ("niche_sim", 1858650): (3, ""),  # In Sink: A Co-op Escape Adventure
    ("niche_sim", 2722360): (3, ""),  # Escape Memoirs: Safe House
    ("niche_sim", 2941660): (3, ""),  # Archean
    ("niche_sim", 3242950): (3, ""),  # Outworld Station
    ("niche_soulslike_solo", 39160): (1, "GENRE_ONLY"),  # Dungeon Siege III
    ("niche_soulslike_solo", 205730): (2, ""),  # Insanely Twisted Shadow Planet
    ("niche_soulslike_solo", 306440): (3, ""),  # Oblitus
    ("niche_soulslike_solo", 655740): (2, ""),  # Tallowmere 2: Curse of the Kittens
    ("niche_soulslike_solo", 750800): (2, ""),  # Egress
    ("niche_soulslike_solo", 919360): (3, ""),  # Alaloth: Champions of The Four Kingdoms
    ("niche_soulslike_solo", 1428710): (3, ""),  # The Tarnishing of Juxtia
    ("niche_soulslike_solo", 2488540): (2, ""),  # Crimson Capes
    ("niche_tactics", 356430): (2, ""),  # Chris Sawyer's Locomotion™
    ("niche_tactics", 602320): (3, ""),  # Train Valley 2
    ("niche_tactics", 856990): (2, ""),  # A Long Way Down
    ("niche_tactics", 1389360): (3, ""),  # 메크 아르마
    ("niche_tactics", 1508400): (3, ""),  # Kaiju Wars
    ("niche_tactics", 1708950): (3, ""),  # Battle Train
    ("niche_tactics", 1832050): (3, ""),  # All Quiet Roads
    ("niche_tactics", 1958190): (2, ""),  # MiniTrans
    ("niche_tactics", 2082410): (3, ""),  # Anomaly Collapse
    ("niche_tactics", 2332970): (3, ""),  # ARC SEED
    ("niche_tactics", 2521630): (3, ""),  # Mini Settlers
}


BLIND_K30_P3 = {
    ("coh_arpg", 39690): (2, ""),  # ArcaniA
    ("coh_arpg", 617290): (3, ""),  # Remnant: From the Ashes
    ("coh_arpg", 727850): (2, ""),  # ELDERBORN
    ("coh_arpg", 831050): (2, ""),  # Dolmen
    ("coh_arpg", 896440): (2, ""),  # Eternal Edge +
    ("coh_arpg", 908060): (1, "MODE_MISMATCH"),  # Talisman: Digital 5th Edition
    ("coh_arpg", 1295900): (1, "GENRE_ONLY"),  # Draconia
    ("coh_arpg", 1373090): (1, "GENRE_ONLY"),  # Yes, Your Grace 2: Snowfall
    ("coh_arpg", 4746070): (1, "MODE_MISMATCH"),  # Eldara
    ("coh_classic_multi", 209670): (2, ""),  # Cortex Command
    ("coh_classic_multi", 270550): (3, ""),  # Yet Another Zombie Defense
    ("coh_classic_multi", 436520): (2, ""),  # Line of Sight
    ("coh_classic_multi", 502550): (1, "LOW_QUALITY"),  # Strike.is: The Game
    ("coh_classic_multi", 1515640): (2, ""),  # Arcadegeddon
    ("coh_classic_multi", 1765300): (3, ""),  # 얼티밋 좀비 디펜스 2
    ("coh_classic_multi", 2157830): (3, ""),  # John Carpenter's Toxic Commando
    ("coh_classic_multi", 2179380): (2, ""),  # Sand:box
    ("coh_classic_multi", 2504090): (2, ""),  # Heroes of Valor
    ("coh_cozy", 252490): (2, ""),  # Rust
    ("coh_cozy", 758870): (3, ""),  # Kynseed
    ("coh_cozy", 834910): (2, ""),  # ATLAS
    ("coh_cozy", 877010): (1, "GENRE_ONLY"),  # Beyond Contact
    ("coh_cozy", 1137750): (2, ""),  # Farmer's Life
    ("coh_cozy", 1156360): (3, ""),  # Peaceful Days
    ("coh_cozy", 1432860): (3, ""),  # Sun Haven
    ("coh_cozy", 2275150): (1, "IRRELEVANT"),  # SLUDGE LIFE 2
    ("coh_cozy", 2678830): (3, ""),  # Halcyon Days at Taoyuan
    ("coh_cozy", 2699820): (1, "IRRELEVANT"),  # 짱구는 못말려 「탄광마을의 흰둥이」
    ("coh_cozy", 2910590): (2, ""),  # Slime Farm
    ("coh_cozy", 3473540): (0, "KEYWORD_MATCH"),  # Bring Slime to Mommy
    ("coh_fps", 16900): (3, ""),  # GROUND BRANCH
    ("coh_fps", 24240): (2, ""),  # PAYDAY™ The Heist
    ("coh_fps", 218230): (3, ""),  # PlanetSide 2
    ("coh_fps", 302830): (2, ""),  # BLOCKADE 3D
    ("coh_fps", 417910): (1, "LOW_QUALITY"),  # Street Warriors Online
    ("coh_fps", 879160): (2, ""),  # Battlerite Royale
    ("coh_fps", 1219480): (2, ""),  # Poly Squad
    ("coh_fps", 1240440): (3, ""),  # Halo Infinite
    ("coh_fps", 2229890): (2, ""),  # Command & Conquer Renegade™
    ("coh_fps", 2687970): (2, ""),  # TRIBES 3: Rivals
    ("coh_fps", 2727490): (2, ""),  # BLITZ ARENA
    ("coh_fps", 3555700): (1, "IRRELEVANT"),  # Liar Game
    ("coh_fps", 3659280): (3, ""),  # Off The Grid
    ("coh_grand_strategy", 289300): (2, ""),  # Faces of War
    ("coh_grand_strategy", 602770): (1, "GENRE_ONLY"),  # Clatter
    ("coh_grand_strategy", 672680): (3, ""),  # Supremacy: Call of War 1942
    ("coh_grand_strategy", 804730): (2, ""),  # Third Front: WWII
    ("coh_grand_strategy", 872410): (3, ""),  # 삼국지14
    ("coh_grand_strategy", 1022780): (0, "IRRELEVANT"),  # Animal Fight Club
    ("coh_grand_strategy", 1671590): (2, ""),  # Paladin's Oath
    ("coh_grand_strategy", 1834920): (2, ""),  # Countryball: Europe 1890
    ("coh_grand_strategy", 2716930): (3, ""),  # 皇帝与社稷 The Emperor and State
    ("coh_grand_strategy", 3458090): (1, "IRRELEVANT"),  # Broadcast Empire
    ("coh_grand_strategy", 3681230): (3, ""),  # Observe
    ("coh_grand_strategy", 3885930): (2, ""),  # 모래판 전쟁：삼국
    ("coh_grand_strategy", 3985030): (2, ""),  # Air Defender
    ("coh_indie_platformer", 115800): (3, ""),  # Owlboy
    ("coh_indie_platformer", 246680): (1, "LOW_QUALITY"),  # Secrets of Rætikon
    ("coh_indie_platformer", 493200): (2, ""),  # RiME
    ("coh_indie_platformer", 751250): (3, ""),  # Big Tower Tiny Square
    ("coh_indie_platformer", 1055540): (2, ""),  # A Short Hike
    ("coh_indie_platformer", 1150950): (1, "GENRE_ONLY"),  # Timelie
    ("coh_indie_platformer", 1281790): (3, ""),  # SHEEPO
    ("coh_indie_platformer", 1517970): (3, ""),  # Aeterna Noctis
    ("coh_indie_platformer", 1679220): (2, ""),  # 포탈 던전
    ("coh_indie_platformer", 1983620): (3, ""),  # Infinitevania
    ("coh_indie_platformer", 2168150): (3, ""),  # Teslagrad Remastered
    ("coh_indie_platformer", 3088410): (2, ""),  # 七度荒域：混沌之树
    ("coh_indie_platformer", 3206200): (2, ""),  # 마법의 룬스톤
    ("coh_openworld_survival", 340490): (3, ""),  # Subterrain
    ("coh_openworld_survival", 536270): (3, ""),  # Ancestors: The Humankind Odyssey
    ("coh_openworld_survival", 568930): (2, ""),  # The Land of Pain
    ("coh_openworld_survival", 1363900): (3, ""),  # Farworld Pioneers
    ("coh_openworld_survival", 1571160): (2, ""),  # Lost Scavenger
    ("coh_openworld_survival", 1967630): (3, ""),  # Survival: Fountain of Youth
    ("coh_openworld_survival", 2644050): (3, ""),  # Echoes of Elysium
    ("coh_openworld_survival", 3922100): (2, ""),  # Forest Doesn’t Care
    ("coh_strategy", 10500): (3, ""),  # 토탈워: 엠파이어 - 데피니티브 에디션
    ("coh_strategy", 240760): (3, ""),  # Wasteland 2: Director's Cut
    ("coh_strategy", 799600): (2, ""),  # Cosmoteer: Starship Architect & Commande
    ("coh_strategy", 979110): (2, ""),  # Space Haven
    ("coh_strategy", 1201700): (3, ""),  # Warnament
    ("coh_strategy", 1574870): (3, ""),  # USC: Counterforce
    ("coh_strategy", 1937750): (2, ""),  # Prime of Flames
    ("coh_strategy", 2012190): (3, ""),  # KAISERPUNK
    ("coh_strategy", 2082410): (2, ""),  # Anomaly Collapse
    ("coh_strategy", 3183500): (2, ""),  # Epoch of Guardians
    ("coh_survival_craft", 280520): (2, ""),  # Crea
    ("coh_survival_craft", 382310): (1, "MODE_MISMATCH"),  # Eco
    ("coh_survival_craft", 482300): (1, "IRRELEVANT"),  # Investigator
    ("coh_survival_craft", 538100): (2, ""),  # Feel The Snow
    ("coh_survival_craft", 574180): (3, ""),  # Rem Survival
    ("coh_survival_craft", 657990): (3, ""),  # Crafting Dead
    ("coh_survival_craft", 1197220): (2, ""),  # Another Try
    ("coh_survival_craft", 1211600): (3, ""),  # ZED ZONE
    ("coh_survival_craft", 1597980): (3, ""),  # City 20
    ("coh_survival_craft", 1755080): (2, ""),  # Away From Life
    ("coh_survival_craft", 2179720): (2, ""),  # The Seekers: Survival
    ("coh_survival_craft", 2941710): (3, ""),  # Project Silverfish
    ("coh_vehicle_sim", 427100): (3, ""),  # Fernbus Simulator
    ("coh_vehicle_sim", 515180): (3, ""),  # Bus Simulator 18
    ("coh_vehicle_sim", 633110): (2, ""),  # FAST BEAT LOOP RACER GT | 環狀賽車GT
    ("coh_vehicle_sim", 899390): (2, ""),  # Xenon Racer
    ("coh_vehicle_sim", 1122720): (1, "GENRE_ONLY"),  # Sayonara Wild Hearts
    ("coh_vehicle_sim", 1222040): (3, ""),  # Offroad Mania
    ("coh_vehicle_sim", 1594060): (2, ""),  # Victory Heat Rally
    ("coh_vehicle_sim", 1681880): (2, ""),  # NHRA Championship Drag Racing: Speed For
    ("coh_vehicle_sim", 1737450): (2, ""),  # Motorcycle Biker Simulator
    ("coh_vehicle_sim", 2077750): (3, ""),  # RENNSPORT
    ("coh_vehicle_sim", 2697250): (2, ""),  # Gran Carismo
    ("coh_vehicle_sim", 2947450): (3, ""),  # Exhibition of Speed
    ("longtail_deckbuilder", 221020): (1, "GENRE_ONLY"),  # Towns
    ("longtail_deckbuilder", 496620): (3, ""),  # Monster Slayers
    ("longtail_deckbuilder", 1098610): (3, ""),  # Crush the Industry
    ("longtail_deckbuilder", 1296610): (3, ""),  # Peglin
    ("longtail_deckbuilder", 1535100): (3, ""),  # Nadir: A Grimdark Deckbuilder
    ("longtail_deckbuilder", 1552080): (3, ""),  # Deepest Chamber: Resurrection
    ("longtail_deckbuilder", 1769830): (3, ""),  # As We Descend: 심연 속으로
    ("longtail_deckbuilder", 2140850): (3, ""),  # Looper Tactics
    ("longtail_deckbuilder", 2311990): (3, ""),  # Rack and Slay
    ("longtail_deckbuilder", 2452820): (3, ""),  # Skogdal
    ("longtail_deckbuilder", 2564520): (3, ""),  # Lost in Random: The Eternal Die
    ("longtail_deckbuilder", 3028310): (2, ""),  # Nordhold
    ("longtail_deckbuilder", 3760520): (3, ""),  # Rogue 'n' Roll: Dice of Fate
    ("longtail_detective", 63660): (3, ""),  # Myst: Masterpiece Edition
    ("longtail_detective", 211160): (0, "IRRELEVANT"),  # Viking: Battle for Asgard
    ("longtail_detective", 269050): (3, ""),  # Year Walk
    ("longtail_detective", 284870): (2, ""),  # 9 Clues: The Secret of Serpent Creek
    ("longtail_detective", 394680): (2, ""),  # Mountain Crime: Requital
    ("longtail_detective", 677160): (3, ""),  # We Were Here Too
    ("longtail_detective", 790740): (2, ""),  # Tick Tock: A Tale for Two
    ("longtail_detective", 1004860): (2, ""),  # The Secret Order 7: Shadow Breach
    ("longtail_detective", 1361320): (3, ""),  # The Room 4: Old Sins
    ("longtail_detective", 1466390): (3, ""),  # Kathy Rain 2: Soothsayer
    ("longtail_detective", 1483780): (3, ""),  # Tested on Humans: Escape Room
    ("longtail_detective", 1542390): (3, ""),  # Monolith
    ("longtail_detective", 2307690): (1, "GENRE_ONLY"),  # Ancient Saga: Vikings Journey
    ("longtail_detective", 3342410): (3, ""),  # Dr Voss’ Escape Room
    ("longtail_metroidvania", 200900): (3, ""),  # Cave Story+
    ("longtail_metroidvania", 236090): (3, ""),  # Dust: An Elysian Tail
    ("longtail_metroidvania", 296870): (1, "GENRE_ONLY"),  # Dreaming Sarah
    ("longtail_metroidvania", 598700): (3, ""),  # The Vagrant
    ("longtail_metroidvania", 764790): (3, ""),  # The Messenger
    ("longtail_metroidvania", 891170): (2, ""),  # The Witch & The 66 Mushrooms
    ("longtail_metroidvania", 960190): (2, ""),  # Maytroid. I swear it's a nice game too
    ("longtail_metroidvania", 1022480): (2, ""),  # KAMIKO
    ("longtail_metroidvania", 1098080): (3, ""),  # 3000th Duel
    ("longtail_metroidvania", 1281790): (3, ""),  # SHEEPO
    ("longtail_metroidvania", 1306630): (3, ""),  # Lost Ruins
    ("longtail_metroidvania", 1630270): (1, "LOW_QUALITY"),  # Death Moon
    ("longtail_metroidvania", 1657630): (0, "KEYWORD_MATCH"),  # Slime Rancher 2
    ("longtail_metroidvania", 1836030): (3, ""),  # TOOK: The Immortal Hero
    ("longtail_metroidvania", 2463220): (3, ""),  # Lone Fungus: Melody of Spores
    ("longtail_metroidvania", 3418990): (3, ""),  # 신역 추락
    ("longtail_puzzle_platformer", 95300): (2, ""),  # Capsized
    ("longtail_puzzle_platformer", 218820): (1, "GENRE_ONLY"),  # Mercenary Kings: Reloaded Edition
    ("longtail_puzzle_platformer", 269670): (3, ""),  # BADLAND: Game of the Year Edition
    ("longtail_puzzle_platformer", 938560): (3, ""),  # INMOST
    ("longtail_puzzle_platformer", 1069740): (2, ""),  # Seen
    ("longtail_puzzle_platformer", 1290760): (3, ""),  # 도마뱀붙이의 모험
    ("longtail_puzzle_platformer", 1444090): (2, ""),  # Colorful Colore
    ("longtail_puzzle_platformer", 1744450): (3, ""),  # Indirection
    ("longtail_puzzle_platformer", 1778840): (3, ""),  # Spirit of the North 2
    ("longtail_puzzle_platformer", 1837330): (3, ""),  # Gibbon: Beyond the Trees
    ("longtail_puzzle_platformer", 1924360): (2, ""),  # Lil' Guardsman
    ("longtail_puzzle_platformer", 1982340): (3, ""),  # Blanc 블랑
    ("longtail_puzzle_platformer", 2584650): (0, "IRRELEVANT"),  # Girlfriend from Hell
    ("longtail_puzzle_platformer", 3358240): (3, ""),  # Little Guardians: The Last Hope
    ("longtail_puzzle_platformer", 3528500): (3, ""),  # Perspective Remastered
    ("lowrev_cozy_narrative", 844660): (3, ""),  # Heart of the Woods
    ("lowrev_cozy_narrative", 864700): (1, "IRRELEVANT"),  # Dinosaur Fossil Hunter
    ("lowrev_cozy_narrative", 1067540): (3, ""),  # Röki
    ("lowrev_cozy_narrative", 1185780): (2, ""),  # Forest Ranger Simulator
    ("lowrev_cozy_narrative", 1335230): (3, ""),  # Sumire
    ("lowrev_cozy_narrative", 1556490): (2, ""),  # Retreat To Enen
    ("lowrev_cozy_narrative", 1564110): (1, "IRRELEVANT"),  # Beside Myself
    ("lowrev_cozy_narrative", 1614270): (3, ""),  # Submerged: Hidden Depths
    ("lowrev_cozy_narrative", 1671570): (3, ""),  # Out and About
    ("lowrev_cozy_narrative", 2113350): (1, "GENRE_ONLY"),  # Heartspell: Horizon Academy
    ("lowrev_cozy_narrative", 2720920): (3, ""),  # ASTROBOTANICA
    ("lowrev_cozy_narrative", 3622260): (3, ""),  # April Grove
    ("lowrev_deckbuilder", 496620): (3, ""),  # Monster Slayers
    ("lowrev_deckbuilder", 1073490): (3, ""),  # Monster X Monster
    ("lowrev_deckbuilder", 1093320): (3, ""),  # Devil's Deck
    ("lowrev_deckbuilder", 1127610): (3, ""),  # Iris and the Giant
    ("lowrev_deckbuilder", 1608040): (3, ""),  # Castle Morihisa
    ("lowrev_deckbuilder", 1769830): (3, ""),  # As We Descend: 심연 속으로
    ("lowrev_deckbuilder", 1803400): (3, ""),  # Beneath Oresa
    ("lowrev_deckbuilder", 1958340): (2, ""),  # Cube Chaos
    ("lowrev_deckbuilder", 1968320): (3, ""),  # Relapse
    ("lowrev_deckbuilder", 2008050): (3, ""),  # Right and Down
    ("lowrev_deckbuilder", 2305500): (2, ""),  # FAIRY TAIL 던전스
    ("lowrev_deckbuilder", 2360210): (3, ""),  # Rune Coliseum
    ("lowrev_deckbuilder", 3064290): (3, ""),  # Trizon
    ("lowrev_deckbuilder", 3636060): (3, ""),  # RogueDiceR
    ("lowrev_detective", 94620): (2, ""),  # Hector: Badge of Carnage - Full Series
    ("lowrev_detective", 566190): (2, ""),  # The Search
    ("lowrev_detective", 593960): (2, ""),  # Lorelai
    ("lowrev_detective", 935580): (2, ""),  # True Fear: Forsaken Souls Part 2
    ("lowrev_detective", 1011190): (3, ""),  # SIMULACRA 2
    ("lowrev_detective", 1170570): (3, ""),  # The Drifter
    ("lowrev_detective", 1172450): (2, ""),  # Carto
    ("lowrev_detective", 1431270): (2, ""),  # Arcana Sands of Destiny
    ("lowrev_detective", 1771980): (3, ""),  # The Operator
    ("lowrev_detective", 3364770): (3, ""),  # 迷雾审判
    ("lowrev_detective", 3525020): (3, ""),  # The Stepfather Remake
    ("lowrev_detective", 3570370): (1, "GENRE_ONLY"),  # CARIMARA: Beneath the forlorn limbs
    ("lowrev_metroidvania", 205230): (2, ""),  # Hell Yeah! Wrath of the Dead Rabbit
    ("lowrev_metroidvania", 286040): (1, "GENRE_ONLY"),  # Dead Effect
    ("lowrev_metroidvania", 332610): (3, ""),  # Mystik Belle
    ("lowrev_metroidvania", 375520): (2, ""),  # Taimumari: Definitive Edition
    ("lowrev_metroidvania", 449610): (3, ""),  # 몬스터 보이와 저주받은 왕국
    ("lowrev_metroidvania", 467850): (3, ""),  # METAGAL
    ("lowrev_metroidvania", 653060): (3, ""),  # The Siege and the Sandfox
    ("lowrev_metroidvania", 668550): (3, ""),  # 사망여각 (8Doors: Arum's Afterlife Adventure
    ("lowrev_metroidvania", 871950): (1, "GENRE_ONLY"),  # DERE EVIL EXE
    ("lowrev_metroidvania", 946030): (3, ""),  # Axiom Verge 2
    ("lowrev_metroidvania", 1478160): (2, ""),  # 咸鱼喵喵
    ("lowrev_metroidvania", 1672810): (3, ""),  # MIO: Memories in Orbit
    ("lowrev_metroidvania", 2751000): (3, ""),  # 페르시아의 왕자: 잃어버린 왕관
    ("lowrev_towerdefense", 70600): (1, "MODE_MISMATCH"),  # Worms Ultimate Mayhem
    ("lowrev_towerdefense", 104000): (3, ""),  # iBomber Defense
    ("lowrev_towerdefense", 204530): (3, ""),  # Infested Planet
    ("lowrev_towerdefense", 236110): (3, ""),  # Dungeon Defenders II
    ("lowrev_towerdefense", 267340): (3, ""),  # Beware Planet Earth
    ("lowrev_towerdefense", 391310): (1, "LOW_QUALITY"),  # Alien Attack
    ("lowrev_towerdefense", 429060): (2, ""),  # Zombie Wars: Invasion
    ("lowrev_towerdefense", 562500): (3, ""),  # Warstone TD
    ("lowrev_towerdefense", 1362560): (2, ""),  # Fire Commander
    ("lowrev_towerdefense", 2224640): (3, ""),  # Artisan TD
    ("lowrev_towerdefense", 2726870): (3, ""),  # Loopstructor
    ("lowrev_towerdefense", 3139620): (3, ""),  # 최전선 위기
    ("lowrev_towerdefense", 4525840): (2, ""),  # Star Castles 2
    ("mix2_arcade_action", 47920): (2, ""),  # Shift 2 Unleashed
    ("mix2_arcade_action", 431600): (2, ""),  # Automobilista
    ("mix2_arcade_action", 441380): (2, ""),  # PAC-MAN™ CHAMPIONSHIP EDITION 2
    ("mix2_arcade_action", 751970): (2, ""),  # Plox Neon
    ("mix2_arcade_action", 989550): (2, ""),  # Million Arthur: Arcana Blood
    ("mix2_arcade_action", 1122720): (3, ""),  # Sayonara Wild Hearts
    ("mix2_arcade_action", 1216060): (3, ""),  # DNF Duel
    ("mix2_arcade_action", 1273440): (2, ""),  # OverDrift Festival
    ("mix2_arcade_action", 1372110): (3, ""),  # 죠죠의 기묘한 모험 올 스타 배틀 R
    ("mix2_arcade_action", 1981820): (2, ""),  # Strike Force Heroes
    ("mix2_arcade_action", 2478970): (1, "GENRE_ONLY"),  # Tomb Raider I-III Remastered Starring La
    ("mix2_arcade_action", 3404260): (3, ""),  # Dead as Disco
    ("mix2_builder_sim", 24780): (3, ""),  # SimCity™ 4 Deluxe Edition
    ("mix2_builder_sim", 424590): (3, ""),  # Farm Expert 2017
    ("mix2_builder_sim", 514290): (3, ""),  # Factory Engineer
    ("mix2_builder_sim", 574720): (3, ""),  # Little Big Workshop
    ("mix2_builder_sim", 704030): (3, ""),  # Professional Farmer: Cattle and Crops
    ("mix2_builder_sim", 1031270): (3, ""),  # Farming Life
    ("mix2_builder_sim", 1127400): (3, ""),  # Mindustry
    ("mix2_builder_sim", 1411740): (3, ""),  # Urbek City Builder
    ("mix2_builder_sim", 1416960): (2, ""),  # Everafter Falls
    ("mix2_builder_sim", 2244130): (3, ""),  # Ratopia
    ("mix2_builder_sim", 2570210): (3, ""),  # Eden Crafters
    ("mix2_builder_sim", 2825870): (2, ""),  # Doodle Factory
    ("mix2_builder_sim", 2827680): (2, ""),  # Block Factory
    ("mix2_builder_sim", 3184110): (3, ""),  # Widget Inc.
    ("mix2_builder_sim", 4039140): (3, ""),  # Subway Builder
    ("mix2_colony_cozy", 246090): (3, ""),  # Spacebase DF-9
    ("mix2_colony_cozy", 284950): (1, "IRRELEVANT"),  # Pixel Puzzles: Japan
    ("mix2_colony_cozy", 454060): (2, ""),  # Blueprint Tycoon
    ("mix2_colony_cozy", 508600): (1, "GENRE_ONLY"),  # ROD: Revolt Of Defense
    ("mix2_colony_cozy", 860890): (3, ""),  # Factory Town
    ("mix2_colony_cozy", 1113120): (3, ""),  # IXION
    ("mix2_colony_cozy", 1180130): (3, ""),  # The Final Earth 2
    ("mix2_colony_cozy", 1333910): (2, ""),  # Sizeable
    ("mix2_colony_cozy", 1883840): (2, ""),  # Ocean Punk
    ("mix2_colony_cozy", 3482510): (3, ""),  # 패킹 라이프
    ("mix2_colony_cozy", 3609750): (3, ""),  # Organized Inside
    ("mix2_colony_cozy", 3693650): (2, ""),  # 우리동네 중고상회
    ("mix2_colony_cozy", 4033610): (3, ""),  # Ecopunk
    ("mix2_colony_cozy", 4244170): (3, ""),  # Cozy Home
    ("mix2_colony_cozy", 4468180): (3, ""),  # AI Society
    ("mix2_coop_horror", 248390): (2, ""),  # Craft The World
    ("mix2_coop_horror", 395570): (1, "GENRE_ONLY"),  # We Are The Dwarves
    ("mix2_coop_horror", 644480): (3, ""),  # Outbreak: The New Nightmare
    ("mix2_coop_horror", 696220): (3, ""),  # Folklore Hunter
    ("mix2_coop_horror", 1555980): (2, ""),  # Mother Machine
    ("mix2_coop_horror", 1577320): (2, ""),  # Somewhere in the Shadow
    ("mix2_coop_horror", 2221490): (1, "MODE_MISMATCH"),  # Tom Clancy’s The Division® 2
    ("mix2_coop_horror", 2242760): (3, ""),  # The Escape: Together
    ("mix2_coop_horror", 2772990): (3, ""),  # Ghost Janitors
    ("mix2_coop_horror", 2846060): (3, ""),  # Subterror
    ("mix2_coop_horror", 3216340): (1, "GENRE_ONLY"),  # Tearscape
    ("mix2_coop_horror", 3253530): (3, ""),  # PIT OF GOBLIN
    ("mix2_coop_horror", 3924610): (3, ""),  # MIND BIOTICS
    ("mix2_coop_horror", 3937360): (3, ""),  # Ocean Keeper Co-op Drill Multiplayer
    ("mix2_crpg_sandbox", 333640): (3, ""),  # Caves of Qud
    ("mix2_crpg_sandbox", 414950): (2, ""),  # Star Nomad 2
    ("mix2_crpg_sandbox", 626690): (1, "KEYWORD_MATCH"),  # Sword Art Online: Fatal Bullet
    ("mix2_crpg_sandbox", 704450): (3, ""),  # Neverwinter Nights: Enhanced Edition
    ("mix2_crpg_sandbox", 758870): (1, "GENRE_ONLY"),  # Kynseed
    ("mix2_crpg_sandbox", 1062810): (2, ""),  # Inkbound
    ("mix2_crpg_sandbox", 1094520): (3, ""),  # 部落与弯刀 Sands of Salzaar
    ("mix2_crpg_sandbox", 1134700): (2, ""),  # Wild Terra 2: New Lands
    ("mix2_crpg_sandbox", 1967820): (2, ""),  # Border Town
    ("mix2_crpg_sandbox", 2774040): (2, ""),  # The Boss Gangster: Criminal Empire
    ("mix2_crpg_sandbox", 2886220): (2, ""),  # Archipelago: Island Survival
    ("mix2_crpg_sandbox", 2953520): (3, ""),  # The End of History
    ("mix2_crpg_sandbox", 3372530): (3, ""),  # Valorborn
    ("mix2_modern_roguelite", 250680): (2, ""),  # BELOW
    ("mix2_modern_roguelite", 497800): (2, ""),  # Golden Krone Hotel
    ("mix2_modern_roguelite", 740080): (2, ""),  # Deadly Days
    ("mix2_modern_roguelite", 858820): (2, ""),  # Tribes of Midgard
    ("mix2_modern_roguelite", 904380): (2, ""),  # Vambrace: Cold Soul
    ("mix2_modern_roguelite", 1304810): (2, ""),  # HANI
    ("mix2_modern_roguelite", 1394130): (3, ""),  # Breach Wanderers
    ("mix2_modern_roguelite", 2026820): (3, ""),  # Die in the Dungeon
    ("mix2_modern_roguelite", 2330750): (1, "MODE_MISMATCH"),  # Cataclysm: Dark Days Ahead
    ("mix2_modern_roguelite", 2400510): (3, ""),  # Dungeons & Degenerate Gamblers
    ("mix2_modern_roguelite", 2564520): (3, ""),  # Lost in Random: The Eternal Die
    ("mix2_modern_roguelite", 2960490): (2, ""),  # Royal Revolt Survivors
    ("mix2_modern_roguelite", 3264570): (3, ""),  # 마지막 집주인
    ("mix2_party_narrative", 258520): (3, ""),  # The Vanishing of Ethan Carter
    ("mix2_party_narrative", 383870): (3, ""),  # Firewatch
    ("mix2_party_narrative", 749580): (3, ""),  # All That Remains
    ("mix2_party_narrative", 790740): (3, ""),  # Tick Tock: A Tale for Two
    ("mix2_party_narrative", 1004490): (3, ""),  # Tools Up!
    ("mix2_party_narrative", 1010860): (1, "LOW_QUALITY"),  # Hide and Seek
    ("mix2_party_narrative", 1126600): (3, ""),  # Before I Forget
    ("mix2_party_narrative", 1412850): (3, ""),  # Spacelines from the Far Out
    ("mix2_party_narrative", 1662570): (2, ""),  # 구조대작전: 라이브! / Rescue Party: Live!
    ("mix2_party_narrative", 2280430): (3, ""),  # The Berlin Apartment
    ("mix2_party_narrative", 2336440): (2, ""),  # 침묵의 땅
    ("mix2_party_narrative", 3445340): (2, ""),  # Sandwich Simulator
    ("mix2_party_narrative", 3795760): (2, ""),  # Burger Bots Inc.
    ("mix2_party_narrative", 4082750): (3, ""),  # Log Riders
    ("mix2_puzzle_survival", 320820): (1, "GENRE_ONLY"),  # Toren
    ("mix2_puzzle_survival", 646570): (3, ""),  # Slay the Spire
    ("mix2_puzzle_survival", 648800): (2, ""),  # Raft
    ("mix2_puzzle_survival", 757300): (2, ""),  # Truberbrook / Trüberbrook
    ("mix2_puzzle_survival", 769560): (2, ""),  # 나이트오브풀문
    ("mix2_puzzle_survival", 1253860): (2, ""),  # Neurodeck : Psychological Deckbuilder
    ("mix2_puzzle_survival", 1371720): (3, ""),  # Card Shark
    ("mix2_puzzle_survival", 1772910): (2, ""),  # Dead District: Survival
    ("mix2_puzzle_survival", 2155770): (3, ""),  # THE MULLER-POWELL PRINCIPLE
    ("mix2_puzzle_survival", 2682580): (2, ""),  # DUCKSIDE
    ("mix2_puzzle_survival", 3079600): (3, ""),  # Au Revoir
    ("mix2_puzzle_survival", 3532360): (2, ""),  # 해적의 도박
    ("mix2_soulslike_narrative", 31830): (2, ""),  # Nancy Drew®: Curse of Blackmoor Manor
    ("mix2_soulslike_narrative", 39160): (1, "GENRE_ONLY"),  # Dungeon Siege III
    ("mix2_soulslike_narrative", 334420): (3, ""),  # Out There: Ω Edition
    ("mix2_soulslike_narrative", 388410): (2, ""),  # Darksiders II Deathinitive Edition
    ("mix2_soulslike_narrative", 768450): (2, ""),  # NUTS
    ("mix2_soulslike_narrative", 890720): (3, ""),  # In Other Waters
    ("mix2_soulslike_narrative", 1132980): (2, ""),  # 빛 없는 세계: There is No Light
    ("mix2_soulslike_narrative", 1444920): (2, ""),  # Doctor Who: The Edge of Reality
    ("mix2_soulslike_narrative", 1501750): (3, ""),  # Lords of the Fallen
    ("mix2_soulslike_narrative", 1656930): (1, "GENRE_ONLY"),  # Coridden
    ("mix2_soulslike_narrative", 4067130): (1, "LOW_QUALITY"),  # Drifter Star: Evolution
    ("mix2_survival_farm", 360170): (2, ""),  # How to Survive 2
    ("mix2_survival_farm", 706020): (2, ""),  # Fade to Silence
    ("mix2_survival_farm", 739650): (3, ""),  # Drake Hollow
    ("mix2_survival_farm", 815370): (3, ""),  # Green Hell
    ("mix2_survival_farm", 1137750): (2, ""),  # Farmer's Life
    ("mix2_survival_farm", 1316230): (3, ""),  # Force of Nature 2: Ghost Keeper
    ("mix2_survival_farm", 1335830): (3, ""),  # Len's Island
    ("mix2_survival_farm", 1928980): (3, ""),  # Nightingale: 나이팅게일
    ("mix2_survival_farm", 2142790): (3, ""),  # Fields of Mistria
    ("mix2_survival_farm", 2340520): (3, ""),  # 세이큐 이야기
    ("mix2_survival_farm", 2496090): (2, ""),  # Whisper of the Swallows
    ("niche_cozy_casual", 204060): (2, ""),  # Superbrothers: Sword & Sworcery EP
    ("niche_cozy_casual", 438730): (2, ""),  # Poly Towns
    ("niche_cozy_casual", 742490): (1, "IRRELEVANT"),  # Nonogram - The Greatest Painter
    ("niche_cozy_casual", 938560): (2, ""),  # INMOST
    ("niche_cozy_casual", 1141220): (2, ""),  # DemonCrawl
    ("niche_cozy_casual", 1552740): (3, ""),  # First Days of Atlantis
    ("niche_cozy_casual", 1655670): (2, ""),  # Rob Riches
    ("niche_cozy_casual", 1924360): (2, ""),  # Lil' Guardsman
    ("niche_cozy_casual", 1956670): (3, ""),  # Biomisland
    ("niche_cozy_casual", 2141770): (3, ""),  # URBO
    ("niche_cozy_casual", 2538390): (0, "LOW_QUALITY"),  # Angry Penguin
    ("niche_puzzle_solo", 759000): (2, ""),  # .projekt
    ("niche_puzzle_solo", 976010): (1, "IRRELEVANT"),  # I hate this game
    ("niche_puzzle_solo", 977230): (2, ""),  # 화성 전력 회사 디럭스
    ("niche_puzzle_solo", 984800): (3, ""),  # Automachef
    ("niche_puzzle_solo", 1226990): (2, ""),  # Mechanica
    ("niche_puzzle_solo", 1381650): (1, "IRRELEVANT"),  # ACTION SANDBOX
    ("niche_puzzle_solo", 1409160): (3, ""),  # Plasma
    ("niche_puzzle_solo", 1720850): (3, ""),  # A=B
    ("niche_roguelite", 218820): (2, ""),  # Mercenary Kings: Reloaded Edition
    ("niche_roguelite", 509570): (3, ""),  # In Celebration of Violence
    ("niche_roguelite", 722560): (3, ""),  # RAD
    ("niche_roguelite", 1389360): (2, ""),  # 메크 아르마
    ("niche_roguelite", 1416050): (3, ""),  # Shovel Knight Dig
    ("niche_roguelite", 1538970): (2, ""),  # Hammerwatch II
    ("niche_roguelite", 1679510): (2, ""),  # Plushie from the Sky
    ("niche_roguelite", 1815530): (2, ""),  # Dungeon Defenders: Going Rogue
    ("niche_roguelite", 2292010): (2, ""),  # Earl vs. the Mutants
    ("niche_roguelite", 3003120): (3, ""),  # Zombiehood
    ("niche_sim", 230290): (2, ""),  # Universe Sandbox
    ("niche_sim", 414720): (2, ""),  # Astrox: Hostile Space Excavation
    ("niche_sim", 514290): (3, ""),  # Factory Engineer
    ("niche_sim", 716490): (3, ""),  # EXAPUNKS
    ("niche_sim", 983870): (3, ""),  # FOUNDRY
    ("niche_sim", 1318690): (3, ""),  # shapez
    ("niche_sim", 2242760): (2, ""),  # The Escape: Together
    ("niche_sim", 4618590): (3, ""),  # The Silver Crow
    ("niche_soulslike_solo", 219830): (0, "MODE_MISMATCH"),  # King Arthur's Gold
    ("niche_soulslike_solo", 307880): (1, "GENRE_ONLY"),  # Savage Lands
    ("niche_soulslike_solo", 356650): (3, ""),  # Death's Gambit: Afterlife
    ("niche_soulslike_solo", 630720): (3, ""),  # Mana Spark
    ("niche_soulslike_solo", 1098080): (3, ""),  # 3000th Duel
    ("niche_soulslike_solo", 1132980): (2, ""),  # 빛 없는 세계: There is No Light
    ("niche_soulslike_solo", 1413990): (3, ""),  # The Heart of Darkness
    ("niche_soulslike_solo", 1437400): (3, ""),  # Salt and Sacrifice
    ("niche_soulslike_solo", 1446900): (3, ""),  # Fallen Tear: The Ascension
    ("niche_soulslike_solo", 1811330): (1, "MODE_MISMATCH"),  # 불가사의의 던전
    ("niche_soulslike_solo", 1911360): (3, ""),  # Trinity Fusion
    ("niche_soulslike_solo", 4729900): (1, "GENRE_ONLY"),  # FussyCraft Survival: Sandbox
    ("niche_tactics", 237350): (3, ""),  # Frozen Cortex
    ("niche_tactics", 253150): (3, ""),  # Halfway
    ("niche_tactics", 614630): (2, ""),  # Tiny Rails
    ("niche_tactics", 874390): (3, ""),  # The Battle of Polytopia
    ("niche_tactics", 955170): (2, ""),  # 铁道物语：陆王（Railway Saga:Land King）
    ("niche_tactics", 984680): (2, ""),  # Interstellar Space: Genesis
    ("niche_tactics", 1075740): (3, ""),  # Banners of Ruin
    ("niche_tactics", 1612770): (3, ""),  # Sweet Transit
    ("niche_tactics", 1769830): (3, ""),  # As We Descend: 심연 속으로
    ("niche_tactics", 1877650): (3, ""),  # Zero Orders Tactics
    ("niche_tactics", 2095480): (3, ""),  # Simple Trains
    ("niche_tactics", 2468100): (2, ""),  # Pyrene
    ("niche_tactics", 4039140): (3, ""),  # Subway Builder
    ("niche_tactics", 4733420): (3, ""),  # Metro Architect
}


BLIND_K50_P4 = {
    ("coh_arpg", 265300): (3, ""),  # Lords Of The Fallen™ 2014
    ("coh_arpg", 385070): (2, ""),  # Atonement: Scourge of Time
    ("coh_arpg", 388410): (2, ""),  # Darksiders II Deathinitive Edition
    ("coh_arpg", 556740): (2, ""),  # LEGRAND LEGACY: Tale of the Fatebounds
    ("coh_arpg", 961200): (1, "MODE_MISMATCH"),  # Predecessor
    ("coh_arpg", 963450): (2, ""),  # The Eternal Castle [REMASTERED]
    ("coh_arpg", 973230): (2, ""),  # We Who Are About To Die
    ("coh_arpg", 1247100): (2, ""),  # SpellMaster: The Saga
    ("coh_arpg", 1580800): (1, "MODE_MISMATCH"),  # Erannorth Chronicles
    ("coh_arpg", 1804470): (2, ""),  # Duel Corp.
    ("coh_classic_multi", 42700): (3, ""),  # Call of Duty®: Black Ops
    ("coh_classic_multi", 45740): (3, ""),  # Dead Rising® 2
    ("coh_classic_multi", 239140): (3, ""),  # Dying Light
    ("coh_classic_multi", 604240): (3, ""),  # Shotgun Farmers
    ("coh_classic_multi", 644480): (3, ""),  # Outbreak: The New Nightmare
    ("coh_classic_multi", 654990): (2, ""),  # Dude Simulator
    ("coh_classic_multi", 786450): (3, ""),  # Don't Bite Me Bro! +
    ("coh_classic_multi", 1118200): (3, ""),  # People Playground
    ("coh_classic_multi", 1228910): (3, ""),  # Ragdoll Party Online
    ("coh_classic_multi", 1286680): (2, ""),  # 타이니 티나의 원더랜드
    ("coh_classic_multi", 1381650): (2, ""),  # ACTION SANDBOX
    ("coh_classic_multi", 2265640): (2, ""),  # X-MODE
    ("coh_classic_multi", 3168930): (2, ""),  # Infect Cam
    ("coh_classic_multi", 3922920): (3, ""),  # PONE
    ("coh_cozy", 378370): (1, "MODE_MISMATCH"),  # Nomad
    ("coh_cozy", 878520): (1, "IRRELEVANT"),  # Sally's Salon: Kiss & Make-Up
    ("coh_cozy", 897450): (3, ""),  # The Survivalists
    ("coh_cozy", 1031270): (3, ""),  # Farming Life
    ("coh_cozy", 1515320): (3, ""),  # Harvest Days: My Dream Farm
    ("coh_cozy", 1603410): (3, ""),  # Lost Nova
    ("coh_cozy", 1629830): (3, ""),  # Research Story
    ("coh_cozy", 1635590): (0, "IRRELEVANT"),  # 내 친구 페파
    ("coh_cozy", 1889810): (2, ""),  # 보보 베이: 뽀짝 몬스터 대작전
    ("coh_cozy", 1990530): (2, ""),  # Sailing alone:Aftermath
    ("coh_cozy", 2508780): (3, ""),  # 목장이야기 Let's! 바람의 그랜드 바자르
    ("coh_cozy", 2845080): (1, "LOW_QUALITY"),  # SHARK SIEGE - TOGETHER SURVIVAL
    ("coh_cozy", 3913270): (1, "KEYWORD_MATCH"),  # Slimeward
    ("coh_fps", 32770): (2, ""),  # Joint Operations: Combined Arms Gold
    ("coh_fps", 319540): (1, "MODE_MISMATCH"),  # A Year Of Rain
    ("coh_fps", 334040): (2, ""),  # Down To One
    ("coh_fps", 386070): (1, "MODE_MISMATCH"),  # Planetary Annihilation: TITANS
    ("coh_fps", 485610): (1, "LOW_QUALITY"),  # 축구 온라인: 볼3D
    ("coh_fps", 750470): (2, ""),  # War Brokers
    ("coh_fps", 916930): (2, ""),  # War Trigger Classic
    ("coh_fps", 980850): (1, "LOW_QUALITY"),  # Pixel Battle Royale
    ("coh_fps", 1721110): (2, ""),  # Abyssus
    ("coh_fps", 2073850): (3, ""),  # THE FINALS
    ("coh_fps", 2089250): (3, ""),  # Alpha Response
    ("coh_grand_strategy", 10500): (3, ""),  # 토탈워: 엠파이어 - 데피니티브 에디션
    ("coh_grand_strategy", 35450): (1, "MODE_MISMATCH"),  # Red Orchestra 2: Heroes of Stalingrad wi
    ("coh_grand_strategy", 312360): (3, ""),  # To End All Wars
    ("coh_grand_strategy", 356430): (1, "IRRELEVANT"),  # Chris Sawyer's Locomotion™
    ("coh_grand_strategy", 410980): (3, ""),  # Master of Orion 2
    ("coh_grand_strategy", 411320): (2, ""),  # Codename: Panzers, Phase Two
    ("coh_grand_strategy", 826630): (2, ""),  # Iron Harvest
    ("coh_grand_strategy", 957720): (3, ""),  # Strategic Command WWII: World at War
    ("coh_grand_strategy", 1084790): (3, ""),  # WarPlan
    ("coh_grand_strategy", 1286010): (2, ""),  # Godless Tactics
    ("coh_grand_strategy", 1599130): (0, "IRRELEVANT"),  # Pavel Repin's Collection
    ("coh_grand_strategy", 2021880): (3, ""),  # Ara History Untold: Anniversary Edition
    ("coh_grand_strategy", 2186320): (2, ""),  # Ages of Conflict: World War Simulator
    ("coh_grand_strategy", 2275440): (3, ""),  # Solar Nations 2
    ("coh_grand_strategy", 2472140): (1, "IRRELEVANT"),  # ALL Chess
    ("coh_grand_strategy", 3204120): (3, ""),  # Maestro's Cold War 2
    ("coh_indie_platformer", 355150): (3, ""),  # gravilon
    ("coh_indie_platformer", 400630): (2, ""),  # Wuppo: Definitive Edition
    ("coh_indie_platformer", 842910): (2, ""),  # The MISSING: J.J. Macfield and the Islan
    ("coh_indie_platformer", 922050): (3, ""),  # DOOMBLADE
    ("coh_indie_platformer", 1145350): (3, ""),  # Hades II
    ("coh_indie_platformer", 1454540): (3, ""),  # 라핀
    ("coh_indie_platformer", 1836030): (3, ""),  # TOOK: The Immortal Hero
    ("coh_indie_platformer", 2620730): (3, ""),  # Deviator
    ("coh_indie_platformer", 2780710): (2, ""),  # Asgard's Fall — Viking Survivors
    ("coh_indie_platformer", 2947280): (3, ""),  # 솔라테리아
    ("coh_indie_platformer", 3642650): (2, ""),  # Boom Ascent
    ("coh_indie_platformer", 4271160): (1, "IRRELEVANT"),  # Zero Stress King: Idle Defense
    ("coh_openworld_survival", 324260): (2, ""),  # VoidExpanse
    ("coh_openworld_survival", 327090): (2, ""),  # Out of Reach
    ("coh_openworld_survival", 371660): (2, ""),  # Far Cry® Primal
    ("coh_openworld_survival", 391730): (3, ""),  # Crashlands
    ("coh_openworld_survival", 568570): (2, ""),  # Force of Nature
    ("coh_openworld_survival", 934630): (2, ""),  # Rocky Planet
    ("coh_openworld_survival", 1054510): (2, ""),  # Survivalist: Invisible Strain
    ("coh_openworld_survival", 1137490): (2, ""),  # Invasion 2037
    ("coh_openworld_survival", 1380220): (3, ""),  # Starsand
    ("coh_openworld_survival", 2868100): (2, ""),  # Shamania
    ("coh_openworld_survival", 3081000): (3, ""),  # Ember's Verge
    ("coh_strategy", 8500): (2, ""),  # 이브 온라인
    ("coh_strategy", 38420): (3, ""),  # Fallout Tactics: Brotherhood of Steel
    ("coh_strategy", 244770): (2, ""),  # StarMade
    ("coh_strategy", 383120): (2, ""),  # Empyrion - Galactic Survival
    ("coh_strategy", 572050): (3, ""),  # Call to Power II
    ("coh_strategy", 826630): (3, ""),  # Iron Harvest
    ("coh_strategy", 921710): (3, ""),  # Galaxy Squad
    ("coh_strategy", 1286010): (2, ""),  # Godless Tactics
    ("coh_strategy", 1466860): (3, ""),  # Age of Empires IV: Anniversary Edition
    ("coh_strategy", 1983990): (3, ""),  # Nexus 5X
    ("coh_strategy", 2340500): (2, ""),  # 창조 삼국지
    ("coh_strategy", 2421430): (2, ""),  # War For Galaxy: New Era
    ("coh_strategy", 2750240): (1, ""),  # AstroForge: Space Pirates
    ("coh_survival_craft", 223490): (1, "GENRE_ONLY"),  # Blockscape
    ("coh_survival_craft", 239140): (3, ""),  # Dying Light
    ("coh_survival_craft", 305780): (3, ""),  # Echo of the Wilds
    ("coh_survival_craft", 340490): (3, ""),  # Subterrain
    ("coh_survival_craft", 622420): (3, ""),  # Stay Alive: Apocalypse
    ("coh_survival_craft", 764920): (3, ""),  # Fear the Night - 恐惧之夜
    ("coh_survival_craft", 786450): (2, ""),  # Don't Bite Me Bro! +
    ("coh_survival_craft", 809210): (3, ""),  # LifeZ - Survival
    ("coh_survival_craft", 1017180): (2, ""),  # The Long Drive
    ("coh_survival_craft", 1027820): (1, "MODE_MISMATCH"),  # Hand Simulator: Horror
    ("coh_survival_craft", 1343520): (3, ""),  # Survival: Lost Way
    ("coh_survival_craft", 1645820): (3, ""),  # SurrounDead
    ("coh_survival_craft", 2075580): (3, ""),  # Under A New Sun
    ("coh_survival_craft", 3616260): (2, ""),  # MAZEBOUND: Hunt, Gather, Run!
    ("coh_vehicle_sim", 302710): (2, ""),  # BlazeRush
    ("coh_vehicle_sim", 362780): (3, ""),  # 18 Wheels of Steel: Extreme Trucker 2
    ("coh_vehicle_sim", 396900): (2, ""),  # GRIP: Combat Racing
    ("coh_vehicle_sim", 588030): (3, ""),  # Derail Valley
    ("coh_vehicle_sim", 1017180): (3, ""),  # The Long Drive
    ("coh_vehicle_sim", 1452250): (3, ""),  # Underground Garage
    ("coh_vehicle_sim", 1592280): (1, "GENRE_ONLY"),  # Selaco
    ("coh_vehicle_sim", 1675090): (2, ""),  # Car Physics Simulator
    ("coh_vehicle_sim", 2143770): (2, ""),  # Drift racing car
    ("coh_vehicle_sim", 2292440): (2, ""),  # inRun
    ("coh_vehicle_sim", 3396460): (2, ""),  # Urban Shadows Racing™ Tokyo
    ("coh_vehicle_sim", 4118250): (2, ""),  # ARC_the beginning
    ("coh_vehicle_sim", 4148530): (3, ""),  # City Transport Simulator 2026 - Bus & Tr
    ("longtail_deckbuilder", 548370): (2, ""),  # Rezrog
    ("longtail_deckbuilder", 975370): (1, "MODE_MISMATCH"),  # Dwarf Fortress
    ("longtail_deckbuilder", 1135810): (3, ""),  # Vault of the Void
    ("longtail_deckbuilder", 1156060): (3, ""),  # Railroads & Catacombs
    ("longtail_deckbuilder", 1229460): (3, ""),  # Dice Legacy
    ("longtail_deckbuilder", 1397130): (3, ""),  # Primateria
    ("longtail_deckbuilder", 1619520): (3, ""),  # Cross Blitz
    ("longtail_deckbuilder", 1866630): (3, ""),  # Throne of Bone
    ("longtail_deckbuilder", 1990110): (3, ""),  # SpellRogue
    ("longtail_deckbuilder", 2179850): (3, ""),  # Cobalt Core
    ("longtail_deckbuilder", 2667120): (3, ""),  # Ballionaire
    ("longtail_deckbuilder", 2746910): (2, ""),  # 삼국 영웅：혈월의 전설
    ("longtail_deckbuilder", 2881800): (2, ""),  # SuperTaxCity
    ("longtail_deckbuilder", 3685190): (3, ""),  # Roll & Reign
    ("longtail_detective", 80340): (3, ""),  # Blackwell Unbound
    ("longtail_detective", 257260): (2, ""),  # Inherit the Earth: Quest for the Orb
    ("longtail_detective", 286480): (2, ""),  # Black Mirror III
    ("longtail_detective", 435120): (3, ""),  # Rusty Lake Hotel
    ("longtail_detective", 656570): (3, ""),  # In the Raven Shadow
    ("longtail_detective", 774201): (3, ""),  # Heaven's Vault
    ("longtail_detective", 850450): (3, ""),  # Escape First
    ("longtail_detective", 1023720): (2, ""),  # Hidden Mysteries: Royal Family Secrets
    ("longtail_detective", 1024160): (2, ""),  # Lord Winklebottom Investigates
    ("longtail_detective", 1685870): (2, ""),  # Near-Mage
    ("longtail_detective", 1742930): (2, ""),  # 올려다 보면 천장 뿐
    ("longtail_detective", 2069040): (3, ""),  # Unboxing the Cryptic Killer
    ("longtail_detective", 3072450): (3, ""),  # FAKEBOOK : 페이크북
    ("longtail_detective", 3148060): (3, ""),  # Penelope Pendrick and the Art of Deceit
    ("longtail_detective", 3176060): (2, ""),  # Emissary Zero
    ("longtail_metroidvania", 368620): (3, ""),  # Timespinner
    ("longtail_metroidvania", 428550): (3, ""),  # Momodora: Reverie Under The Moonlight
    ("longtail_metroidvania", 449610): (3, ""),  # 몬스터 보이와 저주받은 왕국
    ("longtail_metroidvania", 571310): (3, ""),  # SteamWorld Dig 2
    ("longtail_metroidvania", 1225580): (2, ""),  # Fe
    ("longtail_metroidvania", 1349600): (3, ""),  # Slavania
    ("longtail_metroidvania", 1444080): (3, ""),  # Dewdrop Dynasty
    ("longtail_metroidvania", 1446900): (3, ""),  # Fallen Tear: The Ascension
    ("longtail_metroidvania", 1480830): (2, ""),  # Evil Tonight
    ("longtail_metroidvania", 1586700): (2, ""),  # MARSUPILAMI - HOOBADVENTURE
    ("longtail_metroidvania", 1608230): (2, ""),  # Planet of Lana
    ("longtail_metroidvania", 1726760): (2, ""),  # Curse Crackers: For Whom the Belle Toils
    ("longtail_metroidvania", 2085310): (3, ""),  # Emberbane
    ("longtail_metroidvania", 2257790): (2, ""),  # Trapers Platformer
    ("longtail_metroidvania", 2751000): (3, ""),  # 페르시아의 왕자: 잃어버린 왕관
    ("longtail_metroidvania", 3114250): (1, "LOW_QUALITY"),  # ダンジョンビーチ
    ("longtail_metroidvania", 4467090): (3, ""),  # Intertwined
    ("longtail_puzzle_platformer", 239350): (2, ""),  # Spelunky
    ("longtail_puzzle_platformer", 311010): (3, ""),  # The Way
    ("longtail_puzzle_platformer", 399420): (3, ""),  # The Prism
    ("longtail_puzzle_platformer", 720660): (3, ""),  # Overloop
    ("longtail_puzzle_platformer", 1083310): (2, ""),  # OUTBUDDIES DX
    ("longtail_puzzle_platformer", 1181400): (3, ""),  # Path of Giants
    ("longtail_puzzle_platformer", 1632360): (1, "IRRELEVANT"),  # Coolors
    ("longtail_puzzle_platformer", 3840230): (2, ""),  # 사랑스러운 모험기
    ("lowrev_cozy_narrative", 209370): (2, ""),  # Analogue: A Hate Story
    ("lowrev_cozy_narrative", 210970): (1, "MODE_MISMATCH"),  # The Witness
    ("lowrev_cozy_narrative", 307050): (3, ""),  # Shan Gui (山桂)
    ("lowrev_cozy_narrative", 1057680): (2, ""),  # 人间 The Lost We Lost
    ("lowrev_cozy_narrative", 1057750): (2, ""),  # The Suicide of Rachel Foster
    ("lowrev_cozy_narrative", 1337010): (3, ""),  # Alba: A Wildlife Adventure
    ("lowrev_cozy_narrative", 1931940): (2, ""),  # 때 이른 검은 백합
    ("lowrev_cozy_narrative", 2429190): (2, ""),  # Amerzone: 탐험가의 유산
    ("lowrev_cozy_narrative", 2564880): (2, ""),  # Departed Away
    ("lowrev_cozy_narrative", 2600720): (3, ""),  # Cloudy Valley
    ("lowrev_cozy_narrative", 3047750): (3, ""),  # Herdling
    ("lowrev_cozy_narrative", 3394800): (2, ""),  # Koko's Cafe
    ("lowrev_cozy_narrative", 4092670): (1, "GENRE_ONLY"),  # Within of Static - Ashburg Rental
    ("lowrev_deckbuilder", 769560): (2, ""),  # 나이트오브풀문
    ("lowrev_deckbuilder", 1071140): (3, ""),  # ORX
    ("lowrev_deckbuilder", 1201540): (3, ""),  # HELLCARD
    ("lowrev_deckbuilder", 1201830): (3, ""),  # For The Warp
    ("lowrev_deckbuilder", 1385380): (3, ""),  # Across the Obelisk
    ("lowrev_deckbuilder", 2501600): (3, ""),  # DICEOMANCER
    ("lowrev_deckbuilder", 2529010): (3, ""),  # One More Try: Prologue
    ("lowrev_deckbuilder", 2638050): (3, ""),  # Lost For Swords
    ("lowrev_deckbuilder", 2812610): (3, ""),  # Curtain Call
    ("lowrev_deckbuilder", 2824490): (3, ""),  # He is Coming 히 이즈 커밍
    ("lowrev_deckbuilder", 3007390): (3, ""),  # 덕앤롤
    ("lowrev_deckbuilder", 3415570): (2, ""),  # Die For The Lich
    ("lowrev_deckbuilder", 3548520): (3, ""),  # DOG WITCH
    ("lowrev_deckbuilder", 3709000): (3, ""),  # The Drawstring Dungeon
    ("lowrev_detective", 200490): (2, ""),  # Memento Mori
    ("lowrev_detective", 233290): (3, ""),  # Murdered: Soul Suspect
    ("lowrev_detective", 294570): (1, "IRRELEVANT"),  # Freddi Fish 5 featuring Mess Hall Mania®
    ("lowrev_detective", 1055850): (3, ""),  # Inspector Waffles
    ("lowrev_detective", 1159830): (3, ""),  # Scene Investigators
    ("lowrev_detective", 1251980): (3, ""),  # Jessika
    ("lowrev_detective", 1307580): (2, ""),  # TOEM: A Photo Adventure
    ("lowrev_detective", 1414180): (1, "IRRELEVANT"),  # 모태솔로
    ("lowrev_detective", 1787790): (1, "IRRELEVANT"),  # Kirakira Monstars
    ("lowrev_detective", 1977220): (2, ""),  # DELETE AFTER READING
    ("lowrev_detective", 2294450): (3, ""),  # CaseCracker2
    ("lowrev_detective", 2797960): (3, ""),  # 은폐된 살인의 진실들 - 하드코어 본격 추리 탐정 게임
    ("lowrev_detective", 2855580): (2, ""),  # Dr. What & Detective Son
    ("lowrev_detective", 3081240): (2, ""),  # 少女怪奇事件簿:永生
    ("lowrev_detective", 3748110): (3, ""),  # 초동수사 - 증언/증거 추리게임
    ("lowrev_metroidvania", 341870): (2, ""),  # TEMBO THE BADASS ELEPHANT
    ("lowrev_metroidvania", 384280): (2, ""),  # Mute Crimson+
    ("lowrev_metroidvania", 553420): (2, ""),  # TUNIC
    ("lowrev_metroidvania", 595500): (3, ""),  # Freedom Planet 2
    ("lowrev_metroidvania", 634340): (0, "IRRELEVANT"),  # Legend of Himari
    ("lowrev_metroidvania", 1281790): (3, ""),  # SHEEPO
    ("lowrev_metroidvania", 1420080): (3, ""),  # 원더보이・아샤 인 몬스터 월드
    ("lowrev_metroidvania", 1444350): (0, "KEYWORD_MATCH"),  # House
    ("lowrev_metroidvania", 1550760): (3, ""),  # Blast Brigade vs. the Evil Legion of Dr.
    ("lowrev_metroidvania", 1670690): (2, ""),  # Eternity Egg
    ("lowrev_metroidvania", 1848450): (2, ""),  # Nightmare of Decay
    ("lowrev_metroidvania", 2272250): (2, ""),  # Forgive Me Father 2
    ("lowrev_metroidvania", 2433860): (3, ""),  # The Shaman's Ark
    ("lowrev_metroidvania", 2529790): (3, ""),  # GRIME II
    ("lowrev_towerdefense", 22230): (2, ""),  # Rock of Ages
    ("lowrev_towerdefense", 243780): (3, ""),  # PixelJunk™ Monsters Ultimate
    ("lowrev_towerdefense", 336420): (2, ""),  # Bloodsports.TV
    ("lowrev_towerdefense", 356500): (2, ""),  # STAR WARS™ Galactic Battlegrounds Saga
    ("lowrev_towerdefense", 371140): (3, ""),  # Aegis Defenders
    ("lowrev_towerdefense", 412520): (3, ""),  # Evil Defenders
    ("lowrev_towerdefense", 471330): (2, ""),  # VERSUS SQUAD
    ("lowrev_towerdefense", 749800): (3, ""),  # PixelJunk™ Monsters 2
    ("lowrev_towerdefense", 785780): (2, ""),  # OF MICE AND SAND -REVISED-
    ("lowrev_towerdefense", 931280): (3, ""),  # Iron Marines
    ("lowrev_towerdefense", 1516750): (2, ""),  # Alien Marauder
    ("lowrev_towerdefense", 2332980): (2, ""),  # King War [RTS]
    ("lowrev_towerdefense", 3133060): (3, ""),  # 노움들
    ("lowrev_towerdefense", 3978250): (1, "GENRE_ONLY"),  # Sector Space
    ("mix2_arcade_action", 16900): (1, "MODE_MISMATCH"),  # GROUND BRANCH
    ("mix2_arcade_action", 273730): (2, ""),  # Driving School Simulator
    ("mix2_arcade_action", 390560): (3, ""),  # Fantasy Strike
    ("mix2_arcade_action", 412880): (2, ""),  # Drift Streets Japan
    ("mix2_arcade_action", 646910): (3, ""),  # The Crew™ 2
    ("mix2_arcade_action", 655500): (2, ""),  # MX Bikes
    ("mix2_arcade_action", 1358700): (2, ""),  # STRANGER OF PARADISE FINAL FANTASY ORIGI
    ("mix2_arcade_action", 1456760): (3, ""),  # ROBOBEAT
    ("mix2_arcade_action", 1726190): (3, ""),  # No Straight Roads: Encore Edition
    ("mix2_arcade_action", 1951230): (1, "IRRELEVANT"),  # Pizza Possum
    ("mix2_arcade_action", 2483190): (3, ""),  # Forza Horizon 6
    ("mix2_arcade_action", 3803270): (3, ""),  # RhythmStrike
    ("mix2_builder_sim", 92900): (3, ""),  # Agricultural Simulator 2011: Extended Ed
    ("mix2_builder_sim", 486860): (2, ""),  # MMORPG Tycoon 2
    ("mix2_builder_sim", 568570): (2, ""),  # Force of Nature
    ("mix2_builder_sim", 667610): (3, ""),  # Ancient Cities
    ("mix2_builder_sim", 782410): (3, ""),  # Metropolis
    ("mix2_builder_sim", 784150): (3, ""),  # Workers & Resources: Soviet Republic 워커스
    ("mix2_builder_sim", 823950): (2, ""),  # 리: 레전드 Re:Legend
    ("mix2_builder_sim", 1432860): (2, ""),  # Sun Haven
    ("mix2_builder_sim", 2072840): (2, ""),  # Word Factori
    ("mix2_builder_sim", 2389040): (3, ""),  # ShapeHero Factory -쉐이프히어로 팩토리-
    ("mix2_builder_sim", 2795090): (3, ""),  # MR FARMBOY
    ("mix2_builder_sim", 3427850): (3, ""),  # Belts of Iron
    ("mix2_builder_sim", 3474700): (3, ""),  # Plant Nursery Simulator
    ("mix2_builder_sim", 4874860): (3, ""),  # FACTORY I
    ("mix2_colony_cozy", 230290): (2, ""),  # Universe Sandbox
    ("mix2_colony_cozy", 252870): (2, ""),  # PULSAR: Lost Colony
    ("mix2_colony_cozy", 366090): (2, ""),  # Colony Survival
    ("mix2_colony_cozy", 403190): (3, ""),  # Planetbase
    ("mix2_colony_cozy", 596590): (1, "IRRELEVANT"),  # Linked
    ("mix2_colony_cozy", 700820): (2, ""),  # TFM: The First Men
    ("mix2_colony_cozy", 803050): (3, ""),  # Per Aspera
    ("mix2_colony_cozy", 1202730): (2, ""),  # nStations
    ("mix2_colony_cozy", 1318740): (3, ""),  # Farlanders
    ("mix2_colony_cozy", 1629520): (3, ""),  # A Little to the Left
    ("mix2_colony_cozy", 3097690): (2, ""),  # Terra revive
    ("mix2_colony_cozy", 3359320): (1, "IRRELEVANT"),  # SchoolBoy Runaway
    ("mix2_coop_horror", 340520): (1, ""),  # Tallowmere
    ("mix2_coop_horror", 539400): (1, ""),  # Son of a Witch
    ("mix2_coop_horror", 1310510): (3, ""),  # Handy Harry's Haunted House Services
    ("mix2_coop_horror", 1605250): (2, ""),  # 파멸 협약
    ("mix2_coop_horror", 1618540): (3, ""),  # Ghost Exorcism INC.
    ("mix2_coop_horror", 1777600): (3, ""),  # Paranormal Hunter
    ("mix2_coop_horror", 1807080): (3, ""),  # Ghost Exile
    ("mix2_coop_horror", 2204350): (2, ""),  # Midnight Heist
    ("mix2_coop_horror", 2881650): (3, ""),  # Content Warning
    ("mix2_coop_horror", 2963880): (3, ""),  # Murky Divers
    ("mix2_coop_horror", 3237570): (2, ""),  # delivery pals
    ("mix2_coop_horror", 3894440): (3, ""),  # NO BACKUP (백업 없음)
    ("mix2_crpg_sandbox", 344760): (2, ""),  # Reign Of Kings
    ("mix2_crpg_sandbox", 780290): (3, ""),  # Gloomhaven
    ("mix2_crpg_sandbox", 1161830): (3, ""),  # Age of Reforging: The Freelands
    ("mix2_crpg_sandbox", 1818180): (1, "MODE_MISMATCH"),  # The RPG Engine
    ("mix2_crpg_sandbox", 2017480): (2, ""),  # Elengard: Ascension
    ("mix2_crpg_sandbox", 2133520): (2, ""),  # 울루카인의 에르투그룰
    ("mix2_crpg_sandbox", 2186680): (3, ""),  # Warhammer 40,000: Rogue Trader
    ("mix2_crpg_sandbox", 2218970): (2, ""),  # Plains of Pain
    ("mix2_crpg_sandbox", 2343930): (2, ""),  # Eyes of War
    ("mix2_crpg_sandbox", 2399160): (3, ""),  # Soulash 2
    ("mix2_crpg_sandbox", 2738630): (3, ""),  # Dungeons & Dragons Neverwinter Nights 2:
    ("mix2_crpg_sandbox", 3124340): (3, ""),  # Demeo x Dungeons & Dragons: Battlemarked
    ("mix2_crpg_sandbox", 3398110): (1, "IRRELEVANT"),  # The Walking Trade
    ("mix2_modern_roguelite", 237930): (3, ""),  # Transistor
    ("mix2_modern_roguelite", 336940): (2, ""),  # Welcome to Basingstoke
    ("mix2_modern_roguelite", 1065310): (1, "GENRE_ONLY"),  # Evil West
    ("mix2_modern_roguelite", 1098610): (3, ""),  # Crush the Industry
    ("mix2_modern_roguelite", 1280930): (3, ""),  # 애스트럴 어센트
    ("mix2_modern_roguelite", 1494260): (3, ""),  # Loot River
    ("mix2_modern_roguelite", 1811990): (3, ""),  # 와일드프로스트 (Wildfrost)
    ("mix2_modern_roguelite", 2181720): (3, ""),  # Scarlet Tower
    ("mix2_modern_roguelite", 2788310): (3, ""),  # Twilight Survivors
    ("mix2_modern_roguelite", 3057670): (2, ""),  # Pluto
    ("mix2_modern_roguelite", 3219010): (3, ""),  # Fogpiercer 포그피어서
    ("mix2_modern_roguelite", 3284290): (3, ""),  # Moonsigil Atlas 달의 인장
    ("mix2_modern_roguelite", 3600310): (3, ""),  # Texas Twist Poker 3 Bandit's Run
    ("mix2_modern_roguelite", 3965190): (3, ""),  # The Useless Wizard
    ("mix2_party_narrative", 250620): (2, ""),  # Among the Sleep - Enhanced Edition
    ("mix2_party_narrative", 632070): (2, ""),  # The Fidelio Incident
    ("mix2_party_narrative", 1062830): (3, ""),  # Embr
    ("mix2_party_narrative", 1222700): (3, ""),  # A Way Out
    ("mix2_party_narrative", 1271090): (3, ""),  # Let's Cook Together
    ("mix2_party_narrative", 1520380): (2, ""),  # HALF DEAD 3
    ("mix2_party_narrative", 1739070): (3, ""),  # Bone's Cafe
    ("mix2_party_narrative", 2336220): (3, ""),  # Feed the Cups
    ("mix2_party_narrative", 2383120): (1, "LOW_QUALITY"),  # Garten of Banban 4
    ("mix2_party_narrative", 2511290): (3, ""),  # Two Cubes
    ("mix2_party_narrative", 2567870): (3, ""),  # Chained Together
    ("mix2_party_narrative", 2749770): (3, ""),  # Galaxy Burger
    ("mix2_party_narrative", 2828590): (2, ""),  # The Haunting of Joni Evers
    ("mix2_party_narrative", 3644200): (3, ""),  # 2 Cooks 1 Mess
    ("mix2_puzzle_survival", 250400): (2, ""),  # How to Survive
    ("mix2_puzzle_survival", 280520): (2, ""),  # Crea
    ("mix2_puzzle_survival", 381780): (2, ""),  # 80 Days
    ("mix2_puzzle_survival", 441790): (2, ""),  # Fragmented
    ("mix2_puzzle_survival", 477740): (3, ""),  # Zero Escape: The Nonary Games
    ("mix2_puzzle_survival", 574180): (2, ""),  # Rem Survival
    ("mix2_puzzle_survival", 1016730): (3, ""),  # Deck of Ashes
    ("mix2_puzzle_survival", 1049410): (3, ""),  # Superliminal
    ("mix2_puzzle_survival", 1148650): (3, ""),  # The Legend of Bum-Bo
    ("mix2_puzzle_survival", 1803400): (3, ""),  # Beneath Oresa
    ("mix2_puzzle_survival", 2014930): (2, ""),  # 유언장: 모험
    ("mix2_puzzle_survival", 4195010): (2, ""),  # Which hand?
    ("mix2_soulslike_narrative", 912570): (3, ""),  # BEAUTIFUL DESOLATION
    ("mix2_soulslike_narrative", 1088850): (2, ""),  # Marvel's Guardians of the Galaxy
    ("mix2_soulslike_narrative", 1230530): (2, ""),  # Atlas Fallen: Reign Of Sand
    ("mix2_soulslike_narrative", 1371980): (3, ""),  # 노 레스트 포 더 위키드
    ("mix2_soulslike_narrative", 2492290): (2, ""),  # 언커버 더 스모킹 건
    ("mix2_soulslike_narrative", 3539440): (2, ""),  # Metal Garden | 메탈 가든
    ("mix2_soulslike_narrative", 3722660): (2, ""),  # Cosmodrill
    ("mix2_soulslike_narrative", 3864530): (2, ""),  # 炽澜号
    ("mix2_survival_farm", 313120): (3, ""),  # Stranded Deep
    ("mix2_survival_farm", 391730): (3, ""),  # Crashlands
    ("mix2_survival_farm", 427410): (3, ""),  # Abiotic Factor
    ("mix2_survival_farm", 644480): (1, "GENRE_ONLY"),  # Outbreak: The New Nightmare
    ("mix2_survival_farm", 648800): (3, ""),  # Raft
    ("mix2_survival_farm", 858820): (3, ""),  # Tribes of Midgard
    ("mix2_survival_farm", 1045430): (2, ""),  # Circadian City
    ("mix2_survival_farm", 1084600): (3, ""),  # My Time at Sandrock
    ("mix2_survival_farm", 1270010): (2, ""),  # Gone: Survival
    ("mix2_survival_farm", 1604030): (3, ""),  # V Rising
    ("mix2_survival_farm", 1635450): (3, ""),  # Longvinter
    ("mix2_survival_farm", 2111170): (3, ""),  # 목장이야기 Welcome! 원더풀 라이프
    ("mix2_survival_farm", 2442490): (1, "GENRE_ONLY"),  # THE LAST BREATH
    ("mix2_survival_farm", 2455370): (2, ""),  # Russian Village Simulator
    ("niche_cozy_casual", 355760): (2, ""),  # Drizzlepath
    ("niche_cozy_casual", 394700): (3, ""),  # Karma. Incarnation 1
    ("niche_cozy_casual", 733070): (1, "IRRELEVANT"),  # Sudoku Universe / 数独宇宙
    ("niche_cozy_casual", 734920): (2, ""),  # MEANDERS
    ("niche_cozy_casual", 785890): (3, ""),  # Hexologic
    ("niche_cozy_casual", 1075200): (3, ""),  # TOHU
    ("niche_cozy_casual", 1155880): (2, ""),  # The Bonfire 2: Uncharted Shores
    ("niche_cozy_casual", 1216590): (2, ""),  # Lone Land
    ("niche_cozy_casual", 2013240): (1, "IRRELEVANT"),  # Immortal Family
    ("niche_cozy_casual", 2291390): (3, ""),  # Peaceful Lands
    ("niche_cozy_casual", 2434600): (3, ""),  # ShantyTown
    ("niche_cozy_casual", 2440320): (2, ""),  # Gobbo goes adventures
    ("niche_cozy_casual", 3206790): (1, "LOW_QUALITY"),  # JustJump!
    ("niche_cozy_casual", 3655260): (3, ""),  # My Tiny Landscape
    ("niche_cozy_casual", 4032790): (1, "LOW_QUALITY"),  # Mary's Quest
    ("niche_cozy_casual", 4671940): (3, ""),  # Enclaved
    ("niche_puzzle_solo", 241240): (2, ""),  # Contraption Maker
    ("niche_puzzle_solo", 385250): (1, "IRRELEVANT"),  # Paint it Back
    ("niche_puzzle_solo", 654580): (3, ""),  # Senalux
    ("niche_puzzle_solo", 733070): (1, "IRRELEVANT"),  # Sudoku Universe / 数独宇宙
    ("niche_puzzle_solo", 1137350): (3, ""),  # Filament
    ("niche_puzzle_solo", 1584170): (3, ""),  # Block Machine
    ("niche_puzzle_solo", 1636730): (3, ""),  # functional
    ("niche_puzzle_solo", 2423620): (2, ""),  # 星际工业国
    ("niche_puzzle_solo", 4273120): (1, "GENRE_ONLY"),  # Particul
    ("niche_roguelite", 364420): (2, ""),  # Roguelands
    ("niche_roguelite", 539400): (3, ""),  # Son of a Witch
    ("niche_roguelite", 588690): (2, ""),  # Peace, Death!
    ("niche_roguelite", 630720): (3, ""),  # Mana Spark
    ("niche_roguelite", 1794780): (1, "IRRELEVANT"),  # SiNiSistar Lite Version
    ("niche_roguelite", 2068280): (3, ""),  # Nordic Ashes: Survivors of Ragnarok
    ("niche_roguelite", 2427410): (1, "MODE_MISMATCH"),  # S.T.A.L.K.E.R.: Shadow of Chornobyl 인핸스드
    ("niche_roguelite", 2904290): (2, ""),  # RollScape
    ("niche_roguelite", 3512300): (1, "IRRELEVANT"),  # 데드 핑거 게임
    ("niche_sim", 1510): (3, ""),  # Uplink
    ("niche_sim", 359400): (2, ""),  # Why Am I Dead At Sea
    ("niche_sim", 444350): (2, ""),  # HACK_IT
    ("niche_sim", 860890): (2, ""),  # Factory Town
    ("niche_sim", 919260): (2, ""),  # Final Upgrade
    ("niche_sim", 1011190): (3, ""),  # SIMULACRA 2
    ("niche_sim", 1466390): (2, ""),  # Kathy Rain 2: Soothsayer
    ("niche_sim", 1475310): (2, ""),  # Factory Magnate
    ("niche_sim", 1718870): (2, ""),  # Spaceflight Simulator
    ("niche_sim", 1738610): (3, ""),  # First Class Escape: The Train of Thought
    ("niche_sim", 1925320): (3, ""),  # Escape Memoirs: Mansion Heist
    ("niche_sim", 2183320): (3, ""),  # 당신의 안녕을 위하여
    ("niche_sim", 3641800): (3, ""),  # Murder at Ironwood Inn
    ("niche_sim", 4119130): (3, ""),  # HELLO HACKER
    ("niche_soulslike_solo", 265300): (3, ""),  # Lords Of The Fallen™ 2014
    ("niche_soulslike_solo", 320040): (1, "MODE_MISMATCH"),  # Moon Hunters
    ("niche_soulslike_solo", 330270): (2, ""),  # Warlocks vs Shadows
    ("niche_soulslike_solo", 409450): (2, ""),  # The Fall of the Dungeon Guardians - Enha
    ("niche_soulslike_solo", 1158690): (1, "IRRELEVANT"),  # Uncharted Ocean
    ("niche_soulslike_solo", 1315180): (3, ""),  # Spark in the Dark
    ("niche_soulslike_solo", 1316230): (1, "GENRE_ONLY"),  # Force of Nature 2: Ghost Keeper
    ("niche_soulslike_solo", 1419160): (3, ""),  # Souldiers
    ("niche_soulslike_solo", 1676380): (1, "LOW_QUALITY"),  # Kingdom of Atham: Crown of the Champions
    ("niche_soulslike_solo", 1714900): (1, "GENRE_ONLY"),  # First Dwarf
    ("niche_tactics", 304730): (3, ""),  # Train Fever
    ("niche_tactics", 493080): (3, ""),  # Card Quest
    ("niche_tactics", 780210): (3, ""),  # Freeways
    ("niche_tactics", 930780): (3, ""),  # Blood Card
    ("niche_tactics", 954650): (3, ""),  # Druidstone: The Secret of the Menhir For
    ("niche_tactics", 1163740): (2, ""),  # Card Hog
    ("niche_tactics", 1184840): (2, ""),  # Chaos Galaxy
    ("niche_tactics", 1267470): (3, ""),  # Warriors of the Nile
    ("niche_tactics", 1331210): (3, ""),  # Wolfstride
    ("niche_tactics", 1638390): (3, ""),  # Indies' Lies
    ("niche_tactics", 1685460): (3, ""),  # A열차로 가자: 시작해요 관광지 개발
    ("niche_tactics", 2452820): (2, ""),  # Skogdal
    ("niche_tactics", 4219060): (3, ""),  # Idle Ways
}


BLIND_K50_P5 = {
    ("coh_arpg", 207170): (2, ""),  # Legend of Grimrock
    ("coh_arpg", 208730): (2, ""),  # Game of Thrones
    ("coh_arpg", 503830): (1, "LOW_QUALITY"),  # The Life Of Greather
    ("coh_arpg", 632470): (3, ""),  # Disco Elysium - The Final Cut
    ("coh_arpg", 939100): (2, ""),  # Darksburg
    ("coh_arpg", 1088090): (1, "MODE_MISMATCH"),  # Day of Dragons
    ("coh_arpg", 1218320): (2, ""),  # Zack 2: Celestine's Map
    ("coh_arpg", 1230530): (3, ""),  # Atlas Fallen: Reign Of Sand
    ("coh_arpg", 1308700): (2, ""),  # Wizardry: The Five Ordeals
    ("coh_arpg", 1315180): (2, ""),  # Spark in the Dark
    ("coh_arpg", 1343370): (2, ""),  # Old School RuneScape
    ("coh_arpg", 1491410): (3, ""),  # Eternal Strands
    ("coh_arpg", 1893440): (2, ""),  # INTO EVIL
    ("coh_arpg", 2114580): (1, "IRRELEVANT"),  # Spells & Secrets - Character Creator
    ("coh_arpg", 3216340): (3, ""),  # Tearscape
    ("coh_classic_multi", 97000): (1, "IRRELEVANT"),  # Solar 2
    ("coh_classic_multi", 280790): (3, ""),  # Creativerse
    ("coh_classic_multi", 302830): (2, ""),  # BLOCKADE 3D
    ("coh_classic_multi", 321400): (2, ""),  # Supraball
    ("coh_classic_multi", 419520): (2, ""),  # Lifeless
    ("coh_classic_multi", 434570): (2, ""),  # Blood and Bacon
    ("coh_classic_multi", 872200): (2, ""),  # Rogue Company
    ("coh_classic_multi", 1051290): (2, ""),  # ZIC – Zombies in City
    ("coh_classic_multi", 1142500): (3, ""),  # Fun with Ragdolls: The Game
    ("coh_classic_multi", 1240440): (2, ""),  # Halo Infinite
    ("coh_classic_multi", 1281150): (3, ""),  # Modiverse
    ("coh_classic_multi", 1468720): (2, ""),  # Ultimate Epic Battle Simulator 2
    ("coh_classic_multi", 1506620): (2, ""),  # Horror Squad
    ("coh_classic_multi", 1573330): (3, ""),  # Unreal Sandbox
    ("coh_classic_multi", 1962663): (2, ""),  # 콜 오브 듀티®: 워존
    ("coh_classic_multi", 2154730): (3, ""),  # Building Destruction
    ("coh_cozy", 1713350): (3, ""),  # Project Castaway
    ("coh_cozy", 1740300): (2, ""),  # Smushi Come Home
    ("coh_cozy", 2238040): (2, ""),  # 타이니 테리의 터보 여행
    ("coh_cozy", 2987250): (1, "KEYWORD_MATCH"),  # 슬라임이 온다
    ("coh_cozy", 2997230): (1, "GENRE_ONLY"),  # Planet of Lana II
    ("coh_cozy", 3616260): (1, "MODE_MISMATCH"),  # MAZEBOUND: Hunt, Gather, Run!
    ("coh_cozy", 3619420): (1, "KEYWORD_MATCH"),  # Lab Eject
    ("coh_cozy", 3659410): (3, ""),  # 몬스터 빅팜: 방치형 즐거운 시간
    ("coh_fps", 34870): (2, ""),  # Sniper: Ghost Warrior 2
    ("coh_fps", 229870): (2, ""),  # ShootMania Storm
    ("coh_fps", 310950): (1, "MODE_MISMATCH"),  # Street Fighter V
    ("coh_fps", 384960): (2, ""),  # Vengeance
    ("coh_fps", 765410): (1, "LOW_QUALITY"),  # 빈 배틀즈
    ("coh_fps", 823130): (2, ""),  # Totally Accurate Battlegrounds
    ("coh_fps", 860020): (3, ""),  # EXFIL
    ("coh_fps", 923790): (1, "LOW_QUALITY"),  # Battle Royale Survival
    ("coh_fps", 1010860): (1, "LOW_QUALITY"),  # Hide and Seek
    ("coh_fps", 1412190): (2, ""),  # BEACHED
    ("coh_fps", 1515640): (2, ""),  # Arcadegeddon
    ("coh_fps", 1913370): (3, ""),  # OPERATOR
    ("coh_fps", 2476720): (2, ""),  # MetaStrike
    ("coh_fps", 3065800): (3, ""),  # 마라톤
    ("coh_fps", 3649730): (2, ""),  # Legend Battles
    ("coh_grand_strategy", 8170): (2, ""),  # Battlestations Pacific
    ("coh_grand_strategy", 308173): (3, ""),  # Hegemony III: Clash of the Ancients
    ("coh_grand_strategy", 1324780): (2, ""),  # Easy Red 2
    ("coh_grand_strategy", 1337650): (1, "IRRELEVANT"),  # Flak
    ("coh_grand_strategy", 1700300): (3, ""),  # World Warfare & Economics
    ("coh_grand_strategy", 2202120): (2, ""),  # 63 Days
    ("coh_grand_strategy", 3024040): (3, ""),  # Stronghold Crusader: Definitive Edition 
    ("coh_grand_strategy", 3035500): (2, ""),  # Fantasy Map Simulator
    ("coh_grand_strategy", 3428840): (2, ""),  # Countryballs: Rise of Europe
    ("coh_grand_strategy", 4549610): (3, ""),  # 블라인드삼국
    ("coh_indie_platformer", 509570): (2, ""),  # In Celebration of Violence
    ("coh_indie_platformer", 543260): (3, ""),  # Wonder Boy: The Dragon's Trap
    ("coh_indie_platformer", 715680): (2, ""),  # Knight vs Giant: The Broken Excalibur
    ("coh_indie_platformer", 1069530): (3, ""),  # Narita Boy
    ("coh_indie_platformer", 1399750): (3, ""),  # Beholgar
    ("coh_indie_platformer", 1608230): (3, ""),  # Planet of Lana
    ("coh_indie_platformer", 1774220): (3, ""),  # Jubilee
    ("coh_indie_platformer", 1778630): (3, ""),  # Bee: The Knight
    ("coh_indie_platformer", 2100150): (2, ""),  # Shadow of the Depth
    ("coh_indie_platformer", 2330750): (1, "MODE_MISMATCH"),  # Cataclysm: Dark Days Ahead
    ("coh_indie_platformer", 2800900): (1, "GENRE_ONLY"),  # Rift Riff
    ("coh_indie_platformer", 3072640): (2, ""),  # 가이더스 제로
    ("coh_indie_platformer", 3288210): (3, ""),  # Super Meat Boy 3D
    ("coh_indie_platformer", 3431030): (3, ""),  # Slime Climb
    ("coh_openworld_survival", 301280): (2, ""),  # Skin Deep
    ("coh_openworld_survival", 385380): (2, ""),  # Planet Centauri
    ("coh_openworld_survival", 526870): (2, ""),  # Satisfactory
    ("coh_openworld_survival", 876650): (3, ""),  # Aground
    ("coh_openworld_survival", 1077520): (2, ""),  # The Light Remake
    ("coh_openworld_survival", 1326470): (3, ""),  # Sons Of The Forest
    ("coh_openworld_survival", 1343520): (2, ""),  # Survival: Lost Way
    ("coh_openworld_survival", 1356480): (3, ""),  # helionaut
    ("coh_openworld_survival", 1597980): (2, ""),  # City 20
    ("coh_openworld_survival", 1641960): (3, ""),  # Forever Skies
    ("coh_openworld_survival", 1783560): (3, ""),  # The Last Caretaker
    ("coh_openworld_survival", 2633640): (1, "GENRE_ONLY"),  # Eyes: The Horror Game
    ("coh_strategy", 2820): (2, ""),  # X3: Terran Conflict
    ("coh_strategy", 63940): (2, ""),  # Men of War: Vietnam
    ("coh_strategy", 325790): (3, ""),  # Fallen: A2P Protocol
    ("coh_strategy", 346810): (2, ""),  # Marble Age
    ("coh_strategy", 410980): (3, ""),  # Master of Orion 2
    ("coh_strategy", 573490): (2, ""),  # Interstellar Transport Company
    ("coh_strategy", 809230): (3, ""),  # Unity of Command II
    ("coh_strategy", 887370): (2, ""),  # Element Space
    ("coh_strategy", 1021070): (2, ""),  # Spaceland: Sci-Fi Indie Tactics
    ("coh_strategy", 1155330): (2, ""),  # Showgunners
    ("coh_strategy", 1812450): (2, ""),  # Bellwright
    ("coh_strategy", 2994010): (3, ""),  # Chains of Freedom
    ("coh_strategy", 3234520): (2, ""),  # 文明征程
    ("coh_survival_craft", 264710): (3, ""),  # 서브노티카
    ("coh_survival_craft", 280790): (2, ""),  # Creativerse
    ("coh_survival_craft", 340050): (3, ""),  # Survivalist
    ("coh_survival_craft", 638850): (2, ""),  # Animallica
    ("coh_survival_craft", 717790): (3, ""),  # Hold Your Own
    ("coh_survival_craft", 815370): (3, ""),  # Green Hell
    ("coh_survival_craft", 1519090): (3, ""),  # Welcome to ParadiZe
    ("coh_survival_craft", 1641960): (2, ""),  # Forever Skies
    ("coh_survival_craft", 3484300): (3, ""),  # DREADZONE
    ("coh_vehicle_sim", 44680): (3, ""),  # RACE Injection
    ("coh_vehicle_sim", 256390): (2, ""),  # MotoGP™14
    ("coh_vehicle_sim", 299970): (3, ""),  # Project Motor Racing
    ("coh_vehicle_sim", 312650): (1, "MODE_MISMATCH"),  # Battlezone Gold Edition
    ("coh_vehicle_sim", 501220): (1, "LOW_QUALITY"),  # Space Ribbon - Slipstream to the Extreme
    ("coh_vehicle_sim", 747910): (2, ""),  # Disassembly 3D
    ("coh_vehicle_sim", 1062960): (2, ""),  # UNDER the SAND - a road trip simulator
    ("coh_vehicle_sim", 1330660): (3, ""),  # Train Life: A Railway Simulator
    ("coh_vehicle_sim", 1465360): (2, ""),  # SnowRunner
    ("coh_vehicle_sim", 2256990): (3, ""),  # Road Trucker
    ("coh_vehicle_sim", 2577910): (2, ""),  # Car Mechanic: City Driving
    ("coh_vehicle_sim", 2625420): (2, ""),  # Drive Beyond Horizons
    ("coh_vehicle_sim", 3104610): (2, ""),  # MEGARACER QUANTUMRUSH
    ("coh_vehicle_sim", 3656800): (3, ""),  # Train Sim World® 6
    ("longtail_deckbuilder", 930780): (3, ""),  # Blood Card
    ("longtail_deckbuilder", 931270): (2, ""),  # MicroTown
    ("longtail_deckbuilder", 1155880): (2, ""),  # The Bonfire 2: Uncharted Shores
    ("longtail_deckbuilder", 1184820): (3, ""),  # Poker Quest: Swords and Spades
    ("longtail_deckbuilder", 1501690): (3, ""),  # Dice Kingdoms
    ("longtail_deckbuilder", 1691190): (3, ""),  # Rogue Waters
    ("longtail_deckbuilder", 1970580): (3, ""),  # Backpack Hero
    ("longtail_deckbuilder", 2096510): (3, ""),  # The Ouroboros King
    ("longtail_deckbuilder", 2646720): (3, ""),  # Dead Weight
    ("longtail_deckbuilder", 2684510): (3, ""),  # Cards of Divinity
    ("longtail_deckbuilder", 2842800): (2, ""),  # 尸姬之梦
    ("longtail_deckbuilder", 3412270): (3, ""),  # Encounter: The Lost Cards
    ("longtail_deckbuilder", 3693590): (2, ""),  # 보석 전설
    ("longtail_deckbuilder", 3908810): (2, ""),  # Stackflow
    ("longtail_detective", 354680): (3, ""),  # Adventures of Bertram Fiddle 1: A Dreadl
    ("longtail_detective", 431260): (2, ""),  # Cursed
    ("longtail_detective", 543240): (3, ""),  # √Letter - Root Letter -
    ("longtail_detective", 865360): (2, ""),  # 우리는 여기에 있었다 투게더
    ("longtail_detective", 1067540): (3, ""),  # Röki
    ("longtail_detective", 1520380): (1, "MODE_MISMATCH"),  # HALF DEAD 3
    ("longtail_detective", 1689870): (3, ""),  # Meridian 157: Chapter 3
    ("longtail_detective", 1799220): (3, ""),  # Escape: Malice
    ("longtail_detective", 2242760): (2, ""),  # The Escape: Together
    ("longtail_detective", 2754380): (3, ""),  # The Roottrees are Dead
    ("longtail_detective", 3156640): (2, ""),  # Split Brain
    ("longtail_metroidvania", 204060): (2, ""),  # Superbrothers: Sword & Sworcery EP
    ("longtail_metroidvania", 356650): (3, ""),  # Death's Gambit: Afterlife
    ("longtail_metroidvania", 576050): (3, ""),  # Other Worlds India
    ("longtail_metroidvania", 813230): (3, ""),  # ANIMAL WELL
    ("longtail_metroidvania", 835430): (3, ""),  # La-Mulana 2
    ("longtail_metroidvania", 877810): (2, ""),  # Anodyne 2: Return to Dust
    ("longtail_metroidvania", 1013100): (3, ""),  # Mage
    ("longtail_metroidvania", 1669420): (3, ""),  # Islets
    ("longtail_metroidvania", 1701520): (3, ""),  # Afterimage
    ("longtail_metroidvania", 1823930): (2, ""),  # Wavetale
    ("longtail_metroidvania", 1985960): (2, ""),  # 오구와 비밀의 숲
    ("longtail_metroidvania", 2243250): (3, ""),  # Moonlight Pulse
    ("longtail_metroidvania", 2382240): (3, ""),  # Awita: Journey of Hope
    ("longtail_metroidvania", 2483930): (3, ""),  # White Flame Inna: Daughter of the Void
    ("longtail_metroidvania", 2508950): (2, ""),  # DreadFall
    ("longtail_metroidvania", 2973120): (2, ""),  # Dusk City
    ("longtail_metroidvania", 3081830): (2, ""),  # FLARE NUINUI QUEST
    ("longtail_puzzle_platformer", 115800): (2, ""),  # Owlboy
    ("longtail_puzzle_platformer", 257850): (2, ""),  # Hyper Light Drifter
    ("longtail_puzzle_platformer", 371140): (2, ""),  # Aegis Defenders
    ("longtail_puzzle_platformer", 372210): (2, ""),  # Spooky Cats
    ("longtail_puzzle_platformer", 461730): (2, ""),  # Blaite
    ("longtail_puzzle_platformer", 790740): (2, ""),  # Tick Tock: A Tale for Two
    ("longtail_puzzle_platformer", 878670): (1, "IRRELEVANT"),  # Shenmue III
    ("longtail_puzzle_platformer", 956030): (3, ""),  # Creaks
    ("longtail_puzzle_platformer", 1004330): (0, "IRRELEVANT"),  # My Exercise
    ("longtail_puzzle_platformer", 1172520): (2, ""),  # Colorgrid
    ("longtail_puzzle_platformer", 1328840): (3, ""),  # Lost in Play
    ("longtail_puzzle_platformer", 1660960): (3, ""),  # Windswept
    ("longtail_puzzle_platformer", 2231040): (1, "IRRELEVANT"),  # Coloring Game: Studio
    ("longtail_puzzle_platformer", 3106330): (3, ""),  # Chiral
    ("longtail_puzzle_platformer", 3834580): (3, ""),  # Kotenok
    ("lowrev_cozy_narrative", 512790): (2, ""),  # Quern - Undying Thoughts
    ("lowrev_cozy_narrative", 883360): (3, ""),  # Beyond Blue
    ("lowrev_cozy_narrative", 1042490): (3, ""),  # Call of the Sea
    ("lowrev_cozy_narrative", 1575980): (3, ""),  # UsoNatsu ~The Summer Romance Bloomed Fro
    ("lowrev_cozy_narrative", 1740300): (3, ""),  # Smushi Come Home
    ("lowrev_cozy_narrative", 2025610): (3, ""),  # Once Again
    ("lowrev_cozy_narrative", 2052410): (2, ""),  # WITCH ON THE HOLY NIGHT
    ("lowrev_cozy_narrative", 2307160): (2, ""),  # Yukiiro Sign
    ("lowrev_cozy_narrative", 2797180): (2, ""),  # 咫尺遥心-Proof of existence
    ("lowrev_cozy_narrative", 3107900): (1, "GENRE_ONLY"),  # Liminalcore
    ("lowrev_cozy_narrative", 3854310): (2, ""),  # Steventon Street: Deluxe Edition
    ("lowrev_cozy_narrative", 4309030): (2, ""),  # 错过的暑假可以放第二次吗？
    ("lowrev_deckbuilder", 264690): (3, ""),  # Coin Crypt
    ("lowrev_deckbuilder", 557410): (3, ""),  # Dream Quest
    ("lowrev_deckbuilder", 826740): (3, ""),  # Rise of the Slime
    ("lowrev_deckbuilder", 1184820): (3, ""),  # Poker Quest: Swords and Spades
    ("lowrev_deckbuilder", 1332090): (3, ""),  # Void Tyrant
    ("lowrev_deckbuilder", 1394130): (3, ""),  # Breach Wanderers
    ("lowrev_deckbuilder", 1709900): (3, ""),  # Tower Tactics: Liberation
    ("lowrev_deckbuilder", 1724390): (3, ""),  # Power Chord
    ("lowrev_deckbuilder", 2095290): (3, ""),  # Theseus Protocol
    ("lowrev_deckbuilder", 2400510): (3, ""),  # Dungeons & Degenerate Gamblers
    ("lowrev_deckbuilder", 2622820): (3, ""),  # Dobbel Dungeon
    ("lowrev_deckbuilder", 3519530): (2, ""),  # Merge Maestro
    ("lowrev_deckbuilder", 3784030): (2, ""),  # RACCOIN: Coin Pusher Roguelike
    ("lowrev_detective", 46550): (2, ""),  # Post Mortem
    ("lowrev_detective", 330990): (2, ""),  # Demon Hunter: Chronicles from Beyond
    ("lowrev_detective", 359510): (3, ""),  # Tangle Tower
    ("lowrev_detective", 1077560): (3, ""),  # 무연
    ("lowrev_detective", 1259640): (3, ""),  # Tiny Room Stories: Town Mystery
    ("lowrev_detective", 1889040): (1, "IRRELEVANT"),  # Birth
    ("lowrev_detective", 2676840): (3, ""),  # 미제사건은 끝내야 하니까
    ("lowrev_detective", 2739630): (3, ""),  # of the Devil
    ("lowrev_detective", 2806480): (1, "GENRE_ONLY"),  # Marie's Travel
    ("lowrev_detective", 2862610): (1, "IRRELEVANT"),  # 슈퍼스카우트
    ("lowrev_detective", 2933180): (1, "IRRELEVANT"),  # 女神保卫战
    ("lowrev_detective", 2963550): (2, ""),  # 운명 게임
    ("lowrev_detective", 3193510): (3, ""),  # The Ghost of Redstone Manor
    ("lowrev_detective", 3350200): (2, ""),  # 情感反诈模拟器
    ("lowrev_detective", 3636620): (3, ""),  # The Last Case of John Morley
    ("lowrev_detective", 4202730): (2, ""),  # Whispered Secrets: Poisoner's Masquerade
    ("lowrev_metroidvania", 207530): (2, ""),  # Noitu Love 2: Devolution
    ("lowrev_metroidvania", 253330): (1, "GENRE_ONLY"),  # Neverending Nightmares
    ("lowrev_metroidvania", 444720): (3, ""),  # Inexistence
    ("lowrev_metroidvania", 494600): (2, ""),  # EARTH'S DAWN
    ("lowrev_metroidvania", 537430): (2, ""),  # Inner Chains
    ("lowrev_metroidvania", 1126710): (2, ""),  # 魔法使いハナビィ Hanaby the Witch
    ("lowrev_metroidvania", 1252830): (2, ""),  # A Juggler's Tale
    ("lowrev_metroidvania", 1325260): (2, ""),  # Kitsune Tails
    ("lowrev_metroidvania", 1716310): (3, ""),  # Awaken - Astral Blade
    ("lowrev_metroidvania", 1985960): (2, ""),  # 오구와 비밀의 숲
    ("lowrev_metroidvania", 2063480): (3, ""),  # Voltage High Society
    ("lowrev_metroidvania", 2262990): (1, "MODE_MISMATCH"),  # Manager can be Tough!: Case of the Kidna
    ("lowrev_metroidvania", 2394650): (3, ""),  # Crypt Custodian
    ("lowrev_towerdefense", 334210): (3, ""),  # Fortified
    ("lowrev_towerdefense", 434570): (1, "GENRE_ONLY"),  # Blood and Bacon
    ("lowrev_towerdefense", 438480): (3, ""),  # Rock 'N' Roll Defense
    ("lowrev_towerdefense", 476530): (1, "IRRELEVANT"),  # Children of a Dead Earth
    ("lowrev_towerdefense", 524010): (3, ""),  # Siege of Centauri
    ("lowrev_towerdefense", 732160): (3, ""),  # The Wild Age
    ("lowrev_towerdefense", 843200): (1, "GENRE_ONLY"),  # Alien Hominid Invasion
    ("lowrev_towerdefense", 988980): (2, ""),  # GLAD VALAKAS TOWER DEFENCE
    ("lowrev_towerdefense", 1176160): (1, "LOW_QUALITY"),  # Space Wars
    ("lowrev_towerdefense", 1383760): (3, ""),  # Fortification: tower defence
    ("lowrev_towerdefense", 1520380): (0, "IRRELEVANT"),  # HALF DEAD 3
    ("lowrev_towerdefense", 1531540): (2, ""),  # Distant Worlds 2
    ("lowrev_towerdefense", 2223590): (3, ""),  # 2112TD: Tower Defense Survival
    ("lowrev_towerdefense", 2361460): (3, ""),  # Toy Shire
    ("lowrev_towerdefense", 2948680): (2, ""),  # Outlive 25
    ("lowrev_towerdefense", 3054090): (3, ""),  # Idle Tower Defense
    ("lowrev_towerdefense", 3211850): (3, ""),  # 칼릭스
    ("mix2_arcade_action", 730): (1, "MODE_MISMATCH"),  # Counter-Strike 2
    ("mix2_arcade_action", 15740): (1, "IRRELEVANT"),  # Oddworld: Munch's Oddysee
    ("mix2_arcade_action", 299970): (3, ""),  # Project Motor Racing
    ("mix2_arcade_action", 661990): (2, ""),  # Arcana Heart 3 LOVEMAX SIXSTARS!!!!!! XT
    ("mix2_arcade_action", 1350080): (2, ""),  # ABC: AUDIO REACTIVE BEAT CIRCLE
    ("mix2_arcade_action", 1477590): (3, ""),  # EZ2ON REBOOT : R
    ("mix2_arcade_action", 1594040): (3, ""),  # Wreckreation
    ("mix2_arcade_action", 2004380): (2, ""),  # ROD Multiplayer Car Driving
    ("mix2_arcade_action", 2022230): (2, ""),  # Yomi 2
    ("mix2_arcade_action", 2157560): (3, ""),  # Granblue Fantasy Versus: Rising
    ("mix2_arcade_action", 2227440): (2, ""),  # Unlimited Fight Ultimate Strike
    ("mix2_arcade_action", 2263360): (2, ""),  # HYPERBEAT
    ("mix2_arcade_action", 2360770): (2, ""),  # Coreupt
    ("mix2_arcade_action", 2613950): (2, ""),  # Bloody Knuckles Street Boxing
    ("mix2_arcade_action", 3058630): (3, ""),  # 아세토 코르사 에보 Assetto Corsa EVO
    ("mix2_arcade_action", 4323990): (1, "MODE_MISMATCH"),  # EMPULSE
    ("mix2_arcade_action", 4537710): (2, ""),  # Aqua Street
    ("mix2_builder_sim", 4460): (3, ""),  # City Life 2008
    ("mix2_builder_sim", 242920): (3, ""),  # Banished
    ("mix2_builder_sim", 495560): (3, ""),  # Farm Manager 2018
    ("mix2_builder_sim", 690830): (3, ""),  # Foundation
    ("mix2_builder_sim", 1125390): (3, ""),  # Atrio: The Dark Wild
    ("mix2_builder_sim", 1245560): (2, ""),  # Roots of Pacha
    ("mix2_builder_sim", 1735610): (3, ""),  # Multiply Factory
    ("mix2_builder_sim", 1760340): (3, ""),  # ReFactory
    ("mix2_builder_sim", 1989070): (3, ""),  # 시너지 (Synergy)
    ("mix2_builder_sim", 2262080): (3, ""),  # Omega Crafter
    ("mix2_builder_sim", 2426530): (3, ""),  # Nova Roma 노바 로마
    ("mix2_builder_sim", 2666510): (2, ""),  # Rusty's Retirement
    ("mix2_builder_sim", 2823890): (3, ""),  # Center Station Simulator
    ("mix2_builder_sim", 4945600): (2, ""),  # Let's Grow Together
    ("mix2_colony_cozy", 573490): (2, ""),  # Interstellar Transport Company
    ("mix2_colony_cozy", 774351): (2, ""),  # Citystate
    ("mix2_colony_cozy", 1324270): (2, ""),  # Space Station Tycoon
    ("mix2_colony_cozy", 1395760): (3, ""),  # Reshaping Mars
    ("mix2_colony_cozy", 1532200): (2, ""),  # Mars First Logistics
    ("mix2_colony_cozy", 1596310): (1, "IRRELEVANT"),  # Crypto Mining Simulator
    ("mix2_colony_cozy", 1712110): (3, ""),  # Deep Space Outpost
    ("mix2_colony_cozy", 1881940): (3, ""),  # Heard of the Story?
    ("mix2_colony_cozy", 1928080): (3, ""),  # Prospector
    ("mix2_colony_cozy", 2134770): (3, ""),  # SteamWorld Build
    ("mix2_colony_cozy", 2187340): (3, ""),  # ExoColony: Planet Survival
    ("mix2_colony_cozy", 2353250): (2, ""),  # Astrometica
    ("mix2_colony_cozy", 2407830): (2, ""),  # 고블린의 업사이클 공방
    ("mix2_colony_cozy", 2413190): (2, ""),  # Emergency Cleanup Co.
    ("mix2_colony_cozy", 2776790): (2, ""),  # Dr. Planet
    ("mix2_colony_cozy", 4526770): (3, ""),  # Declutterer
    ("mix2_colony_cozy", 4590400): (3, ""),  # Raccoon Packer: A Cozy Puzzle Game
    ("mix2_colony_cozy", 4604410): (3, ""),  # Unbox your Workspace
    ("mix2_colony_cozy", 4800590): (3, ""),  # Supermarket Chaos
    ("mix2_coop_horror", 967050): (3, ""),  # Pacify
    ("mix2_coop_horror", 1481210): (3, ""),  # Psychoscopy
    ("mix2_coop_horror", 1497950): (1, "MODE_MISMATCH"),  # 동~그란 지구가 네모가 됐다고!? 디지복셀 지구방위군 EARTH DEFE
    ("mix2_coop_horror", 1713810): (2, ""),  # Lumencraft
    ("mix2_coop_horror", 1745680): (2, ""),  # Odd Remedy
    ("mix2_coop_horror", 1893440): (2, ""),  # INTO EVIL
    ("mix2_coop_horror", 2111870): (2, ""),  # Section 13
    ("mix2_coop_horror", 2197890): (3, ""),  # Paranormal Cleanup
    ("mix2_coop_horror", 2234150): (3, ""),  # BACKROOMS: APPREHENSION
    ("mix2_coop_horror", 2304620): (3, ""),  # Backrooms Society
    ("mix2_coop_horror", 2524930): (3, ""),  # We Escaped a Twisted Game
    ("mix2_coop_horror", 2619640): (3, ""),  # Evil Hunt - Evil never sleeps
    ("mix2_coop_horror", 2976900): (3, ""),  # Unpossess: Exorcism Simulator
    ("mix2_coop_horror", 3059070): (3, ""),  # The Headliners
    ("mix2_coop_horror", 3099190): (3, ""),  # Anomaly Company
    ("mix2_coop_horror", 3215820): (3, ""),  # Dreadway
    ("mix2_crpg_sandbox", 203770): (3, ""),  # Crusader Kings II
    ("mix2_crpg_sandbox", 223490): (1, "IRRELEVANT"),  # Blockscape
    ("mix2_crpg_sandbox", 280520): (2, ""),  # Crea
    ("mix2_crpg_sandbox", 311290): (3, ""),  # SpellForce 3 Reforced
    ("mix2_crpg_sandbox", 340050): (2, ""),  # Survivalist
    ("mix2_crpg_sandbox", 606880): (3, ""),  # GreedFall
    ("mix2_crpg_sandbox", 980640): (2, ""),  # Empire of Ember
    ("mix2_crpg_sandbox", 1284210): (2, ""),  # Guild Wars 2®
    ("mix2_crpg_sandbox", 1304430): (2, ""),  # Viking Frontiers
    ("mix2_crpg_sandbox", 1597980): (2, ""),  # City 20
    ("mix2_crpg_sandbox", 1724440): (2, ""),  # Epic Fantasy Battle Simulator
    ("mix2_crpg_sandbox", 2389170): (2, ""),  # Huaxia: Warring States
    ("mix2_crpg_sandbox", 3247200): (2, ""),  # Dawn of Ages
    ("mix2_modern_roguelite", 986040): (3, ""),  # The Unliving
    ("mix2_modern_roguelite", 1332340): (2, ""),  # 디스오더
    ("mix2_modern_roguelite", 1911610): (3, ""),  # Windblown
    ("mix2_modern_roguelite", 2097570): (3, ""),  # StarVaders
    ("mix2_modern_roguelite", 2229940): (3, ""),  # [REDACTED]
    ("mix2_modern_roguelite", 2292060): (3, ""),  # Spell Disk
    ("mix2_modern_roguelite", 2697930): (3, ""),  # 커맨더 퀘스트(Commander Quest)
    ("mix2_modern_roguelite", 2904000): (2, ""),  # 마법 주문 팀
    ("mix2_modern_roguelite", 3106070): (2, ""),  # 광대인 내가 어떻게 미소녀랑 카드 게임을 할 수 있겠어
    ("mix2_modern_roguelite", 3129360): (3, ""),  # 악선
    ("mix2_modern_roguelite", 3206200): (2, ""),  # 마법의 룬스톤
    ("mix2_modern_roguelite", 3405580): (3, ""),  # Lunar Rebirth: Dungeon of Eternal Return
    ("mix2_modern_roguelite", 3411020): (2, ""),  # 마을 & 던전
    ("mix2_modern_roguelite", 3419290): (3, ""),  # 신이 버린 땅-GOD FORSAKEN
    ("mix2_modern_roguelite", 4077800): (2, ""),  # Silly Survivors
    ("mix2_modern_roguelite", 4204530): (1, "MODE_MISMATCH"),  # Knights End
    ("mix2_modern_roguelite", 4608410): (3, ""),  # Huntervania
    ("mix2_party_narrative", 244910): (2, ""),  # Homesick
    ("mix2_party_narrative", 258970): (2, ""),  # Gauntlet™ Slayer Edition
    ("mix2_party_narrative", 483980): (1, "GENRE_ONLY"),  # Mad Father
    ("mix2_party_narrative", 507390): (2, ""),  # Gone In November
    ("mix2_party_narrative", 579840): (2, ""),  # Bloody Trapland 2: Curiosity
    ("mix2_party_narrative", 965680): (2, ""),  # Boomerang Fu
    ("mix2_party_narrative", 1071870): (3, ""),  # Biped
    ("mix2_party_narrative", 2124490): (1, "MODE_MISMATCH"),  # SILENT HILL 2
    ("mix2_party_narrative", 2475010): (2, ""),  # 피자 밴딧
    ("mix2_party_narrative", 2592220): (3, ""),  # CUFFBUST
    ("mix2_party_narrative", 3525360): (3, ""),  # Two the Top
    ("mix2_party_narrative", 3533100): (3, ""),  # Wrap House Simulator🌯
    ("mix2_party_narrative", 3744430): (3, ""),  # Together: Moon Escape
    ("mix2_puzzle_survival", 108600): (3, ""),  # Project Zomboid
    ("mix2_puzzle_survival", 378370): (2, ""),  # Nomad
    ("mix2_puzzle_survival", 493080): (2, ""),  # Card Quest
    ("mix2_puzzle_survival", 552160): (2, ""),  # Hade
    ("mix2_puzzle_survival", 556240): (2, ""),  # Disoriented
    ("mix2_puzzle_survival", 603800): (2, ""),  # ReThink
    ("mix2_puzzle_survival", 897450): (2, ""),  # The Survivalists
    ("mix2_puzzle_survival", 920680): (2, ""),  # Fate Hunters
    ("mix2_puzzle_survival", 1285080): (2, ""),  # Gordian Rooms 1: A curious heritage
    ("mix2_puzzle_survival", 1668690): (3, ""),  # Alina of the Arena
    ("mix2_puzzle_survival", 1876000): (1, "IRRELEVANT"),  # IFO
    ("mix2_puzzle_survival", 1967510): (2, ""),  # 레일바운드 - Railbound
    ("mix2_puzzle_survival", 3877790): (2, ""),  # 사냥의 밤：소브린 신디케이트
    ("mix2_soulslike_narrative", 210970): (3, ""),  # The Witness
    ("mix2_soulslike_narrative", 290300): (2, ""),  # Rebel Galaxy
    ("mix2_soulslike_narrative", 335300): (3, ""),  # DARK SOULS™ II: Scholar of the First Sin
    ("mix2_soulslike_narrative", 420740): (2, ""),  # The Deed
    ("mix2_soulslike_narrative", 587520): (2, ""),  # Dungeons of Sundaria
    ("mix2_soulslike_narrative", 612740): (2, ""),  # Bokida - Heartfelt Reunion
    ("mix2_soulslike_narrative", 883360): (2, ""),  # Beyond Blue
    ("mix2_soulslike_narrative", 969760): (3, ""),  # Omno
    ("mix2_soulslike_narrative", 1013750): (2, ""),  # Legal Dungeon
    ("mix2_soulslike_narrative", 1401220): (1, "IRRELEVANT"),  # 冷血症骰子
    ("mix2_soulslike_narrative", 1723260): (2, ""),  # CaseCracker
    ("mix2_soulslike_narrative", 2174380): (3, ""),  # Call of the Elder Gods
    ("mix2_soulslike_narrative", 2246340): (3, ""),  # Monster Hunter Wilds
    ("mix2_survival_farm", 280790): (2, ""),  # Creativerse
    ("mix2_survival_farm", 378370): (2, ""),  # Nomad
    ("mix2_survival_farm", 967410): (3, ""),  # Creatures Inc
    ("mix2_survival_farm", 1156360): (3, ""),  # Peaceful Days
    ("mix2_survival_farm", 1197220): (2, ""),  # Another Try
    ("mix2_survival_farm", 1363350): (3, ""),  # Monster Harvest
    ("mix2_survival_farm", 1515320): (3, ""),  # Harvest Days: My Dream Farm
    ("mix2_survival_farm", 1524630): (3, ""),  # KeepUp Survival
    ("mix2_survival_farm", 1536090): (3, ""),  # Echoes of the Plum Grove
    ("mix2_survival_farm", 1629830): (3, ""),  # Research Story
    ("mix2_survival_farm", 1681600): (3, ""),  # Cornucopia®
    ("mix2_survival_farm", 1780070): (3, ""),  # Seeds of Calamity
    ("mix2_survival_farm", 1898300): (3, ""),  # ASKA
    ("mix2_survival_farm", 2218970): (2, ""),  # Plains of Pain
    ("mix2_survival_farm", 3484300): (2, ""),  # DREADZONE
    ("niche_cozy_casual", 511470): (3, ""),  # Glass Masquerade
    ("niche_cozy_casual", 575640): (1, "IRRELEVANT"),  # Zup! 3
    ("niche_cozy_casual", 638230): (3, ""),  # Journey
    ("niche_cozy_casual", 642610): (3, ""),  # Shape of the World
    ("niche_cozy_casual", 1056610): (2, ""),  # Heal
    ("niche_cozy_casual", 1097100): (3, ""),  # Please, Touch The Artwork
    ("niche_cozy_casual", 1108820): (2, ""),  # Polar Jump
    ("niche_cozy_casual", 1125110): (2, ""),  # Obsurity
    ("niche_cozy_casual", 1135740): (3, ""),  # Utopia
    ("niche_cozy_casual", 1586800): (3, ""),  # Lil Gator Game
    ("niche_cozy_casual", 1781260): (2, ""),  # Promenade
    ("niche_cozy_casual", 1970460): (3, ""),  # Garden Galaxy
    ("niche_cozy_casual", 1983260): (2, ""),  # 힌터베르그의 던전
    ("niche_cozy_casual", 1995590): (3, ""),  # The Block
    ("niche_cozy_casual", 2331880): (3, ""),  # Otok
    ("niche_cozy_casual", 2521630): (3, ""),  # Mini Settlers
    ("niche_cozy_casual", 2717750): (2, ""),  # 토블라 - 신성한 길
    ("niche_cozy_casual", 3959830): (1, "LOW_QUALITY"),  # Rotate Maze: G.O.T.S
    ("niche_puzzle_solo", 41740): (2, ""),  # Cargo! The Quest for Gravity
    ("niche_puzzle_solo", 57200): (2, ""),  # Puzzle Dimension
    ("niche_puzzle_solo", 108500): (2, ""),  # Vessel
    ("niche_puzzle_solo", 204180): (1, "GENRE_ONLY"),  # Waveform
    ("niche_puzzle_solo", 344480): (2, ""),  # Quell
    ("niche_puzzle_solo", 352430): (2, ""),  # Farlight Explorers
    ("niche_puzzle_solo", 552160): (2, ""),  # Hade
    ("niche_puzzle_solo", 602320): (2, ""),  # Train Valley 2
    ("niche_puzzle_solo", 1253450): (3, ""),  # Gnomes & Co: The Art of the Build
    ("niche_puzzle_solo", 1367680): (3, ""),  # Charge!
    ("niche_puzzle_solo", 1372320): (1, "IRRELEVANT"),  # Cloud Gardens
    ("niche_puzzle_solo", 2312770): (3, ""),  # Linkito
    ("niche_puzzle_solo", 2472770): (3, ""),  # Automate It: Factory Puzzle
    ("niche_puzzle_solo", 3247030): (2, ""),  # Rhell: Warped Worlds & Troubled Times
    ("niche_puzzle_solo", 3816700): (3, ""),  # Logic Bombs
    ("niche_puzzle_solo", 4204590): (1, "GENRE_ONLY"),  # Reincarnated as a Dark Lord Before Takin
    ("niche_roguelite", 314410): (3, ""),  # Rampage Knights
    ("niche_roguelite", 356650): (3, ""),  # Death's Gambit: Afterlife
    ("niche_roguelite", 677120): (3, ""),  # Heroes of Hammerwatch
    ("niche_roguelite", 782570): (2, ""),  # 엔드리스 로드：Reborn
    ("niche_roguelite", 881100): (3, ""),  # Noita
    ("niche_roguelite", 1940340): (3, ""),  # Darkest Dungeon® II
    ("niche_roguelite", 2235200): (3, ""),  # Neon Abyss 2
    ("niche_roguelite", 2395770): (3, ""),  # Never Grave: The Witch and The Curse
    ("niche_roguelite", 3583670): (1, "GENRE_ONLY"),  # Where the forest ends
    ("niche_sim", 46480): (3, ""),  # Still Life
    ("niche_sim", 240440): (3, ""),  # Quadrilateral Cowboy
    ("niche_sim", 378110): (3, ""),  # Hack RUN
    ("niche_sim", 476530): (1, "IRRELEVANT"),  # Children of a Dead Earth
    ("niche_sim", 761460): (3, ""),  # Lamplight City
    ("niche_sim", 790740): (3, ""),  # Tick Tock: A Tale for Two
    ("niche_sim", 983380): (2, ""),  # Car Manufacture
    ("niche_sim", 1190970): (2, ""),  # 하우스 플리퍼 2
    ("niche_sim", 1201550): (3, ""),  # Mad Experiments: Escape Room
    ("niche_sim", 1728180): (2, ""),  # Astral Shipwright
    ("niche_sim", 2660460): (2, ""),  # Aviassembly
    ("niche_sim", 2823890): (3, ""),  # Center Station Simulator
    ("niche_sim", 2827680): (2, ""),  # Block Factory
    ("niche_sim", 3159620): (3, ""),  # Emergency Exit
    ("niche_sim", 3684610): (2, ""),  # s.p.l.i.t
    ("niche_soulslike_solo", 347800): (3, ""),  # Ghost Song
    ("niche_soulslike_solo", 371140): (2, ""),  # Aegis Defenders
    ("niche_soulslike_solo", 383230): (3, ""),  # Dungeon Souls
    ("niche_soulslike_solo", 418530): (2, ""),  # Spelunky 2
    ("niche_soulslike_solo", 587520): (2, ""),  # Dungeons of Sundaria
    ("niche_soulslike_solo", 644830): (3, ""),  # The Surge 2
    ("niche_soulslike_solo", 831050): (3, ""),  # Dolmen
    ("niche_soulslike_solo", 1260910): (2, ""),  # Demelia's Isle
    ("niche_soulslike_solo", 1430680): (3, ""),  # Clash: Artifacts of Chaos
    ("niche_soulslike_solo", 1725950): (1, "GENRE_ONLY"),  # 魔塔地牢
    ("niche_soulslike_solo", 1802880): (3, ""),  # 데블위딘 삿갓
    ("niche_soulslike_solo", 3929630): (2, ""),  # Terrinoth®: Heroes of Descent
    ("niche_tactics", 726110): (3, ""),  # Overcrowd: A Commute 'Em Up
    ("niche_tactics", 887370): (2, ""),  # Element Space
    ("niche_tactics", 919370): (3, ""),  # Overdungeon
    ("niche_tactics", 921710): (3, ""),  # Galaxy Squad
    ("niche_tactics", 1364210): (2, ""),  # City Bus Manager
    ("niche_tactics", 1373260): (3, ""),  # Obsidian Prince
    ("niche_tactics", 1608040): (2, ""),  # Castle Morihisa
    ("niche_tactics", 1967510): (3, ""),  # 레일바운드 - Railbound
    ("niche_tactics", 2059170): (2, ""),  # Quasimorph
    ("niche_tactics", 2076450): (2, ""),  # Freenergy
    ("niche_tactics", 2140850): (2, ""),  # Looper Tactics
    ("niche_tactics", 2622820): (2, ""),  # Dobbel Dungeon
    ("niche_tactics", 3107050): (3, ""),  # Stoplights
    ("niche_tactics", 3109280): (3, ""),  # Super Loco World - Cozy Train Automation
}


BLIND_K50_FILL = {
    ("longtail_puzzle_platformer", 3310): (1, "IRRELEVANT"),  # Chuzzle Deluxe
    ("longtail_puzzle_platformer", 250050): (3, ""),  # Life Goes On: Done to Death
    ("longtail_puzzle_platformer", 664180): (1, "GENRE_ONLY"),  # Draw Puzzle 画之谜
    ("longtail_puzzle_platformer", 2539960): (2, ""),  # Orbo's Odyssey
    ("longtail_puzzle_platformer", 2972990): (1, "GENRE_ONLY"),  # hololive Treasure Mountain
    ("longtail_puzzle_platformer", 3204020): (1, "GENRE_ONLY"),  # Juufuutei Raden™'s Guide for Pixel Museu
    ("lowrev_detective", 1444300): (2, ""),  # Teacup
    ("lowrev_detective", 2078350): (0, "IRRELEVANT"),  # 블루이: 비디오 게임
    ("lowrev_detective", 2653470): (3, ""),  # Little Problems: A Cozy Detective Game
    ("mix2_soulslike_narrative", 1630): (1, "IRRELEVANT"),  # Disciples II: Rise of the Elves
    ("mix2_soulslike_narrative", 2294450): (3, ""),  # CaseCracker2
    ("mix2_soulslike_narrative", 3296830): (3, ""),  # CaseCracker3
}


# ------------------------------------------------------------------ 기준 교정 v1
#
# **왜 고치나** — 라운드마다 같은 상황을 다르게 매겼다. 초기 라운드(VAL_V1/DEV_TOPUP_V1/
# BLIND_TAGS)는 "같은 시리즈의 다른 게임"을 `FRANCHISE_OR_VARIANT` 로 1점 처리했는데,
# k=30·k=50 라운드에서는 같은 상황을 3점으로 줬다(The Talos Principle 2, Hades II,
# Cities: Skylines II, Forza Horizon 6, Salt and Sacrifice, Grounded 2 ...).
# 그래서 coh_arpg 는 1·3·5위가 Witcher 1 / Oblivion / DARK SOULS II 인데도 P@50 0.70 이
# 나왔다. 시스템 품질이 아니라 판정자 일관성 문제다.
#
# **어느 쪽이 옳은가** — 시리즈의 **다른 게임**은 좋은 추천이다. Witcher 3 를 좋아하는
# 사람에게 Witcher 1 은 실제로 맞다. 걸러야 할 것은 "이미 가진 게임"인데 그건 판정이 아니라
# `exclude_appids` 가 할 일이다. 반면 **같은 게임의 다른 SKU**(리마스터 묶음 등)는 여전히
# 실패다 — 새 게임이 아니기 때문이다.
#
# **양방향으로 적용했다.** 올린 것만이 아니라 내린 것도 있다(Overcooked! All You Can Eat 은
# 시드 Overcooked! 2 를 포함한 묶음이라 3 → 1). 점수를 올리려는 교정이 아니라 기준을
# 통일하는 교정이다.
CORRECTIONS_V1 = {
    # 시리즈의 다른 게임 → 정상 추천 (1 → 3)
    ("coh_arpg", 20900): (3, ""),          # The Witcher: Enhanced Edition (시드 Witcher 3)
    ("coh_arpg", 570940): (3, ""),         # DARK SOULS: REMASTERED (시드 DS3)
    ("coh_arpg", 22330): (3, ""),          # TES IV: Oblivion (시드 Skyrim)
    ("coh_arpg", 335300): (3, ""),         # DARK SOULS II: SotFS (시드 DS3)
    ("coh_fps", 240): (3, ""),             # Counter-Strike: Source (시드 CS2)
    ("coh_grand_strategy", 25100): (3, ""),        # Hearts of Iron III (시드 HOI4)
    ("coh_vehicle_sim", 232750): (3, ""),  # Euro Truck Simulator (시드 ETS2)
    ("niche_roguelite", 239350): (3, ""),  # Spelunky (시드 Spelunky 2)
    ("niche_roguelite", 241600): (3, ""),  # Rogue Legacy (시드 Rogue Legacy 2)
    # 태그 오적용 — 시리즈 관계가 아예 없었다. 점수는 근거를 다시 보고 매겼다.
    ("lowrev_metroidvania", 1966000): (3, ""),     # Supraland Six Inches Under = 1인칭 메트로배니아
    ("coh_classic_multi", 773850): (1, "GENRE_ONLY"),   # WT2 = 무명 멀티 슈터, 시리즈 무관
    # 같은 게임의 다른 SKU → 실패 (3 → 1)
    ("mix2_party_narrative", 1243830): (1, "SAME_GAME_DIFFERENT_SKU"),  # Overcooked! AYCE ⊇ 시드 Overcooked! 2
    # 은퇴 프로필분도 같은 기준으로 맞춰 둔다 (평가에는 안 쓰이지만 풀을 일관되게 유지)
    ("mix_arpg_survival", 570940): (3, ""),
    ("mix_arpg_survival", 335300): (3, ""),
    ("mix_arpg_survival", 22370): (3, ""),         # Fallout 3 GOTY (시드 Fallout 4)
    ("mix_multi_indie", 500): (3, ""),             # Left 4 Dead (시드 L4D2)
    ("mix_rpg_racing", 22330): (3, ""),            # TES IV: Oblivion (시드 Skyrim)
    ("mix_rpg_racing", 971120): (1, "LOW_QUALITY"),  # MadOut Ice Storm — 시리즈 무관, 저품질
}


BLIND_STRATEGY_S1 = {
    ("coh_arpg", 2623190): (3, ""),  # The Elder Scrolls IV: Oblivion Remastere
    ("coh_classic_multi", 292730): (2, ""),  # Call of Duty®: Infinite Warfare
    ("coh_classic_multi", 453270): (2, ""),  # Madness Cubed
    ("coh_cozy", 1726130): (3, ""),  # Pathless Woods
    ("coh_cozy", 2918500): (3, ""),  # Cozy Island
    ("coh_fps", 1238820): (3, ""),  # Battlefield 3™
    ("coh_strategy", 236130): (3, ""),  # Horizon
    ("coh_strategy", 718850): (3, ""),  # Age of Wonders: Planetfall
    ("coh_strategy", 1176470): (3, ""),  # Terra Invicta 테라 인빅타
    ("longtail_detective", 31830): (3, ""),  # Nancy Drew®: Curse of Blackmoor Manor
    ("longtail_detective", 1580150): (3, ""),  # Between Time: Escape Room
    ("longtail_metroidvania", 347800): (3, ""),  # Ghost Song
    ("longtail_metroidvania", 1489410): (3, ""),  # Alice Escaped!
    ("longtail_puzzle_platformer", 969760): (3, ""),  # Omno
    ("longtail_puzzle_platformer", 1608230): (3, ""),  # Planet of Lana
    ("lowrev_cozy_narrative", 1102130): (3, ""),  # Florence
    ("lowrev_metroidvania", 115800): (2, ""),  # Owlboy
    ("lowrev_metroidvania", 394540): (3, ""),  # Spaceport Hope
    ("lowrev_towerdefense", 449710): (2, ""),  # REDCON
    ("mix2_arcade_action", 491280): (2, ""),  # Drift Horizon Online
    ("mix2_arcade_action", 1070580): (2, ""),  # Drift86
    ("mix2_arcade_action", 1262600): (3, ""),  # Need for Speed™ Rivals
    ("mix2_arcade_action", 1329600): (1, "LOW_QUALITY"),  # Karting
    ("mix2_arcade_action", 2051120): (3, ""),  # HOT WHEELS UNLEASHED™ 2 - Turbocharged
    ("mix2_builder_sim", 898720): (3, ""),  # Kubifaktorium
    ("mix2_builder_sim", 1307890): (3, ""),  # Kingdoms Reborn
    ("mix2_colony_cozy", 383120): (2, ""),  # Empyrion - Galactic Survival
    ("mix2_colony_cozy", 1380910): (3, ""),  # Stardeus
    ("mix2_colony_cozy", 1614550): (3, ""),  # Astro Colony
    ("mix2_crpg_sandbox", 2272900): (2, ""),  # Mirthwood
    ("mix2_puzzle_survival", 219740): (3, ""),  # Don't Starve
    ("mix2_puzzle_survival", 313120): (2, ""),  # Stranded Deep
    ("mix2_puzzle_survival", 538100): (2, ""),  # Feel The Snow
    ("mix2_puzzle_survival", 1169040): (2, ""),  # Necesse: 네세스
    ("mix2_puzzle_survival", 2806640): (1, "SAME_GAME_DIFFERENT_SKU"),  # The Talos Principle: Reawakened
    ("mix2_soulslike_narrative", 367500): (3, ""),  # Dragon's Dogma: Dark Arisen
    ("mix2_soulslike_narrative", 919360): (2, ""),  # Alaloth: Champions of The Four Kingdoms
    ("mix2_soulslike_narrative", 3929630): (1, "GENRE_ONLY"),  # Terrinoth®: Heroes of Descent
    ("niche_cozy_casual", 37600): (3, ""),  # Windosill
    ("niche_cozy_casual", 210970): (3, ""),  # The Witness
    ("niche_cozy_casual", 551110): (2, ""),  # Wayout
    ("niche_cozy_casual", 658560): (1, "LOW_QUALITY"),  # Zup! 7
    ("niche_roguelite", 632360): (3, ""),  # Risk of Rain 2
    ("niche_sim", 870200): (2, ""),  # Juno: New Origins
    ("niche_sim", 3296830): (3, ""),  # CaseCracker3
    ("niche_tactics", 1201830): (3, ""),  # For The Warp
    ("niche_tactics", 1265820): (3, ""),  # Fights in Tight Spaces
}


BLIND_STRATEGY_S2 = {
    ("coh_arpg", 7520): (2, ""),  # Two Worlds II HD
    ("coh_arpg", 253980): (2, ""),  # Enclave
    ("coh_arpg", 311290): (2, ""),  # SpellForce 3 Reforced
    ("coh_arpg", 1430680): (3, ""),  # Clash: Artifacts of Chaos
    ("coh_arpg", 3107880): (1, "LOW_QUALITY"),  # Gnomes and Knights
    ("coh_classic_multi", 8190): (2, ""),  # Just Cause 2
    ("coh_classic_multi", 222880): (2, ""),  # Insurgency
    ("coh_classic_multi", 304930): (3, ""),  # Unturned
    ("coh_cozy", 739650): (3, ""),  # Drake Hollow
    ("coh_cozy", 794490): (2, ""),  # Journey Of Life
    ("coh_cozy", 1062520): (3, ""),  # Dinkum(딩컴)
    ("coh_cozy", 1592110): (3, ""),  # Spirit of the Island
    ("coh_cozy", 1599330): (3, ""),  # Wildmender
    ("coh_cozy", 2350020): (2, ""),  # Start Over
    ("coh_fps", 47790): (2, ""),  # Medal of Honor™
    ("coh_fps", 611500): (2, ""),  # Quake Champions
    ("coh_fps", 1189800): (2, ""),  # Bleeding Edge
    ("coh_grand_strategy", 6250): (3, ""),  # Making History: The Calm & the Storm
    ("coh_indie_platformer", 395160): (3, ""),  # Toby: The Secret Mine
    ("coh_indie_platformer", 753420): (3, ""),  # Dungreed
    ("coh_indie_platformer", 814680): (3, ""),  # Unbound: Worlds Apart
    ("coh_indie_platformer", 1224020): (2, ""),  # Creepy Tale
    ("coh_openworld_survival", 242760): (3, ""),  # The Forest
    ("coh_openworld_survival", 386590): (2, ""),  # Novus Inceptio
    ("coh_openworld_survival", 1335830): (2, ""),  # Len's Island
    ("coh_strategy", 246940): (3, ""),  # Lords of the Black Sun
    ("coh_strategy", 287580): (3, ""),  # Pandora: First Contact
    ("coh_strategy", 1768280): (3, ""),  # Ozymandias: Bronze Age Empire Sim
    ("coh_strategy", 2147380): (3, ""),  # Heroes of Science and Fiction
    ("coh_strategy", 3407390): (3, ""),  # ENDLESS Legend™ 2 엔들리스 레전드 2
    ("coh_vehicle_sim", 324310): (3, ""),  # Bus Simulator 16
    ("coh_vehicle_sim", 671970): (2, ""),  # Junkyard Simulator
    ("coh_vehicle_sim", 2362300): (2, ""),  # Train Sim World® 4
    ("longtail_deckbuilder", 861540): (3, ""),  # Dicey Dungeons
    ("longtail_deckbuilder", 1199030): (3, ""),  # Tainted Grail: Conquest
    ("longtail_detective", 33610): (3, ""),  # Broken Sword 3 - the Sleeping Dragon (20
    ("longtail_detective", 209230): (3, ""),  # Sherlock Holmes and The Hound of The Bas
    ("longtail_detective", 326190): (2, ""),  # Alchemy Mysteries: Prague Legends
    ("longtail_detective", 935580): (2, ""),  # True Fear: Forsaken Souls Part 2
    ("longtail_detective", 1259640): (3, ""),  # Tiny Room Stories: Town Mystery
    ("longtail_metroidvania", 400630): (2, ""),  # Wuppo: Definitive Edition
    ("longtail_metroidvania", 444720): (3, ""),  # Inexistence
    ("longtail_metroidvania", 1436590): (3, ""),  # Phoenotopia: Awakening
    ("longtail_metroidvania", 2230650): (3, ""),  # TEVI
    ("longtail_puzzle_platformer", 221810): (3, ""),  # The Cave
    ("longtail_puzzle_platformer", 331480): (3, ""),  # Pinstripe
    ("longtail_puzzle_platformer", 493200): (3, ""),  # RiME
    ("longtail_puzzle_platformer", 2420660): (3, ""),  # Neva
    ("lowrev_cozy_narrative", 370280): (3, ""),  # Season of 12 Colors
    ("lowrev_cozy_narrative", 685670): (3, ""),  # Know by heart
    ("lowrev_cozy_narrative", 839450): (2, ""),  # 7'scarlet
    ("lowrev_cozy_narrative", 2076140): (3, ""),  # 태번 토크
    ("lowrev_deckbuilder", 861540): (3, ""),  # Dicey Dungeons
    ("lowrev_deckbuilder", 1619520): (3, ""),  # Cross Blitz
    ("lowrev_deckbuilder", 2746910): (2, ""),  # 삼국 영웅：혈월의 전설
    ("lowrev_detective", 1410640): (3, ""),  # Syberia: The World Before
    ("lowrev_detective", 2458980): (2, ""),  # Dimhaven - The Lost Source
    ("lowrev_detective", 2859200): (2, ""),  # Alex Hill: Whispers at White Oak Inn
    ("lowrev_detective", 3072450): (3, ""),  # FAKEBOOK : 페이크북
    ("lowrev_detective", 3148060): (3, ""),  # Penelope Pendrick and the Art of Deceit
    ("lowrev_metroidvania", 409660): (2, ""),  # Angry Video Game Nerd II: ASSimilation
    ("lowrev_metroidvania", 814680): (3, ""),  # Unbound: Worlds Apart
    ("lowrev_metroidvania", 1489410): (3, ""),  # Alice Escaped!
    ("lowrev_metroidvania", 1794780): (1, "MODE_MISMATCH"),  # SiNiSistar Lite Version
    ("lowrev_towerdefense", 221540): (3, ""),  # DG2: Defense Grid 2
    ("lowrev_towerdefense", 275670): (3, ""),  # Space Run
    ("lowrev_towerdefense", 2617400): (3, ""),  # 极简塔防 - Minimalist Tower Defense
    ("lowrev_towerdefense", 2800900): (3, ""),  # Rift Riff
    ("lowrev_towerdefense", 4053520): (2, ""),  # Holo vs Robo
    ("mix2_arcade_action", 311400): (2, ""),  # OCEAN CITY RACING
    ("mix2_arcade_action", 477770): (2, ""),  # Ride 2
    ("mix2_arcade_action", 661290): (2, ""),  # Arizona Derby
    ("mix2_arcade_action", 1301010): (2, ""),  # World Racing 2 - Champion Edition
    ("mix2_arcade_action", 1664220): (2, ""),  # TRAIL OUT
    ("mix2_arcade_action", 1849250): (2, ""),  # EA SPORTS™ WRC
    ("mix2_arcade_action", 2073470): (3, ""),  # Kanjozoku Game レーサー Online Street Racing
    ("mix2_arcade_action", 3278310): (2, ""),  # LANESPLIT
    ("mix2_builder_sim", 1060230): (3, ""),  # Sapiens
    ("mix2_builder_sim", 1084600): (3, ""),  # My Time at Sandrock
    ("mix2_builder_sim", 1455910): (3, ""),  # 영지: 농사와 전쟁 (Territory: Farming and Warfa
    ("mix2_builder_sim", 2195120): (3, ""),  # Go-Go Town!
    ("mix2_colony_cozy", 305940): (3, ""),  # Project AURA
    ("mix2_colony_cozy", 821250): (3, ""),  # Flotsam
    ("mix2_colony_cozy", 1363900): (3, ""),  # Farworld Pioneers
    ("mix2_colony_cozy", 1465550): (3, ""),  # One Lonely Outpost
    ("mix2_colony_cozy", 1620290): (3, ""),  # Zombie Cure Lab
    ("mix2_colony_cozy", 2706020): (3, ""),  # ALL WILL FALL
    ("mix2_coop_horror", 1250): (3, ""),  # Killing Floor
    ("mix2_crpg_sandbox", 1159290): (2, ""),  # The Bloodline
    ("mix2_crpg_sandbox", 3098140): (2, ""),  # Crown and Adventure
    ("mix2_modern_roguelite", 913740): (3, ""),  # WORLD OF HORROR
    ("mix2_modern_roguelite", 2334730): (3, ""),  # Death Must Die
    ("mix2_modern_roguelite", 2780710): (3, ""),  # Asgard's Fall — Viking Survivors
    ("mix2_party_narrative", 2357000): (2, ""),  # KILL IT WITH FIRE! 2
    ("mix2_puzzle_survival", 280790): (2, ""),  # Creativerse
    ("mix2_puzzle_survival", 315460): (2, ""),  # Dig or Die
    ("mix2_puzzle_survival", 391730): (2, ""),  # Crashlands
    ("mix2_puzzle_survival", 846770): (3, ""),  # DYSMANTLE
    ("mix2_puzzle_survival", 1142080): (2, ""),  # Pawnbarian
    ("mix2_puzzle_survival", 1750570): (2, ""),  # Subway Midnight
    ("mix2_puzzle_survival", 2476100): (2, ""),  # Creepy Tale: Some Other Place
    ("mix2_soulslike_narrative", 265300): (3, ""),  # Lords Of The Fallen™ 2014
    ("mix2_soulslike_narrative", 466300): (3, ""),  # Planescape: Torment: Enhanced Edition
    ("mix2_soulslike_narrative", 609150): (2, ""),  # STAR OCEAN™ - THE LAST HOPE -™ 4K & Full
    ("mix2_soulslike_narrative", 628670): (3, ""),  # Hellpoint
    ("mix2_soulslike_narrative", 1620730): (3, ""),  # Hell is Us
    ("mix2_soulslike_narrative", 2186990): (2, ""),  # Fatekeeper
    ("mix2_soulslike_narrative", 2515020): (3, ""),  # FINAL FANTASY XVI
    ("mix2_soulslike_narrative", 3107880): (1, "LOW_QUALITY"),  # Gnomes and Knights
    ("niche_cozy_casual", 204180): (2, ""),  # Waveform
    ("niche_cozy_casual", 284950): (2, ""),  # Pixel Puzzles: Japan
    ("niche_cozy_casual", 355150): (2, ""),  # gravilon
    ("niche_cozy_casual", 395160): (2, ""),  # Toby: The Secret Mine
    ("niche_cozy_casual", 556240): (2, ""),  # Disoriented
    ("niche_cozy_casual", 1333910): (3, ""),  # Sizeable
    ("niche_cozy_casual", 3694480): (2, ""),  # A Game About Feeding A Black Hole
    ("niche_roguelite", 1217060): (3, ""),  # Gunfire Reborn
    ("niche_roguelite", 1679220): (3, ""),  # 포탈 던전
    ("niche_roguelite", 1843840): (2, ""),  # Rogue Point
    ("niche_roguelite", 3363680): (2, ""),  # EverSiege: Untold Ages
    ("niche_sim", 573490): (2, ""),  # Interstellar Transport Company
    ("niche_sim", 1013540): (3, ""),  # Evospace
    ("niche_sim", 1511460): (3, ""),  # InfraSpace
    ("niche_sim", 1638300): (3, ""),  # Solargene
    ("niche_sim", 1723260): (3, ""),  # CaseCracker
    ("niche_sim", 3613270): (3, ""),  # Star Ores Inc.
    ("niche_tactics", 1038370): (3, ""),  # Trials of Fire
    ("niche_tactics", 1909420): (2, ""),  # Tinyfolks
    ("niche_tactics", 2026820): (3, ""),  # Die in the Dungeon
    ("niche_tactics", 3064290): (3, ""),  # Trizon
}


BLIND_STRATEGY_S4 = {
    ("coh_arpg", 1132980): (2, ""),  # 빛 없는 세계: There is No Light
    ("coh_classic_multi", 2795540): (2, ""),  # 미드나잇 워커스
    ("coh_cozy", 105600): (3, ""),  # Terraria
    ("coh_indie_platformer", 1419160): (3, ""),  # Souldiers
    ("coh_indie_platformer", 2420660): (3, ""),  # Neva
    ("coh_strategy", 804850): (3, ""),  # Pax Nova
    ("coh_strategy", 931280): (2, ""),  # Iron Marines
    ("coh_strategy", 2275440): (3, ""),  # Solar Nations 2
    ("longtail_detective", 1042490): (3, ""),  # Call of the Sea
    ("longtail_puzzle_platformer", 1781260): (2, ""),  # Promenade
    ("lowrev_cozy_narrative", 965860): (2, ""),  # SnowNight
    ("lowrev_cozy_narrative", 1238730): (2, ""),  # Flowers -Le volume sur automne-
    ("lowrev_cozy_narrative", 2515070): (1, "MODE_MISMATCH"),  # あくありうむ。
    ("lowrev_cozy_narrative", 3687630): (1, "MODE_MISMATCH"),  # 이런 녀석이 있습니다
    ("lowrev_cozy_narrative", 4179020): (1, "MODE_MISMATCH"),  # 여름미녀, 심장 폭주 주의보!
    ("lowrev_detective", 368390): (3, ""),  # The Darkside Detective
    ("lowrev_detective", 2714620): (3, ""),  # Duck Detective: The Ghost of Glamping
    ("mix2_colony_cozy", 1104330): (3, ""),  # Founders' Fortune
    ("mix2_crpg_sandbox", 810040): (2, ""),  # Swords 'n Magic and Stuff
    ("mix2_crpg_sandbox", 3020510): (2, ""),  # Legend of Heroes: Three Kingdoms
    ("mix2_modern_roguelite", 2665680): (3, ""),  # 바벨탑: 혼돈의 생존자들
    ("mix2_party_narrative", 1335790): (3, ""),  # Operation: Tango
    ("mix2_party_narrative", 1686940): (2, ""),  # Bopl Battle
    ("mix2_party_narrative", 2543180): (2, ""),  # 팝유컴
    ("mix2_puzzle_survival", 876650): (2, ""),  # Aground
    ("mix2_soulslike_narrative", 644830): (3, ""),  # The Surge 2
    ("mix2_soulslike_narrative", 814380): (3, ""),  # Sekiro™: Shadows Die Twice - GOTY Editio
    ("mix2_soulslike_narrative", 1491410): (2, ""),  # Eternal Strands
    ("mix2_soulslike_narrative", 2091020): (2, ""),  # 블레이드 오브 파이어 Blades of Fire
    ("niche_cozy_casual", 367580): (2, ""),  # Hook
    ("niche_cozy_casual", 383270): (3, ""),  # Hue
    ("niche_cozy_casual", 2596420): (3, ""),  # Arranger: A Role-Puzzling Adventure
    ("niche_roguelite", 2015270): (3, ""),  # Rotwood
    ("niche_roguelite", 2543510): (3, ""),  # GUNTOUCHABLES
    ("niche_roguelite", 2956680): (3, ""),  # LORT
}


BLIND_STRATEGY_S5 = {
    ("coh_classic_multi", 636480): (3, ""),  # Ravenfield
    ("coh_cozy", 1644940): (3, ""),  # Loddlenaut
    ("coh_fps", 360940): (2, ""),  # The Mean Greens - Plastic Warfare
    ("coh_grand_strategy", 98300): (1, "GENRE_ONLY"),  # Toy Soldiers
    ("coh_grand_strategy", 298480): (2, ""),  # Victory At Sea
    ("coh_grand_strategy", 338800): (3, ""),  # Seven Kingdoms 2 HD
    ("coh_grand_strategy", 758370): (3, ""),  # Through the Ages
    ("coh_strategy", 637090): (3, ""),  # BATTLETECH
    ("coh_vehicle_sim", 759740): (2, ""),  # RIDE 3
    ("coh_vehicle_sim", 3058630): (3, ""),  # 아세토 코르사 에보 Assetto Corsa EVO
    ("longtail_deckbuilder", 1266430): (3, ""),  # 迷失幻途 Lost In Fantaland
    ("longtail_deckbuilder", 1267470): (3, ""),  # Warriors of the Nile
    ("longtail_metroidvania", 1069530): (3, ""),  # Narita Boy
    ("longtail_puzzle_platformer", 3288210): (2, ""),  # Super Meat Boy 3D
    ("lowrev_towerdefense", 1269690): (2, ""),  # Attack on Toys
    ("mix2_arcade_action", 598980): (1, "IRRELEVANT"),  # The Coin Game
    ("mix2_arcade_action", 808910): (2, ""),  # STAR WARS™ Episode I Racer
    ("mix2_arcade_action", 850980): (3, ""),  # GUILTY GEAR
    ("mix2_arcade_action", 1058450): (3, ""),  # MY HERO ONE'S JUSTICE 2
    ("mix2_arcade_action", 2215260): (2, ""),  # Scott Pilgrim vs. The World™: The Game –
    ("mix2_builder_sim", 860890): (3, ""),  # Factory Town
    ("mix2_colony_cozy", 1651490): (2, ""),  # Tiny Life
    ("mix2_party_narrative", 206440): (3, ""),  # To the Moon
    ("mix2_party_narrative", 493200): (3, ""),  # RiME
    ("mix2_party_narrative", 828900): (3, ""),  # The Stillness of the Wind
    ("mix2_party_narrative", 1410640): (3, ""),  # Syberia: The World Before
    ("mix2_party_narrative", 2726490): (1, "LOW_QUALITY"),  # Hamster Hunter
    ("mix2_party_narrative", 3352690): (2, ""),  # Dead Take
    ("mix2_puzzle_survival", 493200): (2, ""),  # RiME
    ("mix2_puzzle_survival", 2738490): (2, ""),  # Sol Cesto
    ("mix2_soulslike_narrative", 110800): (3, ""),  # L.A. Noire
    ("mix2_soulslike_narrative", 1684350): (3, ""),  # 디 쏘마터지
    ("mix2_soulslike_narrative", 2060790): (2, ""),  # 하운티
    ("mix2_survival_farm", 405710): (3, ""),  # Staxel
    ("mix2_survival_farm", 1245560): (3, ""),  # Roots of Pacha
    ("mix2_survival_farm", 1416960): (3, ""),  # Everafter Falls
    ("mix2_survival_farm", 2918500): (3, ""),  # Cozy Island
    ("niche_sim", 252870): (2, ""),  # PULSAR: Lost Colony
    ("niche_sim", 712730): (3, ""),  # SIMULACRA
    ("niche_tactics", 1937750): (2, ""),  # Prime of Flames
    ("niche_tactics", 2084000): (3, ""),  # Shogun Showdown
}


BLIND_FINAL_F30 = {
    ("coh_arpg", 218680): (0, "IRRELEVANT"),  # Scribblenauts Unlimited
    ("coh_arpg", 1841240): (3, ""),  # FOUNTAINS
    ("coh_classic_multi", 219830): (2, ""),  # King Arthur's Gold
    ("coh_classic_multi", 665120): (2, ""),  # Jetman Go
    ("coh_classic_multi", 667530): (3, ""),  # Drunken Wrestlers 2
    ("coh_classic_multi", 2592220): (3, ""),  # CUFFBUST
    ("coh_cozy", 115800): (2, ""),  # Owlboy
    ("coh_cozy", 210970): (1, "MODE_MISMATCH"),  # The Witness
    ("coh_cozy", 526870): (2, ""),  # Satisfactory
    ("coh_cozy", 1144770): (1, "GENRE_ONLY"),  # SLUDGE LIFE
    ("coh_cozy", 1596940): (1, "LOW_QUALITY"),  # Ryan의 구조 팀
    ("coh_cozy", 1709170): (3, ""),  # Paradise Marsh
    ("coh_cozy", 1865010): (3, ""),  # 皓际大冒险
    ("coh_cozy", 1974220): (3, ""),  # Eden Island
    ("coh_fps", 4683090): (2, ""),  # Hyper Warfare
    ("coh_grand_strategy", 454530): (3, ""),  # Decisive Campaigns: Barbarossa
    ("coh_grand_strategy", 1041710): (2, ""),  # 模拟帝国
    ("coh_grand_strategy", 1135240): (2, ""),  # Men of War: Assault Squad 2 - Cold War
    ("coh_grand_strategy", 3430910): (3, ""),  # Samurai Conqueror
    ("coh_grand_strategy", 4588330): (2, ""),  # Dynastic Era
    ("coh_indie_platformer", 727510): (3, ""),  # Void Memory
    ("coh_indie_platformer", 751640): (3, ""),  # Dead Dungeon
    ("coh_indie_platformer", 1252830): (2, ""),  # A Juggler's Tale
    ("coh_indie_platformer", 1710100): (2, ""),  # 이상한 나라 모험기
    ("coh_indie_platformer", 1848710): (3, ""),  # 妖刀退魔忍
    ("coh_indie_platformer", 3372060): (2, ""),  # Hell Maiden
    ("coh_openworld_survival", 237870): (3, ""),  # Planet Explorers
    ("coh_strategy", 328440): (3, ""),  # Deadlock: Planetary Conquest
    ("coh_strategy", 1154840): (3, ""),  # Shadow Empire
    ("coh_survival_craft", 378370): (2, ""),  # Nomad
    ("coh_survival_craft", 1250220): (3, ""),  # Fallen Region
    ("coh_vehicle_sim", 228760): (2, ""),  # TrackMania² Canyon
    ("coh_vehicle_sim", 273730): (2, ""),  # Driving School Simulator
    ("coh_vehicle_sim", 1209360): (3, ""),  # Monster Truck Championship
    ("coh_vehicle_sim", 2947540): (2, ""),  # Military Logistics Simulator
    ("longtail_deckbuilder", 856990): (3, ""),  # A Long Way Down
    ("longtail_deckbuilder", 1148650): (3, ""),  # The Legend of Bum-Bo
    ("longtail_deckbuilder", 1904860): (3, ""),  # Legends of Kingdom Rush
    ("longtail_deckbuilder", 2082410): (3, ""),  # Anomaly Collapse
    ("longtail_deckbuilder", 2097030): (3, ""),  # Union of Gnomes
    ("longtail_deckbuilder", 2097570): (3, ""),  # StarVaders
    ("longtail_deckbuilder", 2275940): (3, ""),  # Rogue Hex
    ("longtail_deckbuilder", 3009310): (2, ""),  # 마스터 오브 피스
    ("longtail_deckbuilder", 3734140): (2, ""),  # Super Cabbage Kabumi
    ("longtail_detective", 50990): (2, ""),  # Mystery Case Files: Ravenhearst®
    ("longtail_detective", 291630): (2, ""),  # Mirror Mysteries
    ("longtail_detective", 801480): (3, ""),  # Agent A: A puzzle in disguise
    ("longtail_detective", 1097110): (3, ""),  # Willy Morgan and the Curse of Bone Town
    ("longtail_detective", 1352200): (3, ""),  # Detective From The Crypt
    ("longtail_detective", 2458980): (2, ""),  # Dimhaven - The Lost Source
    ("longtail_detective", 2741620): (3, ""),  # Exiled
    ("longtail_detective", 2885870): (3, ""),  # Strange Antiquities
    ("longtail_detective", 3296830): (3, ""),  # CaseCracker3
    ("longtail_detective", 3517980): (3, ""),  # Secrets of Blackrock Manor - Escape Room
    ("longtail_detective", 3636620): (3, ""),  # The Last Case of John Morley
    ("longtail_metroidvania", 482330): (3, ""),  # Legends of the Universe - StarCore
    ("longtail_metroidvania", 1583520): (3, ""),  # Under The Island
    ("longtail_metroidvania", 1772830): (3, ""),  # Rusted Moss
    ("longtail_metroidvania", 2112750): (3, ""),  # Toziuha Night: Order of the Alchemists
    ("longtail_metroidvania", 2997230): (2, ""),  # Planet of Lana II
    ("longtail_metroidvania", 3055950): (3, ""),  # Shantae Advance: Risky Revolution
    ("longtail_puzzle_platformer", 15740): (2, ""),  # Oddworld: Munch's Oddysee
    ("longtail_puzzle_platformer", 562520): (3, ""),  # Attempt[42]
    ("longtail_puzzle_platformer", 1162280): (3, ""),  # Eternal Hope
    ("longtail_puzzle_platformer", 2667950): (3, ""),  # Gloomy Eyes
    ("lowrev_cozy_narrative", 1406810): (3, ""),  # After Us
    ("lowrev_cozy_narrative", 1550010): (3, ""),  # The Witch of Fern Island
    ("lowrev_cozy_narrative", 2342210): (2, ""),  # GENIE
    ("lowrev_cozy_narrative", 3041730): (3, ""),  # Go North
    ("lowrev_deckbuilder", 2452820): (3, ""),  # Skogdal
    ("lowrev_detective", 1844330): (3, ""),  # Profession investigator
    ("lowrev_detective", 1919600): (3, ""),  # Paper Perjury
    ("lowrev_detective", 2757490): (0, "IRRELEVANT"),  # Petit Game Collection vol.1
    ("lowrev_detective", 4054820): (3, ""),  # Murder Mystery Stories
    ("lowrev_metroidvania", 482330): (3, ""),  # Legends of the Universe - StarCore
    ("lowrev_metroidvania", 569290): (2, ""),  # Metamorfose S
    ("lowrev_metroidvania", 664830): (2, ""),  # Zombotron
    ("lowrev_metroidvania", 1262040): (3, ""),  # Super Mombo Quest
    ("lowrev_metroidvania", 1274600): (3, ""),  # The Last Faith
    ("lowrev_metroidvania", 1444080): (3, ""),  # Dewdrop Dynasty
    ("lowrev_towerdefense", 253790): (2, ""),  # rymdkapsel
    ("lowrev_towerdefense", 302970): (3, ""),  # Radiant Defense
    ("lowrev_towerdefense", 538920): (3, ""),  # Fiery Disaster
    ("lowrev_towerdefense", 809440): (3, ""),  # Protolife
    ("lowrev_towerdefense", 915490): (2, ""),  # Frontline Zed
    ("lowrev_towerdefense", 1198740): (3, ""),  # Colony Siege
    ("lowrev_towerdefense", 1252680): (3, ""),  # SENTRY
    ("lowrev_towerdefense", 1540920): (3, ""),  # Neonverse Defenders
    ("lowrev_towerdefense", 1560880): (3, ""),  # Automatrons - Tower Defense
    ("lowrev_towerdefense", 2972530): (3, ""),  # Playground Mayhem
    ("mix2_arcade_action", 452510): (3, ""),  # UNDER NIGHT IN-BIRTH Exe:Late
    ("mix2_arcade_action", 457710): (2, ""),  # Road Madness
    ("mix2_arcade_action", 1103730): (2, ""),  # Shing!
    ("mix2_arcade_action", 2025320): (1, "MODE_MISMATCH"),  # Estencel
    ("mix2_arcade_action", 2283380): (1, "IRRELEVANT"),  # Charrua Soccer - Pro Edition
    ("mix2_arcade_action", 2350280): (2, ""),  # Monster Jam™ Showdown
    ("mix2_arcade_action", 2362050): (3, ""),  # 나의 히어로 아카데미아 올즈 저스티스
    ("mix2_arcade_action", 4182550): (2, ""),  # STICKMAN DX : MEGA FIGHTERS
    ("mix2_builder_sim", 307760): (2, ""),  # Zoo Park
    ("mix2_builder_sim", 361420): (3, ""),  # ASTRONEER
    ("mix2_builder_sim", 1915250): (3, ""),  # Constructed
    ("mix2_builder_sim", 1944210): (3, ""),  # Infindustry
    ("mix2_builder_sim", 3494930): (3, ""),  # Assemblands
    ("mix2_colony_cozy", 284100): (3, ""),  # Unclaimed World
    ("mix2_colony_cozy", 759000): (1, "IRRELEVANT"),  # .projekt
    ("mix2_colony_cozy", 895060): (2, ""),  # My House
    ("mix2_colony_cozy", 2200780): (3, ""),  # My Dream Setup
    ("mix2_colony_cozy", 2352680): (3, ""),  # Colonization Simulator
    ("mix2_colony_cozy", 2443960): (3, ""),  # Room In Dream
    ("mix2_colony_cozy", 3416070): (2, ""),  # 로푸카의 한가한 섬
    ("mix2_colony_cozy", 3441280): (2, ""),  # The Greenening
    ("mix2_coop_horror", 385380): (1, "GENRE_ONLY"),  # Planet Centauri
    ("mix2_coop_horror", 1721110): (3, ""),  # Abyssus
    ("mix2_coop_horror", 2328750): (2, ""),  # 나의 작은 우주 (My Little Universe)
    ("mix2_crpg_sandbox", 506480): (2, ""),  # Archmage Rises
    ("mix2_crpg_sandbox", 839500): (3, ""),  # 血战夹皮沟
    ("mix2_crpg_sandbox", 1468720): (1, "MODE_MISMATCH"),  # Ultimate Epic Battle Simulator 2
    ("mix2_crpg_sandbox", 1769420): (3, ""),  # Noble Fates
    ("mix2_crpg_sandbox", 2336720): (2, ""),  # Greed of Man
    ("mix2_crpg_sandbox", 2761000): (3, ""),  # Tales of Old: Dominus
    ("mix2_crpg_sandbox", 3074020): (2, ""),  # 异世界生存指南
    ("mix2_modern_roguelite", 787810): (3, ""),  # Rogue Heroes: Ruins of Tasos
    ("mix2_modern_roguelite", 1761380): (3, ""),  # Otherworld Legends
    ("mix2_modern_roguelite", 3001330): (3, ""),  # 포레스트 히어로즈
    ("mix2_modern_roguelite", 3734140): (2, ""),  # Super Cabbage Kabumi
    ("mix2_party_narrative", 43600): (2, ""),  # Nancy Drew®: Warnings at Waverly Academy
    ("mix2_party_narrative", 294550): (1, "MODE_MISMATCH"),  # Freddi Fish 4: The Case of the Hogfish R
    ("mix2_party_narrative", 399890): (1, "GENRE_ONLY"),  # The Secret Order 2: Masked Intent
    ("mix2_party_narrative", 466980): (3, ""),  # Can't Drive This
    ("mix2_party_narrative", 600990): (3, ""),  # The Gardens Between
    ("mix2_party_narrative", 843200): (2, ""),  # Alien Hominid Invasion
    ("mix2_party_narrative", 1458300): (3, ""),  # The Gap
    ("mix2_party_narrative", 1859270): (3, ""),  # Perfect Partner
    ("mix2_puzzle_survival", 466300): (2, ""),  # Planescape: Torment: Enhanced Edition
    ("mix2_puzzle_survival", 473950): (3, ""),  # Manifold Garden
    ("mix2_puzzle_survival", 699700): (2, ""),  # DYING: Reborn
    ("mix2_puzzle_survival", 960690): (3, ""),  # One Step From Eden
    ("mix2_puzzle_survival", 1622770): (2, ""),  # Doors: Paradox
    ("mix2_puzzle_survival", 2470270): (2, ""),  # The Endless Dream
    ("mix2_puzzle_survival", 2822980): (2, ""),  # ARSONATE
    ("mix2_puzzle_survival", 3495730): (2, ""),  # Lucid Blocks
    ("mix2_soulslike_narrative", 1137300): (3, ""),  # Sherlock Holmes Chapter One
    ("mix2_soulslike_narrative", 2321880): (2, ""),  # Spire Horizon
    ("mix2_soulslike_narrative", 3552430): (1, "GENRE_ONLY"),  # Dead Silence - Echoes of the Damned
    ("mix2_survival_farm", 1432860): (3, ""),  # Sun Haven
    ("niche_cozy_casual", 420770): (2, ""),  # The Legend of Dark Witch
    ("niche_cozy_casual", 562520): (2, ""),  # Attempt[42]
    ("niche_cozy_casual", 759000): (3, ""),  # .projekt
    ("niche_cozy_casual", 775530): (2, ""),  # Puzzlement
    ("niche_cozy_casual", 969760): (3, ""),  # Omno
    ("niche_cozy_casual", 1898290): (3, ""),  # LEGO® Bricktales
    ("niche_cozy_casual", 2573930): (3, ""),  # Skolly's Adventure
    ("niche_cozy_casual", 3013010): (3, ""),  # Arbor Island
    ("niche_cozy_casual", 3214530): (2, ""),  # Treasure 'n Trio
    ("niche_cozy_casual", 3219450): (1, "IRRELEVANT"),  # Nonogram Halloween
    ("niche_roguelite", 1802880): (3, ""),  # 데블위딘 삿갓
    ("niche_sim", 1247940): (3, ""),  # Greyhat - A Digital Detective Adventure
    ("niche_sim", 1803550): (3, ""),  # MegaFactory Titan
    ("niche_sim", 2421410): (3, ""),  # Cyber Manhunt 2: New World - The Hacking
    ("niche_sim", 3433600): (2, ""),  # Space Sim Tycoon
    ("niche_tactics", 253790): (3, ""),  # rymdkapsel
    ("niche_tactics", 598960): (2, ""),  # Mashinky
    ("niche_tactics", 993690): (2, ""),  # Transport Services
    ("niche_tactics", 1046030): (3, ""),  # ISLANDERS
    ("niche_tactics", 1957990): (2, ""),  # Tile Cities
    ("niche_tactics", 2450400): (3, ""),  # Trash of the Titans
    ("niche_tactics", 3009310): (3, ""),  # 마스터 오브 피스
    ("niche_tactics", 3187880): (3, ""),  # The Fable: Manga Build Roguelike
    ("niche_tactics", 3623530): (3, ""),  # Deified
}


BLIND_FINAL_F50 = {
    ("coh_arpg", 367500): (3, ""),  # Dragon's Dogma: Dark Arisen
    ("coh_arpg", 609150): (2, ""),  # STAR OCEAN™ - THE LAST HOPE -™ 4K & Full
    ("coh_arpg", 1045180): (3, ""),  # Shattered - Tale of the Forgotten King
    ("coh_arpg", 1116210): (1, ""),  # Our world has not decayed
    ("coh_arpg", 1173220): (3, ""),  # Bleak Faith: Forsaken
    ("coh_arpg", 1297900): (3, ""),  # Gothic 1 Remake
    ("coh_arpg", 1634940): (2, ""),  # The Altered Lands
    ("coh_arpg", 2362060): (2, ""),  # 코드 베인 II
    ("coh_arpg", 2429270): (2, ""),  # The RPG
    ("coh_arpg", 2552450): (2, ""),  # KINGDOM HEARTS III + Re Mind (DLC)
    ("coh_arpg", 4433150): (1, ""),  # Idle Realms: The Eternal Spire
    ("coh_classic_multi", 24240): (2, ""),  # PAYDAY™ The Heist
    ("coh_classic_multi", 28000): (2, ""),  # Kane & Lynch 2: Dog Days
    ("coh_classic_multi", 104900): (3, ""),  # ORION: Prelude
    ("coh_classic_multi", 300260): (2, ""),  # Planetoid Pioneers
    ("coh_classic_multi", 357670): (2, ""),  # Home Improvisation: Furniture Sandbox
    ("coh_classic_multi", 433850): (2, ""),  # Z1 Battle Royale
    ("coh_classic_multi", 500260): (2, ""),  # Ben and Ed - Blood Party
    ("coh_classic_multi", 512160): (1, ""),  # Volleying
    ("coh_classic_multi", 729040): (3, ""),  # Borderlands Game of the Year Enhanced
    ("coh_classic_multi", 1797680): (2, ""),  # Wee Tanks!
    ("coh_classic_multi", 2627570): (2, ""),  # Goofy Gorillas
    ("coh_classic_multi", 3111080): (1, ""),  # Secret Agent Wizard Boy and the Internat
    ("coh_classic_multi", 3504700): (1, ""),  # 4Wheel Challenge (휠체어 시뮬레이터)
    ("coh_classic_multi", 3738960): (2, ""),  # ShitSlam2
    ("coh_classic_multi", 4069520): (2, ""),  # Super Battle Golf
    ("coh_classic_multi", 4683090): (2, ""),  # Hyper Warfare
    ("coh_cozy", 203160): (1, ""),  # Tomb Raider Game of the Year
    ("coh_cozy", 363970): (1, ""),  # Clicker Heroes
    ("coh_cozy", 552130): (0, "IRRELEVANT"),  # Whiplash - Crash Valley
    ("coh_cozy", 574180): (2, ""),  # Rem Survival
    ("coh_cozy", 715560): (3, ""),  # Eastshade
    ("coh_cozy", 747200): (2, ""),  # Keplerth
    ("coh_cozy", 1004270): (3, ""),  # My Island
    ("coh_cozy", 1066290): (2, ""),  # 猎魔者战纪
    ("coh_cozy", 1084600): (3, ""),  # My Time at Sandrock
    ("coh_cozy", 1119730): (3, ""),  # Ranch Simulator: Build, Hunt, Farm
    ("coh_cozy", 1329510): (3, ""),  # Big Farm Story
    ("coh_cozy", 1353300): (1, ""),  # Idle Slayer
    ("coh_cozy", 1562260): (2, ""),  # Camping Simulator: The Squad
    ("coh_cozy", 1638340): (1, ""),  # Cat Simulator : Animals on Farm
    ("coh_cozy", 4426330): (1, ""),  # Slime Kingdom: Idle
    ("coh_fps", 6000): (2, ""),  # STAR WARS™ Republic Commando™
    ("coh_fps", 298240): (1, ""),  # War Trigger 3
    ("coh_fps", 393420): (2, ""),  # Hurtworld
    ("coh_fps", 530630): (1, ""),  # Empires Apart
    ("coh_fps", 674020): (3, ""),  # World War 3
    ("coh_fps", 753650): (3, ""),  # Due Process
    ("coh_fps", 1092050): (2, ""),  # Fire On Fight : Online Multiplayer Shoot
    ("coh_fps", 1358710): (2, ""),  # Battle Cry of Freedom
    ("coh_fps", 1679010): (2, ""),  # BLOCKPOST MOBILE
    ("coh_fps", 2215270): (2, ""),  # Army Troop
    ("coh_fps", 2281730): (3, ""),  # Combat Master: Season 5
    ("coh_fps", 2484960): (1, ""),  # 1v1 Deathmatch
    ("coh_fps", 3286280): (3, ""),  # Operation : Strike & Secure
    ("coh_fps", 3889960): (3, ""),  # Warface
    ("coh_grand_strategy", 654890): (3, ""),  # Grand Tactician: The Civil War (1861-186
    ("coh_grand_strategy", 674190): (3, ""),  # Conquest of the New World
    ("coh_grand_strategy", 867210): (3, ""),  # Songs of Conquest
    ("coh_grand_strategy", 874390): (3, ""),  # The Battle of Polytopia
    ("coh_grand_strategy", 1116880): (3, ""),  # 伊格利亚战记/The Heroic Legend Of Eagarlnia
    ("coh_grand_strategy", 1248060): (3, ""),  # Realpolitiks II
    ("coh_grand_strategy", 1270100): (2, ""),  # 兵法：战国篇
    ("coh_grand_strategy", 1501690): (2, ""),  # Dice Kingdoms
    ("coh_grand_strategy", 2219390): (3, ""),  # Imperial Ambitions
    ("coh_grand_strategy", 3867100): (3, ""),  # Of Crowns & Chains
    ("coh_grand_strategy", 4280860): (2, ""),  # Warhammer: Dark Omen (Classic)
    ("coh_indie_platformer", 209190): (2, ""),  # Stealth Bastard Deluxe
    ("coh_indie_platformer", 281750): (2, ""),  # Munin
    ("coh_indie_platformer", 331440): (2, ""),  # bit Dungeon II
    ("coh_indie_platformer", 549260): (3, ""),  # Alwa's Awakening
    ("coh_indie_platformer", 1123770): (3, ""),  # Curse of the Dead Gods
    ("coh_indie_platformer", 1367300): (3, ""),  # Blade Assault
    ("coh_indie_platformer", 2230650): (3, ""),  # TEVI
    ("coh_indie_platformer", 2322440): (1, ""),  # METAL SLUG ATTACK RELOADED
    ("coh_indie_platformer", 2650350): (2, ""),  # Super Puzzled Cat
    ("coh_indie_platformer", 2721890): (3, ""),  # Öoo
    ("coh_indie_platformer", 2845630): (2, ""),  # Ocean Keeper: Dome Survival
    ("coh_indie_platformer", 4718470): (3, ""),  # Dungeon door
    ("coh_openworld_survival", 210970): (1, ""),  # The Witness
    ("coh_openworld_survival", 280520): (2, ""),  # Crea
    ("coh_openworld_survival", 383120): (3, ""),  # Empyrion - Galactic Survival
    ("coh_openworld_survival", 1125390): (2, ""),  # Atrio: The Dark Wild
    ("coh_openworld_survival", 1145290): (2, ""),  # Out There: Oceans of Time
    ("coh_openworld_survival", 1270010): (2, ""),  # Gone: Survival
    ("coh_openworld_survival", 1635450): (2, ""),  # Longvinter
    ("coh_openworld_survival", 2383130): (3, ""),  # Project: Mist
    ("coh_openworld_survival", 3054440): (3, ""),  # 짧은 눈 / Short Snow
    ("coh_openworld_survival", 3223650): (2, ""),  # Trapped In The Omniverse
    ("coh_openworld_survival", 3576870): (3, ""),  # Blossom: The Seed of Life
    ("coh_strategy", 2210): (1, ""),  # Quake 4
    ("coh_strategy", 237950): (3, ""),  # UFO: Afterlight
    ("coh_strategy", 308173): (3, ""),  # Hegemony III: Clash of the Ancients
    ("coh_strategy", 314980): (3, ""),  # Supreme Ruler Ultimate
    ("coh_strategy", 396480): (3, ""),  # Battlevoid: Harbinger
    ("coh_strategy", 465490): (3, ""),  # Stellar Tactics
    ("coh_strategy", 867210): (3, ""),  # Songs of Conquest
    ("coh_strategy", 1351080): (2, ""),  # Pharaoh™: A New Era
    ("coh_strategy", 1482820): (3, ""),  # Void Marauders
    ("coh_strategy", 2449450): (2, ""),  # 우주 식민 회사
    ("coh_strategy", 2920380): (3, ""),  # Xenowars
    ("coh_strategy", 3336660): (3, ""),  # Star Titans : War of the Galaxy
    ("coh_survival_craft", 252490): (3, ""),  # Rust
    ("coh_survival_craft", 351290): (2, ""),  # SURVIVAL: Postapocalypse Now
    ("coh_survival_craft", 372750): (2, ""),  # Girl Amazon Survival
    ("coh_survival_craft", 453270): (1, ""),  # Madness Cubed
    ("coh_survival_craft", 509770): (3, ""),  # PostCollapse
    ("coh_survival_craft", 648800): (3, ""),  # Raft
    ("coh_survival_craft", 696250): (2, ""),  # Lost Shipwreck
    ("coh_survival_craft", 846770): (3, ""),  # DYSMANTLE
    ("coh_survival_craft", 924140): (2, ""),  # Hand Simulator: Survival
    ("coh_survival_craft", 962130): (3, ""),  # Grounded
    ("coh_survival_craft", 1253220): (2, ""),  # The Gold River Project
    ("coh_survival_craft", 1355780): (3, ""),  # 크러시드 Crushed
    ("coh_survival_craft", 2350020): (3, ""),  # Start Over
    ("coh_survival_craft", 2935280): (2, ""),  # Zector 7
    ("coh_survival_craft", 3174500): (3, ""),  # 겨울 생존 프로토콜
    ("coh_vehicle_sim", 47920): (2, ""),  # Shift 2 Unleashed
    ("coh_vehicle_sim", 750170): (3, ""),  # Diesel Railcar Simulator
    ("coh_vehicle_sim", 824720): (3, ""),  # Pure Rock Crawling
    ("coh_vehicle_sim", 1273440): (3, ""),  # OverDrift Festival
    ("coh_vehicle_sim", 2083210): (2, ""),  # Super Woden GP 2
    ("coh_vehicle_sim", 2248760): (3, ""),  # 자동차 판매 시뮬레이터 2023
    ("coh_vehicle_sim", 2486820): (2, ""),  # 소닉 레이싱 크로스월드
    ("coh_vehicle_sim", 2961110): (1, ""),  # SUSHI Race
    ("coh_vehicle_sim", 2986370): (2, ""),  # Food Delivery Simulator
    ("longtail_deckbuilder", 333640): (2, ""),  # Caves of Qud
    ("longtail_deckbuilder", 339400): (3, ""),  # Runestone Keeper
    ("longtail_deckbuilder", 722730): (2, ""),  # Cogmind
    ("longtail_deckbuilder", 1203360): (3, ""),  # Core Defense
    ("longtail_deckbuilder", 1314770): (3, ""),  # Three Kingdom: The Journey
    ("longtail_deckbuilder", 1332090): (3, ""),  # Void Tyrant
    ("longtail_deckbuilder", 1803400): (3, ""),  # Beneath Oresa
    ("longtail_deckbuilder", 2289750): (3, ""),  # Super Fantasy Kingdom 슈퍼 판타지 킹덤
    ("longtail_deckbuilder", 2450400): (3, ""),  # Trash of the Titans
    ("longtail_deckbuilder", 2663790): (3, ""),  # TaxingTiles
    ("longtail_deckbuilder", 2832280): (3, ""),  # Blightstone
    ("longtail_deckbuilder", 2954750): (3, ""),  # The Dreamers Foresight
    ("longtail_deckbuilder", 3410180): (3, ""),  # Overlooting
    ("longtail_deckbuilder", 3720630): (3, ""),  # Caemdale
    ("longtail_deckbuilder", 3833300): (3, ""),  # Infinite Cards
    ("longtail_deckbuilder", 3877790): (3, ""),  # 사냥의 밤：소브린 신디케이트
    ("longtail_detective", 34800): (3, ""),  # Chronicles of Mystery: The Scorpio Ritua
    ("longtail_detective", 63610): (3, ""),  # Riven (1997)
    ("longtail_detective", 284390): (3, ""),  # The Last Door - Collector's Edition
    ("longtail_detective", 454250): (3, ""),  # The Eyes of Ara
    ("longtail_detective", 501990): (2, ""),  # Phantasmagoria
    ("longtail_detective", 566190): (2, ""),  # The Search
    ("longtail_detective", 568930): (2, ""),  # The Land of Pain
    ("longtail_detective", 652950): (2, ""),  # Maggie's Apartment
    ("longtail_detective", 1098770): (3, ""),  # The Crimson Diamond
    ("longtail_detective", 1334010): (2, ""),  # Last Room
    ("longtail_detective", 1553120): (3, ""),  # The Inheritance of Crimson Manor
    ("longtail_detective", 1589500): (2, ""),  # Fate of Dynasty
    ("longtail_detective", 1603640): (3, ""),  # The House of Da Vinci 3
    ("longtail_detective", 1861440): (3, ""),  # CLeM
    ("longtail_detective", 2476100): (2, ""),  # Creepy Tale: Some Other Place
    ("longtail_detective", 2797960): (3, ""),  # 은폐된 살인의 진실들 - 하드코어 본격 추리 탐정 게임
    ("longtail_detective", 2852930): (1, ""),  # Ethel
    ("longtail_detective", 3373950): (3, ""),  # 古き薔薇の館 Old Rose Mansion
    ("longtail_metroidvania", 233510): (1, ""),  # Lunnye Devitsy
    ("longtail_metroidvania", 263980): (3, ""),  # Out There Somewhere
    ("longtail_metroidvania", 311010): (3, ""),  # The Way
    ("longtail_metroidvania", 463220): (3, ""),  # Dungeon of Zolthan
    ("longtail_metroidvania", 664830): (2, ""),  # Zombotron
    ("longtail_metroidvania", 789840): (3, ""),  # A Dream of Burning Sand
    ("longtail_metroidvania", 1005450): (3, ""),  # Vision Soft Reset
    ("longtail_metroidvania", 1220150): (3, ""),  # Blue Fire
    ("longtail_metroidvania", 1379870): (2, ""),  # Tribal Hunter
    ("longtail_metroidvania", 2085800): (3, ""),  # Kalinur
    ("longtail_metroidvania", 2951840): (2, ""),  # Arisen Force: HeroTest
    ("longtail_puzzle_platformer", 385250): (1, ""),  # Paint it Back
    ("longtail_puzzle_platformer", 444770): (3, ""),  # Mimpi Dreams
    ("longtail_puzzle_platformer", 448720): (2, ""),  # Puzzle Box
    ("longtail_puzzle_platformer", 463760): (3, ""),  # The Beggar's Ride
    ("longtail_puzzle_platformer", 646010): (3, ""),  # Golem
    ("longtail_puzzle_platformer", 1206060): (3, ""),  # Happy Game
    ("longtail_puzzle_platformer", 1225580): (3, ""),  # Fe
    ("longtail_puzzle_platformer", 1304610): (3, ""),  # Lab Rat
    ("longtail_puzzle_platformer", 1542060): (3, ""),  # Forgotten Spirits
    ("longtail_puzzle_platformer", 1542810): (2, ""),  # Sunshine Heavy Industries
    ("longtail_puzzle_platformer", 1549550): (2, ""),  # Haven Park
    ("longtail_puzzle_platformer", 1632230): (3, ""),  # Newtonian Inversion
    ("longtail_puzzle_platformer", 1698220): (3, ""),  # Teslagrad 2
    ("longtail_puzzle_platformer", 1717510): (2, ""),  # Time on Frog Island
    ("longtail_puzzle_platformer", 1733230): (3, ""),  # Vortex
    ("longtail_puzzle_platformer", 2067050): (2, ""),  # Squirrel with a Gun
    ("longtail_puzzle_platformer", 2393920): (2, ""),  # Angeline Era
    ("longtail_puzzle_platformer", 2727650): (2, ""),  # WHAT THE CAR?
    ("lowrev_cozy_narrative", 296870): (2, ""),  # Dreaming Sarah
    ("lowrev_cozy_narrative", 438340): (2, ""),  # Drizzlepath: Genie
    ("lowrev_cozy_narrative", 696280): (2, ""),  # SOLE
    ("lowrev_cozy_narrative", 748300): (1, ""),  # Treasure Hunter Simulator
    ("lowrev_cozy_narrative", 880400): (2, ""),  # Lingering Fragrance
    ("lowrev_cozy_narrative", 954080): (3, ""),  # Memorrha
    ("lowrev_cozy_narrative", 978870): (2, ""),  # Lilycle Rainbow Stage!!!
    ("lowrev_cozy_narrative", 1119710): (3, ""),  # looK INside - Chapter 1
    ("lowrev_cozy_narrative", 1227890): (1, ""),  # Summer Memories
    ("lowrev_cozy_narrative", 1387000): (3, ""),  # Island of the Lost
    ("lowrev_cozy_narrative", 1462740): (2, ""),  # Your way
    ("lowrev_cozy_narrative", 1490340): (2, ""),  # We Are OFK
    ("lowrev_cozy_narrative", 1574820): (3, ""),  # Until Then
    ("lowrev_cozy_narrative", 1899620): (2, ""),  # Outdoor Adventures With Marisa Kirisame
    ("lowrev_cozy_narrative", 1970060): (2, ""),  # Tomboy Adventure
    ("lowrev_cozy_narrative", 1996210): (1, ""),  # Sakura Hime 3
    ("lowrev_cozy_narrative", 2495450): (1, ""),  # Winter Memories
    ("lowrev_cozy_narrative", 2642760): (3, ""),  # Back to Hearth
    ("lowrev_cozy_narrative", 2818450): (1, ""),  # Putrika 1st.cut:The Reason She Must Peri
    ("lowrev_cozy_narrative", 3447690): (1, ""),  # iDigging
    ("lowrev_cozy_narrative", 3548580): (3, ""),  # Chill with You : Lo-Fi Story
    ("lowrev_cozy_narrative", 3813140): (2, ""),  # This Winter of Ours
    ("lowrev_deckbuilder", 856990): (3, ""),  # A Long Way Down
    ("lowrev_deckbuilder", 1135810): (3, ""),  # Vault of the Void
    ("lowrev_deckbuilder", 1449070): (3, ""),  # 신주지 서유
    ("lowrev_deckbuilder", 1866630): (3, ""),  # Throne of Bone
    ("lowrev_deckbuilder", 2088160): (3, ""),  # 天外武林 (Traveler of Wuxia)
    ("lowrev_deckbuilder", 2179850): (3, ""),  # Cobalt Core
    ("lowrev_deckbuilder", 2356780): (3, ""),  # Dungeon Clawler
    ("lowrev_deckbuilder", 2877770): (3, ""),  # Into the Restless Ruins
    ("lowrev_detective", 214700): (3, ""),  # Thirty Flights of Loving
    ("lowrev_detective", 384630): (3, ""),  # Aviary Attorney
    ("lowrev_detective", 1057750): (3, ""),  # The Suicide of Rachel Foster
    ("lowrev_detective", 1773180): (1, ""),  # The Matriarch
    ("lowrev_detective", 1787990): (2, ""),  # Hidden Investigation 3: Crime Files
    ("lowrev_detective", 2000090): (3, ""),  # The Isle Tide Hotel 아일 타이드 호텔
    ("lowrev_detective", 2482920): (2, ""),  # 사람 속에 피는 꽃
    ("lowrev_detective", 2852930): (1, ""),  # Ethel
    ("lowrev_detective", 2871140): (3, ""),  # The Real Face of a VTuber
    ("lowrev_detective", 3979550): (3, ""),  # The Game Maker: A Carol Reed Mystery
    ("lowrev_detective", 4198660): (2, ""),  # Demons' Timeline
    ("lowrev_detective", 4333760): (2, ""),  # Detectives United: Vengeance from the Pa
    ("lowrev_detective", 4805120): (3, ""),  # The Disappearing Municipality Worker
    ("lowrev_metroidvania", 200900): (3, ""),  # Cave Story+
    ("lowrev_metroidvania", 236090): (3, ""),  # Dust: An Elysian Tail
    ("lowrev_metroidvania", 296870): (2, ""),  # Dreaming Sarah
    ("lowrev_metroidvania", 454410): (3, ""),  # GIGA WRECKER
    ("lowrev_metroidvania", 463270): (3, ""),  # Ghost 1.0
    ("lowrev_metroidvania", 589510): (3, ""),  # Shovel Knight: Specter of Torment
    ("lowrev_metroidvania", 612390): (3, ""),  # Dandara: Trials of Fear Edition
    ("lowrev_metroidvania", 628550): (2, ""),  # Crypt Stalker
    ("lowrev_metroidvania", 717310): (3, ""),  # Aggelos
    ("lowrev_metroidvania", 838310): (3, ""),  # Bloodstained: Curse of the Moon
    ("lowrev_metroidvania", 851180): (2, ""),  # Blood Harvest 3
    ("lowrev_metroidvania", 1001800): (3, ""),  # KUNAI
    ("lowrev_metroidvania", 1127850): (2, ""),  # Apple Slash
    ("lowrev_metroidvania", 1546710): (3, ""),  # Vernal Edge
    ("lowrev_metroidvania", 1669420): (3, ""),  # Islets
    ("lowrev_metroidvania", 1772830): (3, ""),  # Rusted Moss
    ("lowrev_metroidvania", 1983620): (3, ""),  # Infinitevania
    ("lowrev_metroidvania", 2134320): (0, "IRRELEVANT"),  # ENA: Dream BBQ
    ("lowrev_metroidvania", 2715870): (3, ""),  # Before I Go
    ("lowrev_towerdefense", 18120): (3, ""),  # Unstoppable Gorg
    ("lowrev_towerdefense", 98300): (3, ""),  # Toy Soldiers
    ("lowrev_towerdefense", 214730): (1, ""),  # Space Rangers HD: A War Apart
    ("lowrev_towerdefense", 306020): (3, ""),  # Bloons TD 5
    ("lowrev_towerdefense", 343360): (1, ""),  # Particula
    ("lowrev_towerdefense", 406850): (2, ""),  # Crush Your Enemies
    ("lowrev_towerdefense", 674750): (3, ""),  # Yet Another Zombie Defense HD
    ("lowrev_towerdefense", 1093950): (3, ""),  # Inbound UFO
    ("lowrev_towerdefense", 1422440): (3, ""),  # Cataclismo 카타클리스모
    ("lowrev_towerdefense", 1424860): (3, ""),  # 비시 배시 봇스
    ("lowrev_towerdefense", 1647420): (3, ""),  # GROSS
    ("lowrev_towerdefense", 2296550): (3, ""),  # 강철 요새：시냅스 TD
    ("lowrev_towerdefense", 2474180): (3, ""),  # Save the settlers
    ("lowrev_towerdefense", 3419220): (3, ""),  # Dungeon Warfare 3
    ("lowrev_towerdefense", 4471000): (3, ""),  # Tower Defense Ultimate: Crashfall Protoc
    ("mix2_arcade_action", 7520): (1, ""),  # Two Worlds II HD
    ("mix2_arcade_action", 379720): (2, ""),  # DOOM
    ("mix2_arcade_action", 550340): (2, ""),  # Umineko: Golden Fantasia
    ("mix2_arcade_action", 667530): (2, ""),  # Drunken Wrestlers 2
    ("mix2_arcade_action", 978300): (1, ""),  # Saints Row®: The Third™ Remastered
    ("mix2_arcade_action", 1072100): (2, ""),  # Action Arcade Wrestling
    ("mix2_arcade_action", 1345820): (3, ""),  # Ragnarock
    ("mix2_arcade_action", 1364690): (3, ""),  # First Racer
    ("mix2_arcade_action", 1732650): (2, ""),  # Fighting Robots
    ("mix2_arcade_action", 2131200): (2, ""),  # NeverSynth
    ("mix2_arcade_action", 3050220): (3, ""),  # THE KING OF FIGHTERS XIII GLOBAL MATCH
    ("mix2_arcade_action", 4448950): (1, ""),  # The Turkish Legacy
    ("mix2_builder_sim", 223490): (2, ""),  # Blockscape
    ("mix2_builder_sim", 362620): (2, ""),  # Software Inc.
    ("mix2_builder_sim", 382310): (2, ""),  # Eco
    ("mix2_builder_sim", 454060): (2, ""),  # Blueprint Tycoon
    ("mix2_builder_sim", 591370): (3, ""),  # Production Line : Car factory simulation
    ("mix2_builder_sim", 758870): (2, ""),  # Kynseed
    ("mix2_builder_sim", 803050): (3, ""),  # Per Aspera
    ("mix2_builder_sim", 911430): (3, ""),  # Good Company
    ("mix2_builder_sim", 931270): (3, ""),  # MicroTown
    ("mix2_builder_sim", 977510): (2, ""),  # Time to Morp
    ("mix2_builder_sim", 1104330): (2, ""),  # Founders' Fortune
    ("mix2_builder_sim", 1644500): (3, ""),  # Masterplan Tycoon
    ("mix2_builder_sim", 2414110): (3, ""),  # Builderment
    ("mix2_builder_sim", 2615630): (3, ""),  # Global Farmer
    ("mix2_builder_sim", 3293260): (2, ""),  # Waterpark Simulator
    ("mix2_builder_sim", 3566200): (3, ""),  # Pocket City 2
    ("mix2_builder_sim", 3868320): (1, ""),  # Forage Wizard
    ("mix2_colony_cozy", 210970): (1, ""),  # The Witness
    ("mix2_colony_cozy", 366910): (2, ""),  # The Long Journey Home
    ("mix2_colony_cozy", 704510): (3, ""),  # Mercury Fallen
    ("mix2_colony_cozy", 1148510): (0, "IRRELEVANT"),  # Pretty Angel
    ("mix2_colony_cozy", 1200570): (3, ""),  # AColony
    ("mix2_colony_cozy", 1542810): (3, ""),  # Sunshine Heavy Industries
    ("mix2_colony_cozy", 3269180): (1, ""),  # Piñata Go Boom
    ("mix2_colony_cozy", 3276050): (2, ""),  # SpaceCraft
    ("mix2_colony_cozy", 3804740): (3, ""),  # Cozy Jozy
    ("mix2_coop_horror", 238280): (2, ""),  # Legend of Dungeon
    ("mix2_coop_horror", 258970): (2, ""),  # Gauntlet™ Slayer Edition
    ("mix2_coop_horror", 374320): (1, ""),  # DARK SOULS™ III
    ("mix2_coop_horror", 1245620): (1, ""),  # ELDEN RING
    ("mix2_coop_horror", 1442840): (1, ""),  # DIG - Deep In Galaxies
    ("mix2_coop_horror", 1562420): (3, ""),  # FOREWARNED
    ("mix2_coop_horror", 1571440): (3, ""),  # Lunch Lady
    ("mix2_coop_horror", 1609570): (3, ""),  # Agonize
    ("mix2_coop_horror", 1672970): (2, ""),  # Minecraft Dungeons
    ("mix2_coop_horror", 1708270): (1, ""),  # Running Robot Man 4
    ("mix2_coop_horror", 1715730): (3, ""),  # Boo Men
    ("mix2_coop_horror", 1785120): (3, ""),  # Forsake: Urban horror
    ("mix2_coop_horror", 1791910): (3, ""),  # Silent Rain
    ("mix2_coop_horror", 2189670): (2, ""),  # The Black Pool
    ("mix2_coop_horror", 2208570): (3, ""),  # Dark Hours
    ("mix2_coop_horror", 2541520): (3, ""),  # Deep Dark Space
    ("mix2_coop_horror", 2793380): (2, ""),  # Starground
    ("mix2_coop_horror", 3527290): (3, ""),  # PEAK
    ("mix2_coop_horror", 4400300): (3, ""),  # HAUNTMATES
    ("mix2_crpg_sandbox", 208730): (2, ""),  # Game of Thrones
    ("mix2_crpg_sandbox", 214730): (2, ""),  # Space Rangers HD: A War Apart
    ("mix2_crpg_sandbox", 223430): (1, ""),  # Miner Wars 2081
    ("mix2_crpg_sandbox", 349730): (3, ""),  # Popup Dungeon
    ("mix2_crpg_sandbox", 352220): (3, ""),  # King of Dragon Pass
    ("mix2_crpg_sandbox", 492150): (1, ""),  # RPG World - Action RPG Maker
    ("mix2_crpg_sandbox", 568570): (2, ""),  # Force of Nature
    ("mix2_crpg_sandbox", 696370): (2, ""),  # BROKE PROTOCOL
    ("mix2_crpg_sandbox", 1546770): (2, ""),  # The Vagabond Emperor
    ("mix2_crpg_sandbox", 1635450): (2, ""),  # Longvinter
    ("mix2_crpg_sandbox", 1804470): (2, ""),  # Duel Corp.
    ("mix2_crpg_sandbox", 1812450): (3, ""),  # Bellwright
    ("mix2_crpg_sandbox", 2178070): (2, ""),  # Towers of Aghasba
    ("mix2_crpg_sandbox", 2358040): (2, ""),  # Magisterium: Fantasy Craft
    ("mix2_modern_roguelite", 811320): (2, ""),  # Jupiter Hell
    ("mix2_modern_roguelite", 919370): (3, ""),  # Overdungeon
    ("mix2_modern_roguelite", 1201540): (3, ""),  # HELLCARD
    ("mix2_modern_roguelite", 1637320): (2, ""),  # Dome Keeper 돔 키퍼
    ("mix2_modern_roguelite", 1803400): (3, ""),  # Beneath Oresa
    ("mix2_modern_roguelite", 2100150): (3, ""),  # Shadow of the Depth
    ("mix2_modern_roguelite", 2273430): (3, ""),  # BlazBlue Entropy Effect
    ("mix2_modern_roguelite", 2288470): (3, ""),  # 에테르맨서 (Aethermancer)
    ("mix2_modern_roguelite", 2356780): (3, ""),  # Dungeon Clawler
    ("mix2_modern_roguelite", 2687400): (3, ""),  # GODBREAKERS
    ("mix2_modern_roguelite", 2877540): (3, ""),  # Heretical
    ("mix2_modern_roguelite", 3054210): (2, ""),  # Identifile: Desktop Dungeon
    ("mix2_party_narrative", 322330): (2, ""),  # Don't Starve Together
    ("mix2_party_narrative", 508790): (2, ""),  # 플랫 히어로즈
    ("mix2_party_narrative", 555150): (3, ""),  # The First Tree
    ("mix2_party_narrative", 581270): (3, ""),  # Old Man's Journey
    ("mix2_party_narrative", 935580): (1, ""),  # True Fear: Forsaken Souls Part 2
    ("mix2_party_narrative", 1428470): (2, ""),  # Stowaway
    ("mix2_party_narrative", 2492290): (1, ""),  # 언커버 더 스모킹 건
    ("mix2_party_narrative", 3059070): (2, ""),  # The Headliners
    ("mix2_party_narrative", 3296830): (1, ""),  # CaseCracker3
    ("mix2_party_narrative", 3359320): (1, ""),  # SchoolBoy Runaway
    ("mix2_party_narrative", 4069520): (2, ""),  # Super Battle Golf
    ("mix2_puzzle_survival", 210970): (3, ""),  # The Witness
    ("mix2_puzzle_survival", 359510): (2, ""),  # Tangle Tower
    ("mix2_puzzle_survival", 391720): (2, ""),  # Layers of Fear (2016)
    ("mix2_puzzle_survival", 406970): (2, ""),  # The Uncertain: Last Quiet Day
    ("mix2_puzzle_survival", 566190): (2, ""),  # The Search
    ("mix2_puzzle_survival", 763250): (3, ""),  # The Spectrum Retreat
    ("mix2_puzzle_survival", 990630): (2, ""),  # The Last Campfire
    ("mix2_puzzle_survival", 1141580): (3, ""),  # Taiji
    ("mix2_puzzle_survival", 1257850): (3, ""),  # SOLAS 128
    ("mix2_puzzle_survival", 1343520): (2, ""),  # Survival: Lost Way
    ("mix2_puzzle_survival", 1425130): (2, ""),  # Forgotten Hill First Steps
    ("mix2_puzzle_survival", 1569580): (3, ""),  # Blue Prince
    ("mix2_puzzle_survival", 2179850): (2, ""),  # Cobalt Core
    ("mix2_puzzle_survival", 2631960): (1, ""),  # 14가지 변형 지뢰찾기 2
    ("mix2_puzzle_survival", 3428520): (2, ""),  # Escape If You Can
    ("mix2_puzzle_survival", 3547010): (3, ""),  # Occlude
    ("mix2_puzzle_survival", 3616260): (2, ""),  # MAZEBOUND: Hunt, Gather, Run!
    ("mix2_soulslike_narrative", 264710): (3, ""),  # 서브노티카
    ("mix2_soulslike_narrative", 1442530): (2, ""),  # Outbreak Island
    ("mix2_soulslike_narrative", 1522140): (2, ""),  # Demon Skin
    ("mix2_soulslike_narrative", 1528220): (2, ""),  # Gordian Rooms 2: A curious island
    ("mix2_soulslike_narrative", 1624470): (2, ""),  # Mirror Of Life
    ("mix2_soulslike_narrative", 2179370): (3, ""),  # Retrace the Light
    ("mix2_soulslike_narrative", 2336440): (2, ""),  # 침묵의 땅
    ("mix2_soulslike_narrative", 2438330): (3, ""),  # Pale Coins
    ("mix2_soulslike_narrative", 2774040): (1, ""),  # The Boss Gangster: Criminal Empire
    ("mix2_soulslike_narrative", 3504780): (1, ""),  # 와일드 게이트
    ("mix2_soulslike_narrative", 3785560): (2, ""),  # The Narcotic Operation
    ("mix2_soulslike_narrative", 4609830): (3, ""),  # Allelon
    ("mix2_survival_farm", 211820): (3, ""),  # Starbound
    ("mix2_survival_farm", 280520): (3, ""),  # Crea
    ("mix2_survival_farm", 304950): (3, ""),  # Castaway Paradise - live among the anima
    ("mix2_survival_farm", 307880): (2, ""),  # Savage Lands
    ("mix2_survival_farm", 322330): (3, ""),  # Don't Starve Together
    ("mix2_survival_farm", 823950): (3, ""),  # 리: 레전드 Re:Legend
    ("mix2_survival_farm", 840010): (3, ""),  # Garden Paws
    ("mix2_survival_farm", 895400): (2, ""),  # Deadside
    ("mix2_survival_farm", 1044870): (2, ""),  # Zero-based World-从零开始
    ("mix2_survival_farm", 1592110): (3, ""),  # Spirit of the Island
    ("mix2_survival_farm", 1621690): (3, ""),  # Core Keeper
    ("mix2_survival_farm", 1638340): (1, ""),  # Cat Simulator : Animals on Farm
    ("mix2_survival_farm", 1865010): (3, ""),  # 皓际大冒险
    ("mix2_survival_farm", 2119830): (2, ""),  # MISERY
    ("mix2_survival_farm", 2272900): (3, ""),  # Mirthwood
    ("mix2_survival_farm", 3856280): (3, ""),  # holoVillage: Our Cozy Days
    ("niche_cozy_casual", 219680): (3, ""),  # Proteus
    ("niche_cozy_casual", 296910): (1, ""),  # 8BitBoy™
    ("niche_cozy_casual", 366320): (3, ""),  # Seasons after Fall
    ("niche_cozy_casual", 728870): (2, ""),  # Viaerium
    ("niche_cozy_casual", 935880): (3, ""),  # Glass Masquerade 2: Illusions
    ("niche_cozy_casual", 956140): (2, ""),  # Lifeslide
    ("niche_cozy_casual", 1185700): (3, ""),  # Arrog
    ("niche_cozy_casual", 1346420): (2, ""),  # Transpire
    ("niche_cozy_casual", 1549550): (3, ""),  # Haven Park
    ("niche_cozy_casual", 1550730): (2, ""),  # Ikonei Island: An Earthlock Adventure
    ("niche_cozy_casual", 1787160): (2, ""),  # Cupid Nonogram
    ("niche_cozy_casual", 1979640): (3, ""),  # 7Groves
    ("niche_cozy_casual", 2013650): (2, ""),  # Logicality
    ("niche_cozy_casual", 2207440): (3, ""),  # LOK Digital
    ("niche_puzzle_solo", 26500): (3, ""),  # Cogs
    ("niche_puzzle_solo", 29180): (1, ""),  # Osmos
    ("niche_puzzle_solo", 214340): (1, ""),  # Deponia
    ("niche_puzzle_solo", 249590): (2, ""),  # Teslagrad
    ("niche_puzzle_solo", 348440): (2, ""),  # Quell Memento
    ("niche_puzzle_solo", 461840): (1, ""),  # Zenge
    ("niche_puzzle_solo", 493200): (1, ""),  # RiME
    ("niche_puzzle_solo", 497780): (3, ""),  # Recursed
    ("niche_puzzle_solo", 556240): (2, ""),  # Disoriented
    ("niche_puzzle_solo", 572430): (0, "IRRELEVANT"),  # Party Hard 2
    ("niche_puzzle_solo", 870200): (2, ""),  # Juno: New Origins
    ("niche_puzzle_solo", 923710): (3, ""),  # uFactory
    ("niche_puzzle_solo", 1289990): (3, ""),  # Alan's Automaton Workshop
    ("niche_puzzle_solo", 1644500): (3, ""),  # Masterplan Tycoon
    ("niche_puzzle_solo", 1831530): (1, ""),  # PC Creator - PC Building Simulator
    ("niche_puzzle_solo", 1960480): (2, ""),  # AutoForge
    ("niche_puzzle_solo", 2158190): (3, ""),  # Robot Programmer
    ("niche_puzzle_solo", 2631960): (1, ""),  # 14가지 변형 지뢰찾기 2
    ("niche_puzzle_solo", 2659900): (3, ""),  # Sixty Four
    ("niche_puzzle_solo", 2674590): (3, ""),  # Beltmatic
    ("niche_puzzle_solo", 2802710): (3, ""),  # Quantum Odyssey
    ("niche_puzzle_solo", 2823890): (2, ""),  # Center Station Simulator
    ("niche_puzzle_solo", 2917850): (3, ""),  # Photochemistry
    ("niche_puzzle_solo", 2941660): (2, ""),  # Archean
    ("niche_puzzle_solo", 3175190): (3, ""),  # QuantumPulse 2A
    ("niche_puzzle_solo", 3669570): (2, ""),  # Alchemy Factory
    ("niche_puzzle_solo", 3829620): (1, ""),  # Execute
    ("niche_puzzle_solo", 3868320): (1, ""),  # Forage Wizard
    ("niche_roguelite", 268750): (3, ""),  # Magicite
    ("niche_roguelite", 383230): (3, ""),  # Dungeon Souls
    ("niche_roguelite", 814380): (2, ""),  # Sekiro™: Shadows Die Twice - GOTY Editio
    ("niche_roguelite", 1546790): (1, ""),  # Peace, Death! 2
    ("niche_roguelite", 1710100): (2, ""),  # 이상한 나라 모험기
    ("niche_roguelite", 1969100): (2, ""),  # MIGHT'N MOW'EM: CO-OP SURVIVORS ONLINE
    ("niche_roguelite", 2218750): (3, ""),  # Halls of Torment
    ("niche_roguelite", 2780710): (2, ""),  # Asgard's Fall — Viking Survivors
    ("niche_roguelite", 3126220): (3, ""),  # Make It Count
    ("niche_roguelite", 3375890): (2, ""),  # Ogre Chambers 2222
    ("niche_roguelite", 3734140): (2, ""),  # Super Cabbage Kabumi
    ("niche_sim", 43600): (2, ""),  # Nancy Drew®: Warnings at Waverly Academy
    ("niche_sim", 292330): (3, ""),  # Starship Corporation
    ("niche_sim", 352430): (3, ""),  # Farlight Explorers
    ("niche_sim", 481190): (3, ""),  # Stable Orbit - Build your own space stat
    ("niche_sim", 781480): (3, ""),  # Tech Support: Error Unknown
    ("niche_sim", 975620): (3, ""),  # Project DeepWeb
    ("niche_sim", 1293540): (3, ""),  # Outerverse
    ("niche_sim", 1435670): (3, ""),  # Song of Farca
    ("niche_sim", 2012320): (3, ""),  # Parallel Experiment
    ("niche_sim", 2294450): (3, ""),  # CaseCracker2
    ("niche_sim", 3702930): (1, ""),  # Bum: Revenge
    ("niche_sim", 4054820): (3, ""),  # Murder Mystery Stories
    ("niche_sim", 4194800): (3, ""),  # 스타베스터 - 별의 수확자
    ("niche_soulslike_solo", 238280): (2, ""),  # Legend of Dungeon
    ("niche_soulslike_solo", 374320): (3, ""),  # DARK SOULS™ III
    ("niche_soulslike_solo", 512900): (2, ""),  # Streets of Rogue
    ("niche_soulslike_solo", 584400): (1, ""),  # 소닉 매니아
    ("niche_soulslike_solo", 598700): (3, ""),  # The Vagrant
    ("niche_soulslike_solo", 631980): (3, ""),  # Immortal Planet
    ("niche_soulslike_solo", 718590): (3, ""),  # Dark Devotion
    ("niche_soulslike_solo", 720560): (3, ""),  # Vigil: The Longest Night
    ("niche_soulslike_solo", 946610): (2, ""),  # Pocket Rogues
    ("niche_soulslike_solo", 1110910): (3, ""),  # Mortal Shell
    ("niche_soulslike_solo", 1123770): (3, ""),  # Curse of the Dead Gods
    ("niche_soulslike_solo", 1291790): (1, ""),  # Lakeview Cabin 2
    ("niche_soulslike_solo", 1804470): (2, ""),  # Duel Corp.
    ("niche_soulslike_solo", 1841240): (3, ""),  # FOUNTAINS
    ("niche_soulslike_solo", 1887840): (3, ""),  # Another Crab's Treasure
    ("niche_soulslike_solo", 2006140): (3, ""),  # Withering Rooms
    ("niche_soulslike_solo", 2025320): (3, ""),  # Estencel
    ("niche_soulslike_solo", 2736690): (3, ""),  # Void Sols
    ("niche_soulslike_solo", 3088410): (2, ""),  # 七度荒域：混沌之树
    ("niche_soulslike_solo", 3108510): (2, ""),  # FlyKnight
    ("niche_soulslike_solo", 3123920): (3, ""),  # Coral & The Abyss
    ("niche_tactics", 226620): (2, ""),  # Desktop Dungeons
    ("niche_tactics", 339400): (2, ""),  # Runestone Keeper
    ("niche_tactics", 339570): (3, ""),  # Enemy
    ("niche_tactics", 370020): (3, ""),  # Templar Battleforce
    ("niche_tactics", 596590): (1, ""),  # Linked
    ("niche_tactics", 609490): (2, ""),  # Minit
    ("niche_tactics", 1084020): (2, ""),  # TheoTown
    ("niche_tactics", 1308500): (3, ""),  # Soul Elegy
    ("niche_tactics", 1691190): (3, ""),  # Rogue Waters
    ("niche_tactics", 1895860): (2, ""),  # Monster Girls and the Mysterious Adventu
    ("niche_tactics", 1904860): (3, ""),  # Legends of Kingdom Rush
    ("niche_tactics", 2281820): (2, ""),  # 이볼빙
    ("niche_tactics", 2677310): (3, ""),  # GRIDROAD
    ("niche_tactics", 2730290): (3, ""),  # Journey to Monolith
    ("niche_tactics", 3077400): (3, ""),  # 代号肉鸽：流放之地（WhatRogue：Exile Land）
    ("niche_tactics", 3208560): (2, ""),  # Trainatic
    ("niche_tactics", 3694480): (1, ""),  # A Game About Feeding A Black Hole
}


BLIND_ADAPT = {
    ("coh_classic_multi", 320): (3, ""),  # Half-Life 2: Deathmatch
    ("coh_classic_multi", 22350): (2, ""),  # BRINK
    ("coh_classic_multi", 214730): (1, "IRRELEVANT"),  # Space Rangers HD: A War Apart
    ("coh_classic_multi", 568570): (1, "IRRELEVANT"),  # Force of Nature
    ("coh_classic_multi", 823130): (2, ""),  # Totally Accurate Battlegrounds
    ("coh_classic_multi", 1058020): (2, ""),  # STAR WARS™ Battlefront (Classic, 2004)
    ("coh_classic_multi", 1621070): (2, ""),  # DeadPoly
    ("coh_classic_multi", 1635450): (1, "MODE_MISMATCH"),  # Longvinter
    ("coh_classic_multi", 1686940): (2, ""),  # Bopl Battle
    ("coh_cozy", 253650): (1, "IRRELEVANT"),  # Sparkle 2 Evo
    ("coh_fps", 4920): (2, ""),  # Natural Selection 2
    ("coh_fps", 21970): (1, "MODE_MISMATCH"),  # R.U.S.E.™
    ("coh_fps", 22350): (2, ""),  # BRINK
    ("coh_fps", 35450): (3, ""),  # Red Orchestra 2: Heroes of Stalingrad wi
    ("coh_fps", 291550): (1, "MODE_MISMATCH"),  # Brawlhalla
    ("coh_fps", 302670): (3, ""),  # Call to Arms
    ("coh_fps", 374280): (3, ""),  # Hired Ops
    ("coh_fps", 436520): (2, ""),  # Line of Sight
    ("coh_fps", 489940): (3, ""),  # BATTALION: Legacy
    ("coh_fps", 784080): (2, ""),  # MechWarrior 5: Mercenaries
    ("coh_fps", 866570): (2, ""),  # System Shock 2: 25th Anniversary Remaste
    ("coh_fps", 914260): (2, ""),  # HALF DEAD 2
    ("coh_fps", 961200): (1, "MODE_MISMATCH"),  # Predecessor
    ("coh_fps", 1874880): (3, ""),  # Arma Reforger
    ("coh_fps", 2138720): (1, "IRRELEVANT"),  # REMATCH
    ("coh_fps", 2827230): (3, ""),  # Wild Assault / 兽猎突袭
    ("coh_grand_strategy", 1300700): (3, ""),  # Kingdom Wars 4
    ("coh_strategy", 263860): (2, ""),  # SPACECOM
    ("coh_strategy", 464100): (3, ""),  # Codex of Victory
    ("coh_strategy", 615250): (3, ""),  # M.A.X.: Mechanized Assault & Exploration
    ("coh_vehicle_sim", 1551360): (3, ""),  # Forza Horizon 5
    ("longtail_detective", 1441180): (2, ""),  # Dofamine
    ("longtail_metroidvania", 420770): (2, ""),  # The Legend of Dark Witch
    ("longtail_metroidvania", 1983620): (3, ""),  # Infinitevania
    ("longtail_puzzle_platformer", 590950): (1, "IRRELEVANT"),  # Toaster Jam
    ("longtail_puzzle_platformer", 881040): (2, ""),  # Bug Academy
    ("lowrev_cozy_narrative", 634410): (3, ""),  # A Story Beside
    ("lowrev_cozy_narrative", 1291330): (3, ""),  # Wakamarina Valley, New Zealand
    ("lowrev_cozy_narrative", 2943740): (1, "MODE_MISMATCH"),  # Prologue: Go Wayback!
    ("lowrev_detective", 332610): (1, "IRRELEVANT"),  # Mystik Belle
    ("lowrev_detective", 603700): (3, ""),  # The Low Road
    ("lowrev_detective", 1125480): (1, "MODE_MISMATCH"),  # 初恋日记 - School Years
    ("lowrev_detective", 1178230): (3, ""),  # The Wild Case
    ("lowrev_detective", 1548750): (3, ""),  # Dahlia View
    ("lowrev_detective", 1562910): (3, ""),  # MAKOTO WAKAIDO’s Case Files TRILOGY DELU
    ("lowrev_detective", 1825930): (3, ""),  # 孤岛海妖 The Sea-demon
    ("lowrev_detective", 2013120): (3, ""),  # 1997
    ("lowrev_detective", 2510890): (3, ""),  # 프리콜라주 -IDOLIZED-
    ("lowrev_detective", 3126340): (2, ""),  # Burden Street Station
    ("lowrev_towerdefense", 764510): (2, ""),  # Phantom Signal — Sci-Fi Strategy Game
    ("lowrev_towerdefense", 2103950): (3, ""),  # Kritter: Defend Together
    ("mix2_builder_sim", 1383150): (3, ""),  # Final Factory
    ("mix2_colony_cozy", 521340): (0, "IRRELEVANT"),  # True or False
    ("mix2_coop_horror", 323220): (2, ""),  # Vagante
    ("mix2_coop_horror", 599080): (3, ""),  # The Blackout Club
    ("mix2_coop_horror", 3282300): (2, ""),  # Mistfall Hunter
    ("mix2_puzzle_survival", 3963490): (1, "GENRE_ONLY"),  # LYSA HORA: The Haunted Hill
    ("mix2_soulslike_narrative", 1697700): (2, ""),  # Who's Lila?
    ("mix2_soulslike_narrative", 2025320): (2, ""),  # Estencel
    ("mix2_survival_farm", 345330): (2, ""),  # Eden Rising
    ("niche_puzzle_solo", 454320): (3, ""),  # [the Sequence]
    ("niche_puzzle_solo", 922550): (2, ""),  # keyg: the last prison
    ("niche_puzzle_solo", 1053250): (1, "GENRE_ONLY"),  # TOK HARDCORE
    ("niche_puzzle_solo", 2536720): (3, ""),  # U.V.S. Nirmana
    ("niche_puzzle_solo", 2827680): (2, ""),  # Block Factory
    ("niche_roguelite", 3570060): (2, ""),  # 엠버 가디언 The Ember Guardian
    ("niche_sim", 287220): (2, ""),  # Autocraft
    ("niche_tactics", 595930): (2, ""),  # Steam: Rails to Riches
    ("niche_tactics", 2280060): (3, ""),  # Tenebris: Terra Incognita
}


BLIND_CONSENSUS = {
    ("coh_arpg", 2416880): (0, "IRRELEVANT"),  # Duck Life 9: The Flock
    ("coh_classic_multi", 1866220): (1, "IRRELEVANT"),  # HexoJago
    ("coh_cozy", 1450250): (3, ""),  # Distant Bloom
    ("coh_cozy", 1571990): (2, ""),  # Galaxy Pass Station
    ("coh_cozy", 2343920): (2, ""),  # Trackline Express
    ("coh_grand_strategy", 2659830): (2, ""),  # WarDoom ssp Wargame
    ("coh_grand_strategy", 3174310): (2, ""),  # Battle Match: Samurai Wars
    ("coh_indie_platformer", 1059880): (3, ""),  # Devious Dungeon
    ("coh_indie_platformer", 1141220): (2, ""),  # DemonCrawl
    ("coh_indie_platformer", 1280930): (3, ""),  # 애스트럴 어센트
    ("coh_indie_platformer", 1835240): (3, ""),  # Spiritfall
    ("coh_indie_platformer", 1995870): (2, ""),  # Pixel Survivors : Roguelike
    ("coh_indie_platformer", 4539000): (3, ""),  # SoulSwap
    ("coh_openworld_survival", 2183110): (1, "IRRELEVANT"),  # Paper Needs Inspiration!
    ("longtail_puzzle_platformer", 1715330): (1, "LOW_QUALITY"),  # Infinity Treasures
    ("longtail_puzzle_platformer", 1767700): (3, ""),  # Kid Hallow
    ("longtail_puzzle_platformer", 2070030): (2, ""),  # Super Kenney
    ("lowrev_cozy_narrative", 1054110): (2, ""),  # There The Light
    ("lowrev_metroidvania", 2319920): (3, ""),  # Cantirium: God Slayer
    ("mix2_arcade_action", 1460750): (1, "IRRELEVANT"),  # SUPER DRINK BROS.
    ("mix2_arcade_action", 3463190): (0, "IRRELEVANT"),  # Archaeology - Frozen Pirates
    ("mix2_colony_cozy", 1953860): (3, ""),  # Garden In!
    ("mix2_colony_cozy", 2702260): (2, ""),  # Map Map - 지도에 관한 게임
    ("mix2_colony_cozy", 3469990): (2, ""),  # Mama Station
    ("mix2_colony_cozy", 3633420): (3, ""),  # My Tiny Room
    ("mix2_colony_cozy", 4272830): (3, ""),  # Haven Restored
    ("mix2_coop_horror", 577410): (2, ""),  # The Pit: Infinity
    ("mix2_coop_horror", 1305310): (2, ""),  # Drox Operative 2
    ("mix2_crpg_sandbox", 1162010): (2, ""),  # 奇幻与砍杀2 Fantasy & Blade Ⅱ
    ("mix2_crpg_sandbox", 1556230): (2, ""),  # Gobs of Glory
    ("mix2_party_narrative", 298260): (2, ""),  # Only If
    ("mix2_party_narrative", 391720): (2, ""),  # Layers of Fear (2016)
    ("mix2_party_narrative", 410430): (2, ""),  # Morphine
    ("mix2_party_narrative", 573600): (2, ""),  # Think of the Children
    ("mix2_party_narrative", 1205040): (1, "GENRE_ONLY"),  # Granny: Chapter Two
    ("mix2_party_narrative", 1448000): (1, "GENRE_ONLY"),  # 迷离诡夜 blurred weird night
    ("mix2_party_narrative", 1544540): (2, ""),  # ALONE
    ("mix2_party_narrative", 1916310): (2, ""),  # Remnant Records
    ("mix2_party_narrative", 3587490): (2, ""),  # Terrors to Unveil - Day Off
    ("mix2_party_narrative", 3593780): (2, ""),  # Lumberjacked
    ("mix2_soulslike_narrative", 7520): (2, ""),  # Two Worlds II HD
    ("mix2_soulslike_narrative", 202670): (2, ""),  # Nancy Drew®: Alibi in Ashes
    ("mix2_soulslike_narrative", 220240): (2, ""),  # Far Cry 3
    ("mix2_soulslike_narrative", 328270): (2, ""),  # Leviathan: The Last Day of the Decade
    ("mix2_soulslike_narrative", 520720): (3, ""),  # Dear Esther: Landmark Edition
    ("mix2_soulslike_narrative", 757310): (3, ""),  # Sable
    ("mix2_soulslike_narrative", 824070): (2, ""),  # Objects in Space
    ("mix2_soulslike_narrative", 882560): (1, "IRRELEVANT"),  # Door
    ("mix2_soulslike_narrative", 961010): (3, ""),  # The Hand of Glory
    ("mix2_soulslike_narrative", 986130): (3, ""),  # Shadows of Doubt
    ("mix2_soulslike_narrative", 1175500): (1, "IRRELEVANT"),  # Jack In Town
    ("mix2_soulslike_narrative", 1503180): (2, ""),  # Elementite
    ("mix2_soulslike_narrative", 2215430): (3, ""),  # Ghost of Tsushima 디렉터스 컷
    ("mix2_soulslike_narrative", 2372150): (2, ""),  # Escape from Norwood
    ("mix2_soulslike_narrative", 2375480): (2, ""),  # Space Saga
    ("mix2_soulslike_narrative", 2721530): (2, ""),  # Afterplace
    ("mix2_soulslike_narrative", 3276050): (2, ""),  # SpaceCraft
    ("mix2_soulslike_narrative", 3400000): (2, ""),  # Cubic Odyssey
    ("mix2_soulslike_narrative", 3756080): (3, ""),  # Bilson
    ("niche_sim", 726980): (3, ""),  # Cyber Warrior
    ("niche_sim", 1199830): (3, ""),  # How To Hack In?
    ("niche_sim", 1799220): (3, ""),  # Escape: Malice
    ("niche_soulslike_solo", 1522140): (3, ""),  # Demon Skin
    ("niche_tactics", 1455840): (3, ""),  # Dorfromantik
    ("niche_tactics", 1481630): (3, ""),  # Command Line Pilot
    ("niche_tactics", 1606620): (2, ""),  # Train Tycoon
    ("niche_tactics", 2069630): (1, "LOW_QUALITY"),  # HERO YOUSEIJYO
    ("niche_tactics", 2162350): (3, ""),  # Routemania
    ("niche_tactics", 2583340): (3, ""),  # Hexarium
    ("niche_tactics", 2820100): (3, ""),  # HexoCity
    ("niche_tactics", 2855510): (2, ""),  # Rail Maze 2
    ("niche_tactics", 2963570): (3, ""),  # INDUSTRING
    ("niche_tactics", 3491430): (2, ""),  # Mini Star Quest
    ("niche_tactics", 3767740): (2, ""),  # Outhold
    ("niche_tactics", 3833760): (2, ""),  # You Know The Drill
    ("niche_tactics", 4360900): (3, ""),  # Rails & Rivals
    ("niche_tactics", 4475930): (3, ""),  # Distributrains
    ("niche_tactics", 4684810): (2, ""),  # DICEPATH
}


# k=75 확장으로 새로 노출된 436쌍. 블라인드 판정(설정·순위·전략 비공개, 시드만 보고 판단).
# 3 매우 타당 / 2 타당 / 1 약함(장르만) / 0 부적절
BLIND_K75 = {
    # coh_arpg — DARK SOULS III / Witcher 3 / Skyrim
    ("coh_arpg", 20920): (3, ""),
    ("coh_arpg", 39510): (3, ""),
    ("coh_arpg", 216910): (2, ""),
    ("coh_arpg", 236430): (3, ""),
    ("coh_arpg", 241930): (3, ""),
    ("coh_arpg", 257350): (2, ""),
    ("coh_arpg", 320040): (1, "GENRE_ONLY"),
    ("coh_arpg", 351970): (2, ""),
    ("coh_arpg", 632360): (0, "IRRELEVANT"),
    ("coh_arpg", 1108590): (2, ""),
    ("coh_arpg", 1264880): (2, ""),
    ("coh_arpg", 1265780): (1, "GENRE_ONLY"),
    ("coh_arpg", 1975810): (1, "KEYWORD_MATCH"),
    ("coh_arpg", 2091020): (3, ""),
    ("coh_arpg", 2265990): (1, "MODE_MISMATCH"),
    ("coh_arpg", 2321880): (1, "TOO_NICHE"),
    ("coh_arpg", 2520410): (0, "IRRELEVANT"),
    ("coh_arpg", 2949580): (2, ""),
    ("coh_arpg", 3238670): (1, "GENRE_ONLY"),
    # coh_classic_multi — Garry's Mod / TF2 / L4D2
    ("coh_classic_multi", 232090): (3, ""),
    ("coh_classic_multi", 433350): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 513000): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 1850930): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 1866470): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 1938090): (2, ""),
    ("coh_classic_multi", 2000950): (2, ""),
    ("coh_classic_multi", 2432770): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 2627740): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 2857900): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 3552820): (2, ""),
    # coh_cozy — Stardew / Slime Rancher / Raft
    ("coh_cozy", 420930): (2, ""),
    ("coh_cozy", 962130): (3, ""),
    ("coh_cozy", 1281790): (1, "GENRE_ONLY"),
    ("coh_cozy", 1363350): (3, ""),
    ("coh_cozy", 1549550): (3, ""),
    ("coh_cozy", 1808680): (2, ""),
    ("coh_cozy", 2358040): (2, ""),
    ("coh_cozy", 2378130): (2, ""),
    ("coh_cozy", 2408920): (2, ""),
    ("coh_cozy", 2570210): (2, ""),
    ("coh_cozy", 3907050): (2, ""),
    # coh_fps — CS2 / PUBG / R6 Siege (경쟁 멀티 FPS)
    ("coh_fps", 10): (3, ""),
    ("coh_fps", 20): (2, ""),
    ("coh_fps", 70): (1, "MODE_MISMATCH"),
    ("coh_fps", 440): (3, ""),
    ("coh_fps", 13540): (3, ""),
    ("coh_fps", 15120): (3, ""),
    ("coh_fps", 292730): (2, ""),
    ("coh_fps", 321400): (1, "GENRE_ONLY"),
    ("coh_fps", 1130700): (0, "IRRELEVANT"),
    ("coh_fps", 1517290): (3, ""),
    ("coh_fps", 1938090): (3, ""),
    ("coh_fps", 2505020): (2, ""),
    ("coh_fps", 3595230): (3, ""),
    # coh_grand_strategy — HOI4 / EU4 / Civ V
    ("coh_grand_strategy", 244410): (3, ""),
    ("coh_grand_strategy", 374380): (2, ""),
    ("coh_grand_strategy", 887570): (1, "KEYWORD_MATCH"),
    ("coh_grand_strategy", 1088790): (2, ""),
    ("coh_grand_strategy", 1099410): (3, ""),
    ("coh_grand_strategy", 1268590): (3, ""),
    ("coh_grand_strategy", 2093410): (3, ""),
    ("coh_grand_strategy", 2236080): (2, ""),
    ("coh_grand_strategy", 2511040): (1, "LOW_QUALITY"),
    ("coh_grand_strategy", 2714760): (1, "LOW_QUALITY"),
    ("coh_grand_strategy", 2772750): (3, ""),
    ("coh_grand_strategy", 3450310): (3, ""),
    ("coh_grand_strategy", 4314010): (1, "TOO_NICHE"),
    ("coh_grand_strategy", 4614650): (1, "IRRELEVANT"),
    # coh_indie_platformer — Hollow Knight / Dead Cells / Celeste
    ("coh_indie_platformer", 655380): (1, "GENRE_ONLY"),
    ("coh_indie_platformer", 864540): (2, ""),
    ("coh_indie_platformer", 888630): (1, "GENRE_ONLY"),
    ("coh_indie_platformer", 986040): (1, "GENRE_ONLY"),
    ("coh_indie_platformer", 1323470): (3, ""),
    ("coh_indie_platformer", 1366260): (2, ""),
    ("coh_indie_platformer", 1910740): (1, "TOO_NICHE"),
    ("coh_indie_platformer", 2085310): (2, ""),
    ("coh_indie_platformer", 3021630): (1, "GENRE_ONLY"),
    ("coh_indie_platformer", 3088420): (2, ""),
    ("coh_indie_platformer", 3199390): (3, ""),
    ("coh_indie_platformer", 3720420): (1, "GENRE_ONLY"),
    ("coh_indie_platformer", 3985860): (2, ""),
    # coh_openworld_survival — 서브노티카 / The Long Dark / No Man's Sky
    ("coh_openworld_survival", 758690): (3, ""),
    ("coh_openworld_survival", 977720): (2, ""),
    ("coh_openworld_survival", 1241040): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 1273480): (2, ""),
    ("coh_openworld_survival", 1355780): (2, ""),
    ("coh_openworld_survival", 1772910): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 2126990): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 2496090): (2, ""),
    # coh_strategy — Civ VI / XCOM 2 / Stellaris
    ("coh_strategy", 3910): (3, ""),
    ("coh_strategy", 65980): (3, ""),
    ("coh_strategy", 200510): (3, ""),
    ("coh_strategy", 204880): (3, ""),
    ("coh_strategy", 310470): (2, ""),
    ("coh_strategy", 312370): (2, ""),
    ("coh_strategy", 410970): (2, ""),
    ("coh_strategy", 464880): (3, ""),
    ("coh_strategy", 1357210): (3, ""),
    ("coh_strategy", 1549870): (1, "LOW_QUALITY"),
    ("coh_strategy", 2165300): (1, "LOW_QUALITY"),
    ("coh_strategy", 2326740): (1, "TOO_NICHE"),
    # coh_survival_craft — Project Zomboid / Don't Starve / The Forest
    ("coh_survival_craft", 299740): (2, ""),
    ("coh_survival_craft", 329430): (3, ""),
    ("coh_survival_craft", 360150): (2, ""),
    ("coh_survival_craft", 463860): (2, ""),
    ("coh_survival_craft", 617030): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 663110): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 747200): (2, ""),
    ("coh_survival_craft", 844390): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 967410): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 1139860): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 1273480): (2, ""),
    ("coh_survival_craft", 2496090): (2, ""),
    ("coh_survival_craft", 4134600): (2, ""),
    # coh_vehicle_sim — ETS2 / BeamNG / Assetto Corsa
    ("coh_vehicle_sim", 805550): (3, ""),
    ("coh_vehicle_sim", 1165530): (3, ""),
    ("coh_vehicle_sim", 1190000): (2, ""),
    ("coh_vehicle_sim", 1754150): (2, ""),
    ("coh_vehicle_sim", 1846380): (2, ""),
    ("coh_vehicle_sim", 2434120): (2, ""),
    ("coh_vehicle_sim", 3059520): (3, ""),
    ("coh_vehicle_sim", 3525060): (1, "GENRE_ONLY"),
    # longtail_deckbuilder — Roots of Yggdrasil / Ancient Gods / Flick Shot Rogues
    ("longtail_deckbuilder", 339570): (1, "GENRE_ONLY"),
    ("longtail_deckbuilder", 1269540): (3, ""),
    ("longtail_deckbuilder", 1408590): (2, ""),
    ("longtail_deckbuilder", 1946930): (0, "IRRELEVANT"),
    ("longtail_deckbuilder", 2216660): (2, ""),
    ("longtail_deckbuilder", 3177890): (3, ""),
    ("longtail_deckbuilder", 3290010): (2, ""),
    ("longtail_deckbuilder", 3621050): (3, ""),
    ("longtail_deckbuilder", 4856250): (3, ""),
    # longtail_detective — The Raven Remastered / The Frostrune / Mad Experiments
    ("longtail_detective", 11150): (3, ""),
    ("longtail_detective", 31850): (3, ""),
    ("longtail_detective", 208110): (3, ""),
    ("longtail_detective", 215160): (2, ""),
    ("longtail_detective", 288160): (3, ""),
    ("longtail_detective", 302390): (2, ""),
    ("longtail_detective", 370910): (3, ""),
    ("longtail_detective", 399890): (2, ""),
    ("longtail_detective", 615350): (2, ""),
    ("longtail_detective", 615770): (3, ""),
    ("longtail_detective", 1374290): (3, ""),
    ("longtail_detective", 1410640): (3, ""),
    ("longtail_detective", 1573720): (3, ""),
    ("longtail_detective", 3115860): (3, ""),
    ("longtail_detective", 3612480): (3, ""),
    # longtail_metroidvania — DOOMBLADE / Transiruby / Blast Brigade
    ("longtail_metroidvania", 306440): (2, ""),
    ("longtail_metroidvania", 332200): (3, ""),
    ("longtail_metroidvania", 888990): (2, ""),
    ("longtail_metroidvania", 1072110): (2, ""),
    ("longtail_metroidvania", 1126710): (2, ""),
    ("longtail_metroidvania", 1229390): (3, ""),
    ("longtail_metroidvania", 3114890): (3, ""),
    # longtail_puzzle_platformer — Pid / ChromaGun / Deer & Boy
    ("longtail_puzzle_platformer", 96100): (3, ""),
    ("longtail_puzzle_platformer", 204180): (2, ""),
    ("longtail_puzzle_platformer", 1494110): (3, ""),
    ("longtail_puzzle_platformer", 1731340): (2, ""),
    ("longtail_puzzle_platformer", 1983650): (2, ""),
    ("longtail_puzzle_platformer", 4565980): (2, ""),
    # lowrev_cozy_narrative — Dordogne / A YEAR OF SPRINGS / Botany Manor
    ("lowrev_cozy_narrative", 581370): (2, ""),
    ("lowrev_cozy_narrative", 1073900): (2, ""),
    ("lowrev_cozy_narrative", 1126600): (3, ""),
    ("lowrev_cozy_narrative", 2058970): (3, ""),
    ("lowrev_cozy_narrative", 2066230): (2, ""),
    ("lowrev_cozy_narrative", 3107740): (3, ""),
    ("lowrev_cozy_narrative", 3841730): (2, ""),
    # lowrev_deckbuilder — Deck of Ashes / Dicefolk / Cubic Cosmos
    ("lowrev_deckbuilder", 681730): (3, ""),
    ("lowrev_deckbuilder", 1134610): (3, ""),
    ("lowrev_deckbuilder", 1148650): (2, ""),
    ("lowrev_deckbuilder", 1162480): (3, ""),
    ("lowrev_deckbuilder", 1397130): (3, ""),
    ("lowrev_deckbuilder", 1600910): (3, ""),
    ("lowrev_deckbuilder", 1727590): (3, ""),
    ("lowrev_deckbuilder", 2223420): (3, ""),
    ("lowrev_deckbuilder", 2416920): (3, ""),
    ("lowrev_deckbuilder", 2423290): (3, ""),
    ("lowrev_deckbuilder", 2868840): (3, ""),
    ("lowrev_deckbuilder", 3112170): (3, ""),
    ("lowrev_deckbuilder", 3412270): (3, ""),
    ("lowrev_deckbuilder", 3833300): (3, ""),
    ("lowrev_deckbuilder", 4090730): (3, ""),
    # lowrev_detective — Still Life / Jenny LeClue / Telling Lies
    ("lowrev_detective", 31800): (3, ""),
    ("lowrev_detective", 80350): (3, ""),
    ("lowrev_detective", 972850): (2, ""),
    ("lowrev_detective", 1775170): (2, ""),
    ("lowrev_detective", 1939640): (3, ""),
    ("lowrev_detective", 2332050): (0, "IRRELEVANT"),
    ("lowrev_detective", 2458840): (1, "KEYWORD_MATCH"),
    ("lowrev_detective", 2678970): (3, ""),
    ("lowrev_detective", 2695260): (2, ""),
    ("lowrev_detective", 2917100): (2, ""),
    ("lowrev_detective", 3213430): (3, ""),
    ("lowrev_detective", 3244670): (3, ""),
    # lowrev_metroidvania — Vomitoreum / Tales of Kenzera / Shantae: Half-Genie Hero
    ("lowrev_metroidvania", 246680): (2, ""),
    ("lowrev_metroidvania", 277890): (3, ""),
    ("lowrev_metroidvania", 525380): (3, ""),
    ("lowrev_metroidvania", 576050): (2, ""),
    ("lowrev_metroidvania", 764300): (1, "SAME_GAME_DIFFERENT_SKU"),
    ("lowrev_metroidvania", 1125110): (2, ""),
    ("lowrev_metroidvania", 1368410): (3, ""),
    ("lowrev_metroidvania", 1724300): (3, ""),
    ("lowrev_metroidvania", 1969720): (3, ""),
    ("lowrev_metroidvania", 2463220): (3, ""),
    ("lowrev_metroidvania", 2524340): (3, ""),
    ("lowrev_metroidvania", 3443080): (3, ""),
    ("lowrev_metroidvania", 3870310): (3, ""),
    # lowrev_towerdefense — Iron Brigade / Ancient Planet TD / Creeper World IXE
    ("lowrev_towerdefense", 280220): (3, ""),
    ("lowrev_towerdefense", 391250): (3, ""),
    ("lowrev_towerdefense", 422920): (3, ""),
    ("lowrev_towerdefense", 744650): (2, ""),
    ("lowrev_towerdefense", 1510070): (3, ""),
    ("lowrev_towerdefense", 1674160): (2, ""),
    ("lowrev_towerdefense", 2687370): (2, ""),
    ("lowrev_towerdefense", 2767040): (2, ""),
    ("lowrev_towerdefense", 2976230): (1, "GENRE_ONLY"),
    ("lowrev_towerdefense", 3165500): (2, ""),
    # mix2_arcade_action — Forza Horizon 5 / Hi-Fi RUSH / Street Fighter 6
    ("mix2_arcade_action", 376300): (3, ""),
    ("mix2_arcade_action", 454180): (1, "LOW_QUALITY"),
    ("mix2_arcade_action", 571720): (2, ""),
    ("mix2_arcade_action", 600720): (3, ""),
    ("mix2_arcade_action", 1051960): (2, ""),
    ("mix2_arcade_action", 1058020): (1, "GENRE_ONLY"),
    ("mix2_arcade_action", 1456200): (2, ""),
    ("mix2_arcade_action", 1846380): (3, ""),
    ("mix2_arcade_action", 2556990): (2, ""),
    ("mix2_arcade_action", 2910140): (1, "KEYWORD_MATCH"),
    # mix2_builder_sim — Cities: Skylines / Factorio / Farming Simulator 22
    ("mix2_builder_sim", 673950): (3, ""),
    ("mix2_builder_sim", 787860): (3, ""),
    ("mix2_builder_sim", 1084020): (3, ""),
    ("mix2_builder_sim", 1504110): (2, ""),
    ("mix2_builder_sim", 1823950): (3, ""),
    ("mix2_builder_sim", 2004440): (3, ""),
    ("mix2_builder_sim", 2207490): (2, ""),
    ("mix2_builder_sim", 2300320): (3, ""),
    ("mix2_builder_sim", 2379810): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 2878420): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 3312130): (3, ""),
    # mix2_colony_cozy — RimWorld / Oxygen Not Included / Unpacking
    ("mix2_colony_cozy", 1291340): (3, ""),
    ("mix2_colony_cozy", 1485630): (3, ""),
    ("mix2_colony_cozy", 1795570): (2, ""),
    ("mix2_colony_cozy", 2208840): (3, ""),
    ("mix2_colony_cozy", 2217540): (1, "TOO_NICHE"),
    ("mix2_colony_cozy", 2318800): (3, ""),
    ("mix2_colony_cozy", 2363460): (1, "LOW_QUALITY"),
    ("mix2_colony_cozy", 2434600): (3, ""),
    ("mix2_colony_cozy", 2707950): (3, ""),
    ("mix2_colony_cozy", 2767480): (3, ""),
    ("mix2_colony_cozy", 3179050): (2, ""),
    ("mix2_colony_cozy", 3396290): (2, ""),
    ("mix2_colony_cozy", 3986710): (3, ""),
    ("mix2_colony_cozy", 4927290): (2, ""),
    # mix2_coop_horror — Lethal Company / Phasmophobia / Deep Rock Galactic
    ("mix2_coop_horror", 209330): (0, "IRRELEVANT"),
    ("mix2_coop_horror", 241760): (1, "GENRE_ONLY"),
    ("mix2_coop_horror", 293860): (3, ""),
    ("mix2_coop_horror", 1139870): (1, "GENRE_ONLY"),
    ("mix2_coop_horror", 1683860): (2, ""),
    ("mix2_coop_horror", 1764660): (3, ""),
    ("mix2_coop_horror", 2480930): (2, ""),
    ("mix2_coop_horror", 2635640): (3, ""),
    ("mix2_coop_horror", 3383720): (3, ""),
    ("mix2_coop_horror", 4436880): (3, ""),
    ("mix2_coop_horror", 4896710): (3, ""),
    # mix2_crpg_sandbox — Baldur's Gate 3 / Bannerlord / Kenshi
    ("mix2_crpg_sandbox", 20920): (2, ""),
    ("mix2_crpg_sandbox", 230230): (3, ""),
    ("mix2_crpg_sandbox", 380750): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 587520): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 1656930): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 1750740): (2, ""),
    ("mix2_crpg_sandbox", 1815530): (0, "IRRELEVANT"),
    ("mix2_crpg_sandbox", 1845910): (3, ""),
    ("mix2_crpg_sandbox", 1975810): (1, "KEYWORD_MATCH"),
    ("mix2_crpg_sandbox", 2776640): (3, ""),
    ("mix2_crpg_sandbox", 3209930): (2, ""),
    # mix2_modern_roguelite — Balatro / Hades / Vampire Survivors
    ("mix2_modern_roguelite", 241760): (2, ""),
    ("mix2_modern_roguelite", 340520): (2, ""),
    ("mix2_modern_roguelite", 585710): (3, ""),
    ("mix2_modern_roguelite", 610960): (2, ""),
    ("mix2_modern_roguelite", 919220): (2, ""),
    ("mix2_modern_roguelite", 1029210): (3, ""),
    ("mix2_modern_roguelite", 1038370): (3, ""),
    ("mix2_modern_roguelite", 1679220): (2, ""),
    ("mix2_modern_roguelite", 1839820): (2, ""),
    ("mix2_modern_roguelite", 2868840): (3, ""),
    ("mix2_modern_roguelite", 3079160): (2, ""),
    ("mix2_modern_roguelite", 3985950): (3, ""),
    # mix2_party_narrative — It Takes Two / Overcooked! 2 / What Remains of Edith Finch
    ("mix2_party_narrative", 61730): (2, ""),
    ("mix2_party_narrative", 257750): (2, ""),
    ("mix2_party_narrative", 271570): (2, ""),
    ("mix2_party_narrative", 620650): (1, "GENRE_ONLY"),
    ("mix2_party_narrative", 663670): (2, ""),
    ("mix2_party_narrative", 893690): (2, ""),
    ("mix2_party_narrative", 905340): (3, ""),
    ("mix2_party_narrative", 1146940): (1, "LOW_QUALITY"),
    ("mix2_party_narrative", 1198860): (1, "LOW_QUALITY"),
    ("mix2_party_narrative", 1607680): (3, ""),
    ("mix2_party_narrative", 1733480): (3, ""),
    ("mix2_party_narrative", 2153350): (3, ""),
    ("mix2_party_narrative", 2477770): (2, ""),
    ("mix2_party_narrative", 2548650): (2, ""),
    ("mix2_party_narrative", 2644470): (3, ""),
    ("mix2_party_narrative", 2808570): (1, "LOW_QUALITY"),
    ("mix2_party_narrative", 2947650): (2, ""),
    ("mix2_party_narrative", 2960770): (2, ""),
    ("mix2_party_narrative", 3030360): (2, ""),
    ("mix2_party_narrative", 3044620): (2, ""),
    ("mix2_party_narrative", 3052150): (1, "SAME_GAME_DIFFERENT_SKU"),
    ("mix2_party_narrative", 3404170): (2, ""),
    ("mix2_party_narrative", 3572120): (2, ""),
    ("mix2_party_narrative", 3697560): (2, ""),
    ("mix2_party_narrative", 3955660): (2, ""),
    ("mix2_party_narrative", 4373040): (2, ""),
    ("mix2_party_narrative", 4638460): (2, ""),
    # mix2_puzzle_survival — The Talos Principle / Rust / Inscryption
    ("mix2_puzzle_survival", 311240): (3, ""),
    ("mix2_puzzle_survival", 513710): (3, ""),
    ("mix2_puzzle_survival", 526160): (2, ""),
    ("mix2_puzzle_survival", 897730): (2, ""),
    ("mix2_puzzle_survival", 1483780): (2, ""),
    ("mix2_puzzle_survival", 1641960): (3, ""),
    ("mix2_puzzle_survival", 1769830): (2, ""),
    ("mix2_puzzle_survival", 2789770): (3, ""),
    ("mix2_puzzle_survival", 3686540): (2, ""),
    ("mix2_puzzle_survival", 3922100): (2, ""),
    ("mix2_puzzle_survival", 4199750): (2, ""),
    # mix2_soulslike_narrative — ELDEN RING / Disco Elysium / Outer Wilds
    ("mix2_soulslike_narrative", 50): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 204180): (0, "IRRELEVANT"),
    ("mix2_soulslike_narrative", 221260): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 274520): (3, ""),
    ("mix2_soulslike_narrative", 279070): (2, ""),
    ("mix2_soulslike_narrative", 314230): (2, ""),
    ("mix2_soulslike_narrative", 331870): (3, ""),
    ("mix2_soulslike_narrative", 359320): (2, ""),
    ("mix2_soulslike_narrative", 554810): (2, ""),
    ("mix2_soulslike_narrative", 632360): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 664830): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 712190): (2, ""),
    ("mix2_soulslike_narrative", 788270): (2, ""),
    ("mix2_soulslike_narrative", 1111930): (2, ""),
    ("mix2_soulslike_narrative", 1141580): (3, ""),
    ("mix2_soulslike_narrative", 1173220): (3, ""),
    ("mix2_soulslike_narrative", 1264880): (2, ""),
    ("mix2_soulslike_narrative", 1316230): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 1574240): (3, ""),
    ("mix2_soulslike_narrative", 1807810): (2, ""),
    ("mix2_soulslike_narrative", 1944360): (1, "TOO_NICHE"),
    ("mix2_soulslike_narrative", 2054970): (3, ""),
    ("mix2_soulslike_narrative", 2094750): (3, ""),
    ("mix2_soulslike_narrative", 2362060): (3, ""),
    ("mix2_soulslike_narrative", 2842040): (2, ""),
    ("mix2_soulslike_narrative", 3121470): (3, ""),
    ("mix2_soulslike_narrative", 3321460): (3, ""),
    ("mix2_soulslike_narrative", 3646460): (2, ""),
    ("mix2_soulslike_narrative", 3681010): (3, ""),
    # mix2_survival_farm — Valheim / Grounded / My Time at Portia
    ("mix2_survival_farm", 344760): (2, ""),
    ("mix2_survival_farm", 715400): (3, ""),
    ("mix2_survival_farm", 1119730): (3, ""),
    ("mix2_survival_farm", 1329510): (3, ""),
    ("mix2_survival_farm", 1363900): (2, ""),
    ("mix2_survival_farm", 1370040): (2, ""),
    ("mix2_survival_farm", 1599330): (3, ""),
    ("mix2_survival_farm", 1631470): (3, ""),
    ("mix2_survival_farm", 1931180): (3, ""),
    ("mix2_survival_farm", 2115760): (2, ""),
    ("mix2_survival_farm", 2358040): (2, ""),
    ("mix2_survival_farm", 2418520): (3, ""),
    ("mix2_survival_farm", 2644050): (2, ""),
    ("mix2_survival_farm", 2681030): (3, ""),
    # niche_cozy_casual — A Short Hike / ISLANDERS / Gorogoa
    ("niche_cozy_casual", 566160): (1, "GENRE_ONLY"),
    ("niche_cozy_casual", 693030): (2, ""),
    ("niche_cozy_casual", 768460): (3, ""),
    ("niche_cozy_casual", 968640): (1, "MODE_MISMATCH"),
    ("niche_cozy_casual", 1202480): (2, ""),
    ("niche_cozy_casual", 1487360): (2, ""),
    ("niche_cozy_casual", 1499940): (3, ""),
    ("niche_cozy_casual", 1580560): (2, ""),
    ("niche_cozy_casual", 1632640): (1, "GENRE_ONLY"),
    ("niche_cozy_casual", 2964540): (3, ""),
    # niche_puzzle_solo — Opus Magnum (단일 시드)
    ("niche_puzzle_solo", 271900): (2, ""),
    ("niche_puzzle_solo", 367450): (3, ""),
    ("niche_puzzle_solo", 504210): (3, ""),
    ("niche_puzzle_solo", 827940): (3, ""),
    ("niche_puzzle_solo", 1098840): (1, "IRRELEVANT"),  # 게임이 아니라 설정집(책)
    ("niche_puzzle_solo", 1168880): (3, ""),
    ("niche_puzzle_solo", 1577860): (2, ""),
    ("niche_puzzle_solo", 1698690): (2, ""),
    ("niche_puzzle_solo", 1791580): (3, ""),
    ("niche_puzzle_solo", 1888130): (3, ""),
    ("niche_puzzle_solo", 2113210): (1, "TOO_NICHE"),
    ("niche_puzzle_solo", 2340280): (2, ""),
    ("niche_puzzle_solo", 4105760): (3, ""),
    ("niche_puzzle_solo", 4792210): (3, ""),
    # niche_roguelite — 10개 시드(Death's Door / Nuclear Throne / Spelunky 2 …)
    ("niche_roguelite", 375480): (2, ""),
    ("niche_roguelite", 405290): (2, ""),
    ("niche_roguelite", 434650): (3, ""),
    ("niche_roguelite", 610960): (2, ""),
    ("niche_roguelite", 1206600): (1, "GENRE_ONLY"),
    ("niche_roguelite", 1522140): (1, "GENRE_ONLY"),
    ("niche_roguelite", 1786830): (3, ""),
    ("niche_roguelite", 1792700): (1, "LOW_QUALITY"),
    ("niche_roguelite", 1793330): (2, ""),
    ("niche_roguelite", 1902710): (2, ""),
    ("niche_roguelite", 2025870): (3, ""),
    ("niche_roguelite", 2220870): (2, ""),
    ("niche_roguelite", 2286600): (2, ""),
    ("niche_roguelite", 2321350): (3, ""),
    ("niche_roguelite", 2881650): (1, "GENRE_ONLY"),
    ("niche_roguelite", 3021630): (3, ""),
    # niche_sim — Hardspace / Hacknet / Golden Idol / 방 탈출 시뮬레이터 / shapez 2
    ("niche_sim", 890520): (2, ""),
    ("niche_sim", 1324270): (2, ""),
    ("niche_sim", 1771980): (3, ""),
    ("niche_sim", 1866300): (3, ""),
    ("niche_sim", 2069040): (3, ""),
    ("niche_sim", 2345480): (2, ""),
    ("niche_sim", 2506480): (2, ""),
    ("niche_sim", 3324570): (2, ""),
    # niche_soulslike_solo — Salt and Sanctuary (단일 시드)
    ("niche_soulslike_solo", 239350): (1, "GENRE_ONLY"),
    ("niche_soulslike_solo", 335300): (3, ""),
    ("niche_soulslike_solo", 340520): (1, "GENRE_ONLY"),
    ("niche_soulslike_solo", 568670): (2, ""),
    ("niche_soulslike_solo", 1054700): (2, ""),
    ("niche_soulslike_solo", 1154090): (3, ""),
    ("niche_soulslike_solo", 1484140): (2, ""),
    ("niche_soulslike_solo", 1836030): (3, ""),
    ("niche_soulslike_solo", 1952670): (2, ""),
    ("niche_soulslike_solo", 2139840): (3, ""),
    ("niche_soulslike_solo", 2170410): (2, ""),
    ("niche_soulslike_solo", 2265990): (1, "MODE_MISMATCH"),
    ("niche_soulslike_solo", 2527750): (3, ""),
    ("niche_soulslike_solo", 3308200): (3, ""),
    # niche_tactics — Into the Breach / Monster Train / Mini Metro
    ("niche_tactics", 353640): (3, ""),
    ("niche_tactics", 636320): (3, ""),
    ("niche_tactics", 657090): (2, ""),
    ("niche_tactics", 2068460): (3, ""),
    ("niche_tactics", 2190400): (3, ""),
    ("niche_tactics", 2244470): (3, ""),
    ("niche_tactics", 2451820): (3, ""),
    ("niche_tactics", 3127770): (3, ""),  # 메카 전술 + 덱빌딩 = Into the Breach + Monster Train
}

# 인기시드 조건부 리뷰 하한 300 설정에서 새로 노출된 213쌍. 같은 블라인드 기준.
BLIND_CF300 = {
    ("coh_arpg", 39160): (2, ""),
    ("coh_arpg", 283640): (3, ""),
    ("coh_arpg", 552500): (2, ""),
    ("coh_arpg", 858820): (2, ""),
    ("coh_arpg", 899770): (3, ""),
    ("coh_arpg", 1203620): (3, ""),
    ("coh_arpg", 1206410): (2, ""),
    ("coh_arpg", 1222690): (3, ""),
    ("coh_arpg", 1501750): (3, ""),
    ("coh_arpg", 3681010): (3, ""),
    ("coh_classic_multi", 49520): (3, ""),
    ("coh_classic_multi", 223470): (2, ""),
    ("coh_classic_multi", 304950): (0, "IRRELEVANT"),
    ("coh_classic_multi", 394690): (3, ""),
    ("coh_classic_multi", 914290): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 1324780): (2, ""),
    ("coh_classic_multi", 1428470): (2, ""),
    ("coh_classic_multi", 1843730): (2, ""),
    ("coh_cozy", 290300): (1, "GENRE_ONLY"),
    ("coh_cozy", 823950): (3, ""),
    ("coh_cozy", 855740): (1, "GENRE_ONLY"),
    ("coh_cozy", 897730): (3, ""),
    ("coh_cozy", 1052990): (3, ""),
    ("coh_cozy", 1124660): (0, "IRRELEVANT"),
    ("coh_cozy", 1550730): (3, ""),
    ("coh_cozy", 2555430): (2, ""),
    ("coh_cozy", 2993780): (3, ""),
    ("coh_cozy", 3280350): (1, "GENRE_ONLY"),
    ("coh_fps", 414740): (2, ""),
    ("coh_fps", 439700): (1, "SAME_GAME_DIFFERENT_SKU"),
    ("coh_fps", 578310): (1, "LOW_QUALITY"),
    ("coh_fps", 1237950): (3, ""),
    ("coh_grand_strategy", 4700): (3, ""),
    ("coh_grand_strategy", 17710): (1, "MODE_MISMATCH"),
    ("coh_grand_strategy", 22130): (3, ""),
    ("coh_grand_strategy", 25800): (3, ""),
    ("coh_grand_strategy", 279140): (3, ""),
    ("coh_grand_strategy", 294100): (1, "GENRE_ONLY"),
    ("coh_grand_strategy", 319050): (1, "GENRE_ONLY"),
    ("coh_grand_strategy", 346810): (2, ""),
    ("coh_grand_strategy", 736820): (3, ""),
    ("coh_grand_strategy", 905970): (1, "GENRE_ONLY"),
    ("coh_grand_strategy", 1148490): (2, ""),
    ("coh_grand_strategy", 1295660): (3, ""),
    ("coh_grand_strategy", 2012190): (3, ""),
    ("coh_indie_platformer", 215510): (2, ""),
    ("coh_indie_platformer", 277870): (1, "GENRE_ONLY"),
    ("coh_indie_platformer", 444720): (3, ""),
    ("coh_indie_platformer", 630720): (2, ""),
    ("coh_indie_platformer", 1061090): (2, ""),
    ("coh_indie_platformer", 1144910): (3, ""),
    ("coh_indie_platformer", 1250440): (3, ""),
    ("coh_indie_platformer", 1374970): (3, ""),
    ("coh_indie_platformer", 1436590): (3, ""),
    ("coh_indie_platformer", 1809540): (3, ""),
    ("coh_indie_platformer", 2181930): (1, "LOW_QUALITY"),
    ("coh_indie_platformer", 2229940): (2, ""),
    ("coh_indie_platformer", 2564520): (2, ""),
    ("coh_indie_platformer", 2989180): (3, ""),
    ("coh_indie_platformer", 2997230): (3, ""),
    ("coh_indie_platformer", 3242750): (2, ""),
    ("coh_openworld_survival", 307880): (2, ""),
    ("coh_openworld_survival", 307940): (2, ""),
    ("coh_openworld_survival", 582270): (1, "GENRE_ONLY"),
    ("coh_openworld_survival", 609150): (1, "IRRELEVANT"),
    ("coh_openworld_survival", 671510): (2, ""),
    ("coh_openworld_survival", 711980): (2, ""),
    ("coh_openworld_survival", 751780): (2, ""),
    ("coh_openworld_survival", 767490): (2, ""),
    ("coh_openworld_survival", 815370): (3, ""),
    ("coh_openworld_survival", 1135260): (2, ""),
    ("coh_openworld_survival", 1253220): (2, ""),
    ("coh_openworld_survival", 1392060): (3, ""),
    ("coh_openworld_survival", 1620730): (2, ""),
    ("coh_openworld_survival", 1645820): (2, ""),
    ("coh_openworld_survival", 2218970): (2, ""),
    ("coh_openworld_survival", 2561580): (2, ""),
    ("coh_openworld_survival", 3400000): (3, ""),
    ("coh_openworld_survival", 3484300): (2, ""),
    ("coh_strategy", 4920): (1, "MODE_MISMATCH"),
    ("coh_strategy", 34330): (3, ""),
    ("coh_strategy", 208140): (3, ""),
    ("coh_strategy", 290790): (2, ""),
    ("coh_strategy", 603850): (3, ""),
    ("coh_strategy", 894630): (2, ""),
    ("coh_strategy", 1135240): (2, ""),
    ("coh_strategy", 1501690): (2, ""),
    ("coh_strategy", 3337140): (3, ""),
    ("coh_survival_craft", 105600): (3, ""),
    ("coh_survival_craft", 513710): (2, ""),
    ("coh_survival_craft", 526870): (2, ""),
    ("coh_survival_craft", 536270): (3, ""),
    ("coh_survival_craft", 768200): (3, ""),
    ("coh_survival_craft", 1169040): (3, ""),
    ("coh_survival_craft", 1203370): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 1419850): (2, ""),
    ("coh_survival_craft", 1562260): (2, ""),
    ("coh_survival_craft", 2010030): (1, "GENRE_ONLY"),
    ("coh_survival_craft", 2184150): (2, ""),
    ("coh_survival_craft", 2633640): (1, "GENRE_ONLY"),
    ("coh_vehicle_sim", 253770): (2, ""),
    ("coh_vehicle_sim", 258760): (3, ""),
    ("coh_vehicle_sim", 370350): (2, ""),
    ("coh_vehicle_sim", 431600): (3, ""),
    ("coh_vehicle_sim", 461430): (2, ""),
    ("coh_vehicle_sim", 497180): (3, ""),
    ("coh_vehicle_sim", 553520): (2, ""),
    ("coh_vehicle_sim", 873840): (3, ""),
    ("coh_vehicle_sim", 2141690): (2, ""),
    ("coh_vehicle_sim", 2404880): (2, ""),
    ("coh_vehicle_sim", 3278310): (2, ""),
    ("coh_vehicle_sim", 3598130): (2, ""),
    ("mix2_builder_sim", 278970): (1, "LOW_QUALITY"),
    ("mix2_builder_sim", 280790): (2, ""),
    ("mix2_builder_sim", 569480): (3, ""),
    ("mix2_builder_sim", 758990): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 789660): (2, ""),
    ("mix2_builder_sim", 1366540): (3, ""),
    ("mix2_builder_sim", 1726130): (2, ""),
    ("mix2_builder_sim", 1782570): (3, ""),
    ("mix2_builder_sim", 2706020): (3, ""),
    ("mix2_builder_sim", 3242950): (3, ""),
    ("mix2_colony_cozy", 48240): (3, ""),
    ("mix2_colony_cozy", 204880): (1, "GENRE_ONLY"),
    ("mix2_colony_cozy", 244030): (2, ""),
    ("mix2_colony_cozy", 297920): (3, ""),
    ("mix2_colony_cozy", 360510): (2, ""),
    ("mix2_colony_cozy", 387370): (3, ""),
    ("mix2_colony_cozy", 414720): (2, ""),
    ("mix2_colony_cozy", 474020): (3, ""),
    ("mix2_colony_cozy", 758690): (3, ""),
    ("mix2_colony_cozy", 799600): (3, ""),
    ("mix2_colony_cozy", 919260): (3, ""),
    ("mix2_colony_cozy", 1012610): (3, ""),
    ("mix2_colony_cozy", 1013540): (2, ""),
    ("mix2_colony_cozy", 1060230): (3, ""),
    ("mix2_colony_cozy", 1069030): (2, ""),
    ("mix2_colony_cozy", 1133120): (2, ""),
    ("mix2_colony_cozy", 1155880): (3, ""),
    ("mix2_colony_cozy", 1284190): (3, ""),
    ("mix2_colony_cozy", 1315980): (2, ""),
    ("mix2_colony_cozy", 1336180): (3, ""),
    ("mix2_colony_cozy", 1369700): (2, ""),
    ("mix2_colony_cozy", 1516750): (2, ""),
    ("mix2_colony_cozy", 1734390): (3, ""),
    ("mix2_colony_cozy", 2719750): (3, ""),
    ("mix2_coop_horror", 2210): (1, "MODE_MISMATCH"),
    ("mix2_coop_horror", 249050): (1, "GENRE_ONLY"),
    ("mix2_coop_horror", 383230): (0, "IRRELEVANT"),
    ("mix2_coop_horror", 455120): (3, ""),
    ("mix2_coop_horror", 509570): (0, "IRRELEVANT"),
    ("mix2_coop_horror", 509980): (3, ""),
    ("mix2_coop_horror", 655740): (1, "GENRE_ONLY"),
    ("mix2_coop_horror", 692890): (3, ""),
    ("mix2_coop_horror", 1125710): (2, ""),
    ("mix2_coop_horror", 1520380): (3, ""),
    ("mix2_coop_horror", 1547670): (3, ""),
    ("mix2_coop_horror", 1911610): (2, ""),
    ("mix2_coop_horror", 1987080): (3, ""),
    ("mix2_coop_horror", 2827200): (3, ""),
    ("mix2_coop_horror", 2835530): (3, ""),
    ("mix2_coop_horror", 3070520): (3, ""),
    ("mix2_coop_horror", 3722330): (3, ""),
    ("mix2_coop_horror", 3815750): (3, ""),
    ("mix2_coop_horror", 4244510): (3, ""),
    ("mix2_crpg_sandbox", 22100): (3, ""),
    ("mix2_crpg_sandbox", 73210): (2, ""),
    ("mix2_crpg_sandbox", 202710): (0, "IRRELEVANT"),
    ("mix2_crpg_sandbox", 436780): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 527230): (2, ""),
    ("mix2_crpg_sandbox", 538030): (2, ""),
    ("mix2_crpg_sandbox", 612720): (2, ""),
    ("mix2_crpg_sandbox", 670260): (2, ""),
    ("mix2_crpg_sandbox", 794490): (1, "LOW_QUALITY"),
    ("mix2_crpg_sandbox", 909660): (3, ""),
    ("mix2_crpg_sandbox", 919360): (3, ""),
    ("mix2_crpg_sandbox", 1316230): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 1768780): (3, ""),
    ("mix2_crpg_sandbox", 2344320): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 3767850): (1, "GENRE_ONLY"),
    ("mix2_modern_roguelite", 606150): (3, ""),
    ("mix2_modern_roguelite", 619820): (3, ""),
    ("mix2_modern_roguelite", 920680): (3, ""),
    ("mix2_modern_roguelite", 1040420): (3, ""),
    ("mix2_modern_roguelite", 1108370): (3, ""),
    ("mix2_modern_roguelite", 1869780): (3, ""),
    ("mix2_modern_roguelite", 1930600): (3, ""),
    ("mix2_modern_roguelite", 2305500): (2, ""),
    ("mix2_modern_roguelite", 2717880): (3, ""),
    ("mix2_modern_roguelite", 3021630): (2, ""),
    ("mix2_modern_roguelite", 3288300): (3, ""),
    ("mix2_puzzle_survival", 205650): (2, ""),
    ("mix2_puzzle_survival", 323720): (2, ""),
    ("mix2_puzzle_survival", 938560): (2, ""),
    ("mix2_puzzle_survival", 1042490): (3, ""),
    ("mix2_puzzle_survival", 1125390): (3, ""),
    ("mix2_puzzle_survival", 1149460): (3, ""),
    ("mix2_puzzle_survival", 1889040): (2, ""),
    ("mix2_puzzle_survival", 2379780): (3, ""),
    ("mix2_puzzle_survival", 2840480): (3, ""),
    ("mix2_puzzle_survival", 3223390): (3, ""),
    ("mix2_soulslike_narrative", 8500): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 12320): (2, ""),
    ("mix2_soulslike_narrative", 311290): (2, ""),
    ("mix2_soulslike_narrative", 385380): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 477740): (3, ""),
    ("mix2_soulslike_narrative", 493540): (2, ""),
    ("mix2_soulslike_narrative", 582270): (2, ""),
    ("mix2_soulslike_narrative", 865680): (3, ""),
    ("mix2_soulslike_narrative", 1222690): (3, ""),
    ("mix2_soulslike_narrative", 1233070): (3, ""),
    ("mix2_soulslike_narrative", 2787320): (2, ""),
    ("mix2_soulslike_narrative", 4570720): (2, ""),
}

# 합의태그 개수 부스트(0.3 / 0.8) 합집합으로 새로 노출된 525쌍. 같은 블라인드 기준.
BLIND_CBOOST = {
    ("coh_arpg", 22320): (3, ""), ("coh_arpg", 39500): (3, ""), ("coh_arpg", 40390): (2, ""),
    ("coh_arpg", 204030): (3, ""), ("coh_arpg", 219780): (3, ""), ("coh_arpg", 233350): (1, "GENRE_ONLY"),
    ("coh_arpg", 249230): (2, ""), ("coh_arpg", 345350): (1, "GENRE_ONLY"), ("coh_arpg", 369440): (2, ""),
    ("coh_arpg", 485510): (3, ""), ("coh_arpg", 574050): (1, "GENRE_ONLY"), ("coh_arpg", 582010): (3, ""),
    ("coh_arpg", 589360): (2, ""), ("coh_arpg", 595520): (2, ""), ("coh_arpg", 637650): (2, ""),
    ("coh_arpg", 649950): (3, ""), ("coh_arpg", 1042780): (2, ""), ("coh_arpg", 1171690): (2, ""),
    ("coh_arpg", 1206600): (1, "GENRE_ONLY"), ("coh_arpg", 1453790): (1, "GENRE_ONLY"),
    ("coh_arpg", 1534840): (2, ""), ("coh_arpg", 1716740): (2, ""), ("coh_arpg", 1758450): (0, "IRRELEVANT"),
    ("coh_arpg", 2195190): (1, "TOO_NICHE"), ("coh_arpg", 2240890): (2, ""), ("coh_arpg", 2246340): (3, ""),
    ("coh_arpg", 2277560): (3, ""), ("coh_arpg", 2284600): (2, ""), ("coh_arpg", 2383650): (2, ""),
    ("coh_arpg", 2556990): (1, "GENRE_ONLY"), ("coh_arpg", 2622380): (3, ""), ("coh_arpg", 3173440): (2, ""),
    ("coh_arpg", 3768380): (2, ""), ("coh_arpg", 4069210): (2, ""),

    ("coh_classic_multi", 730): (3, ""), ("coh_classic_multi", 4920): (3, ""),
    ("coh_classic_multi", 215470): (3, ""), ("coh_classic_multi", 218620): (3, ""),
    ("coh_classic_multi", 241720): (2, ""), ("coh_classic_multi", 261640): (3, ""),
    ("coh_classic_multi", 359550): (2, ""), ("coh_classic_multi", 360940): (3, ""),
    ("coh_classic_multi", 437350): (1, "SAME_GAME_DIFFERENT_SKU"), ("coh_classic_multi", 571740): (2, ""),
    ("coh_classic_multi", 651380): (1, "LOW_QUALITY"), ("coh_classic_multi", 706990): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 838410): (2, ""), ("coh_classic_multi", 1359090): (2, ""),
    ("coh_classic_multi", 1508030): (1, "LOW_QUALITY"), ("coh_classic_multi", 1548850): (2, ""),
    ("coh_classic_multi", 1605640): (1, "MODE_MISMATCH"), ("coh_classic_multi", 1785150): (3, ""),
    ("coh_classic_multi", 2080690): (1, "GENRE_ONLY"), ("coh_classic_multi", 2494370): (1, "LOW_QUALITY"),
    ("coh_classic_multi", 3029450): (2, ""), ("coh_classic_multi", 3864450): (2, ""),
    ("coh_classic_multi", 4127370): (2, ""), ("coh_classic_multi", 4397060): (2, ""),

    ("coh_cozy", 108600): (1, "GENRE_ONLY"), ("coh_cozy", 223490): (1, "LOW_QUALITY"),
    ("coh_cozy", 227860): (1, "GENRE_ONLY"), ("coh_cozy", 280520): (2, ""),
    ("coh_cozy", 352950): (1, "GENRE_ONLY"), ("coh_cozy", 387990): (2, ""),
    ("coh_cozy", 438100): (0, "IRRELEVANT"), ("coh_cozy", 441790): (1, "GENRE_ONLY"),
    ("coh_cozy", 576050): (0, "IRRELEVANT"), ("coh_cozy", 660880): (0, "IRRELEVANT"),
    ("coh_cozy", 670260): (1, "GENRE_ONLY"), ("coh_cozy", 788470): (2, ""),
    ("coh_cozy", 840010): (3, ""), ("coh_cozy", 894940): (3, ""), ("coh_cozy", 1158160): (3, ""),
    ("coh_cozy", 1169040): (2, ""), ("coh_cozy", 1241040): (1, "LOW_QUALITY"),
    ("coh_cozy", 1270010): (1, "LOW_QUALITY"), ("coh_cozy", 1326470): (1, "GENRE_ONLY"),
    ("coh_cozy", 1403650): (3, ""), ("coh_cozy", 1442360): (1, "LOW_QUALITY"),
    ("coh_cozy", 1465550): (3, ""), ("coh_cozy", 1594980): (2, ""), ("coh_cozy", 1617270): (2, ""),
    ("coh_cozy", 1670810): (1, "GENRE_ONLY"), ("coh_cozy", 1702760): (1, "GENRE_ONLY"),
    ("coh_cozy", 1723780): (3, ""), ("coh_cozy", 1733860): (3, ""), ("coh_cozy", 1805110): (3, ""),
    ("coh_cozy", 2392400): (2, ""), ("coh_cozy", 2868100): (1, "LOW_QUALITY"),
    ("coh_cozy", 2941660): (1, "GENRE_ONLY"), ("coh_cozy", 2997840): (3, ""),
    ("coh_cozy", 3968160): (3, ""), ("coh_cozy", 4238620): (2, ""),

    ("coh_fps", 320): (2, ""), ("coh_fps", 202970): (3, ""), ("coh_fps", 550650): (2, ""),
    ("coh_fps", 611300): (2, ""), ("coh_fps", 671860): (3, ""), ("coh_fps", 677620): (3, ""),
    ("coh_fps", 680940): (1, "LOW_QUALITY"), ("coh_fps", 872200): (3, ""),
    ("coh_fps", 915320): (1, "LOW_QUALITY"), ("coh_fps", 1000500): (1, "LOW_QUALITY"),
    ("coh_fps", 1172470): (3, ""), ("coh_fps", 1216600): (1, "LOW_QUALITY"),
    ("coh_fps", 1218470): (2, ""), ("coh_fps", 1341160): (2, ""), ("coh_fps", 1359090): (3, ""),
    ("coh_fps", 1539860): (2, ""), ("coh_fps", 1915040): (0, "IRRELEVANT"),
    ("coh_fps", 1985810): (3, ""), ("coh_fps", 2175780): (1, "GENRE_ONLY"),
    ("coh_fps", 2257910): (2, ""), ("coh_fps", 2357570): (3, ""), ("coh_fps", 2440220): (1, "LOW_QUALITY"),
    ("coh_fps", 2489880): (1, "LOW_QUALITY"), ("coh_fps", 2492600): (1, "MODE_MISMATCH"),
    ("coh_fps", 2645030): (1, "LOW_QUALITY"), ("coh_fps", 2790190): (1, "LOW_QUALITY"),
    ("coh_fps", 2807960): (3, ""), ("coh_fps", 2870600): (1, "LOW_QUALITY"),
    ("coh_fps", 2883690): (2, ""), ("coh_fps", 2890270): (1, "LOW_QUALITY"),

    ("coh_grand_strategy", 3910): (3, ""), ("coh_grand_strategy", 34030): (3, ""),
    ("coh_grand_strategy", 529340): (3, ""), ("coh_grand_strategy", 583590): (2, ""),
    ("coh_grand_strategy", 703510): (1, "GENRE_ONLY"), ("coh_grand_strategy", 781960): (1, "TOO_NICHE"),
    ("coh_grand_strategy", 919640): (2, ""), ("coh_grand_strategy", 951670): (2, ""),
    ("coh_grand_strategy", 959150): (3, ""), ("coh_grand_strategy", 979920): (2, ""),
    ("coh_grand_strategy", 1176470): (3, ""), ("coh_grand_strategy", 1401170): (1, "LOW_QUALITY"),
    ("coh_grand_strategy", 1605220): (3, ""), ("coh_grand_strategy", 1724860): (1, "LOW_QUALITY"),
    ("coh_grand_strategy", 1889320): (2, ""), ("coh_grand_strategy", 2147380): (2, ""),
    ("coh_grand_strategy", 2162610): (2, ""), ("coh_grand_strategy", 2284800): (2, ""),
    ("coh_grand_strategy", 2374280): (2, ""), ("coh_grand_strategy", 2685070): (1, "LOW_QUALITY"),
    ("coh_grand_strategy", 2819870): (2, ""), ("coh_grand_strategy", 3105440): (2, ""),
    ("coh_grand_strategy", 3158990): (0, "IRRELEVANT"), ("coh_grand_strategy", 3558380): (2, ""),
    ("coh_grand_strategy", 3867040): (2, ""), ("coh_grand_strategy", 4253970): (2, ""),
    ("coh_grand_strategy", 4498250): (2, ""), ("coh_grand_strategy", 4539870): (2, ""),

    ("coh_indie_platformer", 96100): (2, ""), ("coh_indie_platformer", 218740): (2, ""),
    ("coh_indie_platformer", 252030): (3, ""), ("coh_indie_platformer", 261570): (3, ""),
    ("coh_indie_platformer", 262960): (3, ""), ("coh_indie_platformer", 318530): (3, ""),
    ("coh_indie_platformer", 350070): (3, ""), ("coh_indie_platformer", 356650): (3, ""),
    ("coh_indie_platformer", 418120): (2, ""), ("coh_indie_platformer", 418150): (2, ""),
    ("coh_indie_platformer", 446840): (3, ""), ("coh_indie_platformer", 447780): (2, ""),
    ("coh_indie_platformer", 590590): (3, ""), ("coh_indie_platformer", 612390): (3, ""),
    ("coh_indie_platformer", 619280): (2, ""), ("coh_indie_platformer", 630310): (3, ""),
    ("coh_indie_platformer", 653210): (1, "TOO_NICHE"), ("coh_indie_platformer", 718590): (3, ""),
    ("coh_indie_platformer", 745250): (2, ""), ("coh_indie_platformer", 765520): (2, ""),
    ("coh_indie_platformer", 843360): (2, ""), ("coh_indie_platformer", 940910): (3, ""),
    ("coh_indie_platformer", 1000760): (3, ""), ("coh_indie_platformer", 1024210): (2, ""),
    ("coh_indie_platformer", 1083310): (2, ""), ("coh_indie_platformer", 1123050): (3, ""),
    ("coh_indie_platformer", 1147560): (3, ""), ("coh_indie_platformer", 1167380): (2, ""),
    ("coh_indie_platformer", 1253920): (3, ""), ("coh_indie_platformer", 1264880): (2, ""),
    ("coh_indie_platformer", 1274600): (3, ""), ("coh_indie_platformer", 1599420): (2, ""),
    ("coh_indie_platformer", 1642790): (2, ""), ("coh_indie_platformer", 1672810): (3, ""),
    ("coh_indie_platformer", 1701520): (3, ""), ("coh_indie_platformer", 1702180): (1, "LOW_QUALITY"),
    ("coh_indie_platformer", 1771620): (2, ""), ("coh_indie_platformer", 1802880): (2, ""),
    ("coh_indie_platformer", 1842730): (3, ""), ("coh_indie_platformer", 2023360): (2, ""),
    ("coh_indie_platformer", 2109060): (2, ""), ("coh_indie_platformer", 2741830): (2, ""),
    ("coh_indie_platformer", 2925120): (3, ""), ("coh_indie_platformer", 3032830): (3, ""),
    ("coh_indie_platformer", 3138520): (2, ""),

    ("coh_openworld_survival", 108600): (2, ""), ("coh_openworld_survival", 223490): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 238240): (2, ""), ("coh_openworld_survival", 299740): (2, ""),
    ("coh_openworld_survival", 312210): (1, "LOW_QUALITY"), ("coh_openworld_survival", 322330): (3, ""),
    ("coh_openworld_survival", 324080): (2, ""), ("coh_openworld_survival", 329430): (2, ""),
    ("coh_openworld_survival", 332500): (2, ""), ("coh_openworld_survival", 333340): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 346110): (3, ""), ("coh_openworld_survival", 351100): (2, ""),
    ("coh_openworld_survival", 382310): (3, ""), ("coh_openworld_survival", 393420): (2, ""),
    ("coh_openworld_survival", 420930): (2, ""), ("coh_openworld_survival", 509770): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 677480): (2, ""), ("coh_openworld_survival", 809210): (1, "LOW_QUALITY"),
    ("coh_openworld_survival", 895400): (2, ""), ("coh_openworld_survival", 1169040): (2, ""),
    ("coh_openworld_survival", 1250220): (1, "LOW_QUALITY"), ("coh_openworld_survival", 1284190): (3, ""),
    ("coh_openworld_survival", 1309820): (2, ""), ("coh_openworld_survival", 1323060): (2, ""),
    ("coh_openworld_survival", 1463730): (2, ""), ("coh_openworld_survival", 1524630): (2, ""),
    ("coh_openworld_survival", 1631270): (3, ""), ("coh_openworld_survival", 1670810): (2, ""),
    ("coh_openworld_survival", 1759350): (2, ""), ("coh_openworld_survival", 1805110): (2, ""),
    ("coh_openworld_survival", 1852180): (1, "LOW_QUALITY"), ("coh_openworld_survival", 2348700): (1, "LOW_QUALITY"),

    ("coh_strategy", 4700): (3, ""), ("coh_strategy", 8800): (3, ""), ("coh_strategy", 34030): (3, ""),
    ("coh_strategy", 273070): (2, ""), ("coh_strategy", 292160): (2, ""), ("coh_strategy", 357310): (3, ""),
    ("coh_strategy", 394360): (3, ""), ("coh_strategy", 437440): (2, ""), ("coh_strategy", 462940): (2, ""),
    ("coh_strategy", 489630): (3, ""), ("coh_strategy", 573410): (3, ""), ("coh_strategy", 597180): (3, ""),
    ("coh_strategy", 779340): (3, ""), ("coh_strategy", 791400): (2, ""), ("coh_strategy", 872410): (2, ""),
    ("coh_strategy", 1011390): (3, ""), ("coh_strategy", 1025440): (2, ""), ("coh_strategy", 1072040): (2, ""),
    ("coh_strategy", 1088790): (2, ""), ("coh_strategy", 1268590): (3, ""), ("coh_strategy", 1306770): (2, ""),
    ("coh_strategy", 1430140): (1, "LOW_QUALITY"), ("coh_strategy", 1481170): (3, ""),
    ("coh_strategy", 1546080): (2, ""), ("coh_strategy", 1605220): (3, ""), ("coh_strategy", 1669000): (3, ""),
    ("coh_strategy", 1873540): (1, "LOW_QUALITY"), ("coh_strategy", 1937780): (3, ""),
    ("coh_strategy", 2117940): (1, "TOO_NICHE"), ("coh_strategy", 2156540): (1, "LOW_QUALITY"),
    ("coh_strategy", 2214890): (1, "LOW_QUALITY"), ("coh_strategy", 2216750): (2, ""),
    ("coh_strategy", 2231270): (2, ""), ("coh_strategy", 2310980): (1, "LOW_QUALITY"),
    ("coh_strategy", 2346150): (2, ""), ("coh_strategy", 2381290): (2, ""), ("coh_strategy", 2494920): (2, ""),
    ("coh_strategy", 2925130): (1, "LOW_QUALITY"), ("coh_strategy", 2948440): (2, ""),
    ("coh_strategy", 3815550): (2, ""), ("coh_strategy", 3839930): (1, "LOW_QUALITY"),
    ("coh_strategy", 4538450): (2, ""), ("coh_strategy", 4549610): (2, ""),
    ("coh_strategy", 4661910): (1, "LOW_QUALITY"),

    ("coh_survival_craft", 254200): (2, ""), ("coh_survival_craft", 327090): (2, ""),
    ("coh_survival_craft", 344760): (2, ""), ("coh_survival_craft", 361420): (3, ""),
    ("coh_survival_craft", 366220): (2, ""), ("coh_survival_craft", 369080): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 393420): (2, ""), ("coh_survival_craft", 414190): (2, ""),
    ("coh_survival_craft", 548480): (1, "LOW_QUALITY"), ("coh_survival_craft", 568570): (2, ""),
    ("coh_survival_craft", 625340): (1, "GENRE_ONLY"), ("coh_survival_craft", 670260): (2, ""),
    ("coh_survival_craft", 677480): (2, ""), ("coh_survival_craft", 760800): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 794490): (1, "LOW_QUALITY"), ("coh_survival_craft", 839630): (2, ""),
    ("coh_survival_craft", 895400): (2, ""), ("coh_survival_craft", 1134700): (2, ""),
    ("coh_survival_craft", 1377380): (2, ""), ("coh_survival_craft", 1635450): (2, ""),
    ("coh_survival_craft", 1670810): (1, "GENRE_ONLY"), ("coh_survival_craft", 1702760): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 1749200): (1, "LOW_QUALITY"), ("coh_survival_craft", 1861630): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 1898300): (3, ""), ("coh_survival_craft", 2094600): (2, ""),
    ("coh_survival_craft", 2119830): (3, ""), ("coh_survival_craft", 2175700): (2, ""),
    ("coh_survival_craft", 2218970): (2, ""), ("coh_survival_craft", 2442490): (1, "LOW_QUALITY"),
    ("coh_survival_craft", 2868100): (1, "LOW_QUALITY"), ("coh_survival_craft", 2909870): (2, ""),

    ("coh_vehicle_sim", 211500): (3, ""), ("coh_vehicle_sim", 365960): (3, ""),
    ("coh_vehicle_sim", 491540): (3, ""), ("coh_vehicle_sim", 645630): (2, ""),
    ("coh_vehicle_sim", 690790): (3, ""), ("coh_vehicle_sim", 767390): (2, ""),
    ("coh_vehicle_sim", 957050): (2, ""), ("coh_vehicle_sim", 1066890): (3, ""),
    ("coh_vehicle_sim", 1153410): (3, ""), ("coh_vehicle_sim", 1273400): (2, ""),
    ("coh_vehicle_sim", 1285310): (1, "LOW_QUALITY"), ("coh_vehicle_sim", 1944790): (2, ""),
    ("coh_vehicle_sim", 2327720): (3, ""), ("coh_vehicle_sim", 2488620): (3, ""),
    ("coh_vehicle_sim", 3239770): (2, ""), ("coh_vehicle_sim", 3371650): (2, ""),
    ("coh_vehicle_sim", 3498800): (2, ""), ("coh_vehicle_sim", 3532150): (2, ""),

    ("mix2_builder_sim", 2700): (3, ""), ("mix2_builder_sim", 233450): (3, ""),
    ("mix2_builder_sim", 245620): (3, ""), ("mix2_builder_sim", 311260): (2, ""),
    ("mix2_builder_sim", 375910): (3, ""), ("mix2_builder_sim", 447020): (3, ""),
    ("mix2_builder_sim", 873840): (1, "GENRE_ONLY"), ("mix2_builder_sim", 916440): (3, ""),
    ("mix2_builder_sim", 1069910): (2, ""), ("mix2_builder_sim", 1106840): (3, ""),
    ("mix2_builder_sim", 1244460): (3, ""), ("mix2_builder_sim", 1244910): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 1287530): (3, ""), ("mix2_builder_sim", 1302860): (3, ""),
    ("mix2_builder_sim", 1318690): (3, ""), ("mix2_builder_sim", 1457320): (3, ""),
    ("mix2_builder_sim", 1475310): (3, ""), ("mix2_builder_sim", 1547890): (3, ""),
    ("mix2_builder_sim", 1638300): (3, ""), ("mix2_builder_sim", 2001070): (2, ""),
    ("mix2_builder_sim", 2019920): (3, ""), ("mix2_builder_sim", 2152810): (3, ""),
    ("mix2_builder_sim", 2489330): (3, ""), ("mix2_builder_sim", 2670630): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 2811830): (1, "SAME_GAME_DIFFERENT_SKU"), ("mix2_builder_sim", 2868100): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 2961880): (1, "GENRE_ONLY"), ("mix2_builder_sim", 3320980): (3, ""),
    ("mix2_builder_sim", 3408110): (1, "GENRE_ONLY"), ("mix2_builder_sim", 3732960): (1, "GENRE_ONLY"),
    ("mix2_builder_sim", 4090330): (1, "GENRE_ONLY"),

    ("mix2_colony_cozy", 346420): (2, ""), ("mix2_colony_cozy", 352720): (3, ""),
    ("mix2_colony_cozy", 544550): (3, ""), ("mix2_colony_cozy", 677340): (3, ""),
    ("mix2_colony_cozy", 727570): (3, ""), ("mix2_colony_cozy", 1105330): (2, ""),
    ("mix2_colony_cozy", 1140130): (2, ""), ("mix2_colony_cozy", 1243360): (3, ""),
    ("mix2_colony_cozy", 1403740): (3, ""), ("mix2_colony_cozy", 1465470): (3, ""),
    ("mix2_colony_cozy", 1571990): (2, ""), ("mix2_colony_cozy", 1613630): (2, ""),
    ("mix2_colony_cozy", 1616540): (2, ""), ("mix2_colony_cozy", 1620870): (2, ""),
    ("mix2_colony_cozy", 2001070): (2, ""), ("mix2_colony_cozy", 2060240): (2, ""),
    ("mix2_colony_cozy", 2201620): (3, ""), ("mix2_colony_cozy", 2209140): (3, ""),
    ("mix2_colony_cozy", 2244130): (3, ""), ("mix2_colony_cozy", 2628570): (2, ""),
    ("mix2_colony_cozy", 2810070): (2, ""), ("mix2_colony_cozy", 3651310): (2, ""),
    ("mix2_colony_cozy", 4548960): (3, ""),

    ("mix2_coop_horror", 812380): (2, ""), ("mix2_coop_horror", 816090): (3, ""),
    ("mix2_coop_horror", 1351000): (1, "GENRE_ONLY"), ("mix2_coop_horror", 1487390): (2, ""),
    ("mix2_coop_horror", 1547090): (3, ""), ("mix2_coop_horror", 1575830): (1, "GENRE_ONLY"),
    ("mix2_coop_horror", 1712050): (3, ""), ("mix2_coop_horror", 1804170): (3, ""),
    ("mix2_coop_horror", 1869500): (2, ""), ("mix2_coop_horror", 1916310): (3, ""),
    ("mix2_coop_horror", 1975370): (3, ""), ("mix2_coop_horror", 1980770): (2, ""),
    ("mix2_coop_horror", 2103950): (2, ""), ("mix2_coop_horror", 2141730): (3, ""),
    ("mix2_coop_horror", 2283880): (1, "GENRE_ONLY"), ("mix2_coop_horror", 2886590): (3, ""),
    ("mix2_coop_horror", 2956680): (2, ""), ("mix2_coop_horror", 2959560): (3, ""),
    ("mix2_coop_horror", 3228590): (3, ""), ("mix2_coop_horror", 3321400): (2, ""),
    ("mix2_coop_horror", 3712230): (3, ""), ("mix2_coop_horror", 4408510): (3, ""),

    ("mix2_crpg_sandbox", 48720): (3, ""), ("mix2_crpg_sandbox", 109600): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 206480): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 212500): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 257350): (3, ""), ("mix2_crpg_sandbox", 290080): (2, ""),
    ("mix2_crpg_sandbox", 311260): (2, ""), ("mix2_crpg_sandbox", 324260): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 381640): (0, "IRRELEVANT"), ("mix2_crpg_sandbox", 405710): (0, "IRRELEVANT"),
    ("mix2_crpg_sandbox", 427510): (2, ""), ("mix2_crpg_sandbox", 487120): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 529180): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 638650): (0, "IRRELEVANT"),
    ("mix2_crpg_sandbox", 688470): (2, ""), ("mix2_crpg_sandbox", 716350): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 739650): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 768200): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 1009290): (0, "IRRELEVANT"), ("mix2_crpg_sandbox", 1063730): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 1084600): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 1129580): (2, ""),
    ("mix2_crpg_sandbox", 1130700): (0, "IRRELEVANT"), ("mix2_crpg_sandbox", 1172710): (2, ""),
    ("mix2_crpg_sandbox", 1203620): (2, ""), ("mix2_crpg_sandbox", 1238000): (2, ""),
    ("mix2_crpg_sandbox", 1371580): (2, ""), ("mix2_crpg_sandbox", 1827180): (0, "IRRELEVANT"),
    ("mix2_crpg_sandbox", 1874880): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 1990050): (1, "GENRE_ONLY"),
    ("mix2_crpg_sandbox", 2246340): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 2521740): (1, "LOW_QUALITY"),
    ("mix2_crpg_sandbox", 2803490): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 2868100): (1, "LOW_QUALITY"),
    ("mix2_crpg_sandbox", 2880010): (2, ""), ("mix2_crpg_sandbox", 2984450): (1, "LOW_QUALITY"),
    ("mix2_crpg_sandbox", 3107230): (1, "GENRE_ONLY"), ("mix2_crpg_sandbox", 3114070): (1, "LOW_QUALITY"),
    ("mix2_crpg_sandbox", 3383930): (1, "LOW_QUALITY"), ("mix2_crpg_sandbox", 3752810): (1, "LOW_QUALITY"),
    ("mix2_crpg_sandbox", 3808550): (2, ""), ("mix2_crpg_sandbox", 4369490): (1, "GENRE_ONLY"),

    ("mix2_modern_roguelite", 113200): (3, ""), ("mix2_modern_roguelite", 238280): (2, ""),
    ("mix2_modern_roguelite", 373290): (2, ""), ("mix2_modern_roguelite", 452060): (3, ""),
    ("mix2_modern_roguelite", 493080): (3, ""), ("mix2_modern_roguelite", 509570): (2, ""),
    ("mix2_modern_roguelite", 542350): (1, "LOW_QUALITY"), ("mix2_modern_roguelite", 622370): (2, ""),
    ("mix2_modern_roguelite", 657000): (3, ""), ("mix2_modern_roguelite", 685400): (2, ""),
    ("mix2_modern_roguelite", 704950): (2, ""), ("mix2_modern_roguelite", 878420): (2, ""),
    ("mix2_modern_roguelite", 960690): (3, ""), ("mix2_modern_roguelite", 1016790): (2, ""),
    ("mix2_modern_roguelite", 1074610): (2, ""), ("mix2_modern_roguelite", 1123770): (3, ""),
    ("mix2_modern_roguelite", 1184820): (3, ""), ("mix2_modern_roguelite", 1323470): (3, ""),
    ("mix2_modern_roguelite", 1536210): (2, ""), ("mix2_modern_roguelite", 1569090): (3, ""),
    ("mix2_modern_roguelite", 1608040): (2, ""), ("mix2_modern_roguelite", 1684080): (2, ""),
    ("mix2_modern_roguelite", 1736550): (2, ""), ("mix2_modern_roguelite", 1808320): (2, ""),
    ("mix2_modern_roguelite", 1842730): (2, ""), ("mix2_modern_roguelite", 1920650): (2, ""),
    ("mix2_modern_roguelite", 1930350): (2, ""), ("mix2_modern_roguelite", 1964200): (3, ""),
    ("mix2_modern_roguelite", 2026250): (2, ""), ("mix2_modern_roguelite", 2056210): (3, ""),
    ("mix2_modern_roguelite", 2088570): (3, ""), ("mix2_modern_roguelite", 2106670): (3, ""),
    ("mix2_modern_roguelite", 2126370): (2, ""), ("mix2_modern_roguelite", 2225960): (3, ""),
    ("mix2_modern_roguelite", 2266980): (2, ""), ("mix2_modern_roguelite", 2280440): (2, ""),
    ("mix2_modern_roguelite", 2321350): (2, ""), ("mix2_modern_roguelite", 2452820): (2, ""),
    ("mix2_modern_roguelite", 2493510): (2, ""), ("mix2_modern_roguelite", 2514910): (2, ""),
    ("mix2_modern_roguelite", 2543510): (2, ""), ("mix2_modern_roguelite", 2576200): (3, ""),
    ("mix2_modern_roguelite", 2727210): (3, ""), ("mix2_modern_roguelite", 2877790): (1, "LOW_QUALITY"),
    ("mix2_modern_roguelite", 2908120): (2, ""), ("mix2_modern_roguelite", 3094800): (2, ""),
    ("mix2_modern_roguelite", 3332790): (2, ""), ("mix2_modern_roguelite", 3352240): (2, ""),
    ("mix2_modern_roguelite", 3375890): (2, ""), ("mix2_modern_roguelite", 3424930): (2, ""),
    ("mix2_modern_roguelite", 3496890): (3, ""), ("mix2_modern_roguelite", 3520420): (2, ""),

    ("mix2_soulslike_narrative", 20900): (3, ""), ("mix2_soulslike_narrative", 22330): (3, ""),
    ("mix2_soulslike_narrative", 33230): (2, ""), ("mix2_soulslike_narrative", 39690): (2, ""),
    ("mix2_soulslike_narrative", 40390): (2, ""), ("mix2_soulslike_narrative", 48190): (2, ""),
    ("mix2_soulslike_narrative", 204030): (2, ""), ("mix2_soulslike_narrative", 219780): (2, ""),
    ("mix2_soulslike_narrative", 225260): (2, ""), ("mix2_soulslike_narrative", 247660): (3, ""),
    ("mix2_soulslike_narrative", 280180): (1, "GENRE_ONLY"), ("mix2_soulslike_narrative", 292030): (3, ""),
    ("mix2_soulslike_narrative", 347000): (2, ""), ("mix2_soulslike_narrative", 351970): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 368500): (2, ""), ("mix2_soulslike_narrative", 372360): (1, "GENRE_ONLY"),
    ("mix2_soulslike_narrative", 403970): (2, ""), ("mix2_soulslike_narrative", 417880): (3, ""),
    ("mix2_soulslike_narrative", 446810): (1, "GENRE_ONLY"), ("mix2_soulslike_narrative", 489830): (3, ""),
    ("mix2_soulslike_narrative", 493200): (3, ""), ("mix2_soulslike_narrative", 574050): (0, "IRRELEVANT"),
    ("mix2_soulslike_narrative", 582010): (2, ""), ("mix2_soulslike_narrative", 589360): (2, ""),
    ("mix2_soulslike_narrative", 595520): (2, ""), ("mix2_soulslike_narrative", 618970): (2, ""),
    ("mix2_soulslike_narrative", 637650): (2, ""), ("mix2_soulslike_narrative", 921570): (2, ""),
    ("mix2_soulslike_narrative", 1042780): (2, ""), ("mix2_soulslike_narrative", 1091500): (3, ""),
    ("mix2_soulslike_narrative", 1093360): (1, "LOW_QUALITY"), ("mix2_soulslike_narrative", 2556990): (2, ""),
    ("mix2_soulslike_narrative", 2623190): (3, ""),
}



# 죽은 멀티(멀티 전용 + 리뷰 미보고) 제거로 새로 노출된 4쌍.
BLIND_DDM = {
    ("coh_classic_multi", 1222210): (2, ""),   # 물리 기반 공성 샌드박스 — GMod 축
    ("mix2_party_narrative", 253690): (3, ""),  # 4인 로컬 협동 퍼즐 — It Takes Two / Overcooked 축
    ("mix2_party_narrative", 1016920): (3, ""), # 협동 혼돈 멀티 — Overcooked 직결
    ("mix2_party_narrative", 2235200): (2, ""), # 4인 온라인 협동 로그라이크 — 협동이지만 파티/서사는 아님
}

# 허브니스 보정 λ=0.5 로 새로 노출된 532쌍. 같은 블라인드 기준.
BLIND_HUB05 = {
 ("coh_arpg",377160):(3,""),("coh_arpg",462770):(2,""),("coh_arpg",587100):(2,""),
 ("coh_arpg",716350):(1,"GENRE_ONLY"),("coh_arpg",1018010):(1,"GENRE_ONLY"),
 ("coh_arpg",1026680):(1,"GENRE_ONLY"),("coh_arpg",1295510):(2,""),("coh_arpg",1710180):(1,"TOO_NICHE"),
 ("coh_arpg",1721060):(3,""),("coh_arpg",1910860):(1,"MODE_MISMATCH"),("coh_arpg",2473480):(3,""),
 ("coh_arpg",3333250):(2,""),
 ("coh_classic_multi",218230):(3,""),("coh_classic_multi",244850):(2,""),("coh_classic_multi",286160):(3,""),
 ("coh_classic_multi",374280):(2,""),("coh_classic_multi",397540):(3,""),("coh_classic_multi",924970):(3,""),
 ("coh_classic_multi",1012110):(2,""),("coh_classic_multi",1969870):(1,"LOW_QUALITY"),
 ("coh_classic_multi",2263920):(2,""),("coh_classic_multi",2281200):(1,"LOW_QUALITY"),
 ("coh_classic_multi",2364570):(1,"LOW_QUALITY"),
 ("coh_cozy",758690):(1,"GENRE_ONLY"),("coh_cozy",815370):(1,"GENRE_ONLY"),("coh_cozy",971030):(3,""),
 ("coh_cozy",1018800):(2,""),("coh_cozy",1370040):(1,"GENRE_ONLY"),("coh_cozy",1716740):(0,"IRRELEVANT"),
 ("coh_cozy",2300320):(2,""),("coh_cozy",2721300):(3,""),("coh_cozy",2977620):(3,""),
 ("coh_cozy",3412850):(3,""),("coh_cozy",3951340):(3,""),
 ("coh_fps",42700):(3,""),("coh_fps",504370):(2,""),("coh_fps",687850):(0,"IRRELEVANT"),
 ("coh_fps",703940):(0,"IRRELEVANT"),("coh_fps",1700920):(1,"LOW_QUALITY"),("coh_fps",1873030):(2,""),
 ("coh_fps",1952920):(0,"IRRELEVANT"),("coh_fps",2252870):(1,"LOW_QUALITY"),("coh_fps",2838380):(1,"LOW_QUALITY"),
 ("coh_fps",3251690):(1,"LOW_QUALITY"),
 ("coh_grand_strategy",42960):(3,""),("coh_grand_strategy",306630):(3,""),("coh_grand_strategy",400470):(2,""),
 ("coh_grand_strategy",792930):(2,""),("coh_grand_strategy",1238330):(2,""),("coh_grand_strategy",1770050):(2,""),
 ("coh_grand_strategy",1985050):(3,""),("coh_grand_strategy",3114680):(1,"GENRE_ONLY"),
 ("coh_grand_strategy",3318550):(2,""),("coh_grand_strategy",3898040):(2,""),
 ("coh_indie_platformer",402120):(2,""),("coh_indie_platformer",681110):(3,""),("coh_indie_platformer",808010):(2,""),
 ("coh_indie_platformer",881100):(3,""),("coh_indie_platformer",1462030):(2,""),("coh_indie_platformer",1550760):(3,""),
 ("coh_indie_platformer",1875580):(3,""),("coh_indie_platformer",1962030):(2,""),("coh_indie_platformer",2059210):(3,""),
 ("coh_indie_platformer",2076580):(3,""),("coh_indie_platformer",2617090):(3,""),("coh_indie_platformer",3697230):(2,""),
 ("coh_indie_platformer",4265120):(2,""),("coh_indie_platformer",4388490):(2,""),("coh_indie_platformer",4458820):(3,""),
 ("coh_indie_platformer",4579210):(2,""),
 ("coh_openworld_survival",511430):(2,""),("coh_openworld_survival",1393830):(3,""),("coh_openworld_survival",2842040):(2,""),
 ("coh_strategy",3990):(3,""),("coh_strategy",216130):(2,""),("coh_strategy",338130):(2,""),
 ("coh_strategy",410990):(2,""),("coh_strategy",593030):(2,""),("coh_strategy",765810):(2,""),
 ("coh_strategy",965310):(2,""),("coh_strategy",1966130):(2,""),("coh_strategy",2374280):(2,""),
 ("coh_strategy",2533020):(2,""),("coh_strategy",2799350):(3,""),("coh_strategy",2874590):(2,""),
 ("coh_strategy",3381680):(3,""),
 ("coh_survival_craft",1048280):(1,"LOW_QUALITY"),("coh_survival_craft",1090800):(3,""),
 ("coh_survival_craft",1370040):(2,""),("coh_survival_craft",1405180):(1,"LOW_QUALITY"),
 ("coh_survival_craft",1409510):(1,"LOW_QUALITY"),("coh_survival_craft",1874190):(2,""),
 ("coh_survival_craft",3576870):(2,""),
 ("coh_vehicle_sim",2002520):(2,""),("coh_vehicle_sim",2099680):(2,""),("coh_vehicle_sim",4078230):(2,""),
 ("longtail_deckbuilder",1038370):(3,""),("longtail_deckbuilder",1162480):(3,""),("longtail_deckbuilder",1531250):(3,""),
 ("longtail_deckbuilder",1681840):(3,""),("longtail_deckbuilder",1724390):(3,""),("longtail_deckbuilder",1996430):(3,""),
 ("longtail_deckbuilder",2345740):(3,""),("longtail_deckbuilder",2573470):(3,""),("longtail_deckbuilder",2638050):(3,""),
 ("longtail_deckbuilder",2720540):(2,""),("longtail_deckbuilder",2784470):(3,""),("longtail_deckbuilder",3035330):(3,""),
 ("longtail_deckbuilder",3157320):(2,""),("longtail_deckbuilder",3217480):(3,""),("longtail_deckbuilder",3312330):(3,""),
 ("longtail_deckbuilder",3399930):(2,""),("longtail_deckbuilder",3490280):(3,""),("longtail_deckbuilder",4398750):(3,""),
 ("longtail_deckbuilder",4732210):(3,""),("longtail_deckbuilder",3127770):(3,""),
 ("longtail_detective",262000):(3,""),("longtail_detective",279560):(2,""),("longtail_detective",287720):(2,""),
 ("longtail_detective",298140):(3,""),("longtail_detective",433580):(2,""),("longtail_detective",498050):(2,""),
 ("longtail_detective",624270):(3,""),("longtail_detective",832540):(2,""),("longtail_detective",1064660):(2,""),
 ("longtail_detective",1330630):(2,""),("longtail_detective",1600460):(2,""),("longtail_detective",1697940):(2,""),
 ("longtail_detective",1848440):(3,""),("longtail_detective",1926590):(3,""),("longtail_detective",2541740):(3,""),
 ("longtail_detective",2722360):(3,""),("longtail_detective",2732790):(3,""),("longtail_detective",2996850):(3,""),
 ("longtail_detective",3246240):(3,""),("longtail_detective",3246260):(3,""),("longtail_detective",3779920):(2,""),
 ("longtail_detective",4030180):(3,""),("longtail_detective",4208950):(2,""),
 ("longtail_metroidvania",568070):(3,""),("longtail_metroidvania",603110):(3,""),("longtail_metroidvania",845030):(2,""),
 ("longtail_metroidvania",851100):(3,""),("longtail_metroidvania",1034910):(3,""),("longtail_metroidvania",1079200):(3,""),
 ("longtail_metroidvania",1122280):(3,""),("longtail_metroidvania",1324150):(3,""),("longtail_metroidvania",1429500):(3,""),
 ("longtail_metroidvania",1430220):(3,""),("longtail_metroidvania",1434340):(1,"SAME_GAME_DIFFERENT_SKU"),
 ("longtail_metroidvania",1546710):(3,""),("longtail_metroidvania",1625510):(3,""),("longtail_metroidvania",1698870):(2,""),
 ("longtail_metroidvania",1825850):(2,""),("longtail_metroidvania",1885670):(2,""),("longtail_metroidvania",2063020):(3,""),
 ("longtail_metroidvania",2097790):(2,""),("longtail_metroidvania",2191540):(3,""),("longtail_metroidvania",2685570):(3,""),
 ("longtail_metroidvania",2883560):(3,""),("longtail_metroidvania",3408600):(3,""),("longtail_metroidvania",3525870):(3,""),
 ("longtail_metroidvania",3837380):(2,""),("longtail_metroidvania",3930550):(2,""),("longtail_metroidvania",4745430):(2,""),
 ("longtail_puzzle_platformer",385710):(3,""),("longtail_puzzle_platformer",416840):(3,""),
 ("longtail_puzzle_platformer",715060):(1,"LOW_QUALITY"),("longtail_puzzle_platformer",866140):(3,""),
 ("longtail_puzzle_platformer",1002540):(2,""),("longtail_puzzle_platformer",1017660):(1,"GENRE_ONLY"),
 ("longtail_puzzle_platformer",1072620):(3,""),("longtail_puzzle_platformer",1107470):(2,""),
 ("longtail_puzzle_platformer",1459540):(3,""),("longtail_puzzle_platformer",1541220):(2,""),
 ("longtail_puzzle_platformer",1651680):(3,""),("longtail_puzzle_platformer",1801110):(3,""),
 ("longtail_puzzle_platformer",1808400):(3,""),("longtail_puzzle_platformer",1892570):(3,""),
 ("longtail_puzzle_platformer",1926390):(2,""),("longtail_puzzle_platformer",1995940):(2,""),
 ("longtail_puzzle_platformer",2685900):(3,""),("longtail_puzzle_platformer",2804680):(2,""),
 ("longtail_puzzle_platformer",2989180):(3,""),("longtail_puzzle_platformer",4275250):(2,""),
 ("lowrev_cozy_narrative",355970):(3,""),("lowrev_cozy_narrative",629960):(1,"GENRE_ONLY"),
 ("lowrev_cozy_narrative",987480):(2,""),("lowrev_cozy_narrative",1161190):(0,"IRRELEVANT"),
 ("lowrev_cozy_narrative",1362860):(2,""),("lowrev_cozy_narrative",1432850):(3,""),
 ("lowrev_cozy_narrative",1491670):(3,""),("lowrev_cozy_narrative",1575290):(2,""),
 ("lowrev_cozy_narrative",1624210):(3,""),("lowrev_cozy_narrative",1647780):(2,""),
 ("lowrev_cozy_narrative",1689520):(2,""),("lowrev_cozy_narrative",2420350):(2,""),
 ("lowrev_cozy_narrative",2697000):(2,""),("lowrev_cozy_narrative",2960800):(2,""),
 ("lowrev_cozy_narrative",3034450):(2,""),("lowrev_cozy_narrative",3074860):(2,""),
 ("lowrev_cozy_narrative",3269600):(3,""),("lowrev_cozy_narrative",3290610):(1,"GENRE_ONLY"),
 ("lowrev_cozy_narrative",3298390):(2,""),("lowrev_cozy_narrative",3304840):(2,""),
 ("lowrev_cozy_narrative",3363590):(3,""),("lowrev_cozy_narrative",4042330):(2,""),
 ("lowrev_cozy_narrative",4043300):(3,""),("lowrev_cozy_narrative",4183210):(3,""),
 ("lowrev_cozy_narrative",4634070):(3,""),("lowrev_cozy_narrative",4766420):(2,""),
 ("lowrev_cozy_narrative",4855490):(1,"GENRE_ONLY"),
 ("lowrev_deckbuilder",905490):(3,""),("lowrev_deckbuilder",1038370):(3,""),("lowrev_deckbuilder",1231940):(3,""),
 ("lowrev_deckbuilder",1314770):(3,""),("lowrev_deckbuilder",1447590):(3,""),("lowrev_deckbuilder",2294160):(3,""),
 ("lowrev_deckbuilder",3219010):(3,""),("lowrev_deckbuilder",3284290):(3,""),("lowrev_deckbuilder",3312330):(3,""),
 ("lowrev_deckbuilder",3490280):(3,""),("lowrev_deckbuilder",3509430):(3,""),("lowrev_deckbuilder",3885520):(3,""),
 ("lowrev_deckbuilder",4266290):(3,""),
 ("lowrev_detective",31850):(3,""),("lowrev_detective",205650):(3,""),("lowrev_detective",711430):(3,""),
 ("lowrev_detective",809000):(2,""),("lowrev_detective",1247940):(3,""),("lowrev_detective",1352200):(2,""),
 ("lowrev_detective",1395030):(3,""),("lowrev_detective",1551820):(2,""),("lowrev_detective",1592420):(2,""),
 ("lowrev_detective",1626950):(2,""),("lowrev_detective",1632770):(2,""),("lowrev_detective",1777200):(3,""),
 ("lowrev_detective",2114300):(3,""),("lowrev_detective",2183320):(3,""),("lowrev_detective",2259330):(3,""),
 ("lowrev_detective",2333590):(2,""),("lowrev_detective",2608420):(3,""),("lowrev_detective",2852480):(3,""),
 ("lowrev_detective",3233260):(3,""),("lowrev_detective",3406690):(2,""),("lowrev_detective",4081900):(2,""),
 ("lowrev_detective",4262600):(2,""),
 ("lowrev_metroidvania",630310):(3,""),("lowrev_metroidvania",889770):(1,"LOW_QUALITY"),
 ("lowrev_metroidvania",891170):(2,""),("lowrev_metroidvania",1079200):(3,""),("lowrev_metroidvania",1402120):(3,""),
 ("lowrev_metroidvania",1625510):(3,""),("lowrev_metroidvania",1747560):(3,""),("lowrev_metroidvania",1747760):(3,""),
 ("lowrev_metroidvania",1898610):(3,""),("lowrev_metroidvania",2230650):(3,""),("lowrev_metroidvania",2827750):(3,""),
 ("lowrev_metroidvania",2945380):(2,""),
 ("lowrev_towerdefense",386070):(2,""),("lowrev_towerdefense",469600):(3,""),("lowrev_towerdefense",538170):(3,""),
 ("lowrev_towerdefense",912900):(3,""),("lowrev_towerdefense",1202130):(2,""),("lowrev_towerdefense",1265460):(3,""),
 ("lowrev_towerdefense",1358000):(2,""),("lowrev_towerdefense",1595090):(3,""),("lowrev_towerdefense",1601630):(3,""),
 ("lowrev_towerdefense",1651940):(2,""),("lowrev_towerdefense",1937820):(3,""),("lowrev_towerdefense",2369760):(3,""),
 ("lowrev_towerdefense",2547320):(2,""),("lowrev_towerdefense",3408490):(2,""),("lowrev_towerdefense",3908920):(2,""),
 ("lowrev_towerdefense",4051980):(2,""),("lowrev_towerdefense",4500660):(2,""),
 ("mix2_arcade_action",315210):(1,"GENRE_ONLY"),("mix2_arcade_action",326340):(2,""),("mix2_arcade_action",329050):(3,""),
 ("mix2_arcade_action",561600):(2,""),("mix2_arcade_action",671370):(2,""),("mix2_arcade_action",702120):(3,""),
 ("mix2_arcade_action",985890):(3,""),("mix2_arcade_action",1017160):(2,""),("mix2_arcade_action",1018160):(2,""),
 ("mix2_arcade_action",1114290):(3,""),("mix2_arcade_action",1275760):(2,""),("mix2_arcade_action",1484180):(2,""),
 ("mix2_arcade_action",1620750):(1,"GENRE_ONLY"),("mix2_arcade_action",1953520):(2,""),("mix2_arcade_action",2165350):(2,""),
 ("mix2_arcade_action",2217580):(3,""),("mix2_arcade_action",2292440):(2,""),("mix2_arcade_action",2499990):(3,""),
 ("mix2_arcade_action",2584550):(1,"LOW_QUALITY"),("mix2_arcade_action",2651280):(3,""),("mix2_arcade_action",3359910):(2,""),
 ("mix2_arcade_action",3454980):(3,""),("mix2_arcade_action",3474450):(2,""),("mix2_arcade_action",4129690):(2,""),
 ("mix2_builder_sim",672630):(3,""),("mix2_builder_sim",781180):(3,""),("mix2_builder_sim",1079510):(3,""),
 ("mix2_builder_sim",1138580):(2,""),("mix2_builder_sim",1933930):(2,""),("mix2_builder_sim",2779120):(3,""),
 ("mix2_builder_sim",3582250):(2,""),("mix2_builder_sim",3585520):(2,""),("mix2_builder_sim",3706960):(2,""),
 ("mix2_builder_sim",3891760):(3,""),("mix2_builder_sim",3928990):(3,""),("mix2_builder_sim",4094120):(3,""),
 ("mix2_colony_cozy",1383150):(3,""),("mix2_colony_cozy",1513160):(3,""),("mix2_colony_cozy",1866300):(3,""),
 ("mix2_colony_cozy",2577420):(3,""),("mix2_colony_cozy",2623390):(3,""),("mix2_colony_cozy",2671970):(1,"GENRE_ONLY"),
 ("mix2_colony_cozy",3501510):(3,""),
 ("mix2_coop_horror",1374490):(1,"GENRE_ONLY"),("mix2_coop_horror",1568620):(3,""),("mix2_coop_horror",1843840):(3,""),
 ("mix2_coop_horror",2479150):(3,""),("mix2_coop_horror",3223330):(2,""),
 ("mix2_crpg_sandbox",7110):(3,""),("mix2_crpg_sandbox",700030):(1,"GENRE_ONLY"),("mix2_crpg_sandbox",763890):(3,""),
 ("mix2_crpg_sandbox",1116050):(1,"GENRE_ONLY"),("mix2_crpg_sandbox",1158310):(3,""),("mix2_crpg_sandbox",1192440):(2,""),
 ("mix2_crpg_sandbox",1371760):(2,""),("mix2_crpg_sandbox",1500820):(2,""),("mix2_crpg_sandbox",1516820):(2,""),
 ("mix2_crpg_sandbox",1545830):(3,""),("mix2_crpg_sandbox",2393370):(3,""),("mix2_crpg_sandbox",2732100):(1,"GENRE_ONLY"),
 ("mix2_crpg_sandbox",3518390):(2,""),("mix2_crpg_sandbox",3972520):(1,"LOW_QUALITY"),
 ("mix2_modern_roguelite",1627210):(2,""),("mix2_modern_roguelite",1940340):(3,""),("mix2_modern_roguelite",2027510):(2,""),
 ("mix2_modern_roguelite",2316660):(1,"SAME_GAME_DIFFERENT_SKU"),("mix2_modern_roguelite",2338040):(3,""),
 ("mix2_modern_roguelite",2454830):(3,""),("mix2_modern_roguelite",2605790):(3,""),("mix2_modern_roguelite",2924580):(3,""),
 ("mix2_party_narrative",289690):(3,""),("mix2_party_narrative",359840):(3,""),("mix2_party_narrative",489110):(3,""),
 ("mix2_party_narrative",673750):(3,""),("mix2_party_narrative",896950):(2,""),("mix2_party_narrative",945360):(2,""),
 ("mix2_party_narrative",1132030):(3,""),("mix2_party_narrative",1260320):(2,""),("mix2_party_narrative",1299000):(2,""),
 ("mix2_party_narrative",1637400):(3,""),("mix2_party_narrative",1977530):(3,""),("mix2_party_narrative",2109090):(3,""),
 ("mix2_party_narrative",2172370):(2,""),("mix2_party_narrative",2225610):(3,""),("mix2_party_narrative",2876640):(2,""),
 ("mix2_party_narrative",3253150):(3,""),("mix2_party_narrative",3468170):(3,""),("mix2_party_narrative",4164940):(3,""),
 ("mix2_puzzle_survival",304410):(2,""),("mix2_puzzle_survival",395350):(2,""),("mix2_puzzle_survival",427410):(3,""),
 ("mix2_puzzle_survival",950360):(2,""),("mix2_puzzle_survival",1003890):(2,""),("mix2_puzzle_survival",1284190):(3,""),
 ("mix2_puzzle_survival",1424980):(2,""),("mix2_puzzle_survival",1553120):(2,""),("mix2_puzzle_survival",1601570):(3,""),
 ("mix2_puzzle_survival",1694420):(3,""),("mix2_puzzle_survival",2111090):(2,""),("mix2_puzzle_survival",2367750):(1,"LOW_QUALITY"),
 ("mix2_puzzle_survival",2646460):(3,""),("mix2_puzzle_survival",2727210):(2,""),("mix2_puzzle_survival",2819030):(2,""),
 ("mix2_puzzle_survival",2984860):(2,""),("mix2_puzzle_survival",3292780):(2,""),("mix2_puzzle_survival",3342630):(2,""),
 ("mix2_puzzle_survival",3414580):(3,""),("mix2_puzzle_survival",3517980):(2,""),("mix2_puzzle_survival",3555780):(2,""),
 ("mix2_puzzle_survival",3743220):(2,""),("mix2_puzzle_survival",3837690):(3,""),
 ("mix2_soulslike_narrative",40300):(2,""),("mix2_soulslike_narrative",200260):(3,""),
 ("mix2_soulslike_narrative",215280):(2,""),("mix2_soulslike_narrative",253980):(1,"GENRE_ONLY"),
 ("mix2_soulslike_narrative",306130):(2,""),("mix2_soulslike_narrative",377160):(3,""),
 ("mix2_soulslike_narrative",570940):(3,""),("mix2_soulslike_narrative",900040):(2,""),
 ("mix2_soulslike_narrative",1380220):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",1708850):(1,"GENRE_ONLY"),
 ("mix2_soulslike_narrative",1716740):(2,""),("mix2_soulslike_narrative",1750770):(2,""),
 ("mix2_soulslike_narrative",1829520):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",1898880):(2,""),
 ("mix2_soulslike_narrative",1949030):(3,""),("mix2_soulslike_narrative",2215200):(1,"GENRE_ONLY"),
 ("mix2_soulslike_narrative",2265990):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",2284600):(2,""),
 ("mix2_soulslike_narrative",2313330):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",2422590):(1,"LOW_QUALITY"),
 ("mix2_soulslike_narrative",3644710):(1,"LOW_QUALITY"),("mix2_soulslike_narrative",4286990):(2,""),
 ("mix2_survival_farm",220260):(2,""),("mix2_survival_farm",673950):(3,""),("mix2_survival_farm",758690):(2,""),
 ("mix2_survival_farm",1248130):(2,""),("mix2_survival_farm",1307550):(3,""),("mix2_survival_farm",1702760):(1,"LOW_QUALITY"),
 ("mix2_survival_farm",1713350):(2,""),("mix2_survival_farm",2059530):(3,""),("mix2_survival_farm",2372030):(2,""),
 ("mix2_survival_farm",2826570):(1,"LOW_QUALITY"),("mix2_survival_farm",2909870):(2,""),("mix2_survival_farm",3041230):(3,""),
 ("mix2_survival_farm",4134600):(2,""),
 ("niche_cozy_casual",336380):(3,""),("niche_cozy_casual",923260):(3,""),("niche_cozy_casual",944260):(1,"LOW_QUALITY"),
 ("niche_cozy_casual",999830):(2,""),("niche_cozy_casual",1040550):(3,""),("niche_cozy_casual",1140400):(2,""),
 ("niche_cozy_casual",1676910):(3,""),("niche_cozy_casual",1881180):(3,""),("niche_cozy_casual",2060590):(3,""),
 ("niche_cozy_casual",2122810):(3,""),("niche_cozy_casual",2154400):(3,""),("niche_cozy_casual",2223000):(1,"LOW_QUALITY"),
 ("niche_cozy_casual",2382260):(1,"LOW_QUALITY"),("niche_cozy_casual",2538380):(2,""),("niche_cozy_casual",2600720):(3,""),
 ("niche_cozy_casual",2727970):(3,""),("niche_cozy_casual",3285260):(3,""),("niche_cozy_casual",3457390):(3,""),
 ("niche_cozy_casual",3623490):(3,""),("niche_cozy_casual",4297410):(2,""),
 ("niche_puzzle_solo",319270):(3,""),("niche_puzzle_solo",845590):(3,""),("niche_puzzle_solo",943190):(3,""),
 ("niche_puzzle_solo",1093370):(2,""),("niche_puzzle_solo",1248420):(3,""),("niche_puzzle_solo",1714050):(3,""),
 ("niche_puzzle_solo",1811750):(3,""),("niche_puzzle_solo",1952740):(3,""),("niche_puzzle_solo",1972830):(2,""),
 ("niche_puzzle_solo",2072840):(3,""),("niche_puzzle_solo",2166920):(3,""),("niche_puzzle_solo",2279060):(3,""),
 ("niche_puzzle_solo",2562040):(3,""),("niche_puzzle_solo",2609910):(2,""),("niche_puzzle_solo",2844290):(3,""),
 ("niche_puzzle_solo",3197170):(2,""),("niche_puzzle_solo",3385260):(3,""),("niche_puzzle_solo",3412380):(2,""),
 ("niche_puzzle_solo",3453400):(2,""),
 ("niche_roguelite",311480):(2,""),("niche_roguelite",452060):(3,""),("niche_roguelite",595140):(3,""),
 ("niche_roguelite",917950):(2,""),("niche_roguelite",1099170):(2,""),("niche_roguelite",2205870):(1,"GENRE_ONLY"),
 ("niche_roguelite",2288470):(3,""),("niche_roguelite",2352710):(2,""),("niche_roguelite",2717880):(3,""),
 ("niche_roguelite",3065870):(2,""),("niche_roguelite",3100310):(2,""),("niche_roguelite",3237640):(1,"GENRE_ONLY"),
 ("niche_roguelite",3985950):(3,""),
 ("niche_sim",70120):(3,""),("niche_sim",205650):(2,""),("niche_sim",392160):(2,""),("niche_sim",403620):(3,""),
 ("niche_sim",491950):(3,""),("niche_sim",544550):(3,""),("niche_sim",1200570):(2,""),("niche_sim",1273400):(2,""),
 ("niche_sim",1393320):(3,""),("niche_sim",1668130):(3,""),("niche_sim",1826570):(3,""),("niche_sim",2487060):(3,""),
 ("niche_sim",2527160):(3,""),("niche_sim",2665100):(2,""),("niche_sim",3667680):(3,""),("niche_sim",4054430):(3,""),
 ("niche_sim",4276210):(2,""),("niche_sim",4651320):(3,""),
 ("niche_soulslike_solo",333330):(1,"GENRE_ONLY"),("niche_soulslike_solo",351970):(1,"GENRE_ONLY"),
 ("niche_soulslike_solo",619780):(1,"GENRE_ONLY"),("niche_soulslike_solo",653060):(3,""),
 ("niche_soulslike_solo",1244090):(1,"GENRE_ONLY"),("niche_soulslike_solo",1349600):(3,""),
 ("niche_soulslike_solo",1371980):(3,""),("niche_soulslike_solo",1426490):(3,""),("niche_soulslike_solo",1501750):(3,""),
 ("niche_soulslike_solo",1546760):(2,""),("niche_soulslike_solo",1613080):(2,""),("niche_soulslike_solo",1869500):(2,""),
 ("niche_soulslike_solo",1982480):(2,""),("niche_soulslike_solo",2054660):(0,"IRRELEVANT"),
 ("niche_soulslike_solo",2130510):(2,""),("niche_soulslike_solo",2201910):(3,""),
 ("niche_soulslike_solo",2383180):(1,"GENRE_ONLY"),("niche_soulslike_solo",2417340):(2,""),
 ("niche_soulslike_solo",3087910):(3,""),("niche_soulslike_solo",4033210):(2,""),
 ("niche_soulslike_solo",4467090):(3,""),("niche_soulslike_solo",4506630):(3,""),
 ("niche_tactics",238910):(3,""),("niche_tactics",253630):(3,""),("niche_tactics",738610):(2,""),
 ("niche_tactics",1135200):(3,""),("niche_tactics",1184810):(3,""),("niche_tactics",1291610):(3,""),
 ("niche_tactics",1605520):(3,""),("niche_tactics",1812970):(2,""),("niche_tactics",1990110):(3,""),
 ("niche_tactics",2171550):(3,""),("niche_tactics",2345740):(3,""),("niche_tactics",2366190):(2,""),
 ("niche_tactics",2473410):(3,""),("niche_tactics",2738210):(3,""),("niche_tactics",3041130):(3,""),
 ("niche_tactics",3786930):(2,""),("niche_tactics",3846410):(3,""),("niche_tactics",4048820):(2,""),
}



# 허브니스 보정 λ=0.25 로 새로 노출된 14쌍.
BLIND_HUB25 = {
    ("coh_cozy", 645320): (2, ""),                 # 귀여운 퍼즐 3D 플랫포머 — 아늑하지만 농사/샌드박스 아님
    ("coh_indie_platformer", 1205170): (3, ""),    # 정밀 퍼즐 플랫포머 — Celeste 축
    ("coh_strategy", 2820110): (3, ""),            # Anno 클래식, 도시건설 + 4X
    ("longtail_deckbuilder", 2858880): (2, ""),    # 덱빌딩 + 실시간 영토전, 덱빌더 정체성은 옅다
    ("longtail_detective", 762520): (2, ""),       # 히든오브젝트 어드벤처
    ("longtail_detective", 1790490): (3, ""),      # 방탈출 퍼즐 — Mad Experiments 직결
    ("lowrev_cozy_narrative", 3645660): (3, ""),   # 야생 식물 채집 인터랙티브 픽션 — Botany Manor 직결
    ("mix2_arcade_action", 536560): (3, ""),       # 2D 대전격투 — SF6 축
    ("niche_cozy_casual", 435400): (3, ""),        # 손그림 숨은그림찾기, 편안함 — Gorogoa 축
    ("niche_cozy_casual", 1295260): (2, ""),       # 잔잔한 탐험 플랫포머
    ("niche_puzzle_solo", 3385920): (3, ""),       # 레이저로 디지털 회로 구성 — Opus Magnum 축
    ("niche_roguelite", 506870): (3, ""),          # 액션 로그라이트, 평가 좋음
    ("niche_roguelite", 1145350): (3, ""),         # Hades II — 시드 취향 정중앙
    ("niche_tactics", 3464540): (3, ""),           # 로그라이크 전술 + 카드 — Into the Breach + Monster Train
}



# 허브니스 보정 λ=0.15 / 0.35 로 새로 노출된 11쌍.
BLIND_HUBFINE = {
    ("coh_arpg", 568670): (2, ""),                    # 3인칭 소울라이크 루터, 무명
    ("mix2_arcade_action", 2486820): (3, ""),         # 아케이드 레이싱 — Forza Horizon 축
    ("mix2_arcade_action", 3856040): (1, "KEYWORD_MATCH"),  # 태그만 잔뜩, 실체 불명
    ("mix2_coop_horror", 1794830): (3, ""),           # 1~4인 협동 호러 — Lethal Company 직결
    ("mix2_crpg_sandbox", 2156440): (1, "GENRE_ONLY"),      # MMO 던전 레이드
    ("mix2_puzzle_survival", 2766330): (3, ""),       # 시뮬레이션 탈출 퍼즐 — Talos Principle 축
    ("mix2_survival_farm", 2570210): (2, ""),         # 서바이벌 크래프트 + 공장, 농사 축은 약함
    ("niche_cozy_casual", 1807370): (2, ""),          # 미니멀 편안한 퍼즐
    ("niche_puzzle_solo", 3095910): (2, ""),          # 빛 기반 논리 퍼즐, 소코반 계열
    ("niche_roguelite", 622370): (2, ""),             # 액션 로그라이트 협동
    ("niche_roguelite", 1641240): (2, ""),            # 단순 로그라이트 RPG
}

# k=100 확장(채택 설정: top2_mean + 단일시드 희귀태그 + 하한개방 + 조건부 합의태그
# + 죽은멀티 제거 + 허브니스 λ=0.35)으로 새로 노출된 241쌍.
BLIND_K100 = {
 ("coh_arpg",212500):(1,"GENRE_ONLY"),("coh_arpg",628670):(2,""),("coh_arpg",1114220):(2,""),
 ("coh_arpg",1944360):(1,"TOO_NICHE"),("coh_arpg",2186990):(3,""),("coh_arpg",2246390):(2,""),
 ("coh_arpg",3882670):(2,""),
 ("coh_classic_multi",386180):(2,""),("coh_classic_multi",678800):(2,""),("coh_classic_multi",694280):(3,""),
 ("coh_classic_multi",916930):(1,"LOW_QUALITY"),("coh_classic_multi",1086410):(2,""),
 ("coh_classic_multi",2868100):(1,"LOW_QUALITY"),("coh_classic_multi",3213850):(1,"GENRE_ONLY"),
 ("coh_classic_multi",3674980):(2,""),
 ("coh_cozy",90200):(1,"GENRE_ONLY"),("coh_cozy",1335830):(3,""),("coh_cozy",1504570):(2,""),
 ("coh_cozy",2099190):(2,""),("coh_cozy",2408700):(1,"LOW_QUALITY"),("coh_cozy",4032230):(3,""),
 ("coh_fps",60):(1,"GENRE_ONLY"),("coh_fps",15000):(2,""),("coh_fps",624970):(2,""),
 ("coh_fps",1665460):(0,"IRRELEVANT"),("coh_fps",1681730):(2,""),("coh_fps",3511270):(2,""),
 ("coh_grand_strategy",1727750):(2,""),("coh_grand_strategy",2533020):(2,""),("coh_grand_strategy",4546580):(2,""),
 ("coh_indie_platformer",571310):(3,""),("coh_indie_platformer",1371570):(3,""),
 ("coh_indie_platformer",2725260):(3,""),("coh_indie_platformer",2966850):(3,""),("coh_indie_platformer",3784950):(2,""),
 ("coh_openworld_survival",1394960):(3,""),("coh_openworld_survival",1588560):(2,""),
 ("coh_openworld_survival",4724610):(2,""),
 ("coh_strategy",1610):(2,""),("coh_strategy",214150):(2,""),("coh_strategy",236850):(3,""),
 ("coh_strategy",984680):(3,""),("coh_strategy",1758080):(2,""),("coh_strategy",2842790):(2,""),
 ("coh_survival_craft",263200):(2,""),("coh_survival_craft",305620):(3,""),
 ("coh_survival_craft",681750):(1,"LOW_QUALITY"),("coh_survival_craft",1335830):(3,""),
 ("coh_survival_craft",3249910):(2,""),
 ("coh_vehicle_sim",514160):(3,""),("coh_vehicle_sim",673830):(1,"MODE_MISMATCH"),
 ("coh_vehicle_sim",1793420):(1,"GENRE_ONLY"),("coh_vehicle_sim",3649530):(2,""),
 ("longtail_deckbuilder",1057800):(3,""),("longtail_deckbuilder",1264240):(3,""),("longtail_deckbuilder",2084000):(3,""),
 ("longtail_deckbuilder",2181950):(3,""),("longtail_deckbuilder",2198070):(3,""),("longtail_deckbuilder",2428020):(3,""),
 ("longtail_deckbuilder",2473260):(3,""),("longtail_deckbuilder",2539880):(3,""),("longtail_deckbuilder",3265700):(3,""),
 ("longtail_deckbuilder",3298430):(3,""),("longtail_deckbuilder",3488580):(2,""),("longtail_deckbuilder",3709000):(3,""),
 ("longtail_deckbuilder",4784740):(3,""),
 ("longtail_detective",11180):(3,""),("longtail_detective",299110):(2,""),("longtail_detective",410120):(2,""),
 ("longtail_detective",493720):(3,""),("longtail_detective",1107300):(3,""),("longtail_detective",1364010):(2,""),
 ("longtail_detective",1672240):(2,""),("longtail_detective",2012320):(3,""),("longtail_detective",2859200):(3,""),
 ("longtail_detective",3196770):(3,""),
 ("longtail_metroidvania",1562140):(3,""),("longtail_metroidvania",1614440):(3,""),
 ("longtail_metroidvania",1672810):(3,""),("longtail_metroidvania",2064000):(3,""),("longtail_metroidvania",3123920):(3,""),
 ("longtail_puzzle_platformer",111800):(3,""),("longtail_puzzle_platformer",263980):(3,""),
 ("longtail_puzzle_platformer",383270):(3,""),("longtail_puzzle_platformer",765520):(2,""),
 ("longtail_puzzle_platformer",1553800):(2,""),("longtail_puzzle_platformer",2141780):(3,""),
 ("longtail_puzzle_platformer",2467000):(2,""),("longtail_puzzle_platformer",3170770):(2,""),
 ("lowrev_cozy_narrative",261680):(3,""),("lowrev_cozy_narrative",1427300):(2,""),
 ("lowrev_cozy_narrative",3048770):(2,""),("lowrev_cozy_narrative",3668830):(3,""),
 ("lowrev_cozy_narrative",4436910):(1,"GENRE_ONLY"),
 ("lowrev_deckbuilder",1260590):(3,""),("lowrev_deckbuilder",1535100):(3,""),("lowrev_deckbuilder",1716940):(3,""),
 ("lowrev_deckbuilder",2088770):(3,""),("lowrev_deckbuilder",2697930):(3,""),("lowrev_deckbuilder",2852980):(3,""),
 ("lowrev_deckbuilder",3127770):(3,""),("lowrev_deckbuilder",3132780):(3,""),("lowrev_deckbuilder",3157320):(2,""),
 ("lowrev_deckbuilder",3217480):(3,""),("lowrev_deckbuilder",3438640):(3,""),("lowrev_deckbuilder",3481020):(3,""),
 ("lowrev_detective",226820):(3,""),("lowrev_detective",1294230):(3,""),("lowrev_detective",2477490):(2,""),
 ("lowrev_detective",2564260):(1,"GENRE_ONLY"),("lowrev_detective",3036350):(3,""),
 ("lowrev_metroidvania",374190):(3,""),("lowrev_metroidvania",400910):(3,""),("lowrev_metroidvania",1191630):(3,""),
 ("lowrev_metroidvania",1516250):(2,""),("lowrev_metroidvania",1565990):(2,""),("lowrev_metroidvania",1756020):(3,""),
 ("lowrev_metroidvania",1872680):(3,""),("lowrev_metroidvania",2973120):(1,"LOW_QUALITY"),
 ("lowrev_metroidvania",3392510):(3,""),("lowrev_metroidvania",3639650):(3,""),("lowrev_metroidvania",4381200):(3,""),
 ("lowrev_towerdefense",315460):(3,""),("lowrev_towerdefense",422910):(3,""),("lowrev_towerdefense",1556270):(2,""),
 ("lowrev_towerdefense",1746140):(2,""),("lowrev_towerdefense",2186360):(3,""),("lowrev_towerdefense",2250560):(2,""),
 ("lowrev_towerdefense",2276500):(2,""),("lowrev_towerdefense",2442770):(2,""),
 ("lowrev_towerdefense",2593390):(1,"LOW_QUALITY"),("lowrev_towerdefense",2676390):(2,""),
 ("mix2_arcade_action",292410):(2,""),("mix2_arcade_action",390710):(3,""),("mix2_arcade_action",425240):(2,""),
 ("mix2_arcade_action",520440):(3,""),("mix2_arcade_action",571260):(3,""),
 ("mix2_arcade_action",1628940):(1,"LOW_QUALITY"),("mix2_arcade_action",1930340):(1,"LOW_QUALITY"),
 ("mix2_arcade_action",2442380):(3,""),("mix2_arcade_action",2581700):(2,""),
 ("mix2_builder_sim",1351500):(2,""),("mix2_builder_sim",4210580):(3,""),
 ("mix2_colony_cozy",3635320):(3,""),("mix2_colony_cozy",4196740):(3,""),("mix2_colony_cozy",4245820):(3,""),
 ("mix2_coop_horror",2245610):(3,""),("mix2_coop_horror",2278360):(3,""),("mix2_coop_horror",2576150):(2,""),
 ("mix2_coop_horror",2902470):(3,""),("mix2_coop_horror",4606900):(3,""),
 ("mix2_crpg_sandbox",1101400):(2,""),("mix2_crpg_sandbox",3159770):(3,""),
 ("mix2_modern_roguelite",1146230):(3,""),("mix2_modern_roguelite",1296660):(2,""),("mix2_modern_roguelite",1608640):(3,""),
 ("mix2_modern_roguelite",1793330):(2,""),("mix2_modern_roguelite",2068280):(3,""),("mix2_modern_roguelite",2436940):(3,""),
 ("mix2_modern_roguelite",2735580):(2,""),("mix2_modern_roguelite",2824490):(3,""),("mix2_modern_roguelite",2985450):(2,""),
 ("mix2_modern_roguelite",3181290):(2,""),
 ("mix2_party_narrative",341800):(3,""),("mix2_party_narrative",423810):(3,""),("mix2_party_narrative",2145460):(3,""),
 ("mix2_party_narrative",2909960):(3,""),("mix2_party_narrative",3021850):(3,""),("mix2_party_narrative",3124540):(2,""),
 ("mix2_party_narrative",3444220):(2,""),("mix2_party_narrative",3564340):(3,""),("mix2_party_narrative",3922920):(2,""),
 ("mix2_puzzle_survival",307940):(2,""),("mix2_puzzle_survival",351100):(2,""),("mix2_puzzle_survival",1424600):(3,""),
 ("mix2_puzzle_survival",1783680):(3,""),("mix2_puzzle_survival",2536960):(2,""),("mix2_puzzle_survival",3636620):(2,""),
 ("mix2_soulslike_narrative",20920):(3,""),("mix2_soulslike_narrative",356190):(3,""),
 ("mix2_soulslike_narrative",649950):(3,""),("mix2_soulslike_narrative",1172710):(2,""),
 ("mix2_soulslike_narrative",1771300):(3,""),("mix2_soulslike_narrative",1962700):(2,""),
 ("mix2_soulslike_narrative",2622380):(3,""),("mix2_soulslike_narrative",3672400):(2,""),
 ("mix2_soulslike_narrative",3839600):(2,""),
 ("mix2_survival_farm",943260):(3,""),("mix2_survival_farm",1062520):(3,""),("mix2_survival_farm",1159690):(3,""),
 ("mix2_survival_farm",1253220):(2,""),("mix2_survival_farm",1548910):(3,""),("mix2_survival_farm",1812450):(3,""),
 ("mix2_survival_farm",2268560):(1,"GENRE_ONLY"),("mix2_survival_farm",2313330):(2,""),
 ("mix2_survival_farm",2617700):(3,""),("mix2_survival_farm",3922920):(1,"GENRE_ONLY"),
 ("niche_cozy_casual",2323660):(3,""),("niche_cozy_casual",2887670):(3,""),("niche_cozy_casual",3721240):(2,""),
 ("niche_cozy_casual",4452310):(2,""),
 ("niche_puzzle_solo",265890):(2,""),("niche_puzzle_solo",290020):(3,""),("niche_puzzle_solo",716490):(3,""),
 ("niche_puzzle_solo",990900):(3,""),("niche_puzzle_solo",1044980):(1,"GENRE_ONLY"),
 ("niche_puzzle_solo",1303670):(0,"IRRELEVANT"),("niche_puzzle_solo",1383150):(2,""),
 ("niche_puzzle_solo",1511780):(3,""),("niche_puzzle_solo",4579720):(3,""),
 ("niche_roguelite",200710):(2,""),("niche_roguelite",462860):(2,""),("niche_roguelite",691280):(2,""),
 ("niche_roguelite",989450):(1,"LOW_QUALITY"),("niche_roguelite",1095040):(3,""),
 ("niche_roguelite",1101190):(1,"GENRE_ONLY"),("niche_roguelite",1629200):(2,""),("niche_roguelite",2320890):(3,""),
 ("niche_roguelite",2564520):(2,""),("niche_roguelite",2960490):(2,""),("niche_roguelite",3816930):(2,""),
 ("niche_sim",720250):(3,""),("niche_sim",1341450):(3,""),("niche_sim",4012690):(3,""),
 ("niche_soulslike_solo",269670):(1,"GENRE_ONLY"),("niche_soulslike_solo",386080):(3,""),
 ("niche_soulslike_solo",678960):(3,""),("niche_soulslike_solo",1016600):(2,""),
 ("niche_soulslike_solo",1232810):(1,"GENRE_ONLY"),("niche_soulslike_solo",1448440):(3,""),
 ("niche_soulslike_solo",1634940):(2,""),("niche_soulslike_solo",1867070):(3,""),
 ("niche_soulslike_solo",2227040):(2,""),("niche_soulslike_solo",2650730):(0,"IRRELEVANT"),
 ("niche_soulslike_solo",2850110):(2,""),
 ("niche_tactics",492090):(3,""),("niche_tactics",681730):(3,""),("niche_tactics",2699280):(3,""),
 ("niche_tactics",2987110):(3,""),("niche_tactics",3530170):(3,""),("niche_tactics",4776920):(3,""),
}



# PvP 의존 보강 후 k=100 에서 새로 노출된 13쌍.
BLIND_K100_PVP = {
    ("coh_fps", 212070): (1, "GENRE_ONLY"),          # 우주 MMO 시뮬 — 경쟁 FPS 아님
    ("coh_vehicle_sim", 851110): (1, "LOW_QUALITY"),  # 방치형 클리커
    ("coh_vehicle_sim", 2837580): (2, ""),            # 아케이드 랠리
    ("coh_vehicle_sim", 601170): (2, ""),             # 트럭 랠리크로스 — ETS2/BeamNG 축 접점
    ("niche_puzzle_solo", 2736930): (2, ""),          # 공장 자동화 퍼즐, 무명
    ("niche_tactics", 2427450): (3, ""),              # 턴제 로그라이트 전술 — Into the Breach 축
    ("mix2_arcade_action", 357190): (3, ""),          # 2D 대전격투 명작 — SF6 축
    ("mix2_arcade_action", 2064820): (2, ""),         # 스트릿볼 + 격투 아케이드
    ("mix2_arcade_action", 342560): (1, "GENRE_ONLY"),  # 비행선 전략/4X — 세 시드 어디에도 안 맞음
    ("mix2_arcade_action", 2488290): (3, ""),         # 벨트스크롤 액션, 상쾌한 콤보 — Hi-Fi RUSH 축
    ("mix2_arcade_action", 1888160): (3, ""),         # 아머드 코어 VI — 고속 액션 스펙터클
    ("mix2_arcade_action", 999020): (2, ""),          # 고전 액션 플랫포머 컬렉션
    ("mix2_arcade_action", 635260): (3, ""),          # 드리프트 레이싱 — Forza Horizon 축
}

# 시드 인기도 비례 하한(계수 0.001 / 0.0005) 합집합으로 새로 노출된 277쌍.
BLIND_SSF = {
 ("coh_arpg",268540):(1,"GENRE_ONLY"),("coh_arpg",373420):(3,""),("coh_arpg",544750):(1,"GENRE_ONLY"),
 ("coh_arpg",774241):(2,""),("coh_arpg",919360):(3,""),("coh_arpg",1358700):(3,""),("coh_arpg",1627720):(3,""),
 ("coh_arpg",1656930):(2,""),("coh_arpg",1710170):(3,""),("coh_arpg",1845910):(3,""),("coh_arpg",2438330):(2,""),
 ("coh_arpg",2515020):(3,""),("coh_arpg",2536520):(2,""),
 ("coh_classic_multi",238430):(3,""),("coh_classic_multi",274560):(1,"GENRE_ONLY"),("coh_classic_multi",311210):(3,""),
 ("coh_classic_multi",326460):(3,""),("coh_classic_multi",383120):(2,""),("coh_classic_multi",383180):(3,""),
 ("coh_classic_multi",387990):(3,""),("coh_classic_multi",410900):(3,""),("coh_classic_multi",512490):(2,""),
 ("coh_classic_multi",674020):(2,""),("coh_classic_multi",758690):(1,"GENRE_ONLY"),("coh_classic_multi",843200):(2,""),
 ("coh_classic_multi",895400):(2,""),("coh_classic_multi",914260):(2,""),("coh_classic_multi",1189800):(2,""),
 ("coh_classic_multi",1195460):(3,""),("coh_classic_multi",1430190):(3,""),("coh_classic_multi",1562260):(1,"GENRE_ONLY"),
 ("coh_classic_multi",1755350):(1,"GENRE_ONLY"),("coh_classic_multi",2131350):(2,""),("coh_classic_multi",2154650):(2,""),
 ("coh_classic_multi",2218970):(1,"GENRE_ONLY"),("coh_classic_multi",2221490):(2,""),("coh_classic_multi",3140620):(2,""),
 ("coh_classic_multi",3438990):(3,""),("coh_classic_multi",3720460):(2,""),
 ("coh_cozy",824000):(3,""),("coh_cozy",962580):(3,""),("coh_cozy",978780):(3,""),("coh_cozy",1253220):(2,""),
 ("coh_cozy",1363900):(2,""),("coh_cozy",1385780):(1,"LOW_QUALITY"),("coh_cozy",1536090):(3,""),
 ("coh_cozy",1631470):(3,""),("coh_cozy",1780070):(3,""),("coh_cozy",1898300):(2,""),("coh_cozy",1940200):(3,""),
 ("coh_cozy",2168260):(2,""),("coh_cozy",2238470):(3,""),("coh_cozy",2285550):(3,""),
 ("coh_cozy",2379810):(1,"GENRE_ONLY"),("coh_cozy",2455370):(2,""),("coh_cozy",2631140):(1,"GENRE_ONLY"),
 ("coh_cozy",2681030):(3,""),
 ("coh_fps",1520):(1,"GENRE_ONLY"),("coh_fps",218620):(2,""),("coh_fps",236510):(2,""),("coh_fps",272230):(2,""),
 ("coh_fps",312660):(2,""),("coh_fps",365590):(2,""),("coh_fps",379720):(2,""),("coh_fps",394510):(2,""),
 ("coh_fps",397540):(2,""),("coh_fps",447820):(3,""),("coh_fps",581320):(3,""),("coh_fps",589290):(3,""),
 ("coh_fps",692890):(1,"GENRE_ONLY"),("coh_fps",1012110):(2,""),("coh_fps",1029690):(2,""),
 ("coh_fps",1151340):(1,"GENRE_ONLY"),("coh_fps",1158940):(2,""),("coh_fps",1237980):(3,""),
 ("coh_fps",1268750):(2,""),("coh_fps",1276760):(1,"GENRE_ONLY"),("coh_fps",1430190):(2,""),
 ("coh_fps",1677350):(0,"IRRELEVANT"),("coh_fps",1714420):(1,"LOW_QUALITY"),("coh_fps",1922010):(2,""),
 ("coh_fps",2012510):(0,"IRRELEVANT"),("coh_fps",2157560):(0,"IRRELEVANT"),("coh_fps",2272540):(2,""),
 ("coh_fps",2446550):(2,""),("coh_fps",2479810):(3,""),("coh_fps",2504090):(2,""),("coh_fps",3105890):(2,""),
 ("coh_fps",3932890):(3,""),("coh_fps",4384550):(3,""),
 ("coh_grand_strategy",73170):(3,""),("coh_grand_strategy",333420):(2,""),("coh_grand_strategy",383740):(2,""),
 ("coh_grand_strategy",410970):(2,""),("coh_grand_strategy",607050):(2,""),("coh_grand_strategy",844980):(2,""),
 ("coh_grand_strategy",1128810):(2,""),("coh_grand_strategy",1561960):(2,""),("coh_grand_strategy",1901910):(3,""),
 ("coh_grand_strategy",2228280):(2,""),("coh_grand_strategy",2248900):(0,"IRRELEVANT"),
 ("coh_grand_strategy",2307400):(1,"GENRE_ONLY"),("coh_grand_strategy",2583300):(2,""),
 ("coh_indie_platformer",262770):(2,""),("coh_indie_platformer",595790):(3,""),("coh_indie_platformer",657000):(3,""),
 ("coh_indie_platformer",791180):(3,""),("coh_indie_platformer",2877540):(2,""),
 ("coh_openworld_survival",244850):(2,""),("coh_openworld_survival",282140):(2,""),
 ("coh_openworld_survival",351290):(1,"LOW_QUALITY"),("coh_openworld_survival",545040):(2,""),
 ("coh_openworld_survival",574180):(2,""),("coh_openworld_survival",617030):(1,"LOW_QUALITY"),
 ("coh_openworld_survival",664750):(2,""),("coh_openworld_survival",706020):(3,""),
 ("coh_openworld_survival",1295920):(0,"IRRELEVANT"),("coh_openworld_survival",1360000):(3,""),
 ("coh_openworld_survival",1442530):(2,""),("coh_openworld_survival",1504570):(2,""),
 ("coh_openworld_survival",1850570):(2,""),("coh_openworld_survival",2379910):(1,"GENRE_ONLY"),
 ("coh_openworld_survival",2662780):(2,""),("coh_openworld_survival",2769780):(2,""),
 ("coh_openworld_survival",2949910):(3,""),("coh_openworld_survival",3976500):(2,""),
 ("coh_strategy",228200):(2,""),("coh_strategy",237470):(3,""),("coh_strategy",242570):(2,""),
 ("coh_strategy",252450):(3,""),("coh_strategy",277450):(2,""),("coh_strategy",1017900):(2,""),
 ("coh_strategy",1479730):(2,""),("coh_strategy",2298690):(3,""),("coh_strategy",2871870):(1,"GENRE_ONLY"),
 ("coh_survival_craft",224260):(2,""),("coh_survival_craft",249360):(2,""),("coh_survival_craft",250400):(2,""),
 ("coh_survival_craft",253250):(2,""),("coh_survival_craft",269310):(2,""),("coh_survival_craft",304930):(2,""),
 ("coh_survival_craft",383150):(2,""),("coh_survival_craft",387990):(2,""),("coh_survival_craft",738520):(3,""),
 ("coh_survival_craft",767490):(2,""),("coh_survival_craft",897450):(3,""),("coh_survival_craft",1069640):(2,""),
 ("coh_survival_craft",1125390):(3,""),("coh_survival_craft",1360000):(3,""),("coh_survival_craft",1380220):(2,""),
 ("coh_survival_craft",1504570):(2,""),("coh_survival_craft",1524630):(2,""),("coh_survival_craft",1759350):(2,""),
 ("coh_survival_craft",2126990):(1,"LOW_QUALITY"),
 ("coh_vehicle_sim",12520):(3,""),("coh_vehicle_sim",273840):(3,""),("coh_vehicle_sim",302080):(2,""),
 ("coh_vehicle_sim",412880):(2,""),("coh_vehicle_sim",577990):(1,"GENRE_ONLY"),("coh_vehicle_sim",658700):(2,""),
 ("coh_vehicle_sim",675010):(3,""),("coh_vehicle_sim",940580):(2,""),("coh_vehicle_sim",1109840):(0,"IRRELEVANT"),
 ("coh_vehicle_sim",1303990):(1,"GENRE_ONLY"),("coh_vehicle_sim",1314140):(2,""),("coh_vehicle_sim",1346010):(2,""),
 ("coh_vehicle_sim",1387500):(2,""),("coh_vehicle_sim",1416520):(2,""),("coh_vehicle_sim",1697880):(3,""),
 ("coh_vehicle_sim",1826420):(2,""),("coh_vehicle_sim",2064810):(3,""),("coh_vehicle_sim",2095420):(3,""),
 ("coh_vehicle_sim",2304680):(1,"LOW_QUALITY"),("coh_vehicle_sim",2487300):(2,""),("coh_vehicle_sim",2658040):(2,""),
 ("coh_vehicle_sim",3358360):(2,""),("coh_vehicle_sim",3501070):(2,""),
 ("mix2_builder_sim",226100):(2,""),("mix2_builder_sim",559910):(2,""),("mix2_builder_sim",919260):(3,""),
 ("mix2_builder_sim",1123830):(3,""),("mix2_builder_sim",1150090):(3,""),("mix2_builder_sim",1231520):(2,""),
 ("mix2_builder_sim",1342330):(2,""),("mix2_builder_sim",1450900):(3,""),("mix2_builder_sim",1981570):(3,""),
 ("mix2_builder_sim",2569670):(3,""),("mix2_builder_sim",2939600):(2,""),("mix2_builder_sim",3070880):(2,""),
 ("mix2_colony_cozy",280720):(3,""),("mix2_colony_cozy",394220):(1,"GENRE_ONLY"),("mix2_colony_cozy",738520):(2,""),
 ("mix2_colony_cozy",923970):(3,""),("mix2_colony_cozy",954870):(2,""),("mix2_colony_cozy",1198740):(2,""),
 ("mix2_colony_cozy",1309820):(2,""),("mix2_colony_cozy",1366540):(3,""),("mix2_colony_cozy",1857080):(3,""),
 ("mix2_colony_cozy",2877660):(2,""),("mix2_colony_cozy",3580160):(3,""),("mix2_colony_cozy",3681820):(2,""),
 ("mix2_colony_cozy",4194800):(2,""),
 ("mix2_coop_horror",223710):(2,""),("mix2_coop_horror",274480):(0,"IRRELEVANT"),("mix2_coop_horror",307110):(3,""),
 ("mix2_coop_horror",321270):(2,""),("mix2_coop_horror",381210):(3,""),("mix2_coop_horror",397310):(1,"GENRE_ONLY"),
 ("mix2_coop_horror",408900):(3,""),("mix2_coop_horror",434570):(2,""),("mix2_coop_horror",582500):(3,""),
 ("mix2_coop_horror",611350):(2,""),("mix2_coop_horror",677160):(3,""),("mix2_coop_horror",704270):(2,""),
 ("mix2_coop_horror",858820):(1,"GENRE_ONLY"),("mix2_coop_horror",891360):(3,""),
 ("mix2_coop_horror",1159690):(1,"GENRE_ONLY"),("mix2_coop_horror",1346070):(3,""),("mix2_coop_horror",1361000):(3,""),
 ("mix2_coop_horror",1526490):(2,""),("mix2_coop_horror",1558830):(3,""),("mix2_coop_horror",1609730):(1,"LOW_QUALITY"),
 ("mix2_coop_horror",1627850):(3,""),("mix2_coop_horror",1707540):(2,""),("mix2_coop_horror",1918970):(3,""),
 ("mix2_coop_horror",1929610):(3,""),("mix2_coop_horror",2115390):(3,""),("mix2_coop_horror",2321470):(2,""),
 ("mix2_coop_horror",2444750):(1,"GENRE_ONLY"),("mix2_coop_horror",2599370):(3,""),("mix2_coop_horror",2764750):(3,""),
 ("mix2_coop_horror",2845630):(1,"GENRE_ONLY"),("mix2_coop_horror",2903710):(3,""),
 ("mix2_coop_horror",2996040):(1,"GENRE_ONLY"),("mix2_coop_horror",3417410):(3,""),("mix2_coop_horror",3585800):(2,""),
 ("mix2_coop_horror",3712080):(2,""),("mix2_coop_horror",3736520):(3,""),
 ("mix2_crpg_sandbox",200710):(1,"GENRE_ONLY"),("mix2_crpg_sandbox",335620):(3,""),("mix2_crpg_sandbox",369530):(2,""),
 ("mix2_crpg_sandbox",973230):(2,""),("mix2_crpg_sandbox",1400660):(2,""),("mix2_crpg_sandbox",1786010):(3,""),
 ("mix2_crpg_sandbox",1857090):(3,""),("mix2_crpg_sandbox",2503770):(2,""),("mix2_crpg_sandbox",2525510):(2,""),
 ("mix2_crpg_sandbox",2975950):(3,""),("mix2_crpg_sandbox",3172700):(3,""),("mix2_crpg_sandbox",3407390):(2,""),
 ("mix2_modern_roguelite",1494810):(3,""),("mix2_modern_roguelite",2111870):(2,""),
 ("mix2_modern_roguelite",2250080):(2,""),("mix2_modern_roguelite",2285630):(3,""),
 ("mix2_modern_roguelite",2342150):(2,""),("mix2_modern_roguelite",2348610):(3,""),
 ("mix2_modern_roguelite",2360210):(3,""),("mix2_modern_roguelite",2429240):(2,""),
 ("mix2_modern_roguelite",2784470):(3,""),("mix2_modern_roguelite",2803280):(3,""),
 ("mix2_modern_roguelite",3100310):(2,""),("mix2_modern_roguelite",3762660):(2,""),
 ("mix2_puzzle_survival",434620):(1,"GENRE_ONLY"),("mix2_puzzle_survival",1374290):(3,""),
 ("mix2_puzzle_survival",1441180):(2,""),("mix2_puzzle_survival",1780370):(2,""),
 ("mix2_puzzle_survival",1889740):(3,""),("mix2_puzzle_survival",3083300):(2,""),
 ("mix2_soulslike_narrative",108710):(3,""),("mix2_soulslike_narrative",241930):(3,""),
 ("mix2_soulslike_narrative",243930):(2,""),("mix2_soulslike_narrative",411300):(3,""),
 ("mix2_soulslike_narrative",460930):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",794260):(3,""),
 ("mix2_soulslike_narrative",992910):(2,""),("mix2_soulslike_narrative",1550180):(2,""),
 ("mix2_soulslike_narrative",1928980):(2,""),
}



# 제품 기본값(page_size=10)으로 k=100 재생성 시 새로 노출된 117쌍 — 평가-제품 충실도 검증용.
BLIND_PS10 = {
 ("coh_arpg",29720):(1,"GENRE_ONLY"),("coh_arpg",39190):(2,""),("coh_arpg",40300):(3,""),
 ("coh_classic_multi",209160):(2,""),("coh_classic_multi",476600):(2,""),
 ("coh_cozy",313160):(2,""),("coh_cozy",322330):(2,""),("coh_cozy",447020):(2,""),
 ("coh_cozy",787860):(2,""),("coh_cozy",1201230):(3,""),("coh_cozy",1350840):(3,""),
 ("coh_cozy",1812450):(2,""),("coh_cozy",3408110):(2,""),
 ("coh_fps",30):(3,""),("coh_fps",209650):(2,""),("coh_fps",393080):(3,""),("coh_fps",476600):(2,""),
 ("coh_fps",553850):(2,""),("coh_fps",555570):(1,"LOW_QUALITY"),("coh_fps",1238810):(3,""),
 ("coh_fps",1238840):(3,""),("coh_fps",1262600):(0,"IRRELEVANT"),
 ("coh_grand_strategy",42850):(3,""),
 ("coh_strategy",16810):(3,""),("coh_strategy",957720):(2,""),
 ("coh_survival_craft",383180):(2,""),
 ("coh_vehicle_sim",516750):(3,""),("coh_vehicle_sim",1282590):(2,""),("coh_vehicle_sim",1784570):(2,""),
 ("coh_vehicle_sim",1822450):(2,""),("coh_vehicle_sim",1924170):(2,""),("coh_vehicle_sim",2380050):(3,""),
 ("coh_vehicle_sim",2486740):(2,""),
 ("longtail_detective",11130):(2,""),("longtail_detective",31800):(3,""),("longtail_detective",31810):(3,""),
 ("longtail_detective",31820):(3,""),("longtail_detective",43600):(3,""),("longtail_detective",615780):(3,""),
 ("longtail_detective",1676520):(2,""),
 ("longtail_puzzle_platformer",321370):(1,"LOW_QUALITY"),("longtail_puzzle_platformer",420060):(3,""),
 ("lowrev_detective",31820):(3,""),("lowrev_detective",31840):(3,""),("lowrev_detective",31930):(3,""),
 ("lowrev_detective",43600):(3,""),("lowrev_detective",200080):(3,""),
 ("lowrev_metroidvania",3055950):(3,""),
 ("lowrev_towerdefense",473560):(2,""),
 ("mix2_arcade_action",204360):(3,""),("mix2_arcade_action",348550):(3,""),("mix2_arcade_action",702890):(3,""),
 ("mix2_arcade_action",1328660):(3,""),("mix2_arcade_action",2440510):(3,""),
 ("mix2_builder_sim",90200):(2,""),("mix2_builder_sim",313160):(2,""),
 ("mix2_colony_cozy",39000):(1,"GENRE_ONLY"),("mix2_colony_cozy",400080):(3,""),
 ("mix2_colony_cozy",537520):(2,""),("mix2_colony_cozy",839310):(2,""),
 ("mix2_coop_horror",550):(3,""),("mix2_coop_horror",43190):(2,""),("mix2_coop_horror",67370):(1,"MODE_MISMATCH"),
 ("mix2_coop_horror",214870):(2,""),("mix2_coop_horror",415590):(3,""),("mix2_coop_horror",1027820):(2,""),
 ("mix2_coop_horror",1274570):(3,""),("mix2_coop_horror",1304930):(3,""),("mix2_coop_horror",1392860):(2,""),
 ("mix2_coop_horror",1492070):(3,""),("mix2_coop_horror",1592290):(3,""),("mix2_coop_horror",1644490):(2,""),
 ("mix2_coop_horror",1799220):(2,""),("mix2_coop_horror",1802330):(3,""),("mix2_coop_horror",1813170):(2,""),
 ("mix2_coop_horror",1889640):(2,""),("mix2_coop_horror",2232600):(2,""),("mix2_coop_horror",2275020):(2,""),
 ("mix2_coop_horror",2569760):(3,""),("mix2_coop_horror",2714970):(2,""),("mix2_coop_horror",2790330):(2,""),
 ("mix2_coop_horror",2802820):(2,""),("mix2_coop_horror",2808570):(2,""),("mix2_coop_horror",2840210):(3,""),
 ("mix2_coop_horror",3121110):(3,""),("mix2_coop_horror",3288600):(1,"LOW_QUALITY"),
 ("mix2_coop_horror",4121170):(3,""),
 ("mix2_crpg_sandbox",699330):(3,""),("mix2_crpg_sandbox",1466060):(3,""),("mix2_crpg_sandbox",1611910):(2,""),
 ("mix2_party_narrative",1195420):(2,""),
 ("mix2_puzzle_survival",2868840):(3,""),
 ("mix2_soulslike_narrative",249230):(2,""),("mix2_soulslike_narrative",268670):(1,"LOW_QUALITY"),
 ("mix2_soulslike_narrative",306670):(1,"LOW_QUALITY"),("mix2_soulslike_narrative",518790):(1,"GENRE_ONLY"),
 ("mix2_soulslike_narrative",529180):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",1071290):(1,"LOW_QUALITY"),
 ("mix2_soulslike_narrative",1206600):(1,"GENRE_ONLY"),("mix2_soulslike_narrative",1466060):(3,""),
 ("mix2_soulslike_narrative",1482320):(1,"TOO_NICHE"),("mix2_soulslike_narrative",1745510):(3,""),
 ("mix2_soulslike_narrative",2191500):(1,"GENRE_ONLY"),
 ("niche_cozy_casual",512740):(2,""),("niche_cozy_casual",732430):(3,""),("niche_cozy_casual",757860):(2,""),
 ("niche_cozy_casual",1587500):(2,""),("niche_cozy_casual",1804450):(2,""),("niche_cozy_casual",2389440):(3,""),
 ("niche_cozy_casual",2396090):(2,""),("niche_cozy_casual",2887660):(2,""),
 ("niche_roguelite",330270):(2,""),("niche_roguelite",986040):(2,""),("niche_roguelite",1074610):(2,""),
 ("niche_roguelite",1243890):(2,""),("niche_roguelite",2066020):(2,""),("niche_roguelite",3216790):(2,""),
}



# 홀드아웃 7프로필 × k=100 = 700쌍. 1회성 일반화 검증 — 이 판정으로 어떤 튜닝도 하지 않는다.
# ho_contradictory 기준: 시드 간 합의가 없으므로 최소 한 시드 취향에 강하게 부합하면 타당.
BLIND_HOLDOUT = {
    ("ho_jrpg", 991270): (3, ""),
    ("ho_jrpg", 1062040): (2, ""),
    ("ho_jrpg", 2410170): (2, ""),
    ("ho_jrpg", 1562940): (3, ""),
    ("ho_jrpg", 207350): (3, ""),
    ("ho_jrpg", 367500): (2, ""),
    ("ho_jrpg", 1969060): (2, ""),
    ("ho_jrpg", 809890): (2, ""),
    ("ho_jrpg", 219780): (1, ""),
    ("ho_jrpg", 280140): (2, ""),
    ("ho_jrpg", 429660): (3, ""),
    ("ho_jrpg", 1845910): (2, ""),
    ("ho_jrpg", 1198090): (3, ""),
    ("ho_jrpg", 407230): (2, ""),
    ("ho_jrpg", 1139160): (2, ""),
    ("ho_jrpg", 2844850): (3, ""),
    ("ho_jrpg", 1382330): (3, ""),
    ("ho_jrpg", 3508730): (2, ""),
    ("ho_jrpg", 552700): (3, ""),
    ("ho_jrpg", 207320): (3, ""),
    ("ho_jrpg", 2701660): (3, ""),
    ("ho_jrpg", 1668510): (3, ""),
    ("ho_jrpg", 589360): (3, ""),
    ("ho_jrpg", 47810): (2, ""),
    ("ho_jrpg", 4249150): (3, ""),
    ("ho_jrpg", 418190): (2, ""),
    ("ho_jrpg", 2532770): (2, ""),
    ("ho_jrpg", 1539620): (2, ""),
    ("ho_jrpg", 524580): (2, ""),
    ("ho_jrpg", 2701350): (1, ""),
    ("ho_jrpg", 251150): (3, ""),
    ("ho_jrpg", 525240): (2, ""),
    ("ho_jrpg", 1918130): (1, ""),
    ("ho_jrpg", 1027250): (3, ""),
    ("ho_jrpg", 2515020): (3, ""),
    ("ho_jrpg", 1863430): (1, ""),
    ("ho_jrpg", 510540): (2, ""),
    ("ho_jrpg", 384570): (2, ""),
    ("ho_jrpg", 1732340): (2, ""),
    ("ho_jrpg", 990050): (2, ""),
    ("ho_jrpg", 726950): (2, ""),
    ("ho_jrpg", 359870): (3, ""),
    ("ho_jrpg", 1549990): (2, ""),
    ("ho_jrpg", 981750): (2, ""),
    ("ho_jrpg", 3124340): (1, ""),
    ("ho_jrpg", 1150690): (3, ""),
    ("ho_jrpg", 351970): (3, ""),
    ("ho_jrpg", 878090): (2, ""),
    ("ho_jrpg", 1082710): (3, ""),
    ("ho_jrpg", 277470): (2, ""),
    ("ho_jrpg", 283270): (1, ""),
    ("ho_jrpg", 1266220): (2, ""),
    ("ho_jrpg", 2490990): (3, ""),
    ("ho_jrpg", 2367710): (2, ""),
    ("ho_jrpg", 1668520): (3, ""),
    ("ho_jrpg", 1958220): (3, ""),
    ("ho_jrpg", 321800): (1, ""),
    ("ho_jrpg", 984560): (2, ""),
    ("ho_jrpg", 878670): (2, ""),
    ("ho_jrpg", 2769210): (2, ""),
    ("ho_jrpg", 1551520): (2, ""),
    ("ho_jrpg", 399790): (2, ""),
    ("ho_jrpg", 2856450): (1, ""),
    ("ho_jrpg", 1331210): (2, ""),
    ("ho_jrpg", 1105500): (3, ""),
    ("ho_jrpg", 1640): (1, ""),
    ("ho_jrpg", 377840): (3, ""),
    ("ho_jrpg", 1685260): (2, ""),
    ("ho_jrpg", 33670): (1, ""),
    ("ho_jrpg", 854940): (2, ""),
    ("ho_jrpg", 854080): (2, ""),
    ("ho_jrpg", 3155540): (2, ""),
    ("ho_jrpg", 921570): (3, ""),
    ("ho_jrpg", 1456650): (1, ""),
    ("ho_jrpg", 896960): (1, ""),
    ("ho_jrpg", 613830): (3, ""),
    ("ho_jrpg", 3575960): (2, ""),
    ("ho_jrpg", 387290): (1, ""),
    ("ho_jrpg", 1072300): (2, ""),
    ("ho_jrpg", 626690): (2, ""),
    ("ho_jrpg", 1773540): (2, ""),
    ("ho_jrpg", 740550): (2, ""),
    ("ho_jrpg", 1895810): (2, ""),
    ("ho_jrpg", 2244210): (2, ""),
    ("ho_jrpg", 1042550): (3, ""),
    ("ho_jrpg", 3908250): (1, ""),
    ("ho_jrpg", 2246670): (3, ""),
    ("ho_jrpg", 1446650): (3, ""),
    ("ho_jrpg", 3929630): (1, ""),
    ("ho_jrpg", 1770400): (1, ""),
    ("ho_jrpg", 1875830): (3, ""),
    ("ho_jrpg", 2175540): (3, ""),
    ("ho_jrpg", 1551510): (1, ""),
    ("ho_jrpg", 2445990): (3, ""),
    ("ho_jrpg", 257350): (2, ""),
    ("ho_jrpg", 1783360): (3, ""),
    ("ho_jrpg", 1372000): (2, ""),
    ("ho_jrpg", 3140180): (2, ""),
    ("ho_jrpg", 995070): (2, ""),
    ("ho_jrpg", 2161700): (3, ""),
    ("ho_horror_solo", 921780): (2, ""),
    ("ho_horror_solo", 2506110): (2, ""),
    ("ho_horror_solo", 57300): (3, ""),
    ("ho_horror_solo", 636520): (2, ""),
    ("ho_horror_solo", 2796460): (2, ""),
    ("ho_horror_solo", 414700): (3, ""),
    ("ho_horror_solo", 1251300): (2, ""),
    ("ho_horror_solo", 1234430): (3, ""),
    ("ho_horror_solo", 3735770): (2, ""),
    ("ho_horror_solo", 537430): (1, ""),
    ("ho_horror_solo", 1412000): (1, ""),
    ("ho_horror_solo", 418950): (2, ""),
    ("ho_horror_solo", 3916260): (2, ""),
    ("ho_horror_solo", 3799320): (2, ""),
    ("ho_horror_solo", 1819180): (2, ""),
    ("ho_horror_solo", 871810): (1, ""),
    ("ho_horror_solo", 282140): (3, ""),
    ("ho_horror_solo", 2026880): (2, ""),
    ("ho_horror_solo", 2412490): (2, ""),
    ("ho_horror_solo", 1092530): (2, ""),
    ("ho_horror_solo", 2180310): (1, ""),
    ("ho_horror_solo", 2601030): (2, ""),
    ("ho_horror_solo", 1437050): (2, ""),
    ("ho_horror_solo", 2338120): (2, ""),
    ("ho_horror_solo", 2631880): (2, ""),
    ("ho_horror_solo", 1177660): (2, ""),
    ("ho_horror_solo", 2582100): (2, ""),
    ("ho_horror_solo", 792300): (3, ""),
    ("ho_horror_solo", 1905020): (2, ""),
    ("ho_horror_solo", 3723450): (1, ""),
    ("ho_horror_solo", 3706160): (1, ""),
    ("ho_horror_solo", 252330): (3, ""),
    ("ho_horror_solo", 384110): (2, ""),
    ("ho_horror_solo", 2314180): (2, ""),
    ("ho_horror_solo", 248470): (2, ""),
    ("ho_horror_solo", 3099670): (2, ""),
    ("ho_horror_solo", 239200): (3, ""),
    ("ho_horror_solo", 733800): (2, ""),
    ("ho_horror_solo", 4115350): (1, ""),
    ("ho_horror_solo", 3461060): (2, ""),
    ("ho_horror_solo", 22120): (3, ""),
    ("ho_horror_solo", 1884800): (2, ""),
    ("ho_horror_solo", 1092660): (3, ""),
    ("ho_horror_solo", 3919190): (2, ""),
    ("ho_horror_solo", 1168690): (2, ""),
    ("ho_horror_solo", 1275450): (1, ""),
    ("ho_horror_solo", 1096570): (3, ""),
    ("ho_horror_solo", 2482650): (2, ""),
    ("ho_horror_solo", 3319120): (2, ""),
    ("ho_horror_solo", 1080130): (1, ""),
    ("ho_horror_solo", 22180): (3, ""),
    ("ho_horror_solo", 3979350): (2, ""),
    ("ho_horror_solo", 3906770): (1, ""),
    ("ho_horror_solo", 4097370): (2, ""),
    ("ho_horror_solo", 1517340): (2, ""),
    ("ho_horror_solo", 1926620): (2, ""),
    ("ho_horror_solo", 4043280): (2, ""),
    ("ho_horror_solo", 2257460): (2, ""),
    ("ho_horror_solo", 1128140): (2, ""),
    ("ho_horror_solo", 786910): (2, ""),
    ("ho_horror_solo", 1946700): (3, ""),
    ("ho_horror_solo", 2233120): (3, ""),
    ("ho_horror_solo", 1697720): (2, ""),
    ("ho_horror_solo", 453670): (2, ""),
    ("ho_horror_solo", 409320): (2, ""),
    ("ho_horror_solo", 1444170): (2, ""),
    ("ho_horror_solo", 3916390): (2, ""),
    ("ho_horror_solo", 1825390): (2, ""),
    ("ho_horror_solo", 1357870): (2, ""),
    ("ho_horror_solo", 3211490): (2, ""),
    ("ho_horror_solo", 4589210): (2, ""),
    ("ho_horror_solo", 1319570): (1, ""),
    ("ho_horror_solo", 2419900): (2, ""),
    ("ho_horror_solo", 3353810): (2, ""),
    ("ho_horror_solo", 332950): (2, ""),
    ("ho_horror_solo", 1368400): (2, ""),
    ("ho_horror_solo", 284990): (2, ""),
    ("ho_horror_solo", 4519130): (2, ""),
    ("ho_horror_solo", 692100): (3, ""),
    ("ho_horror_solo", 3271870): (2, ""),
    ("ho_horror_solo", 339230): (2, ""),
    ("ho_horror_solo", 1729740): (3, ""),
    ("ho_horror_solo", 4139810): (1, ""),
    ("ho_horror_solo", 269790): (3, ""),
    ("ho_horror_solo", 4181410): (2, ""),
    ("ho_horror_solo", 4256690): (2, ""),
    ("ho_horror_solo", 1693070): (2, ""),
    ("ho_horror_solo", 3351140): (2, ""),
    ("ho_horror_solo", 2820890): (1, ""),
    ("ho_horror_solo", 218640): (2, ""),
    ("ho_horror_solo", 1670870): (3, ""),
    ("ho_horror_solo", 4616930): (2, ""),
    ("ho_horror_solo", 337880): (2, ""),
    ("ho_horror_solo", 2311190): (1, ""),
    ("ho_horror_solo", 10680): (1, ""),
    ("ho_horror_solo", 594160): (1, ""),
    ("ho_horror_solo", 253330): (3, ""),
    ("ho_horror_solo", 2593650): (2, ""),
    ("ho_horror_solo", 1311670): (1, ""),
    ("ho_horror_solo", 1805040): (2, ""),
    ("ho_mmo", 39120): (3, ""),
    ("ho_mmo", 716350): (2, ""),
    ("ho_mmo", 761890): (3, ""),
    ("ho_mmo", 1821680): (2, ""),
    ("ho_mmo", 1286830): (3, ""),
    ("ho_mmo", 22330): (2, ""),
    ("ho_mmo", 1995520): (3, ""),
    ("ho_mmo", 249550): (2, ""),
    ("ho_mmo", 9900): (2, ""),
    ("ho_mmo", 3767850): (3, ""),
    ("ho_mmo", 22320): (2, ""),
    ("ho_mmo", 3929630): (1, ""),
    ("ho_mmo", 2344320): (3, ""),
    ("ho_mmo", 1189290): (3, ""),
    ("ho_mmo", 2265990): (1, ""),
    ("ho_mmo", 1170950): (3, ""),
    ("ho_mmo", 1114220): (2, ""),
    ("ho_mmo", 2352620): (3, ""),
    ("ho_mmo", 1284210): (3, ""),
    ("ho_mmo", 2382520): (3, ""),
    ("ho_mmo", 228280): (1, ""),
    ("ho_mmo", 834910): (2, ""),
    ("ho_mmo", 489830): (2, ""),
    ("ho_mmo", 109600): (3, ""),
    ("ho_mmo", 1771300): (2, ""),
    ("ho_mmo", 2506440): (3, ""),
    ("ho_mmo", 2414270): (1, ""),
    ("ho_mmo", 487120): (2, ""),
    ("ho_mmo", 1161830): (1, ""),
    ("ho_mmo", 1134700): (3, ""),
    ("ho_mmo", 3074020): (2, ""),
    ("ho_mmo", 2623190): (2, ""),
    ("ho_mmo", 3336530): (3, ""),
    ("ho_mmo", 908060): (1, ""),
    ("ho_mmo", 3672400): (3, ""),
    ("ho_mmo", 365360): (1, ""),
    ("ho_mmo", 215280): (3, ""),
    ("ho_mmo", 1887020): (1, ""),
    ("ho_mmo", 2429640): (3, ""),
    ("ho_mmo", 699330): (1, ""),
    ("ho_mmo", 626690): (2, ""),
    ("ho_mmo", 2438330): (1, ""),
    ("ho_mmo", 670260): (2, ""),
    ("ho_mmo", 2230980): (1, ""),
    ("ho_mmo", 2321880): (1, ""),
    ("ho_mmo", 1669000): (1, ""),
    ("ho_mmo", 1245620): (1, ""),
    ("ho_mmo", 1247100): (1, ""),
    ("ho_mmo", 3997970): (3, ""),
    ("ho_mmo", 720620): (1, ""),
    ("ho_mmo", 61500): (1, ""),
    ("ho_mmo", 704450): (2, ""),
    ("ho_mmo", 2622380): (2, ""),
    ("ho_mmo", 2294660): (2, ""),
    ("ho_mmo", 1159090): (1, ""),
    ("ho_mmo", 3107230): (3, ""),
    ("ho_mmo", 1527950): (1, ""),
    ("ho_mmo", 230230): (1, ""),
    ("ho_mmo", 1100260): (1, ""),
    ("ho_mmo", 1461810): (2, ""),
    ("ho_mmo", 1088090): (2, ""),
    ("ho_mmo", 4800): (1, ""),
    ("ho_mmo", 858820): (2, ""),
    ("ho_mmo", 496730): (1, ""),
    ("ho_mmo", 311290): (1, ""),
    ("ho_mmo", 594570): (1, ""),
    ("ho_mmo", 435150): (2, ""),
    ("ho_mmo", 1598730): (2, ""),
    ("ho_mmo", 1588560): (2, ""),
    ("ho_mmo", 1025600): (1, ""),
    ("ho_mmo", 326160): (2, ""),
    ("ho_mmo", 1154040): (1, ""),
    ("ho_mmo", 492150): (1, ""),
    ("ho_mmo", 228260): (1, ""),
    ("ho_mmo", 2679100): (2, ""),
    ("ho_mmo", 291650): (1, ""),
    ("ho_mmo", 240620): (2, ""),
    ("ho_mmo", 1930): (1, ""),
    ("ho_mmo", 2241380): (2, ""),
    ("ho_mmo", 655780): (2, ""),
    ("ho_mmo", 810040): (2, ""),
    ("ho_mmo", 65530): (1, ""),
    ("ho_mmo", 3098140): (1, ""),
    ("ho_mmo", 4124950): (3, ""),
    ("ho_mmo", 2738630): (2, ""),
    ("ho_mmo", 200710): (2, ""),
    ("ho_mmo", 418180): (2, ""),
    ("ho_mmo", 1656930): (2, ""),
    ("ho_mmo", 39160): (1, ""),
    ("ho_mmo", 202710): (1, ""),
    ("ho_mmo", 3234720): (3, ""),
    ("ho_mmo", 246760): (1, ""),
    ("ho_mmo", 1804470): (2, ""),
    ("ho_mmo", 280520): (1, ""),
    ("ho_mmo", 774241): (2, ""),
    ("ho_mmo", 261550): (2, ""),
    ("ho_mmo", 2853730): (2, ""),
    ("ho_mmo", 1096530): (1, ""),
    ("ho_mmo", 1676380): (2, ""),
    ("ho_mmo", 1111930): (1, ""),
    ("ho_sports", 1951410): (3, ""),
    ("ho_sports", 3472040): (3, ""),
    ("ho_sports", 1506830): (3, ""),
    ("ho_sports", 1196470): (3, ""),
    ("ho_sports", 1875310): (2, ""),
    ("ho_sports", 1665460): (2, ""),
    ("ho_sports", 1252000): (2, ""),
    ("ho_sports", 2453660): (3, ""),
    ("ho_sports", 3405690): (3, ""),
    ("ho_sports", 485610): (2, ""),
    ("ho_sports", 3059520): (2, ""),
    ("ho_sports", 1785650): (2, ""),
    ("ho_sports", 3468640): (2, ""),
    ("ho_sports", 2742550): (1, ""),
    ("ho_sports", 1461100): (2, ""),
    ("ho_sports", 324090): (1, ""),
    ("ho_sports", 280830): (1, ""),
    ("ho_sports", 2385530): (2, ""),
    ("ho_sports", 932980): (2, ""),
    ("ho_sports", 2138720): (3, ""),
    ("ho_sports", 3077390): (2, ""),
    ("ho_sports", 891020): (2, ""),
    ("ho_sports", 2098130): (2, ""),
    ("ho_sports", 4415290): (2, ""),
    ("ho_sports", 554750): (2, ""),
    ("ho_sports", 3127230): (2, ""),
    ("ho_sports", 1620540): (2, ""),
    ("ho_sports", 2185930): (2, ""),
    ("ho_sports", 988910): (2, ""),
    ("ho_sports", 4154410): (1, ""),
    ("ho_sports", 407190): (1, ""),
    ("ho_sports", 2337630): (2, ""),
    ("ho_sports", 2384430): (2, ""),
    ("ho_sports", 2422110): (2, ""),
    ("ho_sports", 299970): (1, ""),
    ("ho_sports", 4653510): (2, ""),
    ("ho_sports", 3465170): (2, ""),
    ("ho_sports", 1415920): (2, ""),
    ("ho_sports", 270450): (1, ""),
    ("ho_sports", 3217240): (2, ""),
    ("ho_sports", 1701380): (2, ""),
    ("ho_sports", 310950): (1, ""),
    ("ho_sports", 1698080): (2, ""),
    ("ho_sports", 1702130): (1, ""),
    ("ho_sports", 3621200): (1, ""),
    ("ho_sports", 1412780): (2, ""),
    ("ho_sports", 1126990): (2, ""),
    ("ho_sports", 730): (0, ""),
    ("ho_sports", 1453850): (2, ""),
    ("ho_sports", 3907880): (2, ""),
    ("ho_sports", 45760): (1, ""),
    ("ho_sports", 2210520): (2, ""),
    ("ho_sports", 314070): (2, ""),
    ("ho_sports", 1913210): (2, ""),
    ("ho_sports", 687850): (2, ""),
    ("ho_sports", 2591310): (2, ""),
    ("ho_sports", 212480): (1, ""),
    ("ho_sports", 1281170): (1, ""),
    ("ho_sports", 1595050): (2, ""),
    ("ho_sports", 505740): (0, ""),
    ("ho_sports", 1005530): (2, ""),
    ("ho_sports", 745530): (2, ""),
    ("ho_sports", 2488620): (2, ""),
    ("ho_sports", 1488560): (2, ""),
    ("ho_sports", 464850): (2, ""),
    ("ho_sports", 2824810): (1, ""),
    ("ho_sports", 3137350): (2, ""),
    ("ho_sports", 3873970): (2, ""),
    ("ho_sports", 240): (0, ""),
    ("ho_sports", 703940): (2, ""),
    ("ho_sports", 971900): (2, ""),
    ("ho_sports", 209120): (1, ""),
    ("ho_sports", 3284110): (1, ""),
    ("ho_sports", 1214520): (1, ""),
    ("ho_sports", 222880): (0, ""),
    ("ho_sports", 1923690): (2, ""),
    ("ho_sports", 518040): (2, ""),
    ("ho_sports", 2103640): (2, ""),
    ("ho_sports", 4054680): (1, ""),
    ("ho_sports", 2935400): (2, ""),
    ("ho_sports", 1178400): (2, ""),
    ("ho_sports", 3585230): (2, ""),
    ("ho_sports", 3219190): (2, ""),
    ("ho_sports", 3514580): (1, ""),
    ("ho_sports", 1748900): (1, ""),
    ("ho_sports", 2574120): (2, ""),
    ("ho_sports", 946880): (2, ""),
    ("ho_sports", 3052520): (1, ""),
    ("ho_sports", 4032350): (2, ""),
    ("ho_sports", 2215910): (2, ""),
    ("ho_sports", 1641830): (2, ""),
    ("ho_sports", 1609870): (2, ""),
    ("ho_sports", 3230400): (2, ""),
    ("ho_sports", 547900): (2, ""),
    ("ho_sports", 3499820): (1, ""),
    ("ho_sports", 883130): (3, ""),
    ("ho_sports", 1360040): (1, ""),
    ("ho_sports", 230650): (2, ""),
    ("ho_sports", 3486980): (2, ""),
    ("ho_sports", 3795200): (1, ""),
    ("ho_rhythm", 2717010): (3, ""),
    ("ho_rhythm", 3077570): (2, ""),
    ("ho_rhythm", 1477590): (3, ""),
    ("ho_rhythm", 2949320): (3, ""),
    ("ho_rhythm", 1761390): (3, ""),
    ("ho_rhythm", 938220): (2, ""),
    ("ho_rhythm", 4737700): (2, ""),
    ("ho_rhythm", 3115500): (1, ""),
    ("ho_rhythm", 3006240): (2, ""),
    ("ho_rhythm", 2250500): (3, ""),
    ("ho_rhythm", 406940): (2, ""),
    ("ho_rhythm", 2756930): (3, ""),
    ("ho_rhythm", 3921360): (2, ""),
    ("ho_rhythm", 656600): (2, ""),
    ("ho_rhythm", 4282500): (2, ""),
    ("ho_rhythm", 1947940): (1, ""),
    ("ho_rhythm", 372690): (2, ""),
    ("ho_rhythm", 744060): (3, ""),
    ("ho_rhythm", 3988570): (2, ""),
    ("ho_rhythm", 1064610): (2, ""),
    ("ho_rhythm", 4142580): (2, ""),
    ("ho_rhythm", 2073250): (3, ""),
    ("ho_rhythm", 2951580): (2, ""),
    ("ho_rhythm", 370460): (1, ""),
    ("ho_rhythm", 2263360): (2, ""),
    ("ho_rhythm", 2928330): (2, ""),
    ("ho_rhythm", 3666470): (2, ""),
    ("ho_rhythm", 2585040): (2, ""),
    ("ho_rhythm", 368570): (2, ""),
    ("ho_rhythm", 1683200): (2, ""),
    ("ho_rhythm", 893030): (2, ""),
    ("ho_rhythm", 2295790): (1, ""),
    ("ho_rhythm", 1802720): (3, ""),
    ("ho_rhythm", 2359170): (2, ""),
    ("ho_rhythm", 1945220): (2, ""),
    ("ho_rhythm", 40800): (1, ""),
    ("ho_rhythm", 977950): (3, ""),
    ("ho_rhythm", 4146170): (2, ""),
    ("ho_rhythm", 3446490): (2, ""),
    ("ho_rhythm", 2978440): (2, ""),
    ("ho_rhythm", 357720): (2, ""),
    ("ho_rhythm", 930620): (2, ""),
    ("ho_rhythm", 554800): (1, ""),
    ("ho_rhythm", 1216230): (3, ""),
    ("ho_rhythm", 1282210): (3, ""),
    ("ho_rhythm", 1998530): (2, ""),
    ("ho_rhythm", 49600): (2, ""),
    ("ho_rhythm", 1233260): (0, ""),
    ("ho_rhythm", 2758510): (2, ""),
    ("ho_rhythm", 1345820): (3, ""),
    ("ho_rhythm", 3998760): (1, ""),
    ("ho_rhythm", 322170): (3, ""),
    ("ho_rhythm", 2442660): (2, ""),
    ("ho_rhythm", 1384160): (1, ""),
    ("ho_rhythm", 1563310): (1, ""),
    ("ho_rhythm", 2608510): (2, ""),
    ("ho_rhythm", 3500840): (2, ""),
    ("ho_rhythm", 2685840): (2, ""),
    ("ho_rhythm", 3705340): (2, ""),
    ("ho_rhythm", 1126750): (1, ""),
    ("ho_rhythm", 4013260): (2, ""),
    ("ho_rhythm", 2005050): (2, ""),
    ("ho_rhythm", 511680): (0, ""),
    ("ho_rhythm", 1546100): (3, ""),
    ("ho_rhythm", 1829470): (2, ""),
    ("ho_rhythm", 986800): (3, ""),
    ("ho_rhythm", 531510): (3, ""),
    ("ho_rhythm", 3516620): (2, ""),
    ("ho_rhythm", 2058180): (0, ""),
    ("ho_rhythm", 1396140): (2, ""),
    ("ho_rhythm", 3803270): (2, ""),
    ("ho_rhythm", 326340): (0, ""),
    ("ho_rhythm", 513510): (2, ""),
    ("ho_rhythm", 2704460): (1, ""),
    ("ho_rhythm", 411370): (1, ""),
    ("ho_rhythm", 3233820): (2, ""),
    ("ho_rhythm", 618740): (2, ""),
    ("ho_rhythm", 3717340): (0, ""),
    ("ho_rhythm", 2738520): (2, ""),
    ("ho_rhythm", 2736580): (2, ""),
    ("ho_rhythm", 452510): (1, ""),
    ("ho_rhythm", 3656360): (2, ""),
    ("ho_rhythm", 2153280): (2, ""),
    ("ho_rhythm", 45760): (1, ""),
    ("ho_rhythm", 671370): (2, ""),
    ("ho_rhythm", 661000): (2, ""),
    ("ho_rhythm", 877850): (2, ""),
    ("ho_rhythm", 2057840): (2, ""),
    ("ho_rhythm", 1797480): (2, ""),
    ("ho_rhythm", 221680): (2, ""),
    ("ho_rhythm", 1058830): (3, ""),
    ("ho_rhythm", 3463900): (2, ""),
    ("ho_rhythm", 251630): (2, ""),
    ("ho_rhythm", 416790): (2, ""),
    ("ho_rhythm", 4258870): (2, ""),
    ("ho_rhythm", 921590): (1, ""),
    ("ho_rhythm", 4123750): (2, ""),
    ("ho_rhythm", 1475840): (3, ""),
    ("ho_rhythm", 711210): (1, ""),
    ("ho_rhythm", 2216360): (2, ""),
    ("ho_contradictory", 2311760): (3, ""),
    ("ho_contradictory", 2456740): (3, ""),
    ("ho_contradictory", 379720): (3, ""),
    ("ho_contradictory", 1962663): (2, ""),
    ("ho_contradictory", 3710840): (3, ""),
    ("ho_contradictory", 2212670): (3, ""),
    ("ho_contradictory", 214870): (3, ""),
    ("ho_contradictory", 2280): (3, ""),
    ("ho_contradictory", 2616140): (3, ""),
    ("ho_contradictory", 47890): (3, ""),
    ("ho_contradictory", 2300120): (3, ""),
    ("ho_contradictory", 3595230): (2, ""),
    ("ho_contradictory", 4025700): (2, ""),
    ("ho_contradictory", 1214470): (3, ""),
    ("ho_contradictory", 1098293): (2, ""),
    ("ho_contradictory", 476600): (2, ""),
    ("ho_contradictory", 2834600): (2, ""),
    ("ho_contradictory", 1591900): (2, ""),
    ("ho_contradictory", 221100): (1, ""),
    ("ho_contradictory", 1384160): (1, ""),
    ("ho_contradictory", 3314070): (3, ""),
    ("ho_contradictory", 3326230): (3, ""),
    ("ho_contradictory", 3017860): (3, ""),
    ("ho_contradictory", 45760): (1, ""),
    ("ho_contradictory", 1651490): (3, ""),
    ("ho_contradictory", 1120320): (3, ""),
    ("ho_contradictory", 3850560): (0, ""),
    ("ho_contradictory", 730): (2, ""),
    ("ho_contradictory", 1405810): (2, ""),
    ("ho_contradictory", 2185060): (2, ""),
    ("ho_contradictory", 1098291): (2, ""),
    ("ho_contradictory", 40): (2, ""),
    ("ho_contradictory", 1118520): (3, ""),
    ("ho_contradictory", 1389840): (2, ""),
    ("ho_contradictory", 1250): (2, ""),
    ("ho_contradictory", 209120): (1, ""),
    ("ho_contradictory", 1954160): (2, ""),
    ("ho_contradictory", 944250): (3, ""),
    ("ho_contradictory", 2941710): (2, ""),
    ("ho_contradictory", 2157560): (1, ""),
    ("ho_contradictory", 1244630): (3, ""),
    ("ho_contradictory", 2726460): (2, ""),
    ("ho_contradictory", 381210): (1, ""),
    ("ho_contradictory", 320): (2, ""),
    ("ho_contradictory", 1429880): (3, ""),
    ("ho_contradictory", 1190970): (3, ""),
    ("ho_contradictory", 2406950): (1, ""),
    ("ho_contradictory", 584400): (0, ""),
    ("ho_contradictory", 978780): (2, ""),
    ("ho_contradictory", 4599650): (2, ""),
    ("ho_contradictory", 985830): (2, ""),
    ("ho_contradictory", 470210): (1, ""),
    ("ho_contradictory", 1296400): (3, ""),
    ("ho_contradictory", 2844910): (2, ""),
    ("ho_contradictory", 1765300): (2, ""),
    ("ho_contradictory", 204300): (3, ""),
    ("ho_contradictory", 2742710): (3, ""),
    ("ho_contradictory", 3086630): (2, ""),
    ("ho_contradictory", 42700): (2, ""),
    ("ho_contradictory", 577940): (1, ""),
    ("ho_contradictory", 1125890): (2, ""),
    ("ho_contradictory", 1602000): (2, ""),
    ("ho_contradictory", 3517910): (1, ""),
    ("ho_contradictory", 348550): (1, ""),
    ("ho_contradictory", 1009560): (3, ""),
    ("ho_contradictory", 2487150): (2, ""),
    ("ho_contradictory", 285190): (1, ""),
    ("ho_contradictory", 1522820): (2, ""),
    ("ho_contradictory", 1493760): (2, ""),
    ("ho_contradictory", 1510580): (2, ""),
    ("ho_contradictory", 709840): (1, ""),
    ("ho_contradictory", 336420): (2, ""),
    ("ho_contradictory", 2400550): (2, ""),
    ("ho_contradictory", 3339330): (2, ""),
    ("ho_contradictory", 2027330): (2, ""),
    ("ho_contradictory", 2437170): (3, ""),
    ("ho_contradictory", 2508780): (2, ""),
    ("ho_contradictory", 1080020): (2, ""),
    ("ho_contradictory", 10680): (2, ""),
    ("ho_contradictory", 310950): (1, ""),
    ("ho_contradictory", 3571710): (2, ""),
    ("ho_contradictory", 2510970): (2, ""),
    ("ho_contradictory", 1583230): (2, ""),
    ("ho_contradictory", 2075050): (1, ""),
    ("ho_contradictory", 1151380): (2, ""),
    ("ho_contradictory", 1937500): (2, ""),
    ("ho_contradictory", 2210): (2, ""),
    ("ho_contradictory", 2344520): (1, ""),
    ("ho_contradictory", 3314060): (3, ""),
    ("ho_contradictory", 1703180): (2, ""),
    ("ho_contradictory", 1238820): (2, ""),
    ("ho_contradictory", 1361210): (2, ""),
    ("ho_contradictory", 1084600): (2, ""),
    ("ho_contradictory", 3017730): (3, ""),
    ("ho_contradictory", 1098292): (2, ""),
    ("ho_contradictory", 291550): (1, ""),
    ("ho_contradictory", 2698780): (2, ""),
    ("ho_contradictory", 1138640): (3, ""),
    ("ho_contradictory", 550010): (2, ""),
    ("ho_contradictory", 601050): (1, ""),
    ("ho_biglib", 238010): (3, ""),
    ("ho_biglib", 287390): (3, ""),
    ("ho_biglib", 614570): (3, ""),
    ("ho_biglib", 329830): (2, ""),
    ("ho_biglib", 537430): (1, ""),
    ("ho_biglib", 565120): (3, ""),
    ("ho_biglib", 460810): (2, ""),
    ("ho_biglib", 1681430): (3, ""),
    ("ho_biglib", 1577250): (2, ""),
    ("ho_biglib", 301280): (3, ""),
    ("ho_biglib", 1509960): (1, ""),
    ("ho_biglib", 1244950): (2, ""),
    ("ho_biglib", 403640): (3, ""),
    ("ho_biglib", 57300): (3, ""),
    ("ho_biglib", 1319420): (2, ""),
    ("ho_biglib", 6920): (2, ""),
    ("ho_biglib", 1692240): (3, ""),
    ("ho_biglib", 286690): (3, ""),
    ("ho_biglib", 1670780): (2, ""),
    ("ho_biglib", 612880): (3, ""),
    ("ho_biglib", 2210): (2, ""),
    ("ho_biglib", 1341050): (1, ""),
    ("ho_biglib", 1347780): (2, ""),
    ("ho_biglib", 2444350): (1, ""),
    ("ho_biglib", 244910): (2, ""),
    ("ho_biglib", 1193990): (1, ""),
    ("ho_biglib", 1184790): (2, ""),
    ("ho_biglib", 973580): (2, ""),
    ("ho_biglib", 1363360): (2, ""),
    ("ho_biglib", 258180): (2, ""),
    ("ho_biglib", 203140): (3, ""),
    ("ho_biglib", 893690): (2, ""),
    ("ho_biglib", 1170760): (2, ""),
    ("ho_biglib", 214490): (3, ""),
    ("ho_biglib", 10680): (2, ""),
    ("ho_biglib", 3823950): (2, ""),
    ("ho_biglib", 6910): (3, ""),
    ("ho_biglib", 6000): (2, ""),
    ("ho_biglib", 3920610): (2, ""),
    ("ho_biglib", 1874190): (1, ""),
    ("ho_biglib", 3527760): (2, ""),
    ("ho_biglib", 202750): (3, ""),
    ("ho_biglib", 3621700): (1, ""),
    ("ho_biglib", 3710): (2, ""),
    ("ho_biglib", 239200): (3, ""),
    ("ho_biglib", 17470): (3, ""),
    ("ho_biglib", 2506160): (2, ""),
    ("ho_biglib", 70): (3, ""),
    ("ho_biglib", 310740): (2, ""),
    ("ho_biglib", 1715130): (2, ""),
    ("ho_biglib", 2321780): (1, ""),
    ("ho_biglib", 6850): (2, ""),
    ("ho_biglib", 11550): (2, ""),
    ("ho_biglib", 3263320): (1, ""),
    ("ho_biglib", 2124100): (2, ""),
    ("ho_biglib", 409320): (2, ""),
    ("ho_biglib", 444580): (2, ""),
    ("ho_biglib", 482300): (1, ""),
    ("ho_biglib", 40): (1, ""),
    ("ho_biglib", 7010): (2, ""),
    ("ho_biglib", 1343520): (1, ""),
    ("ho_biglib", 13570): (3, ""),
    ("ho_biglib", 1607680): (2, ""),
    ("ho_biglib", 50): (3, ""),
    ("ho_biglib", 2419090): (2, ""),
    ("ho_biglib", 411330): (2, ""),
    ("ho_biglib", 1905020): (2, ""),
    ("ho_biglib", 365160): (3, ""),
    ("ho_biglib", 2446540): (2, ""),
    ("ho_biglib", 379720): (3, ""),
    ("ho_biglib", 32660): (1, ""),
    ("ho_biglib", 3626760): (2, ""),
    ("ho_biglib", 1316680): (2, ""),
    ("ho_biglib", 390520): (0, ""),
    ("ho_biglib", 281410): (1, ""),
    ("ho_biglib", 2503690): (2, ""),
    ("ho_biglib", 20550): (2, ""),
    ("ho_biglib", 402020): (2, ""),
    ("ho_biglib", 251110): (3, ""),
    ("ho_biglib", 261530): (2, ""),
    ("ho_biglib", 1150760): (3, ""),
    ("ho_biglib", 393080): (2, ""),
    ("ho_biglib", 3105890): (2, ""),
    ("ho_biglib", 211600): (3, ""),
    ("ho_biglib", 2397250): (1, ""),
    ("ho_biglib", 1554840): (1, ""),
    ("ho_biglib", 2096610): (2, ""),
    ("ho_biglib", 2905090): (1, ""),
    ("ho_biglib", 3342650): (1, ""),
    ("ho_biglib", 258520): (3, ""),
    ("ho_biglib", 2257460): (2, ""),
    ("ho_biglib", 2961530): (2, ""),
    ("ho_biglib", 91700): (2, ""),
    ("ho_biglib", 24240): (1, ""),
    ("ho_biglib", 1693980): (3, ""),
    ("ho_biglib", 1643320): (3, ""),
    ("ho_biglib", 390030): (2, ""),
    ("ho_biglib", 3295360): (1, ""),
    ("ho_biglib", 92000): (2, ""),
    ("ho_biglib", 1300): (2, ""),
}



# 동적 시나리오(좋아요 증가 전이) 판정 122쌍. 문맥은 DYN_SCENARIOS 참고.
# 성장 1·2단계는 dyn_*@n 키(그 시점 시드가 문맥), pivot post 는 base+새시드 4개가 문맥.
BLIND_DYN = {
    ("dyn_grow_cozy@1", 405710): (3, ""),
    ("dyn_grow_cozy@1", 2418520): (2, ""),
    ("dyn_grow_cozy@1", 666140): (3, ""),
    ("dyn_grow_cozy@1", 1245560): (3, ""),
    ("dyn_grow_cozy@1", 1137750): (2, ""),
    ("dyn_grow_cozy@1", 220260): (2, ""),
    ("dyn_grow_cozy@1", 2508780): (3, ""),
    ("dyn_grow_cozy@1", 2678830): (2, ""),
    ("dyn_grow_cozy@1", 2252680): (2, ""),
    ("dyn_grow_cozy@1", 1158160): (3, ""),
    ("dyn_grow_cozy@1", 1248130): (2, ""),
    ("dyn_grow_cozy@1", 673950): (2, ""),
    ("dyn_grow_cozy@1", 758870): (3, ""),
    ("dyn_grow_cozy@1", 1536090): (3, ""),
    ("dyn_grow_cozy@1", 1432860): (3, ""),
    ("dyn_grow_cozy@1", 1329510): (2, ""),
    ("dyn_grow_cozy@1", 840010): (3, ""),
    ("dyn_grow_cozy@1", 678900): (2, ""),
    ("dyn_grow_cozy@1", 3659410): (1, ""),
    ("dyn_grow_cozy@1", 2142790): (3, ""),
    ("dyn_grow_cozy@2", 666140): (3, ""),
    ("dyn_grow_cozy@2", 1657630): (3, ""),
    ("dyn_grow_cozy@2", 1137750): (2, ""),
    ("dyn_grow_cozy@2", 580200): (3, ""),
    ("dyn_grow_cozy@2", 758870): (3, ""),
    ("dyn_grow_cozy@2", 214730): (0, ""),
    ("dyn_grow_cozy@2", 405710): (3, ""),
    ("dyn_grow_cozy@2", 1144770): (1, ""),
    ("dyn_grow_cozy@2", 1245560): (3, ""),
    ("dyn_grow_cozy@2", 2993780): (3, ""),
    ("dyn_grow_cozy@2", 2418520): (2, ""),
    ("dyn_grow_cozy@2", 855740): (1, ""),
    ("dyn_grow_cozy@2", 978780): (3, ""),
    ("dyn_grow_cozy@2", 1740300): (3, ""),
    ("dyn_grow_cozy@2", 2508780): (3, ""),
    ("dyn_grow_cozy@2", 715560): (2, ""),
    ("dyn_grow_cozy@2", 2918500): (2, ""),
    ("dyn_grow_cozy@2", 1018800): (2, ""),
    ("dyn_grow_cozy@2", 1350840): (3, ""),
    ("dyn_grow_cozy@2", 1092590): (1, ""),
    ("dyn_grow_det@1", 31800): (2, ""),
    ("dyn_grow_det@1", 1466390): (3, ""),
    ("dyn_grow_det@1", 350640): (3, ""),
    ("dyn_grow_det@1", 1055850): (2, ""),
    ("dyn_grow_det@1", 286480): (3, ""),
    ("dyn_grow_det@1", 430020): (2, ""),
    ("dyn_grow_det@1", 1689870): (2, ""),
    ("dyn_grow_det@1", 46550): (3, ""),
    ("dyn_grow_det@1", 236930): (3, ""),
    ("dyn_grow_det@1", 935580): (2, ""),
    ("dyn_grow_det@1", 615770): (2, ""),
    ("dyn_grow_det@1", 1677770): (3, ""),
    ("dyn_grow_det@1", 284770): (2, ""),
    ("dyn_grow_det@1", 340020): (2, ""),
    ("dyn_grow_det@1", 1919600): (2, ""),
    ("dyn_grow_det@1", 370910): (3, ""),
    ("dyn_grow_det@1", 94620): (2, ""),
    ("dyn_grow_det@1", 205650): (3, ""),
    ("dyn_grow_det@1", 11150): (3, ""),
    ("dyn_grow_det@1", 233290): (3, ""),
    ("dyn_grow_det@2", 615770): (3, ""),
    ("dyn_grow_det@2", 392970): (2, ""),
    ("dyn_grow_det@2", 1055850): (3, ""),
    ("dyn_grow_det@2", 1140290): (3, ""),
    ("dyn_grow_det@2", 1919600): (3, ""),
    ("dyn_grow_det@2", 2450840): (2, ""),
    ("dyn_grow_det@2", 1466390): (3, ""),
    ("dyn_grow_det@2", 1271300): (2, ""),
    ("dyn_grow_det@2", 3979550): (2, ""),
    ("dyn_grow_det@2", 368370): (3, ""),
    ("dyn_grow_det@2", 31800): (3, ""),
    ("dyn_grow_det@2", 4623310): (1, ""),
    ("dyn_grow_det@2", 350640): (3, ""),
    ("dyn_grow_det@2", 1546920): (3, ""),
    ("dyn_grow_det@2", 1787990): (1, ""),
    ("dyn_grow_det@2", 2875630): (2, ""),
    ("dyn_grow_det@2", 1677770): (3, ""),
    ("dyn_grow_det@2", 2871140): (2, ""),
    ("dyn_grow_det@2", 3148060): (2, ""),
    ("dyn_grow_det@2", 1977220): (2, ""),
    ("dyn_pivot_cozy", 220260): (2, ""),
    ("dyn_pivot_cozy", 892970): (3, ""),
    ("dyn_pivot_cozy", 617290): (3, ""),
    ("dyn_pivot_cozy", 1592110): (2, ""),
    ("dyn_pivot_cozy", 834910): (1, ""),
    ("dyn_pivot_cozy", 858820): (2, ""),
    ("dyn_pivot_cozy", 1599330): (3, ""),
    ("dyn_pivot_cozy", 670260): (2, ""),
    ("dyn_pivot_cozy", 715400): (3, ""),
    ("dyn_pivot_cozy", 1062520): (3, ""),
    ("dyn_pivot_cozy", 2661300): (3, ""),
    ("dyn_pivot_cozy", 1804470): (2, ""),
    ("dyn_pivot_cozy", 1329510): (2, ""),
    ("dyn_pivot_cozy", 3041230): (3, ""),
    ("dyn_pivot_cozy", 1203620): (3, ""),
    ("dyn_pivot_cozy", 2300320): (2, ""),
    ("dyn_pivot_cozy", 1504570): (2, ""),
    ("dyn_pivot_cozy", 2683150): (3, ""),
    ("dyn_pivot_cozy", 794490): (1, ""),
    ("dyn_pivot_cozy", 2679100): (2, ""),
    ("dyn_pivot_tactics", 2870340): (3, ""),
    ("dyn_pivot_tactics", 1909420): (3, ""),
    ("dyn_pivot_tactics", 993690): (2, ""),
    ("dyn_pivot_tactics", 1985420): (3, ""),
    ("dyn_pivot_tactics", 861540): (3, ""),
    ("dyn_pivot_tactics", 370020): (3, ""),
    ("dyn_pivot_tactics", 2244470): (3, ""),
    ("dyn_pivot_tactics", 1302240): (3, ""),
    ("dyn_pivot_tactics", 981430): (2, ""),
    ("dyn_pivot_tactics", 2622820): (2, ""),
    ("dyn_pivot_tactics", 595930): (2, ""),
    ("dyn_pivot_tactics", 2309400): (2, ""),
    ("dyn_pivot_tactics", 2026820): (3, ""),
    ("dyn_pivot_tactics", 2730290): (2, ""),
    ("dyn_pivot_tactics", 1124180): (3, ""),
    ("dyn_pivot_tactics", 3004140): (2, ""),
    ("dyn_pivot_tactics", 2468100): (2, ""),
    ("dyn_pivot_tactics", 1267470): (3, ""),
    ("dyn_pivot_tactics", 955170): (2, ""),
    ("dyn_pivot_tactics", 2960770): (2, ""),
    ("dyn_stable_arpg", 1903340): (3, ""),
    ("dyn_stable_cozy", 2142790): (3, ""),
}



# 시리즈 세션 상한(3) 적용 후 새로 노출된 10쌍.
BLIND_SCAP = {
    ("coh_fps", 1506830): (0, "IRRELEVANT"),     # FIFA 22 — 경쟁 FPS 취향에 축구
    ("coh_fps", 1090630): (1, "GENRE_ONLY"),     # 대전 격투 — 경쟁 PvP 인접일 뿐
    ("coh_strategy", 1679290): (2, ""),           # 빅토리아 시대 턴제 전략
    ("coh_cozy", 678900): (2, ""),                # 농장 생활 RPG
    ("coh_cozy", 2142790): (3, ""),               # Fields of Mistria — 정중앙
    ("coh_cozy", 943260): (3, ""),                # 난파 후 농사+탐험 — Raft+Stardew 가교
    ("coh_vehicle_sim", 1618290): (2, ""),        # 철도 운전 시뮬
    ("coh_classic_multi", 579840): (2, ""),       # 4인 협동 플랫포머
    ("niche_cozy_casual", 530320): (3, ""),       # Wandersong — 감성 모험, 축 정중앙
    ("longtail_detective", 286360): (3, ""),      # 포인트앤클릭 미스터리
}

ALL = {**REP_V2, **POSTPROCESS, **DEPTH_V1, **FULL_V1, **REP_V2_TOPUP, **MIN_REV_300,
       **RANKER_DIAG, **BOOST_SWEEP, **VAL_V1, **DEV_TOPUP_V1, **NICHE_V1,
       **BLIND_ROUND1, **BLIND_TAGS, **BLIND_FLOOR, **BLIND_LOWREV,
       **BLIND_NOTAGS31, **BLIND_TAGSFULL, **BLIND_K30_P1, **BLIND_K30_P2, **BLIND_K30_P3, **BLIND_K50_P4, **BLIND_K50_P5, **BLIND_K50_FILL, **BLIND_STRATEGY_S1, **BLIND_STRATEGY_S2, **BLIND_STRATEGY_S4, **BLIND_STRATEGY_S5, **BLIND_FINAL_F30, **BLIND_FINAL_F50, **BLIND_ADAPT, **BLIND_CONSENSUS, **BLIND_K75, **BLIND_CF300, **BLIND_CBOOST, **BLIND_DDM, **BLIND_HUB05, **BLIND_HUB25, **BLIND_HUBFINE, **BLIND_K100, **BLIND_K100_PVP, **BLIND_SSF, **BLIND_PS10, **BLIND_HOLDOUT, **BLIND_DYN, **BLIND_SCAP, **CORRECTIONS_V1}
