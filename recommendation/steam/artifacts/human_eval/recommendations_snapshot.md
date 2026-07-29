# 추천 결과 스냅샷

생성 2026-07-29 06:10 · git `db1f51b`

## 어떤 상태에서 나온 결과인가

| 항목 | 값 |
|---|---|
| 코퍼스 | 19,476개 |
| 임베딩 아티팩트 | `artifacts/rep_v2` (지문 `f314b18b5f4a`) |
| 임베딩 모델 | Qwen/Qwen3-Embedding-0.6B, 1024차원, L2 정규화 |
| 표현 | `Description` + `Genres` + `Modes`(게임플레이 모드만, 플랫폼 문구 제거) |
| 집계 | MAX (시드별 최대 유사도) |
| 인기도 부스트 | 15% |
| 후처리 | 시드 라운드로빈 인터리빙 · 시리즈 상한 1 · hard filter(성인/VR전용/미출시) |
| 품질 하한 | 리뷰 수가 보고되는 게임만 (`has_recommendations`) |
| 판정자 | Claude (Opus 5) — 사람 판정 아님 |
| 판정 범위 | 20개 프로필 중 8개 |

평가 표기: ★★★ 매우 타당 · ★★ 타당 · ★ 약함 · ✗ 부적절 · – 미채점

> 이 코퍼스는 Steam 전체(176,609개)의 11%다. 전체 크롤링이 진행 중이며, 완료 후 같은
> 스크립트로 다시 뽑아 비교한다.

---

### Garry's Mod + Team Fortress 2 + Left 4 Dead 2

`coherent` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | s&box | 액션, 어드벤처, 캐주얼, 인디, 레이싱, 시뮬 | ★★★ | 4,919 |
| 2 | Killing Floor | 액션 | ★★★ | 39,967 |
| 3 | Half-Life | 액션 | ★★ | 116,142 |
| 4 | BROKE PROTOCOL | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | ★★ | 1,606 |
| 5 | Left 4 Dead | 액션 | ★ | 57,627 |
| 6 | Nuclear Dawn | 액션, 전략 | ★★ | 1,614 |
| 7 | Modbox | 액션, 어드벤처, 캐주얼, 인디, 시뮬레이션,  | – | 199 |
| 8 | Zombie Grinder | 액션, 어드벤처, 인디, RPG, 앞서 해보기 | ★ | 257 |
| 9 | War Trigger 3 | 액션, 캐주얼, 인디, 시뮬레이션, 무료 플레이 | ★ | 161 |
| 10 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | – | 2,646 |

### Stardew Valley + Slime Rancher + Raft

`coherent` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Staxel | 인디, RPG, 시뮬레이션 | ★★★ | 3,109 |
| 2 | Stranded Deep | 어드벤처, 인디 | ★★★ | 43,436 |
| 3 | Eventide: Slavic Fable | 어드벤처, 캐주얼 | – | 590 |
| 4 | Stonehearth | 인디, 시뮬레이션, 전략 | ★★ | 10,625 |
| 5 | The Caribbean Sail | 액션, 어드벤처, 캐주얼, 인디, RPG, 시뮬 | ★ | 500 |
| 6 | Star Control®: Origins | 액션, 어드벤처, 시뮬레이션 | – | 2,057 |
| 7 | Farm Together | 캐주얼, 인디, 시뮬레이션 | ★★★ | 17,629 |
| 8 | Feel The Snow | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | ★★ | 2,622 |
| 9 | Rayman® Origins | 액션, 어드벤처 | – | 5,761 |
| 10 | Farming Simulator 2013 Titanium Edition | 캐주얼, 시뮬레이션 | ★★ | 3,915 |

### Hollow Knight + Dead Cells + Celeste

`coherent` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Camera Obscura | 액션, 어드벤처, 인디, 전략 | ★★ | 1,336 |
| 2 | Nidhogg | 액션, 인디, 스포츠 | ★ | 5,689 |
| 3 | Dex | 액션, 어드벤처, 인디, RPG | ★★★ | 2,868 |
| 4 | TowerClimb | 액션, 어드벤처, 인디, RPG | ★★★ | 317 |
| 5 | LOST EMBER | 액션, 어드벤처, 인디 | ★ | 4,189 |
| 6 | Deadly Days | 액션, 캐주얼, 인디, 전략 | ★ | 1,381 |
| 7 | Kingdom Rush Frontiers - Tower Defense | 전략 | ✗ | 9,812 |
| 8 | Darkwood 다크우드 | 액션, 어드벤처, RPG, 전략 | ★ | 21,910 |
| 9 | Dead Rising® 2 | 액션, 어드벤처 | ✗ | 6,180 |
| 10 | Tangle Tower | 어드벤처, 인디 | ✗ | 6,031 |

### Raft + Stardew Valley + Counter-Strike 2

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Staxel | 인디, RPG, 시뮬레이션 | ★★★ | 3,109 |
| 2 | Counter-Strike: Source | 액션 | ★ | 134,732 |
| 3 | Stranded Deep | 어드벤처, 인디 | ★★★ | 43,436 |
| 4 | Stonehearth | 인디, 시뮬레이션, 전략 | ★★ | 10,625 |
| 5 | Day of Defeat: Source | 액션 | ★★★ | 15,932 |
| 6 | The Caribbean Sail | 액션, 어드벤처, 캐주얼, 인디, RPG, 시뮬 | ★ | 500 |
| 7 | Farm Together | 캐주얼, 인디, 시뮬레이션 | ★★★ | 17,629 |
| 8 | Half-Life | 액션 | – | 116,142 |
| 9 | Feel The Snow | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | ★★ | 2,622 |
| 10 | Farming Simulator 2013 Titanium Edition | 캐주얼, 시뮬레이션 | ★★ | 3,915 |

### Counter-Strike 2 + PUBG: BATTLEGROUNDS + Stardew Valley

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Staxel | 인디, RPG, 시뮬레이션 | ★★★ | 3,109 |
| 2 | Counter-Strike: Source | 액션 | ★ | 134,732 |
| 3 | Skillshot City | 액션, 인디, 무료 플레이, 앞서 해보기 | ★ | 120 |
| 4 | Stonehearth | 인디, 시뮬레이션, 전략 | ★★ | 10,625 |
| 5 | Day of Defeat: Source | 액션 | ★★★ | 15,932 |
| 6 | 빈 배틀즈 | 액션, 인디 | ★★ | 6,766 |
| 7 | Farm Together | 캐주얼, 인디, 시뮬레이션 | ★★★ | 17,629 |
| 8 | Half-Life | 액션 | ★★ | 116,142 |
| 9 | Alien Swarm | 액션, 무료 플레이 | ★★ | 294 |
| 10 | Farming Simulator 2013 Titanium Edition | 캐주얼, 시뮬레이션 | ★★ | 3,915 |

### No Man's Sky + 서브노티카 + Slime Rancher

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | FAR: Lone Sails | 액션, 어드벤처, 인디 | ★★ | 24,205 |
| 2 | Space Rangers HD: A War Apart | 액션, 어드벤처, RPG, 시뮬레이션, 전략 | ★★ | 5,811 |
| 3 | Rayman® Origins | 액션, 어드벤처 | – | 5,761 |
| 4 | Zombotron | 액션, 어드벤처, 인디 | ✗ | 2,944 |
| 5 | ENDLESS Space™ 2 | 전략 | – | 17,181 |
| 6 | Breakneck | 액션, 레이싱 | – | 174 |
| 7 | AER Memories of Old | 어드벤처, 인디 | ★★ | 5,027 |
| 8 | Creativerse | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | – | 2,641 |
| 9 | 소닉 매니아 | 액션, 어드벤처, 캐주얼 | – | 24,011 |
| 10 | Narcosis | 어드벤처, 인디 | ★★★ | 1,200 |

### The Witcher 3: Wild Hunt + The Elder Scrolls V: Skyrim Sp + BeamNG.drive

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | The Witcher: Enhanced Edition Director's | 액션, RPG | ★ | 81,934 |
| 2 | Drift Horizon Online | 액션, 레이싱, 시뮬레이션, 스포츠 | ★★ | 697 |
| 3 | The Elder Scrolls® Online | 액션, 어드벤처, 대규모 멀티플레이어, RPG | ★★ | 133,822 |
| 4 | The Surge 2 | 액션, RPG | ★★ | 7,368 |
| 5 | City Car Driving | 인디, 레이싱, 시뮬레이션 | ★★ | 33,172 |
| 6 | FINAL FANTASY XV WINDOWS EDITION | RPG | – | 44,609 |
| 7 | 아토믹 하트 | 액션, 어드벤처, RPG | ★★★ | 37,446 |
| 8 | Kart Racing Pro | 레이싱, 시뮬레이션, 스포츠 | ★★ | 1,005 |
| 9 | Shadowrun Returns | 어드벤처, 인디, RPG, 전략 | – | 9,487 |
| 10 | Zanzarah: The Hidden Portal | 액션, 어드벤처, RPG | ★ | 975 |

### Euro Truck Simulator 2 + 아세토 코르사 Assetto Corsa + Left 4 Dead 2

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Killing Floor | 액션 | ★★★ | 39,967 |
| 2 | American Truck Simulator | 인디, 시뮬레이션 | ★★★ | 165,605 |
| 3 | RIDE 3 | 레이싱, 시뮬레이션, 스포츠 | – | 4,522 |
| 4 | Left 4 Dead | 액션 | ★ | 57,627 |
| 5 | Chris Sawyer's Locomotion™ | 시뮬레이션, 전략 | ★ | 598 |
| 6 | Barro | 캐주얼, 인디, 레이싱 | – | 3,218 |
| 7 | Zombie Grinder | 액션, 어드벤처, 인디, RPG, 앞서 해보기 | ★ | 257 |
| 8 | Truck Driver | 어드벤처, 캐주얼, 시뮬레이션 | ★★★ | 157 |
| 9 | MXGP3 - The Official Motocross Videogame | 레이싱, 시뮬레이션, 스포츠 | – | 1,023 |
| 10 | Splatter - Zombiecalypse Now | 액션, 어드벤처, 인디 | ★ | 2,089 |

### DARK SOULS™ III + The Witcher 3: Wild Hunt + The Elder Scrolls V: Skyrim Sp

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | The Witcher: Enhanced Edition Director's | 액션, RPG | – | 81,934 |
| 2 | Darkwood 다크우드 | 액션, 어드벤처, RPG, 전략 | – | 21,910 |
| 3 | The Elder Scrolls® Online | 액션, 어드벤처, 대규모 멀티플레이어, RPG | – | 133,822 |
| 4 | The Surge 2 | 액션, RPG | – | 7,368 |
| 5 | DARK SOULS™: REMASTERED | 액션 | – | 98,893 |
| 6 | FINAL FANTASY XV WINDOWS EDITION | RPG | – | 44,609 |
| 7 | 아토믹 하트 | 액션, 어드벤처, RPG | – | 37,446 |
| 8 | Yet Another Zombie Defense | 액션, 캐주얼, 인디, 전략 | – | 7,681 |
| 9 | Shadowrun Returns | 어드벤처, 인디, RPG, 전략 | – | 9,487 |
| 10 | Zanzarah: The Hidden Portal | 액션, 어드벤처, RPG | – | 975 |

### Counter-Strike 2 + PUBG: BATTLEGROUNDS + 톰 클랜시의 레인보우식스 시즈

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Counter-Strike: Source | 액션 | – | 134,732 |
| 2 | Skillshot City | 액션, 인디, 무료 플레이, 앞서 해보기 | – | 120 |
| 3 | Dirty Bomb® | 액션, 무료 플레이 | – | 1,265 |
| 4 | Day of Defeat: Source | 액션 | – | 15,932 |
| 5 | 빈 배틀즈 | 액션, 인디 | – | 6,766 |
| 6 | Tom Clancy's Rainbow Six Lockdown™ | 액션 | – | 510 |
| 7 | Half-Life | 액션 | – | 116,142 |
| 8 | Alien Swarm | 액션, 무료 플레이 | – | 294 |
| 9 | 토탈워: 워해머 2 | 액션, 전략 | – | 87,831 |
| 10 | Killing Floor | 액션 | – | 39,967 |

### Hearts of Iron IV + Europa Universalis IV + Sid Meier's Civilization® V

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Hearts of Iron III | 전략 | – | 4,948 |
| 2 | Crusader Kings II | RPG, 시뮬레이션, 전략, 무료 플레이 | – | 30,864 |
| 3 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | – | 2,646 |
| 4 | Making History: The Second World War | 인디, 시뮬레이션, 전략 | – | 266 |
| 5 | For The Glory: A Europa Universalis Game | 시뮬레이션, 전략 | – | 147 |
| 6 | ROD: Revolt Of Defense | 액션, 인디, 시뮬레이션, 전략 | – | 524 |
| 7 | Holdfast: Nations At War | 액션, 인디, 대규모 멀티플레이어, RPG, 시 | – | 26,286 |
| 8 | Supreme Ruler Ultimate | 인디, 시뮬레이션, 전략 | – | 1,168 |
| 9 | Frozen Synapse | 인디, 전략 | – | 1,105 |
| 10 | Order of Battle: World War II | 시뮬레이션, 전략, 무료 플레이 | – | 540 |

### 서브노티카 + The Long Dark + No Man's Sky

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | FAR: Lone Sails | 액션, 어드벤처, 인디 | – | 24,205 |
| 2 | Space Rangers HD: A War Apart | 액션, 어드벤처, RPG, 시뮬레이션, 전략 | – | 5,811 |
| 3 | Don't Starve | 어드벤처, 인디, 시뮬레이션 | – | 96,710 |
| 4 | Zombotron | 액션, 어드벤처, 인디 | – | 2,944 |
| 5 | ENDLESS Space™ 2 | 전략 | – | 17,181 |
| 6 | Darkout | 액션, 어드벤처, 인디 | – | 562 |
| 7 | AER Memories of Old | 어드벤처, 인디 | – | 5,027 |
| 8 | Creativerse | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | – | 2,641 |
| 9 | Amnesia: The Dark Descent | 액션, 어드벤처, 인디 | – | 20,238 |
| 10 | Narcosis | 어드벤처, 인디 | – | 1,200 |

### Sid Meier’s Civilization® VI + XCOM® 2 + Stellaris

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Elite Dangerous | 액션, 어드벤처, 대규모 멀티플레이어, RPG, | – | 81,809 |
| 2 | XCOM: Enemy Unknown | 전략 | – | 40,145 |
| 3 | Knights of Honor II: Sovereign | 시뮬레이션, 전략 | – | 4,892 |
| 4 | X4: Foundations | 액션, 시뮬레이션, 전략 | – | 24,706 |
| 5 | Mutant Year Zero: Road to Eden | RPG, 전략 | – | 13,274 |
| 6 | Men of War: Vietnam | 전략 | – | 1,103 |
| 7 | Sid Meier's Starships | 전략 | – | 1,959 |
| 8 | Xenonauts | 인디, 시뮬레이션, 전략 | – | 3,801 |
| 9 | Empires Apart | 무료 플레이, 전략 | – | 326 |
| 10 | Imagine Earth | 캐주얼, 인디, 시뮬레이션, 전략 | – | 804 |

### Project Zomboid + Don't Starve + The Forest

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Zombasite | 액션, 인디, RPG | – | 191 |
| 2 | Stranded Deep | 어드벤처, 인디 | – | 43,436 |
| 3 | Forsaken Isle | 액션, 어드벤처, 캐주얼, 인디, RPG | – | 459 |
| 4 | State of Decay 2: Juggernaut Edition | 액션, RPG, 시뮬레이션 | – | 58,506 |
| 5 | Survival Zombies The Inverted Evolution | 액션, 어드벤처, 인디, 시뮬레이션 | – | 978 |
| 6 | The Isle | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | – | 107,375 |
| 7 | Zombie Grinder | 액션, 어드벤처, 인디, RPG, 앞서 해보기 | – | 257 |
| 8 | Eco | 어드벤처, 인디, 시뮬레이션, 앞서 해보기 | – | 11,289 |
| 9 | Dustbowl | 어드벤처, 인디, RPG | – | 109 |
| 10 | Over 9000 Zombies! | 액션, 인디 | – | 1,213 |

### Euro Truck Simulator 2 + BeamNG.drive + 아세토 코르사 Assetto Corsa

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | American Truck Simulator | 인디, 시뮬레이션 | – | 165,605 |
| 2 | Drift Horizon Online | 액션, 레이싱, 시뮬레이션, 스포츠 | – | 697 |
| 3 | 리볼트 | 레이싱 | – | 1,336 |
| 4 | Chris Sawyer's Locomotion™ | 시뮬레이션, 전략 | – | 598 |
| 5 | City Car Driving | 인디, 레이싱, 시뮬레이션 | – | 33,172 |
| 6 | Sonic & SEGA All-Stars Racing | 레이싱 | – | 2,612 |
| 7 | Truck Driver | 어드벤처, 캐주얼, 시뮬레이션 | – | 157 |
| 8 | Kart Racing Pro | 레이싱, 시뮬레이션, 스포츠 | – | 1,005 |
| 9 | Razortron 2000 | 캐주얼, 인디, 레이싱 | – | 3,013 |
| 10 | Euro Truck Simulator | 인디, 시뮬레이션 | – | 6,430 |

### DARK SOULS™ III + Fallout 4 + The Forest

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Stranded Deep | 어드벤처, 인디 | – | 43,436 |
| 2 | Darkwood 다크우드 | 액션, 어드벤처, RPG, 전략 | – | 21,910 |
| 3 | Dark and Light | 액션, 어드벤처, 대규모 멀티플레이어, RPG, | – | 10,573 |
| 4 | Survival Zombies The Inverted Evolution | 액션, 어드벤처, 인디, 시뮬레이션 | – | 978 |
| 5 | DARK SOULS™: REMASTERED | 액션 | – | 98,893 |
| 6 | Fallout 3: Game of the Year Edition | RPG | – | 43,687 |
| 7 | Eco | 어드벤처, 인디, 시뮬레이션, 앞서 해보기 | – | 11,289 |
| 8 | Yet Another Zombie Defense | 액션, 캐주얼, 인디, 전략 | – | 7,681 |
| 9 | Falcon 4.0 | 시뮬레이션 | – | 282 |
| 10 | State of Decay 2: Juggernaut Edition | 액션, RPG, 시뮬레이션 | – | 58,506 |

### Sid Meier's Civilization® V + Europa Universalis IV + Human Fall Flat

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Crusader Kings II | RPG, 시뮬레이션, 전략, 무료 플레이 | – | 30,864 |
| 2 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | – | 2,646 |
| 3 | Deadfall Adventures | 액션, 어드벤처 | – | 1,300 |
| 4 | For The Glory: A Europa Universalis Game | 시뮬레이션, 전략 | – | 147 |
| 5 | ROD: Revolt Of Defense | 액션, 인디, 시뮬레이션, 전략 | – | 524 |
| 6 | City Climber | 액션, 어드벤처, 캐주얼, 인디, 시뮬레이션,  | – | 365 |
| 7 | Supreme Ruler Ultimate | 인디, 시뮬레이션, 전략 | – | 1,168 |
| 8 | Frozen Synapse | 인디, 전략 | – | 1,105 |
| 9 | Bloody Trapland 2: Curiosity | 액션, 어드벤처, 캐주얼, 인디 | – | 817 |
| 10 | Ashes of the Singularity: Escalation | 시뮬레이션, 전략 | – | 2,690 |

### Hollow Knight + Dead Cells + Garry's Mod

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | s&box | 액션, 어드벤처, 캐주얼, 인디, 레이싱, 시뮬 | – | 4,919 |
| 2 | Nidhogg | 액션, 인디, 스포츠 | – | 5,689 |
| 3 | Dex | 액션, 어드벤처, 인디, RPG | – | 2,868 |
| 4 | BROKE PROTOCOL | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | – | 1,606 |
| 5 | LOST EMBER | 액션, 어드벤처, 인디 | – | 4,189 |
| 6 | Deadly Days | 액션, 캐주얼, 인디, 전략 | – | 1,381 |
| 7 | Modbox | 액션, 어드벤처, 캐주얼, 인디, 시뮬레이션,  | – | 199 |
| 8 | Darkwood 다크우드 | 액션, 어드벤처, RPG, 전략 | – | 21,910 |
| 9 | Dead Rising® 2 | 액션, 어드벤처 | – | 6,180 |
| 10 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | – | 2,646 |

### Team Fortress 2 + Left 4 Dead 2 + Terraria

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Dungeons of Sundaria | 액션, 어드벤처, 인디, RPG | – | 2,984 |
| 2 | Killing Floor | 액션 | – | 39,967 |
| 3 | Half-Life | 액션 | – | 116,142 |
| 4 | Project 5: Sightseer | 액션, 어드벤처, 인디, RPG | – | 423 |
| 5 | Left 4 Dead | 액션 | – | 57,627 |
| 6 | Nuclear Dawn | 액션, 전략 | – | 1,614 |
| 7 | Feel The Snow | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | – | 2,622 |
| 8 | Zombie Grinder | 액션, 어드벤처, 인디, RPG, 앞서 해보기 | – | 257 |
| 9 | War Trigger 3 | 액션, 캐주얼, 인디, 시뮬레이션, 무료 플레이 | – | 161 |
| 10 | Hyper Light Drifter | 액션, 어드벤처, 인디, RPG | – | 15,345 |

### The Forest + 7 Days to Die + Sid Meier’s Civilization® VI

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Stranded Deep | 어드벤처, 인디 | – | 43,436 |
| 2 | Survive the Nights | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | – | 4,624 |
| 3 | Age of Wonders: Planetfall | 전략 | – | 4,744 |
| 4 | Survival Zombies The Inverted Evolution | 액션, 어드벤처, 인디, 시뮬레이션 | – | 978 |
| 5 | Days of War: Definitive Edition | 액션, 인디 | – | 1,357 |
| 6 | Galactic Civilizations III | 전략 | – | 8,328 |
| 7 | Eco | 어드벤처, 인디, 시뮬레이션, 앞서 해보기 | – | 11,289 |
| 8 | Space Rangers HD: A War Apart | 액션, 어드벤처, RPG, 시뮬레이션, 전략 | – | 5,811 |
| 9 | Europa Universalis IV | 시뮬레이션, 전략 | – | 95,453 |
| 10 | State of Decay 2: Juggernaut Edition | 액션, RPG, 시뮬레이션 | – | 58,506 |
---

## 20개를 다 보고 나서 — 반복되는 문제 3가지

판정된 8개만 볼 때보다 미판정 12개를 함께 볼 때 경향이 뚜렷하다.

### ① 품질 하한이 무디다 — 무명 게임이 여전히 상위에 온다

리뷰 100~500개짜리가 1~5위에 나온다:

| 프로필 | 순위 | 게임 | 리뷰 |
|---|---|---|---|
| Project Zomboid + Don't Starve + The Forest | 1 | Zombasite | 191 |
| Project Zomboid + Don't Starve + The Forest | 3 | Forsaken Isle | 459 |
| HoI4 + EU4 + Civilization | 5 | For The Glory | 147 |
| TF2 + L4D2 + Terraria | 4 | Project 5: Sightseer | 423 |

원인: 하한이 **"리뷰 수가 보고되는가"라는 이진 조건**이라 100개나 100만 개나 똑같이 통과한다.
Steam 이 값을 주기 시작하는 하한선만 넘으면 되기 때문이다.

→ 개선안: 이진 통과가 아니라 리뷰 수를 연속값으로 랭킹에 반영하거나, 절대 하한(예: 1,000)을 둔다.
   다만 앞서 측정했듯 하한을 올리면 좋은 추천도 함께 잘린다(현재 하한도 3점짜리의 6% 를 제거).

### ② 전작이 계속 1위를 먹는다 — 시리즈 상한이 못 잡는다

| 시드 | 추천된 전작 | 순위 |
|---|---|---|
| Hearts of Iron IV | Hearts of Iron III | **1** |
| The Witcher 3 | The Witcher 1 | **1** |
| Euro Truck Simulator 2 | Euro Truck Simulator | 10 |
| Counter-Strike 2 | Counter-Strike: Source | 2 |

`series_key` 가 이름 앞 2단어로 추측하는데 `Hearts of` vs `Hearts of` 는 같아야 하는데도
정규화 과정에서 어긋나는 경우가 있고, `WT2` vs `War Trigger 3` 처럼 이름이 아예 다른 경우는
원리적으로 못 잡는다.

→ 퍼블리셔 기반 판정(`series_group`)을 구현해뒀지만 **현재 데이터에 `publisher` 컬럼이 없다.**
   전체 크롤링이 끝나야 효과가 난다.

### ③ 장르가 튀는 것이 섞인다

| 프로필 | 튀는 추천 |
|---|---|
| 서브노티카 + The Long Dark + No Man's Sky | ENDLESS Space 2 (전략) |
| DARK SOULS III + Witcher 3 + Skyrim | Yet Another Zombie Defense |
| Civilization + EU4 + Human Fall Flat | City Climber, Bloody Trapland 2 |
| Hollow Knight + Dead Cells + Celeste | Kingdom Rush(타워디펜스), Dead Rising 2 |

---

## 잘 되는 것과 안 되는 것의 경계

**취향이 좁고 명확한 장르일수록 잘 되고, 넓거나 분위기로 정의되는 장르일수록 무너진다.**

| 잘 됨 | 무너짐 |
|---|---|
| 전략 (Civ+XCOM+Stellaris — 10/10 전략) | 플랫포머 (Hollow Knight+Dead Cells+Celeste) |
| 레이싱 (ETS2+BeamNG+Assetto — 10/10 운전) | 생존 (Project Zomboid+Don't Starve+The Forest) |
| 그랜드 전략 (HoI4+EU4+Civ) | 탐험 (서브노티카+Long Dark+NMS) |

이유는 표현에 있다. 전략·레이싱 게임의 설명문에는 **구체적 명사**(문명, 함선, 제국, 트럭,
드리프트)가 많아 임베딩이 구분할 수 있다. 반면 플랫포머·생존·탐험은 설명문이 분위기 위주라
(`왕국`, `동굴`, `생존`, `탐험`) 수천 개 게임이 같은 단어를 공유한다.

이것이 지금 시스템의 **성능 상한**이고, 랭킹을 더 다듬어서는 넘을 수 없다.
필요한 것은 게임플레이를 서술하는 신호(Steam 사용자 태그: `메트로베니아`, `소울라이크`,
`로그라이트`)인데 `appdetails` API 가 제공하지 않는다는 것이 크롤러 코드로 확인됐다.
