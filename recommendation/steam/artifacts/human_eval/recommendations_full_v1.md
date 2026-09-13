# 추천 결과 스냅샷

생성 2026-08-08 03:26 · git `ee0542a`

## 어떤 상태에서 나온 결과인가

| 항목 | 값 |
|---|---|
| 코퍼스 | 173,691개 |
| 임베딩 아티팩트 | `artifacts/full_v1` (지문 `cfd55010bf76`) |
| 임베딩 모델 | Qwen/Qwen3-Embedding-0.6B, 1024차원, L2 정규화 |
| 표현 | `Description` + `Genres` + `Modes`(게임플레이 모드만, 플랫폼 문구 제거) |
| 집계 | MAX (시드별 최대 유사도) |
| 인기도 부스트 | 15% |
| 후처리 | 시드 라운드로빈 인터리빙 · 시리즈 상한 1 · 퍼블리셔 상한 2 · hard filter(성인/VR전용/미출시) |
| 품질 하한 | 리뷰 300개 이상 (전체 코퍼스에서 롱테일이 유명작을 밀어냄) |
| 판정자 | Claude (Opus 5) — 사람 판정 아님 |
| 판정 범위 | 20개 프로필 중 8개 |

평가 표기: ★★★ 매우 타당 · ★★ 타당 · ★ 약함 · ✗ 부적절 · – 미채점

> Steam 전체 게임은 176,609개다(2026-08 기준). 이 코퍼스는 그중 98%.

---

### Garry's Mod + Team Fortress 2 + Left 4 Dead 2

`coherent` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | s&box | 액션, 어드벤처, 캐주얼, 인디, 레이싱, 시뮬 | ★★★ | 4,965 |
| 2 | Killing Floor | 액션 | ★★★ | 39,993 |
| 3 | Half-Life | 액션 | ★★ | 116,412 |
| 4 | BROKE PROTOCOL | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | ★★ | 1,606 |
| 5 | Left 4 Dead | 액션 | ★ | 57,756 |
| 6 | Battlefield 3™ | 액션 | ★★ | 16,026 |
| 7 | GoreBox | 액션, 캐주얼, 인디, 시뮬레이션 | ★★★ | 22,237 |
| 8 | Splatter - Zombiecalypse Now | 액션, 어드벤처, 인디 | ★ | 2,090 |
| 9 | Nuclear Dawn | 액션, 전략 | ★★ | 1,617 |
| 10 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | ★★ | 2,645 |

### Stardew Valley + Slime Rancher + Raft

`coherent` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Slime Rancher 2 | 액션, 어드벤처, 캐주얼, 인디, 시뮬레이션 | ★★★ | 42,923 |
| 2 | Staxel | 인디, RPG, 시뮬레이션 | ★★★ | 3,112 |
| 3 | Stranded Deep | 어드벤처, 인디 | ★★★ | 43,464 |
| 4 | 虫潮 | 액션, 어드벤처, 캐주얼, 인디, RPG | ✗ | 2,452 |
| 5 | Stonehearth | 인디, 시뮬레이션, 전략 | ★★ | 10,627 |
| 6 | Breakwaters: Crystal Tides | 액션, 어드벤처, 인디, 시뮬레이션, 앞서 해보 | ★★ | 867 |
| 7 | Eventide: Slavic Fable | 어드벤처, 캐주얼 | ★ | 592 |
| 8 | Roots of Pacha | 인디, RPG, 시뮬레이션 | ★★★ | 3,750 |
| 9 | Aground | 어드벤처, 인디, RPG | ★★ | 1,801 |
| 10 | Farming Simulator 22 | 시뮬레이션 | ★★ | 68,953 |

### Hollow Knight + Dead Cells + Celeste

`coherent` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Windblown | 액션, 인디, 앞서 해보기 | ★★★ | 8,972 |
| 2 | Hollow Knight: Silksong | 액션, 어드벤처, 인디 | ★★★ | 381,373 |
| 3 | Camera Obscura | 액션, 어드벤처, 인디, 전략 | ★★ | 1,336 |
| 4 | Dragon Is Dead | 액션, 인디, RPG | ★★ | 656 |
| 5 | Minishoot' Adventures | 액션, 어드벤처, 캐주얼, RPG | ★★ | 7,668 |
| 6 | TowerClimb | 액션, 어드벤처, 인디, RPG | ★★★ | 317 |
| 7 | Ravenswatch | 액션, 어드벤처 | ★★★ | 16,605 |
| 8 | Dunjungle | 액션, 어드벤처, 인디 | ★★ | 687 |
| 9 | Kingdom Rush Frontiers - Tower Defense | 전략 | ✗ | 9,833 |
| 10 | Nuclear Blaze | 액션, 어드벤처, 인디 | ★★ | 753 |

### Raft + Stardew Valley + Counter-Strike 2

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Staxel | 인디, RPG, 시뮬레이션 | ★★★ | 3,112 |
| 2 | Counter-Strike: Source | 액션 | ★ | 134,865 |
| 3 | Stranded Deep | 어드벤처, 인디 | ★★★ | 43,464 |
| 4 | Stonehearth | 인디, 시뮬레이션, 전략 | ★★ | 10,627 |
| 5 | Day of Defeat: Source | 액션 | ★★★ | 15,951 |
| 6 | Breakwaters: Crystal Tides | 액션, 어드벤처, 인디, 시뮬레이션, 앞서 해보 | ★★ | 867 |
| 7 | Roots of Pacha | 인디, RPG, 시뮬레이션 | ★★★ | 3,750 |
| 8 | Killing Floor | 액션 | ★★★ | 39,993 |
| 9 | Aground | 어드벤처, 인디, RPG | ★★ | 1,801 |
| 10 | Farming Simulator 22 | 시뮬레이션 | ★★ | 68,953 |

### Counter-Strike 2 + PUBG: BATTLEGROUNDS + Stardew Valley

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Staxel | 인디, RPG, 시뮬레이션 | ★★★ | 3,112 |
| 2 | Counter-Strike: Source | 액션 | ★ | 134,865 |
| 3 | RUSSIA BATTLEGROUNDS | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | ★ | 3,788 |
| 4 | Stonehearth | 인디, 시뮬레이션, 전략 | ★★ | 10,627 |
| 5 | Day of Defeat: Source | 액션 | ★★★ | 15,951 |
| 6 | Battlefield™ REDSEC | 액션, 무료 플레이 | ★★★ | 9,369 |
| 7 | Roots of Pacha | 인디, RPG, 시뮬레이션 | ★★★ | 3,750 |
| 8 | Killing Floor | 액션 | ★★ | 39,993 |
| 9 | Deadside | 액션, 어드벤처, 인디, 대규모 멀티플레이어 | ★★★ | 34,475 |
| 10 | Farming Simulator 22 | 시뮬레이션 | ★★ | 68,953 |

### No Man's Sky + 서브노티카 + Slime Rancher

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Slime Rancher 2 | 액션, 어드벤처, 캐주얼, 인디, 시뮬레이션 | ★★★ | 42,923 |
| 2 | In Other Waters | 어드벤처, 인디 | ★★★ | 2,072 |
| 3 | Space Rangers HD: A War Apart | 액션, 어드벤처, RPG, 시뮬레이션, 전략 | ★★ | 5,816 |
| 4 | Humans are not that against Lizardwomen  | 어드벤처, 캐주얼, 인디, RPG, 시뮬레이션 | ✗ | 380 |
| 5 | Sail Forth | 액션, 어드벤처, 인디, 시뮬레이션 | ★★ | 601 |
| 6 | ENDLESS Space™ 2 | 전략 | ★ | 17,192 |
| 7 | Zombotron | 액션, 어드벤처, 인디 | ✗ | 2,950 |
| 8 | Moon Mystery | 액션, 어드벤처, 캐주얼 | ★ | 544 |
| 9 | FAR: Lone Sails | 액션, 어드벤처, 인디 | ★★ | 24,278 |
| 10 | Creativerse | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | ★★ | 2,641 |

### The Witcher 3: Wild Hunt + The Elder Scrolls V: Skyrim Sp + BeamNG.drive

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | The Witcher: Enhanced Edition Director's | 액션, RPG | ★ | 82,053 |
| 2 | Drift Horizon Online | 액션, 레이싱, 시뮬레이션, 스포츠 | ★★ | 698 |
| 3 | The Elder Scrolls® Online | 액션, 어드벤처, 대규모 멀티플레이어, RPG | ★★ | 133,919 |
| 4 | The Surge 2 | 액션, RPG | ★★ | 7,395 |
| 5 | Sailwind | 어드벤처, 캐주얼, 인디, 시뮬레이션, 앞서 해 | ★★ | 2,260 |
| 6 | Angels of Death Episode.Eddie | 어드벤처, 인디, RPG | ✗ | 442 |
| 7 | Rogue : Genesia | 액션, 캐주얼, RPG, 전략 | ★ | 8,056 |
| 8 | Kart Racing Pro | 레이싱, 시뮬레이션, 스포츠 | ★★ | 1,010 |
| 9 | Nice Day for Fishing | 어드벤처, RPG | ★ | 2,545 |
| 10 | Dream Car Builder | 인디, 레이싱, 시뮬레이션 | ★★ | 961 |

### Euro Truck Simulator 2 + 아세토 코르사 Assetto Corsa + Left 4 Dead 2

`mixed` · 판정됨

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Truck Life | 캐주얼, 인디, 시뮬레이션, 스포츠 | ★★ | 1,412 |
| 2 | Killing Floor | 액션 | ★★★ | 39,993 |
| 3 | 아세토 코르사 랠리 Assetto Corsa Rally | 레이싱, 앞서 해보기 | ★★★ | 8,404 |
| 4 | American Truck Simulator | 인디, 시뮬레이션 | ★★★ | 165,963 |
| 5 | Left 4 Dead | 액션 | ★ | 57,756 |
| 6 | Chris Sawyer's Locomotion™ | 시뮬레이션, 전략 | ★ | 600 |
| 7 | Splatter - Zombiecalypse Now | 액션, 어드벤처, 인디 | ★ | 2,090 |
| 8 | Bus Driving Sim 22 | 시뮬레이션 | ★★ | 441 |
| 9 | Atom Zombie Smasher | 인디, 전략 | ★ | 1,104 |
| 10 | Alaskan Road Truckers | 액션, 어드벤처, 캐주얼, 레이싱, 시뮬레이션 | ★★★ | 2,711 |

### DARK SOULS™ III + The Witcher 3: Wild Hunt + The Elder Scrolls V: Skyrim Sp

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | The Witcher: Enhanced Edition Director's | 액션, RPG | – | 82,053 |
| 2 | Sekiro™: Shadows Die Twice - GOTY Editio | 액션, 어드벤처 | – | 257,490 |
| 3 | The Elder Scrolls® Online | 액션, 어드벤처, 대규모 멀티플레이어, RPG | – | 133,919 |
| 4 | The Surge 2 | 액션, RPG | – | 7,395 |
| 5 | The Dark Pictures Anthology: Man of Meda | 어드벤처 | – | 9,157 |
| 6 | Angels of Death Episode.Eddie | 어드벤처, 인디, RPG | – | 442 |
| 7 | Rogue : Genesia | 액션, 캐주얼, RPG, 전략 | – | 8,056 |
| 8 | Darkwood 다크우드 | 액션, 어드벤처, RPG, 전략 | – | 21,988 |
| 9 | Evoland Legendary Edition | 액션, 어드벤처, 캐주얼, 인디, RPG | – | 1,408 |
| 10 | Nice Day for Fishing | 어드벤처, RPG | – | 2,545 |

### Counter-Strike 2 + PUBG: BATTLEGROUNDS + 톰 클랜시의 레인보우식스 시즈

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Counter-Strike: Source | 액션 | – | 134,865 |
| 2 | RUSSIA BATTLEGROUNDS | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | – | 3,788 |
| 3 | Tom Clancy’s Rainbow Six® Extraction | 액션 | – | 2,845 |
| 4 | Day of Defeat: Source | 액션 | – | 15,951 |
| 5 | Battlefield™ REDSEC | 액션, 무료 플레이 | – | 9,369 |
| 6 | Returnal™ | 액션 | – | 6,494 |
| 7 | Killing Floor | 액션 | – | 39,993 |
| 8 | Deadside | 액션, 어드벤처, 인디, 대규모 멀티플레이어 | – | 34,475 |
| 9 | Iron Harvest | 전략 | – | 11,628 |
| 10 | Nuclear Dawn | 액션, 전략 | – | 1,617 |

### Hearts of Iron IV + Europa Universalis IV + Sid Meier's Civilization® V

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Hearts of Iron III | 전략 | – | 4,954 |
| 2 | Crusader Kings II | RPG, 시뮬레이션, 전략, 무료 플레이 | – | 30,868 |
| 3 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | – | 2,645 |
| 4 | Headquarters: World War II | 시뮬레이션, 전략 | – | 505 |
| 5 | Imperiums: Greek Wars | 시뮬레이션, 전략 | – | 449 |
| 6 | ROD: Revolt Of Defense | 액션, 인디, 시뮬레이션, 전략 | – | 525 |
| 7 | Order of Battle: World War II | 시뮬레이션, 전략, 무료 플레이 | – | 539 |
| 8 | 창조 삼국지 | 인디, RPG, 전략, 앞서 해보기 | – | 950 |
| 9 | Frozen Synapse | 인디, 전략 | – | 1,105 |
| 10 | Strategic Command WWII: War in Europe | 전략 | – | 329 |

### 서브노티카 + The Long Dark + No Man's Sky

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | In Other Waters | 어드벤처, 인디 | – | 2,072 |
| 2 | 침묵의 땅 | 어드벤처, 인디 | – | 408 |
| 3 | Space Rangers HD: A War Apart | 액션, 어드벤처, RPG, 시뮬레이션, 전략 | – | 5,816 |
| 4 | Sail Forth | 액션, 어드벤처, 인디, 시뮬레이션 | – | 601 |
| 5 | Don't Starve | 어드벤처, 인디, 시뮬레이션 | – | 96,781 |
| 6 | ENDLESS Space™ 2 | 전략 | – | 17,192 |
| 7 | Zombotron | 액션, 어드벤처, 인디 | – | 2,950 |
| 8 | Darkout | 액션, 어드벤처, 인디 | – | 561 |
| 9 | Moon Mystery | 액션, 어드벤처, 캐주얼 | – | 544 |
| 10 | FAR: Lone Sails | 액션, 어드벤처, 인디 | – | 24,278 |

### Sid Meier’s Civilization® VI + XCOM® 2 + Stellaris

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | SpaceCraft | 어드벤처, 인디, 대규모 멀티플레이어, 시뮬레이 | – | 2,422 |
| 2 | XCOM®: Chimera Squad | 전략 | – | 20,533 |
| 3 | Age of History 3 | 시뮬레이션, 전략 | – | 17,780 |
| 4 | Astro Colony | 액션, 어드벤처, 캐주얼, 인디, 시뮬레이션,  | – | 2,349 |
| 5 | Mutant Year Zero: Road to Eden | RPG, 전략 | – | 13,276 |
| 6 | Elite Dangerous | 액션, 어드벤처, 대규모 멀티플레이어, RPG, | – | 81,894 |
| 7 | Xenonauts | 인디, 시뮬레이션, 전략 | – | 3,802 |
| 8 | X4: Foundations | 액션, 시뮬레이션, 전략 | – | 24,818 |
| 9 | Fort Triumph | 인디, 전략 | – | 711 |
| 10 | Solar Expanse - Space Exploration Manage | 인디, 시뮬레이션, 전략, 앞서 해보기 | – | 1,140 |

### Project Zomboid + Don't Starve + The Forest

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | SILENT BREATH | 어드벤처, 인디, 시뮬레이션, 앞서 해보기 | – | 1,078 |
| 2 | Among Trees | 어드벤처, 인디, 시뮬레이션 | – | 2,064 |
| 3 | Over 9000 Zombies! | 액션, 인디 | – | 1,212 |
| 4 | Survival Zombies The Inverted Evolution | 액션, 어드벤처, 인디, 시뮬레이션 | – | 978 |
| 5 | 森林之子 | 어드벤처, 인디, RPG, 시뮬레이션, 앞서 해 | – | 312 |
| 6 | State of Decay 2: Juggernaut Edition | 액션, RPG, 시뮬레이션 | – | 58,622 |
| 7 | Stranded Deep | 어드벤처, 인디 | – | 43,464 |
| 8 | Forsaken Isle | 액션, 어드벤처, 캐주얼, 인디, RPG | – | 459 |
| 9 | Delivery from the Pain:Survival / 末日方舟:生 | 액션, 어드벤처, 인디, RPG, 시뮬레이션,  | – | 815 |
| 10 | Beyond Contact | 액션, 어드벤처, 인디 | – | 991 |

### Euro Truck Simulator 2 + BeamNG.drive + 아세토 코르사 Assetto Corsa

`coherent` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Truck Life | 캐주얼, 인디, 시뮬레이션, 스포츠 | – | 1,412 |
| 2 | Drift Horizon Online | 액션, 레이싱, 시뮬레이션, 스포츠 | – | 698 |
| 3 | 아세토 코르사 랠리 Assetto Corsa Rally | 레이싱, 앞서 해보기 | – | 8,404 |
| 4 | American Truck Simulator | 인디, 시뮬레이션 | – | 165,963 |
| 5 | Sailwind | 어드벤처, 캐주얼, 인디, 시뮬레이션, 앞서 해 | – | 2,260 |
| 6 | Chris Sawyer's Locomotion™ | 시뮬레이션, 전략 | – | 600 |
| 7 | Kart Racing Pro | 레이싱, 시뮬레이션, 스포츠 | – | 1,010 |
| 8 | Bus Driving Sim 22 | 시뮬레이션 | – | 441 |
| 9 | Dream Car Builder | 인디, 레이싱, 시뮬레이션 | – | 961 |
| 10 | Alaskan Road Truckers | 액션, 어드벤처, 캐주얼, 레이싱, 시뮬레이션 | – | 2,711 |

### DARK SOULS™ III + Fallout 4 + The Forest

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Starfield | RPG | – | 114,625 |
| 2 | SILENT BREATH | 어드벤처, 인디, 시뮬레이션, 앞서 해보기 | – | 1,078 |
| 3 | Sekiro™: Shadows Die Twice - GOTY Editio | 액션, 어드벤처 | – | 257,490 |
| 4 | Survival Zombies The Inverted Evolution | 액션, 어드벤처, 인디, 시뮬레이션 | – | 978 |
| 5 | The Dark Pictures Anthology: Man of Meda | 어드벤처 | – | 9,157 |
| 6 | Stranded Deep | 어드벤처, 인디 | – | 43,464 |
| 7 | Darkwood 다크우드 | 액션, 어드벤처, RPG, 전략 | – | 21,988 |
| 8 | Beyond Contact | 액션, 어드벤처, 인디 | – | 991 |
| 9 | DARK SOULS™: REMASTERED | 액션 | – | 99,123 |
| 10 | Aground | 어드벤처, 인디, RPG | – | 1,801 |

### Sid Meier's Civilization® V + Europa Universalis IV + Human Fall Flat

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Crusader Kings II | RPG, 시뮬레이션, 전략, 무료 플레이 | – | 30,868 |
| 2 | Ancient Warfare 3 | 액션, 인디, 시뮬레이션, 전략 | – | 2,645 |
| 3 | Soundfall | 액션, 어드벤처 | – | 988 |
| 4 | Imperiums: Greek Wars | 시뮬레이션, 전략 | – | 449 |
| 5 | ROD: Revolt Of Defense | 액션, 인디, 시뮬레이션, 전략 | – | 525 |
| 6 | Log Riders | 어드벤처, 캐주얼, 인디 | – | 610 |
| 7 | 창조 삼국지 | 인디, RPG, 전략, 앞서 해보기 | – | 950 |
| 8 | Frozen Synapse | 인디, 전략 | – | 1,105 |
| 9 | Chained Together | 어드벤처, 캐주얼, 인디, 시뮬레이션 | – | 53,724 |
| 10 | Ozymandias: Bronze Age Empire Sim | 시뮬레이션, 전략 | – | 1,292 |

### Hollow Knight + Dead Cells + Garry's Mod

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | s&box | 액션, 어드벤처, 캐주얼, 인디, 레이싱, 시뮬 | – | 4,965 |
| 2 | Windblown | 액션, 인디, 앞서 해보기 | – | 8,972 |
| 3 | Hollow Knight: Silksong | 액션, 어드벤처, 인디 | – | 381,373 |
| 4 | BROKE PROTOCOL | 액션, 어드벤처, 캐주얼, 인디, 대규모 멀티플 | – | 1,606 |
| 5 | Dragon Is Dead | 액션, 인디, RPG | – | 656 |
| 6 | Minishoot' Adventures | 액션, 어드벤처, 캐주얼, RPG | – | 7,668 |
| 7 | GoreBox | 액션, 캐주얼, 인디, 시뮬레이션 | – | 22,237 |
| 8 | Ravenswatch | 액션, 어드벤처 | – | 16,605 |
| 9 | Dunjungle | 액션, 어드벤처, 인디 | – | 687 |
| 10 | Guild Wars 2® | 어드벤처, 대규모 멀티플레이어, RPG, 무료  | – | 500 |

### Team Fortress 2 + Left 4 Dead 2 + Terraria

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | Project 5: Sightseer | 액션, 어드벤처, 인디, RPG | – | 423 |
| 2 | Killing Floor | 액션 | – | 39,993 |
| 3 | Half-Life | 액션 | – | 116,412 |
| 4 | Dungeons of Sundaria | 액션, 어드벤처, 인디, RPG | – | 2,989 |
| 5 | Left 4 Dead | 액션 | – | 57,756 |
| 6 | Battlefield 3™ | 액션 | – | 16,026 |
| 7 | Feel The Snow | 액션, 어드벤처, 인디, 대규모 멀티플레이어,  | – | 2,624 |
| 8 | Splatter - Zombiecalypse Now | 액션, 어드벤처, 인디 | – | 2,090 |
| 9 | Nuclear Dawn | 액션, 전략 | – | 1,617 |
| 10 | Hyper Light Drifter | 액션, 어드벤처, 인디, RPG | – | 15,387 |

### The Forest + 7 Days to Die + Sid Meier’s Civilization® VI

`mixed` · 미판정

| # | 추천 게임 | 장르 | 평가 | 리뷰 |
|---|---|---|---|---|
| 1 | SILENT BREATH | 어드벤처, 인디, 시뮬레이션, 앞서 해보기 | – | 1,078 |
| 2 | 세븐데이즈 오리진 | 어드벤처, 캐주얼, 인디, 시뮬레이션 | – | 313 |
| 3 | Age of History 3 | 시뮬레이션, 전략 | – | 17,780 |
| 4 | Survival Zombies The Inverted Evolution | 액션, 어드벤처, 인디, 시뮬레이션 | – | 978 |
| 5 | 我与你共度的七日 | 어드벤처, 캐주얼, 인디, RPG, 시뮬레이션 | – | 924 |
| 6 | 시드 마이어의 문명 VII | 시뮬레이션, 전략 | – | 43,640 |
| 7 | Stranded Deep | 어드벤처, 인디 | – | 43,464 |
| 8 | Night of the Dead | 액션, 어드벤처, 인디, RPG, 시뮬레이션 | – | 10,345 |
| 9 | Beyond Contact | 액션, 어드벤처, 인디 | – | 991 |
| 10 | Days of War: Definitive Edition | 액션, 인디 | – | 1,357 |
