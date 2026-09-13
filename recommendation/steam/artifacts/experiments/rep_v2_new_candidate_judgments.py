# rep_v2 Top-10 에 새로 올라온 후보 28쌍에 대한 Claude 판정 (실제 appid 기준)
J2 = {
 # P01 mix_vehicle_fps: ETS2 / Assetto Corsa / Left 4 Dead 2
 #   v2 에서 처음으로 L4D2(좀비 협동) 축이 제대로 살아났다 — v1 은 전작+저품질 2D 뿐
 ("mix_vehicle_fps", 1250):   (3, ""),                    # Killing Floor — 6인 협동 생존호러 FPS
 ("mix_vehicle_fps", 17500):  (3, ""),                    # Zombie Panic! Source — Source 협동 좀비 FPS
 ("mix_vehicle_fps", 689030): (2, ""),                    # Infection Rate — 협동 좀비 생존(저품질)
 ("mix_vehicle_fps", 281920): (1, "GENRE_ONLY"),          # Splatter — 탑다운 슈터
 ("mix_vehicle_fps", 494220): (0, "IRRELEVANT"),          # Blight of the Immortals — 언데드 전략

 # P02 coh_classic_multi: Garry's Mod / TF2 / Left 4 Dead 2
 ("coh_classic_multi", 17500):  (3, ""),                  # Zombie Panic! Source — 세 시드 모두와 결이 맞음
 ("coh_classic_multi", 689030): (2, ""),                  # Infection Rate
 ("coh_classic_multi", 281920): (1, "GENRE_ONLY"),        # Splatter
 ("coh_classic_multi", 494220): (0, "IRRELEVANT"),        # Blight of the Immortals
 ("coh_classic_multi", 657990): (2, ""),                  # Crafting Dead — 멀티 좀비 생존
 ("coh_classic_multi", 298240): (1, "TOO_NICHE"),         # War Trigger 3 — 저품질 무료 FPS

 # P03 mix_openworld_cozy: No Man's Sky / 서브노티카 / Slime Rancher
 ("mix_openworld_cozy", 632880): (2, ""),                 # Majesty of Colors — 수중 생명체 탐험

 # P04 coh_cozy: Stardew / Slime Rancher / Raft
 ("coh_cozy", 673950): (3, ""),                           # Farm Together — 협동 농장, Stardew 직결
 ("coh_cozy", 598240): (0, "KEYWORD_MATCH"),              # Stupid Raft Battle Simulator — 'Raft' 단어만
 ("coh_cozy", 280790): (2, ""),                           # Creativerse — 샌드박스 크래프팅

 # P05 mix_rpg_racing: Witcher 3 / Skyrim / BeamNG
 #   RPG 축이 통째로 사라지고 차량이 8칸을 먹었다 — seed 독점 악화
 ("mix_rpg_racing", 491280): (2, ""),                     # Drift Horizon Online — 드리프트 시뮬
 ("mix_rpg_racing", 352170): (1, "TOO_NICHE"),            # MadOut — 저품질 GTA 아류
 ("mix_rpg_racing", 400500): (1, "FRANCHISE_OR_VARIANT"), # MadOut Ice Storm — 위 게임의 변형
 ("mix_rpg_racing", 415600): (2, ""),                     # Kart Racing Pro — 사실적 카트 시뮬
 ("mix_rpg_racing", 41740):  (2, ""),                     # Cargo! — 커스텀 차량 제작
 ("mix_rpg_racing", 488550): (2, ""),                     # Dream Car Builder — 차 설계+레이싱
 ("mix_rpg_racing", 582390): (0, "IRRELEVANT"),           # Ski Sport: Jumping VR — 스키점프 VR

 # P06 mix_fps_cozy: CS2 / PUBG / Stardew
 ("mix_fps_cozy", 656240): (1, "GENRE_ONLY"),             # Heat — 야생 생존 샌드박스

 # P07 coh_indie_platformer: Hollow Knight / Dead Cells / Celeste
 ("coh_indie_platformer", 634700): (1, "TOO_NICHE"),      # PLUTONIUM — 마이너 2D 어드벤처
 ("coh_indie_platformer", 94400):  (1, "MODE_MISMATCH"),  # Nidhogg — 좋은 인디지만 대전 게임
 ("coh_indie_platformer", 541230): (0, "IRRELEVANT"),     # Alien Shooter TD — 타워디펜스

 # P08 mix_cozy_fps: Raft / Stardew / CS2
 ("mix_cozy_fps", 673950): (3, ""),                       # Farm Together
 ("mix_cozy_fps", 598240): (0, "KEYWORD_MATCH"),          # Stupid Raft Battle Simulator
}
