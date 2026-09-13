# D-56 사전등록 — Steam "격" 축: `has_metacritic`

## 왜

Steam 은 세 플랫폼 중 유일하게 꼬리가 무너진다 — P@50 **0.8215**, 0.8 미만 **18개**.
D-52 재채점 후에도 그대로다. 최하위 8개 프로필의 실패 항목을 눈으로 보니
**두 종류**로 갈렸다:

**A. 태그는 완벽한데 격이 미달** — 무명 아류작
```
coh_survival_craft  SURVIVAL: Postapocalypse Now · SurrounDead · Novus Inceptio · Urge
                    (전부 Open World Survival Craft · Survival · Open World, g=1)
coh_vehicle_sim     Alaskan Road Truckers · My Truck Game · Used Cars Simulator (g=1)
mix2_coop_horror    Bunker Invaders · DEATH IN UNISON · Horror Squad (g=1)
coh_sports          축구 온라인: 볼3D · World of Football · Disc Golf Masters (g=1)
```
**B. 대작인데 태그가 안 맞음** — `twenty_broad` 에 ELDEN RING·Starfield·Fallout 3,
`coh_mmo` 에 위쳐3·Counter-Strike 2 (전부 g=1)

**D-49 의 `tag_w` 는 B 를 고쳤다. A 는 아무것도 안 건드리고 있다.**
`quality_w`(리뷰 수)가 A 를 막을 것 같지만 안 막는다 — 태그 완벽(1.0)한 아류작이
태그 0.6인 명작을 **0.40×0.4 = 0.16** 앞서는데, 리뷰 수 격차가 만드는 차이는
1천 대 10만에서 **0.50×0.4 = 0.20** 뿐이라 서로 상쇄된다.

## 신호 측정 (변경 전에 쟀다 — D-49 절차)

프로필 내부 상관 중앙 · 부호일치 (52프로필 · 6,252쌍):

| 후보 | 원상관 | 부호일치 | **잔차상관** | 부호일치 |
|---|---|---|---|---|
| `_rev` (현행 quality) | +0.457 | 51/52 | — | — |
| 개발사 격 `_devmax` | **+0.465** | 51/52 | **+0.019** | 37/52 |
| **`has_metacritic`** | +0.333 | 49/52 | **+0.131** | **42/52** |
| metacritic 점수 | — | — | +0.060 | 39/52 |

잔차 = 등급을 `[1, quality, tag_fit]` 에 프로필별로 회귀한 나머지.
**`_devmax` 는 원상관이 제일 높지만 잔차가 +0.019 로 사실상 0** — 리뷰 수의 재표현이다.
**남는 건 `has_metacritic` 하나다.** 점수가 아니라 **보유 여부**가 신호다:
"언론이 다뤘는가"가 격이고, 그 다음 미세 구분은 리뷰 수가 이미 잡는다.

눈가림 시트에는 metacritic 을 **노출한 적이 없다** — 순환이 아니다.

## 변경

`quality_w=0.50` · `tag_w=0.40` · `rec_boost=0.03` 고정. 항 하나만 더한다.

    final = sim × (1 + rec_pct·rec_boost + quality·quality_w + tag_fit·tag_w
                     + has_metacritic·mc_w)

## 후보

`mc_w` = **0.10 · 0.20 · 0.30** (현행 0.00)

## 판정 기준 (관측 전 확정)

**최소 효과크기를 먼저 계산한다(h44).** A 유형으로 분류한 프로필은 dev 30개 중
`coh_vehicle_sim` · `mix2_coop_horror` · `coh_sports` **3개**
(`coh_survival_craft` 는 val 이다). 셋이 각각 +0.10 오르면 전체 dev 는
3/30 × 0.10 = **+0.010**. 여기에 다른 프로필의 잡음을 감안해 문턱을 **+0.015** 로 잡는다.
D-49 의 +0.03 은 꼬리 전체를 고치는 축이라 그랬고, 이건 부분군을 겨냥한 축이다.

**dev 자격 (넷 다):**
- **(A)** 전체 dev P@50 이 현행 대비 **+0.015 이상**
- **(B)** 저인기 축∩dev(7개, `lowrev_*`·`longtail_*`·`niche_*`) 평균이 **−0.02 이상**
- **(C)** 어떤 dev 프로필도 **0.10 초과 하락 없음**
- **(E)** 전체 dev **ILS@50 이 +0.03 이내**

**자격자가 여럿이면 전체 dev P@50 이 가장 높은 것.** 동점(0.005 미만)이면 **작은 `mc_w`.**

**val 확증:** 전체 val P@50 **+0.005 이상** · 저인기 축∩val(8개) **−0.02 이상** ·
ILS **+0.03 이내**.

**기각 시 현행 유지. 문턱을 낮춰 다시 재지 않는다.**

## 함께 보고할 진단 (관문 아님)

- A 유형 3개 프로필(`coh_vehicle_sim`·`mix2_coop_horror`·`coh_sports`)의 개별 변화
- top-50 안 **metacritic 보유 비율**의 변화
- 위에 나열한 실패 항목들이 실제로 빠졌는지 눈검수

## 예상 실패 모드

1. **(B) 위반 — 저인기 축이 무너진다. 가장 가능성이 높다고 본다.**
   코퍼스 전체에서 metacritic 보유는 **2.4%** 뿐인데 은행 항목에서는 25.7% 다.
   `mc_w` 를 밀면 그 2.4% 가 통째로 올라와, metacritic 페이지가 없는 좋은 인디를
   밀어낸다. `niche_*`·`lowrev_*` 는 정확히 그런 후보로 채워져 있다.
2. **(A) 미달 — 잔차상관 +0.131 이 순위를 바꿀 만큼 크지 않다.**
3. 통과하면 Steam 에도 "격" 축이 생기고, TMDB D-45(미해결)와 같은 문제를
   서로 다른 데이터로 푼 사례가 된다.

## 불확실도

D-52 재채점 후 자카드 ≥0.5 기준 Steam 모순은 **0건**이다. 이 축의 판정에
은행 자기모순이 개입할 여지는 없다. **단 D-25(채점자=제작자)는 그대로다.**
