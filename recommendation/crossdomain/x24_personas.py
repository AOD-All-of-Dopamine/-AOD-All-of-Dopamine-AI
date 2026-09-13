"""X-24 근거 있는 교차 페르소나 — 조합이 전부 공개 사실인 val 보완 집합.

만드는 규칙 (관측 전 확정):
  1. 페르소나 하나 = **IP 하나**. 교차 플랫폼 조합이 "같은 작품"이므로 내 취향 추측이 없다.
  2. 게임 IP: Steam 쪽 = 프랜차이즈 정규식 일치 ∧ 리뷰>0 중 상위 3,
     TMDB 쪽 = 그 IP 의 각색물 중 vote_count 상위 3.
  3. 라이트노벨 IP: 웹소설 쪽 = X-23 에서 봉인한 원작(+같은 프랜차이즈 형제 있으면 상위 3),
     TMDB 쪽 = 그 애니.
  4. **동명이인 제거**: 정규식에 걸렸지만 그 IP 가 아닌 것은 공개 사실로 뺀다(아래 DROP).
     이건 취향 판단이 아니라 "이 작품이 그 프랜차이즈인가"라는 사실 확인이다.
  5. 시드는 플랫폼당 최대 3. 결과를 본 뒤에는 목록을 고치지 않는다.
"""
import pandas as pd, json, re
st=pd.read_parquet("steam/artifacts/tags_full/dataset.parquet",columns=["steam_appid","name","recommendations_total"])
tm=pd.read_parquet("tmdb/artifacts/tmdb_v1/dataset.parquet",columns=["item_id","name","vote_count"])
wn=pd.read_parquet("/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4/dataset.parquet",
                   columns=["item_id","name","interest_count"])
st["rc"]=st["recommendations_total"].fillna(0)

# 그 IP 가 아닌 것 (공개 사실로 제외)
DROP_S={1158690,1113780,2527540,   # Uncharted Ocean/Tides — Naughty Dog 언차티드 아님
        2193990,                    # Monster Hunters: Frost Giant — 캡콤 아님
        4231820,                    # Castlevania: Belmont's Curse — 코나미 아님
        1533560,2198280}            # Warcraft 유사명
DROP_T={"movie_795514","movie_1305642",   # 폴아웃(2021/2024 영화) — 아마존 시리즈 아님
        "movie_17478",                     # 둠(2004) — 프랜차이즈 확인 불가
        "movie_207774","movie_13548",      # 보더랜드(2007/2014) — 게임 각색 아님
        "movie_974635","movie_808023",     # 히트맨(2024/2021) — 게임 각색 아님
        "movie_390582"}                    # 라스트 오브 어스(2016) — HBO/너티독 아님

FR={"witcher":("위쳐",r"^The Witcher",r"^위쳐"),"fallout":("폴아웃",r"^Fallout",r"^폴아웃$"),
"resident_evil":("레지던트 이블",r"^BIOHAZARD|^Resident Evil|^Biohazard",r"^레지던트 이블"),
"assassins_creed":("어쌔신 크리드",r"^Assassin.s Creed",r"^어쌔신 ?크리드"),"halo":("헤일로",r"^Halo",r"^헤일로"),
"sonic":("소닉",r"^Sonic",r"^수퍼 소닉"),"fnaf":("프레디의 피자가게",r"^Five Nights at Freddy",r"^프레디의 피자가게"),
"doom":("둠",r"^DOOM|^Doom",r"^둠$|^둠:"),"tekken":("철권",r"^TEKKEN|^Tekken",r"^철권$"),
"street_fighter":("스트리트 파이터",r"^Street Fighter|^Ultra Street Fighter",r"^스트리트 파이터"),
"nfs":("니드 포 스피드",r"^Need for Speed",r"^니드 포 스피드"),
"monster_hunter":("몬스터 헌터",r"^Monster Hunter|^MONSTER HUNTER",r"^몬스터 헌터"),
"tomb_raider":("툼 레이더",r"Tomb Raider",r"^툼 레이더"),
"last_of_us":("라스트 오브 어스",r"^The Last of Us",r"^더 라스트 오브 어스"),
"hitman":("히트맨",r"^HITMAN|^Hitman",r"^히트맨"),"max_payne":("맥스 페인",r"^Max Payne",r"^맥스 페인"),
"prince_persia":("페르시아의 왕자",r"^Prince of Persia",r"^페르시아의 왕자"),
"castlevania":("캐슬바니아",r"Castlevania",r"^캐슬바니아"),
"devil_may_cry":("데빌 메이 크라이",r"Devil May Cry|^DmC",r"데빌 메이 크라이"),
"mortal_kombat":("모탈 컴뱃",r"^Mortal Kombat",r"^모탈 컴뱃"),"borderlands":("보더랜드",r"^Borderlands",r"^보더랜드$"),
"minecraft":("마인크래프트",r"^Minecraft",r"마인크래프트"),
"silent_hill":("사일런트 힐",r"^SILENT HILL|^Silent Hill",r"사일런트 힐"),
"postal":("포스탈",r"^POSTAL|^Postal",r"^포스탈$"),"uncharted":("언차티드",r"^UNCHARTED",r"^언차티드$"),
"cuphead":("컵헤드",r"^Cuphead",r"^컵헤드"),"angry_birds":("앵그리버드",r"Angry Birds",r"^앵그리 ?버드"),
"dota":("도타",r"^Dota 2$",r"^도타:"),"cyberpunk":("사이버펑크",r"^Cyberpunk 2077$",r"^사이버펑크")}

P=[]
for key,(nm,sp,tp) in FR.items():
    s=st[st["name"].str.contains(sp,case=False,na=False,regex=True) & (st["rc"]>0)]
    s=s[~s["steam_appid"].astype(int).isin(DROP_S)].sort_values("rc",ascending=False).head(3)
    t=tm[tm["name"].astype(str).str.contains(tp,na=False,regex=True) & (tm["vote_count"]>0)]
    t=t[~t["item_id"].isin(DROP_T)].sort_values("vote_count",ascending=False).head(3)
    if len(s)==0 or len(t)==0: print("건너뜀",key); continue
    P.append(dict(pid=f"g_{key}",kind="game",label=f"{nm} IP 팬",
        steam=[int(x) for x in s["steam_appid"]], tmdb=list(t["item_id"]), wn=[],
        seed_names=dict(steam=[x[:40] for x in s["name"]], tmdb=[str(x)[:40] for x in t["name"]])))

# 라이트노벨 IP — X-23 봉인 쌍 + 같은 프랜차이즈 형제(제목 접두 8자 일치)
pairs=json.load(open("crossdomain/x23_pairs.json"))
for p in [x for x in pairs if x["kind"]=="novel"]:
    base=str(p["name"]); pre=re.sub(r"[\s~!]+","",base)[:8]
    sib=wn[wn["name"].astype(str).str.replace(r"[\s~!]+","",regex=True).str.startswith(pre)]
    sib=sib.sort_values("interest_count",ascending=False).head(3)
    ids=[int(p["wn"])]+[int(x) for x in sib["item_id"] if int(x)!=int(p["wn"])]
    ids=ids[:3]
    tt=tm[tm["item_id"]==p["tmdb"]]
    P.append(dict(pid=f"n_{p['tmdb']}",kind="novel",label=f"{base[:24]} 원작 팬",
        steam=[], tmdb=[p["tmdb"]], wn=ids,
        seed_names=dict(wn=[str(wn[wn['item_id']==i]['name'].iloc[0])[:40] for i in ids],
                        tmdb=[str(tt['name'].iloc[0])[:40]])))
json.dump(P,open("crossdomain/x24_personas.json","w"),ensure_ascii=False,indent=1)
print(f"\n근거 페르소나 {len(P)}개 (게임 {sum(1 for x in P if x['kind']=='game')} · 라노벨 {sum(1 for x in P if x['kind']=='novel')})")
import collections
c=collections.Counter((len(x["steam"]),len(x["tmdb"]),len(x["wn"])) for x in P)
print("시드 구성 (steam,tmdb,wn):", dict(c))
