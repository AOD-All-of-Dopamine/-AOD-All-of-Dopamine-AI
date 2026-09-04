"""X-23 각색 쌍 생성 — 공개 사실로 정한 목록을 코퍼스 id 로 고정한다.

선정 규칙 (관측 전 확정):
  A. 게임↔영상: 아래 프랜차이즈 목록은 공개된 각색 사실이다. 각 쌍의
     게임 쪽 = 그 프랜차이즈에서 Steam 리뷰 수가 가장 많은 작품,
     영상 쪽 = 그 게임의 실제 각색물 중 TMDB vote_count 가 가장 높은 것.
  B. 라이트노벨↔애니: 제목 완전일치 ∧ 웹소설 장르 == "라이트노벨"
     ∧ TMDB 장르에 "애니메이션" 포함. (기계적 규칙, 내 판단 없음)
  C. 한국 웹소설↔영상: 제목 완전일치 ∧ 공개된 각색 사실. (전지적 독자 시점)
어떤 검색 결과도 보기 전에 확정한다.
"""
import pandas as pd, re, json
ST="steam/artifacts/tags_full/dataset.parquet"; TM="tmdb/artifacts/tmdb_v1/dataset.parquet"
WN="/home/ubuntu/aod-webnovel/recommendation/webnovel/artifacts/wn_v4/dataset.parquet"
st=pd.read_parquet(ST,columns=["steam_appid","name","recommendations_total"])
tm=pd.read_parquet(TM,columns=["item_id","name","media","date","vote_count","genres"])
wn=pd.read_parquet(WN,columns=["item_id","name","genres","interest_count"])

# ── A. 게임 ↔ 영상 (공개 각색 사실) ─────────────────────────────────
GAMES = [
 ("위쳐",292030,"tv_71912"),("폴아웃",377160,"tv_106379"),("레지던트 이블",2050650,"movie_1576"),
 ("어쌔신 크리드",812140,"movie_121856"),("헤일로",976730,"tv_52814"),("소닉",213610,"movie_454626"),
 ("FNAF",747660,"movie_507089"),("둠",782330,"movie_8814"),("철권",389730,"movie_42194"),
 ("스트리트 파이터",1364780,"movie_11667"),("니드 포 스피드",1222680,"movie_136797"),
 ("몬스터 헌터",582010,"movie_458576"),("컵헤드",268910,"tv_103786"),("앵그리버드",1001140,"movie_153518"),
 ("툼 레이더",203160,"movie_338970"),("언차티드",1659420,"movie_335787"),("라스트 오브 어스",1888930,"tv_100088"),
 ("히트맨",1659040,"movie_1620"),("맥스 페인",204100,"movie_13051"),("페르시아의 왕자",13600,"movie_9543"),
 ("도타",570,"tv_118956"),("캐슬바니아",2369900,"tv_71024"),("데빌 메이 크라이",601150,"tv_235930"),
 ("모탈 컴뱃",1971870,"movie_460465"),("보더랜드",49520,"movie_365177"),("마인크래프트",1672970,"movie_950387"),
 ("사일런트 힐",2124490,"movie_588"),("사이버펑크",1091500,"tv_105248"),("포스탈",223470,"movie_2728"),
]
# ── B/C. 웹소설 ↔ 영상 (기계 규칙) ──────────────────────────────────
norm=lambda s: re.sub(r"[\s\[\]()<>:·,~!?\-–—’'\"]+","",str(s)).lower()
lst=lambda g: list(g) if hasattr(g,"__iter__") and not isinstance(g,str) else ([g] if g else [])
tmi={}
for _,r in tm.iterrows(): tmi.setdefault(norm(r["name"]),[]).append(r)
NOVELS=[]
for _,r in wn.iterrows():
    k=norm(r["name"])
    if len(k)<4 or k not in tmi: continue
    if "라이트노벨" not in lst(r["genres"]): continue
    cand=[t for t in tmi[k] if "애니메이션" in lst(t["genres"])]
    if not cand: continue
    t=max(cand,key=lambda x:x["vote_count"])
    NOVELS.append((str(r["name"]),int(r["item_id"]),t["item_id"]))
NOVELS.append(("전지적 독자 시점",6020546,"movie_1131759"))   # C: 공개 각색 사실

pairs=[]
for nm,a,b in GAMES:  pairs.append(dict(kind="game",name=nm,steam=a,tmdb=b))
for nm,a,b in NOVELS: pairs.append(dict(kind="novel",name=nm,wn=a,tmdb=b))
# 존재 확인
sids=set(st["steam_appid"].astype(int)); tids=set(tm["item_id"]); wids=set(wn["item_id"].astype(int))
bad=[p for p in pairs if (p.get("steam") and p["steam"] not in sids) or (p.get("wn") and p["wn"] not in wids) or p["tmdb"] not in tids]
assert not bad, bad
json.dump(pairs,open("crossdomain/x23_pairs.json","w"),ensure_ascii=False,indent=1)
print(f"게임↔영상 {len(GAMES)} · 라노벨/웹소설↔영상 {len(NOVELS)} · 합계 {len(pairs)}")
for p in pairs[-len(NOVELS):]: print("  ",p["name"][:34],p["wn"],p["tmdb"])
