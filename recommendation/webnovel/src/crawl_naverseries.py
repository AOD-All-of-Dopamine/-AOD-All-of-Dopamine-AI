# src/crawl_naverseries.py
"""네이버 시리즈 웹소설을 크롤링해 `webnovels.jsonl` 을 만든다.

백엔드의 Java 크롤러(`NaverSeriesCrawler.java`)를 포팅했다. 같은 페이지·같은 필드를 쓰되
출력만 jsonl 로 맞춘 얇은 크롤러다. `crawl_steam.py` 의 이어받기·적응형 레이트 구조를 따른다.

**목록에 10,000개 하드 캡이 있다.** `page=400` 까지는 25개씩 나오고 401부터 0개다.
그래서 전체 목록 하나로는 1만 개가 최대다. 장르(8개) × 완결여부(2)로 슬라이스하면
각 슬라이스가 독립적으로 1만까지 나오므로 그만큼 더 열거된다.
실측(2026-07): 로맨스·로판·판타지·현판·BL 5개 장르가 단독으로 캡에 걸린다
(= 각각 1만 개 이상 존재). 무협 6,275 / 라이트노벨 1,125 / 미스터리 700.

    python -m src.crawl_naverseries --out data/webnovels.jsonl [--limit N] [--rps 0.5]

**19금 작품은 목록에는 나오지만 상세는 못 받는다.** 실측 결과 실제 차단 방식은
`#adult_msg`/`enctp=19` 마커가 아니라 **네이버 로그인 페이지로의 302 리다이렉트**였다
(`nidlogin.login?svctype=128...`). 파일럿에서 처리한 200건 중 107건(53%)이 이 경우다 —
특히 로맨스·BL 슬라이스에 몰려 있다. 마커 검사도 남겨 두지만(로그인 상태에 따라 그 경로로
올 수 있다) 실제로 걸러내는 것은 `is_redirected_away` 다.
따라서 **열거 건수의 절반 정도만 실제 코퍼스가 된다** — 5.8만 열거 → 2.7만 내외.

**1화 날짜는 수집하지 않는다.** Java 크롤러는 `volumeList.series?sortOrder=ASC` 의
`lastVolumeUpdateDate` 첫 등장값을 1화 등록일로 쓰는데, 실측(2026-07-29) 결과 그 필드는
회차별 값이 아니라 응답 시각이다 — 104화짜리 작품의 전 회차가 동일한 값을 갖고, 같은 작품을
ASC/DESC 로 부르면 값이 달라진다(00:00:25 vs 23:00:09). 즉 백엔드의 `firstDate` 는 계속
"오늘"을 저장해 왔다. 이 엔드포인트에 회차별 발행일은 존재하지 않으므로 호출 자체를 뺐다
(작품당 요청이 절반으로 준다). 신작/트렌드 신호가 필요해지면 별도 경로를 찾아야 한다.
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.config import PROJECT_ROOT

BASE = "https://series.naver.com"
LIST_URL = BASE + "/novel/categoryProductList.series"
DETAIL_URL = BASE + "/novel/detail.series"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Referer": BASE + "/",
}

# 목록 슬라이스용 장르 코드. 204 는 결번이다.
GENRE_CODES = {
    "201": "로맨스", "202": "판타지", "203": "미스터리", "205": "라이트노벨",
    "206": "무협", "207": "로판", "208": "현판", "209": "BL",
}
LIST_PAGE_CAP = 400  # 25개 × 400 = 10,000 (그 이상은 서버가 빈 목록을 준다)

# 제목의 [독점] · [단행본] · [시리즈 에디션] 류 태그
_TITLE_TAG = re.compile(r"\s*\[[^\]]+\]\s*")
_WS = re.compile(r"\s+")
_INTEREST = re.compile(r"관심\s*([\d.,]+\s*(?:억|만|천)|[\d,]+)")
_NUM = re.compile(r"\d+(?:\.\d+)?")


class RateLimited(Exception):
    pass


class AdaptiveRate:
    """429 를 맞으면 느려지고, 잘 되면 조금씩 빨라진다. `crawl_steam.AdaptiveRate` 와 동일.

    네이버는 Steam 과 달리 한계가 공개돼 있지 않고 차단이 더 공격적이라 기본 목표를
    0.5 req/s 로 낮춰 잡는다.
    """

    def __init__(self, target_rps: float, min_rps: float = 0.1):
        self.delay = 1.0 / target_rps
        self.min_delay = 1.0 / target_rps
        self.max_delay = 1.0 / min_rps
        self.ok_streak = 0
        self.n_429 = 0

    def on_success(self) -> None:
        self.ok_streak += 1
        if self.ok_streak >= 200 and self.delay > self.min_delay:
            self.delay = max(self.min_delay, self.delay * 0.95)
            self.ok_streak = 0

    def on_429(self) -> float:
        self.n_429 += 1
        self.ok_streak = 0
        self.delay = min(self.max_delay, self.delay * 1.5)
        return min(120.0, 15.0 * min(self.n_429, 4))

    @property
    def rps(self) -> float:
        return 1.0 / self.delay


def parse_korean_count(s: str | None) -> int | None:
    """"2억 5,006만" · "139.3만" · "2.5천" · "1,393,475" 를 정수로.

    Java `parseKoreanCount` 포팅. 네이버는 큰 수를 항상 축약해서 보여준다.
    """
    if not s:
        return None
    s = s.strip().replace(",", "")

    if "억" in s:
        head, _, tail = s.partition("억")
        m = _NUM.search(head)
        if m:
            total = round(float(m.group()) * 100_000_000)
            man = _NUM.search(tail.replace("만", " "))
            if man:
                total += round(float(man.group()) * 10_000)
            return total

    for unit, mult in (("만", 10_000), ("천", 1_000)):
        m = re.search(rf"(\d+(?:\.\d+)?)\s*{unit}", s)
        if m:
            return round(float(m.group(1)) * mult)

    m = re.fullmatch(r"\d+", s)
    return int(m.group()) if m else None


def clean_title(raw: str | None) -> str:
    """[독점] 류 태그 제거. Java `cleanTitle` 포팅."""
    if not raw:
        return ""
    return _WS.sub(" ", _TITLE_TAG.sub(" ", raw)).strip()


def _text(node) -> str:
    return _WS.sub(" ", node.get_text(" ", strip=True)).replace("\xa0", " ").strip() if node else ""


def _meta(soup, prop: str) -> str:
    tag = soup.find("meta", property=prop)
    return (tag.get("content") or "").strip() if tag else ""


def is_adult(soup) -> bool:
    """19금 작품은 상세 내용이 아예 안 내려온다."""
    if soup.select_one("#adult_msg"):
        return True
    enctp = soup.select_one("input[name=enctp]")
    return bool(enctp and enctp.get("value") == "19")


def parse_info_list(soup) -> dict:
    """작품정보란(`ul.end_info li.info_lst > ul`)에서 상태/장르/작가/출판사/이용가를 뽑는다.

    구조가 `<li>연재중</li><li><span>글</span><a>작가</a></li>` 처럼 라벨-값 쌍과
    라벨 없는 항목이 섞여 있다.

    **장르는 라벨이 아니라 링크 대상으로 판정한다.** 라벨 목록(글/출판사/이용가)만 걸러내고
    나머지 `<a>` 를 장르로 보면 크레딧이 새어 들어간다 — 실제로 `그림`(삽화가) 라벨이 있는
    작품에서 삽화가 이름이 장르가 됐다(390건 중 4건). 장르 링크는
    `categoryProductList.series` 로 가고 크레딧 링크는 `/search/search.series` 로 가므로
    href 로 구분하는 쪽이 라벨 화이트리스트보다 안전하다(새 크레딧 라벨이 생겨도 안 샌다).
    """
    out = {"status": "", "genres": [], "author": "", "publisher": "", "age_rating": ""}
    ul = soup.select_one("ul.end_info li.info_lst > ul") or soup.select_one("ul.info_lst")
    if not ul:
        return out

    for li in ul.find_all("li", recursive=False):
        label = _text(li.find("span", recursive=False))
        whole = _text(li)
        if whole in ("연재중", "완결"):
            out["status"] = whole
            continue
        if label == "글":
            a = li.find("a")
            out["author"] = _text(a) if a else whole.removeprefix("글").strip()
            continue
        if label == "출판사":
            a = li.find("a")
            out["publisher"] = _text(a) if a else whole.removeprefix("출판사").strip()
            continue
        if "이용가" in whole:
            out["age_rating"] = whole
            continue
        for a in li.find_all("a"):
            if "categoryProductList.series" not in (a.get("href") or ""):
                continue
            g = _text(a)
            if g and g not in out["genres"]:
                out["genres"].append(g)
    return out


def parse_detail(html: str, product_no: str) -> dict | None:
    """상세 페이지 1건 → 평평한 dict. 19금·제목없음이면 None."""
    soup = BeautifulSoup(html, "lxml")
    if is_adult(soup):
        return None

    title = clean_title(_meta(soup, "og:title") or _text(soup.find("h2")))
    if not title:
        return None

    info = parse_info_list(soup)

    # 시놉시스: 접힘/펼침 두 노드가 있어 Java 는 마지막을 쓴다. 가장 긴 것이 펼친 본문이다.
    synopsis = ""
    for node in soup.select("div.end_dsc ._synopsis") or soup.select("._synopsis"):
        s = _text(node).removesuffix("접기").strip()
        if len(s) > len(synopsis):
            synopsis = s

    # 평점 — 참여자 수가 노출되지 않으므로 단독으로는 신뢰할 수 없다(관심 2에 평점 10.0).
    rating = None
    if m := _NUM.search(_text(soup.select_one("div.score_area"))):
        rating = float(m.group())

    # 관심 수 — Java 의 1차 셀렉터 `a.btn_download > span` 은 현재 페이지에 없다.
    # end_head 텍스트("평점 10.0 관심 2 공유")가 현재의 유일한 경로다. 이게 주 인기도 신호다.
    interest = None
    if m := _INTEREST.search(_text(soup.select_one("div.end_head"))):
        interest = parse_korean_count(m.group(1))
    if interest is None:
        interest = parse_korean_count(_text(soup.select_one("a.btn_download > span")))

    comments = parse_korean_count(_text(soup.select_one("span#commentCount")))

    episodes = None
    if strong := soup.select_one("h5.end_total_episode strong"):
        episodes = parse_korean_count(_text(strong))

    return {
        "product_no": int(product_no),
        "title": title,
        "synopsis": synopsis,
        "genres": info["genres"],
        "author": info["author"],
        "publisher": info["publisher"],
        "age_rating": info["age_rating"],
        "status": info["status"],
        "rating": rating,
        "interest_count": interest,
        "comment_count": comments,
        "episode_count": episodes,
        "cover_image": _meta(soup, "og:image"),
        "url": f"{DETAIL_URL}?productNo={product_no}",
    }


# ---------------------------------------------------------------- 네트워크

def _get(session: requests.Session, url: str, params: dict, timeout: int = 20) -> requests.Response:
    r = session.get(url, params=params, headers=HEADERS, timeout=timeout)
    if r.status_code == 429:
        raise RateLimited()
    return r


def is_redirected_away(r: requests.Response) -> bool:
    """상세 페이지를 못 받은 경우를 판정한다. 실측된 두 가지:

      · 19금 작품   → `nidlogin.login?svctype=128...` (네이버 로그인) — 스킵의 대부분
      · 판매중지    → `/error/stopSale.series`

    requests 가 리다이렉트를 따라가므로 status 는 200 이고, 그 페이지를 그대로 파싱하면
    제목이 'SERIES'(사이트 이름)인 빈 레코드가 코퍼스에 들어간다. 실제로 걸렸던 문제라
    `data_loader.record_to_row` 에도 같은 방어가 하나 더 있다.
    """
    return bool(r.history) or "/error/" in r.url or "nidlogin" in r.url


def list_page_ids(session: requests.Session, params: dict) -> list[str]:
    r = _get(session, LIST_URL, params)
    if r.status_code != 200:
        return []
    return list(dict.fromkeys(re.findall(r"productNo=(\d+)", r.text)))


def enumerate_ids(session: requests.Session, rate: AdaptiveRate, max_pages: int = LIST_PAGE_CAP) -> list[str]:
    """장르 × 완결여부로 슬라이스해 productNo 를 모은다.

    슬라이스마다 10,000개 캡이 따로 걸리므로 전체 목록 하나로 도는 것보다 훨씬 많이 얻는다.
    빈 페이지가 나오면 그 슬라이스는 끝난 것으로 보고 다음으로 넘어간다.
    """
    seen: dict[str, None] = {}
    slices = [
        {"categoryTypeCode": "genre", "genreCode": g, "isFinished": f}
        for g in GENRE_CODES
        for f in ("false", "true")
    ]
    for sl in slices:
        name = GENRE_CODES[sl["genreCode"]]
        label = f"{name}/{'완결' if sl['isFinished'] == 'true' else '연재중'}"
        before = len(seen)
        for page in range(1, max_pages + 1):
            try:
                ids = list_page_ids(session, {**sl, "page": page})
                rate.on_success()
            except RateLimited:
                wait = rate.on_429()
                print(f"    429 — {wait:.0f}초 대기 ({rate.rps:.2f} req/s)", flush=True)
                time.sleep(wait)
                continue
            except Exception:
                break
            if not ids:
                break
            seen.update(dict.fromkeys(ids))
            time.sleep(rate.delay)
        print(f"  {label:16} +{len(seen) - before:>6,}  (누적 {len(seen):,})", flush=True)
    return list(seen)


def already_done(path: Path) -> set[int]:
    """이어받기 — 이미 저장한 product_no. 깨진 줄은 건너뛴다."""
    done: set[int] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                pid = json.loads(line).get("product_no")
                if isinstance(pid, int):
                    done.add(pid)
            except json.JSONDecodeError:
                continue
    return done


def fetch_one(session: requests.Session, product_no: str) -> dict | None:
    r = _get(session, DETAIL_URL, {"productNo": product_no})
    if r.status_code != 200 or is_redirected_away(r):
        return None
    return parse_detail(r.text, product_no)


def main():
    ap = argparse.ArgumentParser(description="네이버 시리즈 웹소설 크롤러")
    ap.add_argument("--out", default="data/webnovels.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="앞에서 N개만 (파일럿용)")
    ap.add_argument("--rps", type=float, default=0.5, help="초당 요청 수 목표")
    ap.add_argument("--list-pages", type=int, default=LIST_PAGE_CAP, help="슬라이스당 최대 목록 페이지")
    ap.add_argument("--ids-cache", default="data/product_ids.json", help="열거 결과 캐시")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    cache = Path(args.ids_cache)
    if not cache.is_absolute():
        cache = PROJECT_ROOT / cache

    session = requests.Session()
    rate = AdaptiveRate(args.rps)

    if cache.exists():
        ids = json.loads(cache.read_text())
        print(f"목록 캐시 재사용: {len(ids):,}건 ({cache})", flush=True)
    else:
        print("작품 목록 열거 중 (장르 × 완결여부 슬라이스)...", flush=True)
        ids = enumerate_ids(session, rate, args.list_pages)
        cache.write_text(json.dumps(ids))
        print(f"열거 완료: {len(ids):,}건 → {cache}", flush=True)

    done = already_done(out)
    todo = [i for i in ids if int(i) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"전체 {len(ids):,} | 이미 받음 {len(done):,} | 이번에 받을 것 {len(todo):,}", flush=True)
    if not todo:
        return

    t0 = time.time()
    saved = skipped = failed = 0
    n = 0
    queue = list(todo)

    with open(out, "a", encoding="utf-8") as f:
        while queue:
            pid = queue.pop(0)
            n += 1
            try:
                rec = fetch_one(session, pid)
                if rec is None:
                    skipped += 1  # 19금 · 판매중지 · 제목 없음
                else:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    saved += 1
                rate.on_success()
            except RateLimited:
                wait = rate.on_429()
                print(f"  429 (#{rate.n_429}) — {wait:.0f}초 대기, {rate.rps:.2f} req/s 로 조정 "
                      f"| 진행 {n:,}/{len(todo):,}", flush=True)
                f.flush()
                time.sleep(wait)
                queue.insert(0, pid)
                n -= 1
                continue
            except Exception as e:
                failed += 1
                if failed <= 20:
                    print(f"  실패 productNo={pid}: {type(e).__name__}", flush=True)

            if n % 200 == 0:
                el = time.time() - t0
                obs = n / el if el else 0
                eta = (len(todo) - n) / obs / 3600 if obs else 0
                f.flush()
                print(f"  {n:,}/{len(todo):,}  저장 {saved:,} 제외 {skipped:,} 실패 {failed:,} "
                      f"429 {rate.n_429} | 실측 {obs:.2f} req/s | 남은 {eta:.1f}h", flush=True)
            time.sleep(rate.delay)

    el = time.time() - t0
    print(f"\n완료: 저장 {saved:,} / 제외(19금·삭제) {skipped:,} / 실패 {failed:,} / 429 {rate.n_429}")
    print(f"소요 {el / 3600:.2f}시간 → {out}")


if __name__ == "__main__":
    main()
