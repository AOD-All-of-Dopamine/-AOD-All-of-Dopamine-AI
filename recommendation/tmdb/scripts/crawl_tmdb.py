"""TMDB 영화·드라마 코퍼스 수집.

설계 결정(측정 전에 적는다):

1. **vote_count >= 30.** 임계 없이 받으면 영화만 117만 건인데 대부분 줄거리가
   비어 있다. 30 은 영화 5.0만·드라마 0.96만으로 Steam(17.4만)과 같은 자릿수이고,
   이 구간은 한국어 줄거리 커버리지가 100% 다(표본 40건 확인).
   코퍼스 크기가 top-1 유사도를 좌우한다는 건 이미 측정돼 있으므로
   (top1 ≈ 0.0217·ln(n) + 0.5169, R²=0.999) Steam 과 비교할 때는
   Steam 을 같은 크기로 서브샘플링해서 맞춘다.

2. **한국어 우선, 없으면 영어.** 웹소설 코퍼스는 순한국어다. 언어가 섞이면
   그 자체가 교란이 될 수 있으므로 `lang` 컬럼에 기록해 사후 검증한다.

3. **discover 는 500페이지(1만 건) 상한이 있다.** 연도로 쪼개고,
   한 연도가 상한을 넘으면 월 단위로 다시 쪼갠다.

4. 키워드(Steam 의 tags 에 대응)는 아이템당 1요청이라 2단계로 분리한다.
   1단계 discover 로 목록을 확보하고, 2단계에서 키워드를 붙인다.
   중간에 죽어도 이어서 받도록 done 슬라이스를 캐시에 남긴다.
"""
import json, os, sys, time, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env') if l.startswith('TMDB_API_KEY')][0]
OUT = ROOT / 'data'
OUT.mkdir(exist_ok=True)
VOTE_MIN = 30


def get(path, **q):
    q['api_key'] = KEY
    url = f"https://api.themoviedb.org/3{path}?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))


def discover(media, lo, hi):
    """[lo, hi] 구간을 받는다. 500페이지를 넘으면 호출부가 더 쪼갠다."""
    dk = 'primary_release_date' if media == 'movie' else 'first_air_date'
    q = {'language': 'ko-KR', 'sort_by': 'vote_count.desc',
         'vote_count.gte': VOTE_MIN, f'{dk}.gte': lo, f'{dk}.lte': hi}
    first = get(f'/discover/{media}', page=1, **q)
    pages = min(first['total_pages'], 500)
    if first['total_pages'] > 500:
        return None, first['total_results']          # 호출부가 쪼개도록 신호
    rows = list(first['results'])
    for p in range(2, pages + 1):
        rows += get(f'/discover/{media}', page=p, **q)['results']
    return rows, first['total_results']


def slices(media):
    for year in range(1900, 2027):
        lo, hi = f'{year}-01-01', f'{year}-12-31'
        rows, total = discover(media, lo, hi)
        if rows is None:                              # 1만 건 초과 → 월별로
            for m in range(1, 13):
                last = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
                r2, _ = discover(media, f'{year}-{m:02d}-01', f'{year}-{m:02d}-{last}')
                if r2:
                    yield year, r2
        elif rows:
            yield year, rows


def main():
    for media in ('movie', 'tv'):
        path = OUT / f'{media}_discover.jsonl'
        done = set()
        if path.exists():
            for line in open(path):
                done.add(json.loads(line)['_year'])
        n = sum(1 for _ in open(path)) if path.exists() else 0
        with open(path, 'a') as f:
            for year, rows in slices(media):
                if year in done:
                    continue
                for r in rows:
                    r['_year'] = year
                    r['_media'] = media
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                    n += 1
                f.flush()
                print(f"[{media}] {year} +{len(rows):4d}  누적 {n:,}", flush=True)
        print(f"[{media}] 완료 총 {n:,}건 → {path}", flush=True)


if __name__ == '__main__':
    main()
