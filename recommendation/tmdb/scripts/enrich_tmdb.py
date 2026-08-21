"""2단계 — discover 목록에 키워드를 붙인다. **스레드 풀로 병렬 수집.**

순차로 돌려보니 요청당 왕복 300ms 대라 6만 건에 5시간이 넘었다.
TMDB 는 초당 수십 요청을 허용하므로 워커 16개로 나눈다.
쓰기는 락으로 직렬화하고 500건마다 flush 해서 중간에 죽어도 이어받는다.

키워드 커버리지는 vote_count 에 강하게 의존한다(vote 30~50 구간 33%가 0개).
거르지 않고 n_keywords 로 남겨 사후 검증한다 — 사전 등록 h43 보정 1.
"""
import json, sys, threading, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env') if l.startswith('TMDB_API_KEY')][0]
DATA = ROOT / 'data'
WORKERS = 16

_lock = threading.Lock()
_done = 0


def get(path, **q):
    q['api_key'] = KEY
    url = f"https://api.themoviedb.org/3{path}?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:                      # 레이트 리밋 — 물러선다
                time.sleep(2.0 * (attempt + 1))
                continue
            if attempt == 4:
                return None
            time.sleep(1.0 * (attempt + 1))
        except Exception:
            if attempt == 4:
                return None
            time.sleep(1.0 * (attempt + 1))
    return None


def fetch(args):
    media, i, base, kwkey = args
    d = get(f'/{media}/{i}', language='ko-KR', append_to_response='keywords')
    if d is None:
        return None
    out = {
        'id': i, 'media': media,
        'title': d.get('title') or d.get('name') or '',
        'overview': (d.get('overview') or '').strip(),
        'overview_en': '',
        'genres': [g['name'] for g in d.get('genres', [])],
        'keywords': [k['name'] for k in (d.get('keywords') or {}).get(kwkey, [])],
        'vote_count': d.get('vote_count', 0), 'vote_average': d.get('vote_average', 0.0),
        'date': d.get('release_date') or d.get('first_air_date') or '',
        'adult': bool(d.get('adult', False)), 'year': base.get('_year'),
    }
    if not out['overview']:
        e = get(f'/{media}/{i}', language='en-US')
        if e:
            out['overview_en'] = (e.get('overview') or '').strip()
    return out


def main(media):
    global _done
    src, dst = DATA / f'{media}_discover.jsonl', DATA / f'{media}_enriched.jsonl'
    kwkey = 'keywords' if media == 'movie' else 'results'
    seen = set()
    if dst.exists():
        for line in open(dst):
            try:
                seen.add(json.loads(line)['id'])
            except Exception:
                pass
    rows = {}
    for line in open(src):
        r = json.loads(line)
        rows[r['id']] = r
    todo = [(media, i, rows[i], kwkey) for i in rows if i not in seen]
    print(f"[{media}] 목록 {len(rows):,} · 완료 {len(seen):,} · 남음 {len(todo):,} · 워커 {WORKERS}", flush=True)
    _done = 0
    t0 = time.time()
    with open(dst, 'a') as f, ThreadPoolExecutor(WORKERS) as ex:
        for out in ex.map(fetch, todo, chunksize=8):
            if out is None:
                continue
            with _lock:
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
                _done += 1
                if _done % 1000 == 0:
                    f.flush()
                    el = time.time() - t0
                    print(f"[{media}] {_done:,}/{len(todo):,}  {_done/el:.1f}건/s  "
                          f"남은 시간 {(len(todo)-_done)/max(_done/el,.01)/60:.0f}분", flush=True)
    print(f"[{media}] 완료 → {dst}", flush=True)


if __name__ == '__main__':
    for m in (sys.argv[1:] or ['movie', 'tv']):
        main(m)
