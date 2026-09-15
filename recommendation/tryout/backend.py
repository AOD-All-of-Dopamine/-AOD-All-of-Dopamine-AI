"""실유저 시험용 백엔드 — 플랫폼 하나를 HTTP 로 띄운다.

**평가와 같은 경로를 쓴다.** 시험에서 본 목록이 곧 P@k 를 낸 그 목록이어야 하므로
확정 설정(Steam quality_w=0.50·tag_w=0.40 · TMDB PRODUCTION · 웹소설 기본값)을 그대로 넣는다.

    python backend.py --platform steam   --port 8011
    python backend.py --platform tmdb    --port 8012
    python backend.py --platform webnovel --port 8013
"""
import argparse, json, sys, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

AOD = Path("/home/ubuntu/-AOD-All-of-Dopamine-AI/recommendation")
WN = Path("/home/ubuntu/aod-webnovel/recommendation/webnovel")


def _num(v, default=0):
    """nullable Int64 의 pd.NA / NaN / None 을 삼킨다.

    `v or 0` 은 pd.NA 에서 TypeError 를 던진다. Steam 의 recommendations_total 은
    87% 가 결측이라(D-39) 이 경로를 반드시 밟는다.
    """
    try:
        if v is None or pd.isna(v): return default
        return type(default)(v)
    except (TypeError, ValueError):
        return default


def _clean(v):
    """numpy/NaN 을 JSON 이 먹을 수 있는 것으로 바꾼다."""
    if v is None: return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else round(f, 3)
    if isinstance(v, (np.ndarray, list, tuple)): return [_clean(x) for x in v]
    if isinstance(v, float): return None if np.isnan(v) else round(v, 3)
    return v


class Steam:
    label = "Steam"
    id_field = "appid"

    def __init__(self):
        sys.path.insert(0, str(AOD / "steam"))
        import os; os.environ.setdefault("AOD_ARTIFACTS", "artifacts/tags_full")
        os.chdir(AOD / "steam")
        from src.personalized_retrieve import build_components, run_multi
        from src.config import PRODUCTION as STEAM_PROD
        self._run = run_multi
        self._strategy = STEAM_PROD["strategy"]   # 리터럴 두 벌을 두지 않는다
        self.comps = build_components()   # 인자 없음 = config.PRODUCTION 확정값 (D-55)
        self.ds = pd.read_parquet(AOD / "steam/artifacts/tags_full/dataset.parquet")
        self.by_id = self.ds.set_index("steam_appid")
        self._lname = self.ds["name"].fillna("").str.lower()

    def _card(self, appid):
        try: r = self.by_id.loc[int(appid)]
        except KeyError: return None
        # 한 편이 터져도 목록 전체가 죽지 않게 한다.
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        tags = r.get("tags")
        tags = list(tags)[:5] if tags is not None and len(tags) else []
        n = _num(r.get("recommendations_total"))
        return dict(id=int(appid), name=str(r["name"]),
                    meta=" · ".join(str(t) for t in tags) or "-",
                    stat=f"리뷰 {n:,}" if n else "리뷰 수 미보고",
                    link=f"https://store.steampowered.com/app/{int(appid)}")

    def search(self, q, n=20):
        hit = self.ds[self._lname.str.contains(q.lower(), regex=False)]
        hit = hit.sort_values("recommendations_total", ascending=False).head(n)
        return [c for c in (self._card(a) for a in hit["steam_appid"]) if c]

    def recommend(self, seeds, k=20):
        res = self._run([int(s) for s in seeds], strategies=[self._strategy], top_n=k,
                        components=self.comps, postprocess=True)
        df = list(res.values())[0]
        if isinstance(df, dict): df = pd.DataFrame(df.get("recommendations", df))
        col = next(c for c in ("steam_appid", "appid", "candidate_appid") if c in df.columns)
        return [c for c in (self._card(a) for a in df[col].head(k)) if c]


class Tmdb:
    label = "TMDB"
    id_field = "row"

    def __init__(self):
        sys.path.insert(0, str(AOD / "tmdb"))
        import os; os.environ.setdefault("AOD_ARTIFACTS", "artifacts/tmdb_v1")
        os.chdir(AOD / "tmdb")
        from src.personalized_retrieve import build_components, recommend
        from src.config import PRODUCTION
        self.PROD = PRODUCTION
        self._rec = recommend
        self.comps = build_components(**{k: v for k, v in PRODUCTION.items() if k != "strategy"})
        idx = pd.read_parquet(AOD / "tmdb/artifacts/tmdb_v1/corpus_index.parquet").sort_values("embedding_row")
        ds = pd.read_parquet(AOD / "tmdb/artifacts/tmdb_v1/dataset.parquet").set_index("item_id")
        self.ds = ds.loc[idx["item_id"].to_numpy()].reset_index()
        self._lname = self.ds["name"].fillna("").str.lower()
        # 표시명 — 한글·라틴 문자가 없는 원제(데바나가리·텔루구 등)만 영어 제목으로 보여준다.
        # 랭킹·임베딩과 무관하다. 파일이 없으면 원래 이름 그대로 (scripts/fetch_display_titles.py).
        dn = AOD / "tmdb/artifacts/tmdb_v1/display_names.parquet"
        self._display = dict(pd.read_parquet(dn).itertuples(index=False, name=None)) if dn.exists() else {}

    def _card(self, row):
        r = self.ds.iloc[int(row)]
        g = r["genres"]
        g = list(g) if g is not None and len(g) else []
        media = "영화" if r["media"] == "movie" else "TV"
        # 연도 — 코퍼스의 9.5% 가 제목이 겹친다(퍼니 게임 1997/2007 · 왓치맨 2009 영화/2019 TV).
        # 서빙 top-10 카드의 14% 가 그런 작품이라 연도 없이는 어느 작품인지 못 가린다. 검색(시드 고르기)도 같다.
        y = str(r["date"])[:4] if r.get("date") is not None and str(r["date"])[:4].isdigit() else ""
        return dict(id=int(row), name=str(self._display.get(r["item_id"], r["name"])),
                    meta=f"{media}{' · ' + y if y else ''} · " + (", ".join(str(x) for x in g) or "-"),
                    stat=f"평점 {_num(r['vote_average'], 0.0):.1f} · 투표 {_num(r['vote_count']):,}",
                    link=f"https://www.themoviedb.org/{r['media']}/{int(r['tmdb_id'])}")

    def search(self, q, n=20):
        hit = self.ds[self._lname.str.contains(q.lower(), regex=False)]
        hit = hit.sort_values("vote_count", ascending=False).head(n)
        return [self._card(i) for i in hit.index]

    def recommend(self, seeds, k=20):
        df = self._rec([int(s) for s in seeds], components=self.comps, top_n=k,
                       strategy=self.PROD["strategy"], postprocess_on=True)
        return [self._card(r) for r in df["row"].head(k)]


class Webnovel:
    label = "웹소설"
    id_field = "item_id"

    def __init__(self):
        sys.path.insert(0, str(WN))
        # wn_v6(29,494편) — 웹소설 확정값(W-1~W-4)을 전부 이 코퍼스에서 쟀다.
        # 2026-09-15 전까지 wn_v4(파일럿 7,062편)로 떠서 시험 목록 ≠ 평가 목록이었다.
        import os; os.environ.setdefault("AOD_ARTIFACTS", "artifacts/wn_v6")
        os.chdir(WN)
        from src.wn_eval import Engine
        self.eng = Engine()
        self.ds = self.eng.ds
        self.by_id = self.ds.set_index("item_id")
        self._lname = self.ds["name"].fillna("").str.lower()
        # item_id 의 실제 dtype 이 int 일 수 있다. URL 로는 문자열로 오므로 되돌린다.
        self._native = {str(i): i for i in self.ds["item_id"]}

    def _card(self, iid):
        r = self.by_id.loc[iid]
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        g = r["genres"]
        g = list(g) if g is not None and len(g) else []
        rating = _num(r["rating"], 0.0)
        return dict(id=str(iid), name=str(r["name"]),
                    meta=(", ".join(str(x) for x in g) or "-") + f" · {r['author']}",
                    stat=(f"평점 {rating:.1f} · " if rating > 0 else "평점 없음 · ")
                         + f"관심 {_num(r['interest_count']):,} · {_num(r['episode_count'])}화",
                    link=r["url"])

    def search(self, q, n=20):
        hit = self.ds[self._lname.str.contains(q.lower(), regex=False)]
        hit = hit.sort_values("interest_count", ascending=False).head(n)
        return [self._card(i) for i in hit["item_id"]]

    def recommend(self, seeds, k=20):
        ids = [self._native.get(str(s), s) for s in seeds]
        df = self.eng.recommend(ids, k=k)                       # 확정 = 전부 기본값
        return [self._card(i) for i in df["item_id"].head(k)]


class Webtoon:
    """2026-09-15 추가 — 네 플랫폼 중 웹툰만 시험대가 없어 사람이 직접 볼 길이 없었다.
    평가(T-1~T-9)와 같은 `Engine.recommend` 기본값 = PRODUCTION(tag_w 0.2 등)으로 돈다."""
    label = "웹툰"
    id_field = "item_id"

    def __init__(self):
        sys.path.insert(0, str(AOD / "webtoon"))
        import os; os.environ.setdefault("AOD_WT_ARTIFACTS", str(AOD / "webtoon/artifacts/wt_v1"))
        os.chdir(AOD / "webtoon")
        from src.personalized_retrieve import Engine
        self.eng = Engine()
        self.ds = self.eng.ds
        self.by_id = self.ds.set_index("item_id")
        self._lname = self.ds["name"].fillna("").str.lower()

    def _card(self, iid):
        try: r = self.by_id.loc[int(iid)]
        except (KeyError, ValueError): return None
        if isinstance(r, pd.DataFrame): r = r.iloc[0]
        g = list(r["genres"]) if r["genres"] is not None else []
        tags = [t for t in (list(r["tags"]) if r["tags"] is not None else [])
                if not str(t).startswith("완결") and t not in g][:4]     # 태그에 장르명이 또 들어 있다
        who = ", ".join(list(r["artists"])[:2]) if r["artists"] is not None and len(r["artists"]) else str(r["author"] or "")
        state = "휴재" if bool(r["rest"]) else ("완결" if bool(r["finished"]) else "연재중")
        return dict(id=int(iid), name=str(r["name"]),
                    meta=" · ".join(x for x in [", ".join(map(str, g)), ", ".join(map(str, tags)), who] if x) or "-",
                    stat=f"관심 {_num(r['favorite_count']):,} · {_num(r['episode_count'])}화 · {state}",
                    link=r.get("url"))

    def search(self, q, n=20):
        hit = self.ds[self._lname.str.contains(q.lower(), regex=False)]
        hit = hit.sort_values("favorite_count", ascending=False).head(n)
        return [c for c in (self._card(i) for i in hit["item_id"]) if c]

    def recommend(self, seeds, k=20):
        df = self.eng.recommend([int(s) for s in seeds], k=k)   # 확정 = 전부 기본값
        return [c for c in (self._card(i) for i in df["item_id"].head(k)) if c]


ENGINES = {"steam": Steam, "tmdb": Tmdb, "webnovel": Webnovel, "webtoon": Webtoon}


def make_handler(eng):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def _send(self, obj, code=200):
            b = json.dumps(obj, ensure_ascii=False, default=_clean).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path); q = parse_qs(u.query)
            try:
                if u.path == "/health":
                    return self._send({"ok": True, "label": eng.label, "n": len(eng.ds)})
                if u.path == "/search":
                    s = (q.get("q") or [""])[0].strip()
                    return self._send(eng.search(s, int((q.get("n") or [20])[0])) if s else [])
                if u.path == "/card":
                    # 크로스 탭이 시드 이름을 보여주는 데 쓴다 — 시드가 안 보이면 채점이 성립하지 않는다 (D-58)
                    ids = q.get("id") or []
                    out = []
                    for i in ids:
                        key = eng._native.get(str(i), i) if hasattr(eng, "_native") else int(i)
                        c = eng._card(key)
                        if c: out.append(c)
                    return self._send(out)
                if u.path == "/recommend":
                    seeds = [s for s in (q.get("seed") or []) if s != ""]
                    if not seeds: return self._send({"error": "시드가 없습니다"}, 400)
                    return self._send(eng.recommend(seeds, int((q.get("k") or [20])[0])))
            except Exception as e:
                traceback.print_exc()
                return self._send({"error": f"{type(e).__name__}: {e}"}, 500)
            self._send({"error": "not found"}, 404)
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=list(ENGINES))
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    eng = ENGINES[a.platform]()
    print(f"[{a.platform}] 준비됨 · 코퍼스 {len(eng.ds):,} · :{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(eng)).serve_forever()
