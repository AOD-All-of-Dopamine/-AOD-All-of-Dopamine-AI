"""시험 페이지 + 백엔드 3개로의 프록시. 표준 라이브러리만 쓴다.

    python gateway.py --port 8000

프록시를 두는 이유는 CORS 를 없애기 위해서다. 페이지도, /api/* 도 같은 오리진에서 나온다.
채점은 `recommendation/eval/user_grades.jsonl` 에 줄단위로 쌓인다 — **내가 매긴 은행과
섞지 않는다.** 사용자가 매긴 등급은 D-25(채점자=설계자)를 치는 유일한 독립 증거다.
"""
import argparse, json, time, sys
from urllib.parse import parse_qs
# 크로스 도메인 믹싱 — 세 백엔드의 top-N 카드를 받아 순위만으로 섞는다 (crossdomain/DESIGN.md)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "crossdomain"))
from profiles import PROFILES as XPROFILES
from mix import RULES as XRULES
XSEEDS = json.loads((__import__("pathlib").Path(__file__).resolve().parent.parent / "crossdomain" / "seed_index.json").read_text())
XPLAT = {"steam": "steam", "tmdb": "tmdb", "wn": "webnovel"}   # 프로필 키 → 백엔드 이름
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

HERE = Path(__file__).resolve().parent
GRADES = HERE.parent / "eval" / "user_grades.jsonl"
PORTS = {"steam": 8011, "tmdb": 8012, "webnovel": 8013}


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        # 채점 POST 가 도달했는지 추적할 수 있어야 한다 — 등급이 사라졌는데 확인할 길이 없었다 (D-74)
        if "POST" in (a[0] if a else "") or "/grade" in fmt % a if a else False:
            sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % a}\n")

    def _bytes(self, b, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code=200):
        self._bytes(json.dumps(obj, ensure_ascii=False).encode(),
                    "application/json; charset=utf-8", code)

    def do_GET(self):
        try:
            self._get()
        except Exception as e:                      # 핸들러가 죽으면 응답이 비어
            import traceback; traceback.print_exc()  # 페이지에 원인이 안 보인다
            try: self._json({"error": f"게이트웨이 오류 — {type(e).__name__}: {e}"}, 500)
            except Exception: pass

    def _get(self):
        # http.server 는 요청 경로를 latin-1 로 디코드한다. 한글을 그대로 친 URL 은
        # 여기서 깨진 문자로 들어오므로 원래 바이트로 되돌려 UTF-8 로 읽는다.
        raw = self.path
        try: raw = raw.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError): pass
        u = urlparse(raw)
        if u.path == "/status":
            out = {}
            for p, port in PORTS.items():
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                        out[p] = json.loads(r.read())
                except Exception as e:
                    out[p] = {"ok": False, "why": str(e)}
            return self._json(out)
        if u.path in ("/", "/index.html"):
            return self._bytes((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/cross/profiles":
            return self._json([dict(pid=p["pid"], steam=p["steam"], tmdb=p["tmdb"], wn=p["wn"]) for p in XPROFILES])
        if u.path == "/api/cross/recommend":
            return self._cross(parse_qs(u.query))
        if u.path.startswith("/api/"):
            _, _, plat, *rest = u.path.split("/")
            if plat not in PORTS: return self._json({"error": "unknown platform"}, 404)
            # 쿼리를 그대로 붙이면 비ASCII 에서 urlopen 이 UnicodeEncodeError 로 죽는다.
            # 브라우저는 encodeURIComponent 를 쓰므로 페이지 경로는 안전하지만,
            # 손으로 URL 을 치면 여기로 온다. 안전문자만 남기고 다시 인코딩한다.
            tail = "/" + "/".join(rest) + (f"?{quote(u.query, safe='=&%')}" if u.query else "")
            for attempt in (1, 2):
                try:
                    with urlopen(f"http://127.0.0.1:{PORTS[plat]}{tail}", timeout=180) as r:
                        return self._bytes(r.read(), "application/json; charset=utf-8")
                except HTTPError as e:
                    # 백엔드가 **응답은 했다.** 죽었다고 말하면 안 된다 — 그쪽이 실은
                    # 무슨 예외를 던졌는지 그대로 넘긴다.
                    try:    body = json.loads(e.read())
                    except Exception: body = {"error": f"HTTP {e.code}"}
                    body["error"] = f"{plat} 백엔드 오류 — {body.get('error', e)}"
                    # **백엔드의 상태 코드를 그대로 넘긴다.** 전부 500 으로 뭉개면
                    # "시드가 없습니다"(400, 사용자 입력 문제)와 진짜 서버 예외(500)가
                    # 화면에서 구분되지 않는다.
                    return self._json(body, e.code)
                except URLError as e:
                    if attempt == 1:
                        time.sleep(1.5)      # 막 뜨는 중일 수 있다. 한 번만 더 기다린다.
                        continue
                    return self._json({"error": f"{plat} 백엔드({PORTS[plat]})가 응답하지 않습니다. "
                                                f"터미널에서 tryout/run.sh 로 다시 띄우세요. ({e})"}, 502)
        self._json({"error": "not found"}, 404)

    def _backend(self, plat, seeds, k):
        qs = "&".join(f"seed={quote(str(x))}" for x in seeds) + f"&k={k}"
        with urlopen(f"http://127.0.0.1:{PORTS[plat]}/recommend?{qs}", timeout=180) as r:
            return json.loads(r.read())

    def _cross(self, q):
        pid = (q.get("pid") or [""])[0]; rule = (q.get("rule") or ["M0"])[0]; k = int((q.get("k") or ["10"])[0])
        prof = next((p for p in XPROFILES if p["pid"] == pid), None)
        if prof is None: return self._json({"error": f"unknown profile {pid}"}, 404)
        if rule not in XRULES: return self._json({"error": f"unknown rule {rule}"}, 400)
        lists, seeds, coh, cards = {}, {}, {}, {}
        for key, plat in XPLAT.items():
            sub = prof[key]
            if not sub: continue
            sid = XSEEDS["seeds"][key][sub]
            try:
                rows = self._backend(plat, sid, 50)
            except HTTPError as e:
                try: body = json.loads(e.read())
                except Exception: body = {"error": f"HTTP {e.code}"}
                return self._json({"error": f"{plat} 백엔드 오류 — {body.get('error')}"}, e.code)
            except URLError as e:
                return self._json({"error": f"{plat} 백엔드({PORTS[plat]})가 응답하지 않습니다. ({e})"}, 502)
            lists[key] = [r["id"] for r in rows]; cards[key] = {str(r["id"]): r for r in rows}
            seeds[key] = len(sid); coh[key] = XSEEDS["coh"][key][sub]
        if rule == "M3":
            import re
            eps = {}
            for _id, c in cards.get("wn", {}).items():
                mm = re.search(r"(\d+)화", c.get("stat", "")); eps[str(_id)] = int(mm.group(1)) if mm else 0
            mixed = XRULES["M3"](lists, seeds, coh, k=k, episodes=eps)
        else:
            mixed = XRULES[rule](lists, seeds, coh, k=k)
        out = []
        for key, item, rank in mixed:
            c = dict(cards[key][str(item)]); c["plat"] = XPLAT[key]; c["plat_rank"] = rank; out.append(c)
        # 시드 이름도 같이 — 화면이 "이 조합"이 뭔지 보여줘야 채점이 성립한다
        seed_names = {}
        for key, plat in XPLAT.items():
            if not prof[key]: continue
            sid = XSEEDS["seeds"][key][prof[key]]
            try:
                qs = "&".join(f"id={quote(str(x))}" for x in sid)
                with urlopen(f"http://127.0.0.1:{PORTS[plat]}/card?{qs}", timeout=30) as r:
                    seed_names[plat] = [c["name"] for c in json.loads(r.read())]
            except Exception:
                seed_names[plat] = [str(x) for x in sid]
        return self._json({"pid": pid, "rule": rule, "seeds": seed_names, "items": out})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/grade": return self._json({"error": "not found"}, 404)
        n = int(self.headers.get("Content-Length") or 0)
        rec = json.loads(self.rfile.read(n) or b"{}")
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        GRADES.parent.mkdir(parents=True, exist_ok=True)
        with GRADES.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._json({"ok": True, "total": sum(1 for _ in GRADES.open(encoding="utf-8"))})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"http://localhost:{a.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
