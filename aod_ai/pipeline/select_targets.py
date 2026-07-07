import hashlib

from aod_ai.models import SelectedTarget

_CANDIDATE_SQL = """
SELECT c.content_id, c.domain, c.master_title, c.original_title,
       c.synopsis, COALESCE(w.genres, '{}') AS genres
FROM public.contents c
JOIN public.webnovel_contents w ON w.content_id = c.content_id
WHERE c.domain = %s
ORDER BY c.content_id
"""
_EXISTING_SQL = "SELECT content_id, content_hash FROM aod_ai.content_semantic_profile"


def compute_content_hash(master_title, original_title, synopsis, genres):
    payload = f"{master_title}|{original_title}|{synopsis}|{sorted(genres)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_targets(conn, domain: str, limit: int) -> list[SelectedTarget]:
    with conn.cursor() as cur:
        cur.execute(_EXISTING_SQL)
        existing = {row[0]: row[1] for row in cur.fetchall()}
        cur.execute(_CANDIDATE_SQL, (domain,))
        rows = cur.fetchall()

    out: list[SelectedTarget] = []
    for content_id, dom, master_title, original_title, synopsis, genres in rows:
        genres = list(genres or [])
        content_hash = compute_content_hash(master_title, original_title, synopsis, genres)
        if existing.get(content_id) == content_hash:
            continue
        out.append(
            SelectedTarget(
                content_id=content_id,
                domain=dom,
                master_title=master_title,
                original_title=original_title,
                synopsis=synopsis,
                genres=genres,
                content_hash=content_hash,
            )
        )
        if len(out) >= limit:
            break
    return out
