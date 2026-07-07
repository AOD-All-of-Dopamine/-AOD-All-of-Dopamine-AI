import argparse

from dotenv import load_dotenv

from aod_ai import db
from aod_ai.clients.embedding import EmbeddingClient
from aod_ai.clients.llm import LlmClient
from aod_ai.clients.vane import VaneClient
from aod_ai.config import Settings
from aod_ai.eyeball import write_eyeball_dump
from aod_ai.pipeline.collect_reviews import collect_reviews
from aod_ai.pipeline.embed import build_profile_embedding
from aod_ai.pipeline.extract import extract_profile
from aod_ai.pipeline.quality import compute_quality
from aod_ai.pipeline.select_targets import select_targets
from aod_ai.pipeline.upsert import upsert_assets


def _load_active_tags(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM aod_ai.fun_tag_dict WHERE status = 'active'")
        return [r[0] for r in cur.fetchall()]


def run_pipeline(conn, domain, limit, *, vane, llm, emb):
    active_tags = _load_active_tags(conn)
    records = []
    for target in select_targets(conn, domain, limit):
        sources = collect_reviews(vane, target)
        extraction = extract_profile(llm, target, sources, active_tags)
        vector = build_profile_embedding(emb, extraction)
        quality = compute_quality(conn, target)
        upsert_assets(conn, target, extraction, vector, quality, sources)
        records.append((target, extraction))
    return records


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="aod_ai.run")
    p.add_argument("--domain", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--dump", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    # M0 클라이언트의 os.getenv 기반 튜닝값(VANE_MAX_SOURCES, EXTRACT_MAX_RETRIES)이
    # .env를 보도록 process env에 로드 (M0 최종리뷰 #1).
    load_dotenv()
    s = Settings()
    conn = db.connect(s)
    vane = VaneClient(base_url=s.vane_base_url)
    llm = LlmClient(base_url=s.openai_base_url, api_key=s.openai_api_key, model=s.llm_model)
    emb = EmbeddingClient(
        base_url=s.openai_base_url, api_key=s.openai_api_key,
        model=s.embedding_model, dim=s.embedding_dim,
    )
    records = run_pipeline(conn, args.domain, args.limit, vane=vane, llm=llm, emb=emb)
    if args.dump:
        write_eyeball_dump(records, args.dump)
    print(f"processed {len(records)} contents")


if __name__ == "__main__":
    main()
