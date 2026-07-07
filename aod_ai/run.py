import argparse
import logging
import sys

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
from aod_ai.pipeline.quality import compute_global_stats, compute_quality
from aod_ai.pipeline.select_targets import select_targets
from aod_ai.pipeline.upsert import upsert_assets

logger = logging.getLogger(__name__)


def _load_active_tags(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM aod_ai.fun_tag_dict WHERE status = 'active'")
        return [r[0] for r in cur.fetchall()]


def run_pipeline(conn, domain, limit, *, vane, llm, emb, failed_ids: list | None = None):
    active_tags = _load_active_tags(conn)
    global_stats = compute_global_stats(conn, domain)  # 배치당 1회 (리뷰 F#2)
    records = []
    for target in select_targets(conn, domain, limit):
        # 리뷰 F#1: 콘텐츠 1건 실패가 배치 전체·후속 재실행을 막지 않도록 격리
        try:
            sources = collect_reviews(vane, target)
            extraction = extract_profile(llm, target, sources, active_tags)
            vector = build_profile_embedding(emb, extraction)
            quality = compute_quality(conn, target, global_stats)
            upsert_assets(conn, target, extraction, vector, quality, sources)
        except Exception:
            logger.exception("content %s 처리 실패 — 건너뜀", target.content_id)
            if failed_ids is not None:
                failed_ids.append(target.content_id)
            continue
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
    failed_ids: list = []
    records = run_pipeline(
        conn, args.domain, args.limit, vane=vane, llm=llm, emb=emb, failed_ids=failed_ids
    )
    if args.dump:
        write_eyeball_dump(records, args.dump)
    print(f"processed {len(records)} contents")
    if failed_ids:
        print(f"failed {len(failed_ids)} contents: {failed_ids}")
        sys.exit(1)


if __name__ == "__main__":
    main()
