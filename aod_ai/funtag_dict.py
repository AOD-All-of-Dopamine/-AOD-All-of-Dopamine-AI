from __future__ import annotations

import argparse
from pathlib import Path

import psycopg
import yaml

SEED_PATH = Path(__file__).resolve().parent.parent / "resources" / "seed_fun_tags.yaml"


def load_seed_tags(path: Path = SEED_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["tags"]


def sync_seed_tags(conn: psycopg.Connection, tags: list[dict]) -> int:
    count = 0
    for t in tags:
        conn.execute(
            """
            INSERT INTO aod_ai.fun_tag_dict (name, aliases, description, status)
            VALUES (%s, %s, %s, 'active')
            ON CONFLICT (name) DO UPDATE
              SET aliases = EXCLUDED.aliases,
                  description = EXCLUDED.description,
                  status = 'active'
            """,
            (t["name"], t.get("aliases", []), t.get("description")),
        )
        count += 1
    return count


def list_active_tags(conn: psycopg.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM aod_ai.fun_tag_dict WHERE status = 'active' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def main() -> None:
    # CONTRACTS §9 [funtag_dict --sync 인터페이스 고정]: 시드 YAML을 status='active'로 upsert.
    parser = argparse.ArgumentParser(description="fun_tag dictionary utilities")
    parser.add_argument("--sync", action="store_true", help="seed_fun_tags.yaml을 fun_tag_dict에 active로 upsert")
    args = parser.parse_args()
    if not args.sync:
        parser.print_help()
        return

    from aod_ai.config import Settings
    from aod_ai.db import connect

    conn = connect(Settings(), register_pgvector=False)
    try:
        n = sync_seed_tags(conn, load_seed_tags())
        print(f"synced {n} active fun tags")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
