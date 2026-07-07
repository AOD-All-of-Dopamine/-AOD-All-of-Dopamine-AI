from aod_ai.funtag_dict import list_active_tags, load_seed_tags, sync_seed_tags


def test_seed_yaml_has_30_to_60_tags_with_alias_shape():
    # spec §1/§10 #6 이 고정하는 것은 '시드 30~60개'뿐. 개수 + 스키마 형태만 검증.
    tags = load_seed_tags()
    assert 30 <= len(tags) <= 60
    names = {t["name"] for t in tags}
    assert len(names) == len(tags)  # 이름 유니크
    for t in tags:
        assert isinstance(t["name"], str) and t["name"]
        assert isinstance(t.get("aliases", []), list)


def test_sync_upserts_active_and_is_idempotent(conn):
    tags = load_seed_tags()
    inserted = sync_seed_tags(conn, tags)
    assert inserted == len(tags)
    sync_seed_tags(conn, tags)  # 재실행 멱등

    active = list_active_tags(conn)
    assert len(active) == len(tags)
    assert "회귀" in active

    statuses = {
        r[0]
        for r in conn.execute("SELECT DISTINCT status FROM aod_ai.fun_tag_dict").fetchall()
    }
    assert statuses == {"active"}
