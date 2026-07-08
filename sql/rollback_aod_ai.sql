-- ============================================================================
-- aod_ai 전체 롤백 스크립트
-- 실행: DBA/마스터 계정 (psql "<dba-url>" -f sql/rollback_aod_ai.sql)
--
-- 제거 대상 (M0/M1이 RDS에 추가한 것 전부):
--   1) aod_ai 스키마 + 그 안의 테이블 8개 + 적재 데이터 전부
--   2) 배치 계정(aod_ai)과 읽기전용 롤(aod_public_ro), 관련 권한
--
-- 유지되는 것:
--   - public 스키마의 모든 백엔드 테이블/데이터 (애초에 수정된 적 없음)
--   - vector 확장 (public에 설치됨; 다른 용도가 생길 수 있어 기본 유지 — 필요 시 맨 아래 주석 해제)
--
-- 참고: aod_ai 데이터는 배치 재실행으로 언제든 재생성 가능(멱등 파이프라인)이라
--       롤백에 따른 데이터 손실 비용은 낮다. 되돌리려면 README의 셋업 절차를 다시 수행.
-- ============================================================================

-- 1) 스키마 + 테이블 + 데이터 제거
DROP SCHEMA IF EXISTS aod_ai CASCADE;

-- 2) 배치 계정 정리 (소유 객체/권한 정리 후 제거해야 함)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aod_ai') THEN
    EXECUTE 'DROP OWNED BY aod_ai';          -- 잔여 소유물·부여받은 권한 일괄 정리
    EXECUTE 'REVOKE aod_public_ro FROM aod_ai';
    EXECUTE 'DROP ROLE aod_ai';
  END IF;
END $$;

-- 3) 읽기전용 롤 정리
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aod_public_ro') THEN
    EXECUTE 'REVOKE ALL ON ALL TABLES IN SCHEMA public FROM aod_public_ro';
    EXECUTE 'REVOKE USAGE ON SCHEMA public FROM aod_public_ro';
    EXECUTE 'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE SELECT ON TABLES FROM aod_public_ro';
    EXECUTE 'DROP ROLE aod_public_ro';
  END IF;
END $$;

-- 4) (선택) vector 확장 제거 — aod_ai 외 다른 곳에서 벡터 타입을 안 쓸 때만.
--    M2 서빙(백엔드)이 배포돼 있으면 제거 금지.
-- DROP EXTENSION IF EXISTS vector;

-- 5) (해당 시) M2 백엔드가 배포된 후라면 Flyway 이력도 정리 대상:
--    V4(grant)는 스키마가 사라지면 무의미하며, 재적용을 원치 않으면 이력 유지,
--    aod_ai를 재구축해 V4를 다시 태우려면 아래 주석 해제.
-- DELETE FROM public.flyway_schema_history WHERE version = '4';
