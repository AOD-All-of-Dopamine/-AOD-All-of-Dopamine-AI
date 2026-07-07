-- 실행: DBA/슈퍼유저 (CREATEROLE 필요) — aod_ai 마이그레이션 러너(aod_ai 계정)와 별개.
--   psql "postgresql://<dba>@<host>:<port>/<db>" -v ai_user=aod_ai -f sql/roles.sql
-- 근거: spec §2.2 / §1 '데이터 접근' — 배치 AI 계정 = public 읽기전용 + aod_ai 읽기쓰기.

-- 1) public 읽기전용 role (멱등)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'aod_public_ro') THEN
    CREATE ROLE aod_public_ro NOLOGIN;
  END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO aod_public_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO aod_public_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO aod_public_ro;

-- 2) 배치 AI 로그인 계정: public 읽기전용 role 부여 (백엔드 테이블 실수 수정 차단)
GRANT aod_public_ro TO :"ai_user";

-- 3) 배치 AI 로그인 계정: aod_ai 읽기쓰기
GRANT USAGE, CREATE ON SCHEMA aod_ai TO :"ai_user";
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA aod_ai TO :"ai_user";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA aod_ai TO :"ai_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA aod_ai
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO :"ai_user";
ALTER DEFAULT PRIVILEGES IN SCHEMA aod_ai
  GRANT USAGE, SELECT ON SEQUENCES TO :"ai_user";
