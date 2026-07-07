CREATE SCHEMA IF NOT EXISTS aod_ai;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS aod_ai.schema_migrations (
  version    text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);
