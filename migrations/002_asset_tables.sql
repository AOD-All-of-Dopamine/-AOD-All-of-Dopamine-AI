CREATE TABLE aod_ai.fun_tag_dict (
  id          bigserial PRIMARY KEY,
  name        text NOT NULL UNIQUE,
  aliases     text[] NOT NULL DEFAULT '{}',
  description text,
  status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active','proposed','rejected')),
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE aod_ai.content_semantic_profile (
  content_id         bigint PRIMARY KEY,
  domain             text NOT NULL,
  normalized_summary text,
  profile_text       text NOT NULL,
  evidence           jsonb NOT NULL DEFAULT '[]'::jsonb,
  extraction_quality real  NOT NULL DEFAULT 0,
  source_count       int   NOT NULL DEFAULT 0,
  content_hash       text  NOT NULL,
  processed_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE aod_ai.content_fun_tag (
  content_id     bigint NOT NULL,
  tag            text   NOT NULL,
  tag_score      real   NOT NULL,
  tag_confidence real   NOT NULL,
  PRIMARY KEY (content_id, tag)
);
CREATE INDEX idx_content_fun_tag_tag ON aod_ai.content_fun_tag (tag);

CREATE TABLE aod_ai.content_embedding (
  content_id bigint PRIMARY KEY,
  embedding  vector(1024) NOT NULL,
  model      text NOT NULL,
  dim        int  NOT NULL DEFAULT 1024
);

CREATE TABLE aod_ai.content_quality_score (
  content_id               bigint PRIMARY KEY,
  bayesian_score           real,
  platform_rank_score      real,
  review_count_score       real,
  recency_score            real,
  quality_popularity_score real,
  computed_at              timestamptz NOT NULL DEFAULT now()
);
