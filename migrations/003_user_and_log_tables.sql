CREATE TABLE aod_ai.user_profile_cache (
  user_id          bigint PRIMARY KEY,
  fun_tag_profile  jsonb NOT NULL DEFAULT '{}'::jsonb,
  negative_fun_tag jsonb NOT NULL DEFAULT '{}'::jsonb,
  profile_vector   vector(1024),
  positive_count   int NOT NULL DEFAULT 0,
  updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE aod_ai.rec_impression (
  id                  bigserial PRIMARY KEY,
  request_id          uuid   NOT NULL,
  user_id             bigint,
  location            text   NOT NULL,
  selected_content_id bigint,
  content_id          bigint NOT NULL,
  candidate_source    text,
  rank_position       int,
  score_breakdown     jsonb,
  served_at           timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE aod_ai.rec_event (
  id         bigserial PRIMARY KEY,
  request_id uuid,
  user_id    bigint,
  content_id bigint NOT NULL,
  event_type text   NOT NULL,
  value      real,
  created_at timestamptz NOT NULL DEFAULT now()
);
