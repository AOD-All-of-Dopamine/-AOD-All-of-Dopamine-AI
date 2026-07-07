-- M0 결정: 빈 테이블 위 HNSW 생성은 의도된 것 (M0 체크포인트가 인덱스 존재를 요구,
-- HNSW는 이후 INSERT마다 점증 구축됨). 대량 재적재(M1+) 전에는 REINDEX로 재빌드.
CREATE INDEX idx_content_embedding_hnsw ON aod_ai.content_embedding
  USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_user_profile_vector_hnsw ON aod_ai.user_profile_cache
  USING hnsw (profile_vector vector_cosine_ops);
