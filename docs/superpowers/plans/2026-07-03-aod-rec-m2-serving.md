# M2 — 홈/related 서빙 (AOD 추천) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (권장) 또는 superpowers:executing-plans. 스텝은 `- [ ]` 체크박스.

**Goal:** 기존 백엔드 api 모듈에 추천 서빙을 구현 — `aod_ai` 완제품 조회 + pgvector DB내 HNSW ANN + 자바 스코어링/랭킹/후처리 → `GET /api/recommendations`(home/related) Top-N. 별도 FastAPI 없음.

**Architecture:** 요청 시 후보생성(pgvector `<=>` + fun_tag SQL, union→dedupe) → 5피처 계산 → home/related 랭킹 → 후처리(hard filter·soft penalty·diversity) → Top-N + `rec_impression` 로깅. 홈 Top-N 캐싱(t3.small 대비). M2는 무조건 콜드스타트(quality/popularity fallback); positive_count 분기·온보딩 시드는 M3.

**Tech Stack:** Java 17, Spring Boot 3, JPA 네이티브 쿼리(pgvector), Spring Cache, JUnit 5(+Mockito, 통합은 Testcontainers pgvector), Flyway(V4 grant).

**전제:** M0의 aod_ai 마이그레이션 선행(grant 대상 존재), M1이 자산 적재. **공유 계약:** `2026-07-03-aod-rec-00-contracts.md` (특히 §1.2·§7 서빙, §9 통합 보정 [마이그레이션 순서·소유권]·[정규화 태그 저장]) 를 그대로 인용.

---

## M2 — 홈 서빙 (기존 백엔드 api 모듈)

**Deliverable/Checkpoint:** `com.example.AOD.recommend` 패키지에 후보→피처→랭킹→**후처리(hard filter → soft penalty/boost → diversity → Top-N, spec §5.4 순서 준수)**→로깅 서빙 경로를 구현하고, `GET /api/recommendations?location=home`가 콜드스타트(프로파일 0건) 경로로 품질/인기 기반 Top-N `RecommendationItem`을 반환 — 오프라인 검수 가능(contracts §1.2·§7, spec §5·§6.2·§9 M2행). Testcontainers pgvector 통합 테스트(Task 10)로 grant 실효 + 네이티브 aod_ai 쿼리 동작을 **행위 검증(비-optional)** 한다.

> 실행 위치: 모든 gradle 명령은 `-AOD-All-of-Dopamine-back/`에서 실행. 테스트 태스크 경로의 선행 하이픈 때문에 `":-AOD-All-of-Dopamine-api:test"`처럼 반드시 따옴표로 감싼다. 커밋은 태스크당 1커밋(contracts §8), 각 커밋에 `Co-Authored-By` 트레일러.
> 테스트 전략: (a) 순수 로직/오케스트레이션은 **Mockito 목(경량, 무DB)**, (b) 네이티브 SQL·grant는 **리플렉션/문자열로 계약 고정**(빠른 RED→GREEN), (c) 실제 DB 행위(grant 실효, pgvector 쿼리 실행, Flyway migrate)는 **Task 10의 Testcontainers pgvector 통합 테스트로 필수 검증**(Docker 필요 — optional 아님). 이로써 "테스트 통과 = 계약 실제 성립"이 성립한다.

> **범위 및 §9 M2 재조정 노트 (silently drop 방지):**
> - 이 섹션은 §9 M2의 **서빙(자바) 절반**만 다룬다: 홈 서빙 후보→피처→랭킹→후처리(§5.4 4단계)→로깅.
> - **Airflow 도입(배치 DAG화)** 와 **테스트 유저 프로파일 시드**(§9 M2 나머지 line-item)는 파이썬 배치(AI 모듈) 소관이므로 **별도 plan 섹션 "M2-Batch (Airflow DAG화 + 테스트 유저 프로파일)"** 로 이관한다(본 섹션 범위 밖, 드롭 아님).
> - **related 서빙**(location=related, selected-content 후보 생성/랭킹)은 spec §9상 **M3**. 본 섹션 title은 "홈 서빙"이며 `Ranker`에 related_score 미포함(M3에서 추가).
> - **콜드스타트 온보딩 시드**(§6.2 0건 전략의 `onboarding-page.tsx` 초기선호 → fun_tag 프로파일 시드)와 **positive_count 분기**(§7)는 **M3**로 연기. M2는 **무조건 콜드스타트(quality/popularity fallback)**. `AiAssetRepository.findPositiveCount`는 리포지토리에 두되 M2 서빙 경로에서 호출하지 않는다(M3용, Task 5에서 계약 고정).
> - **metadata_match**는 M2에서 **장르 Jaccard만**(§5.2의 creator/platform/domain 확장은 M3). 콜드스타트에서 metadata=0이라 서빙 결과에 영향 없음(의도된 단순화).

---

### Task 1: Flyway V4 — 백엔드 API 서빙 계정에 aod_ai 읽기 + 로그 쓰기 grant

**Files:**
- Create: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-back/-AOD-All-of-Dopamine-api/src/main/resources/db/migration/V4__grant_aod_ai_access.sql`
- Modify: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-back/-AOD-All-of-Dopamine-api/build.gradle`
- Modify: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-back/-AOD-All-of-Dopamine-api/src/main/resources/application.properties`
- Test: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-back/-AOD-All-of-Dopamine-api/src/test/java/com/example/AOD/recommend/migration/V4GrantMigrationTest.java`

근거: contracts §1.2(백엔드 DB 계정에 `aod_ai` 읽기 + `rec_impression`/`rec_event` 쓰기 grant 필요), §2.2, §5(백엔드 property 키 `spring.datasource.username`), §6(Flyway `V<n>__`; 기존 V1·V3 → 다음은 **V4**). grant 대상은 **실제 서빙 커넥션 계정 = `spring.datasource.username`(현재 `postgres`, application.properties:14)** 이다. 계약에 없는 새 롤(`aod_api` 등)을 발명하지 않는다 — 대상이 데이터소스 유저와 어긋나면 grant가 서빙 커넥션에 도달하지 못한다. 최소권한 전용 롤로 분리하려면 **그 롤을 생성 + `spring.datasource.username` 재지정**이 선행돼야 하므로 M3 인프라 과제로 남기고, 본 마이그레이션은 대상 이름만 교체하면 되도록 작성한다.
Flyway 도입 정합: 현재 백엔드는 Flyway 부재 + `spring.jpa.hibernate.ddl-auto=update`로 스키마가 Hibernate 관리(비어있지 않음)라, flyway-core를 그냥 추가하면 부팅 시 "Found non-empty schema(s) without schema history table"로 실패한다. 따라서 `baseline-on-migrate=true` + `baseline-version=3`(기존 V1·V3을 baseline 처리 → **오직 V4만 적용**)으로 설정한다. Flyway는 Hibernate보다 먼저 실행되며 `ddl-auto=update`와 공존한다. V4는 `aod_ai` 스키마/테이블(M0 Python 마이그레이션 산출물)에만 grant하므로 M0 완료가 선행 조건이다. **실제 migrate 정상 동작 검증은 Task 10(Testcontainers)에서 필수 수행.**

- [ ] 실패 테스트 작성 — 마이그레이션 리소스 존재/버전/grant 대상(서빙 계정) 고정 검증:
```java
package com.example.AOD.recommend.migration;

import org.junit.jupiter.api.Test;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import static org.junit.jupiter.api.Assertions.*;

class V4GrantMigrationTest {

    @Test
    void v4GrantFileExistsAndGrantsToServingDatasourceUser() throws Exception {
        URL url = getClass().getResource("/db/migration/V4__grant_aod_ai_access.sql");
        assertNotNull(url, "V4__grant_aod_ai_access.sql must exist (V3 다음 순번)");
        String sql = Files.readString(Paths.get(url.toURI()), StandardCharsets.UTF_8);
        // grant 대상 = spring.datasource.username(=postgres). 계약에 없는 롤 이름 금지.
        assertTrue(sql.contains("GRANT USAGE ON SCHEMA aod_ai TO postgres"));
        assertTrue(sql.contains("GRANT SELECT ON aod_ai.content_embedding TO postgres"));
        assertTrue(sql.contains("GRANT SELECT ON aod_ai.content_fun_tag TO postgres"));
        assertTrue(sql.contains("GRANT SELECT ON aod_ai.content_quality_score TO postgres"));
        assertTrue(sql.contains("GRANT INSERT ON aod_ai.rec_impression TO postgres"));
        assertTrue(sql.contains("GRANT INSERT ON aod_ai.rec_event TO postgres"));
        assertFalse(sql.contains("aod_api"), "존재하지 않는 롤에 grant 금지(서빙 계정과 불일치)");
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.migration.V4GrantMigrationTest"`
  예상 FAIL: `V4GrantMigrationTest > v4GrantFileExistsAndGrantsToServingDatasourceUser FAILED` / `org.opentest4j.AssertionFailedError: V4__grant_aod_ai_access.sql must exist (V3 다음 순번) ==> expected: not <null>` → `BUILD FAILED`.
- [ ] 최소 구현 — 마이그레이션 SQL 작성 (`V4__grant_aod_ai_access.sql`):
```sql
-- V4: 백엔드 API 서빙 DB 계정(= spring.datasource.username, 현재 postgres)에
--     aod_ai 스키마 자산 읽기 + 로그 테이블 쓰기 권한 부여.
-- 근거: contracts §1.2, §2.2, §5. aod_ai 스키마/테이블은 M0 Python 마이그레이션 산출물(선행).
-- ※ 최소권한 전용 롤로 분리 시 이 대상 이름(postgres)만 교체하고 spring.datasource.username을 함께 재지정.
GRANT USAGE ON SCHEMA aod_ai TO postgres;

GRANT SELECT ON aod_ai.content_semantic_profile TO postgres;
GRANT SELECT ON aod_ai.content_fun_tag TO postgres;
GRANT SELECT ON aod_ai.content_embedding TO postgres;
GRANT SELECT ON aod_ai.content_quality_score TO postgres;
GRANT SELECT ON aod_ai.user_profile_cache TO postgres;
GRANT SELECT ON aod_ai.fun_tag_dict TO postgres;

GRANT INSERT ON aod_ai.rec_impression TO postgres;
GRANT INSERT ON aod_ai.rec_event TO postgres;

GRANT USAGE, SELECT ON SEQUENCE aod_ai.rec_impression_id_seq TO postgres;
GRANT USAGE, SELECT ON SEQUENCE aod_ai.rec_event_id_seq TO postgres;
```
  `build.gradle`의 `// === DB ===` 블록 바로 아래에 Flyway 의존성 추가(버전은 Spring Boot 3.4 BOM 관리; `implementation`이라 테스트 런타임에서도 사용 가능):
```gradle
	// === DB ===
	runtimeOnly 'org.postgresql:postgresql'   // BOM으로 버전 관리 (중복/명시 버전 제거)
	runtimeOnly 'com.h2database:h2'

	// === Flyway (aod_ai grant 마이그레이션 V4 적용) ===
	implementation 'org.flywaydb:flyway-core'
	implementation 'org.flywaydb:flyway-database-postgresql'
```
  `application.properties`에 Flyway baseline 설정 추가(비어있지 않은 Hibernate 관리 스키마에서 부팅 실패 방지; V1·V3 baseline 처리 → V4만 적용). `spring.datasource.password` 아래에 삽입:
```properties
# ===== Flyway (aod_ai grant 마이그레이션 V4) =====
# 기존 스키마는 Hibernate(ddl-auto=update)가 관리하므로 baseline 후 V4만 적용.
spring.flyway.enabled=true
spring.flyway.baseline-on-migrate=true
spring.flyway.baseline-version=3
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.migration.V4GrantMigrationTest"`
  예상 PASS: `V4GrantMigrationTest > v4GrantFileExistsAndGrantsToServingDatasourceUser PASSED` → `BUILD SUCCESSFUL`.
  (행위 검증) 실제 grant 실효 + Flyway migrate 정상 동작은 **Task 10**에서 Testcontainers로 필수 확인.
- [ ] 커밋:
  `git add -A && git commit -m "feat: V4 grant aod_ai read + rec log insert to serving datasource user; add Flyway with baseline" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 2: FeatureCalculator — 5개 피처 (순수 로직)

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/dto/FunTag.java`
- Create: `.../src/main/java/com/example/AOD/recommend/dto/QualityScore.java`
- Create: `.../src/main/java/com/example/AOD/recommend/feature/FeatureVector.java`
- Create: `.../src/main/java/com/example/AOD/recommend/feature/FeatureCalculator.java`
- Test: `.../src/test/java/com/example/AOD/recommend/feature/FeatureCalculatorTest.java`

(경로 접두어 공통: `C:/Users/dfdfg/IdeaProjects/AOD/-AOD-All-of-Dopamine-back/-AOD-All-of-Dopamine-api`)

근거: spec §5.2(fun_tag overlap=tag_score×tag_confidence 가중, profile cosine, metadata match, quality lookup, recency). 컬럼명은 contracts §2 `content_fun_tag(tag,tag_score,tag_confidence)`·`content_quality_score(quality_popularity_score)` 재사용. **metadata_match는 M2에서 장르 Jaccard만**(§5.2의 creator/platform/domain은 M3 확장; 콜드스타트에서 metadata=0이라 서빙 무영향 — 상단 재조정 노트).

- [ ] 실패 테스트 작성:
```java
package com.example.AOD.recommend.feature;

import com.example.AOD.recommend.dto.FunTag;
import com.example.AOD.recommend.dto.QualityScore;
import org.junit.jupiter.api.Test;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.assertEquals;

class FeatureCalculatorTest {

    private final FeatureCalculator calc = new FeatureCalculator();

    @Test
    void funTagMatchScoreWeightsScoreTimesConfidence() {
        double s = calc.funTagMatchScore(Map.of("힐링", 1.0),
                List.of(new FunTag("힐링", 0.8, 0.5), new FunTag("코믹", 0.9, 0.9)));
        assertEquals(0.40, s, 1e-9); // 1.0 * (0.8*0.5), 미매칭 태그 제외
    }

    @Test
    void funTagMatchScoreZeroForEmptyProfile() {
        assertEquals(0.0, calc.funTagMatchScore(Map.of(),
                List.of(new FunTag("힐링", 0.8, 0.5))), 1e-9);
    }

    @Test
    void profileSimilarityCosine() {
        assertEquals(1.0, calc.profileSimilarityScore(new float[]{1, 0}, new float[]{1, 0}), 1e-9);
        assertEquals(0.0, calc.profileSimilarityScore(new float[]{1, 0}, new float[]{0, 1}), 1e-9);
    }

    @Test
    void metadataMatchIsGenreJaccard() {
        assertEquals(1.0 / 3, calc.metadataMatchScore(Set.of("A", "B"), Set.of("B", "C")), 1e-9);
    }

    @Test
    void qualityLookupReturnsPopularityScoreOrZero() {
        assertEquals(0.7, calc.qualityPopularityScore(new QualityScore(0, 0, 0, 0, 0.7)), 1e-9);
        assertEquals(0.0, calc.qualityPopularityScore(null), 1e-9);
    }

    @Test
    void recencyDecaysToZeroOverOneYear() {
        LocalDate now = LocalDate.of(2026, 7, 3);
        assertEquals(1.0, calc.recencyScore(now, now), 1e-9);
        assertEquals(0.0, calc.recencyScore(now.minusDays(365), now), 1e-9);
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.feature.FeatureCalculatorTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class FeatureCalculator` → `BUILD FAILED`.
- [ ] 최소 구현:
  `dto/FunTag.java`:
```java
package com.example.AOD.recommend.dto;

public record FunTag(String tag, double tagScore, double tagConfidence) {}
```
  `dto/QualityScore.java` (contracts §2 content_quality_score 컬럼 미러):
```java
package com.example.AOD.recommend.dto;

public record QualityScore(
        double bayesianScore,
        double platformRankScore,
        double reviewCountScore,
        double recencyScore,
        double qualityPopularityScore) {}
```
  `feature/FeatureVector.java`:
```java
package com.example.AOD.recommend.feature;

public record FeatureVector(double funTag, double profileSim, double quality,
                            double metadata, double recency) {}
```
  `feature/FeatureCalculator.java`:
```java
package com.example.AOD.recommend.feature;

import com.example.AOD.recommend.dto.FunTag;
import com.example.AOD.recommend.dto.QualityScore;
import org.springframework.stereotype.Component;
import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Component
public class FeatureCalculator {

    /** spec §5.2 fun_tag overlap: Σ(공유 태그) profileWeight × (tag_score × tag_confidence). */
    public double funTagMatchScore(Map<String, Double> profileTagWeights, List<FunTag> candidateTags) {
        if (profileTagWeights == null || profileTagWeights.isEmpty() || candidateTags == null) return 0.0;
        double sum = 0.0;
        for (FunTag t : candidateTags) {
            Double w = profileTagWeights.get(t.tag());
            if (w != null) sum += w * (t.tagScore() * t.tagConfidence());
        }
        return sum;
    }

    /** spec §5.2 profile cosine similarity. */
    public double profileSimilarityScore(float[] profileVector, float[] candidateVector) {
        if (profileVector == null || candidateVector == null
                || profileVector.length == 0 || profileVector.length != candidateVector.length) return 0.0;
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < profileVector.length; i++) {
            dot += profileVector[i] * candidateVector[i];
            na += (double) profileVector[i] * profileVector[i];
            nb += (double) candidateVector[i] * candidateVector[i];
        }
        if (na == 0 || nb == 0) return 0.0;
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    /** spec §5.2 metadata match. **M2 단순화: 장르 Jaccard만** (creator/platform/domain은 M3). */
    public double metadataMatchScore(Set<String> profileGenres, Set<String> candidateGenres) {
        if (profileGenres == null || candidateGenres == null
                || profileGenres.isEmpty() || candidateGenres.isEmpty()) return 0.0;
        long inter = candidateGenres.stream().filter(profileGenres::contains).count();
        long union = profileGenres.size() + candidateGenres.size() - inter;
        return union == 0 ? 0.0 : (double) inter / union;
    }

    /** spec §5.2 quality lookup: content_quality_score.quality_popularity_score. */
    public double qualityPopularityScore(QualityScore quality) {
        return quality == null ? 0.0 : quality.qualityPopularityScore();
    }

    /** spec §5.2 recency: 발매일 1.0, 365일 선형 감쇠 후 0. */
    public double recencyScore(LocalDate releaseDate, LocalDate now) {
        if (releaseDate == null || now == null) return 0.0;
        long days = ChronoUnit.DAYS.between(releaseDate, now);
        if (days < 0) days = 0;
        if (days >= 365) return 0.0;
        return 1.0 - (days / 365.0);
    }
}
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.feature.FeatureCalculatorTest"`
  예상 PASS: `FeatureCalculatorTest > ... PASSED`(6건) → `BUILD SUCCESSFUL`.
- [ ] 커밋:
  `git add -A && git commit -m "feat: FeatureCalculator 5 features (funtag/cosine/metadata-jaccard/quality/recency)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 3: Ranker — home_score (순수 로직)

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/rank/Ranker.java`
- Test: `.../src/test/java/com/example/AOD/recommend/rank/RankerTest.java`

근거: contracts §7 / spec §5.3 — `home = 0.45·funtag + 0.25·profile_sim + 0.15·quality + 0.10·metadata + 0.05·recency`. **related_score(0.4·home + 0.6·selected_sim)는 related 서빙과 함께 M3로 연기**(spec §9; 본 M2는 홈 서빙만 — 상단 재조정 노트). 서빙 미연결 dead code를 만들지 않도록 M2에는 넣지 않는다.

- [ ] 실패 테스트 작성:
```java
package com.example.AOD.recommend.rank;

import com.example.AOD.recommend.feature.FeatureVector;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

class RankerTest {

    private final Ranker ranker = new Ranker();

    @Test
    void homeScoreUsesContractWeights() {
        // 0.45+0.25+0.15+0.10+0.05 = 1.0
        assertEquals(1.0, ranker.homeScore(new FeatureVector(1, 1, 1, 1, 1)), 1e-9);
        // 0.45*0.2 + 0.25*0.4 + 0.15*0.6 + 0.10*0.8 + 0.05*1.0 = 0.41
        assertEquals(0.41, ranker.homeScore(new FeatureVector(0.2, 0.4, 0.6, 0.8, 1.0)), 1e-9);
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.rank.RankerTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class Ranker` → `BUILD FAILED`.
- [ ] 최소 구현 (`rank/Ranker.java`):
```java
package com.example.AOD.recommend.rank;

import com.example.AOD.recommend.feature.FeatureVector;
import org.springframework.stereotype.Component;

@Component
public class Ranker {

    /** contracts §7: home = 0.45·funtag + 0.25·profile + 0.15·quality + 0.10·metadata + 0.05·recency. */
    public double homeScore(FeatureVector f) {
        return 0.45 * f.funTag()
             + 0.25 * f.profileSim()
             + 0.15 * f.quality()
             + 0.10 * f.metadata()
             + 0.05 * f.recency();
    }
    // related_score(0.4·home + 0.6·selected_sim)는 related 서빙과 함께 M3에서 추가.
}
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.rank.RankerTest"`
  예상 PASS: `RankerTest > homeScoreUsesContractWeights PASSED`(1건) → `BUILD SUCCESSFUL`.
- [ ] 커밋:
  `git add -A && git commit -m "feat: Ranker home score with contract §7 weights (related deferred to M3)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 4: PostProcessor — hard filter / soft penalty·boost / diversity (순수 로직) + Candidate

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/candidate/Candidate.java`
- Create: `.../src/main/java/com/example/AOD/recommend/postprocess/PostProcessor.java`
- Test: `.../src/test/java/com/example/AOD/recommend/postprocess/PostProcessorTest.java`

근거: spec §5.4 — (1) hard filter age_rating/hide/unavailable, (2) soft: negative penalty ≤20%·recency boost ≤8%, (3) diversity Top-N 한 domain ≤60%·한 platform ≤50%. **diversity 규칙 보정**: 플랫폼 정보가 없는(빈) 후보는 특정 플랫폼 점유로 볼 수 없으므로 platform 캡에서 **면제**한다(빈 문자열이 하나의 가짜 버킷으로 묶여 전부 캡되는 것을 방지 — §5.4-3의 "한 platform ≤50%"는 실제 플랫폼에만 적용). 세 후처리 메서드는 Task 8에서 hardFilter → applySoftAdjust → diversify 순으로 실제 서빙에 연결된다.

- [ ] 실패 테스트 작성:
```java
package com.example.AOD.recommend.postprocess;

import com.example.AOD.recommend.candidate.Candidate;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

class PostProcessorTest {

    private final PostProcessor pp = new PostProcessor();

    private Candidate c(long id, String domain, double score) {
        Candidate x = new Candidate(id, domain);
        x.score = score;
        return x;
    }

    @Test
    void hardFilterRemovesHiddenUnavailableAndAdultWhenNotAllowed() {
        Candidate hidden = c(1, "MOVIE", 1); hidden.hidden = true;
        Candidate gone = c(2, "MOVIE", 1); gone.unavailable = true;
        Candidate adult = c(3, "MOVIE", 1); adult.ageRating = "청소년이용불가";
        Candidate ok = c(4, "MOVIE", 1);
        List<Candidate> out = pp.hardFilter(List.of(hidden, gone, adult, ok), false);
        assertEquals(1, out.size());
        assertEquals(4L, out.get(0).contentId);
    }

    @Test
    void hardFilterKeepsAdultWhenAllowed() {
        Candidate adult = c(3, "MOVIE", 1); adult.ageRating = "19";
        assertEquals(1, pp.hardFilter(List.of(adult), true).size());
    }

    @Test
    void softAdjustCapsPenaltyAt20AndBoostAt8Percent() {
        Candidate a = c(1, "MOVIE", 1.0);
        pp.applySoftAdjust(List.of(a), Map.of(1L, 0.50), Map.of(1L, 0.50));
        // penalty min(0.5,0.2)*1.0=0.2, boost min(0.5,0.08)*1.0=0.08 → 1.0-0.2+0.08
        assertEquals(0.88, a.score, 1e-9);
    }

    @Test
    void diversityPromotesLowerDomainAboveExcessSameDomain() {
        // topN=3 → domainCap=floor(1.8)=1 → MOVIE 1개만 1차 통과.
        // 후보 모두 platforms 비어있음 → platform 캡 면제(가짜 "" 버킷 캡 방지) → GAME(c2) 승격.
        // 1차: [5](MOVIE), c4·c3 domain 초과 defer, [5,2](GAME 통과) → defer=[4,3] 채움 → [5,2,4].
        List<Candidate> ranked = List.of(
                c(5, "MOVIE", 5), c(4, "MOVIE", 4), c(3, "MOVIE", 3), c(2, "GAME", 2));
        List<Candidate> out = pp.diversify(ranked, 3);
        assertEquals(3, out.size());
        assertEquals(List.of(5L, 2L, 4L),
                out.stream().map(x -> x.contentId).toList());
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.postprocess.PostProcessorTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class Candidate` (및 `class PostProcessor`) → `BUILD FAILED`.
- [ ] 최소 구현:
  `candidate/Candidate.java` (서빙 내부 작업 타입):
```java
package com.example.AOD.recommend.candidate;

import com.example.AOD.recommend.dto.FunTag;
import com.example.AOD.recommend.dto.QualityScore;
import com.example.AOD.recommend.feature.FeatureVector;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Candidate {
    public Long contentId;
    public String domain;                 // MOVIE/TV/GAME/WEBTOON/WEBNOVEL
    public List<String> platforms = new ArrayList<>();
    public String ageRating;              // nullable (spec §10 이슈5: 없을 수 있음)
    public boolean hidden;
    public boolean unavailable;
    public String candidateSource;        // "vector_ann" | "fun_tag" | "quality"
    public double score;
    public float[] embedding;
    public List<FunTag> funTags = new ArrayList<>();
    public QualityScore quality;
    public Set<String> genres = new HashSet<>();
    public LocalDate releaseDate;
    public FeatureVector features;

    public Candidate(Long contentId, String domain) {
        this.contentId = contentId;
        this.domain = domain;
    }
}
```
  `postprocess/PostProcessor.java`:
```java
package com.example.AOD.recommend.postprocess;

import com.example.AOD.recommend.candidate.Candidate;
import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Component
public class PostProcessor {

    private static final Set<String> ADULT_RATINGS = Set.of("19", "청소년이용불가", "ADULT");

    /** spec §5.4-1 hard filter: hidden / unavailable / 성인등급(비허용) 제외. */
    public List<Candidate> hardFilter(List<Candidate> candidates, boolean allowAdult) {
        List<Candidate> out = new ArrayList<>();
        for (Candidate c : candidates) {
            if (c.hidden || c.unavailable) continue;
            if (!allowAdult && c.ageRating != null && ADULT_RATINGS.contains(c.ageRating)) continue;
            out.add(c);
        }
        return out;
    }

    /** spec §5.4-2 soft: negative penalty ≤20%, recency boost ≤8% (score 기준 비율 상한). */
    public void applySoftAdjust(List<Candidate> candidates,
                                Map<Long, Double> negativePenalty,
                                Map<Long, Double> recencyBoost) {
        for (Candidate c : candidates) {
            double penalty = Math.min(negativePenalty.getOrDefault(c.contentId, 0.0), 0.20) * c.score;
            double boost = Math.min(recencyBoost.getOrDefault(c.contentId, 0.0), 0.08) * c.score;
            c.score = c.score - penalty + boost;
        }
    }

    /** spec §5.4-3 diversity: Top-N 내 한 domain ≤60%, 한 platform ≤50%(플랫폼 미상은 캡 면제, 초과분은 후순위 채움). */
    public List<Candidate> diversify(List<Candidate> ranked, int topN) {
        List<Candidate> pool = new ArrayList<>(ranked);
        pool.sort(Comparator.comparingDouble((Candidate c) -> c.score).reversed());
        int domainCap = (int) Math.floor(topN * 0.60);
        int platformCap = (int) Math.floor(topN * 0.50);
        Map<String, Integer> domainCount = new HashMap<>();
        Map<String, Integer> platformCount = new HashMap<>();
        List<Candidate> result = new ArrayList<>();
        List<Candidate> deferred = new ArrayList<>();
        for (Candidate c : pool) {
            if (result.size() >= topN) break;
            String plat = c.platforms.isEmpty() ? null : c.platforms.get(0);
            boolean platformOk = (plat == null) || platformCount.getOrDefault(plat, 0) < platformCap; // 미상 플랫폼 면제
            if (domainCount.getOrDefault(c.domain, 0) < domainCap && platformOk) {
                result.add(c);
                domainCount.merge(c.domain, 1, Integer::sum);
                if (plat != null) platformCount.merge(plat, 1, Integer::sum);
            } else {
                deferred.add(c);
            }
        }
        for (Candidate c : deferred) {
            if (result.size() >= topN) break;
            result.add(c);
        }
        return result;
    }
}
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.postprocess.PostProcessorTest"`
  예상 PASS: `PostProcessorTest > ... PASSED`(4건: hardFilterRemoves…, hardFilterKeeps…, softAdjustCaps…, diversityPromotes…) → `BUILD SUCCESSFUL`.
- [ ] 커밋:
  `git add -A && git commit -m "feat: PostProcessor hard filter, soft caps, diversity re-rank (empty-platform cap-exempt)" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 5: AiAssetRepository — aod_ai 네이티브 조회 (@Query nativeQuery)

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/candidate/AiAssetRepository.java`
- Test: `.../src/test/java/com/example/AOD/recommend/candidate/AiAssetRepositoryTest.java`

근거: contracts §7(pgvector 후보 `ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :k`, fun_tag 후보 `tag = ANY(:tags)`), §1.2(pgvector 파라미터는 `CAST(:vec AS vector)` 문자열 바인딩), §2(테이블/컬럼), §6(`public.contents`). 리포지토리는 기존 엔티티 `com.example.shared.entity.Content`에 바인딩하되 콘텐츠 id/프로젝션만 반환. **구현하는 모든 @Query는 test-driven** — 계약 문자열은 리플렉션으로 고정하고, 실제 실행(pgvector)은 Task 10 통합 테스트로 검증한다. M2 서빙에 쓰이는 메서드만 둔다: `findVectorCandidates`·`findFunTagCandidates`·`findQualityFallback`·`findPositiveCount`(M3 콜드스타트 분기용, 계약 고정)·`loadCandidateRows`(Task 8 하이드레이션). M2 미사용 dead code(loadFunTags/loadEmbeddings)는 넣지 않는다.

- [ ] 실패 테스트 작성 — @Query 계약 문자열/nativeQuery 고정(구현하는 5개 전부):
```java
package com.example.AOD.recommend.candidate;

import org.junit.jupiter.api.Test;
import org.springframework.data.jpa.repository.Query;
import java.lang.reflect.Method;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class AiAssetRepositoryTest {

    private Query queryOf(String name, Class<?>... params) throws Exception {
        Method m = AiAssetRepository.class.getMethod(name, params);
        Query q = m.getAnnotation(Query.class);
        assertNotNull(q, name + " must have @Query");
        assertTrue(q.nativeQuery(), name + " must be nativeQuery");
        return q;
    }

    @Test
    void vectorCandidateQueryMatchesContract() throws Exception {
        assertEquals("SELECT content_id FROM aod_ai.content_embedding "
                + "ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :k",
                queryOf("findVectorCandidates", String.class, int.class).value());
    }

    @Test
    void funTagCandidateQueryMatchesContract() throws Exception {
        assertEquals("SELECT content_id FROM aod_ai.content_fun_tag WHERE tag = ANY(:tags)",
                queryOf("findFunTagCandidates", String[].class).value());
    }

    @Test
    void qualityFallbackQueryOrdersByPopularity() throws Exception {
        assertEquals("SELECT content_id FROM aod_ai.content_quality_score "
                + "ORDER BY quality_popularity_score DESC NULLS LAST LIMIT :k",
                queryOf("findQualityFallback", int.class).value());
    }

    @Test
    void positiveCountQueryReadsUserProfileCache() throws Exception {
        assertEquals("SELECT positive_count FROM aod_ai.user_profile_cache WHERE user_id = :userId",
                queryOf("findPositiveCount", long.class).value());
    }

    @Test
    void loadCandidateRowsJoinsContentsAndQuality() throws Exception {
        assertEquals("SELECT c.content_id, c.domain, c.release_date, q.quality_popularity_score "
                + "FROM public.contents c "
                + "LEFT JOIN aod_ai.content_quality_score q ON q.content_id = c.content_id "
                + "WHERE c.content_id IN (:ids)",
                queryOf("loadCandidateRows", List.class).value());
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.candidate.AiAssetRepositoryTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class AiAssetRepository` → `BUILD FAILED`.
- [ ] 최소 구현 (`candidate/AiAssetRepository.java`):
```java
package com.example.AOD.recommend.candidate;

import com.example.shared.entity.Content;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.Repository;
import org.springframework.data.repository.query.Param;
import java.util.List;

/** aod_ai 스키마 네이티브 조회. contracts §7 SQL 그대로. vec는 '[0.1,0.2,...]' 문자열. */
public interface AiAssetRepository extends Repository<Content, Long> {

    @Query(value = "SELECT content_id FROM aod_ai.content_embedding "
            + "ORDER BY embedding <=> CAST(:vec AS vector) LIMIT :k", nativeQuery = true)
    List<Long> findVectorCandidates(@Param("vec") String vec, @Param("k") int k);

    @Query(value = "SELECT content_id FROM aod_ai.content_fun_tag WHERE tag = ANY(:tags)",
            nativeQuery = true)
    List<Long> findFunTagCandidates(@Param("tags") String[] tags);

    @Query(value = "SELECT content_id FROM aod_ai.content_quality_score "
            + "ORDER BY quality_popularity_score DESC NULLS LAST LIMIT :k", nativeQuery = true)
    List<Long> findQualityFallback(@Param("k") int k);

    /** M3 콜드스타트 분기(§7 positive_count)용 — 계약 고정, M2 서빙 경로에서는 미호출. */
    @Query(value = "SELECT positive_count FROM aod_ai.user_profile_cache WHERE user_id = :userId",
            nativeQuery = true)
    Integer findPositiveCount(@Param("userId") long userId);

    /** 후보 집합 하이드레이션: domain·release_date(public.contents) + quality(aod_ai) 로드. */
    @Query(value = "SELECT c.content_id, c.domain, c.release_date, q.quality_popularity_score "
            + "FROM public.contents c "
            + "LEFT JOIN aod_ai.content_quality_score q ON q.content_id = c.content_id "
            + "WHERE c.content_id IN (:ids)", nativeQuery = true)
    List<Object[]> loadCandidateRows(@Param("ids") List<Long> ids);
}
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.candidate.AiAssetRepositoryTest"`
  예상 PASS: `AiAssetRepositoryTest > ... PASSED`(5건) → `BUILD SUCCESSFUL`.
  (행위 검증) `findVectorCandidates`·`findQualityFallback`·`loadCandidateRows`의 실제 실행은 **Task 10**에서 pgvector Testcontainers로 필수 확인.
- [ ] 커밋:
  `git add -A && git commit -m "feat: AiAssetRepository native pgvector/fun_tag/quality/hydration queries" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 6: CandidateGenerator — union → dedupe (콜드스타트 fallback 포함)

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/candidate/CandidateGenerator.java`
- Test: `.../src/test/java/com/example/AOD/recommend/candidate/CandidateGeneratorTest.java`

근거: contracts §7 / spec §5.1(user profile vector ANN + fun_tag match + selected ANN/tag + quality fallback → 중복 제거 500~1000). 콜드스타트(profile 비었음)면 vector/fun_tag 소스 스킵 → quality fallback만.

- [ ] 실패 테스트 작성 (Mockito 목 리포지토리):
```java
package com.example.AOD.recommend.candidate;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

class CandidateGeneratorTest {

    private final AiAssetRepository repo = mock(AiAssetRepository.class);
    private final CandidateGenerator gen = new CandidateGenerator(repo);

    @Test
    void coldStartUsesOnlyQualityFallback() {
        when(repo.findQualityFallback(5)).thenReturn(List.of(10L, 11L, 12L));
        List<Long> ids = gen.generate(null, new String[0], null, new String[0], 5);
        assertEquals(List.of(10L, 11L, 12L), ids);
        verify(repo, never()).findVectorCandidates(anyString(), anyInt());
        verify(repo, never()).findFunTagCandidates(any());
    }

    @Test
    void unionDedupesAcrossSourcesPreservingOrder() {
        when(repo.findVectorCandidates("[0.1]", 300)).thenReturn(List.of(1L, 2L));
        when(repo.findFunTagCandidates(new String[]{"a"})).thenReturn(List.of(2L, 3L));
        when(repo.findQualityFallback(5)).thenReturn(List.of(3L, 4L));
        List<Long> ids = gen.generate("[0.1]", new String[]{"a"}, null, new String[0], 5);
        assertEquals(List.of(1L, 2L, 3L, 4L), ids);
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.candidate.CandidateGeneratorTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class CandidateGenerator` → `BUILD FAILED`.
- [ ] 최소 구현 (`candidate/CandidateGenerator.java`):
```java
package com.example.AOD.recommend.candidate;

import org.springframework.stereotype.Component;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

@Component
public class CandidateGenerator {

    private static final int ANN_LIMIT = 300; // spec §5.1 소스별 ~300
    private final AiAssetRepository repo;

    public CandidateGenerator(AiAssetRepository repo) {
        this.repo = repo;
    }

    /** spec §5.1 union → dedupe. 비어있는 소스는 스킵(콜드스타트는 fallback만). */
    public List<Long> generate(String profileVector, String[] userTags,
                               String selectedVector, String[] selectedTags, int fallbackK) {
        Set<Long> pool = new LinkedHashSet<>();
        if (profileVector != null) pool.addAll(repo.findVectorCandidates(profileVector, ANN_LIMIT));
        if (userTags != null && userTags.length > 0) pool.addAll(repo.findFunTagCandidates(userTags));
        if (selectedVector != null) pool.addAll(repo.findVectorCandidates(selectedVector, ANN_LIMIT));
        if (selectedTags != null && selectedTags.length > 0) pool.addAll(repo.findFunTagCandidates(selectedTags));
        pool.addAll(repo.findQualityFallback(fallbackK));
        return new ArrayList<>(pool);
    }
}
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.candidate.CandidateGeneratorTest"`
  예상 PASS: `CandidateGeneratorTest > ... PASSED`(2건) → `BUILD SUCCESSFUL`.
- [ ] 커밋:
  `git add -A && git commit -m "feat: CandidateGenerator union/dedupe with cold-start fallback" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 7: RecLogWriter — rec_impression insert

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/dto/RecommendationItem.java`
- Create: `.../src/main/java/com/example/AOD/recommend/log/RecLogWriter.java`
- Test: `.../src/test/java/com/example/AOD/recommend/log/RecLogWriterTest.java`

근거: contracts §2 `rec_impression(request_id, user_id, location, selected_content_id, content_id, candidate_source, rank_position, score_breakdown jsonb)`, §7(각 노출 → rec_impression insert). JdbcTemplate 목으로 경량 검증(무DB).

- [ ] 실패 테스트 작성:
```java
package com.example.AOD.recommend.log;

import com.example.AOD.recommend.dto.RecommendationItem;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import java.util.List;
import java.util.UUID;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

class RecLogWriterTest {

    private final JdbcTemplate jdbc = mock(JdbcTemplate.class);
    private final RecLogWriter writer = new RecLogWriter(jdbc);

    @Test
    void writesOneImpressionRowPerItem() {
        List<RecommendationItem> items = List.of(
                new RecommendationItem(10L, "MOVIE", 0.9, "quality", 0, "{}"),
                new RecommendationItem(11L, "GAME", 0.8, "quality", 1, "{}"));
        writer.writeImpressions(UUID.randomUUID(), 7L, "home", null, items);
        verify(jdbc, times(2)).update(
                contains("INSERT INTO aod_ai.rec_impression"),
                any(), any(), any(), any(), any(), any(), any(), any());
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.log.RecLogWriterTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class RecommendationItem` (및 `class RecLogWriter`) → `BUILD FAILED`.
- [ ] 최소 구현:
  `dto/RecommendationItem.java`:
```java
package com.example.AOD.recommend.dto;

public record RecommendationItem(
        Long contentId,
        String domain,
        double score,
        String candidateSource,
        int rankPosition,
        String scoreBreakdownJson) {}
```
  `log/RecLogWriter.java`:
```java
package com.example.AOD.recommend.log;

import com.example.AOD.recommend.dto.RecommendationItem;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import java.util.List;
import java.util.UUID;

@Component
public class RecLogWriter {

    private static final String INSERT_SQL =
            "INSERT INTO aod_ai.rec_impression "
          + "(request_id, user_id, location, selected_content_id, content_id, "
          + "candidate_source, rank_position, score_breakdown) "
          + "VALUES (?, ?, ?, ?, ?, ?, ?, CAST(? AS jsonb))";

    private final JdbcTemplate jdbc;

    public RecLogWriter(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    /** contracts §7: 노출 아이템마다 rec_impression 1행 insert. */
    public void writeImpressions(UUID requestId, Long userId, String location,
                                 Long selectedContentId, List<RecommendationItem> items) {
        for (RecommendationItem it : items) {
            jdbc.update(INSERT_SQL,
                    requestId, userId, location, selectedContentId, it.contentId(),
                    it.candidateSource(), it.rankPosition(), it.scoreBreakdownJson());
        }
    }
}
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.log.RecLogWriterTest"`
  예상 PASS: `RecLogWriterTest > writesOneImpressionRowPerItem PASSED` → `BUILD SUCCESSFUL`.
- [ ] 커밋:
  `git add -A && git commit -m "feat: RecLogWriter inserts rec_impression per served item" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 8: RecommendService — 오케스트레이션 (hard→soft→diversity→Top-N) + 무조건 콜드스타트 + @Cacheable

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/dto/RecRequest.java`
- Create: `.../src/main/java/com/example/AOD/recommend/dto/ScoreBreakdown.java`
- Create: `.../src/main/java/com/example/AOD/recommend/RecommendService.java`
- Modify: `.../src/main/resources/application.properties`
- Test: `.../src/test/java/com/example/AOD/recommend/RecommendServiceTest.java`

근거: contracts §1.2(캐시 `@Cacheable("homeRecommendations")` 유저별 Top-N), §7(콜드스타트 분기 `user_profile_cache.positive_count`), spec §5.4(후처리 순서 **hard filter → soft penalty/boost → diversity → Top-N**)·§6.2(0건 → quality/popularity fallback). **M2는 무조건 콜드스타트**: 프로파일 비어있음(funtag/profile/metadata=0, quality·recency로 랭킹). **§5.4-2 soft adjust를 실제 서빙에 연결** — 콜드스타트에도 recency_boost(≤8%)는 적용, negative penalty는 유저 negative 프로파일이 없어 빈 맵. **positive_count 분기(§7)와 온보딩 시드 콜드스타트(§6.2)는 M3로 연기**(상단 재조정 노트) — `findPositiveCount`는 리포지토리에 유지하되 M2 경로에서 호출하지 않는다(테스트로 미호출 고정). 캐시 히트 시 로깅 미재실행은 M4 로깅 마일스톤에서 재검토(알려진 M2 한계).

- [ ] 실패 테스트 작성 (CandidateGenerator/AiAssetRepository/RecLogWriter 목, 순수 컴포넌트는 실제 사용):
```java
package com.example.AOD.recommend;

import com.example.AOD.recommend.candidate.CandidateGenerator;
import com.example.AOD.recommend.candidate.AiAssetRepository;
import com.example.AOD.recommend.dto.RecRequest;
import com.example.AOD.recommend.dto.RecommendationItem;
import com.example.AOD.recommend.feature.FeatureCalculator;
import com.example.AOD.recommend.log.RecLogWriter;
import com.example.AOD.recommend.postprocess.PostProcessor;
import com.example.AOD.recommend.rank.Ranker;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import java.sql.Date;
import java.time.LocalDate;
import java.util.List;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class RecommendServiceTest {

    private final CandidateGenerator gen = mock(CandidateGenerator.class);
    private final AiAssetRepository repo = mock(AiAssetRepository.class);
    private final RecLogWriter log = mock(RecLogWriter.class);
    private final RecommendService service = new RecommendService(
            gen, repo, new FeatureCalculator(), new Ranker(), new PostProcessor(), log, new ObjectMapper());

    @Test
    void coldStartRanksByQualityAppliesSoftBoostAndLogsImpressions() {
        Date today = Date.valueOf(LocalDate.now());
        when(gen.generate(isNull(), any(String[].class), isNull(), any(String[].class), anyInt()))
                .thenReturn(List.of(1L, 2L, 3L));
        when(repo.loadCandidateRows(anyList())).thenReturn(List.of(
                new Object[]{1L, "MOVIE", today, 0.9f},
                new Object[]{2L, "GAME", today, 0.5f},
                new Object[]{3L, "TV", today, 0.7f}));

        RecRequest req = new RecRequest("home", null, 0, 20, null);
        List<RecommendationItem> out = service.recommend(req);

        assertEquals(List.of(1L, 3L, 2L), out.stream().map(RecommendationItem::contentId).toList());
        assertEquals(0, out.get(0).rankPosition());
        assertEquals("quality", out.get(0).candidateSource());
        assertTrue(out.get(0).scoreBreakdownJson().contains("quality"));
        // §5.4-2 soft adjust 실제 적용 확인: homeScore(0.15*0.9+0.05*1.0=0.185) + recency_boost 8% = 0.185*1.08.
        assertEquals(0.185 * 1.08, out.get(0).score(), 1e-9);
        verify(repo, never()).findPositiveCount(anyLong()); // M2 무조건 콜드스타트 → 분기 조회 안함
        verify(log).writeImpressions(any(), isNull(), eq("home"), isNull(), anyList());
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.RecommendServiceTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class RecommendService` (및 `RecRequest`) → `BUILD FAILED`.
- [ ] 최소 구현:
  `dto/RecRequest.java`:
```java
package com.example.AOD.recommend.dto;

public record RecRequest(String location, Long selectedContentId, int page, int size, Long userId) {}
```
  `dto/ScoreBreakdown.java` (score_breakdown jsonb 페이로드):
```java
package com.example.AOD.recommend.dto;

public record ScoreBreakdown(
        double funTag, double profileSim, double quality,
        double metadata, double recency, double finalScore) {}
```
  `RecommendService.java`:
```java
package com.example.AOD.recommend;

import com.example.AOD.recommend.candidate.AiAssetRepository;
import com.example.AOD.recommend.candidate.Candidate;
import com.example.AOD.recommend.candidate.CandidateGenerator;
import com.example.AOD.recommend.dto.QualityScore;
import com.example.AOD.recommend.dto.RecRequest;
import com.example.AOD.recommend.dto.RecommendationItem;
import com.example.AOD.recommend.dto.ScoreBreakdown;
import com.example.AOD.recommend.feature.FeatureCalculator;
import com.example.AOD.recommend.feature.FeatureVector;
import com.example.AOD.recommend.log.RecLogWriter;
import com.example.AOD.recommend.postprocess.PostProcessor;
import com.example.AOD.recommend.rank.Ranker;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import java.sql.Date;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
public class RecommendService {

    private static final int POOL_SIZE = 1000;      // spec §5.1 500~1000
    private static final double RECENCY_BOOST_MAX = 0.08; // spec §5.4-2 recency boost ≤8%

    private final CandidateGenerator candidateGenerator;
    private final AiAssetRepository aiAssetRepository;
    private final FeatureCalculator featureCalculator;
    private final Ranker ranker;
    private final PostProcessor postProcessor;
    private final RecLogWriter recLogWriter;
    private final ObjectMapper objectMapper;

    public RecommendService(CandidateGenerator candidateGenerator, AiAssetRepository aiAssetRepository,
                            FeatureCalculator featureCalculator, Ranker ranker, PostProcessor postProcessor,
                            RecLogWriter recLogWriter, ObjectMapper objectMapper) {
        this.candidateGenerator = candidateGenerator;
        this.aiAssetRepository = aiAssetRepository;
        this.featureCalculator = featureCalculator;
        this.ranker = ranker;
        this.postProcessor = postProcessor;
        this.recLogWriter = recLogWriter;
        this.objectMapper = objectMapper;
    }

    @Cacheable(value = "homeRecommendations",
               key = "#req.userId() + ':' + #req.page()",
               condition = "#req.location() == 'home'")
    public List<RecommendationItem> recommend(RecRequest req) {
        // M2: 무조건 콜드스타트(quality/popularity fallback). positive_count 분기·온보딩 시드는 M3.
        List<Long> ids = candidateGenerator.generate(null, new String[0], null, new String[0], POOL_SIZE);
        List<Candidate> pool = loadCandidates(ids);

        LocalDate now = LocalDate.now();
        for (Candidate c : pool) {
            FeatureVector f = new FeatureVector(
                    0.0, // cold-start: fun_tag 프로파일 없음
                    0.0, // cold-start: profile vector 없음
                    featureCalculator.qualityPopularityScore(c.quality),
                    0.0, // cold-start: profile 장르 없음(metadata=0)
                    featureCalculator.recencyScore(c.releaseDate, now));
            c.features = f;
            c.score = ranker.homeScore(f);
        }

        // spec §5.4 순서: hard filter → soft penalty/boost → diversity → Top-N.
        List<Candidate> filtered = postProcessor.hardFilter(pool, false);
        Map<Long, Double> noPenalty = Collections.emptyMap(); // 콜드스타트: 유저 negative 프로파일 없음
        Map<Long, Double> recencyBoost = new HashMap<>();
        for (Candidate c : filtered) {
            recencyBoost.put(c.contentId, c.features.recency() * RECENCY_BOOST_MAX); // §5.4-2 recency boost(콜드스타트에도 적용)
        }
        postProcessor.applySoftAdjust(filtered, noPenalty, recencyBoost);
        List<Candidate> top = postProcessor.diversify(filtered, req.size());

        List<RecommendationItem> items = toItems(top);
        recLogWriter.writeImpressions(UUID.randomUUID(), req.userId(), req.location(),
                req.selectedContentId(), items);
        return items;
    }

    private List<Candidate> loadCandidates(List<Long> ids) {
        if (ids.isEmpty()) return Collections.emptyList();
        List<Candidate> out = new ArrayList<>();
        for (Object[] r : aiAssetRepository.loadCandidateRows(ids)) {
            Candidate c = new Candidate(((Number) r[0]).longValue(), (String) r[1]);
            c.releaseDate = r[2] == null ? null : ((Date) r[2]).toLocalDate();
            double qp = r[3] == null ? 0.0 : ((Number) r[3]).doubleValue();
            c.quality = new QualityScore(0, 0, 0, 0, qp);
            c.candidateSource = "quality";
            out.add(c);
        }
        return out;
    }

    private List<RecommendationItem> toItems(List<Candidate> top) {
        List<RecommendationItem> items = new ArrayList<>();
        for (int i = 0; i < top.size(); i++) {
            Candidate c = top.get(i);
            FeatureVector f = c.features;
            ScoreBreakdown sb = new ScoreBreakdown(
                    f.funTag(), f.profileSim(), f.quality(), f.metadata(), f.recency(), c.score);
            String json;
            try {
                json = objectMapper.writeValueAsString(sb);
            } catch (JsonProcessingException e) {
                json = "{}";
            }
            items.add(new RecommendationItem(c.contentId, c.domain, c.score, c.candidateSource, i, json));
        }
        return items;
    }
}
```
  `application.properties`의 캐시 이름에 `homeRecommendations` 추가:
```properties
spring.cache.cache-names=traditional-recommendations,llm-recommendations,homeRecommendations
```
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.RecommendServiceTest"`
  예상 PASS: `RecommendServiceTest > coldStartRanksByQualityAppliesSoftBoostAndLogsImpressions PASSED` → `BUILD SUCCESSFUL`.
- [ ] 커밋:
  `git add -A && git commit -m "feat: RecommendService orchestration (hard->soft->diversity->topN) + cold-start + homeRecommendations cache" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 9: RecommendController — GET /api/recommendations

**Files:**
- Create: `.../src/main/java/com/example/AOD/recommend/RecommendController.java`
- Test: `.../src/test/java/com/example/AOD/recommend/RecommendControllerTest.java`

근거: contracts §7 엔드포인트 `GET /api/recommendations?location=home|related&selectedContentId={id?}&page=0&size=20`. M2는 principal→userId 매핑(로그인 유저) 없이 익명(userId null) → 콜드스타트 경로 검증; principal 기반 userId 주입은 M3에서 연결(spec §9). `location=related`는 파라미터로 받되 **서빙은 홈만**(related 후보/랭킹은 M3). 컨트롤러 테스트는 `standaloneSetup`으로 보안 미포함 경량 MockMvc.

- [ ] 실패 테스트 작성:
```java
package com.example.AOD.recommend;

import com.example.AOD.recommend.dto.RecRequest;
import com.example.AOD.recommend.dto.RecommendationItem;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;
import java.util.List;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

class RecommendControllerTest {

    @Test
    void homeReturnsRankedItems() throws Exception {
        RecommendService service = mock(RecommendService.class);
        when(service.recommend(any(RecRequest.class))).thenReturn(List.of(
                new RecommendationItem(1L, "MOVIE", 0.9, "quality", 0, "{}")));
        MockMvc mvc = MockMvcBuilders.standaloneSetup(new RecommendController(service)).build();

        mvc.perform(get("/api/recommendations").param("location", "home"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].contentId").value(1))
                .andExpect(jsonPath("$[0].candidateSource").value("quality"))
                .andExpect(jsonPath("$[0].rankPosition").value(0));

        verify(service).recommend(argThat(r ->
                r.location().equals("home") && r.size() == 20 && r.userId() == null));
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.RecommendControllerTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `symbol: class RecommendController` → `BUILD FAILED`.
- [ ] 최소 구현 (`RecommendController.java`):
```java
package com.example.AOD.recommend;

import com.example.AOD.recommend.dto.RecRequest;
import com.example.AOD.recommend.dto.RecommendationItem;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import java.util.List;

@RestController
@RequestMapping("/api/recommendations")
public class RecommendController {

    private final RecommendService recommendService;

    public RecommendController(RecommendService recommendService) {
        this.recommendService = recommendService;
    }

    /** contracts §7: GET /api/recommendations?location=&selectedContentId=&page=&size=.
     *  M2는 홈 서빙 + 익명(userId null) 콜드스타트; principal→userId·related 서빙은 M3. */
    @GetMapping
    public ResponseEntity<List<RecommendationItem>> recommend(
            @RequestParam(defaultValue = "home") String location,
            @RequestParam(required = false) Long selectedContentId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        RecRequest req = new RecRequest(location, selectedContentId, page, size, null);
        return ResponseEntity.ok(recommendService.recommend(req));
    }
}
```
- [ ] 실행하여 통과 확인 (전체 recommend 패키지 회귀, Docker 통합 테스트 제외):
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.*" --tests "!com.example.AOD.recommend.AiAssetIntegrationTest"`
  예상 PASS: `RecommendControllerTest > homeReturnsRankedItems PASSED` 및 M2 단위/계약 테스트 전체 통과 → `BUILD SUCCESSFUL`.
  체크포인트: `GET /api/recommendations?location=home`가 콜드스타트 경로로 랭킹된 `RecommendationItem` Top-N을 반환(오프라인 검수 가능).
- [ ] 커밋:
  `git add -A && git commit -m "feat: RecommendController GET /api/recommendations home serving" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task 10: 통합 검증 (Testcontainers pgvector) — V4 grant 실효 + 네이티브 aod_ai 쿼리 실행 [필수, Docker]

**Files:**
- Create: `.../src/test/resources/db/it/aod_ai_it_setup.sql`
- Test: `.../src/test/java/com/example/AOD/recommend/AiAssetIntegrationTest.java`

근거: contracts §1.2·§7 — grant가 실제로 서빙 계정에 적용되는지, 그리고 pgvector 네이티브 쿼리(`CAST(:vec AS vector)`, `ORDER BY ... DESC NULLS LAST`, 조인)가 실DB에서 실행되는지를 **행위로 검증**(문자열/리플렉션만으론 "틀렸지만 일관된 SQL/grant"를 못 잡음). Flyway가 **Hibernate가 이미 관리하는(비어있지 않은) 스키마 위에서 baseline(3) 후 V4만 깨끗이 migrate** 됨을 확인해 Task 1의 Flyway 도입을 실증한다. Docker 필요 — **optional 아님**. (서빙 계정 postgres는 슈퍼유저라 has_table_privilege가 자명하게 true지만, migrate 성공 + 쿼리 실행이 실질 검증이며, 최소권한 롤 분리 시 동일 단언이 비자명해진다.)

- [ ] 실패 테스트 작성 — 테스트 셋업 SQL(`db/it/aod_ai_it_setup.sql`, 컨테이너 init 시 실행: aod_ai 스키마/테이블 + 최소 public.contents + 시드):
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS aod_ai;

CREATE TABLE aod_ai.content_embedding (
  content_id bigint PRIMARY KEY, embedding vector(1024) NOT NULL, model text NOT NULL, dim int NOT NULL DEFAULT 1024
);
CREATE TABLE aod_ai.content_fun_tag (
  content_id bigint NOT NULL, tag text NOT NULL, tag_score real NOT NULL, tag_confidence real NOT NULL,
  PRIMARY KEY (content_id, tag)
);
CREATE TABLE aod_ai.content_semantic_profile (content_id bigint PRIMARY KEY, domain text, profile_text text, content_hash text);
CREATE TABLE aod_ai.content_quality_score (
  content_id bigint PRIMARY KEY, bayesian_score real, platform_rank_score real, review_count_score real,
  recency_score real, quality_popularity_score real, computed_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE aod_ai.fun_tag_dict (id bigserial PRIMARY KEY, name text NOT NULL UNIQUE);
CREATE TABLE aod_ai.user_profile_cache (user_id bigint PRIMARY KEY, positive_count int NOT NULL DEFAULT 0);
CREATE TABLE aod_ai.rec_impression (
  id bigserial PRIMARY KEY, request_id uuid NOT NULL, user_id bigint, location text NOT NULL,
  selected_content_id bigint, content_id bigint NOT NULL, candidate_source text, rank_position int,
  score_breakdown jsonb, served_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE aod_ai.rec_event (
  id bigserial PRIMARY KEY, request_id uuid, user_id bigint, content_id bigint NOT NULL,
  event_type text NOT NULL, value real, created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE public.contents (
  content_id bigint PRIMARY KEY, domain varchar(50), master_title varchar(500), release_date date
);

INSERT INTO public.contents(content_id, domain, master_title, release_date) VALUES
  (1,'MOVIE','A', DATE '2026-07-01'), (2,'GAME','B', DATE '2026-01-01'), (3,'TV','C', DATE '2026-06-01');
INSERT INTO aod_ai.content_quality_score(content_id, quality_popularity_score) VALUES (1,0.9),(2,0.5),(3,0.7);
INSERT INTO aod_ai.content_fun_tag(content_id,tag,tag_score,tag_confidence) VALUES (1,'힐링',0.8,0.9),(2,'긴장감',0.7,0.6);
INSERT INTO aod_ai.content_embedding(content_id, embedding, model) VALUES
  (1, array_fill(0.1::real, ARRAY[1024])::vector, 'test'),
  (2, array_fill(0.2::real, ARRAY[1024])::vector, 'test'),
  (3, array_fill(0.3::real, ARRAY[1024])::vector, 'test');
```
  통합 테스트(`AiAssetIntegrationTest.java`):
```java
package com.example.AOD.recommend;

import com.example.AOD.recommend.candidate.AiAssetRepository;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import javax.sql.DataSource;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers
@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
class AiAssetIntegrationTest {

    @Container
    static final PostgreSQLContainer<?> PG = new PostgreSQLContainer<>(
            DockerImageName.parse("pgvector/pgvector:pg16").asCompatibleSubstituteFor("postgres"))
            .withUsername("postgres").withPassword("password")
            .withInitScript("db/it/aod_ai_it_setup.sql");

    @DynamicPropertySource
    static void props(DynamicPropertyRegistry r) {
        r.add("spring.datasource.url", PG::getJdbcUrl);
        r.add("spring.datasource.username", PG::getUsername); // = postgres (V4 grant 대상과 일치)
        r.add("spring.datasource.password", PG::getPassword);
        r.add("spring.jpa.hibernate.ddl-auto", () -> "none"); // 스키마는 init script가 제공
        r.add("spring.flyway.enabled", () -> "false");         // 아래에서 Flyway API로 직접 migrate
    }

    @Autowired AiAssetRepository repo;
    @Autowired DataSource dataSource;

    @Test
    void v4GrantAppliesAndNativeAodAiQueriesExecuteOnPgvector() {
        // (1) 실제 Flyway migrate — 비어있지 않은 스키마 위에서 baseline(3) 후 V4만 적용(Task 1 실증).
        Flyway.configure()
                .dataSource(PG.getJdbcUrl(), PG.getUsername(), PG.getPassword())
                .locations("classpath:db/migration")
                .baselineOnMigrate(true).baselineVersion("3")
                .load().migrate();

        JdbcTemplate jdbc = new JdbcTemplate(dataSource);

        // (2) V4 grant 실효 — 서빙 계정(spring.datasource.username=postgres)이 실제 권한 보유(contracts §1.2).
        assertThat(jdbc.queryForObject(
                "SELECT has_table_privilege('postgres','aod_ai.rec_impression','INSERT')", Boolean.class)).isTrue();
        assertThat(jdbc.queryForObject(
                "SELECT has_table_privilege('postgres','aod_ai.content_embedding','SELECT')", Boolean.class)).isTrue();

        // (3) 네이티브 pgvector ANN 실행 — CAST(:vec AS vector) 바인딩 실제 동작(contracts §7).
        String vec = "[" + "0.1,".repeat(1023) + "0.1]";
        List<Long> ann = repo.findVectorCandidates(vec, 3);
        assertThat(ann).containsExactlyInAnyOrder(1L, 2L, 3L);
        assertThat(ann.get(0)).isEqualTo(1L); // embedding=0.1 콘텐츠가 최근접

        // (4) quality fallback ORDER BY 실행 + 하이드레이션 조인 실행(contracts §7).
        assertThat(repo.findQualityFallback(2)).containsExactly(1L, 3L); // 0.9, 0.7
        assertThat(repo.loadCandidateRows(List.of(1L, 2L, 3L))).hasSize(3);
    }
}
```
- [ ] 실행하여 실패 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend.AiAssetIntegrationTest"`
  예상 FAIL: `> Task :-AOD-All-of-Dopamine-api:compileTestJava FAILED` / `error: cannot find symbol` `class AiAssetIntegrationTest` (또는 셋업 리소스 부재로 컨테이너 init 실패) → `BUILD FAILED`.
- [ ] 최소 구현: 위 `aod_ai_it_setup.sql` 리소스 배치 완료(테스트 본문은 실패 단계에서 이미 작성). Flyway 의존성(Task 1)·Testcontainers(build.gradle 기존)·pgvector 이미지로 실행 가능. (`ann.get(0)==1` 등 시드 기대치가 실제 pgvector 거리 계산과 맞도록 셋업 SQL 확정.)
- [ ] 실행하여 통과 확인:
  `./gradlew ":-AOD-All-of-Dopamine-api:test" --tests "com.example.AOD.recommend
.AiAssetIntegrationTest"`
  예상 PASS: `AiAssetIntegrationTest > v4GrantAppliesAndNativeAodAiQueriesExecuteOnPgvector PASSED` → `BUILD SUCCESSFUL` (Docker 필요).
  체크포인트: Flyway V4가 Hibernate 관리 스키마 위에서 깨끗이 migrate + grant 실효 + pgvector 네이티브 쿼리 실행 — 계약(§1.2·§7)이 실제로 성립함을 행위로 확인.
- [ ] 커밋:
  `git add -A && git commit -m "test: Testcontainers pgvector IT verifies V4 grant effective + native aod_ai queries execute" -m "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`
