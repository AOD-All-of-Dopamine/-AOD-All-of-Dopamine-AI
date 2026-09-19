# 빌드 컨텍스트 = recommendation/  (docker build -f serving/Dockerfile --target dev recommendation/)
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# 네이티브 안전장치. 코드(aod_serving/native.py)도 같은 값을 세팅하지만 — 어떤 기동 방식이든 먹게 —
# 이미지에도 박아 둔다. `system` 인 이유: pyarrow 25 기본 풀인 mimalloc 이 Arrow 를 쓰던 스레드가
# 죽고 그 TLS 블록을 새 스레드가 재활용하면 `mi_thread_init()` 에서 SIGSEGV 를 낸다.
# 할당자만 바뀌므로 수치 결과는 그대로다. 자세한 근거·재현은 aod_serving/native.py 머리말 참고.
ENV ARROW_DEFAULT_MEMORY_POOL=system PYTHONFAULTHANDLER=1

# 개발·테스트용: 코드는 복사하지 않고 레포를 /rec 에 마운트해 쓴다
FROM base AS dev
COPY serving/requirements/dev.txt /tmp/requirements/dev.txt
RUN pip install -r /tmp/requirements/dev.txt
ENV PYTHONPATH=/rec/serving AOD_REC_ROOT=/rec
WORKDIR /rec/serving

# ── 엔진 의존성: PLATFORM 인자와 무관하다 — 아래 engine 타깃보다 먼저 떼어 둬야
# 플랫폼(steam·tmdb·webtoon·webnovel)마다 다시 빌드해도 이 pip 레이어가 캐시로 재사용된다.
# (ARG 는 선언된 지점 아래의 모든 RUN 레이어 캐시 키에 들어간다 — ARG PLATFORM 을 pip install 보다
# 앞에 두면 플랫폼이 바뀔 때마다 무거운 설치가 다시 돈다.)
FROM base AS engine-deps
COPY serving/requirements/serving.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ── 엔진: 코드만 담는다. 아티팩트는 /artifacts 에 읽기 전용으로 마운트한다(§8-6) ──
FROM engine-deps AS engine
ARG PLATFORM
ARG GIT_SHA=dev
RUN test -n "$PLATFORM" || (echo "--build-arg PLATFORM=steam|tmdb|webtoon|webnovel 이 필요하다" && false)
COPY serving/aod_serving /app/serving/aod_serving
COPY schemas /app/schemas
COPY ${PLATFORM}/src /app/${PLATFORM}/src
ENV PYTHONPATH=/app/serving AOD_REC_ROOT=/app ARTIFACTS_ROOT=/artifacts PLATFORM=${PLATFORM} GIT_SHA=${GIT_SHA} PORT=8000 \
    HOME=/tmp
WORKDIR /app
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "aod_serving.engine"]

# ── 라우터: numpy·pandas 없음. 평가된 혼합 규칙(mix.py) 파일 하나만 가져온다 ──
FROM base AS router
ARG GIT_SHA=dev
COPY serving/requirements/router.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY serving/aod_serving /app/serving/aod_serving
COPY crossdomain/mix.py /app/crossdomain/mix.py
ENV PYTHONPATH=/app/serving AOD_REC_ROOT=/app GIT_SHA=${GIT_SHA} PORT=8080 \
    HOME=/tmp
WORKDIR /app
USER 10001:10001
EXPOSE 8080
CMD ["python", "-m", "aod_serving.router"]
