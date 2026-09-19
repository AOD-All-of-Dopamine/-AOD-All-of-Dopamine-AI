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
