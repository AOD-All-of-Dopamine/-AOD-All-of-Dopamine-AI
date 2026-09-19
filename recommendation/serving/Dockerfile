# 빌드 컨텍스트 = recommendation/  (docker build -f serving/Dockerfile --target dev recommendation/)
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# 개발·테스트용: 코드는 복사하지 않고 레포를 /rec 에 마운트해 쓴다
FROM base AS dev
COPY serving/requirements/dev.txt /tmp/requirements/dev.txt
RUN pip install -r /tmp/requirements/dev.txt
ENV PYTHONPATH=/rec/serving AOD_REC_ROOT=/rec
WORKDIR /rec/serving
