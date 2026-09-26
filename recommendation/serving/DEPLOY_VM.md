# 추천 서빙 — 리눅스 VM 배포 런북

작성: 2026-09-26 (같은 날 명령어 모음 추가) · 대상: 추천 엔진 4개(Steam · TMDB · 웹툰 · 웹소설) + 라우터를 VM 한 대에 올려 백엔드와 잇는다.

- **서버를 띄우는 법**은 `scripts/deploy.sh` 가 한다(README §5-0). 이 문서는 그 앞뒤 — **VM 준비와, 띄운 서버를 서비스에 붙이는 법**까지 순서대로 적는다.
- 단계마다 **누가**(직접 = 콘솔·서버 작업 / 코드 = 저장소 수정)와 **성공하면 보이는 것**을 적었다. 성공 표시가 안 보이면 다음 단계로 가지 않는다.

```
0 VM 결정 → 1 코드 준비 → 2 VM 준비 → 3 기동·검증 → 4 네트워크 → 5 백엔드 연결 → 6 팀만 켜기 → 전체 공개
```

---

## 따라 하기 — 명령어 모음

처음 올리는 사람이 **위에서부터 그대로** 따라 하면 끝나도록 적었다. 각 단계의 이유·배경은 아래 0~6절에 있다.

> ⚠️ **임베딩 파일(약 1.2GB)은 GitHub 에 없다**(`.gitignore`). 코퍼스를 만든 개발 머신(`guest-a`, 저장소 `/home/ubuntu/-AOD-All-of-Dopamine-AI`)에만 있으므로 **③ push 는 그 머신에서** 돌린다.
> 다른 PC 에서 clone 해 push 하면 `… 가 없다` 로 멈춘다. VM 의 SSH 키(.pem)를 그 머신에 두거나, 그 머신에서 VM 으로 SSH 가 되게 한다.

### ① VM 만들기 — AWS 콘솔

1. **API 서버 정보 확인**: EC2 → 인스턴스 → 백엔드 API 서버 → **네트워킹** 탭에서 VPC ID · 서브넷 ID · **보안 그룹 ID(`sg-…`)** 를 적는다.
2. **인스턴스 시작**

   | 항목 | 값 |
   |---|---|
   | 이름 | `aod-rec` |
   | AMI | **Ubuntu Server 24.04 LTS** |
   | 인스턴스 유형 | **`m6i.xlarge`**(4 vCPU · 16GB, 권장) / 최소 `m6i.large`(2 vCPU · 8GB) |
   | 키 페어 | 새로 만들고 `aod-rec.pem` 다운로드 |
   | 네트워크 | 1번의 **같은 VPC · 같은 서브넷** |
   | 퍼블릭 IP 자동 할당 | 활성화 (SSH 용) |
   | 스토리지 | **30GB gp3** |

3. **보안 그룹** 새로 만들기(`aod-rec-sg`)

   | 유형 | 포트 | 소스 |
   |---|---|---|
   | SSH | 22 | 작업자 IP · push 할 개발 머신 IP |
   | 사용자 지정 TCP | **8080** | **1번의 API 서버 보안 그룹 ID** |

   8080 을 `0.0.0.0/0` 으로 열면 **안 된다** — 라우터에는 인증이 없다.
4. 시작 후 **퍼블릭 IP**(SSH 용)와 **프라이빗 IP**(`10.x.x.x`, 서비스 용)를 적는다.

### ② VM 설정 — 작업자 PC 에서 SSH

```bash
chmod 400 aod-rec.pem
ssh -i aod-rec.pem ubuntu@<퍼블릭IP>
```
```bash
# ── VM 안 ──
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
sudo systemctl enable docker
exit                                   # docker 그룹 반영을 위해 다시 접속
```
```bash
ssh -i aod-rec.pem ubuntu@<퍼블릭IP>
docker compose version                 # 버전이 나오면 OK

git clone https://github.com/AOD-All-of-Dopamine/-AOD-All-of-Dopamine-AI.git
cd ./-AOD-All-of-Dopamine-AI/recommendation/serving   # './' 필수 — '-' 로 시작하면 cd 가 옵션으로 읽는다

cp deploy.env.example deploy.env
sed -i "s/^BIND_IP=.*/BIND_IP=$(hostname -I | awk '{print $1}')/" deploy.env
grep BIND_IP deploy.env                # BIND_IP=10.x.x.x 면 OK
```

### ③ 임베딩 보내기 — **임베딩이 있는 개발 머신에서**

보내는 것 — 저장소 `/home/ubuntu/-AOD-All-of-Dopamine-AI` 안의 코퍼스 4개를 VM 의 `/srv/aod-artifacts/` 로 복사한다(`deploy.env` 의 `ARTIFACTS_BASE` · `*_CORPUS` 기본값).

| 플랫폼 | 원본 (개발 머신) | 도착 (VM) | 용량 |
|---|---|---|---|
| steam | `recommendation/steam/artifacts/tags_full/` | `/srv/aod-artifacts/steam/tags_full/` | 764M |
| tmdb | `recommendation/tmdb/artifacts/tmdb_v1/` | `/srv/aod-artifacts/tmdb/tmdb_v1/` | 281M |
| webnovel | `recommendation/webnovel/artifacts/wn_v6/` | `/srv/aod-artifacts/webnovel/wn_v6/` | 143M |
| webtoon | `recommendation/webtoon/artifacts/wt_v1/` | `/srv/aod-artifacts/webtoon/wt_v1/` | 17M |

```bash
chmod 400 ~/aod-rec.pem
cat >> ~/.ssh/config <<'EOF'
Host aod-rec
  HostName <퍼블릭IP>
  User ubuntu
  IdentityFile ~/aod-rec.pem
EOF

cd /home/ubuntu/-AOD-All-of-Dopamine-AI
git pull
recommendation/serving/scripts/deploy.sh push aod-rec
```
끝에 `✓ 전송 끝` 이 보이면 OK (약 1.2GB, 수 분).

### ④ 띄우기 — VM 에서

```bash
cd ~/-AOD-All-of-Dopamine-AI/recommendation/serving
scripts/deploy.sh all
```
점검 → 이미지 빌드(첫 빌드 수 분) → 파일 검증 → 기동(Steam 최대 5분) → 추천 한 건. 성공하면 마지막 줄:
```
✓ 항목 5 개 · [...] · partial []
✓ 끝. 백엔드에 REC_ROUTER_BASE_URL=http://10.x.x.x:8080 를 넣으면 ...
```
이 `http://10.x.x.x:8080` 을 복사해 둔다.

### ⑤ 백엔드에서 닿는지 — API 서버에서

```bash
# API 서버(EC2)에 SSH 로 들어가서
curl -s http://<VM프라이빗IP>:8080/health
```
`{"ready":true, ... "engines":{...}}` 가 나오면 OK. **작업자 PC 에서** `curl http://<VM퍼블릭IP>:8080/health` 는 **응답이 없어야** 정상(외부 차단).

### ⑥ 백엔드 연결 — GitHub

1. 백엔드 저장소 → Settings → Secrets and variables → Actions → **New repository secret**

   | 이름 | 값 |
   |---|---|
   | `REC_ROUTER_BASE_URL` | `http://<VM프라이빗IP>:8080` |
   | `REC_ALLOWED_USERS` | 팀원 로그인 아이디를 쉼표로 (예: `user1,user2`) |

2. Actions → **CI/CD Pipeline - API & Crawler** → **Run workflow** → `main` → 초록불까지 대기(약 5분)
3. 팀 계정으로 https://allofdophamin.com 로그인 → `/for-you` → 개발자 도구 → Network → `recommendations` 응답

   | 응답 | 뜻 |
   |---|---|
   | `"fallback": false` | ✅ 성공 — 추천 엔진 결과 |
   | `"fallbackReason": "disabled"` | 허용 목록 밖 계정 (아이디 확인) |
   | `"timeout"` · `"service_error"` · `"circuit_open"` | 백엔드가 VM 에 못 닿음 → ⑤ 다시 |
   | `"no_seed"` | 좋아요가 없는 계정 → 작품 좋아요 후 다시 |

### ⑦ 전체 공개

1. 팀이 한 시간쯤 써 보고 이상이 없으면
2. 비밀값 **`REC_ALLOWED_USERS` 를 삭제**(GitHub 는 빈 값을 못 넣는다 — 삭제하면 전원 허용) → Run workflow
3. 끝. **프론트는 할 일이 없다** — 홈 추천 릴은 이미 항상 켜져 있다(프론트 #52, 2026-09-26). 추천 서버가 붙기 전·허용 목록 밖에서는 릴이 인기 목록으로 보이다가, 전체 공개하면 로그인 사용자 홈에 개인화 추천이 뜬다.

### 문제가 생기면

| 증상 | 명령 · 조치 |
|---|---|
| `deploy.sh all` 이 `✗` 로 멈춤 | 그 줄 안내대로 고친 뒤 `scripts/deploy.sh all` |
| 기동이 15분 넘게 안 끝남 | `scripts/deploy.sh logs rec-steam` |
| 상태 보기 | `scripts/deploy.sh ps` · `docker stats --no-stream` |
| **추천을 급히 꺼야 함** | 비밀값 `REC_ENABLED` = `false` 추가 → Run workflow (화면은 인기 목록으로 정상) |
| VM 을 내림 | `scripts/deploy.sh down` (백엔드는 자동으로 인기 목록으로 돌아간다) |

---

## 0. VM 결정 〔직접〕

### 위치 — 이것부터 정한다

| VM 위치 | 백엔드와 통신 | 할 일 |
|---|---|---|
| **AWS 서울(ap-northeast-2), 백엔드 API 서버와 같은 VPC** (권장) | 사설 IP 로 직접 | 보안 그룹만 연다 |
| 그 밖(다른 클라우드 · 사내·집 서버) | 공인 인터넷을 거친다 | **VPN(Tailscale · WireGuard) 필수** — 라우터에는 인증이 없어 인터넷에 열면 누구나 부를 수 있다 |

### 사양

| 항목 | 최소 | 권장 | 근거 |
|---|---|---|---|
| 메모리 | 6 GB | **8 GB** | 4개 엔진 실측 약 3.3 GB(anon 2.0 + 임베딩 캐시 1.3) + 라우터·OS. 4 GB 는 여유가 없어 Steam 캐시가 밀리면 요청마다 디스크를 다시 읽는다 |
| vCPU | 2 | **4 이상** | compose 한도 합계 5.5. 부족하면 동시 요청에서 스로틀링 |
| 디스크 | 15 GB | 20 GB | 이미지 5장 + 아티팩트 1.2 GB |
| OS | Ubuntu 22.04 / 24.04 | | Docker 공식 지원 |
| CPU 종류 | x86_64 · arm64 모두 | | 이미지를 VM 에서 빌드하므로 상관없다 |

AWS 예: `m6i.large`(2 vCPU · 8 GB, 최소) · `m6i.xlarge`(4 vCPU · 16 GB, 여유) · `c6i.2xlarge`(8 vCPU · 16 GB).

**성공하면**: VM 에 SSH 로 들어갈 수 있고, VM 의 사설 IP 를 알고 있다.

---

## 1. 코드 준비 〔코드 — 완료〕

배포 전에 막히던 것들. 브랜치 `feature/serving-deploy-prep` 에서 끝냈다.

| 커밋 | 내용 |
|---|---|
| `87bd068` | Steam 이 서빙에 안 쓰는 컬럼 6개를 적재하지 않는다 — 피크 메모리 2,151 → 1,824 MB, 결과 불변(기준선 59 사례 어긋남 0) |
| `b407bd7` | TMDB `manifest.json` — 없으면 운영 모드에서 TMDB 엔진이 기동을 거부했다 |
| `2c0b06b` | 전체 탭 앞 k 개를 평가된 M6@k 그대로(REC_TAB_DESIGN 부록 D A5) |
| `9eb9e20` | `scripts/deploy.sh` |

**완료**: AI #2 로 `main` 에 병합됐다(2026-09-26 `2a5f0fe`). VM 은 `main` 을 clone 한다.

---

## 2. VM 준비 〔직접〕

1. **Docker · Compose 설치**
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER      # 그 뒤 로그아웃·재로그인
   docker compose version             # 플러그인 확인
   ```
2. **저장소 clone** (`main`)
   ```bash
   git clone https://github.com/AOD-All-of-Dopamine/-AOD-All-of-Dopamine-AI.git
   ```
3. **아티팩트 보내기** — **개발 PC 에서** 돌린다. 임베딩(`.npy`)과 TMDB 파일은 git 에 없어서 따로 보내야 한다
   ```bash
   recommendation/serving/scripts/deploy.sh push ubuntu@<VM>
   ```
   코퍼스 4개(steam 764M · tmdb 281M · webnovel 143M · webtoon 17M, 약 1.2 GB)를 `/srv/aod-artifacts/` 로 보내고 읽기 권한까지 준다. 다음부터는 바뀐 파일만 간다.
4. **설정 파일**
   ```bash
   cd ./-AOD-All-of-Dopamine-AI/recommendation/serving
   cp deploy.env.example deploy.env
   # BIND_IP= 에 VM 사설 IP (hostname -I 로 확인)
   ```
   `deploy.env` 는 커밋되지 않는다(gitignore).

**성공하면**: `scripts/deploy.sh check` 가 `✓ 점검 통과` 로 끝난다.

---

## 3. 기동 · 검증 〔직접 — 명령 하나〕

```bash
scripts/deploy.sh all
```

| 단계 | 하는 일 | 실패하면 |
|---|---|---|
| `check` | 도커 권한 · 메모리 · vCPU · 디스크 · 아티팩트 4종과 권한 · `BIND_IP` | `✗` 줄을 고친다 |
| `build` | 엔진 4장 + 라우터 1장 빌드(태그 = git 커밋) · 첫 빌드 수 분 | 네트워크(pip) 확인 |
| `verify` | 전송된 파일을 커밋된 manifest 의 sha256 과 대조 | `push` 를 다시 하고 `verify` |
| `up` | 기동, 라우터 응답까지 대기(최대 15분 — Steam 이 가장 길다) | `scripts/deploy.sh logs rec-steam` |
| `smoke` | 라우터 `/health` + 게임 탭 추천 한 건 | 로그 확인 |

**성공하면**: 마지막에 `✓ 항목 5 개 · [...] · partial []` 와 `✓ 끝` 이 보인다.

### 3-1. 이 VM 에서 부하 관문 다시 재기 (권장)

지금 `compose.yaml` 의 메모리·CPU 한도는 **개발 PC 에서 잰 값**이다(`LOADGATE_RESULTS.md`). 실제 VM 에서 다시 잰다.

```bash
docker run --rm --network aod-rec_aod-rec -v "$PWD/..:/rec:ro" --tmpfs /tmp aod-rec-dev:latest \
  python -m aod_serving.tools.loadgate --out /tmp/g.json --platforms steam --skip-router
```
(`aod-rec-dev` 는 `docker build -f Dockerfile --target dev -t aod-rec-dev:latest ..` 로 한 번 만든다.)
동시 5 에서 Steam p95 가 예산(2.5초)을 넘으면 `LOADGATE_RESULTS.md` "판정과 권고"의 순서(카탈로그 켜기 → Steam 복제 → cpus 상향)를 따른다.

---

## 4. 네트워크 〔직접〕

1. **보안 그룹 · 방화벽**: VM 의 **8080 을 백엔드 API 서버에서 오는 것만** 허용한다.
   - AWS 같은 VPC: VM 보안 그룹 인바운드 `TCP 8080` · 소스 = API 서버의 보안 그룹
   - 엔진 포트 8000 은 열지 않는다(compose 가 호스트에 노출하지 않는다)
2. **API 서버에서 불러 보기** — API 서버에 SSH 로 들어가서
   ```bash
   curl -s http://<VM 사설 IP>:8080/health
   ```
3. (카탈로그를 켤 때만) VM → `api.allofdophamin.com` 으로 나가는 HTTPS 가 열려 있어야 한다.

**성공하면**: API 서버에서 `{"ready":true, ... "engines":{...}}` 가 보인다. **인터넷에서는 안 보여야 한다**(개인 PC 에서 같은 주소가 막히는지도 확인).

---

## 5. 백엔드 연결 〔코드 + 직접〕

지금 백엔드는 라우터 주소를 받을 통로가 없다 — 기본값 `http://localhost:18080` 으로 부르다 실패해 **모든 요청이 대체 목록**으로 끝난다.

### 5-1. 코드 (백엔드 저장소) — ✅ 완료 (#126, 2026-09-26 운영 배포)

| 파일 | 추가 |
|---|---|
| `.github/workflows/main.yml` · Deploy API 단계 `env:` | `REC_ROUTER_BASE_URL: ${{ secrets.REC_ROUTER_BASE_URL }}` · `REC_ALLOWED_USERS: ${{ secrets.REC_ALLOWED_USERS }}` |
| 같은 단계 원격 셸 | `export REC_ROUTER_BASE_URL="${REC_ROUTER_BASE_URL}"` · `export REC_ALLOWED_USERS="${REC_ALLOWED_USERS}"` |
| `-AOD-All-of-Dopamine-api/docker-compose.yml` `environment:` | `REC_ROUTER_BASE_URL: ${REC_ROUTER_BASE_URL}` · `REC_ALLOWED_USERS: ${REC_ALLOWED_USERS}` |

Spring 이 환경변수 `REC_ROUTER_BASE_URL` 을 `rec.router.base-url` 로, `REC_ALLOWED_USERS` 를 `rec.allowed-users` 로 읽는다(코드 변경 없음).

### 5-2. 비밀값 등록 〔직접〕

GitHub → 백엔드 저장소 → Settings → Secrets and variables → Actions:
- `REC_ROUTER_BASE_URL` = `http://<VM 사설 IP>:8080`
- `REC_ALLOWED_USERS` = 팀원 아이디를 쉼표로 (6단계에서 비운다)

**주의**: 백엔드는 `main` 에 병합하면 **바로 운영에 배포**된다(`main.yml`). 5-1 PR 과 5-2 비밀값이 같이 준비된 뒤 병합한다.

---

## 6. 팀만 켜기 → 전체 공개 〔직접〕

1. 5-1 PR 병합 → 배포 끝나기 기다림
2. **팀 계정**으로 로그인해 `/for-you` 를 연다. 응답을 개발자 도구에서 보면:
   - `"fallback": false` → **엔진에서 온 추천** (성공)
   - `"fallbackReason": "disabled"` → 허용 목록 밖 계정 (정상 — 팀 외 사용자는 이렇게 보인다)
   - `"fallbackReason": "service_error" | "timeout" | "circuit_open"` → 라우터에 못 닿는다 → 4단계로
3. 한 시간쯤 지켜본다
   ```sql
   SELECT fallback, fallback_reason, count(*) FROM aod_log.rec_request
    WHERE served_at > now() - interval '1 hour' GROUP BY 1, 2 ORDER BY 3 DESC;
   ```
   팀 계정 요청의 `fallback = false` 비율이 높고 `timeout` 이 거의 없으면 통과.
4. **전체 공개**: `REC_ALLOWED_USERS` 비밀값을 비우고 백엔드를 다시 배포(허용 목록이 비면 전원 허용).
5. 홈은 따로 켤 것이 없다 — 추천 릴은 플래그 없이 항상 나간다(프론트 #52, 예전 `VITE_HOME_REC` 는 삭제됨). 전체 공개 뒤로 로그인 사용자 홈에 "내 취향 추천"이 뜬다.

---

## 되돌리기

| 상황 | 할 일 | 결과 |
|---|---|---|
| 추천이 이상하다 · 느리다 | 백엔드 `REC_ENABLED=false` 로 재배포(킬 스위치 `RecFeatureFlag`) | 모든 요청이 대체 목록(`disabled`). 화면은 정상 |
| VM 이 죽었다 | 할 일 없음 | 백엔드 서킷이 열려 대체 목록으로 답한다(`circuit_open`). VM 이 살아나면 30초 뒤 다시 시도 |
| 배포한 버전이 문제다 | VM 에서 이전 커밋으로 `git checkout` → `scripts/deploy.sh build && scripts/deploy.sh up` | 이미지 태그가 커밋이라 이전 이미지가 남아 있으면 빌드도 금방 |
| 서버를 내린다 | `scripts/deploy.sh down` | 컨테이너만 내린다. 아티팩트·이미지는 남는다 |

(`REC_ENABLED` 통로도 #126 에 들어갔다 — 비밀값 `REC_ENABLED=false` 만으로 끌 수 있다. 비밀값이 없으면 `true`.)

---

## 운영

- **상태**: `scripts/deploy.sh ps` · 메모리 `docker stats --no-stream` · 로그 `scripts/deploy.sh logs rec-steam`
- **재부팅**: 컨테이너가 `restart: unless-stopped` 라 도커가 켜지면 같이 뜬다(`sudo systemctl enable docker`)
- **코퍼스 교체**(새 임베딩): 새 폴더를 `push` → `deploy.env` 의 `*_CORPUS` 변경 → `scripts/deploy.sh verify && scripts/deploy.sh up`. 평가 기준과 다른 코퍼스는 운영 모드에서 `config.json` 승인이 있어야 뜬다(README §6)
- **알려진 한계**: 라우터 `/health` 는 엔진이 모두 죽어도 200 이다(REC_TAB_DESIGN 부록 D B7) — 장애 감시는 엔진 쪽(`ps` 의 health 상태)을 본다

---

## 체크리스트

- [ ] 0 VM 위치·사양 결정, SSH 가능
- [x] 1 코드 준비 — AI #2 병합(`2a5f0fe`)
- [ ] 2 Docker 설치 · clone · `push` · `deploy.env` → `check` 통과
- [ ] 3 `all` → `✓ 끝` · (권장) 부하 관문 재측정
- [ ] 4 보안 그룹 · API 서버에서 `/health` 보임 · 인터넷에서 안 보임
- [x] 5-1 백엔드 통로(`REC_ROUTER_BASE_URL` · `REC_ALLOWED_USERS` · `REC_ENABLED`) — 백엔드 #126 병합·배포
- [ ] 5-2 비밀값 등록
- [ ] 6 팀 계정 `fallback:false` 확인 → 한 시간 관찰 → 전체 공개 (홈 릴은 이미 켜져 있음)
