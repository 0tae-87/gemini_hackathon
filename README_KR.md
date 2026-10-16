# 🤖 Argon-SafeKinematics (국문 소개서)
**Gemini 4 Argon 기반 7자유도 로봇 매니퓰레이터 궤적 생성 및 물리·불확실성 검증 루프**

- **출품 트랙:** Track 2 — Gemini 4 Argon in Hard Tech (로보틱스 및 물리 시스템)
- **실시간 Google Cloud Run 서비스 URL:** [https://argon-safekinematics-508335790284.asia-northeast3.run.app](https://argon-safekinematics-508335790284.asia-northeast3.run.app)
- **GitHub 공개 저장소:** [https://github.com/0tae-87/gemini_hackathon](https://github.com/0tae-87/gemini_hackathon)

---

## 1. 해결하려는 문제 및 타겟 사용자
- **타겟 사용자:** 7자유도 협동/산업용 로봇 팔(**Franka Emika Panda**)의 제어 및 모션 플래닝 엔지니어.
- **현장의 문제:** 거대언어모델(LLM)이 생성한 로봇 제어 코드는 물리적 안전성이 보장되지 않아 실제 하드웨어에 즉시 배포할 수 없습니다. 단 한 번의 관절 각도 한계 이탈, 동역학 토크 초과($87\text{ Nm} / 12\text{ Nm}$), 혹은 3D 포인트클라우드 센서 불확실성($\sigma$)으로 인한 충돌도 고가의 로봇 장비 파손이나 안전사고로 이어집니다.

---

## 2. 핵심 솔루션 및 Gemini 4 Argon 활용 아키텍처
**Argon-SafeKinematics**는 **`gemini-4-argon`** (`google-genai>=2.3.0`, **Interactions API**)의 추론 능력과 수정 불가능한(Immutable) **결정론적 파이썬 물리·기구학·토크 검증기(`verifier.py`)**를 폐루프(Closed-loop)로 결합한 클라우드 엔지니어링 워크벤치입니다.

1. **실시간 3D 디지털 트윈 및 포인트클라우드 프리뷰 (`app.py`):**
   - 추론 실행 전에도 목표 위치(Target X, Y, Z), 장애물 위치·반경, LiDAR/RGB-D 포인트클라우드 불확실성 마진($\sigma$)을 실시간으로 조절하며 3D 로봇 모션 애니메이션을 확인할 수 있습니다.
   - 한국어 / 영어 UI 실시간 전환을 지원합니다.
2. **데이터시트 기반 궤적 합성 (`argon_agent.py`):**
   - Franka Emika Panda 공식 데이터시트의 Modified Craig DH 파라미터, 관절 한계($q_{\min}, q_{\max}$), 최대 각속도($\dot{q}_{\max}$), 최대 토크($\tau_{\max}$)를 근거로 3D 직교 공간(Cartesian Task-Space) 궤적을 생성합니다.
3. **5단계 정량적 물리 검증 하네스 (`verifier.py`):**
   - 모델의 자기평가에 의존하지 않고 순수 NumPy 행렬 연산으로 모든 타임스텝($N=40$)을 검증합니다.
   - ① **목표 도달 오차:** 엔드이펙터(TCP) 위치 오차 $\le 20.0\text{ mm}$
   - ② **관절 각도 한계:** 전 구간 $q_{\min} \le q_t \le q_{\max}$ 준수
   - ③ **관절 각속도 한계:** $|\Delta q_t / \Delta t| \le \dot{q}_{\max}$ 준수
   - ④ **동역학 관절 토크 한계:** 중력 모멘트 + 관성 가속도 기반 피크 토크가 데이터시트 허용치($1\text{--}4$축 $87\text{ Nm}$, $5\text{--}7$축 $12\text{ Nm}$) 이내인지 검증
   - ⑤ **3D 장애물 + 센서 불확실성($\sigma$) 충돌 검사:** 모든 로봇 링크 선분과 장애물 사이의 최단 거리가 $r_{\text{obs}} + r_{\text{link}} + \sigma$ 이상인지 검증
4. **자가 교정 루프 (`previous_interaction_id` & `background=True`):**
   - 1회차 직선 경로에서 충돌이나 토크 위반이 감지되면, 위반 타임스텝·링크 번호·안전 마진 수치를 `previous_interaction_id`를 통해 `gemini-4-argon`에 피드백하여 **최소 우회 3D 곡선(Minimal-Detour Bezier Curve)**으로 자율 교정합니다.
5. **인간 개입(Human-in-the-Loop) 배포 승인 게이트:**
   - 모든 물리 검증을 통과한 후 엔지니어가 3D 안전 마진과 토크 그래프를 확인하고 승인 체크박스를 선택해야만 제어기용 JSON 파일을 다운로드할 수 있습니다. 도달 불가능한 목표(S3)는 즉시 출력을 잠그고 인간에게 에스컬레이션합니다.

---

## 3. 검증 시나리오 3종
- **S1 (개방 공간 기본 이동):** 낮은 바닥 장애물 환경에서 직선 경로가 1회차(`Iteration 1/1`)에 즉시 통과함을 확인합니다.
- **S2 (다중 장애물 및 포인트클라우드 회피):** 1회차 직선 경로가 중앙 장애물의 불확실성($\sigma$) 구체와 충돌(`FAIL`)한 뒤, 2회차에 최소 우회 곡선으로 자가 교정되어 통과(`PASS`)합니다.
- **S3 (도달 불가 영역 안전 잠금):** 로봇의 물리적 작업 반경 밖($1.15\text{ m}$) 목표 입력 시 억지로 통과시키지 않고 하드웨어 출력을 안전 잠금(`LOCKED`)합니다.

---

## 4. 로컬 실행 및 재현 방법
```bash
git clone https://github.com/0tae-87/gemini_hackathon.git
cd gemini_hackathon

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export GEMINI_API_KEY="발급받은_API_KEY"
streamlit run app.py --server.port=8080