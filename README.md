# 🤖 Argon-SafeKinematics
**Datasheet-Grounded Robotic Trajectory Synthesis & Physical Verification Loop with Gemini 4 Argon**

- **Track:** Track 2 — Gemini 4 Argon in Hard Tech (Robotics & Physical Systems)
- **Live Google Cloud Run Service URL:** [https://argon-safekinematics-508335790284.asia-northeast3.run.app](https://argon-safekinematics-508335790284.asia-northeast3.run.app)
- **Public GitHub Repository:** [https://github.com/0tae-87/gemini_hackathon](https://github.com/0tae-87/gemini_hackathon)

---

## 1. Target User & Engineering Problem
- **Target User:** Robotics Control, Manipulation, and Automation Engineers deploying 7-DOF collaborative/industrial robot arms (**Franka Emika Panda**).
- **Problem:** Frontier LLMs can draft motion plans, but unverified trajectories cannot be executed on physical robots. A single joint limit violation, dynamic torque spike ($>87\text{ Nm} / 12\text{ Nm}$), or collision under 3D depth sensor / point-cloud completion uncertainty ($\sigma$) damages hardware or endangers operators.

## 2. Solution & Gemini 4 Argon Architecture
**Argon-SafeKinematics** couples **`gemini-4-argon`** (`google-genai>=2.3.0` on the **Interactions API**) with an immutable deterministic physics, kinematics, and torque verification harness:
1. **Live Pre-Execution 3D Digital Twin & Point-Cloud Preview:** Before running inference, engineers can interactively adjust 3D target poses, obstacle coordinates, and LiDAR/RGB-D point-cloud uncertainty ($\sigma$) while previewing unverified trajectories and animated robot motion.
2. **Datasheet-Grounded Trajectory Synthesis (`gemini-4-argon`):** Queries `client.interactions.create(model="gemini-4-argon", background=True, generation_config={"thinking_level": ...})` with official Franka Emika Panda Craig Modified DH parameters, joint limits ($q_{\min}, q_{\max}$), velocity bounds ($\dot{q}_{\max}$), and torque limits ($\tau_{\max}$).
3. **Immutable 5-Stage Physical Verifier (`verifier.py`):**
   - **Target Pose Accuracy:** Euclidean TCP error $\| \text{FK}(q_T) - p_{\text{target}} \|_2 \le 20.0\text{ mm}$.
   - **Joint Angle & Velocity Limits:** Strict compliance across all $N=40$ timesteps.
   - **Dynamic Joint Torque Verification:** Evaluates gravitational + inertial acceleration torques against datasheet bounds ($87\text{ Nm}$ on $J_1\text{--}J_4$, $12\text{ Nm}$ on $J_5\text{--}J_7$).
   - **3D Obstacle + Point-Cloud Uncertainty ($\sigma$) Clearance:** Ensures minimum link-to-obstacle distance exceeds $r_{\text{obs}} + r_{\text{link}} + \sigma$.
4. **Stateful Self-Healing Loop (`previous_interaction_id`):** When Iteration 1 fails (e.g., colliding with the uncertainty-inflated obstacle shell), exact offending timesteps, link indices, and negative margins are fed back to `gemini-4-argon` via `previous_interaction_id` to synthesize a collision-free arched trajectory.
5. **Mandatory Human-in-the-Loop Hardware Gate:** Controller JSON export remains locked until all deterministic checks pass and the engineer explicitly approves the 3D clearance envelope and torque profile.

---

## 3. Repository Structure
- `app.py`: Streamlit engineering workbench with instant 3D preview, animated trajectory playback (`▶ Play Motion Video`), live target/obstacle sliders, 7-DOF torque bar charts, and human approval gate.
- `argon_agent.py`: Gemini 4 Argon Interactions API orchestrator (`background=True`, `previous_interaction_id`, numerical IK + C1 smooth spline generation).
- `verifier.py`: Deterministic NumPy 7-DOF Forward Kinematics, gravitational/inertial torque calculator, and 3D segment-to-obstacle clearance verifier.
- `sample_specs.py`: Official Franka Emika Panda 7-DOF datasheet specifications and benchmark scenarios.
- `TODO.md`: Roadmap for CAD mesh rendering and multi-robot URDF expansion.

---

## 4. Local Setup & Reproducibility
```bash
git clone https://github.com/0tae-87/gemini_hackathon.git
cd gemini_hackathon

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your Gemini API key via environment variable (never hardcode keys)
export GEMINI_API_KEY="your_api_key_here"

# Run local server
streamlit run app.py --server.port=8080