import os, json
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from sample_specs import FRANKA_PANDA_SPEC, SCENARIOS
from argon_agent import run_verification_loop, solve_ik_numerical, build_smooth_trajectory
from verifier import verify_trajectory

st.set_page_config(page_title="Argon-SafeKinematics | Track 2 Hard Tech", layout="wide")

with st.sidebar:
    lang = st.radio("🌐 Language / 언어", ["English", "한국어"], horizontal=True)
    is_ko = (lang == "한국어")

T = {
    "title": "🤖 Argon-SafeKinematics — 데이터시트 기반 로봇 물리 검증 루프" if is_ko else "🤖 Argon-SafeKinematics — Datasheet-Grounded Robotic Verification Loop",
    "sub": "7자유도 Franka Emika Panda 궤적 생성 · 동역학 토크 · 3D 포인트클라우드 불확실성(σ) 검증" if is_ko else "7-DOF Franka Emika Panda Trajectory Synthesis · Dynamic Torque · 3D Point-Cloud Uncertainty (σ) Verification",
    "badge": "Track 2: Hard Tech · Gemini 4 Argon",
    "cfg_hdr": "⚙️ 시나리오 및 센서 설정" if is_ko else "⚙️ Scenario & Sensor Config",
    "scen_sel": "검증 시나리오" if is_ko else "Benchmark Scenario",
    "sigma_lbl": "포인트클라우드 불확실성 마진 (σ, mm)" if is_ko else "Point-Cloud Uncertainty Margin (σ, mm)",
    "pcd_chk": "3D 포인트클라우드 표시" if is_ko else "Show 3D Point-Cloud",
    "builder_exp": "🎯 목표점 & 장애물 위치 조정" if is_ko else "🎯 Adjust Target & Obstacle Pose",
    "adv_exp": "🧠 추론 설정 및 데이터시트 제원" if is_ko else "🧠 Reasoning & Datasheet Specs",
    "think_lbl": "추론 깊이 (Thinking Level)" if is_ko else "Thinking Level",
    "iter_lbl": "최대 교정 횟수 (Max Iterations)" if is_ko else "Max Repair Iterations",
    "run_btn": "🚀 Gemini 4 Argon 궤적 생성 및 검증 실행" if is_ko else "🚀 Run Gemini 4 Argon Verification Loop",
    "preview_info": "👀 **사전 시뮬레이션 프리뷰:** 좌측 설정 변경 시 즉시 반영됩니다. 실행 버튼을 누르면 최소 우회 안전 경로를 생성·검증합니다." if is_ko else "👀 **Pre-Execution Preview:** Adjust parameters on the left or click **Run Gemini 4 Argon** to synthesize a minimal-detour verified path.",
    "pass_banner": "✅ **모든 기구학·토크·데이터시트 검증 통과!**" if is_ko else "✅ **ALL KINEMATIC, TORQUE & DATASHEET CHECKS PASSED**",
    "fail_banner": "🛑 **검증 실패 — 하드웨어 출력 잠금 및 엔지니어 확인 필요**" if is_ko else "🛑 **VERIFICATION FAILED — Hardware export locked & escalated to engineer.**",
    "kpi1": "검증 상태" if is_ko else "Verification Status",
    "kpi2": "목표 위치 오차" if is_ko else "Target Pose Error",
    "kpi3": "최소 안전거리 마진" if is_ko else "Min Safety Clearance",
    "kpi4": "최대 관절 토크 비율" if is_ko else "Peak Joint Torque Ratio",
    "twin_hdr": "🌐 3D 작업공간 시뮬레이션" if is_ko else "🌐 3D Workspace Simulation",
    "trace_hdr": "🔬 자율 검증 루프 로그" if is_ko else "🔬 Verification Loop Log",
    "gate_hdr": "🛡️ 하드웨어 제어기 배포 승인" if is_ko else "🛡️ Hardware Controller Export Gate",
    "gate_chk": "안전 마진과 관절 토크 한계를 모두 확인했으며, 제어기 배포를 승인합니다." if is_ko else "I have verified the clearance margin and joint torque bounds, and approve controller export.",
    "gate_btn": "📥 검증 완료 궤적 다운로드 (JSON)" if is_ko else "📥 Export Verified Trajectory (JSON)",
    "gate_lock": "⚠️ 모든 물리 검사를 통과하고 엔지니어가 승인하기 전까지 하드웨어 출력이 잠겨 있습니다." if is_ko else "⚠️ Controller export remains LOCKED until all deterministic physical checks pass and human approval is granted."
}

st.markdown(f"""
<style>
.block-container {{padding-top: 3.2rem; padding-bottom: 1.5rem; max-width: 96%;}}
div[data-baseweb="select"] span {{white-space: normal !important; line-height: 1.3 !important;}}
ul[data-testid="stSelectboxVirtualDropdown"] li {{white-space: normal !important; height: auto !important; padding-top: 8px !important; padding-bottom: 8px !important;}}
.hero-header {{
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
    padding: 16px 24px; border-radius: 12px; color: white; margin-bottom: 14px;
    border-left: 5px solid #38bdf8;
}}
.hero-title {{font-size: 1.38rem; font-weight: 800; margin: 0; color: #f8fafc;}}
.hero-sub {{font-size: 0.85rem; color: #94a3b8; margin-top: 4px;}}
.hero-badge {{
    background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8;
    padding: 6px 14px; border-radius: 999px; font-size: 0.80rem; font-weight: 700; white-space: nowrap;
}}
.kpi-grid {{display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 10px 0 16px 0;}}
.kpi-card {{
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
    padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}}
.kpi-label {{font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase;}}
.kpi-val {{font-size: 1.55rem; font-weight: 800; color: #0f172a; margin: 3px 0;}}
.kpi-sub {{font-size: 0.74rem; font-weight: 700; display: inline-block; padding: 2px 8px; border-radius: 99px;}}
.sub-ok {{background: #dcfce7; color: #15803d;}}
.sub-warn {{background: #fee2e2; color: #b91c1c;}}
.sub-info {{background: #e0f2fe; color: #0369a1;}}
.metric-table {{width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 8px 0;}}
.metric-table td {{padding: 5px 8px; border-bottom: 1px solid #e2e8f0;}}
.viol-box {{background: #fff1f2; border-left: 4px solid #f43f5e; padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; color: #9f1239; margin-top: 6px;}}
</style>

<div class="hero-header">
  <div>
    <div class="hero-title">{T['title']}</div>
    <div class="hero-sub">{T['sub']}</div>
  </div>
  <div class="hero-badge">{T['badge']}</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader(T["cfg_hdr"])
    scen_keys = list(SCENARIOS.keys())
    scenario_name = st.selectbox(
        T["scen_sel"],
        scen_keys,
        index=1,
        format_func=lambda k: SCENARIOS[k].get("ko_title", k) if is_ko else k
    )
    base_scen = SCENARIOS[scenario_name]
    st.caption(base_scen.get("ko_desc", base_scen["description"]) if is_ko else base_scen["description"])

    sigma_mm = st.slider(
        T["sigma_lbl"],
        min_value=0, max_value=60, value=int(base_scen["default_uncertainty"]*1000), step=5
    )
    sigma_m = sigma_mm / 1000.0
    show_pcd = st.checkbox(T["pcd_chk"], value=True)

    with st.expander(T["builder_exp"], expanded=False):
        tx = st.slider("Target X (m)", -0.30, 0.75, float(base_scen["target_pos"][0]), 0.01)
        ty = st.slider("Target Y (m)", -0.50, 0.60, float(base_scen["target_pos"][1]), 0.01)
        tz = st.slider("Target Z (m)", 0.10, 0.85, float(base_scen["target_pos"][2]), 0.01)

        custom_obs = [dict(o) for o in base_scen["obstacles"]]
        if len(custom_obs) > 0:
            st.caption("Primary Obstacle (#1):")
            o0 = custom_obs[0]
            ox = st.slider("Obstacle #1 X (m)", -0.20, 0.70, float(o0["center"][0]), 0.01)
            oy = st.slider("Obstacle #1 Y (m)", -0.40, 0.40, float(o0["center"][1]), 0.01)
            oz = st.slider("Obstacle #1 Z (m)", 0.05, 0.70, float(o0["center"][2]), 0.01)
            orad = st.slider("Obstacle #1 Radius (m)", 0.03, 0.14, float(o0["radius"]), 0.005)
            custom_obs[0] = {"type": "sphere", "center": [ox, oy, oz], "radius": orad}

    with st.expander(T["adv_exp"], expanded=False):
        thinking_level = st.selectbox(T["think_lbl"], ["medium", "high", "low"], index=0)
        max_iters = st.slider(T["iter_lbl"], 1, 4, 3)
        st.json({
            "platform": FRANKA_PANDA_SPEC["name"],
            "q_min_rad": FRANKA_PANDA_SPEC["q_min"],
            "q_max_rad": FRANKA_PANDA_SPEC["q_max"],
            "dq_max_rad_s": FRANKA_PANDA_SPEC["dq_max"],
            "tau_max_nm": FRANKA_PANDA_SPEC.get("tau_max", [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        })

    scen = {
        "description": base_scen["description"],
        "start_q": base_scen["start_q"],
        "target_pos": [tx, ty, tz],
        "obstacles": custom_obs,
        "default_uncertainty": sigma_m,
        "via_pos": base_scen.get("via_pos", [0.34, 0.0, 0.52])
    }

def sample_point_cloud(center, radius, sigma, n_pts=220):
    rng = np.random.default_rng(42)
    dirs = rng.normal(size=(n_pts, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r_samples = radius + rng.normal(0.0, max(0.004, sigma * 0.6), size=(n_pts, 1))
    return np.round(np.array(center) + dirs * r_samples, 4).tolist()

def build_preview_history(scen, sigma_m):
    start_q = np.array(scen["start_q"], dtype=float)
    goal_q = solve_ik_numerical(scen["target_pos"], start_q, FRANKA_PANDA_SPEC)
    traj = build_smooth_trajectory([start_q, goal_q], num_steps=40)
    ver = verify_trajectory(traj, FRANKA_PANDA_SPEC, scen, sigma_m, dt=0.08)
    return [{
        "iteration": 0,
        "interaction_id": "PREVIEW_BASELINE",
        "cited_clauses": [
            "Franka Datasheet Sec 3.1: Joint Position & Velocity Limits",
            f"3D Point-Cloud Safety Margin: r_obs + r_link + σ ({int(sigma_m*1000)} mm)"
        ],
        "strategy": "Direct unverified Cartesian straight-line interpolation (Pre-Argon baseline).",
        "trajectory": traj,
        "verification": ver
    }]

def render_threejs_cad_viewer(history, scen, sigma_m, show_pcd=True, is_preview=False):
    all_iters = []
    for rec in history:
        raw_links = np.asarray(rec["verification"]["link_trajectories"], dtype=float)
        all_iters.append({
            "iteration": int(rec["iteration"]),
            "passed": bool(rec["verification"]["passed"]),
            "frames": np.round(raw_links, 4).tolist(),
            "ee": np.round(raw_links[:, -1, :], 4).tolist()
        })

    obs_data = []
    for obs in scen["obstacles"]:
        c = [float(x) for x in obs["center"]]
        r = float(obs["radius"])
        pcd = sample_point_cloud(c, r, sigma_m) if show_pcd else []
        obs_data.append({
            "center": c,
            "radius": r,
            "envelope": float(r + sigma_m + FRANKA_PANDA_SPEC["link_radius"]),
            "pcd": pcd
        })

    payload = json.dumps({
        "iters": all_iters,
        "target": [float(x) for x in scen["target_pos"]],
        "obstacles": obs_data,
        "is_preview": bool(is_preview)
    })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
      <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
      <style>
        body {{ margin: 0; overflow: hidden; background: #0b1120; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
        #hud {{
          position: absolute; top: 12px; left: 12px; z-index: 10;
          background: rgba(15, 23, 42, 0.86); border: 1px solid #334155;
          padding: 8px 12px; border-radius: 8px; color: #f8fafc; font-size: 11.5px;
        }}
        #iter-switch {{
          position: absolute; top: 12px; right: 12px; z-index: 10;
          display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end;
        }}
        .mode-btn {{
          background: rgba(15, 23, 42, 0.9); color: #cbd5e1; border: 1px solid #475569;
          padding: 5px 10px; border-radius: 7px; font-size: 11.5px; font-weight: 700; cursor: pointer;
        }}
        .mode-btn.active-pass {{ background: #166534; color: #ffffff; border-color: #22c55e; }}
        .mode-btn.active-fail {{ background: #991b1b; color: #ffffff; border-color: #ef4444; }}
        .leg-row {{ display: flex; align-items: center; gap: 7px; margin: 3px 0; }}
        .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
        #controls {{
          position: absolute; bottom: 12px; left: 12px; right: 12px; z-index: 10;
          background: rgba(15, 23, 42, 0.92); border: 1px solid #334155;
          padding: 7px 14px; border-radius: 10px; display: flex; align-items: center; gap: 12px; color: white;
        }}
        button.play {{
          background: #0284c7; color: white; border: none; padding: 5px 13px;
          border-radius: 6px; font-weight: 700; cursor: pointer; font-size: 12px;
        }}
        input[type=range] {{ flex: 1; accent-color: #38bdf8; cursor: pointer; }}
        #step-label {{ font-size: 12px; font-weight: 700; color: #38bdf8; min-width: 120px; text-align: right; }}
      </style>
    </head>
    <body>
      <div id="hud">
        <div class="leg-row"><span class="dot" style="background:#22c55e;"></span> Verified Trajectory <span style="color:#64748b;">(Blue=Preview)</span></div>
        <div class="leg-row"><span class="dot" style="background:#ef4444;"></span> Failed Attempt Path & Obstacle</div>
        <div class="leg-row"><span class="dot" style="background:#f97316;"></span> σ Point-Cloud Safety Bound</div>
      </div>
      <div id="iter-switch"></div>
      <div id="controls">
        <button id="play-btn" class="play">⏸ Pause</button>
        <input id="slider" type="range" min="0" max="39" value="0">
        <span id="step-label">Step: 0 / 39</span>
      </div>
      <div id="canvas-container" style="width:100%; height:490px;"></div>

      <script>
        const DATA = {payload};
        const container = document.getElementById('canvas-container');
        const W = container.clientWidth || 800;
        const H = 490;
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0b1120);

        function toThree(pt) {{ return new THREE.Vector3(pt[0], pt[2], -pt[1]); }}

        const camera = new THREE.PerspectiveCamera(40, W / H, 0.01, 20);
        camera.position.set(1.12, 0.82, 1.05);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(W, H);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.target.set(0.22, 0.30, 0.0);
        controls.enableDamping = true;

        scene.add(new THREE.AmbientLight(0xffffff, 0.72));
        const dLight = new THREE.DirectionalLight(0xffffff, 0.9);
        dLight.position.set(2.5, 4.0, 2.0);
        scene.add(dLight);
        const rLight = new THREE.DirectionalLight(0x38bdf8, 0.4);
        rLight.position.set(-2.0, 2.0, -2.0);
        scene.add(rLight);

        scene.add(new THREE.GridHelper(1.8, 18, 0x334155, 0x1e293b));
        const basePlate = new THREE.Mesh(
          new THREE.CylinderGeometry(0.075, 0.09, 0.04, 32),
          new THREE.MeshStandardMaterial({{ color: 0x334155, metalness: 0.6, roughness: 0.35 }})
        );
        basePlate.position.set(0, 0.02, 0);
        scene.add(basePlate);

        const tPos = toThree(DATA.target);
        const targetMesh = new THREE.Mesh(
          new THREE.OctahedronGeometry(0.022, 0),
          new THREE.MeshStandardMaterial({{ color: 0xfacc15, emissive: 0xca8a04, emissiveIntensity: 0.6 }})
        );
        targetMesh.position.copy(tPos);
        scene.add(targetMesh);

        DATA.obstacles.forEach(obs => {{
          const c = toThree(obs.center);
          const core = new THREE.Mesh(
            new THREE.SphereGeometry(obs.radius, 28, 20),
            new THREE.MeshStandardMaterial({{ color: 0xef4444, roughness: 0.3, transparent: true, opacity: 0.86 }})
          );
          core.position.copy(c);
          scene.add(core);

          const env = new THREE.Mesh(
            new THREE.SphereGeometry(obs.envelope, 24, 18),
            new THREE.MeshStandardMaterial({{ color: 0xf97316, transparent: true, opacity: 0.14, depthWrite: false }})
          );
          env.position.copy(c);
          scene.add(env);

          const wire = new THREE.Mesh(
            new THREE.SphereGeometry(obs.envelope, 14, 10),
            new THREE.MeshBasicMaterial({{ color: 0xf97316, wireframe: true, transparent: true, opacity: 0.24 }})
          );
          wire.position.copy(c);
          scene.add(wire);

          if (obs.pcd && obs.pcd.length > 0) {{
            const ptsGeo = new THREE.BufferGeometry();
            const pos = new Float32Array(obs.pcd.length * 3);
            obs.pcd.forEach((p, idx) => {{
              const v = toThree(p);
              pos[idx*3] = v.x; pos[idx*3+1] = v.y; pos[idx*3+2] = v.z;
            }});
            ptsGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            const ptsMat = new THREE.PointsMaterial({{ size: 0.008, color: 0x38bdf8, transparent: true, opacity: 0.85 }});
            scene.add(new THREE.Points(ptsGeo, ptsMat));
          }}
        }});

        // Draw trajectories for ALL iterations in history
        DATA.iters.forEach((itObj, idx) => {{
          const eePts = itObj.ee.map(toThree);
          const curve = new THREE.CatmullRomCurve3(eePts);
          const isLast = (idx === DATA.iters.length - 1);
          const col = DATA.is_preview ? 0x38bdf8 : (itObj.passed ? 0x22c55e : 0xff4b4b);
          const radius = isLast ? 0.006 : 0.0042;
          const opacity = isLast ? 0.95 : 0.72;
          scene.add(new THREE.Mesh(
            new THREE.TubeGeometry(curve, 56, radius, 10, false),
            new THREE.MeshStandardMaterial({{ color: col, emissive: col, emissiveIntensity: 0.3, transparent: true, opacity: opacity }})
          ));
        }});

        const robotGroup = new THREE.Group();
        scene.add(robotGroup);

        const whiteMat = new THREE.MeshStandardMaterial({{ color: 0xf8fafc, metalness: 0.2, roughness: 0.32 }});
        const redWarnMat = new THREE.MeshStandardMaterial({{ color: 0xfca5a5, metalness: 0.2, roughness: 0.32 }});
        const darkMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, metalness: 0.65, roughness: 0.28 }});
        const ringMat = new THREE.MeshStandardMaterial({{ color: 0x0284c7, emissive: 0x0284c7, emissiveIntensity: 0.3 }});
        const radii = [0.054, 0.050, 0.046, 0.042, 0.038, 0.034, 0.030, 0.026];

        let activeIterIdx = DATA.iters.length - 1;
        const switchBox = document.getElementById('iter-switch');

        function renderIterButtons() {{
          if (DATA.is_preview || DATA.iters.length <= 1) return;
          switchBox.innerHTML = '';
          DATA.iters.forEach((itObj, idx) => {{
            const b = document.createElement('button');
            const icon = itObj.passed ? '🟢' : '🔴';
            const statusTxt = itObj.passed ? 'PASS' : 'FAIL';
            b.innerText = `${{icon}} Iter ${{itObj.iteration}} (${{statusTxt}})`;
            b.className = 'mode-btn' + (idx === activeIterIdx ? (itObj.passed ? ' active-pass' : ' active-fail') : '');
            b.onclick = () => {{
              activeIterIdx = idx;
              renderIterButtons();
              setStep(curStep);
            }};
            switchBox.appendChild(b);
          }});
        }}
        renderIterButtons();

        function buildConnectedCapsule(p1, p2, r1, r2, bodyMat) {{
          const group = new THREE.Group();
          const dir = new THREE.Vector3().subVectors(p2, p1);
          const len = dir.length();

          const hub1 = new THREE.Mesh(new THREE.SphereGeometry(r1 * 1.12, 24, 18), darkMat);
          hub1.position.copy(p1);
          group.add(hub1);

          if (len > 0.02) {{
            const collar = new THREE.Mesh(new THREE.CylinderGeometry(r1 * 1.14, r1 * 1.14, 0.018, 24), ringMat);
            collar.position.copy(p1);
            collar.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
            group.add(collar);

            const linkBody = new THREE.Mesh(new THREE.CylinderGeometry(r2 * 0.95, r1 * 0.98, len, 24), bodyMat);
            const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
            linkBody.position.copy(mid);
            linkBody.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
            group.add(linkBody);
          }}
          return group;
        }}

        function updateRobot(frameIdx) {{
          while (robotGroup.children.length > 0) {{ robotGroup.remove(robotGroup.children[0]); }}
          const curIter = DATA.iters[activeIterIdx];
          const pts = curIter.frames[frameIdx].map(toThree);
          const bodyMat = (!DATA.is_preview && !curIter.passed) ? redWarnMat : whiteMat;

          const cleanPts = [pts[0]];
          for (let i = 1; i < pts.length; i++) {{
            if (pts[i].distanceTo(cleanPts[cleanPts.length - 1]) > 0.015) {{
              cleanPts.push(pts[i]);
            }}
          }}

          for (let i = 0; i < cleanPts.length - 1; i++) {{
            const r1 = radii[Math.min(i, radii.length - 1)];
            const r2 = radii[Math.min(i + 1, radii.length - 1)];
            robotGroup.add(buildConnectedCapsule(cleanPts[i], cleanPts[i+1], r1, r2, bodyMat));
          }}

          const tcp = cleanPts[cleanPts.length - 1];
          const prev = cleanPts[cleanPts.length - 2] || new THREE.Vector3(0, 1, 0);
          const dir = new THREE.Vector3().subVectors(tcp, prev).normalize();

          const wristFlange = new THREE.Mesh(new THREE.SphereGeometry(0.030, 20, 16), darkMat);
          wristFlange.position.copy(tcp);
          robotGroup.add(wristFlange);

          const handGroup = new THREE.Group();
          handGroup.position.copy(tcp);
          handGroup.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);

          const palm = new THREE.Mesh(new THREE.BoxGeometry(0.084, 0.030, 0.040), darkMat);
          palm.position.set(0, 0.016, 0);
          handGroup.add(palm);

          const fL = new THREE.Mesh(new THREE.BoxGeometry(0.011, 0.042, 0.016), bodyMat);
          fL.position.set(0.030, 0.044, 0);
          handGroup.add(fL);

          const fR = new THREE.Mesh(new THREE.BoxGeometry(0.011, 0.042, 0.016), bodyMat);
          fR.position.set(-0.030, 0.044, 0);
          handGroup.add(fR);

          robotGroup.add(handGroup);
        }}

        let curStep = 0;
        let playing = true;
        let lastTick = performance.now();
        const slider = document.getElementById('slider');
        const stepLabel = document.getElementById('step-label');
        const playBtn = document.getElementById('play-btn');

        function setStep(s) {{
          curStep = s;
          slider.value = s;
          const curIter = DATA.iters[activeIterIdx];
          const tag = DATA.is_preview ? "Preview" : `Iter ${{curIter.iteration}}`;
          stepLabel.innerText = `${{tag}} · ${{s}} / ${{curIter.frames.length - 1}}`;
          updateRobot(s);
        }}

        playBtn.onclick = () => {{
          playing = !playing;
          playBtn.innerText = playing ? '⏸ Pause' : '▶ Play';
        }};

        slider.oninput = (e) => {{
          playing = false;
          playBtn.innerText = '▶ Play';
          setStep(parseInt(e.target.value));
        }};

        setStep(0);

        function animate(now) {{
          requestAnimationFrame(animate);
          const curIter = DATA.iters[activeIterIdx];
          if (playing && now - lastTick > 65) {{
            lastTick = now;
            setStep((curStep + 1) % curIter.frames.length);
          }}
          targetMesh.rotation.y += 0.03;
          controls.update();
          renderer.render(scene, camera);
        }}
        requestAnimationFrame(animate);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=505)

def plot_torque_profile(ver, is_ko=False):
    m = ver["metrics"]
    peaks = m.get("joint_peak_torques_nm", [0]*7)
    limits = m.get("tau_max_nm", [87,87,87,87,12,12,12])
    joints = [f"J{i+1}" for i in range(7)]
    ratios = [p / l * 100.0 for p, l in zip(peaks, limits)]
    colors = ["#22c55e" if r <= 100 else "#ef4444" for r in ratios]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=joints, y=peaks, marker_color=colors,
        text=[f"{p:.1f}Nm ({r:.0f}%)" for p, r in zip(peaks, ratios)],
        textposition="outside", name="최대 토크 (Nm)" if is_ko else "Peak Torque (Nm)"
    ))
    fig.add_trace(go.Scatter(
        x=joints, y=limits, mode="lines+markers",
        line=dict(color="#facc15", width=2, dash="dot"),
        name="데이터시트 한계 (Nm)" if is_ko else "Datasheet Limit (Nm)"
    ))
    fig.update_layout(
        template="plotly_dark",
        title="7축 관절 토크 vs 데이터시트 한계 (Nm)" if is_ko else "7-DOF Joint Torque vs. Datasheet Limits (Nm)",
        yaxis=dict(title="Torque (Nm)", range=[0, 105]),
        height=240,
        margin=dict(l=20, r=20, t=36, b=16),
        legend=dict(orientation="h", y=1.15)
    )
    return fig

scen_signature = f"{scenario_name}_{tx}_{ty}_{tz}_{sigma_mm}_{ len(custom_obs) and custom_obs[0]['center'] }_{ len(custom_obs) and custom_obs[0]['radius'] }"
if st.session_state.get("last_sig") != scen_signature:
    st.session_state["last_sig"] = scen_signature
    st.session_state.pop("history", None)

col_btn, col_info = st.columns([2, 3])
with col_btn:
    run_btn = st.button(T["run_btn"], type="primary", use_container_width=True)

if run_btn:
    status_box = st.empty()
    def update_status(msg):
        status_box.info(f"⏳ {msg}")

    with st.spinner("Running Gemini 4 Argon Verification Loop..."):
        history = run_verification_loop(
            FRANKA_PANDA_SPEC, scenario_name, scen,
            sigma_uncertainty=sigma_m, max_iters=max_iters,
            thinking_level=thinking_level, status_cb=update_status
        )
        st.session_state["history"] = history
    status_box.empty()

is_preview = "history" not in st.session_state
active_history = build_preview_history(scen, sigma_m) if is_preview else st.session_state["history"]
final_rec = active_history[-1]
ver = final_rec["verification"]
m = ver["metrics"]

with col_info:
    if is_preview:
        st.info(T["preview_info"])
    elif ver["passed"]:
        st.success(f"{T['pass_banner']} (Iteration {final_rec['iteration']}/{len(active_history)})")
    else:
        st.error(T["fail_banner"])

status_text = ("프리뷰" if is_ko else "PREVIEW") if is_preview else (("통과 (PASS)" if is_ko else "PASS (Verified)") if ver["passed"] else ("실패 (LOCKED)" if is_ko else "FAIL (Locked)"))
status_cls = "sub-info" if is_preview else ("sub-ok" if ver["passed"] else "sub-warn")
status_sub = ("미검증 초기 경로" if is_ko else "Unverified Baseline") if is_preview else f"Iteration {final_rec['iteration']}/{len(active_history)}"

clr_val = m["min_clearance_margin_m"] * 1000.0
clr_cls = "sub-ok" if clr_val >= 0 else "sub-warn"
clr_sub = ("안전 (>= 0 mm)" if is_ko else "SAFE (>= 0 mm)") if clr_val >= 0 else ("충돌 (< 0 mm)" if is_ko else "COLLISION")

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">{T['kpi1']}</div>
    <div class="kpi-val">{status_text}</div>
    <span class="kpi-sub {status_cls}">{status_sub}</span>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{T['kpi2']}</div>
    <div class="kpi-val">{m['target_error_m']*1000:.1f} mm</div>
    <span class="kpi-sub sub-ok">Limit: ≤ 20.0 mm</span>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{T['kpi3']}</div>
    <div class="kpi-val">{clr_val:+.1f} mm</div>
    <span class="kpi-sub {clr_cls}">{clr_sub}</span>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">{T['kpi4']}</div>
    <div class="kpi-val">{m.get('max_torque_ratio', 0)*100:.1f}%</div>
    <span class="kpi-sub sub-ok">Max: 87 / 12 Nm</span>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader(T["twin_hdr"])
    render_threejs_cad_viewer(active_history, scen, sigma_m, show_pcd=show_pcd, is_preview=is_preview)
    st.plotly_chart(plot_torque_profile(ver, is_ko=is_ko), use_container_width=True)

with c2:
    st.subheader(T["trace_hdr"])
    for rec in active_history:
        r_ver = rec["verification"]
        rm = r_ver["metrics"]
        badge = "🔵 PREVIEW" if is_preview else ("🟢 PASS" if r_ver["passed"] else "🔴 FAIL")
        title = ("초기 기구학 프리뷰 (미검증)" if is_ko else "Baseline Preview (Unverified)") if is_preview else f"Iteration {rec['iteration']} · {badge} (ID: {str(rec['interaction_id'])[:12]}..)"
        with st.expander(title, expanded=True):
            st.markdown(f"**Argon Strategy:** {rec['strategy']}")
            st.markdown(f"""
            <table class="metric-table">
              <tr><td><b>Target Error</b></td><td><code>{rm['target_error_m']*1000:.1f} mm</code> (≤ 20 mm)</td>
                  <td><b>Max Velocity</b></td><td><code>{rm['max_velocity_ratio']*100:.1f}%</code> (≤ 100%)</td></tr>
              <tr><td><b>Peak Torque</b></td><td><code>{rm.get('max_torque_ratio', 0)*100:.1f}%</code> (≤ 100%)</td>
                  <td><b>Min Clearance</b></td><td><code>{rm['min_clearance_margin_m']*1000:+.1f} mm</code> (≥ 0 mm)</td></tr>
            </table>
            """, unsafe_allow_html=True)
            if r_ver["violations"]:
                v_summary = f"🚨 {len(r_ver['violations'])} Violations (e.g., {r_ver['violations'][0]})"
                st.markdown(f'<div class="viol-box">{v_summary}</div>', unsafe_allow_html=True)
            st.caption("Cited: " + " | ".join(rec["cited_clauses"]))

st.divider()
st.subheader(T["gate_hdr"])
if (not is_preview) and ver["passed"]:
    approved = st.checkbox(T["gate_chk"])
    export_payload = json.dumps({
        "robot": FRANKA_PANDA_SPEC["name"],
        "scenario": scenario_name,
        "uncertainty_margin_m": sigma_m,
        "verified": True,
        "argon_interaction_id": final_rec["interaction_id"],
        "cited_datasheet_clauses": final_rec["cited_clauses"],
        "peak_joint_torques_nm": ver["metrics"].get("joint_peak_torques_nm", []),
        "joint_trajectory_rad": final_rec["trajectory"].tolist()
    }, indent=2)
    st.download_button(
        T["gate_btn"],
        data=export_payload,
        file_name="franka_verified_trajectory.json",
        mime="application/json",
        disabled=not approved
    )
else:
    st.warning(T["gate_lock"])