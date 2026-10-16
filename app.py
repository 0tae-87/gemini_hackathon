import os, json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sample_specs import FRANKA_PANDA_SPEC, SCENARIOS
from argon_agent import run_verification_loop

st.set_page_config(page_title="Argon-SafeKinematics | Track 2 Hard Tech", layout="wide")

st.title("🤖 Argon-SafeKinematics: Datasheet-Grounded Robotic Verification Loop")
st.caption("Track 2: Gemini 4 Argon in Hard Tech | 7-DOF Franka Emika Panda Kinematics, Torque & 3D Point-Cloud Uncertainty Verification")

with st.sidebar:
    st.header("⚙️ Hardware & Scenario Config")
    st.markdown("**Robot Platform:** `Franka Emika Panda (7-DOF)`")
    scenario_name = st.selectbox("Select Engineering Scenario", list(SCENARIOS.keys()), index=1)
    base_scen = SCENARIOS[scenario_name]
    st.info(base_scen["description"])

    sigma_mm = st.slider(
        "3D Point-Cloud / Sensor Uncertainty (σ, mm)",
        min_value=0, max_value=60, value=int(base_scen["default_uncertainty"]*1000), step=5
    )
    sigma_m = sigma_mm / 1000.0
    show_pcd = st.checkbox("Render 3D Depth Point-Cloud (LiDAR/RGB-D Noise)", value=True)

    with st.expander("🎯 Interactive Target & Obstacle Builder", expanded=False):
        st.caption("Customize 3D Target Pose & Obstacle Coordinates live:")
        tx = st.slider("Target X (m)", -0.30, 0.70, float(base_scen["target_pos"][0]), 0.01)
        ty = st.slider("Target Y (m)", -0.50, 0.60, float(base_scen["target_pos"][1]), 0.01)
        tz = st.slider("Target Z (m)", 0.10, 0.85, float(base_scen["target_pos"][2]), 0.01)

        custom_obs = []
        if len(base_scen["obstacles"]) > 0:
            o0 = base_scen["obstacles"][0]
            ox = st.slider("Obstacle X (m)", -0.20, 0.70, float(o0["center"][0]), 0.01)
            oy = st.slider("Obstacle Y (m)", -0.40, 0.40, float(o0["center"][1]), 0.01)
            oz = st.slider("Obstacle Z (m)", 0.05, 0.70, float(o0["center"][2]), 0.01)
            orad = st.slider("Obstacle Radius (m)", 0.03, 0.14, float(o0["radius"]), 0.005)
            custom_obs.append({"type": "sphere", "center": [ox, oy, oz], "radius": orad})

    scen = {
        "description": base_scen["description"],
        "start_q": base_scen["start_q"],
        "target_pos": [tx, ty, tz],
        "obstacles": custom_obs,
        "default_uncertainty": sigma_m
    }

    thinking_level = st.selectbox("Gemini 4 Argon Thinking Level", ["medium", "high", "low"], index=0)
    max_iters = st.slider("Max Autonomous Repair Iterations", 1, 4, 3)

    with st.expander("📄 Official Franka Datasheet Specs"):
        st.json({
            "q_min_rad": FRANKA_PANDA_SPEC["q_min"],
            "q_max_rad": FRANKA_PANDA_SPEC["q_max"],
            "dq_max_rad_s": FRANKA_PANDA_SPEC["dq_max"],
            "tau_max_nm": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
            "link_safety_radius_m": FRANKA_PANDA_SPEC["link_radius"]
        })

def create_sphere_mesh(center, radius, color, opacity, name):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 14)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v)).flatten()
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v)).flatten()
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v)).flatten()
    return go.Mesh3d(x=x, y=y, z=z, color=color, opacity=opacity, alphahull=0, name=name, showscale=False)

def sample_point_cloud(center, radius, sigma, n_pts=240):
    rng = np.random.default_rng(42)
    dirs = rng.normal(size=(n_pts, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r_samples = radius + rng.normal(0.0, max(0.004, sigma * 0.6), size=(n_pts, 1))
    return np.array(center) + dirs * r_samples

def plot_3d_workspace(history, scen, sigma_m, step_idx=-1, show_pcd=True):
    fig = go.Figure()

    # Robot Base Pedestal
    fig.add_trace(create_sphere_mesh([0, 0, 0.05], 0.08, "#475569", 0.85, "Robot Base Pedestal"))

    # Target Pose
    tp = scen["target_pos"]
    fig.add_trace(go.Scatter3d(
        x=[tp[0]], y=[tp[1]], z=[tp[2]],
        mode="markers+text", text=["TARGET"], textposition="top center",
        marker=dict(size=9, color="#facc15", symbol="diamond", line=dict(width=2, color="white")),
        name="Target Pose"
    ))

    # Obstacles + Uncertainty Shell + 3D Point-Cloud
    for idx, obs in enumerate(scen["obstacles"]):
        c = obs["center"]
        r = obs["radius"]
        fig.add_trace(create_sphere_mesh(c, r, "#ef4444", 0.65, f"Obstacle {idx+1} Solid Core"))
        if sigma_m > 0:
            fig.add_trace(create_sphere_mesh(
                c, r + sigma_m + FRANKA_PANDA_SPEC["link_radius"],
                "#f97316", 0.16, f"Obs {idx+1} + σ Safety Envelope"
            ))
        if show_pcd:
            pcd = sample_point_cloud(c, r, sigma_m)
            fig.add_trace(go.Scatter3d(
                x=pcd[:,0], y=pcd[:,1], z=pcd[:,2],
                mode="markers",
                marker=dict(size=2.2, color=pcd[:,2], colorscale="Turbo", opacity=0.85),
                name=f"3D Sensor Point-Cloud (σ={int(sigma_m*1000)}mm)"
            ))

    # Iteration 1 Failed Trajectory vs Final Verified Trajectory
    if len(history) > 1 and not history[0]["verification"]["passed"]:
        ee_init = history[0]["verification"]["link_trajectories"][:, -1, :]
        fig.add_trace(go.Scatter3d(
            x=ee_init[:,0], y=ee_init[:,1], z=ee_init[:,2],
            mode="lines", line=dict(color="#ff4b4b", width=6, dash="dash"),
            name="Iter 1: Unverified Path (Collision)"
        ))

    final_rec = history[-1]
    link_trajs = final_rec["verification"]["link_trajectories"]
    ee_final = link_trajs[:, -1, :]
    traj_color = "#22c55e" if final_rec["verification"]["passed"] else "#f97316"
    fig.add_trace(go.Scatter3d(
        x=ee_final[:,0], y=ee_final[:,1], z=ee_final[:,2],
        mode="lines+markers", marker=dict(size=3.5, color=traj_color),
        line=dict(color=traj_color, width=8),
        name=f"Iter {final_rec['iteration']}: Verified Trajectory"
    ))

    # Franka 7-DOF Volumetric Arm Links + Joints + End-Effector Gripper
    robot_pts = link_trajs[step_idx]
    fig.add_trace(go.Scatter3d(
        x=robot_pts[:,0], y=robot_pts[:,1], z=robot_pts[:,2],
        mode="lines+markers",
        marker=dict(size=8, color="#38bdf8", line=dict(width=2, color="#0284c7")),
        line=dict(color="#e2e8f0", width=16),
        name="Franka Panda 7-DOF Arm"
    ))
    ee_pt = robot_pts[-1]
    fig.add_trace(go.Scatter3d(
        x=[ee_pt[0]], y=[ee_pt[1]], z=[ee_pt[2]],
        mode="markers",
        marker=dict(size=10, color="#a855f7", symbol="cross"),
        name="Parallel Gripper TCP"
    ))

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(range=[-0.5, 0.8], title="X (m)", backgroundcolor="#0f172a", gridcolor="#334155"),
            yaxis=dict(range=[-0.5, 0.8], title="Y (m)", backgroundcolor="#0f172a", gridcolor="#334155"),
            zaxis=dict(range=[0.0, 1.0], title="Z (m)", backgroundcolor="#0f172a", gridcolor="#334155"),
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=25),
        height=560,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(15,23,42,0.75)")
    )
    return fig

def plot_torque_profile(ver):
    m = ver["metrics"]
    peaks = m.get("joint_peak_torques_nm", [0]*7)
    limits = m.get("tau_max_nm", [87,87,87,87,12,12,12])
    joints = [f"J{i+1}" for i in range(7)]
    ratios = [p / l * 100.0 for p, l in zip(peaks, limits)]
    colors = ["#22c55e" if r <= 100 else "#ef4444" for r in ratios]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=joints, y=peaks, marker_color=colors,
        text=[f"{p:.1f} Nm ({r:.0f}%)" for p, r in zip(peaks, ratios)],
        textposition="outside", name="Peak Joint Torque (Nm)"
    ))
    fig.add_trace(go.Scatter(
        x=joints, y=limits, mode="lines+markers",
        line=dict(color="#facc15", width=2, dash="dot"),
        name="Datasheet Max Torque Limit (Nm)"
    ))
    fig.update_layout(
        template="plotly_dark",
        title="7-DOF Joint Peak Torque vs. Datasheet Limits (Nm)",
        yaxis=dict(title="Torque (Nm)", range=[0, 105]),
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.12)
    )
    return fig

run_btn = st.button("🚀 Run Argon Synthesis & Verification Loop", type="primary", use_container_width=True)

if run_btn:
    status_box = st.empty()
    def update_status(msg):
        status_box.info(f"⏳ {msg}")

    with st.spinner("Executing Gemini 4 Argon Verification Loop..."):
        history = run_verification_loop(
            FRANKA_PANDA_SPEC, scenario_name, scen,
            sigma_uncertainty=sigma_m, max_iters=max_iters,
            thinking_level=thinking_level, status_cb=update_status
        )
        st.session_state["history"] = history
    status_box.empty()

if "history" in st.session_state:
    history = st.session_state["history"]
    final_rec = history[-1]
    passed = final_rec["verification"]["passed"]

    if passed:
        st.success(f"✅ ALL PHYSICAL, TORQUE & DATASHEET CHECKS PASSED on Iteration {final_rec['iteration']}/{len(history)}!")
    else:
        st.error(f"🛑 VERIFICATION FAILED after {len(history)} iterations. Mandatory Human Escalation Triggered.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("🌐 3D Point-Cloud Workspace & Kinematic Envelope")
        step_view = st.slider("Inspect Trajectory Timestep (0 = Start, 39 = Goal)", 0, 39, 20)
        st.plotly_chart(plot_3d_workspace(history, scen, sigma_m, step_idx=step_view, show_pcd=show_pcd), use_container_width=True)
        st.plotly_chart(plot_torque_profile(final_rec["verification"]), use_container_width=True)

    with c2:
        st.subheader("🔬 Autonomous Verification Loop Trace")
        for rec in history:
            ver = rec["verification"]
            badge = "🟢 PASS" if ver["passed"] else "🔴 FAIL"
            with st.expander(f"Iteration {rec['iteration']}: {badge} (ID: {str(rec['interaction_id'])[:14]}...)", expanded=True):
                st.markdown(f"**Argon Reasoning:** {rec['strategy']}")
                m = ver["metrics"]
                st.write(f"- **Target Error:** `{m['target_error_m']*1000:.1f} mm` (Limit: `20.0 mm`)")
                st.write(f"- **Max Joint Velocity Ratio:** `{m['max_velocity_ratio']*100:.1f}%` (Limit: `100%`)")
                st.write(f"- **Max Joint Torque Ratio:** `{m.get('max_torque_ratio', 0)*100:.1f}%` (Limit: `100%`)")
                st.write(f"- **Min Safety Clearance Margin:** `{m['min_clearance_margin_m']*1000:.1f} mm` (Req: `>= 0 mm`)")
                if ver["violations"]:
                    for v in ver["violations"]:
                        st.warning(v)
                st.caption("Cited Datasheet Constraints:")
                for c in rec["cited_clauses"]:
                    st.code(c, language="text")

    st.divider()
    st.subheader("🛡️ Human-in-the-Loop Hardware Deployment Gate")
    if passed:
        approved = st.checkbox("I have inspected the 3D point-cloud clearance envelope, joint torque limits, and approve hardware execution.")
        export_payload = json.dumps({
            "robot": FRANKA_PANDA_SPEC["name"],
            "scenario": scenario_name,
            "uncertainty_margin_m": sigma_m,
            "verified": True,
            "argon_interaction_id": final_rec["interaction_id"],
            "cited_datasheet_clauses": final_rec["cited_clauses"],
            "peak_joint_torques_nm": final_rec["verification"]["metrics"].get("joint_peak_torques_nm", []),
            "joint_trajectory_rad": final_rec["trajectory"].tolist()
        }, indent=2)
        st.download_button(
            "📥 Export Verified Controller Trajectory (JSON)",
            data=export_payload,
            file_name="franka_verified_trajectory.json",
            mime="application/json",
            disabled=not approved
        )
    else:
        st.warning("⚠️ Hardware export is LOCKED because deterministic physical verification did not pass. Adjust target pose or obstacle clearance.")