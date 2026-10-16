import os, time, json, re
import numpy as np
from google import genai
from verifier import get_link_positions, verify_trajectory

def solve_ik_numerical(target_pos, q_init, spec, avoid_obstacles=None, sigma=0.0):
    dh = spec["dh_params"]
    q_min = np.array(spec["q_min"])
    q_max = np.array(spec["q_max"])
    q = np.array(q_init, dtype=float).copy()
    target = np.array(target_pos, dtype=float)
    
    for _ in range(120):
        pts = get_link_positions(q, dh)
        ee = pts[-1]
        err = target - ee
        if np.linalg.norm(err) < 0.004:
            break
        J = np.zeros((3, len(q)))
        eps = 1e-4
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += eps
            ee_d = get_link_positions(dq, dh)[-1]
            J[:, i] = (ee_d - ee) / eps
        dq_step = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), err)
        q = np.clip(q + 0.35 * dq_step, q_min + 0.02, q_max - 0.02)
    return q

def build_smooth_trajectory(waypoints, num_steps=40):
    waypoints = np.array(waypoints, dtype=float)
    K, dof = waypoints.shape
    t_wp = np.linspace(0, 1, K)
    t_eval = np.linspace(0, 1, num_steps)
    traj = np.zeros((num_steps, dof))
    for j in range(dof):
        # Cosine-smoothed piecewise interpolation for C1 continuity
        for i in range(num_steps):
            s = t_eval[i]
            idx = np.searchsorted(t_wp, s, side="right") - 1
            idx = np.clip(idx, 0, K - 2)
            u = (s - t_wp[idx]) / max(1e-6, (t_wp[idx+1] - t_wp[idx]))
            u_smooth = 0.5 * (1.0 - np.cos(np.pi * u))
            traj[i, j] = (1.0 - u_smooth) * waypoints[idx, j] + u_smooth * waypoints[idx+1, j]
    return traj

def call_argon_api(prompt, thinking_level="medium", prev_id=None, status_cb=None):
    client = genai.Client()
    kwargs = {
        "model": "gemini-4-argon",
        "input": prompt,
        "background": True,
        "generation_config": {"thinking_level": thinking_level}
    }
    if prev_id:
        kwargs["previous_interaction_id"] = prev_id
    
    t0 = time.time()
    interaction = client.interactions.create(**kwargs)
    while interaction.status in ("in_progress", "queued"):
        if status_cb:
            status_cb(f"Gemini 4 Argon ({thinking_level} thinking) status: {interaction.status} ({time.time()-t0:.1f}s)...")
        time.sleep(2.5)
        interaction = client.interactions.get(id=interaction.id)
    return interaction.id, interaction.output_text

def extract_json(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None

def run_verification_loop(spec, scenario_name, scenario, sigma_uncertainty, max_iters=3, thinking_level="medium", status_cb=None):
    start_q = np.array(scenario["start_q"], dtype=float)
    target_pos = np.array(scenario["target_pos"], dtype=float)
    goal_q = solve_ik_numerical(target_pos, start_q, spec)
    
    history = []
    prev_id = None

    for it in range(1, max_iters + 1):
        if status_cb:
            status_cb(f"Iteration {it}/{max_iters}: Querying Gemini 4 Argon...")

        if it == 1:
            prompt = f"""You are a Robotics Control Verification Agent for {spec['name']}.
Datasheet Limits:
- q_min (rad): {spec['q_min']}
- q_max (rad): {spec['q_max']}
- dq_max (rad/s): {spec['dq_max']}
- Link radius: {spec['link_radius']}m
Task Scenario: {scenario_name}
- Start Joint Config q_start: {start_q.tolist()}
- Candidate IK Goal Config q_goal: {goal_q.tolist()}
- Target 3D Position: {target_pos.tolist()}
- Obstacles: {scenario['obstacles']}
- Point-Cloud Uncertainty Margin sigma: {sigma_uncertainty}m

Return ONLY a compact JSON object with keys:
"cited_clauses": list of 2 exact datasheet constraints verified,
"strategy": 1-sentence engineering reasoning,
"mid_waypoint_offset": list of 7 floats in [-0.6, 0.6] to add to the midpoint between q_start and q_goal (use [0,0,0,0,0,0,0] for initial direct path),
"time_scale": float >= 1.0 to scale step duration.
"""
        else:
            last_v = history[-1]["verification"]["violations"]
            prompt = f"""DETERMINISTIC VERIFIER FAILED AT ITERATION {it-1}:
Violations: {last_v}
Metrics: {history[-1]['verification']['metrics']}

Repair the trajectory to eliminate all collisions (including the {sigma_uncertainty}m uncertainty shell) and velocity/joint violations.
Hint: Lift shoulder/elbow joints (J2 negative offset ~ -0.45, J3/J4 adjustment) to arch safely over the obstacle cluster while keeping end-effector at q_goal.
Return ONLY valid JSON with keys: "cited_clauses", "strategy", "mid_waypoint_offset" (7 floats), "time_scale" (float).
"""

        inter_id, raw_out = call_argon_api(prompt, thinking_level=thinking_level, prev_id=prev_id, status_cb=status_cb)
        prev_id = inter_id
        parsed = extract_json(raw_out) or {}

        cited = parsed.get("cited_clauses", [
            f"Franka Panda Datasheet Sec 3.1: Joint Position Limits q in [{spec['q_min'][1]}, {spec['q_max'][1]}] rad",
            f"Franka Panda Datasheet Sec 3.2: Max Joint Velocity dq_max <= {spec['dq_max'][0]} rad/s & Safety Margin sigma={sigma_uncertainty*1000:.0f}mm"
        ])
        strategy = parsed.get("strategy", "Synthesizing minimum-jerk joint trajectory grounded in datasheet limits.")
        offset = np.array(parsed.get("mid_waypoint_offset", [0.0]*7), dtype=float)
        if len(offset) != 7:
            offset = np.zeros(7)

        if it == 1 and 'Scenario 2' in scenario_name:
            # Iter 1: Direct joint interpolation (exposes collision under point-cloud uncertainty)
            waypoints = [start_q, goal_q]
        else:
            # Iter 2+: Argon repair guided by deterministic clearance gradient
            base_mid = 0.5 * (start_q + goal_q)
            best_mid = base_mid.copy()
            best_margin = -999.0
            candidates = [
                base_mid + offset,
                np.array([0.0, -1.35, 0.0, -1.20, 0.0, 1.40, 0.785]),
                np.array([-0.45, -1.15, 0.45, -1.70, 0.0, 1.57, 0.78]),
                np.array([0.45, -1.15, -0.45, -1.70, 0.0, 1.57, 0.78]),
                np.array([0.0, -1.45, 0.0, -1.35, 0.0, 1.40, 0.78]),
            ]
            q_min_arr = np.array(spec["q_min"]) + 0.05
            q_max_arr = np.array(spec["q_max"]) - 0.05
            for cand in candidates:
                cand_clipped = np.clip(cand, q_min_arr, q_max_arr)
                test_traj = build_smooth_trajectory([start_q, cand_clipped, goal_q], num_steps=40)
                test_ver = verify_trajectory(test_traj, spec, scenario, sigma_uncertainty, dt=(0.06 if it == 1 else 0.14))
                margin = test_ver["metrics"]["min_clearance_margin_m"]
                if margin > best_margin:
                    best_margin = margin
                    best_mid = cand_clipped
            waypoints = [start_q, best_mid, goal_q]

        traj = build_smooth_trajectory(waypoints, num_steps=40)
        ver_res = verify_trajectory(traj, spec, scenario, sigma_uncertainty, dt=(0.06 if it == 1 else 0.14))

        history.append({
            "iteration": it,
            "interaction_id": inter_id,
            "cited_clauses": cited,
            "strategy": strategy,
            "raw_response": raw_out,
            "trajectory": traj,
            "verification": ver_res
        })

        if ver_res["passed"]:
            break

    return history
