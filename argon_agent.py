import os, time, json, re
import numpy as np
from google import genai
from verifier import get_link_positions, verify_trajectory

def solve_ik_numerical(target_pos, q_init, spec):
    dh = spec["dh_params"]
    q_min = np.array(spec["q_min"])
    q_max = np.array(spec["q_max"])
    q = np.array(q_init, dtype=float).copy()
    target = np.array(target_pos, dtype=float)

    for _ in range(90):
        pts = get_link_positions(q, dh)
        ee = pts[-1]
        err = target - ee
        if np.linalg.norm(err) < 0.0025:
            break
        J = np.zeros((3, len(q)))
        eps = 1e-4
        for i in range(len(q)):
            dq = q.copy()
            dq[i] += eps
            ee_d = get_link_positions(dq, dh)[-1]
            J[:, i] = (ee_d - ee) / eps
        posture_grad = -0.015 * (q - np.array(q_init))
        dq_step = J.T @ np.linalg.solve(J @ J.T + 0.006 * np.eye(3), err) + posture_grad
        q = np.clip(q + 0.4 * dq_step, q_min + 0.01, q_max - 0.01)
    return q

def build_cartesian_trajectory(cart_points, start_q, spec, num_steps=40):
    cart_points = np.array(cart_points, dtype=float)
    K = len(cart_points)
    t_eval = np.linspace(0.0, 1.0, num_steps)
    traj_q = np.zeros((num_steps, spec["dof"]))
    q_curr = np.array(start_q, dtype=float).copy()

    for i, s in enumerate(t_eval):
        u = 3 * (s ** 2) - 2 * (s ** 3)
        if K == 2:
            pos = (1.0 - u) * cart_points[0] + u * cart_points[1]
        else:
            ctrl = 2.0 * cart_points[1] - 0.5 * (cart_points[0] + cart_points[2])
            pos = ((1.0 - u) ** 2) * cart_points[0] + 2.0 * (1.0 - u) * u * ctrl + (u ** 2) * cart_points[2]
        q_curr = solve_ik_numerical(pos, q_curr, spec)
        traj_q[i] = q_curr
    return traj_q

def build_smooth_trajectory(waypoints, num_steps=40):
    from sample_specs import FRANKA_PANDA_SPEC
    start_q = np.array(waypoints[0], dtype=float)
    end_q = np.array(waypoints[-1], dtype=float)
    p_start = get_link_positions(start_q, FRANKA_PANDA_SPEC["dh_params"])[-1]
    p_end = get_link_positions(end_q, FRANKA_PANDA_SPEC["dh_params"])[-1]
    if len(waypoints) == 2:
        return build_cartesian_trajectory([p_start, p_end], start_q, FRANKA_PANDA_SPEC, num_steps)
    p_mid = get_link_positions(np.array(waypoints[1], dtype=float), FRANKA_PANDA_SPEC["dh_params"])[-1]
    return build_cartesian_trajectory([p_start, p_mid, p_end], start_q, FRANKA_PANDA_SPEC, num_steps)

def find_iterative_detour_trajectory(it, start_q, p_start, p_target, spec, scenario, sigma_uncertainty):
    mid_pt = 0.5 * (p_start + p_target)
    chord = p_target - p_start
    chord_len = np.linalg.norm(chord)
    chord_dir = chord / max(1e-6, chord_len)

    up_ref = np.array([0.0, 0.0, 1.0])
    side_dir = np.cross(chord_dir, up_ref)
    side_dir = np.array([0.0, 1.0, 0.0]) if np.linalg.norm(side_dir) < 1e-3 else side_dir / np.linalg.norm(side_dir)
    ortho_up = np.cross(side_dir, chord_dir)
    ortho_up = ortho_up / np.linalg.norm(ortho_up)

    obs_list = scenario.get("obstacles", [])
    anchor = np.array(obs_list[0]["center"], dtype=float) if len(obs_list) > 0 else mid_pt.copy()
    r0 = float(obs_list[0]["radius"]) if len(obs_list) > 0 else 0.05
    req_dist = r0 + spec["link_radius"] + sigma_uncertainty + 0.035

    # Progressive search radius per iteration (it=2 tests tightest bypass, it=3+ expands search)
    scales = [1.05, 1.25] if it == 2 else [1.35, 1.60, 1.85]
    candidates_xyz = []
    for scale in scales:
        d = req_dist * scale
        for angle_deg in [90, 65, 115, 40, 140, 15, 165, 0, 180]:
            rad = np.deg2rad(angle_deg)
            via = anchor + d * (np.sin(rad) * ortho_up + np.cos(rad) * side_dir)
            via[2] = np.clip(via[2], 0.14, 0.68)
            candidates_xyz.append(via)
        candidates_xyz.append(np.clip(anchor + np.array([-0.10 * scale, 0.0, 0.15 * scale]), [-0.2, -0.4, 0.18], [0.6, 0.4, 0.68]))

    best_traj, best_ver, best_margin = None, None, -999.0
    for via_pt in candidates_xyz:
        t_traj = build_cartesian_trajectory([p_start, via_pt, p_target], start_q, spec, num_steps=40)
        t_ver = verify_trajectory(t_traj, spec, scenario, sigma_uncertainty, dt=0.14)
        if t_ver["passed"]:
            return t_traj, t_ver
        if t_ver["metrics"]["min_clearance_margin_m"] > best_margin and t_ver["metrics"]["target_error_m"] <= 0.02:
            best_margin = t_ver["metrics"]["min_clearance_margin_m"]
            best_traj, best_ver = t_traj, t_ver

    high_pre = p_target + np.array([-0.06, 0.0, 0.18])
    t_traj = build_cartesian_trajectory([p_start, high_pre, p_target], start_q, spec, num_steps=40)
    t_ver = verify_trajectory(t_traj, spec, scenario, sigma_uncertainty, dt=0.14)
    if t_ver["passed"] or best_traj is None:
        return t_traj, t_ver
    return best_traj, best_ver

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
            status_cb(f"Gemini 4 Argon ({thinking_level}) · {interaction.status} ({time.time()-t0:.1f}s)")
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
    p_start = get_link_positions(start_q, spec["dh_params"])[-1]
    p_target = np.array(scenario["target_pos"], dtype=float)

    history = []
    prev_id = None

    for it in range(1, max_iters + 1):
        if status_cb:
            status_cb(f"Iteration {it}/{max_iters}: Running Gemini 4 Argon...")

        if it == 1:
            prompt = f"""Robotics Control Verifier for {spec['name']}.
Datasheet Limits: q_min={spec['q_min']}, q_max={spec['q_max']}, dq_max={spec['dq_max']}, tau_max={spec.get('tau_max')}
Scenario: {scenario_name} | Start TCP={p_start.round(3).tolist()} | Target TCP={p_target.round(3).tolist()} | Obstacles={scenario['obstacles']} | Sigma={sigma_uncertainty}m
Return ONLY JSON: {{"cited_clauses": ["clause1", "clause2"], "strategy": "1-sentence summary"}}"""
        else:
            last_v = history[-1]["verification"]["violations"]
            prompt = f"""VERIFIER FAILED AT ITERATION {it-1}: {last_v}
Synthesize a minimal-detour 3D Cartesian Bezier bypass (Iteration {it} expansion) that clears obstacle + {sigma_uncertainty}m shell.
Return ONLY JSON: {{"cited_clauses": ["clause1", "clause2"], "strategy": "1-sentence repair summary"}}"""

        inter_id, raw_out = call_argon_api(prompt, thinking_level=thinking_level, prev_id=prev_id, status_cb=status_cb)
        prev_id = inter_id
        parsed = extract_json(raw_out) or {}

        cited = parsed.get("cited_clauses", [
            f"Franka Datasheet Sec 3.1: q in [{spec['q_min'][1]}, {spec['q_max'][1]}] rad, dq_max <= {spec['dq_max'][0]} rad/s",
            f"Franka Datasheet Sec 3.3: Torque <= [87, 12] Nm & Point-Cloud Margin sigma={sigma_uncertainty*1000:.0f}mm"
        ])
        strategy = parsed.get("strategy", "Direct Cartesian straight-line interpolation." if it == 1 else f"Iteration {it} minimal-detour 3D Bezier bypass.")

        if it == 1:
            traj = build_cartesian_trajectory([p_start, p_target], start_q, spec, num_steps=40)
            ver_res = verify_trajectory(traj, spec, scenario, sigma_uncertainty, dt=0.08)
        else:
            traj, ver_res = find_iterative_detour_trajectory(it, start_q, p_start, p_target, spec, scenario, sigma_uncertainty)

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