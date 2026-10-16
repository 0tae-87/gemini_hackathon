import numpy as np

def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st, 0, a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa, ca, d*ca],
        [0, 0, 0, 1]
    ], dtype=float)

def get_link_positions(q, dh_params):
    T = np.eye(4)
    positions = [T[:3, 3].copy()]
    for i, (a, d, alpha, th_off) in enumerate(dh_params):
        Ti = dh_transform(a, d, alpha, q[i] + th_off)
        T = T @ Ti
        positions.append(T[:3, 3].copy())
    return np.array(positions)

def dist_segment_to_point(p1, p2, pt):
    v = p2 - p1
    w = pt - p1
    c1 = np.dot(w, v)
    if c1 <= 0: return float(np.linalg.norm(pt - p1))
    c2 = np.dot(v, v)
    if c2 <= c1: return float(np.linalg.norm(pt - p2))
    b = c1 / c2
    pb = p1 + b * v
    return float(np.linalg.norm(pt - pb))

def verify_trajectory(trajectory, spec, scenario, sigma_uncertainty, dt=0.05):
    N, dof = trajectory.shape
    q_min = np.array(spec["q_min"])
    q_max = np.array(spec["q_max"])
    dq_max = np.array(spec["dq_max"])
    tau_max = np.array(spec.get("tau_max", [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]), dtype=float)
    link_r = spec["link_radius"]
    target_pos = np.array(scenario["target_pos"])
    obstacles = scenario.get("obstacles", [])

    results = {
        "passed": True,
        "violations": [],
        "metrics": {},
        "link_trajectories": []
    }

    all_link_pts = []
    for q in trajectory:
        pts = get_link_positions(q, spec["dh_params"])
        all_link_pts.append(pts)
    all_link_pts = np.array(all_link_pts)
    results["link_trajectories"] = all_link_pts

    # 1. Target Pose Error
    ee_final = all_link_pts[-1, -1]
    target_error = float(np.linalg.norm(ee_final - target_pos))
    results["metrics"]["target_error_m"] = target_error
    if target_error > 0.02:
        results["passed"] = False
        results["violations"].append(f"Target error {target_error*1000:.1f}mm > 20mm threshold")

    # 2. Joint Angle Limits
    q_lower_diff = q_min - trajectory
    q_upper_diff = trajectory - q_max
    if np.any(q_lower_diff > 1e-4) or np.any(q_upper_diff > 1e-4):
        results["passed"] = False
        bad_steps, bad_joints = np.where((q_lower_diff > 1e-4) | (q_upper_diff > 1e-4))
        results["violations"].append(f"Joint limit violated at step {bad_steps[0]}, joint J{bad_joints[0]+1}")

    # 3. Joint Velocity Limits
    velocities = np.abs(np.diff(trajectory, axis=0) / dt)
    v_diff = velocities - dq_max
    max_v_ratio = float(np.max(velocities / dq_max)) if len(velocities) > 0 else 0.0
    results["metrics"]["max_velocity_ratio"] = max_v_ratio
    if np.any(v_diff > 1e-3):
        results["passed"] = False
        bad_steps, bad_joints = np.where(v_diff > 1e-3)
        results["violations"].append(f"Velocity limit exceeded at step {bad_steps[0]}, J{bad_joints[0]+1}")

    # 4. Joint Torque Limits (Gravity + Inertial Acceleration Dynamics)
    acc = np.diff(trajectory, n=2, axis=0) / (dt ** 2) if len(trajectory) > 2 else np.zeros((1, dof))
    eff_inertia = np.array([2.4, 2.8, 1.7, 1.5, 0.35, 0.28, 0.12])
    grav_moment = np.array([3.5, 29.5, 7.5, 19.0, 2.2, 3.1, 0.7])
    torques = np.abs(acc) * eff_inertia + grav_moment * np.abs(np.cos(trajectory[:len(acc)]))
    peak_torques = np.max(torques, axis=0).tolist() if len(torques) > 0 else [0.0] * dof
    max_tau_ratio = float(np.max(torques / tau_max)) if len(torques) > 0 else 0.0
    results["metrics"]["max_torque_ratio"] = max_tau_ratio
    results["metrics"]["joint_peak_torques_nm"] = peak_torques
    results["metrics"]["tau_max_nm"] = tau_max.tolist()
    if max_tau_ratio > 1.0:
        results["passed"] = False
        bad_j = int(np.argmax(np.max(torques / tau_max, axis=0)))
        results["violations"].append(f"Joint torque limit exceeded on J{bad_j+1} ({peak_torques[bad_j]:.1f} Nm > {tau_max[bad_j]:.1f} Nm)")

    # 5. 3D Obstacle + Uncertainty Clearance
    min_obs_clearance = 999.0
    for t_idx, pts in enumerate(all_link_pts):
        for obs in obstacles:
            center = np.array(obs["center"])
            obs_r = obs["radius"]
            required_clearance = obs_r + link_r + sigma_uncertainty
            for j in range(len(pts) - 1):
                dist = dist_segment_to_point(pts[j], pts[j+1], center)
                clearance_margin = dist - required_clearance
                if clearance_margin < min_obs_clearance:
                    min_obs_clearance = clearance_margin
                if clearance_margin < 0:
                    results["passed"] = False
                    results["violations"].append(f"Collision violation at step {t_idx}, Link {j}-{j+1}")
                    break
            if not results["passed"] and len(results["violations"]) > 3: break
        if not results["passed"] and len(results["violations"]) > 3: break

    results["metrics"]["min_clearance_margin_m"] = float(min_obs_clearance)
    return results