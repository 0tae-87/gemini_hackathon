FRANKA_PANDA_SPEC = {
    "name": "Franka Emika Panda (7-DOF)",
    "dof": 7,
    "dh_params": [
        [0.0, 0.333, 0.0, 0.0],
        [0.0, 0.0, -1.57079632679, 0.0],
        [0.0, 0.316, 1.57079632679, 0.0],
        [0.0825, 0.0, 1.57079632679, 0.0],
        [-0.0825, 0.384, -1.57079632679, 0.0],
        [0.0, 0.0, 1.57079632679, 0.0],
        [0.088, 0.107, 1.57079632679, 0.0],
    ],
    "q_min": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    "q_max": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    "dq_max": [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
    "tau_max": [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0],
    "link_radius": 0.052
}

SCENARIOS = {
    "S1: Open-Space Baseline (Immediate Pass)": {
        "ko_title": "S1: 개방 공간 기본 이동 (1회차 즉시 통과)",
        "description": "Low floor obstacle with ample clearance. Direct trajectory passes all checks on Iteration 1.",
        "ko_desc": "바닥의 낮은 장애물 환경. 초기 궤적이 모든 물리 검사를 1회차에 즉시 통과합니다.",
        "start_q": [0.0, -0.65, 0.0, -2.25, 0.0, 1.60, 0.785],
        "target_pos": [0.42, 0.25, 0.36],
        "obstacles": [
            {"type": "sphere", "center": [0.30, 0.10, 0.10], "radius": 0.055}
        ],
        "default_uncertainty": 0.010,
        "via_pos": [0.38, 0.12, 0.38]
    },
    "S2: Point-Cloud Clutter Bypass (Self-Healing)": {
        "ko_title": "S2: 3D 포인트클라우드 장애물 회피 (자가 교정)",
        "description": "Direct path collides with the central obstacle + σ uncertainty shell on Iter 1; Argon computes a minimal-detour bypass on Iter 2.",
        "ko_desc": "1회차 직선 경로는 중앙 장애물 및 불확실성(σ) 반경과 충돌하며, 2회차에 최소 우회 경로로 자가 교정합니다.",
        "start_q": [-0.55, -0.50, 0.0, -2.15, 0.0, 1.65, 0.785],
        "target_pos": [0.42, 0.24, 0.22],
        "obstacles": [
            {"type": "sphere", "center": [0.46, 0.02, 0.33], "radius": 0.055},
            {"type": "sphere", "center": [0.28, -0.20, 0.14], "radius": 0.055}
        ],
        "default_uncertainty": 0.020,
        "via_pos": [0.34, 0.02, 0.53]
    },
    "S3: Out-of-Reach Workspace Lockout (Safety Gate)": {
        "ko_title": "S3: 도달 불가 영역 안전 잠금 (하드웨어 보호)",
        "description": "Target at [1.15m, 0.60m, 0.85m] is beyond physical reach. Verifier rejects the trajectory and locks controller export.",
        "ko_desc": "로봇의 물리적 작업 반경 밖(1.15m) 목표 입력 시 억지로 통과시키지 않고 제어기 출력을 안전 잠금합니다.",
        "start_q": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "target_pos": [1.15, 0.60, 0.85],
        "obstacles": [
            {"type": "sphere", "center": [0.45, 0.20, 0.35], "radius": 0.060}
        ],
        "default_uncertainty": 0.015,
        "via_pos": [0.45, 0.20, 0.55]
    }
}