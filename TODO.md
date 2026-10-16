# 📋 Argon-SafeKinematics Development & Enhancement TODO

## 1. 3D Visualization & Physics Engine Upgrade
- [ ] **Three.js / WebGPU Real-Time Mesh Viewer:** Upgrade from Plotly 3D volumetric primitives to full URDF/DAE CAD mesh rendering (`Franka Emika Panda` link STL/DAE meshes) via custom Three.js/WebGL component.
- [ ] **Dense 3D Point-Cloud Completion Overlay:** Integrate real depth-camera / LiDAR point-cloud completion uncertainty heatmaps (per-point covariance ellipsoids $\Sigma_i$) around target objects and obstacles.
- [ ] **Interactive Playback Controls:** Add timeline play/pause/scrubbing animation directly inside the 3D viewport with real-time collision distance vectors drawn between the nearest robot link and obstacle surface.

## 2. Web UI / Engineering Workbench Polish
- [ ] **Industrial Dark-Mode Telemetry Layout:** Add multi-tab engineering views (`3D Digital Twin`, `Joint Velocity/Acceleration Phase Plots`, `Datasheet Clause Diff`, `Raw Argon Interaction JSON`).
- [ ] **Live Interactive Drag-and-Drop Waypoints:** Allow engineers to click or drag 3D target poses and obstacle spheres directly inside the viewport in addition to sidebar coordinate sliders.
- [ ] **Architecture Pipeline Walkthrough Modal:** Add an interactive visual architecture diagram showing the closed-loop flow between `gemini-4-argon` (`thinking_level="high"`), `previous_interaction_id`, and `verifier.py`.

## 3. Scenario 3 + Alpha (Expanded Benchmark Suite)
- [ ] **Scenario 3 Solvable Mode (High-Payload Dynamic Torque & Velocity Bottleneck):** Add a solvable mode where Iteration 1 violates the 87 Nm / 12 Nm joint torque or velocity limit under a 2.5 kg payload, and Iteration 2 scales trajectory timing ($dt$) and re-routes closer to the base to reduce gravitational moment arms.
- [ ] **Scenario 4 (+Alpha): Multi-Obstacle Narrow Shelf Reach & Grasp Approach:** Verify both pre-grasp approach vector alignment and wrist singularity avoidance ($J_4, J_6$).
- [ ] **Scenario 5 (+Alpha): Multi-Robot Platform Preset (6-DOF Universal Robots UR5e):** Allow one-click switching between 7-DOF Franka Emika Panda and 6-DOF UR5e datasheet specifications.