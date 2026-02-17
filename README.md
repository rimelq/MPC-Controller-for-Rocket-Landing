# MPC Controller for Rocket Landing

Model Predictive Control strategies for a 12-state nonlinear rocket model: linear MPC regulators/tracking controllers, cascaded PID–MPC position tracking, offset-free tracking for mass mismatch, robust Tube MPC for safe landing in the z-axis, and full Nonlinear MPC (NMPC) using CasADi.

**Course:** ME-425 Model Predictive Control (EPFL)  
**Project:** Rocket Landing  
**Group AL:** Rim El Qabli, Yasmine Tligui, Ismail Filali  
**Report:** `report.pdf`

---

## Problem Setup

### States and Inputs
The rocket is modeled as a **12-state nonlinear system**:

- Angular velocities: **ω = [ωx, ωy, ωz]**
- Euler angles: **ϕ = [α, β, γ]**
- Linear velocities: **v = [vx, vy, vz]**
- Positions: **p = [x, y, z]**

Control inputs:

- Gimbal servo deflections: **δ1, δ2** (±15°)
- Average throttle: **Pavg** (bounded for safety)
- Differential throttle: **Pdiff** (±20%)

---

## Key Idea: Linearization & Subsystem Decomposition

We trim the rocket at hover (vertical, zero velocities) and linearize around the trim point.
The linearized system exhibits a **block-diagonal structure** that decouples the dynamics into four independent subsystems:

- **roll:** (ωz, γ) driven by **Pdiff**
- **x:** (ωy, β, vx, x) driven by **δ2**
- **y:** (ωx, α, vy, y) driven by **δ1**
- **z:** (vz, z) driven by **Pavg**

This enables independent MPC design per axis (valid near hover; enforced by angle constraints |α|, |β| ≤ 10°).

---

## Controllers Implemented

### 1) Linear MPC (Regulation + Tracking)
Finite-horizon MPC with:
- quadratic stage costs (Q, R)
- terminal cost (P from LQR)
- terminal invariant set constraint for stability and recursive feasibility

**Sampling:** Ts = 0.05 s (20 Hz)  
**Horizon:** N = 35 (H = 1.75 s) for regulation/tracking

Includes:
- **Deliverable 3.1:** velocity regulation to hover
- **Deliverable 3.2:** constant velocity tracking

### 2) Cascaded PID–MPC for Position Tracking
Outer PID loop generates time-varying **velocity references** for inner MPC velocity controllers.

**Extended horizon:** N = 120 (H = 6.0 s) to handle long maneuvers.

### 3) Nonlinear Simulation (Model-Plant Mismatch)
Same controller structure tested on the **full nonlinear model**, with tuning adjustments to handle coupling effects (especially in roll).

### 4) Offset-Free Tracking (z-axis)
Augmented z-model with an unknown disturbance **d** (mass mismatch effect), estimated via a Luenberger observer designed through **LQR duality**.
Uses target selection to compute disturbance-compensated steady-state input.

Scenarios:
- **Constant mass mismatch:** disturbance converges and removes steady-state velocity offset.
- **Time-varying mass:** estimator cannot perfectly track changing disturbance → residual offset and saturation effects.

### 5) Robust Tube MPC for Landing (z-axis safety)
Robust Tube MPC on z-subsystem with bounded input-channel disturbance:
- wk ∈ W = [−15, 5]
- tightened constraints via Pontryagin difference (X ⊖ E, U ⊖ KE)
- robustly enforces **z ≥ 0** near ground

### 6) Merged Landing Controller
Combines:
- **Nominal MPC** for x/y/roll position capture
- **Robust Tube MPC** for z

Landing objective:
(3, 2, 10, 30°) → (1, 0, 3, 0°), with **z ≥ 0**

### 7) Full Nonlinear MPC (CasADi)
NMPC over the full nonlinear model using:
- RK4 discretization
- multiple shooting
- warm-start shifting
- terminal weight from LQR at landing trim

Includes explicit constraints:
- z ≥ 0
- |β| ≤ 80° (avoid Euler singularity)
- actuator bounds

---

## Results Summary

- Linear MPC achieves stable regulation and tracking with constraints satisfied (near-hover validity enforced).
- Cascaded PID–MPC achieves large-range position maneuvers (e.g., (50,50,100) → (0,0,10)) within ~20 s.
- Offset-free tracking eliminates steady-state error for **constant** mass mismatch; time-varying mass yields residual drift and possible saturation.
- Tube MPC robustly maintains **z ≥ 0** under bounded disturbances and enables safe landing behavior.
- NMPC captures coupling and achieves fast landing but can show stronger transient gimbal activity compared to merged linear controllers.
