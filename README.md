# Learning-Based MPC for Autonomous Vehicle Path Tracking

Python implementation of learning-based predictive-error compensation and MPC policy approximation for autonomous vehicle path tracking.

The project was developed for the **Optimization and Machine Learning for Dynamical Systems** course at the University of Florence.

## Project objective

Model Predictive Control can explicitly handle control and state constraints, but its performance depends on the accuracy of the prediction model.

This project investigates two applications of machine learning:

1. estimating the predictive error caused by model mismatch and using it through a feedforward compensator;
2. approximating the nominal MPC policy to reduce online computation time.

The main reference is:

> C. Jiang, H. Tian, J. Hu, J. Zhai, C. Wei, and J. Ni,  
> “Learning based Predictive Error Estimation and Compensator Design for Autonomous Vehicle Path Tracking,” 2020.

## Dynamical models

### Nonlinear plant

The simulated vehicle is represented by a five-state nonlinear dynamic bicycle model:

\[
x = [X,\;Y,\;\psi,\;v_y,\;r]^\top
\]

where:

- \(X,Y\) are the global vehicle coordinates;
- \(\psi\) is the heading angle;
- \(v_y\) is the lateral velocity;
- \(r\) is the yaw rate.

Nonlinear tire saturation, parameter uncertainty, steering bias, and friction variations are considered.

### Linear MPC model

The controller uses a four-state linearized tracking-error model:

\[
\xi = [v_y,\;r,\;e_y,\;e_\psi]^\top
\]

The model is analyzed in terms of:

- open-loop eigenvalues;
- stability;
- reachability;
- observability;
- discrete-time dynamics.

## Control methods

The project implements:

- LQR state-feedback control;
- Luenberger state observer;
- constrained linear MPC;
- curvature-preview MPC;
- ELM predictive-error estimator;
- proportional predictive-error compensator;
- ELM approximation of the MPC policy.

The MPC explicitly constrains:

- steering angle;
- steering rate;
- prediction horizon;
- control effort.

## Predictive-error estimation

The Extreme Learning Machine receives eight inputs:

\[
[v_y,\;r,\;e_y,\;e_\psi,\;\delta,\;\dot\delta,\;\kappa,\;v_x]
\]

and estimates:

- lateral prediction error;
- heading prediction error.

The generated dataset contains approximately 1300 samples collected under multiple paths, velocities, vehicle parameters, tire parameters, and steering biases.

### ELM prediction results

| Target | Mean baseline RMSE | ELM RMSE | Test \(R^2\) |
|---|---:|---:|---:|
| Lateral prediction error | 0.000638 m | 0.000106 m | 0.9722 |
| Heading prediction error | 0.000530 rad | 0.000189 rad | 0.8735 |

## Learning-enhanced MPC results

On the final unseen sinusoidal trajectory:

| Metric | Nominal MPC | MPC + ELM |
|---|---:|---:|
| Lateral-error RMSE | 0.01550 m | 0.01044 m |
| Maximum lateral error | 0.02594 m | 0.01871 m |
| Heading-error RMSE | 0.00565 rad | 0.00559 rad |
| Control effort | 0.001370 | 0.001364 |
| Average computation time | 4.17 ms | 4.45 ms |

The learning-enhanced controller reduced lateral RMSE by **32.64%**.

## Robustness benchmark

The fixed compensator was evaluated without retuning:

| Scenario | RMSE improvement |
|---|---:|
| Smooth sine | 13.90% |
| High-speed sine | 24.31% |
| Double lane change | 43.30% |
| Heavy model mismatch | -19.95% |
| Steering bias | 34.69% |
| **Mean** | **19.25%** |

The heavy-mismatch result shows an important limitation: a learned compensator does not guarantee improvement outside its training distribution.

## MPC policy approximation

A second ELM was trained to approximate the nominal MPC policy.

### Offline approximation

| Metric | Result |
|---|---:|
| Steering RMSE | 0.0153 deg |
| Steering MAE | 0.00364 deg |
| Test \(R^2\) | 0.999925 |
| Constraint violations | 0 |
| ELM inference time | 0.0048 ms/sample |

### Closed-loop comparison

| Metric | Exact MPC | ELM policy |
|---|---:|---:|
| Lateral-error RMSE | 0.01413 m | 0.01161 m |
| Maximum lateral error | 0.02414 m | 0.02012 m |
| Control effort | 0.001403 | 0.001401 |
| Online computation time | 4.20 ms | 0.21 ms |

The ELM policy achieved a measured online speedup of **19.9×** while preserving closed-loop stability and control constraints.

The slightly lower tracking error observed in this scenario is an empirical effect of the smooth policy approximation and is not a theoretical guarantee.

## Project structure

```text
autonomous-vehicle-learning-mpc/
├── data/
│   ├── models/
│   └── predictive_error_dataset.csv
├── experiments/
├── report/
├── results/
├── src/
│   ├── controllers/
│   ├── evaluation/
│   ├── learning/
│   ├── models/
│   └── simulation/
├── tests/
├── README.md
├── reproduce.py
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the tests

```bash
python -m pytest -q
```

Expected result:

```text
32 passed
```

## Reproducing the main results

```bash
python reproduce.py
```

To additionally run all preliminary experiments:

```bash
python reproduce.py --full
```

Generated figures and numerical outputs are stored in `results/`.

## Individual experiments

```bash
python -m experiments.analyze_linear_model
python -m experiments.run_lqr_tracking
python -m experiments.run_observer_tracking
python -m experiments.compare_lqr_mpc
python -m experiments.generate_predictive_error_dataset
python -m experiments.train_elm
python -m experiments.tune_compensator
python -m experiments.compare_nominal_learning_mpc
python -m experiments.run_robustness_benchmark
python -m experiments.train_policy_elm
python -m experiments.compare_mpc_policy_approximation
```

## Main limitations

- The evaluation is simulation-based.
- The nonlinear plant is less detailed than a full CarSim model.
- The ELM is trained offline.
- Improvement is not guaranteed outside the training distribution.
- The policy approximation does not provide the same formal guarantees as the original constrained MPC.
- No full-size vehicle experiments are performed.

## Academic scope

The project covers:

- nonlinear dynamical systems;
- equilibrium and linearization;
- LTI systems;
- stability;
- reachability and observability;
- state-feedback control;
- state estimation;
- convex optimization;
- constrained MPC;
- neural networks;
- data-driven predictive-error estimation;
- ML-based approximation of an MPC policy.