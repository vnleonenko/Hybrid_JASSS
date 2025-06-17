# Hybrid_JASSS

Code for Approximate Bayesian Computation (ABC) and history matching calibration of epidemic models across three frameworks:

- Agent-Based Model (ABM)
- SEIR compartmental model
- Network-based SEIR model

Each framework includes synthetic data generation, parameter inference methods and visualization tools.

---

## Repository Contents

| File        | Description                                                  |
|-------------------------|--------------------------------------------------------------|
| `abm_to_abm.ipynb`       | Agent-Based Model calibration with ABC and history matching. |
| `seir_to_seir.py`        | SEIR compartmental model calibration using Gaussian Process emulator and ABC methods. |
| `networks_to_seir.ipynb` | Network-based SEIR model calibration focusing on transmission rate and initial infection fraction. |

---

## Models description

### 1. Agent-Based Model (ABM)

- **Name:** `abm_to_abm.ipynb`
- **Purpose:** calibrate ABM parameters: susceptibility (`alpha`) and transmissibility (`lambda`).
- **Features:**
  - Synthetic epidemic data generation
  - History matching to identify plausible parameter regions
  - ABC methods: rejection, simulated annealing, and Sequential Monte Carlo (SMC)
  - Visualization of parameter posteriors and epidemic trajectories

---

### 2. SEIR compartmental model

- **Name:** `seir_to_seir.py`
- **Purpose:** calibrate the transmission parameter (`beta`) of the SEIR model.
- **Features:**
  - Load and preprocess SEIR simulation data
  - Gaussian Process emulator for fast surrogate modeling
  - History matching to narrow parameter space
  - ABC methods: rejection sampling, simulated annealing, and SMC
  - Comparison of runtime, acceptance rates, and posterior distributions

---

### 3. Network-based SEIR model

- **Name:** `networks_to_seir.ipynb`
- **Purpose:** calibrate network SEIR parameters: transmission rate (`tau`) and initial infection fraction (`rho`), with fixed incubation (`alpha`) and recovery (`gamma`) rates.
- **Features:**
  - Generate synthetic epidemic data on various network types (Barabasi–Albert, Erdos–Renyi, Watts-Strogatz, etc.)
  - History matching and ABC (rejection, annealing, SMC) for parameter inference
  - Visualization of parameter posteriors and epidemic curves

---

## Parameter inference methods

All models support:

- **History Matching:** filters out implausible parameter regions based on distance metrics.
- **ABC Rejection Sampling:** accepts parameter samples producing simulations close to observed data.
- **ABC Simulated Annealing:** sequentially tightens acceptance threshold using a cooling schedule.
- **ABC Sequential Monte Carlo (SMC):** efficient sampling with adaptive weights and perturbations.

---
