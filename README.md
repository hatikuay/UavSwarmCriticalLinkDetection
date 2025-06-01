# UAV Swarm Connectivity Simulator

**Prepared by Dr. Ercan Erkalkan** (May 2025)  
Email: ercan.erkalkan@marmara.edu.tr

---

This repository contains code and documentation for simulating adaptive critical‐link detection and connectivity preservation in unmanned aerial vehicle (UAV) swarms. It includes both headless (console) parameter sweeps and a real‐time PyQt5 GUI.

---

## Table of Contents

1. [Overview](#overview)  
2. [Installation](#installation)  
3. [Usage](#usage)  
   - [1) Parameter Sweep (Headless)](#1-parameter-sweep-headless)  
   - [2) Console Simulation](#2-console-simulation)  
   - [3) PyQt5 GUI Simulation](#3-pyqt5-gui-simulation)  
4. [File Descriptions](#file-descriptions)  
5. [Example Plots](#example-plots)  
6. [License](#license)  
7. [Contributing](#contributing)  

---

## Overview

This project studies how a UAV swarm (multiple quadrotors) maintains algebraic connectivity (λ₂) under varying adaptive‐threshold control parameters (α, β, γ). Specifically:

- **Critical‐link detection**: Identify “risky” inter-UAV links whose channel quality probability falls below a minimum threshold (pₘᵢₙ).  
- **Adaptive thresholds**: Two thresholds (θ₋, θ₊) that shrink or expand the network graph based on estimated channel state.  
- **Algebraic connectivity (λ₂)**: The second smallest eigenvalue of the Laplacian matrix of the UAV‐to‐UAV graph; used to quantify connectivity.  
- **Parameter sweep**: Evaluate impact of α (gain for lower threshold), β (gain for partition prevention), and γ (trade-off reward parameter) on average λ₂, partition rate, and “reward” = (λ₂ – 0.1·partition_rate).  
- **Headless simulation**: Run many short episodes to compute statistics, logged in a CSV file.  
- **PyQt5 GUI**: Real-time visualization of UAV positions, risky links, connectivity, and live λ₂ values.

---

## Installation

1. **Python 3.8+ is required**. Tested on Python 3.8 and 3.9.

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS / Linux
   venv\Scripts\activate.bat   # Windows
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: If you do not intend to run the PyQt5 GUI, you may omit `pyqt5` from installation. However, for real-time visualization, `pyqt5>=5.12` must be present.

---

## Usage

### 1) Parameter Sweep (Headless)

Performs a sweep over combinations of (α, β, γ), runs multiple episodes, and records summary statistics (`mean_lambda2`, `partition_rate`) into a CSV.

```bash
python parametre.py
```

- Runs all `(α, β, γ)` combinations for N episodes (default = 100) and T time steps each (default = 100).  
- Computes:
  - **mean_lambda2**: The average λ₂ (algebraic connectivity) over all episodes and time steps.  
  - **partition_rate**: Fraction of episodes/time steps where the network split (λ₂ <  ε).  
- Outputs a file named `parameter_sweep_results.csv` in the project root.  
- Also prints progress and final CSV path to the console.

---

### 2) Console Simulation

Runs a single example simulation (without GUI), prints a pandas DataFrame summarizing each time step.

```bash
python simulation.py
```

- Default: 3 UAVs, 50 time steps.  
- Captures at each step:
  - UAV positions (x, y), velocities, commanded accelerations.  
  - Estimated channel qualities (hₕ), remaining energy, process and measurement noise.  
  - Current thresholds θ₋, θ₊, risky link count, and λ₂.  
- Prints the DataFrame head and tail; you can modify `simulation.py` to save `states.csv` if desired.

---

### 3) PyQt5 GUI Simulation

Provides an interactive window that visualizes UAVs, thresholds, risky links, and λ₂ on the fly.

```bash
python simulation_pyQT.py
```

#### GUI Controls

- **UAV Count** (SpinBox): Select the number of UAVs (default = 5). Press “Reset” to reinitialize.  
- **p_min Slider**: Minimum channel-quality probability threshold (default = 0.9).  
- **Pause / Resume**: Temporarily freeze or continue the simulation loop.  
- **Generate Report**: Immediately write a `uav_report.txt` summarizing current λ₂, risk count, and time series.

#### Real-time Visualization

- **Blue dots**: Current UAV positions in the 2D plane.  
- **Red star**: Ground station at (0, 0).  
- **Gray lines**: Safe UAV-to-UAV links (estimated SNR ≥ θ₋).  
- **Red lines**: Risky UAV-to-UAV links (estimated SNR between θ₋ and θ₊).  
- **Window title**: Displays current time step, latest λ₂, and number of risky links.

#### Automatic Reporting

- If λ₂ drops below a critical value (default ε = 0.05), the GUI will automatically pause and write a `uav_report.txt` file.  
- The `uav_report.txt` content looks like:

  ```
  UAV Connectivity Report
  Prepared by Dr. Ercan Erkalkan
  ----------------------------
  Final λ₂: 0.0217
  Total Time Steps: 317
  Number of Connected Components: 4
  Risky Links at Final Step: 6

  λ₂ Time Series:
  1.000, 0.972, 0.945, 0.912, …, 0.051, 0.033, 0.0217
  ```

---

## File Descriptions

- **`parametre.py`**  
  - Defines `run_episode(...)` and `simulate_parameter_sweep(...)` functions.  
  - Sweeps α∈{0.10,0.15,0.20,…,0.50}, β∈{0.10,0.15,…,0.40}, γ∈{0.05,0.075,…,0.20} by default.  
  - Saves results to `parameter_sweep_results.csv`.

- **`simulation.py`**  
  - Runs a single, console‐only simulation.  
  - Uses classes from `uav_swarm.py`.  
  - Prints a pandas DataFrame summarizing each step.

- **`simulation_pyQT.py`**  
  - PyQt5 GUI application.  
  - Real‐time plotting via `matplotlib.backends.backend_qt5agg`.  
  - Buttons/Sliders to control simulation parameters and generate reports.

- **`uav_swarm.py`**  
  - Contains two main classes:  
    - **`UAVState`**: State variables (position p, velocity v, commanded acceleration a_ctrl, jerk j, channel quality h_ch, remaining energy E, process noise ω, measurement noise ε, wind disturbance w_wind).  
    - **`UAVSwarm`**:  
      - Maintains a list of `UAVState` instances.  
      - Implements Extended Kalman Filter (EKF) for position estimation.  
      - Nakagami‐m fading model for channel quality estimation.  
      - Adaptive threshold logic (θ₋ = α·h_low, θ₊ = β·h_high).  
      - Algebraic connectivity (λ₂) computation using the Laplacian matrix.  
      - Risky‐link detection and partition counting.

- **`parameter_sweep_results.csv`**  
  - Generated by `parametre.py`.  
  - Columns:  
    - `alpha`, `beta`, `gamma`  
    - `mean_lambda2` — average λ₂ over all episodes and time steps  
    - `partition_rate` — fraction of time steps where λ₂ < ε (i.e., swarm partitioned)

- **`uav_report.txt`**  
  - Generated by GUI or automatically when swarm becomes disconnected.  
  - Summarizes final λ₂, time step count, number of components, risky link count, and the λ₂ timeline.

---

## Example Plots

Once you have run `parametre.py`, you can generate the following plots in Python to visualize results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("parameter_sweep_results.csv")

# 1) Mean λ₂ vs α
plt.figure(figsize=(10, 5))
for alpha in sorted(df["alpha"].unique()):
    subset = df[df["alpha"] == alpha]
    plt.scatter([alpha]*len(subset), subset["mean_lambda2"],
                color="orange", marker="x", alpha=0.7, s=25)
plt.title("Mean Algebraic Connectivity (λ₂) vs Alpha")
plt.xlabel("Alpha (α)")
plt.ylabel("Mean λ₂")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("mean_lambda2_vs_alpha.png", dpi=300)
plt.close()

# 2) Partition Rate vs β
plt.figure(figsize=(10, 5))
for beta in sorted(df["beta"].unique()):
    subset = df[df["beta"] == beta]
    plt.scatter([beta]*len(subset), subset["partition_rate"],
                color="teal", marker="o", alpha=0.7, s=25)
plt.title("Partition Rate vs Beta")
plt.xlabel("Beta (β)")
plt.ylabel("Partition Rate")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("partition_rate_vs_beta.png", dpi=300)
plt.close()

# 3) Reward vs γ
plt.figure(figsize=(10, 5))
for gamma in sorted(df["gamma"].unique()):
    subset = df[df["gamma"] == gamma]
    reward = subset["mean_lambda2"] - 0.1 * subset["partition_rate"]
    plt.scatter([gamma]*len(subset), reward,
                color="purple", marker="d", alpha=0.7, s=25)
plt.title("Reward vs Gamma")
plt.xlabel("Gamma (γ)")
plt.ylabel("Reward = (mean λ₂ - 0.1·partition_rate)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig("reward_vs_gamma.png", dpi=300)
plt.close()
```

These scripts will produce the three PNG files:
- `mean_lambda2_vs_alpha.png`  
- `partition_rate_vs_beta.png`  
- `reward_vs_gamma.png`

From these, you can analyze:
- How **α** influences average connectivity.
- Which **β** yields the lowest partition rate.
- Which **γ** maximizes the combined “reward.”

---

## License

```
MIT License

Copyright (c) 2025 Dr. Ercan Erkalkan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in  
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN  
THE SOFTWARE.
```
