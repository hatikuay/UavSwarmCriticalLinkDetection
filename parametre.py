import numpy as np
import pandas as pd
from uav_swarm import UAVSwarm


def run_episode(
    swarm, steps, power_fn, wind_inputs, cthr_base, gains, v_max, a_max, j_max
):
    parts = 0
    lambdas = []
    for _ in range(steps):
        power_usage = power_fn(swarm.uavs)
        swarm.step(power_usage, cthr_base, gains, v_max, a_max, j_max, wind_inputs)
        l2 = swarm.last_lambda2
        lambdas.append(l2)
        if l2 < 1e-3:
            parts += 1
    return np.mean(lambdas), parts


def simulate_parameter_sweep(
    n_uav=5,
    dt=0.1,
    steps=100,
    episodes=100,
    alpha_vals=None,
    beta_vals=None,
    gamma_vals=None,
):
    # Default parameter grids
    if alpha_vals is None:
        alpha_vals = np.linspace(0.1, 0.5, 9)
    if beta_vals is None:
        beta_vals = np.linspace(0.1, 0.4, 7)
    if gamma_vals is None:
        gamma_vals = np.linspace(0.05, 0.2, 7)

    # Constants
    v_max, a_max, j_max = 23.0, 60.0, 100.0
    cthr_base = 100.0
    wind_inputs = [np.zeros(3)] * n_uav

    # Covariances
    Q_proc = np.diag([0.01] * 9)
    Q_kf = np.eye(6) * 0.05
    R_kf = np.eye(3) * 0.01

    records = []
    for alpha in alpha_vals:
        for beta in beta_vals:
            for gamma in gamma_vals:
                gains = {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": 0.1,
                    "zeta": 0.05,
                }
                lambda_means = []
                partition_counts = []
                for _ in range(episodes):
                    swarm = UAVSwarm(n_uav, dt, Q_proc, Q_kf, R_kf)
                    for uav in swarm.uavs:
                        uav.p = np.random.randn(3)
                        uav.v = np.random.randn(3)
                        uav.a_ctrl = np.random.randn(3)
                        uav.prev_a_ctrl = uav.a_ctrl.copy()
                        uav.compute_jerk()
                        uav.E = uav.E_max
                    mean_l2, parts = run_episode(
                        swarm,
                        steps,
                        power_fn=lambda uas: [
                            (np.linalg.norm(uav.a_ctrl) * 0.01, 0.1, 0.05)
                            for uav in uas
                        ],
                        wind_inputs=wind_inputs,
                        cthr_base=cthr_base,
                        gains=gains,
                        v_max=v_max,
                        a_max=a_max,
                        j_max=j_max,
                    )
                    lambda_means.append(mean_l2)
                    partition_counts.append(parts)
                records.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "mean_lambda2": np.mean(lambda_means),
                        "partition_rate": np.mean(partition_counts) / steps,
                    }
                )
    return pd.DataFrame(records)


# Run the full grid sweep
df_results = simulate_parameter_sweep(n_uav=5, dt=0.1, steps=100, episodes=100)

# Save to CSV
df_results.to_csv("parameter_sweep_results.csv", index=False)

print(df_results)
