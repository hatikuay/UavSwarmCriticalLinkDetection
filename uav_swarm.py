import numpy as np
import pandas as pd
from scipy.special import gammaincc  # <-- import added for incomplete gamma


class UAVState:
    """
    Single UAV state with:
      - p: ENU-frame position (np.array of shape (3,))
      - v: ENU-frame velocity (np.array of shape (3,))
      - a_ctrl: commanded acceleration (np.array of shape (3,))
      - j: jerk (np.array of shape (3,))
      - h_ch: probabilistic link-quality headroom (float, between 0 and 1)
      - E: remaining energy (float)
      - omega: process noise vector (np.array of shape (9,))
      - eps: measurement noise vector (np.array of shape (3,))
      - w_wind: wind disturbance (np.array of shape (3,))
    """

    def __init__(self, dt, initial_state=None, E_max=1.0):
        self.dt = dt
        init = initial_state or {}

        # Position p_i(t) ∈ ℝ³ [m]
        self.p = np.array(init.get("p", np.zeros(3)))

        # Velocity v_i(t) ∈ ℝ³ [m/s]
        self.v = np.array(init.get("v", np.zeros(3)))

        # Commanded acceleration a_ctrl_i(t) ∈ ℝ³ [m/s²]
        self.a_ctrl = np.array(init.get("a_ctrl", np.zeros(3)))
        self.prev_a_ctrl = self.a_ctrl.copy()

        # Jerk j_i(t) ∈ ℝ³ [m/s³]
        self.j = np.zeros(3)

        # Channel quality headroom h_ch_i(t) ∈ [0,1]
        self.h_ch = init.get("h_ch", 0.0)

        # Remaining energy E_i(t) ∈ [0, E_max]
        self.E = init.get("E", init.get("E_max", 1.0))
        self.E_max = E_max
        self.E_max = init.get("E_max", 1.0)
        self.E_min = 0.15 * self.E_max

        # Process noise ω_i(t) ∈ ℝ⁹
        self.omega = np.zeros(9)

        # Measurement noise ε_i(t) ∈ ℝ³
        self.eps = np.zeros(3)

        # Wind disturbance w_wind_i(t) ∈ ℝ³ [m/s²]
        self.w_wind = np.zeros(3)

    def compute_jerk(self):
        """Discrete derivative of control acceleration."""
        self.j = (self.a_ctrl - self.prev_a_ctrl) / self.dt
        self.prev_a_ctrl = self.a_ctrl.copy()

    def enforce_limits(self, v_max=23.0, a_max=60.0):
        """Hard-limit velocity and control acceleration."""

        # Limit velocity
        v_norm = np.linalg.norm(self.v)
        if v_norm > v_max:
            self.v *= v_max / v_norm

        # Limit commanded acceleration
        a_norm = np.linalg.norm(self.a_ctrl)
        if a_norm > a_max:
            self.a_ctrl *= a_max / a_norm

    def update_energy(self, P_prop, P_comm, P_payload):
        """Update remaining energy based on power consumption [J per second]."""
        self.E -= (P_prop + P_comm + P_payload) * self.dt
        self.E = np.clip(self.E, 0.0, self.E_max)

    def check_low_energy(self):
        """Return True if energy has fallen below threshold."""
        return self.E <= self.E_min

    def apply_process_noise(self, Q):
        """Inject process noise into position, velocity, and acceleration."""
        noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        self.omega = noise
        self.p += noise[0:3]
        self.v += noise[3:6]
        self.a_ctrl += noise[6:9]

    def apply_measurement_noise(self, R):
        """Generate and store measurement noise."""
        noise = np.random.multivariate_normal(np.zeros(R.shape[0]), R)
        self.eps = noise
        return noise

    def apply_wind(self, wind_vector):
        """Apply wind disturbance to velocity."""
        self.w_wind = np.array(wind_vector)
        self.v += self.w_wind * self.dt

    def probabilistic_link_validation(self, distance, m, gamma_0, SNR_0, d0, path_loss_exp):
        """
        Compute probability that SNR ≥ γ₀ under Nakagami-m fading:
        Pr[SNR ≥ γ₀] = Γ(m, m·(γ₀/SNR₀)·(d/d₀)^ξ) / Γ(m)
        where Γ(m, ·) is the upper incomplete gamma function.
        """
        argument = m * (gamma_0 / SNR_0) * (distance / d0)**path_loss_exp
        prob = gammaincc(m, argument)  # normalized: gammaincc = Γ(m, arg)/Γ(m)
        return prob

    def update_channel_quality(self, distance, m, gamma_0, SNR_0, d0, path_loss_exp):
        """
        Update h_ch to the probabilistic headroom for a given link distance.
        """
        self.h_ch = self.probabilistic_link_validation(
            distance, m, gamma_0, SNR_0, d0, path_loss_exp
        )


class UAVSwarm:
    """
    Swarm model with per-UAV EKF fusing IMU accel and GNSS Doppler.
    """

    def __init__(
        self, n_uav, dt, Q_proc=None, Q_kf=None, R_kf=None, channel_params=None, p_s=None
    ):
        self.dt = dt
        self.n = n_uav
        self.uavs = [UAVState(dt, E_max=(channel_params.get('E_max',1.0) if channel_params else 1.0))
                     for _ in range(n_uav)]

        # EKF state & covariances
        self.x_est = [np.hstack((uav.p, uav.v)) for uav in self.uavs]
        self.P = [np.eye(6) * 0.1 for _ in range(n_uav)]
        self.Q_proc = Q_proc  # for process noise injection
        self.Q_kf = Q_kf  # EKF process covariance (6x6)
        self.R_kf = R_kf  # EKF measurement covariance (3x3)

        # Channel (Nakagami-m) parameters
        params = channel_params or {}
        self.m = params.get('m', 2.5)
        # average linear SNR at reference distance d0 (convert from dB if given)
        snr0_db = params.get('snr0_db', 22)
        self.snr0 = 10 ** (snr0_db / 10)
        self.d0 = params.get('d0', 100.0)
        self.pl_exp = params.get('pl_exp', 2.4)
        # SNR target (linear)
        gamma0_db = params.get('gamma0_db', 10)
        self.gamma0 = 10 ** (gamma0_db / 10)

        I3 = np.eye(3)
        self.F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        self.B = np.vstack((0.5 * (dt**2) * I3, dt * I3))
        self.H = np.hstack((np.zeros((3, 3)), I3))

        # connectivity placeholders
        self.last_distances = np.zeros((n_uav, n_uav))
        self.link_up_prob = np.zeros((n_uav, n_uav))
        self.C = np.zeros((n_uav, n_uav))
        self.theta_minus = np.zeros((n_uav, n_uav))
        self.theta_plus = np.zeros((n_uav, n_uav))
        self.last_lambda2 = 0.0
        # Yer istasyonu pozisyonu (3‐lük vektör)
        self.p_s = np.array(p_s) if p_s is not None else None
        self.p_s = np.array(p_s) if p_s is not None else None
        self.time_series = []  # <-- buraya ekle

    def compute_distances(self):
        """Compute pairwise Euclidean distances between UAVs."""
        positions = np.array([uav.p for uav in self.uavs])
        dmat = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2)
        self.last_distances = dmat
        return dmat

    def compute_link_up_probabilities(self):
        """
        Under Nakagami-m fading, compute Pr[SNR >= gamma0] for each link (i,j):
          Pr = Gamma(m, m*(gamma0/snr0)*(d/d0)^pl_exp) / Gamma(m)
        using the regularized upper incomplete gamma (gammaincc).
        """
        d = self.last_distances
        # compute lambda = m * (gamma0/snr0) * (d/d0)**pl_exp
        lam = self.m * (self.gamma0 / self.snr0) * (d / self.d0) ** self.pl_exp
        # regularized upper incomplete gamma: gammaincc(m, lam)
        P = gammaincc(self.m, lam)
        np.fill_diagonal(P, 0.0)
        self.link_up_prob = P
        return P

    def compute_laplacian(self, W):
        """Given adjacency matrix W, compute graph Laplacian L = D - W."""
        D = np.diag(W.sum(axis=1))
        L = D - W
        return L

    def compute_adaptive_thresholds(self, cthr_base, gains, v_max, a_max, j_max):
        """
        Compute adaptive thresholds with clamped multiplier to avoid runaway growth.
        """
        dmat = self.last_distances
        i_u, j_u = np.triu_indices(self.n, k=1)
        d_vals = dmat[i_u, j_u]
        mean_d, std_d = d_vals.mean(), d_vals.std()

        C = np.zeros_like(dmat)
        theta_minus = np.zeros_like(dmat)
        theta_plus = np.zeros_like(dmat)

        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                dv = np.linalg.norm(self.uavs[i].v - self.uavs[j].v)
                # safe‐guard zero‐velocity
                vi_norm = np.linalg.norm(self.uavs[i].v) + 1e-9
                vj_norm = np.linalg.norm(self.uavs[j].v) + 1e-9
                dot = np.clip(self.uavs[i].v.dot(
                    self.uavs[j].v) / (vi_norm*vj_norm), -1, 1)
                dtheta = np.arccos(dot)
                da = np.linalg.norm(self.uavs[i].a_ctrl - self.uavs[j].a_ctrl)
                dj = np.linalg.norm(self.uavs[i].j - self.uavs[j].j)

                raw = (
                    1
                    + gains["alpha"] * (std_d / mean_d)
                    - gains["beta"] * (dv / v_max)
                    - gains["gamma"] * (dtheta / np.pi)
                    + gains["delta"] * (da / a_max)
                    - gains["zeta"] * (dj / j_max)
                )
                # clamp multiplier between 0.5 and 1.5
                raw = np.clip(raw, 0.5, 1.5)

                C_val = cthr_base * raw
                C[i, j] = C_val
                theta_minus[i, j] = 0.9 * C_val
                theta_plus[i, j] = C_val

        self.C = C
        self.theta_minus = theta_minus
        self.theta_plus = theta_plus

    def compute_weighted_adjacency(self, dmat, epsilon=1e-6):
        W = np.zeros_like(dmat)
        for i in range(self.n):
            inv_sq = np.power(dmat[i], -2, where=(np.arange(self.n) != i))
            inv_sq[i] = 0
            denom = inv_sq.sum()
            if denom > 0:
                W[i] = inv_sq / denom
        return W

    def compute_weighted_adjacency_with_ground(self, dmat):
        """
        dmat: (n x n) UAV–UAV mesafe matrisi
        self.p_s varsa, dmat'ı (n+1)x(n+1) yap ve
        Eq. (2) + Eq. (3) uyarınca normalize et.
        """
        n = self.n
        # 1) Eğer ground station yoksa orijinal fonksiyonu çağır
        if self.p_s is None:
            return self.compute_weighted_adjacency(dmat)

        # 2) UAV–ground mesafelerini hesapla
        #    son indeksi s olarak ayarla
        dmat_ext = np.zeros((n+1, n+1))
        dmat_ext[:n, :n] = dmat
        # UAV→s ve s→UAV mesafeleri
        for i in range(n):
            d_us = np.linalg.norm(self.uavs[i].p - self.p_s)
            dmat_ext[i, n] = d_us     # i→s
            dmat_ext[n, i] = d_us     # s→i (mesafe simetrik)
        dmat_ext[n, n] = np.inf       # s→s anlamsız

        # 3) Ağırlıkları normalize et
        W = np.zeros_like(dmat_ext)
        for i in range(n+1):
            inv_sq = np.zeros(n+1)
            # tüm j ≠ i için d^(-2)
            mask = np.arange(n+1) != i
            inv_sq[mask] = dmat_ext[i, mask]**(-2)
            inv_sq[i] = 0
            # ground‐station’dan UAV’a ağırlık sıfır (Eq. 3)
            if i == n:
                inv_sq[:] = 0
            denom = inv_sq.sum()
            if denom > 0:
                W[i, :] = inv_sq / denom
        return W

    def compute_algebraic_connectivity(self, L):
        eigs = np.linalg.eigvalsh(L)
        # second smallest eigenvalue
        return np.sort(eigs)[1]

    def ekf_predict_update(self):
        for i, uav in enumerate(self.uavs):
            # Prediction
            x_pred = self.F.dot(self.x_est[i]) + self.B.dot(uav.a_ctrl)
            P_pred = self.F.dot(self.P[i]).dot(self.F.T) + self.Q_kf

            # Simulated GNSS Doppler measurement
            z = uav.v + np.random.multivariate_normal(np.zeros(3), self.R_kf)

            # Update
            y = z - self.H.dot(x_pred)
            S = self.H.dot(P_pred).dot(self.H.T) + self.R_kf
            K = P_pred.dot(self.H.T).dot(np.linalg.inv(S))
            x_upd = x_pred + K.dot(y)
            P_upd = (np.eye(6) - K.dot(self.H)).dot(P_pred)

            # Save estimates
            self.x_est[i] = x_upd
            self.P[i] = P_upd

            # Overwrite UAVState
            uav.p = x_upd[0:3]
            uav.v = x_upd[3:6]

    def compute_predicted_adjacency(self, p_min):
        """
        Compute predicted adjacency A_hat based on:
          - predicted positions: p + v*dt
          - viability: d_hat <= theta_plus AND link_up_prob >= p_min
        """
        # 1) predicted positions
        preds = np.array([uav.p + uav.v * self.dt for uav in self.uavs])
        # 2) pairwise predicted distances
        d_pred = np.linalg.norm(preds[:, None, :] - preds[None, :, :], axis=2)
        # **Store predicted distances** so the UI can access them:
        self.predicted_distances = d_pred

        # 3) build adjacency
        A_hat = np.zeros_like(d_pred, dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                if (d_pred[i, j] <= self.theta_plus[i, j]
                        and self.link_up_prob[i, j] >= p_min):
                    A_hat[i, j] = 1

        # **Store adjacency** too
        A_hat = np.maximum(A_hat, A_hat.T)
        self.predicted_adjacency = A_hat
        return A_hat

    def step(
        self,
        power_consumptions,
        cthr_base,
        gains,
        v_max,
        a_max,
        j_max,
        wind_inputs=None,
        p_min=0.9,
        k_max=3,       # maksimum hop sayısı
        s_index=0      # ground-station indeksi (örneğin self.n)
    ):
        # 1) Process noise
        if self.Q_proc is not None:
            for uav in self.uavs:
                uav.apply_process_noise(self.Q_proc)

        # 2) EKF predict & update
        self.ekf_predict_update()

        # 3) State updates (jerk, limits, energy, wind)
        for i, uav in enumerate(self.uavs):
            uav.compute_jerk()
            uav.enforce_limits(v_max, a_max)
            P_prop, P_comm, P_payload = power_consumptions[i]
            uav.update_energy(P_prop, P_comm, P_payload)
            if wind_inputs is not None:
                uav.apply_wind(wind_inputs[i])

        # 4) Current connectivity metrics
        dmat = self.compute_distances()
        link_probs = self.compute_link_up_probabilities()

        # 5) Her UAV için kanal kalitesini güncelle (h_ch alanı)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                dist = dmat[i, j]
                self.uavs[i].update_channel_quality(
                    distance=dist,
                    m=self.m,
                    gamma_0=self.gamma0,
                    SNR_0=self.snr0,
                    d0=self.d0,
                    path_loss_exp=self.pl_exp
                )

        # 6) Weighted adjacency & Laplacian
        W = self.compute_weighted_adjacency(dmat)
        L = self.compute_laplacian(W)
        self.last_lambda2 = self.compute_algebraic_connectivity(L)

        # 7) Adaptive thresholds (θ₋, θ₊)
        self.compute_adaptive_thresholds(cthr_base, gains, v_max, a_max, j_max)

        # 8) Predicted adjacency A_hat
        A_hat = self.compute_predicted_adjacency(p_min)

        # --- Algoritma 1–4 çağrıları ---
        # 9) Bileşen tahmini
        components = self.predict_components(k_max)

        # 10) Risky-link sınıflandırması
        R_inter, R_intra = self.catalogue_risky_links()

        # 11) Yer istasyonuyla bağlantı kontrolü
        ok, Rs_uav, Ruav_uav = self.ground_connected(s_index)

        # İstersen sonuçları bir attribute’a atayabilirsin:
        self.predicted_components = components
        self.risky_inter = R_inter
        self.risky_intra = R_intra
        self.ground_ok = ok
        self.risky_s_uav = Rs_uav
        self.risky_uav_uav = Ruav_uav

    def get_states(self):
        records = []
        for i, uav in enumerate(self.uavs):
            records.append(
                {
                    "time": None,
                    "uav": i,
                    "p_x": uav.p[0],
                    "p_y": uav.p[1],
                    "p_z": uav.p[2],
                    "v_x": uav.v[0],
                    "v_y": uav.v[1],
                    "v_z": uav.v[2],
                    "C_mean": self.C.mean(),
                    "lambda2": self.last_lambda2,
                    "E": uav.E,
                }
            )
        return records

    def neighbors_k_hop(self, seed: int, k: int) -> set[int]:
        """
        Algorithm 1: KNeighbors
        Find all nodes reachable from `seed` within k hops, using predicted adjacency
        and the lower risk‐band `theta_minus`.
        Returns a set of node indices (0‐based).
        """
        visited = {seed}
        frontier = {seed}
        for _ in range(k):
            next_frontier = set()
            for u in frontier:
                # look at all neighbors v where A_hat[u,v] == 1
                for v in np.where(self.predicted_adjacency[u] == 1)[0]:
                    if v not in visited and self.predicted_distances[u, v] <= self.theta_minus[u, v]:
                        visited.add(v)
                        next_frontier.add(v)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def predict_components(self, k_max: int) -> list[set[int]]:
        """
        Algorithm 2: PredictComponents
        Partition the full set {0…n-1} into connected components by repeated k-hop expansion.
        """
        unseen = set(range(self.n))
        components: list[set[int]] = []
        while unseen:
            u = next(iter(unseen))
            comp = self.neighbors_k_hop(u, k_max)
            components.append(comp)
            unseen -= comp
        self.predicted_components = components
        return components

    def catalogue_risky_links(self
                              ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Algorithm 3: CatalogueRiskyLinks
        Classify every active link whose predicted distance falls
        in (theta_minus, theta_plus] as inter- or intra-component risky.
        Returns (R_inter, R_intra) as lists of (i,j) pairs.
        """
        # build a map from node to its component index
        comp_of = {}
        for ci, comp in enumerate(self.predicted_components):
            for u in comp:
                comp_of[u] = ci

        R_inter = []
        R_intra = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.predicted_adjacency[i, j] == 1:
                    d = self.predicted_distances[i, j]
                    if self.theta_minus[i, j] < d <= self.theta_plus[i, j]:
                        if comp_of[i] != comp_of[j]:
                            R_inter.append((i, j))
                        else:
                            R_intra.append((i, j))

        self.risky_inter = R_inter
        self.risky_intra = R_intra
        return R_inter, R_intra

    def ground_connected(self,
                         s_index: int
                         ) -> tuple[bool, list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Algorithm 4: GroundConnected
        Given inter-component risky links (self.risky_inter) and a central node index s_index,
        do a BFS from s_index over those links and separate them into s–UAV and UAV–UAV subsets.
        Returns (OK?, R_s–UAV, R_UAV–UAV).
        """
        seen = {s_index}
        queue = [s_index]
        R_s_uav = []
        R_uav_uav = []

        while queue:
            u = queue.pop(0)
            for (i, j) in self.risky_inter:
                # is this link incident to u?
                if i == u and j not in seen:
                    seen.add(j)
                    queue.append(j)
                    if i == s_index or j == s_index:
                        R_s_uav.append((i, j))
                    else:
                        R_uav_uav.append((i, j))
                elif j == u and i not in seen:
                    seen.add(i)
                    queue.append(i)
                    if i == s_index or j == s_index:
                        R_s_uav.append((i, j))
                    else:
                        R_uav_uav.append((i, j))

        # OK if we've seen every node (including the ground station if it's in the adjacency)
        total_nodes = self.predicted_adjacency.shape[0]
        ok = (len(seen) == total_nodes)
        return ok, R_s_uav, R_uav_uav


def main():
    dt = 0.1
    n_uav = 3
    steps = 50
    v_max, a_max, j_max = 23.0, 60.0, 100.0
    cthr_base = 100.0
    gains = {"alpha": 0.5, "beta": 0.3,
             "gamma": 0.2, "delta": 0.1, "zeta": 0.05}

    Q_proc = np.diag([0.01] * 9)
    Q_kf = np.eye(6) * 0.05
    R_kf = np.eye(3) * 0.01

    swarm = UAVSwarm(n_uav, dt, Q_proc, Q_kf, R_kf, p_s=np.array([0, 0, 0]))
    np.random.seed(42)
    init_pos = np.random.randn(n_uav, 3)
    for i, uav in enumerate(swarm.uavs):
        uav.p = init_pos[i]
        uav.v = np.random.randn(3)
        uav.a_ctrl = np.random.randn(3)
        uav.prev_a_ctrl = uav.a_ctrl.copy()
        uav.compute_jerk()
        uav.E = uav.E_max

    records = []
    for t in range(steps):
        power_usage = [
            (np.linalg.norm(uav.a_ctrl) * 0.01, 0.1, 0.05) for uav in swarm.uavs
        ]
        wind = [np.array([0.1, 0.0, 0.0])] * n_uav
        swarm.step(power_usage, cthr_base, gains, v_max,
                a_max, j_max, wind_inputs=wind,
                k_max=3, s_index=n_uav)  # <-- buraya ekle

        states = swarm.get_states()
        for rec in states:
            rec["time"] = t * dt
            records.append(rec)

    df = pd.DataFrame(records)
    print(df)


if __name__ == "__main__":
    main()
