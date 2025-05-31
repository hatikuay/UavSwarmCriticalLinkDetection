import numpy as np
from scipy.special import gammaincc

class UAVState:
    def __init__(self, dt, E_max=1.0):
        self.dt = dt
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.a_ctrl = np.zeros(3)
        self.prev_a_ctrl = np.zeros(3)
        self.j = np.zeros(3)
        self.h_ch = 0.0
        self.p_up = 0.0
        self.E = E_max
        self.E_max = E_max
        self.E_min = 0.15 * E_max
        self.omega = np.zeros(9)
        self.eps = np.zeros(3)
        self.w_wind = np.zeros(3)

    def compute_jerk(self):
        self.j = (self.a_ctrl - self.prev_a_ctrl) / self.dt
        self.prev_a_ctrl = self.a_ctrl.copy()

    def enforce_limits(self, v_max, a_max):
        v_norm = np.linalg.norm(self.v)
        if v_norm > v_max:
            self.v *= (v_max / v_norm)
        a_norm = np.linalg.norm(self.a_ctrl)
        if a_norm > a_max:
            self.a_ctrl *= (a_max / a_norm)

    def update_energy(self, P_prop, P_comm, P_payload):
        self.E = np.clip(self.E - (P_prop + P_comm + P_payload) * self.dt, 0.0, self.E_max)

    def apply_process_noise(self, Q):
        noise = np.random.multivariate_normal(np.zeros(Q.shape[0]), Q)
        self.omega = noise
        self.p += noise[:3]
        self.v += noise[3:6]
        self.a_ctrl += noise[6:9]

    def apply_wind(self, wind_vec):
        self.w_wind = np.array(wind_vec)
        self.v += self.w_wind * self.dt

class UAVSwarm:
    def __init__(
        self, n_uav, dt, Q_proc, Q_kf, R_kf, channel_params=None, p_s=None
    ):
        self.n = n_uav
        self.dt = dt
        self.uavs = [UAVState(dt, E_max=(channel_params.get('E_max',1.0) if channel_params else 1.0))
                     for _ in range(n_uav)]
        self.Q_proc = Q_proc
        self.Q_kf = Q_kf
        self.R_kf = R_kf
        channel_params = channel_params or {}
        self.m = channel_params.get('m', 2.5)
        self.snr0 = 10**(channel_params.get('snr0_db',22)/10)
        self.gamma0 = 10**(channel_params.get('gamma0_db',10)/10)
        self.d0 = channel_params.get('d0',100.0)
        self.xi = channel_params.get('pl_exp',2.4)
        self.p_s = np.array(p_s) if p_s is not None else None

        # Public attributes for simulation.py
        self.last_lambda2 = None
        self.predicted_distances = None
        self.predicted_adjacency = None
        self.theta_minus = None
        self.theta_plus = None
        self.predicted_components = []
        self.risky_inter = []
        self.risky_intra = []
        self.ground_ok = True
        self.risky_s_uav = []
        self.risky_uav_uav = []

        # Working matrices
        self.d = np.zeros((self.n, self.n))
        self.p_up = np.zeros((self.n, self.n))

    def compute_distances(self):
        pos = np.vstack([u.p for u in self.uavs])
        self.d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

    def compute_link_up_prob(self):
        lam = self.m * (self.gamma0 / self.snr0) * (self.d / self.d0) ** self.xi
        self.p_up = gammaincc(self.m, lam)

    def compute_weighted_adjacency(self):
        W = np.zeros_like(self.d)
        for i in range(self.n):
            inv2 = np.zeros(self.n)
            # only invert the non‐self entries
            mask = (np.arange(self.n) != i)
            # and only where distance>0
            valid = mask & (self.d[i] > 0)
            inv2[valid] = 1.0 / (self.d[i, valid] ** 2)
            total = inv2.sum()
            if total > 0:
                W[i, :] = inv2 / total
        return W


    def compute_laplacian(self, W):
        D = np.diag(W.sum(axis=1))
        return D - W

    def compute_algebraic_connectivity(self, L):
        eig = np.linalg.eigvalsh(L)
        return np.sort(eig)[1]

    def compute_adaptive_thresholds(
        self, cthr_base, gains, v_max, a_max, j_max
    ):
        iu, ju = np.triu_indices(self.n, k=1)
        vals = self.d[iu, ju]
        mean_d, std_d = vals.mean(), vals.std()
        self.theta_minus = np.zeros_like(self.d)
        self.theta_plus = np.zeros_like(self.d)
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                dv = np.linalg.norm(self.uavs[i].v - self.uavs[j].v)
                vi, vj = self.uavs[i].v, self.uavs[j].v
                cosang = np.dot(vi, vj) / (
                    np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-9
                )
                dth = np.arccos(np.clip(cosang, -1, 1))
                da = np.linalg.norm(self.uavs[i].a_ctrl - self.uavs[j].a_ctrl)
                dj = np.linalg.norm(self.uavs[i].j - self.uavs[j].j)
                factor = (
                    1
                    + gains['alpha'] * (std_d / mean_d)
                    - gains['beta'] * (dv / v_max)
                    - gains['gamma'] * (dth / np.pi)
                    + gains['delta'] * (da / a_max)
                    - gains['zeta'] * (dj / j_max)
                )
                factor = np.clip(factor, 0.5, 1.5)
                cthr = cthr_base * factor
                self.theta_minus[i, j] = 0.9 * cthr
                self.theta_plus[i, j] = cthr

    def predict_adjacency(self, p_min):
        preds = np.array([u.p + u.v * self.dt for u in self.uavs])
        d_pred = np.linalg.norm(
            preds[:, None, :] - preds[None, :, :], axis=2
        )
        self.predicted_distances = d_pred
        A = ((d_pred <= self.theta_plus) & (self.p_up >= p_min)).astype(int)
        self.predicted_adjacency = np.maximum(A, A.T)

    def neighbors_k_hop(self, seed, k):
        visited, frontier = {seed}, {seed}
        for _ in range(k):
            nxt = set()
            for u in frontier:
                for v in np.where(self.predicted_adjacency[u] == 1)[0]:
                    if (
                        v not in visited
                        and self.predicted_distances[u, v] <= self.theta_minus[u, v]
                    ):
                        nxt.add(v)
                        visited.add(v)
            frontier = nxt
            if not frontier:
                break
        return visited

    def predict_components(self, k_max):
        unseen = set(range(self.n))
        comps = []
        while unseen:
            seed = next(iter(unseen))
            comp = self.neighbors_k_hop(seed, k_max)
            comps.append(comp)
            unseen -= comp
        self.predicted_components = comps

    def catalogue_risky_links(self):
        comp_idx = {
            u: ci for ci, c in enumerate(self.predicted_components) for u in c
        }
        Rint, Rintra = [], []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (
                    self.predicted_adjacency[i, j]
                    and self.theta_minus[i, j] < self.predicted_distances[i, j] <= self.theta_plus[i, j]
                ):
                    if comp_idx[i] != comp_idx[j]:
                        Rint.append((i, j))
                    else:
                        Rintra.append((i, j))
        self.risky_inter = Rint
        self.risky_intra = Rintra

    def ground_connected(self, s_index):
        seen = {s_index}
        queue = [s_index]
        Rs, Ru = [], []
        while queue:
            u = queue.pop(0)
            for i, j in self.risky_inter:
                if i == u and j not in seen:
                    seen.add(j)
                    queue.append(j)
                    (Rs if i == s_index or j == s_index else Ru).append((i, j))
                elif j == u and i not in seen:
                    seen.add(i)
                    queue.append(i)
                    (Rs if i == s_index or j == s_index else Ru).append((i, j))
        ok = len(seen) == self.n
        return ok, Rs, Ru

    def step(
        self,
        power,
        cthr_base,
        gains,
        v_max,
        a_max,
        j_max,
        wind_inputs=None,
        p_min=0.9,
        k_max=3,
        s_index=0,
    ):
        # process noise
        if self.Q_proc is not None:
            for u in self.uavs:
                u.apply_process_noise(self.Q_proc)
        # state updates
        for idx, u in enumerate(self.uavs):
            u.compute_jerk()
            u.enforce_limits(v_max, a_max)
            u.update_energy(*power[idx])
            if wind_inputs is not None:
                u.apply_wind(wind_inputs[idx])
        # connectivity metrics
        self.compute_distances()
        self.compute_link_up_prob()
        W = self.compute_weighted_adjacency()
        L = self.compute_laplacian(W)
        self.last_lambda2 = self.compute_algebraic_connectivity(L)
        self.compute_adaptive_thresholds(cthr_base, gains, v_max, a_max, j_max)
        self.predict_adjacency(p_min)
