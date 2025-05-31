# ------------------------------------------------------------------------------

# Full `simulation_pyQT.py` with updated visualization

import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from uav_swarm import UAVSwarm  # your updated class

class SwarmCanvas(FigureCanvas):
    """Canvas to plot UAVs, ground station, thresholds, and colored links."""
    def __init__(self, swarm):
        fig = Figure(figsize=(6,6))
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        self.swarm = swarm
        

    def plot(self):
        self.ax.clear()
        pos = np.array([u.p for u in self.swarm.uavs])
        self.ax.scatter(pos[:,0], pos[:,1], c='blue', label='UAVs')
        self.ax.scatter(0, 0, c='red', marker='*', s=100, label='Ground Station')

        # Draw per-UAV inner (min) and outer (max) theta_plus circles
        for i, u in enumerate(self.swarm.uavs):
            ths = self.swarm.theta_plus[i]
            r_min, r_max = ths.min(), ths.max()
            c1 = Circle((u.p[0], u.p[1]), r_min,
                        fill=False, linestyle='--', color='green', alpha=0.3)
            c2 = Circle((u.p[0], u.p[1]), r_max,
                        fill=False, linestyle=':', color='orange', alpha=0.3)
            self.ax.add_patch(c1)
            self.ax.add_patch(c2)

        A        = self.swarm.predicted_adjacency
        d_pred   = self.swarm.predicted_distances
        theta_m  = self.swarm.theta_minus

        for i in range(self.swarm.n):
            for j in range(i+1, self.swarm.n):
                if A[i,j]:
                    xi, yi = self.swarm.uavs[i].p[:2]
                    xj, yj = self.swarm.uavs[j].p[:2]
                    color = 'gray' if d_pred[i,j] <= theta_m[i,j] else 'red'
                    self.ax.plot([xi,xj], [yi,yj], color=color, linewidth=1.5)

        title = (
            f"Step: {len(self.swarm.time_series)} | "
            f"λ₂: {self.swarm.last_lambda2:.3f} | "
            f"Risky: {len(self.swarm.risky_inter)}"
        )
        self.ax.set_title(title)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.legend(loc='upper right')
        self.draw()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Swarm Simulator")

        # sim params
        dt = 0.1 
        self.cthr_base = 100.0
        self.gains = {"alpha":0.35, "beta":0.10, "gamma":0.20, "delta":0.05, "zeta":0.05}
        self.v_max, self.a_max, self.j_max = 23.0, 60.0, 100.0

        # init swarm
        self.swarm = UAVSwarm(
            n_uav=5, dt=dt,
            Q_proc=np.diag([0.01]*9),
            Q_kf=np.eye(6)*0.05,
            R_kf=np.eye(3)*0.01
        )
        self.swarm.time_series  = []
        self.swarm.risky_inter   = []

        self.p_min = 0.9
        self.k_max = 3

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step_sim)

        self.canvas = SwarmCanvas(self.swarm)

        # controls
        slider_p = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_p.setRange(0,100)
        slider_p.setValue(int(self.p_min*100))
        slider_p.valueChanged.connect(lambda v: setattr(self, 'p_min', v/100))

        spin_n = QtWidgets.QSpinBox()
        spin_n.setRange(2,20)
        spin_n.setValue(self.swarm.n)
        spin_n.valueChanged.connect(self.reset_swarm)

        btn_pause = QtWidgets.QPushButton("Pause")
        btn_pause.clicked.connect(self.toggle_pause)

        btn_report = QtWidgets.QPushButton("Generate Report")
        btn_report.clicked.connect(self.generate_report)

        ctrl = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(ctrl)
        h.addWidget(QtWidgets.QLabel("p_min")); h.addWidget(slider_p)
        h.addWidget(QtWidgets.QLabel("UAV count")); h.addWidget(spin_n)
        h.addWidget(btn_pause); h.addWidget(btn_report)

        main = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(main)
        v.addWidget(self.canvas); v.addWidget(ctrl)
        self.setCentralWidget(main)

        self.timer.start(100)

    def reset_swarm(self, n):
        self.swarm = self.swarm.__class__(  # rebuild with same covariances
            n_uav=n, dt=self.swarm.dt,
            Q_proc=self.swarm.Q_proc,
            Q_kf=self.swarm.Q_kf,
            R_kf=self.swarm.R_kf
        )
        self.swarm.time_series = []
        self.swarm.risky_inter  = []

    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(100)

    def step_sim(self):
        # run step
        power = [(0.1,0.05,0.02)]*self.swarm.n
        self.swarm.step(power, self.cthr_base, self.gains, self.v_max, self.a_max, self.j_max,
                        wind_inputs=[(0,0,0)]*self.swarm.n,
                        p_min=self.p_min)
        self.swarm.time_series.append(self.swarm.last_lambda2)
        self.swarm.predict_components(self.k_max)
        self.swarm.catalogue_risky_links()

        if self.swarm.last_lambda2 < 0.05:
            self.timer.stop()
            QtWidgets.QMessageBox.warning(self, "Alert", "Connectivity about to break!")
            self.generate_report()

        self.canvas.plot()

    def generate_report(self):
        fname = "uav_report.txt"
        with open(fname, "w") as f:
            f.write("UAV Connectivity Report\n")
            f.write(f"Final λ₂: {self.swarm.last_lambda2:.4f}\n")
            f.write(f"Steps: {len(self.swarm.time_series)}\n")
            f.write(f"p_min: {self.p_min}\n")
            f.write("λ₂ series:\n" + ", ".join(f"{v:.3f}" for v in self.swarm.time_series))
        QtWidgets.QMessageBox.information(self, "Report", f"Saved to {fname}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w   = MainWindow()
    w.show()
    sys.exit(app.exec())