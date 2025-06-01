import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from uav_swarm import UAVSwarm  # uav_swarm.py içindeki sınıf
import matplotlib
matplotlib.use('Qt5Agg')  # PyQt5 için en stabil backend
import matplotlib.pyplot as plt



# -----------------------------------
# Canvas (çizim alanı)
# -----------------------------------
class SwarmCanvas(FigureCanvas):
    def __init__(self, swarm, parent=None):
        fig, self.ax = plt.subplots(figsize=(6,6))
        super().__init__(fig)
        self.setParent(parent)
        self.swarm = swarm

    def plot(self):
        self.ax.clear()
        # UAV pozisyonları
        pos = np.array([u.p for u in self.swarm.uavs])
        self.ax.scatter(pos[:,0], pos[:,1], c='blue', label='UAVs')
        # Yer istasyonu
        self.ax.scatter(0, 0, c='red', marker='*', s=100, label='Ground Station')

        # Bağlantı çizgileri: gri = güvenli, kırmızı = riskli
        A       = self.swarm.predicted_adjacency
        d_pred  = self.swarm.predicted_distances
        theta_m = self.swarm.theta_minus
        for i in range(self.swarm.n):
            xi, yi = self.swarm.uavs[i].p[:2]
            for j in range(i+1, self.swarm.n):
                if A[i,j]:
                    xj, yj = self.swarm.uavs[j].p[:2]
                    color = 'gray' if d_pred[i,j] <= theta_m[i,j] else 'red'
                    self.ax.plot([xi, xj], [yi, yj], color=color, linewidth=1.5)

        # Başlıkta adım sayısı, bileşen ve riskli bağlantı sayısı, lambda2
        comp_count  = len(self.swarm.predicted_components)
        risky_count = len(self.swarm.risky_inter)
        self.ax.set_title(
            f"Step: {len(self.swarm.time_series)}  "
            f"Components: {comp_count}  "
            f"Risky: {risky_count}  "
            f"λ₂: {self.swarm.last_lambda2:.3f}"
        )
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.legend(loc='upper right')
        self.draw()


# -----------------------------------
# Ana Pencere
# -----------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Swarm Simulator")

        # Sim parametreleri
        self.dt        = 0.1
        self.cthr_base = 100.0
        self.gains     = {"alpha":0.35, "beta":0.10, "gamma":0.20, "delta":0.05, "zeta":0.05}
        self.v_max     = 23.0
        self.a_max     = 60.0
        self.j_max     = 100.0

        # Swarm oluştur
        self.swarm = UAVSwarm(
            n_uav=5, dt=self.dt,
            Q_proc=np.diag([0.01]*9),
            Q_kf=np.eye(6)*0.05,
            R_kf=np.eye(3)*0.01
        )
        self.swarm.time_series     = []
        self.swarm.predicted_components = []
        self.swarm.risky_inter     = []

        self.p_min = 0.9
        self.k_max = 3

        # Zamanlayıcı
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step_sim)

        # Çizim alanı
        self.canvas = SwarmCanvas(self.swarm)

        # Kontroller
        slider_p = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider_p.setRange(0,100)
        slider_p.setValue(int(self.p_min*100))
        slider_p.valueChanged.connect(lambda v: setattr(self, 'p_min', v/100))

        spin_n = QtWidgets.QSpinBox()
        spin_n.setRange(2,20)
        spin_n.setValue(self.swarm.n)
        spin_n.valueChanged.connect(self.reset_swarm)

        btn_pause = QtWidgets.QPushButton("Pause/Resume")
        btn_pause.clicked.connect(self.toggle_pause)

        btn_report = QtWidgets.QPushButton("Generate Report")
        btn_report.clicked.connect(self.generate_report)

        ctrl = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(ctrl)
        h.addWidget(QtWidgets.QLabel("p_min"));  h.addWidget(slider_p)
        h.addWidget(QtWidgets.QLabel("UAV count")); h.addWidget(spin_n)
        h.addWidget(btn_pause); h.addWidget(btn_report)

        main = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(main)
        v.addWidget(self.canvas); v.addWidget(ctrl)
        self.setCentralWidget(main)

        self.timer.start(100)  # milisaniye

    def reset_swarm(self, n):
        self.swarm = UAVSwarm(
            n_uav=n, dt=self.dt,
            Q_proc=self.swarm.Q_proc,
            Q_kf=self.swarm.Q_kf,
            R_kf=self.swarm.R_kf,
            p_s=np.array([0, 0, 0])  # Yer istasyonu pozisyonu tanımlandı
        )

        # Rastgele başlangıç değerleri atama
        np.random.seed(42)
        init_pos = np.random.randn(n, 3)
        for i, uav in enumerate(self.swarm.uavs):
            uav.p = init_pos[i]
            uav.v = np.random.randn(3)
            uav.a_ctrl = np.random.randn(3)
            uav.prev_a_ctrl = uav.a_ctrl.copy()
            uav.compute_jerk()
            uav.E = uav.E_max

        self.swarm.time_series = []
        self.swarm.predicted_components = []
        self.swarm.risky_inter = []


    def toggle_pause(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(100)

    def step_sim(self):
        # güç tüketimleri (örnek değerler)
        power = [(0.1, 0.05, 0.02)] * self.swarm.n

        # Swarm adımını çalıştır
        self.swarm.step(
            power,
            self.cthr_base,
            self.gains,
            self.v_max,
            self.a_max,
            self.j_max,
            wind_inputs=[(0, 0, 0)] * self.swarm.n,
            p_min=self.p_min,
            k_max=self.k_max,             # <-- eksik parametre eklendi
            s_index=self.swarm.n          # <-- ground station indeksi açıkça belirtildi
        )

        # λ₂ değerini zaman serisine kaydet
        self.swarm.time_series.append(self.swarm.last_lambda2)

        # Bileşenleri ve riskli bağlantıları hesapla
        self.swarm.predict_components(self.k_max)
        self.swarm.catalogue_risky_links()

        # Kritik eşik altına düşerse uyarı ver ve rapor oluştur
        if self.swarm.last_lambda2 < 0.05:
            self.timer.stop()
            QtWidgets.QMessageBox.warning(self, "Alert", "Connectivity is about to break!")
            self.generate_report()

        # Canvas'ı güncelle
        self.canvas.plot()


    def generate_report(self):
        fname = "uav_report.txt"
        try:
            with open(fname, "w", encoding="utf-8") as f:  # <-- encoding ekle!
                f.write("UAV Connectivity Report\n")
                f.write(f"Final λ₂: {self.swarm.last_lambda2:.4f}\n")
                f.write(f"Steps: {len(self.swarm.time_series)}\n")
                f.write(f"Components: {len(self.swarm.predicted_components)}\n")
                f.write(f"Risky links: {len(self.swarm.risky_inter)}\n")
                f.write("λ₂ time series:\n")
                f.write(", ".join(f"{v:.3f}" for v in self.swarm.time_series))
            QtWidgets.QMessageBox.information(self, "Report", f"Saved to {fname}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Report Error", str(e))



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w   = MainWindow()
    w.show()
    sys.exit(app.exec())
