
import sys
import serial
import serial.tools.list_ports
import threading
import queue
import math
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QPushButton, QLabel, QGridLayout
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF

pv.set_plot_theme("dark")

class ADIWidget(QWidget):
    """姿勢指示器（ADI）を描画するカスタムウィジェット"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._roll = 0
        self._pitch = 0

    def set_attitude(self, roll, pitch):
        self._roll = roll
        self._pitch = pitch
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = min(self.width(), self.height())
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(size / 200.0, size / 200.0)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#2b2b2b"))
        painter.drawRect(-100, -100, 200, 200)

        painter.save()

        pitch_offset = self._pitch * 2
        painter.translate(0, pitch_offset)
        painter.rotate(-self._roll)

        sky_color = QColor("#3282F6")
        ground_color = QColor("#8B4513")

        painter.setPen(Qt.NoPen)
        painter.setBrush(sky_color)
        painter.drawRect(-300, -300, 600, 300)
        painter.setBrush(ground_color)
        painter.drawRect(-300, 0, 600, 300)

        painter.setPen(QPen(Qt.white, 2))
        painter.drawLine(-300, 0, 300, 0)

        painter.restore()

        painter.setPen(QPen(QColor("yellow"), 3))
        painter.drawLine(-50, 0, -10, 0)
        painter.drawLine(10, 0, 50, 0)
        painter.drawLine(0, -5, 0, 5)

class StickWidget(QWidget):
    """プロポのスティックを描画するカスタムウィジェット"""
    def __init__(self, x_label, y_label, parent=None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self._x_label = x_label
        self._y_label = y_label
        self._x = 0
        self._y = 0

    def set_position(self, x, y):
        self._x = x
        self._y = y
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setBrush(QColor("#2b2b2b"))
        painter.setPen(QColor("gray"))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)

        pen = QPen(QColor("gray"), 1, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(self.width()/2, 5, self.width()/2, self.height()-5)
        painter.drawLine(5, self.height()/2, self.width()-5, self.height()/2)

        painter.setPen(QColor("white"))
        painter.drawText(QRectF(0, 0, self.width(), 15), Qt.AlignCenter, self._y_label)
        painter.drawText(QRectF(self.width() - 35, 0, 35, self.height()), Qt.AlignCenter, self._x_label)

        center_x = self.width()/2 + self._x * (self.width()/2 - 10)
        center_y = self.height()/2 - self._y * (self.height()/2 - 10)
        painter.setBrush(QColor("cyan"))
        painter.setPen(QColor("white"))
        painter.drawEllipse(QPointF(center_x, center_y), 5, 5)


class TelemetryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC Telemetry Display (PySide6)")
        self.setGeometry(100, 100, 1400, 800)

        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 3)

        left_layout.addWidget(self._create_connection_group())
        left_layout.addWidget(self._create_instrument_group())
        left_layout.addWidget(self._create_stick_group())
        left_layout.addStretch(1)

        self.plotter = QtInteractor(right_panel)
        right_layout.addWidget(self.plotter.interactor)
        self._setup_3d_model()

        self.update_com_ports()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_serial_queue)
        self.timer.start(50)

    @staticmethod
    def normalize_value(value, min_in, max_in, min_out=-1.0, max_out=1.0):
        if max_in == min_in: return 0
        value = max(min_in, min(value, max_in))
        return min_out + (value - min_in) * (max_out - min_out) / (max_in - min_in)

    @staticmethod
    def normalize_symmetrical(value, min_in, center_in, max_in):
        value = max(min_in, min(value, max_in))
        if value >= center_in:
            span = max_in - center_in
            return (value - center_in) / span if span > 0 else 1.0
        else:
            span = center_in - min_in
            return (value - center_in) / span if span > 0 else -1.0

    def _create_connection_group(self):
        group = QGroupBox("シリアル接続")
        layout = QGridLayout()

        self.com_port_combo = QComboBox()
        self.refresh_button = QPushButton("更新")
        self.connect_button = QPushButton("接続")

        layout.addWidget(QLabel("COMポート:"), 0, 0)
        layout.addWidget(self.com_port_combo, 0, 1)
        layout.addWidget(self.refresh_button, 0, 2)
        layout.addWidget(self.connect_button, 1, 0, 1, 3)

        self.refresh_button.clicked.connect(self.update_com_ports)
        self.connect_button.clicked.connect(self.toggle_connection)

        group.setLayout(layout)
        return group

    def _create_instrument_group(self):
        group = QGroupBox("計器")
        main_layout = QVBoxLayout()

        self.adi_widget = ADIWidget()
        self.altitude_label = QLabel("高度: 0.0 m")
        self.heading_label = QLabel("方位: 0.0 °")

        main_layout.addWidget(self.adi_widget, 0, Qt.AlignCenter)
        main_layout.addWidget(self.altitude_label)
        main_layout.addWidget(self.heading_label)

        aux_group = QGroupBox("AUXスイッチ")
        aux_layout = QHBoxLayout()
        self.aux_labels = []
        for i in range(1, 5):
            label = QLabel(f"AUX{i}: OFF")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: #555; color: white; padding: 5px; border-radius: 3px;")
            self.aux_labels.append(label)
            aux_layout.addWidget(label)
        aux_group.setLayout(aux_layout)
        main_layout.addWidget(aux_group)

        group.setLayout(main_layout)
        return group

    def _create_stick_group(self):
        group = QGroupBox("プロポ入力")
        layout = QGridLayout()

        self.left_stick = StickWidget("ラダー", "エレベーター")
        self.right_stick = StickWidget("エルロン", "スロットル")
        self.left_stick_label = QLabel("R: 0, E: 0")
        self.right_stick_label = QLabel("A: 0, T: 0")

        layout.addWidget(self.left_stick, 0, 0)
        layout.addWidget(self.right_stick, 0, 1)
        layout.addWidget(self.left_stick_label, 1, 0, Qt.AlignCenter)
        layout.addWidget(self.right_stick_label, 1, 1, Qt.AlignCenter)

        group.setLayout(layout)
        return group

    def _setup_3d_model(self):
        try:
            mesh = pv.read("planeLight.stl")
        except Exception as e:
            print(f"3Dモデルの読み込みに失敗しました: {e}")
            mesh = pv.Sphere(radius=1.0)

        self.plane_actor = self.plotter.add_mesh(mesh, smooth_shading=True)
        self.plotter.view_isometric()
        self.plotter.add_axes()
        self.plotter.enable_zoom_style()

    def update_com_ports(self):
        self.com_port_combo.clear()
        ports = [port.device for port in serial.tools.list_ports.comports()]
        if not ports:
            self.com_port_combo.addItem("ポートなし")
        else:
            self.com_port_combo.addItems(ports)

    def toggle_connection(self):
        if not self.is_connected:
            port = self.com_port_combo.currentText()
            if "ポートなし" in port: return
            try:
                self.serial_connection = serial.Serial(port, baudrate=115200, timeout=1)
                self.is_connected = True
                self.connect_button.setText("切断")
                self.com_port_combo.setEnabled(False)
                self.refresh_button.setEnabled(False)
                threading.Thread(target=self.read_serial_data, daemon=True).start()
            except serial.SerialException as e:
                print(f"接続に失敗しました: {e}")
        else:
            self.is_connected = False
            if self.serial_connection:
                self.serial_connection.close()
            self.connect_button.setText("接続")
            self.com_port_combo.setEnabled(True)
            self.refresh_button.setEnabled(True)

    def read_serial_data(self):
        while self.is_connected and self.serial_connection:
            try:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    self.data_queue.put(line)
            except (serial.SerialException, UnicodeDecodeError):
                break
        self.data_queue.put("CONNECTION_LOST")

    def process_serial_queue(self):
        try:
            while not self.data_queue.empty():
                item = self.data_queue.get_nowait()
                if item == "CONNECTION_LOST":
                    if self.is_connected:
                        self.toggle_connection()
                elif isinstance(item, str):
                    self.parse_and_update_ui(item)
        except queue.Empty:
            pass

    def parse_and_update_ui(self, line):
        try:
            parts = [float(p) for p in line.split(',')]
            if len(parts) == 18:
                roll, pitch, yaw, alt, ail, elev, thro, rudd, aux1, aux2, aux3, aux4, *_ = parts

                if self.plane_actor:
                    self.plane_actor.SetOrientation(0, 0, 0)
                    self.plane_actor.RotateZ(yaw)
                    self.plane_actor.RotateY(pitch)
                    self.plane_actor.RotateX(roll)

                self.adi_widget.set_attitude(roll, pitch)
                self.altitude_label.setText(f"高度: {alt:.1f} m")
                self.heading_label.setText(f"方位: {yaw:.1f} °")

                self.update_aux_switches([aux1, aux2, aux3, aux4])

                # 新しい範囲と中心点で正規化し、UIを更新
                # 左スティック: ラダー(X), エレベーター(Y)
                rud_norm = self.normalize_symmetrical(rudd, 830, 1148, 1500)
                ele_norm = self.normalize_symmetrical(elev, 800, 966, 1070)
                # 表示方向を反転
                rud_norm = -rud_norm
                ele_norm = -ele_norm
                self.left_stick.set_position(rud_norm, ele_norm)
                self.left_stick_label.setText(f"R: {int(rudd)}, E: {int(elev)}")

                # 右スティック: エルロン(X), スロットル(Y)
                ail_norm = self.normalize_symmetrical(ail, 560, 1164, 1750)
                thr_norm = self.normalize_value(thro, 360, 1590) # スロットルは-1から1に
                self.right_stick.set_position(ail_norm, thr_norm)
                self.right_stick_label.setText(f"A: {int(ail)}, T: {int(thro)}")

        except (ValueError, IndexError) as e:
            print(f"データ解析エラー: {e} - {line}")

    def update_aux_switches(self, aux_values):
        on_style = "background-color: #28a745; color: white; padding: 5px; border-radius: 3px;"
        off_style = "background-color: #555; color: white; padding: 5px; border-radius: 3px;"
        for i, value in enumerate(aux_values):
            label = self.aux_labels[i]
            if value > 1100:
                label.setText(f"AUX{i+1}: ON")
                label.setStyleSheet(on_style)
            else:
                label.setText(f"AUX{i+1}: OFF")
                label.setStyleSheet(off_style)

    def closeEvent(self, event):
        self.is_connected = False
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.plotter.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TelemetryApp()
    window.show()
    sys.exit(app.exec())
