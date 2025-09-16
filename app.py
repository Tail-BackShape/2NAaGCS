import sys
import serial
import serial.tools.list_ports
import threading
import queue
import math
import socket
import cv2
import numpy as np
import requests
import json

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QPushButton, QLabel, QGridLayout, QLineEdit, QCheckBox
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPolygonF, QImage, QPixmap, QTransform, QLinearGradient
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QThread, Signal, Slot

# --- PID Controller ---
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-1, 1), integral_limits=(-1, 1)):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.last_error = 0
        self.integral = 0

    def update(self, process_variable, dt):
        error = self.setpoint - process_variable
        self.integral += error * dt
        self.integral = max(self.integral_limits[0], min(self.integral, self.integral_limits[1]))
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return max(self.output_limits[0], min(output, self.output_limits[1]))

    def reset(self):
        self.last_error = 0
        self.integral = 0

# --- Video Worker Thread ---
class VideoWorker(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    error_signal = Signal(str)

    def __init__(self, url):
        super().__init__()
        self.url = url
        self._running = True

    def run(self):
        try:
            with requests.get(self.url, stream=True, timeout=5) as r:
                if r.status_code != 200:
                    self.error_signal.emit(f"HTTPエラー {r.status_code}")
                    return

                self.error_signal.emit("ストリームに接続しました。")
                bytes_buffer = b''
                for chunk in r.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    bytes_buffer += chunk
                    a = bytes_buffer.find(b'\xff\xd8') # JPEGの開始マーカー
                    b = bytes_buffer.find(b'\xff\xd9') # JPEGの終了マーカー
                    if a != -1 and b != -1:
                        jpg = bytes_buffer[a:b+2]
                        bytes_buffer = bytes_buffer[b+2:]

                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.change_pixmap_signal.emit(frame)

            if self._running:
                self.error_signal.emit("ストリームが予期せず終了しました。")

        except requests.exceptions.RequestException as e:
            self.error_signal.emit(f"接続エラー: {e}")
        except Exception as e:
            self.error_signal.emit(f"不明なエラー: {e}")

    def stop(self):
        self._running = False
        self.wait()

# --- Custom UI Widgets ---
class ADIWidget(QWidget):
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

class AltimeterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 200)
        self._altitude = 0
        self._max_display_alt = 100 # max altitude for the bar display

    def set_altitude(self, altitude):
        self._altitude = altitude
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        padding = 10

        # Background
        painter.fillRect(self.rect(), QColor("#2b2b2b"))

        # Bar background
        bar_rect = QRectF(padding, padding, width - 2 * padding, height - 2 * padding)
        painter.setPen(QColor("gray"))
        painter.setBrush(QColor("#444"))
        painter.drawRect(bar_rect)

        # Altitude bar fill
        fill_height_ratio = min(self._altitude / self._max_display_alt, 1.0)
        if fill_height_ratio < 0: fill_height_ratio = 0

        fill_height = bar_rect.height() * fill_height_ratio
        fill_rect = QRectF(bar_rect.left(), bar_rect.bottom() - fill_height, bar_rect.width(), fill_height)

        # Gradient for the bar
        gradient = QLinearGradient(bar_rect.topLeft(), bar_rect.bottomLeft())
        gradient.setColorAt(0, QColor("red"))
        gradient.setColorAt(0.5, QColor("yellow"))
        gradient.setColorAt(1, QColor("lime"))

        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRect(fill_rect)

        # Altitude Text
        painter.setPen(QColor("white"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter | Qt.AlignTop, f"{self._altitude:.1f} m")

        # Scale markings
        painter.setPen(QColor("white"))
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        num_ticks = 5
        for i in range(num_ticks + 1):
            tick_alt = (self._max_display_alt / num_ticks) * i
            y_pos = bar_rect.bottom() - (bar_rect.height() * (i / num_ticks))
            painter.drawLine(bar_rect.right(), y_pos, bar_rect.right() + 5, y_pos)
            painter.drawText(QRectF(bar_rect.right() + 7, y_pos - 8, 30, 16), Qt.AlignLeft | Qt.AlignVCenter, str(int(tick_alt)))

class StickWidget(QWidget):
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

class Attitude2DWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            print(f"画像の読み込みに失敗: {image_path}")
        self._angle = 0

    def set_angle(self, angle):
        self._angle = angle
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        widget_rect = self.rect()
        painter.fillRect(widget_rect, QColor("#2b2b2b"))

        if not self.pixmap.isNull():
            # 1. 画像が回転しても収まるようにスケーリング計算
            # 画像の対角線の長さがウィジェットの短辺に収まるようにする
            img_diagonal = math.sqrt(self.pixmap.width()**2 + self.pixmap.height()**2)
            widget_min_side = min(widget_rect.width(), widget_rect.height())

            # パディングを少し加える
            scale_factor = (widget_min_side * 0.9) / img_diagonal if img_diagonal > 0 else 1.0

            scaled_pixmap = self.pixmap.scaled(
                int(self.pixmap.width() * scale_factor),
                int(self.pixmap.height() * scale_factor),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # 2. 回転と描画
            transform = QTransform()
            # ウィジェットの中心に移動
            transform.translate(widget_rect.center().x(), widget_rect.center().y())
            # 回転
            transform.rotate(self._angle)
            # 画像の中心が原点に来るように移動
            transform.translate(-scaled_pixmap.width() / 2, -scaled_pixmap.height() / 2)

            painter.setTransform(transform)
            painter.drawPixmap(0, 0, scaled_pixmap)

# --- Main Application Window ---
class TelemetryApp(QMainWindow):
    GAINS_TO_TUNE = [
        ("Roll P", "roll_p", "0.1"), ("Roll I", "roll_i", "0.01"), ("Roll D", "roll_d", "0.05"),
        ("Pitch P", "pitch_p", "0.1"), ("Pitch I", "pitch_i", "0.01"), ("Pitch D", "pitch_d", "0.05"),
        ("Yaw P", "yaw_p", "0.2"), ("Yaw I", "yaw_i", "0.0"), ("Yaw D", "yaw_d", "0.0"),
    ]
    
    RC_RANGES = {
        'ail': {'min_in': 560, 'center_in': 1164, 'max_in': 1750},
        'elev': {'min_in': 800, 'center_in': 966, 'max_in': 1070},
        'rudd': {'min_in': 830, 'center_in': 1148, 'max_in': 1500},
        'thro': {'min_in': 360, 'max_in': 1590}
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC Telemetry Display (PySide6)")
        self.setGeometry(100, 100, 1600, 900)

        self.serial_connection = None
        self.is_connected = False
        self.read_thread = None
        self.data_queue = queue.Queue()
        self.video_thread = None
        self.pid_gain_edits = {}
        self.current_pid_gains = {}

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.udp_broadcast_address = ('<broadcast>', 12345)

        # --- Autopilot State ---
        self.autopilot_active = False
        self.active_mission_mode = 0  # 0: Manual, 1: Horizontal, 2: Ascending, 3: Fig-8
        self.mission_start_yaw = 0
        self.mission_start_altitude = 0
        self.yaw_diff = 0
        self.last_yaw = 0
        self.last_autopilot_commands = None
        self.previous_aux_values = [0, 0, 0, 0]
        self.latest_attitude = {'roll': 0, 'pitch': 0, 'yaw': 0, 'alt': 0}

        self._setup_ui()
        self.load_pid_gains() # Also initializes PID controllers
        self._setup_timers()
        self.update_com_ports()

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

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)

        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 3)

        left_layout.addWidget(self._create_connection_group())
        left_layout.addWidget(self._create_instrument_group())
        left_layout.addWidget(self._create_stick_group())
        left_layout.addWidget(self._create_video_group())
        left_layout.addStretch(1)

        # --- Right Panel Layout ---
        top_right_widget = QWidget()
        top_right_layout = QHBoxLayout(top_right_widget)
        top_right_layout.setContentsMargins(0, 0, 0, 0)

        video_widget = self._setup_video_display()
        pid_panel = self._create_autopilot_panel()

        top_right_layout.addWidget(video_widget)
        top_right_layout.addWidget(pid_panel)
        top_right_layout.addStretch(1)

        self.right_layout.addWidget(top_right_widget, 1)
        self._setup_2d_attitude_display(right_panel)

    def _setup_timers(self):
        # Telemetry processing timer
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.process_serial_queue)
        self.telemetry_timer.start(50)  # 20Hz

        # 2D Attitude update timer
        self.attitude_2d_timer = QTimer(self)
        self.attitude_2d_timer.timeout.connect(self.update_2d_attitude)
        self.attitude_2d_timer.start(100) # 10Hz

        # Autopilot loop timer
        self.autopilot_timer = QTimer(self)
        self.autopilot_timer.timeout.connect(self.run_autopilot_cycle)
        self.autopilot_timer.start(50) # 20 Hz

    def _init_pid_controllers(self):
        gains = self.current_pid_gains
        self.roll_pid = PIDController(gains.get('roll_p', 0), gains.get('roll_i', 0), gains.get('roll_d', 0))
        self.pitch_pid = PIDController(gains.get('pitch_p', 0), gains.get('pitch_i', 0), gains.get('pitch_d', 0))
        # TODO: Make alt PID gains adjustable
        self.alt_pid = PIDController(Kp=0.1, Ki=0.02, Kd=0.05, output_limits=(-15, 15)) # Output is target pitch angle
        self.yaw_pid = PIDController(gains.get('yaw_p', 0), gains.get('yaw_i', 0), gains.get('yaw_d', 0))
        print("PID controllers initialized/updated.")

    def update_and_save_pid_gains(self):
        gains_to_save = {}
        try:
            for _, key, _ in self.GAINS_TO_TUNE:
                value_str = self.pid_gain_edits[key].text()
                self.current_pid_gains[key] = float(value_str)
                gains_to_save[key] = value_str
            
            with open("coef.txt", "w") as f:
                json.dump(gains_to_save, f, indent=4)
            
            print(f"PID gains updated and saved: {self.current_pid_gains}")
            self._init_pid_controllers()
        except ValueError as e:
            print(f"PIDゲインの値が不正です: {e}")
        except Exception as e:
            print(f"PIDゲインの保存中にエラーが発生しました: {e}")

    def load_pid_gains(self):
        try:
            with open("coef.txt", "r") as f:
                gains = json.load(f)
            
            for key, value in gains.items():
                if key in self.pid_gain_edits:
                    self.pid_gain_edits[key].setText(str(value))
                    self.current_pid_gains[key] = float(value)
            print(f"Loaded PID gains from coef.txt: {self.current_pid_gains}")

        except FileNotFoundError:
            print("coef.txt not found, using default PID gains.")
            for _, key, default_value in self.GAINS_TO_TUNE:
                self.current_pid_gains[key] = float(default_value)
            print(f"Using default PID gains: {self.current_pid_gains}")
        except Exception as e:
            print(f"PIDゲインの読み込み中にエラーが発生しました: {e}")
        
        self._init_pid_controllers()

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

        instrument_layout = QHBoxLayout()
        self.adi_widget = ADIWidget()
        self.altimeter_widget = AltimeterWidget()
        instrument_layout.addWidget(self.adi_widget)
        instrument_layout.addWidget(self.altimeter_widget)

        main_layout.addLayout(instrument_layout)

        self.heading_label = QLabel("方位: 0.0 °")
        self.mission_status_label = QLabel("ミッション: なし")
        self.mission_status_label.setAlignment(Qt.AlignCenter)
        self.mission_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFD700; padding: 5px;")

        main_layout.addWidget(self.heading_label)
        main_layout.addWidget(self.mission_status_label)

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

    def _create_video_group(self):
        group = QGroupBox("ビデオストリーム")
        layout = QGridLayout(group)
        self.video_url_input = QLineEdit("http://tsuna.local:8080/video")
        self.video_toggle_button = QPushButton("FPV")
        self.video_toggle_button.clicked.connect(self.toggle_video_stream)
        layout.addWidget(QLabel("URL:"), 0, 0)
        layout.addWidget(self.video_url_input, 0, 1)
        layout.addWidget(self.video_toggle_button, 1, 0, 1, 2)
        return group

    def _create_autopilot_panel(self):
        group = QGroupBox("自動操縦係数調整")
        layout = QFormLayout(group)

        for label, key, default_value in self.GAINS_TO_TUNE:
            self.pid_gain_edits[key] = QLineEdit(default_value)
            layout.addRow(label, self.pid_gain_edits[key])

        update_button = QPushButton("係数更新")
        update_button.clicked.connect(self.update_and_save_pid_gains)
        layout.addRow(update_button)

        return group

    def _setup_video_display(self):
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setText("ビデオストリーム停止中")
        self.video_status_label = QLabel("ステータス: 非アクティブ")
        self.video_status_label.setFixedHeight(20)
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.video_status_label)
        return video_container

    def _setup_2d_attitude_display(self, parent):
        container = QGroupBox("2D姿勢表示")
        layout = QHBoxLayout(container)

        self.roll_widget = Attitude2DWidget("2NAa_roll.png")
        self.pitch_widget = Attitude2DWidget("2NAa_pitch.png")
        self.yaw_widget = Attitude2DWidget("2NAa_yaw.png")

        layout.addWidget(self.roll_widget)
        layout.addWidget(self.pitch_widget)
        layout.addWidget(self.yaw_widget)

        self.right_layout.addWidget(container, 2)

    def update_2d_attitude(self):
        roll = self.latest_attitude.get('roll', 0)
        pitch = self.latest_attitude.get('pitch', 0)
        yaw = self.latest_attitude.get('yaw', 0)

        self.roll_widget.set_angle(roll)
        self.pitch_widget.set_angle(pitch)
        self.yaw_widget.set_angle(yaw)

    def toggle_video_stream(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_toggle_button.setText("配信開始")
            self.video_status_label.setText("ステータス: ユーザーにより停止")
            self.video_url_input.setEnabled(True)
        else:
            url = self.video_url_input.text()
            self.video_thread = VideoWorker(url)
            self.video_thread.change_pixmap_signal.connect(self.update_video_image)
            self.video_thread.error_signal.connect(self.update_video_status)
            self.video_thread.start()
            self.video_toggle_button.setText("配信停止")
            self.video_status_label.setText("ステータス: 接続試行中...")
            self.video_url_input.setEnabled(False)

    @Slot(np.ndarray)
    def update_video_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    @Slot(str)
    def update_video_status(self, status_text):
        self.video_status_label.setText(f"ステータス: {status_text}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

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
                self.read_thread = threading.Thread(target=self.read_serial_data, daemon=True)
                self.read_thread.start()
                self.connect_button.setText("切断")
                self.com_port_combo.setEnabled(False)
                self.refresh_button.setEnabled(False)
            except serial.SerialException as e:
                print(f"接続に失敗しました: {e}")
        else:
            self.is_connected = False
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=2)
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.close()
                self.serial_connection = None
            self.read_thread = None
            self.connect_button.setText("接続")
            self.com_port_combo.setEnabled(True)
            self.refresh_button.setEnabled(True)

    def read_serial_data(self):
        while self.is_connected:
            try:
                line = self.serial_connection.readline().decode('utf-8').strip()
                if line:
                    self.data_queue.put(line)
            except (serial.SerialException, UnicodeDecodeError):
                print("Serial reading error, closing connection.")
                break
            except Exception as e:
                print(f"Unexpected error in read_serial_data: {e}")
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
            try:
                self.udp_socket.sendto(line.encode('utf-8'), self.udp_broadcast_address)
            except Exception as e:
                print(f"UDP送信エラー: {e}")
            parts = [float(p) for p in line.split(',')]
            if len(parts) == 18:
                roll, pitch, yaw, alt, ail, elev, thro, rudd, aux1, aux2, aux3, aux4, *_ = parts
                self.latest_attitude = {'roll': roll, 'pitch': pitch, 'yaw': yaw, 'alt': alt}
                
                self.adi_widget.set_attitude(roll, pitch)
                self.altimeter_widget.set_altitude(alt)
                self.heading_label.setText(f"方位: {yaw:.1f} °")
                self.update_aux_switches([aux1, aux2, aux3, aux4])
                
                rud_norm = self.normalize_symmetrical(rudd, **self.RC_RANGES['rudd'])
                ele_norm = self.normalize_symmetrical(elev, **self.RC_RANGES['elev'])
                rud_norm = -rud_norm
                ele_norm = -ele_norm
                self.left_stick.set_position(rud_norm, ele_norm)
                self.left_stick_label.setText(f"R: {int(rudd)}, E: {int(elev)}")
                
                ail_norm = self.normalize_symmetrical(ail, **self.RC_RANGES['ail'])
                thr_norm = self.normalize_value(thro, **self.RC_RANGES['thro'])
                self.right_stick.set_position(ail_norm, thr_norm)
                self.right_stick_label.setText(f"A: {int(ail)}, T: {int(thro)}")

                self.check_mission_triggers([aux1, aux2, aux3, aux4])
        except (ValueError, IndexError) as e:
            print(f"データ解析エラー: {e} - {line}")

    def update_aux_switches(self, aux_values):
        on_style = "background-color: #28a745; color: white; padding: 5px; border-radius: 3px;"
        off_style = "background-color: #555; color: white; padding: 5px; border-radius: 3px;"

        mission_text = "なし"
        missions = { 1: "水平旋回", 2: "上昇旋回", 3: "八の字旋回" } # AUX index (0-based) to mission_mode

        for i, value in enumerate(aux_values):
            label = self.aux_labels[i]
            is_on = value > 1100

            if is_on:
                label.setText(f"AUX{i+1}: ON")
                label.setStyleSheet(on_style)
                # i は 0-3, missionsのキーは 1-3. AUX2 is index 1.
                if (i+1) in missions:
                    mission_text = missions[i+1]
            else:
                label.setText(f"AUX{i+1}: OFF")
                label.setStyleSheet(off_style)

        self.mission_status_label.setText(f"ミッション: {mission_text}")

    def check_mission_triggers(self, aux_values):
        # Map AUX SWITCH NUMBER (2, 3, 4) to mission_mode (1, 2, 3)
        mission_map = {
            2: 1, # AUX2 -> Mission 1 (Horizontal)
            3: 2, # AUX3 -> Mission 2 (Ascending)
            4: 3  # AUX4 -> Mission 3 (Fig-8)
        }
        
        mission_found = False
        # Prioritize lower AUX numbers if multiple are on
        for aux_number in sorted(mission_map.keys()):
            aux_index = aux_number - 1
            if aux_values[aux_index] > 1100: # If this mission switch is ON
                mission_found = True
                prev_val = self.previous_aux_values[aux_index]
                if prev_val < 1100: # And it was previously OFF (rising edge)
                    self.start_mission(mission_map[aux_number])
                break # Only handle one mission at a time
        
        if not mission_found:
            self.stop_mission()

        self.previous_aux_values = list(aux_values)

    def start_mission(self, mission_mode):
        if self.autopilot_active and self.active_mission_mode == mission_mode:
            return
        print(f"Starting Mission: {mission_mode}")
        self.autopilot_active = True
        self.active_mission_mode = mission_mode
        self.mission_start_yaw = self.latest_attitude.get('yaw', 0)
        self.mission_start_altitude = self.latest_attitude.get('alt', 0)
        self.last_yaw = self.mission_start_yaw
        self.yaw_diff = 0
        self.roll_pid.reset()
        self.pitch_pid.reset()
        self.alt_pid.reset()
        self.yaw_pid.reset()

    def stop_mission(self):
        if not self.autopilot_active:
            return
        print("Stopping Mission, holding last commands.")
        self.autopilot_active = False
        self.active_mission_mode = 0

    def run_autopilot_cycle(self):
        if not self.is_connected:
            return
            
        if not self.autopilot_active:
            if self.last_autopilot_commands:
                self.send_serial_command(self.last_autopilot_commands)
            return

        current_alt = self.latest_attitude.get('alt', self.mission_start_altitude)
        current_roll = self.latest_attitude.get('roll', 0)
        current_pitch = self.latest_attitude.get('pitch', 0)
        current_yaw = self.latest_attitude.get('yaw', 0)
        dt = 0.05 # 50ms interval

        delta_yaw = current_yaw - self.last_yaw
        if delta_yaw > 180: delta_yaw -= 360
        if delta_yaw < -180: delta_yaw += 360
        self.yaw_diff += delta_yaw
        self.last_yaw = current_yaw

        target_roll = 0
        target_pitch = 0
        target_throttle = 1300 # TODO: Make adjustable

        if self.active_mission_mode == 1: # Horizontal Turn
            target_roll = 20 # TODO: Make adjustable
            self.alt_pid.setpoint = self.mission_start_altitude
            if abs(self.yaw_diff) > 760:
                self.mission_status_label.setText("ミッション: 水平旋回 成功")
        # ... other missions ...

        target_pitch_from_alt = self.alt_pid.update(current_alt, dt)
        self.pitch_pid.setpoint = target_pitch_from_alt + target_pitch
        elev_out = self.pitch_pid.update(current_pitch, dt)

        self.roll_pid.setpoint = target_roll
        ail_out = self.roll_pid.update(current_roll, dt)
        
        rudd_out = 0

        commands = {
            'ail': self.denormalize_symmetrical(ail_out, 'ail'),
            'elev': self.denormalize_symmetrical(elev_out, 'elev'),
            'rudd': self.denormalize_symmetrical(rudd_out, 'rudd'),
            'thro': target_throttle
        }
        self.send_serial_command(commands)
        self.last_autopilot_commands = commands

    def send_serial_command(self, commands):
        try:
            command_str = f"A,{int(commands['ail'])},{int(commands['elev'])},{int(commands['rudd'])},{int(commands['thro'])},{self.active_mission_mode},1500\n"
            if self.serial_connection and self.serial_connection.is_open:
                self.serial_connection.write(command_str.encode('utf-8'))
        except Exception as e:
            print(f"Error sending serial command: {e}")

    def denormalize_value(self, norm_val, channel):
        conf = self.RC_RANGES[channel]
        min_out, max_out = -1.0, 1.0
        return conf['min_in'] + (norm_val - min_out) * (conf['max_in'] - conf['min_in']) / (max_out - min_out)

    def denormalize_symmetrical(self, norm_val, channel):
        conf = self.RC_RANGES[channel]
        if norm_val >= 0:
            return conf['center_in'] + norm_val * (conf['max_in'] - conf['center_in'])
        else:
            return conf['center_in'] + norm_val * (conf['center_in'] - conf['min_in'])

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        
        self.is_connected = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=2)
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()

        self.udp_socket.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TelemetryApp()
    window.showMaximized()
    sys.exit(app.exec())
