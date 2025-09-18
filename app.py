import sys
import serial
import serial.tools.list_ports
import threading
import queue
import math
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
        self._max_display_alt = 15 # max altitude for the bar display

    def set_altitude(self, altitude):
        self._altitude = altitude * 0.001
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
        self._autopilot_x = None  # 自動操縦時のX値（None = 非表示）
        self._autopilot_y = None  # 自動操縦時のY値（None = 非表示）
        self._autopilot_active = False

    def set_position(self, x, y):
        self._x = x
        self._y = y
        self.update()

    def set_autopilot_position(self, x, y):
        self._autopilot_x = x
        self._autopilot_y = y
        self.update()

    def set_autopilot_active(self, active):
        self._autopilot_active = active
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

        # 現在のプロポ入力位置（シアン）
        center_x = self.width()/2 + self._x * (self.width()/2 - 10)
        center_y = self.height()/2 - self._y * (self.height()/2 - 10)
        painter.setBrush(QColor("cyan"))
        painter.setPen(QColor("white"))
        painter.drawEllipse(QPointF(center_x, center_y), 5, 5)

        # 自動操縦時の目標位置（黄色）
        if self._autopilot_active and self._autopilot_x is not None and self._autopilot_y is not None:
            autopilot_x = self.width()/2 + self._autopilot_x * (self.width()/2 - 10)
            autopilot_y = self.height()/2 - self._autopilot_y * (self.height()/2 - 10)
            painter.setBrush(QColor("yellow"))
            painter.setPen(QColor("black"))
            painter.drawEllipse(QPointF(autopilot_x, autopilot_y), 7, 7)

class Attitude2DWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            print(f"画像の読み込みに失敗: {image_path}")
        self._angle = 0
        self._target_angle = None  # 目標角度（None = 非表示）
        self._autopilot_active = False

    def set_angle(self, angle):
        self._angle = angle
        self.update()

    def set_target_angle(self, target_angle):
        self._target_angle = target_angle
        self.update()

    def set_autopilot_active(self, active):
        self._autopilot_active = active
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

            # 3. 目標角度の表示（自動操縦時のみ）
            if self._autopilot_active and self._target_angle is not None:
                painter.resetTransform()
                center = widget_rect.center()
                radius = widget_min_side * 0.4

                # 目標角度のライン（赤色）
                target_rad = math.radians(self._target_angle - 90)  # -90で上方向を0度とする
                target_x = center.x() + radius * math.cos(target_rad)
                target_y = center.y() + radius * math.sin(target_rad)

                painter.setPen(QPen(QColor("red"), 3))
                painter.drawLine(center.x(), center.y(), target_x, target_y)

                # 現在角度のライン（緑色）
                current_rad = math.radians(self._angle - 90)
                current_x = center.x() + radius * 0.8 * math.cos(current_rad)
                current_y = center.y() + radius * 0.8 * math.sin(current_rad)

                painter.setPen(QPen(QColor("lime"), 2))
                painter.drawLine(center.x(), center.y(), current_x, current_y)

                # 角度値の表示
                painter.setPen(QPen(QColor("white"), 1))
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)

                text_rect = QRectF(5, 5, widget_rect.width() - 10, 30)
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop,
                               f"目標: {self._target_angle:.1f}°")

                text_rect = QRectF(5, 25, widget_rect.width() - 10, 30)
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop,
                               f"現在: {self._angle:.1f}°")

# --- Main Application Window ---
class TelemetryApp(QMainWindow):
    GAINS_TO_TUNE = [
        ("Roll P", "roll_p", "0.1"), ("Roll I", "roll_i", "0.01"), ("Roll D", "roll_d", "0.05"),
        ("Pitch P", "pitch_p", "0.1"), ("Pitch I", "pitch_i", "0.01"), ("Pitch D", "pitch_d", "0.05"),
        ("Yaw P", "yaw_p", "0.2"), ("Yaw I", "yaw_i", "0.0"), ("Yaw D", "yaw_d", "0.0"),
    ]

    # 自動操縦パラメータ
    AUTOPILOT_PARAMS = [
        ("バンク角 (度)", "bank_angle", "20.0"),
        ("水平旋回判定角 (度)", "horizontal_turn_target", "760.0"),
        ("上昇旋回判定角 (度)", "ascending_turn_target1", "-760.0"),
        ("八の字右旋回角 (度)", "figure8_right_target", "300.0"),
        ("八の字左旋回角 (度)", "figure8_left_target", "-320.0"),
        ("上昇前高度 (m)", "altitude_low", "2.5"),
        ("上昇後高度 (m)", "altitude_high", "5.0"),
        ("自動スロットル標準", "autopilot_throttle", "1300.0"),
        ("エルロン→ラダーミキシング", "aileron_rudder_mix", "0.3"),
    ]

    RC_RANGES = {
        'ail': {'min_in': 560, 'center_in': 1164, 'max_in': 1750},
        'elev': {'min_in': 800, 'center_in': 966, 'max_in': 1070},
        'rudd': {'min_in': 830, 'center_in': 1148, 'max_in': 1500},
        'thro': {'min_in': 360, 'max_in': 1590}
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("2NAa GCS")
        self.setGeometry(100, 100, 1600, 900)

        self.serial_connection = None
        self.is_connected = False
        self.read_thread = None
        self.data_queue = queue.Queue()
        self.video_thread = None
        self.pid_gain_edits = {}
        self.current_pid_gains = {}
        self.autopilot_param_edits = {}
        self.current_autopilot_params = {}

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

        # --- Figure-8 Mission State ---
        self.figure8_phase = 0  # 0: 右旋回フェーズ, 1: 左旋回フェーズ
        self.figure8_completed = False

        self._setup_ui()
        self.load_pid_gains() # Also initializes PID controllers
        self.load_autopilot_params() # Load autopilot parameters
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
        autopilot_param_panel = self._create_autopilot_param_panel()

        top_right_layout.addWidget(video_widget)
        top_right_layout.addWidget(pid_panel)
        top_right_layout.addWidget(autopilot_param_panel)
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

    def update_and_save_autopilot_params(self):
        params_to_save = {}
        try:
            for _, key, _ in self.AUTOPILOT_PARAMS:
                value_str = self.autopilot_param_edits[key].text()
                self.current_autopilot_params[key] = float(value_str)
                params_to_save[key] = value_str

            # Load existing coef.txt data and add autopilot params
            existing_data = {}
            try:
                with open("coef.txt", "r") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                pass

            existing_data.update(params_to_save)

            with open("coef.txt", "w") as f:
                json.dump(existing_data, f, indent=4)

            print(f"Autopilot parameters updated and saved: {self.current_autopilot_params}")
        except ValueError as e:
            print(f"自動操縦パラメータの値が不正です: {e}")
        except Exception as e:
            print(f"自動操縦パラメータの保存中にエラーが発生しました: {e}")

    def load_autopilot_params(self):
        try:
            with open("coef.txt", "r") as f:
                params = json.load(f)

            for key, value in params.items():
                if key in self.autopilot_param_edits:
                    self.autopilot_param_edits[key].setText(str(value))
                    self.current_autopilot_params[key] = float(value)
            print(f"Loaded autopilot parameters from coef.txt: {self.current_autopilot_params}")

        except FileNotFoundError:
            print("coef.txt not found, using default autopilot parameters.")
            for _, key, default_value in self.AUTOPILOT_PARAMS:
                self.current_autopilot_params[key] = float(default_value)
            print(f"Using default autopilot parameters: {self.current_autopilot_params}")
        except Exception as e:
            print(f"自動操縦パラメータの読み込み中にエラーが発生しました: {e}")

        # Set default values for parameters that aren't loaded
        for _, key, default_value in self.AUTOPILOT_PARAMS:
            if key not in self.current_autopilot_params:
                self.current_autopilot_params[key] = float(default_value)

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
        group = QGroupBox("自動操縦操舵量計算ゲイン")
        layout = QFormLayout(group)

        for label, key, default_value in self.GAINS_TO_TUNE:
            self.pid_gain_edits[key] = QLineEdit(default_value)
            layout.addRow(label, self.pid_gain_edits[key])

        update_button = QPushButton("係数更新")
        update_button.clicked.connect(self.update_and_save_pid_gains)
        layout.addRow(update_button)

        return group

    def _create_autopilot_param_panel(self):
        group = QGroupBox("自動操縦パラメータ")
        layout = QFormLayout(group)

        for label, key, default_value in self.AUTOPILOT_PARAMS:
            self.autopilot_param_edits[key] = QLineEdit(default_value)
            layout.addRow(label, self.autopilot_param_edits[key])

        update_button = QPushButton("パラメータ更新")
        update_button.clicked.connect(self.update_and_save_autopilot_params)
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

        # ヨー軸は累積角度で表示（自動操縦時のみ）
        if self.autopilot_active:
            # 自動操縦時は累積角度（mission_start_yaw + yaw_diff）で表示
            cumulative_yaw = self.mission_start_yaw + self.yaw_diff
            self.yaw_widget.set_angle(cumulative_yaw)
        else:
            # 手動操縦時は受信した値をそのまま表示
            self.yaw_widget.set_angle(yaw)

        # 自動操縦状態を2Dウィジェットに反映
        self.roll_widget.set_autopilot_active(self.autopilot_active)
        self.pitch_widget.set_autopilot_active(self.autopilot_active)
        self.yaw_widget.set_autopilot_active(self.autopilot_active)

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
            # COM22をデフォルトで選択
            com22_index = -1
            for i, port in enumerate(ports):
                if port == "COM22":
                    com22_index = i
                    break
            if com22_index >= 0:
                self.com_port_combo.setCurrentIndex(com22_index)

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

                # スティックウィジェットに自動操縦状態を設定
                self.left_stick.set_autopilot_active(self.autopilot_active)
                self.right_stick.set_autopilot_active(self.autopilot_active)
                self.check_mission_triggers([aux1, aux2, aux3, aux4])
        except (ValueError, IndexError) as e:
            print(f"データ解析エラー: {e} - {line}")

    def update_aux_switches(self, aux_values):
        on_style = "background-color: #28a745; color: white; padding: 5px; border-radius: 3px;"
        off_style = "background-color: #555; color: white; padding: 5px; border-radius: 3px;"

        mission_text = "なし"
        missions = { 2: "水平旋回", 3: "上昇旋回", 4: "八の字旋回" } # AUX switch number to mission name

        for i, value in enumerate(aux_values):
            label = self.aux_labels[i]
            is_on = value > 1100
            aux_number = i + 1  # AUX1, AUX2, AUX3, AUX4

            if is_on:
                if aux_number == 1:
                    label.setText(f"AUX1: ON")
                else:
                    label.setText(f"AUX{aux_number}: ON")
                label.setStyleSheet(on_style)
                # Check if this AUX is assigned to a mission
                if aux_number in missions:
                    mission_text = missions[aux_number]
            else:
                if aux_number == 1:
                    label.setText(f"AUX1: OFF")
                else:
                    label.setText(f"AUX{aux_number}: OFF")
                label.setStyleSheet(off_style)

        self.mission_status_label.setText(f"ミッション: {mission_text}")

    def check_mission_triggers(self, aux_values):
        # Map AUX SWITCH NUMBER to mission_mode
        # AUX1: 物資投下（ミッション処理なし）
        # AUX2: 水平旋回（Mission 1）
        # AUX3: 上昇旋回（Mission 2）
        # AUX4: 八の字旋回（Mission 3）
        mission_map = {
            2: 1, # AUX2 -> Mission 1 (水平旋回)
            3: 2, # AUX3 -> Mission 2 (上昇旋回)
            4: 3  # AUX4 -> Mission 3 (八の字旋回)
        }

        mission_found = False
        # Prioritize lower AUX numbers if multiple are on
        for aux_number in sorted(mission_map.keys()):
            aux_index = aux_number - 1
            if aux_values[aux_index] > 1100: # If this mission switch is ON
                mission_found = True
                prev_val = self.previous_aux_values[aux_index]
                if prev_val < 1100: # And it was previously OFF (rising edge)
                    # AUX2,3,4スイッチが入った瞬間にyaw_diffをリセット
                    self.yaw_diff = 0
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

        # 八の字旋回の状態をリセット
        self.figure8_phase = 0  # 右旋回から開始
        self.figure8_completed = False

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

        # 目標角度の表示をクリア
        self.roll_widget.set_target_angle(None)
        self.pitch_widget.set_target_angle(None)
        self.yaw_widget.set_target_angle(None)

        # スティックウィジェットの自動操縦表示をクリア
        self.left_stick.set_autopilot_position(None, None)
        self.right_stick.set_autopilot_position(None, None)

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

        # ヨー角の差分計算（-180〜180度の範囲で正規化）
        delta_yaw = current_yaw - self.last_yaw

        # 180度を跨ぐ場合の処理
        if delta_yaw > 180:
            delta_yaw -= 360  # 例: 350° - 10° = 340° → -20°
        elif delta_yaw < -180:
            delta_yaw += 360  # 例: 10° - 350° = -340° → 20°

        # 累積ヨー差分に加算
        self.yaw_diff += delta_yaw
        self.last_yaw = current_yaw

        target_roll = 0
        target_pitch = 0
        target_throttle = int(self.current_autopilot_params.get('autopilot_throttle', 1300))

        if self.active_mission_mode == 1: # Horizontal Turn
            target_roll = self.current_autopilot_params.get('bank_angle', 20)
            self.alt_pid.setpoint = self.mission_start_altitude
            horizontal_target = self.current_autopilot_params.get('horizontal_turn_target', 760)
            if abs(self.yaw_diff) > horizontal_target:
                self.mission_status_label.setText("ミッション: 水平旋回 成功")
        elif self.active_mission_mode == 2: # Ascending Turn
            # 上昇旋回時は負のバンク角（水平旋回と逆方向）
            target_roll = -self.current_autopilot_params.get('bank_angle', 20)
            ascending_target = abs(self.current_autopilot_params.get('ascending_turn_target1', -760))
            if abs(self.yaw_diff) > ascending_target:
                self.mission_status_label.setText("ミッション: 上昇旋回 成功")
        elif self.active_mission_mode == 3: # Figure-8 Turn
            # 八の字旋回：右旋回から入り、目標角到達で左バンクに切り替え
            self.alt_pid.setpoint = self.mission_start_altitude

            right_target = self.current_autopilot_params.get('figure8_right_target', 300)
            left_target = self.current_autopilot_params.get('figure8_left_target', -320)
            bank_angle = self.current_autopilot_params.get('bank_angle', 20)

            if not self.figure8_completed:
                if self.figure8_phase == 0:  # 右旋回フェーズ
                    target_roll = bank_angle  # 正のバンク角（右バンク）
                    if self.yaw_diff >= right_target:  # 右旋回目標角に到達
                        self.figure8_phase = 1  # 左旋回フェーズに切り替え
                        print(f"八の字旋回: 左旋回フェーズに切り替え (yaw_diff: {self.yaw_diff:.1f}°)")

                elif self.figure8_phase == 1:  # 左旋回フェーズ
                    target_roll = -bank_angle  # 負のバンク角（左バンク）
                    # 左旋回目標角は負の値なので、yaw_diffがleft_target以下になったら完了
                    if self.yaw_diff <= left_target:  # 左旋回目標角に到達
                        self.figure8_completed = True
                        self.mission_status_label.setText("ミッション: 八の字旋回 成功")
                        print(f"八の字旋回: 完了 (yaw_diff: {self.yaw_diff:.1f}°)")
            else:
                # ミッション完了後は水平飛行
                target_roll = 0

        target_pitch_from_alt = self.alt_pid.update(current_alt, dt)
        final_target_pitch = target_pitch_from_alt + target_pitch
        self.pitch_pid.setpoint = final_target_pitch
        elev_out = self.pitch_pid.update(current_pitch, dt)

        self.roll_pid.setpoint = target_roll
        ail_out = self.roll_pid.update(current_roll, dt)

        # エルロン→ラダーミキシング
        aileron_rudder_mix_coef = self.current_autopilot_params.get('aileron_rudder_mix', 0.3)
        rudd_out = ail_out * aileron_rudder_mix_coef

        """
        # デバッグ情報（必要に応じて）
        if abs(ail_out) > 0.1:  # エルロン操作がある時のみログ出力
            print(f"ミキシング: エルロン={ail_out:.3f}, ラダー={rudd_out:.3f}, 係数={aileron_rudder_mix_coef}")
        """

        # 目標角度を2D表示に送信
        self.roll_widget.set_target_angle(target_roll)
        self.pitch_widget.set_target_angle(final_target_pitch)

        # ヨーの目標角度はミッション成功判定角度
        if self.active_mission_mode == 1:  # 水平旋回
            horizontal_target = self.current_autopilot_params.get('horizontal_turn_target', 760)
            target_yaw_display = self.mission_start_yaw + horizontal_target
        elif self.active_mission_mode == 2:  # 上昇旋回
            ascending_target = self.current_autopilot_params.get('ascending_turn_target1', -760)
            target_yaw_display = self.mission_start_yaw + ascending_target
        elif self.active_mission_mode == 3:  # 八の字旋回
            if self.figure8_phase == 0:  # 右旋回フェーズ
                figure8_target = self.current_autopilot_params.get('figure8_right_target', 300)
                target_yaw_display = self.mission_start_yaw + figure8_target
            else:  # 左旋回フェーズ
                figure8_target = self.current_autopilot_params.get('figure8_left_target', -320)
                target_yaw_display = self.mission_start_yaw + figure8_target
        else:
            # 手動操縦時は現在の累積角度
            target_yaw_display = self.mission_start_yaw + self.yaw_diff

        self.yaw_widget.set_target_angle(target_yaw_display)

        # 自動操縦時の操作量をスティック表示に送信
        # スロットルは正規化する必要がある
        thro_norm = self.normalize_value(target_throttle, **self.RC_RANGES['thro'])

        # 左スティック（ラダー、エレベーター）
        self.left_stick.set_autopilot_position(rudd_out, -elev_out)  # エレベーターは反転

        # 右スティック（エルロン、スロットル）
        self.right_stick.set_autopilot_position(ail_out, thro_norm)

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
            # データ送信フォーマット: エルロン,エレベータ,ラダー,スロットル,ミッションモード,AUXチャンネル
            command_str = f"{int(commands['ail'])},{int(commands['elev'])},{int(commands['rudd'])},{int(commands['thro'])},{self.active_mission_mode},1500\n"
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

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TelemetryApp()
    window.showMaximized()
    sys.exit(app.exec())
