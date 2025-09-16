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
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF, QImage, QPixmap, QTransform
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QThread, Signal, Slot

# QtInteractorは必要時にのみインポート
# QtInteractor = None

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

# --- Custom UI Widgets (Omitted for brevity) ---
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC Telemetry Display (PySide6)")
        self.setGeometry(100, 100, 1600, 900)

        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.video_thread = None
        self.pid_gain_edits = {}
        self.current_pid_gains = {}
        self.latest_attitude = {'roll': 0, 'pitch': 0, 'yaw': 0}
        self.latest_aux_values = []

        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.udp_broadcast_address = ('<broadcast>', 12345)

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
        self.load_pid_gains()

        top_right_layout.addWidget(video_widget)
        top_right_layout.addWidget(pid_panel)
        top_right_layout.addStretch(1)

        self.right_layout.addWidget(top_right_widget, 1)

        self._setup_2d_attitude_display(right_panel)

        self.update_com_ports()

        # --- Timers ---
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.process_serial_queue)
        self.telemetry_timer.start(50)  # 20Hz

        self.control_data_timer = QTimer(self)
        self.control_data_timer.timeout.connect(self.send_control_data)
        self.control_data_timer.start(20) # 50Hz

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
        self.mission_status_label = QLabel("ミッション: なし")
        self.mission_status_label.setAlignment(Qt.AlignCenter)
        self.mission_status_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFD700; padding: 5px;")

        main_layout.addWidget(self.adi_widget, 0, Qt.AlignCenter)
        main_layout.addWidget(self.altitude_label)
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

    def update_and_save_pid_gains(self):
        """PIDゲインを更新し、ファイルに保存する"""
        gains_to_save = {}
        try:
            for _, key, _ in self.GAINS_TO_TUNE:
                value_str = self.pid_gain_edits[key].text()
                self.current_pid_gains[key] = float(value_str) # Update internal state
                gains_to_save[key] = value_str # Save as string
            
            with open("coef.txt", "w") as f:
                json.dump(gains_to_save, f, indent=4)
            
            print(f"PID gains updated and saved: {self.current_pid_gains}")
        except ValueError as e:
            print(f"PIDゲインの値が不正です: {e}")
        except Exception as e:
            print(f"PIDゲインの保存中にエラーが発生しました: {e}")

    def load_pid_gains(self):
        """coef.txtからPIDゲインを読み込んで適用する"""
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
        """2D姿勢表示ウィジェットをセットアップ"""
        container = QGroupBox("2D姿勢表示")
        layout = QHBoxLayout(container)

        self.roll_widget = Attitude2DWidget("2NAa_roll.png")
        self.pitch_widget = Attitude2DWidget("2NAa_pitch.png")
        self.yaw_widget = Attitude2DWidget("2NAa_yaw.png")

        layout.addWidget(self.roll_widget)
        layout.addWidget(self.pitch_widget)
        layout.addWidget(self.yaw_widget)

        self.right_layout.addWidget(container, 2)

        self.attitude_2d_timer = QTimer(self)
        self.attitude_2d_timer.timeout.connect(self.update_2d_attitude)
        self.attitude_2d_timer.start(100) # 10Hz

    def get_mission_mode(self):
        """AUXチャンネルの状態から現在のミッションモードを決定する"""
        # 0:手動, 1:水平旋回(AUX2), 2:八の字(AUX3), 3:上昇旋回(AUX4), 4:自動離着陸(AUX1)
        if len(self.latest_aux_values) == 4:
            if self.latest_aux_values[1] > 1100: return 1
            elif self.latest_aux_values[2] > 1100: return 2
            elif self.latest_aux_values[3] > 1100: return 3
            elif self.latest_aux_values[0] > 1100: return 4
        return 0 # Manual

    def send_control_data(self):
        """PC側で計算した制御データをシリアルポートに送信する"""
        if not self.is_connected or not self.serial_connection.is_open:
            return

        # 操舵量は後で計算するため、現在は仮の値を送信
        aileron_cmd = 1500
        elevator_cmd = 1500
        rudder_cmd = 1500
        throttle_cmd = 1000

        mission_mode = self.get_mission_mode()
        
        # AUXチャンネルの操作 (bitmask)
        aux_bitmask = 0
        if len(self.latest_aux_values) == 4:
            for i, val in enumerate(self.latest_aux_values):
                if val > 1100:
                    aux_bitmask |= (1 << i)

        try:
            data_str = f"{aileron_cmd},{elevator_cmd},{rudder_cmd},{throttle_cmd},{mission_mode},{aux_bitmask}\n"
            self.serial_connection.write(data_str.encode('utf-8'))
        except serial.SerialException as e:
            print(f"Error sending control data: {e}")
            self.toggle_connection() # Assume connection is lost
        except Exception as e:
            print(f"An unexpected error occurred while sending data: {e}")

    def update_2d_attitude(self):
        """2D姿勢表示を更新"""
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
            try:
                self.udp_socket.sendto(line.encode('utf-8'), self.udp_broadcast_address)
            except Exception as e:
                print(f"UDP送信エラー: {e}")
            parts = [float(p) for p in line.split(',')]
            if len(parts) == 18:
                roll, pitch, yaw, alt, ail, elev, thro, rudd, aux1, aux2, aux3, aux4, *_ = parts
                self.latest_attitude = {'roll': roll, 'pitch': pitch, 'yaw': yaw}
                self.latest_aux_values = [aux1, aux2, aux3, aux4]

                self.adi_widget.set_attitude(roll, pitch)
                self.altitude_label.setText(f"高度: {alt:.1f} m")
                self.heading_label.setText(f"方位: {yaw:.1f} °")
                self.update_aux_switches(self.latest_aux_values)
                rud_norm = self.normalize_symmetrical(rudd, 830, 1148, 1500)
                ele_norm = self.normalize_symmetrical(elev, 800, 966, 1070)
                rud_norm = -rud_norm
                ele_norm = -ele_norm
                self.left_stick.set_position(rud_norm, ele_norm)
                self.left_stick_label.setText(f"R: {int(rudd)}, E: {int(elev)}")
                ail_norm = self.normalize_symmetrical(ail, 560, 1164, 1750)
                thr_norm = self.normalize_value(thro, 360, 1590)
                self.right_stick.set_position(ail_norm, thr_norm)
                self.right_stick_label.setText(f"A: {int(ail)}, T: {int(thro)}")
        except (ValueError, IndexError) as e:
            print(f"データ解析エラー: {e} - {line}")

    def update_aux_switches(self, aux_values):
        on_style = "background-color: #28a745; color: white; padding: 5px; border-radius: 3px;"
        off_style = "background-color: #555; color: white; padding: 5px; border-radius: 3px;"

        mission_map = {
            1: "水平旋回",
            2: "八の字飛行",
            3: "上昇旋回",
            4: "自動離着陸"
        }
        mission_mode = self.get_mission_mode()
        mission_text = mission_map.get(mission_mode, "手動操縦")

        self.mission_status_label.setText(f"ミッション: {mission_text}")

        if len(aux_values) == 4:
            for i, value in enumerate(aux_values):
                label = self.aux_labels[i]
                is_on = value > 1100
                if is_on:
                    label.setText(f"AUX{i+1}: ON")
                    label.setStyleSheet(on_style)
                else:
                    label.setText(f"AUX{i+1}: OFF")
                    label.setStyleSheet(off_style)

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        self.is_connected = False
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.udp_socket.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = TelemetryApp()
    window.showMaximized()
    sys.exit(app.exec())
