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
import pyvista as pv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QPushButton, QLabel, QGridLayout, QLineEdit
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF, QImage, QPixmap
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QThread, Signal, Slot

# QtInteractorは必要時にのみインポート
QtInteractor = None

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

# --- Main Application Window ---
class TelemetryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RC Telemetry Display (PySide6)")
        self.setGeometry(100, 100, 1600, 900)

        self.serial_connection = None
        self.is_connected = False
        self.data_queue = queue.Queue()
        self.video_thread = None

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

        # 3D表示の初期化を遅延（QApplicationが確実に存在してから）
        self.plotter = None
        self.has_3d_display = False
        self.plane_actor = None

        self._setup_video_display()

        # QTimerを使って3D表示を遅延初期化
        QTimer.singleShot(100, self._delayed_3d_setup)

        self.update_com_ports()

        # テレメトリータイマー（高頻度でデータ処理）
        self.telemetry_timer = QTimer(self)
        self.telemetry_timer.timeout.connect(self.process_serial_queue)
        self.telemetry_timer.start(50)  # 20Hz: UI表示用

        # 3D表示用の姿勢データバッファ
        self.latest_attitude = {'roll': 0, 'pitch': 0, 'yaw': 0}
        self.enable_3d_display = self.has_3d_display

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

    def _create_video_group(self):
        group = QGroupBox("ビデオストリーム")
        layout = QGridLayout(group)
        self.video_url_input = QLineEdit("http://192.168.0.10:8080/video")
        self.video_toggle_button = QPushButton("配信開始")
        self.video_toggle_button.clicked.connect(self.toggle_video_stream)
        layout.addWidget(QLabel("URL:"), 0, 0)
        layout.addWidget(self.video_url_input, 0, 1)
        layout.addWidget(self.video_toggle_button, 1, 0, 1, 2)
        return group

    def _setup_video_display(self):
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0,0,0,0)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setText("ビデオストリーム停止中")
        self.video_status_label = QLabel("ステータス: 非アクティブ")
        self.video_status_label.setFixedHeight(20)
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.video_status_label)
        self.right_layout.addWidget(video_container, 1)

    def _delayed_3d_setup(self):
        """3D表示の遅延初期化"""
        global QtInteractor
        try:
            # QtInteractorを必要時にのみインポート
            if QtInteractor is None:
                from pyvistaqt import QtInteractor

            # QApplicationが完全に初期化された後に3D表示を設定
            # 引数なしで初期化を試行
            self.plotter = QtInteractor()
            self.right_layout.addWidget(self.plotter.interactor, 2)
            self.has_3d_display = True
            print("3D表示の初期化に成功しました")

            # 3Dモデルをセットアップ
            self._setup_3d_model()

            # 3D描画タイマーを開始
            if hasattr(self, 'telemetry_timer'):  # テレメトリータイマーが既に存在する場合
                self.render_3d_timer = QTimer(self)
                self.render_3d_timer.timeout.connect(self.update_3d_display)
                self.render_3d_timer.start(150)  # 6.7Hz: 3D描画用

        except Exception as e:
            print(f"3D表示の初期化に失敗: {e}")
            # フォールバック：3D表示なしのラベル
            fallback_label = QLabel("3D表示は利用できません\n（PyVistaとPySide6の互換性問題）")
            fallback_label.setAlignment(Qt.AlignCenter)
            fallback_label.setStyleSheet("background-color: #2b2b2b; color: white; padding: 20px;")
            self.right_layout.addWidget(fallback_label, 2)
            self.plotter = None
            self.has_3d_display = False

    def _setup_3d_model(self):
        """3Dモデルのセットアップ（3D表示が利用可能な場合のみ）"""
        if not self.has_3d_display or not self.plotter:
            print("3D表示が利用できないため、3Dモデルの読み込みをスキップします")
            return

        try:
            # 軽量化されたSTLファイルを優先使用
            try:
                mesh = pv.read("planeLight_light.stl")
                print("軽量化STLファイルを使用します")
            except FileNotFoundError:
                mesh = pv.read("planeLight.stl")
                print("標準STLファイルを使用します")
        except FileNotFoundError:
            try:
                mesh = pv.read("plane.stl")
            except FileNotFoundError:
                mesh = pv.read("placeholder_cube.stl")
                print("プレースホルダーCubeを使用します")

        # メッシュ情報を表示
        print(f"メッシュ情報: 頂点数={mesh.n_points}, ポリゴン数={mesh.n_cells}")

        self.plane_actor = self.plotter.add_mesh(mesh, smooth_shading=True)
        self.plotter.view_isometric()
        self.plotter.add_axes()
        self.plotter.enable_zoom_style()

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
        return p.scaled(420, 320, Qt.KeepAspectRatio)

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
                if self.plane_actor:
                    self.plane_actor.SetOrientation(0, 0, 0)
                    self.plane_actor.RotateZ(yaw)
                    self.plane_actor.RotateY(pitch)
                    self.plane_actor.RotateX(roll)
                self.adi_widget.set_attitude(roll, pitch)
                self.altitude_label.setText(f"高度: {alt:.1f} m")
                self.heading_label.setText(f"方位: {yaw:.1f} °")
                self.update_aux_switches([aux1, aux2, aux3, aux4])
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
        for i, value in enumerate(aux_values):
            label = self.aux_labels[i]
            if value > 1100:
                label.setText(f"AUX{i+1}: ON")
                label.setStyleSheet(on_style)
            else:
                label.setText(f"AUX{i+1}: OFF")
                label.setStyleSheet(off_style)

    def toggle_3d_display(self, enabled):
        """3D表示の有効/無効を切り替え"""
        if not self.has_3d_display:
            return  # 3D表示が利用できない場合は何もしない

        self.enable_3d_display = enabled
        if enabled and hasattr(self, 'render_3d_timer'):
            self.render_3d_timer.start(150)
            if self.plotter:
                self.plotter.show()
        elif hasattr(self, 'render_3d_timer'):
            self.render_3d_timer.stop()
            if self.plotter:
                self.plotter.hide()

    def update_3d_display(self):
        """3D表示を低頻度で更新"""
        if not self.has_3d_display or not self.enable_3d_display or not self.plotter:
            return

        if hasattr(self, 'plane_actor') and self.plane_actor:
            roll = self.latest_attitude['roll']
            pitch = self.latest_attitude['pitch']
            yaw = self.latest_attitude['yaw']
            try:
                self.plane_actor.SetOrientation(roll, pitch, yaw)
            except Exception as e:
                print(f"3D表示更新エラー: {e}")

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        self.is_connected = False
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        self.udp_socket.close()
        # 3D表示が利用可能な場合のみplotterを閉じる
        if self.has_3d_display and self.plotter:
            try:
                self.plotter.close()
            except Exception as e:
                print(f"3D表示の終了処理でエラー: {e}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # QApplication作成後にPyVistaテーマを設定
    try:
        pv.set_plot_theme("dark")
    except Exception as e:
        print(f"PyVistaテーマ設定エラー: {e}")

    window = TelemetryApp()
    window.show()
    sys.exit(app.exec())
