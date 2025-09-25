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
import os
import glob
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QPushButton, QLabel, QGridLayout, QLineEdit, QCheckBox,
    QTabWidget, QDoubleSpinBox, QFileDialog
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
        try:
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
        finally:
            painter.end()

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
        try:
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
        finally:
            painter.end()

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
        try:
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

            # 現在の手動操縦入力位置（シアン）
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
        finally:
            painter.end()

class Attitude2DWidget(QWidget):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)

        # Get absolute path to ensure image loading works regardless of working directory
        if not os.path.isabs(image_path):
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(script_dir, image_path)

        self.pixmap = QPixmap(image_path)
        if self.pixmap.isNull():
            print(f"画像の読み込みに失敗: {image_path}")
            print(f"ファイル存在確認: {os.path.exists(image_path)}")
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
        try:
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
        finally:
            painter.end()

# --- Flight State Graph Widget ---
class FlightStateGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)  # Reduced size to fit better

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 8))  # Smaller figure size
        self.canvas = FigureCanvas(self.figure)        # Create layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Data storage
        self.log_data = []

        # Initialize empty plots
        self.clear_plots()

    def clear_plots(self):
        """Clear all plots and set up empty axes"""
        self.figure.clear()

        # Create 4x1 subplot layout (4 rows, 1 column)
        self.ax1 = self.figure.add_subplot(4, 1, 1)  # Altitude
        self.ax2 = self.figure.add_subplot(4, 1, 2)  # Yaw angle
        self.ax3 = self.figure.add_subplot(4, 1, 3)  # Throttle
        self.ax4 = self.figure.add_subplot(4, 1, 4)  # AUX1

        # Set titles and labels in English
        self.ax1.set_title('Altitude Change', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Altitude (mm)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(labelbottom=False)  # Hide x-axis labels for top plots

        self.ax2.set_title('Yaw Displacement (from start)', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Yaw Displacement (deg)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(labelbottom=False)  # Hide x-axis labels for middle plots

        self.ax3.set_title('Throttle Value', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Throttle')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.tick_params(labelbottom=False)  # Hide x-axis labels for middle plots

        self.ax4.set_title('Material Drop Timing (AUX1)', fontsize=12, fontweight='bold')
        self.ax4.set_xlabel('Time (sec)')  # Only show x-axis label on bottom plot
        self.ax4.set_ylabel('AUX1 Value')
        self.ax4.grid(True, alpha=0.3)

        # Adjust layout with proper margins for y-axis labels
        self.figure.subplots_adjust(left=0.9, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
        self.figure.tight_layout(pad=1.0)  # Reduced padding for better space usage
        self.canvas.draw()

    def update_plots(self, log_data):
        """Update plots with new log data"""
        if not log_data:
            self.clear_plots()
            return

        self.log_data = log_data

        # Extract data for plotting
        times = [point['elapsed_time'] for point in log_data]
        altitudes = [point['altitude'] for point in log_data]
        yaw_angles = [point['yaw'] for point in log_data]
        throttles = [point['throttle'] for point in log_data]
        aux1_values = [point['aux1'] for point in log_data]

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        # Plot 1: Altitude
        self.ax1.plot(times, altitudes, 'b-', linewidth=2, label='Altitude')
        self.ax1.set_title('Altitude Change', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Altitude (mm)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(labelbottom=False)  # Hide x-axis labels for top plots
        # Set altitude axis with proper scale
        if altitudes:
            alt_min, alt_max = min(altitudes), max(altitudes)
            alt_range = alt_max - alt_min
            if alt_range > 0:
                self.ax1.set_ylim(alt_min - alt_range*0.1, alt_max + alt_range*0.1)

        # Plot 2: Yaw angle
        self.ax2.plot(times, yaw_angles, 'r-', linewidth=2, label='Yaw Angle')
        self.ax2.set_title('Yaw Displacement (from start)', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Yaw Displacement (deg)')
        self.ax2.set_ylim(-180, 180)  # Fixed scale for yaw displacement (-180 to +180 degrees)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(labelbottom=False)  # Hide x-axis labels for middle plots
        # Add major ticks every 45 degrees
        self.ax2.set_yticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

        # Plot 3: Throttle
        self.ax3.plot(times, throttles, 'g-', linewidth=2, label='Throttle')
        self.ax3.set_title('Throttle Value', fontsize=12, fontweight='bold')
        self.ax3.set_ylabel('Throttle')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.tick_params(labelbottom=False)  # Hide x-axis labels for middle plots
        # Set throttle axis with custom range
        self.ax3.set_ylim(0, 1700)  # Custom throttle range
        self.ax3.set_yticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1600])

        # Plot 4: AUX1
        self.ax4.plot(times, aux1_values, 'm-', linewidth=2, label='AUX1')
        self.ax4.set_title('Material Drop Timing (AUX1)', fontsize=12, fontweight='bold')
        self.ax4.set_xlabel('Time (sec)')  # Only show x-axis label on bottom plot
        self.ax4.set_ylabel('AUX1 Value')
        self.ax4.grid(True, alpha=0.3)
        # Set AUX1 axis with extended range
        self.ax4.set_ylim(0, 2000)  # Extended range from 0 to 2000
        self.ax4.set_yticks([0, 400, 800, 1200, 1600, 2000])

        # Set common time axis for all plots
        if times:
            time_max = max(times)
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.set_xlim(0, time_max + time_max*0.05)
                # Add time ticks every 5 seconds or appropriate interval
                if time_max > 0:
                    tick_interval = max(1, int(time_max / 10))  # About 10 ticks
                    ax.set_xticks(range(0, int(time_max) + tick_interval, tick_interval))

        # Adjust layout and refresh
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

# --- Position Visualization Widgets for Auto Landing ---
class PositionVisualizationWidget(QWidget):
    def __init__(self, view_type="XY", parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.view_type = view_type  # "XY" or "ZY"
        self.aircraft_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # Default marker field positions (manual setting only, separate from image coordinates)
        self.aruco_markers = {
            1: {'x': -3.0, 'y': 6.0, 'z': 0.0, 'image_x': 0, 'image_y': 0, 'size': 0, 'detected': False},
            2: {'x': 0.0, 'y': 0.0, 'z': 0.0, 'image_x': 0, 'image_y': 0, 'size': 0, 'detected': False},
            3: {'x': 3.0, 'y': 6.0, 'z': 0.0, 'image_x': 0, 'image_y': 0, 'size': 0, 'detected': False}
        }
        self.scale = 10.0  # pixels per meter

        # Load marker positions from file if exists
        self.load_marker_positions()

    def set_aircraft_position(self, x, y, z):
        self.aircraft_pos = {'x': x, 'y': y, 'z': z}
        self.update()

    def set_marker_detection(self, marker_id, detected):
        if marker_id in self.aruco_markers:
            self.aruco_markers[marker_id]['detected'] = detected
        self.update()

    def update_marker_data(self, marker_id, size, x, y):
        """Update marker data with real-time image coordinate values (from camera)"""
        if marker_id in self.aruco_markers:
            # Store image coordinates separately from field coordinates
            self.aruco_markers[marker_id]['size'] = size
            self.aruco_markers[marker_id]['image_x'] = x  # Image coordinates from camera
            self.aruco_markers[marker_id]['image_y'] = y  # Image coordinates from camera
            self.aruco_markers[marker_id]['detected'] = size > 0
        self.update()

    def set_marker_position(self, marker_id, x, y, z=0.0):
        """Set marker field position (for flight control layout, manual setting only)"""
        if marker_id in self.aruco_markers:
            # Field coordinates for landing control (separate from image coordinates)
            self.aruco_markers[marker_id]['x'] = x  # Field coordinates in meters
            self.aruco_markers[marker_id]['y'] = y  # Field coordinates in meters
            self.aruco_markers[marker_id]['z'] = z  # Field coordinates in meters
        self.update()

    def get_marker_positions(self):
        """Get all marker positions"""
        positions = {}
        for marker_id, marker in self.aruco_markers.items():
            positions[marker_id] = {
                'x': marker['x'],
                'y': marker['y'],
                'z': marker['z']
            }
        return positions

    def load_marker_positions(self):
        """Load marker positions from file"""
        try:
            if os.path.exists('marker_positions.txt'):
                with open('marker_positions.txt', 'r') as f:
                    for line in f:
                        line = line.strip()
                        if '=' in line:
                            key, value = line.split('=', 1)
                            try:
                                # Parse marker_X_Y=value format
                                parts = key.split('_')
                                if len(parts) == 3 and parts[0] == 'marker':
                                    marker_id = int(parts[1])
                                    coord = parts[2]  # x, y, or z
                                    if marker_id in self.aruco_markers and coord in ['x', 'y', 'z']:
                                        self.aruco_markers[marker_id][coord] = float(value)
                            except (ValueError, IndexError):
                                continue
                print("マーカー位置を読み込みました")
        except Exception as e:
            print(f"マーカー位置読み込みエラー: {e}")

    def save_marker_positions(self):
        """Save marker positions to file"""
        try:
            with open('marker_positions.txt', 'w') as f:
                f.write("# ArUco Marker Positions (Field Layout)\n")
                f.write("# Format: marker_ID_coordinate=value\n\n")
                for marker_id, marker in self.aruco_markers.items():
                    f.write(f"marker_{marker_id}_x={marker['x']}\n")
                    f.write(f"marker_{marker_id}_y={marker['y']}\n")
                    f.write(f"marker_{marker_id}_z={marker['z']}\n")
            print("マーカー位置を保存しました")
            return True
        except Exception as e:
            print(f"マーカー位置保存エラー: {e}")
            return False

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            painter.fillRect(self.rect(), QColor("#2b2b2b"))

            # Get widget center
            center_x = self.width() / 2
            center_y = self.height() / 2

            # Draw coordinate grid
            painter.setPen(QPen(QColor("gray"), 1, Qt.DashLine))
            for i in range(-20, 21, 5):
                # Vertical lines
                x_pos = center_x + i * self.scale
                if 0 <= x_pos <= self.width():
                    painter.drawLine(x_pos, 0, x_pos, self.height())

                # Horizontal lines
                y_pos = center_y - i * self.scale
                if 0 <= y_pos <= self.height():
                    painter.drawLine(0, y_pos, self.width(), y_pos)

            # Draw axes
            painter.setPen(QPen(QColor("white"), 2))
            painter.drawLine(center_x, 0, center_x, self.height())  # Y axis
            painter.drawLine(0, center_y, self.width(), center_y)   # X axis

            # Draw runway representation
            painter.setPen(QPen(QColor("yellow"), 3))
            if self.view_type == "XY":
                # Runway from Y=0 to Y=33
                runway_start_y = center_y
                runway_end_y = center_y + 33 * self.scale  # Y軸正負を反転
                painter.drawLine(center_x - 10, runway_start_y, center_x + 10, runway_start_y)
                painter.drawLine(center_x - 10, runway_end_y, center_x + 10, runway_end_y)
                painter.drawLine(center_x, runway_start_y, center_x, runway_end_y)

            # Draw ArUco markers (both configured positions and real-time detections)
            for marker_id, marker in self.aruco_markers.items():
                if self.view_type == "XY":
                    # Use configured field positions for markers
                    marker_x = center_x + float(marker['x']) * self.scale
                    marker_y = center_y + float(marker['y']) * self.scale  # Y軸正負を反転
                else:  # ZY view
                    marker_x = center_x + float(marker['y']) * self.scale
                    marker_y = center_y - float(marker.get('z', 0)) * self.scale  # Z軸は上が正（変更なし）

                # Color based on detection status
                is_detected = marker.get('detected', False) or marker.get('size', 0) > 0
                color = QColor("lime") if is_detected else QColor("orange")

                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(color))

                # Draw marker as square (more representative of ArUco markers)
                marker_size = 12
                marker_rect = QRectF(marker_x - marker_size/2, marker_y - marker_size/2, marker_size, marker_size)
                painter.drawRect(marker_rect)

                # Draw marker ID and position info with units
                painter.setPen(QPen(QColor("white"), 1))
                if self.view_type == "XY":
                    marker_info = f"ID{marker_id}\n({marker['x']:.1f}m, {marker['y']:.1f}m)"
                    if is_detected:
                        marker_info += f"\nサイズ: {marker.get('size', 0):.0f}px"
                else:
                    marker_info = f"ID{marker_id}\n({marker['y']:.1f}m, {marker.get('z', 0):.1f}m)"
                    if is_detected:
                        marker_info += f"\nサイズ: {marker.get('size', 0):.0f}px"

                painter.drawText(marker_x + marker_size/2 + 5, marker_y - marker_size/2, marker_info)

            # Draw aircraft position
            if self.view_type == "XY":
                aircraft_x = center_x + self.aircraft_pos['x'] * self.scale
                aircraft_y = center_y + self.aircraft_pos['y'] * self.scale  # Y軸正負を反転
            else:  # ZY view
                aircraft_x = center_x + self.aircraft_pos['y'] * self.scale
                aircraft_y = center_y - self.aircraft_pos['z'] * self.scale  # Z軸は上が正（変更なし）

            painter.setPen(QPen(QColor("cyan"), 3))
            painter.setBrush(QColor("cyan"))
            # Draw aircraft as triangle
            aircraft_size = 8
            triangle = QPolygonF([
                QPointF(aircraft_x, aircraft_y - aircraft_size),
                QPointF(aircraft_x - aircraft_size/2, aircraft_y + aircraft_size/2),
                QPointF(aircraft_x + aircraft_size/2, aircraft_y + aircraft_size/2)
            ])
            painter.drawPolygon(triangle)

            # Draw labels
            painter.setPen(QPen(QColor("white"), 1))
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)

            title = f"{self.view_type}平面表示"
            painter.drawText(10, 20, title)

            # Draw axis labels
            if self.view_type == "XY":
                painter.drawText(self.width() - 30, center_y - 10, "X")
                painter.drawText(center_x + 10, self.height() - 15, "Y")  # Y軸ラベルを下に移動
            else:
                painter.drawText(self.width() - 30, center_y - 10, "Y")
                painter.drawText(center_x + 10, 15, "Z (高度)")  # Z軸を高度として明記
        finally:
            painter.end()

# --- Calibration Graph Widget ---
class CalibrationGraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 400)
        self.calibration_data = {}
        self.selected_marker = "全マーカー"

    def set_calibration_data(self, data):
        """Set calibration data for display"""
        self.calibration_data = data
        self.update()

    def set_selected_marker(self, marker_text):
        """Set which marker to display"""
        self.selected_marker = marker_text
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Background
            painter.fillRect(self.rect(), QColor("#2b2b2b"))

            # Draw graph
            self._draw_graph(painter)
        finally:
            painter.end()

    def _draw_graph(self, painter):
        """Draw calibration graph"""
        if not self.calibration_data:
            # No data message
            painter.setPen(QPen(QColor("white"), 1))
            center_point = QPointF(self.rect().center())  # QPointをQPointFに変換
            painter.drawText(center_point + QPointF(-100, 0), "キャリブレーションデータがありません")
            return

        # Graph area
        margin = 50
        graph_rect = QRectF(margin, margin, self.width() - 2 * margin, self.height() - 2 * margin)

        # Draw graph background
        painter.setPen(QPen(QColor("#404040"), 1))
        painter.setBrush(QBrush(QColor("#1e1e1e")))  # Slightly lighter background for graph area
        painter.drawRect(graph_rect)

        # Draw axes
        painter.setPen(QPen(QColor("white"), 2))
        painter.drawLine(graph_rect.bottomLeft(), graph_rect.bottomRight())  # X axis
        painter.drawLine(graph_rect.bottomLeft(), graph_rect.topLeft())      # Y axis

        # Determine data range
        all_distances = []
        all_sizes = []

        markers_to_show = []
        if self.selected_marker == "全マーカー":
            markers_to_show = [1, 2, 3]
        else:
            marker_num = int(self.selected_marker.split()[1])
            markers_to_show = [marker_num]

        for marker_id in markers_to_show:
            if marker_id in self.calibration_data:
                for point in self.calibration_data[marker_id]:
                    if 'distance' in point and 'size' in point:
                        all_distances.append(point['distance'])
                        all_sizes.append(point['size'])

        if not all_distances:
            return

        min_dist, max_dist = min(all_distances), max(all_distances)
        min_size, max_size = min(all_sizes), max(all_sizes)

        # Add some padding
        dist_range = max_dist - min_dist if max_dist > min_dist else 1
        size_range = max_size - min_size if max_size > min_size else 1

        min_dist -= dist_range * 0.1
        max_dist += dist_range * 0.1
        min_size -= size_range * 0.1
        max_size += size_range * 0.1

        # Draw grid lines and tick marks
        self._draw_grid_and_ticks(painter, graph_rect, min_dist, max_dist, min_size, max_size)

        # Draw data points and lines for each marker
        colors = [QColor("red"), QColor("green"), QColor("blue")]

        for i, marker_id in enumerate(markers_to_show):
            if marker_id not in self.calibration_data:
                continue

            color = colors[i % len(colors)]
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))

            points = []
            for point in self.calibration_data[marker_id]:
                if 'distance' in point and 'size' in point:
                    x = graph_rect.left() + (point['distance'] - min_dist) / (max_dist - min_dist) * graph_rect.width()
                    y = graph_rect.bottom() - (point['size'] - min_size) / (max_size - min_size) * graph_rect.height()
                    points.append(QPointF(x, y))

                    # Draw point with white border for better visibility
                    painter.setPen(QPen(QColor("white"), 1))  # White border
                    painter.setBrush(QBrush(color))  # Colored fill
                    painter.drawEllipse(QPointF(x, y), 6, 6)  # Slightly larger point

            # Draw connecting lines
            if len(points) > 1:
                painter.setPen(QPen(color, 2, Qt.DashLine))  # Thicker dashed line
                painter.setBrush(QBrush())  # No fill for lines
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])

        # Draw labels
        painter.setPen(QPen(QColor("white"), 1))
        painter.drawText(graph_rect.bottomRight() + QPointF(-50, 20), "距離 (m)")
        painter.drawText(graph_rect.topLeft() + QPointF(-40, -10), "サイズ (px)")

        # Draw legend
        legend_y = graph_rect.top() + 20
        for i, marker_id in enumerate(markers_to_show):
            if marker_id in self.calibration_data:
                color = colors[i % len(colors)]
                painter.setPen(QPen(color, 2))
                painter.drawLine(graph_rect.right() - 120, legend_y + i * 20,
                               graph_rect.right() - 100, legend_y + i * 20)
                painter.setPen(QPen(QColor("white"), 1))
                painter.drawText(graph_rect.right() - 95, legend_y + i * 20 + 5, f"マーカー {marker_id}")

    def _draw_grid_and_ticks(self, painter, graph_rect, min_dist, max_dist, min_size, max_size):
        """Draw grid lines and tick marks with labels"""

        # Grid line style
        grid_pen = QPen(QColor("#404040"), 1)  # Dark gray grid
        painter.setPen(grid_pen)

        # Tick mark style
        tick_pen = QPen(QColor("white"), 1)

        # Calculate nice tick intervals
        dist_range = max_dist - min_dist
        size_range = max_size - min_size

        # Distance (X-axis) ticks - aim for about 5-8 divisions
        if dist_range > 0:
            dist_step = self._calculate_nice_step(dist_range, 6)
            dist_start = math.ceil(min_dist / dist_step) * dist_step

            # Draw vertical grid lines and X-axis ticks
            dist = dist_start
            while dist <= max_dist:
                x_pos = graph_rect.left() + (dist - min_dist) / (max_dist - min_dist) * graph_rect.width()

                # Grid line
                painter.setPen(grid_pen)
                painter.drawLine(x_pos, graph_rect.top(), x_pos, graph_rect.bottom())

                # Tick mark
                painter.setPen(tick_pen)
                painter.drawLine(x_pos, graph_rect.bottom(), x_pos, graph_rect.bottom() + 5)

                # Tick label
                painter.drawText(QRectF(x_pos - 20, graph_rect.bottom() + 8, 40, 20),
                               Qt.AlignCenter, f"{dist:.1f}")

                dist += dist_step

        # Size (Y-axis) ticks - aim for about 5-8 divisions
        if size_range > 0:
            size_step = self._calculate_nice_step(size_range, 6)
            size_start = math.ceil(min_size / size_step) * size_step

            # Draw horizontal grid lines and Y-axis ticks
            size = size_start
            while size <= max_size:
                y_pos = graph_rect.bottom() - (size - min_size) / (max_size - min_size) * graph_rect.height()

                # Grid line
                painter.setPen(grid_pen)
                painter.drawLine(graph_rect.left(), y_pos, graph_rect.right(), y_pos)

                # Tick mark
                painter.setPen(tick_pen)
                painter.drawLine(graph_rect.left() - 5, y_pos, graph_rect.left(), y_pos)

                # Tick label
                painter.drawText(QRectF(graph_rect.left() - 45, y_pos - 10, 35, 20),
                               Qt.AlignRight | Qt.AlignVCenter, f"{size:.0f}")

                size += size_step

    def _calculate_nice_step(self, range_val, target_divisions):
        """Calculate a nice step size for grid divisions"""
        rough_step = range_val / target_divisions
        magnitude = math.pow(10, math.floor(math.log10(rough_step)))

        # Normalize to 1-10 range
        normalized = rough_step / magnitude

        # Choose nice step size
        if normalized <= 1.0:
            nice_step = 1.0
        elif normalized <= 2.0:
            nice_step = 2.0
        elif normalized <= 5.0:
            nice_step = 5.0
        else:
            nice_step = 10.0

        return nice_step * magnitude

# --- Main Application Window ---
class TelemetryApp(QMainWindow):
    GAINS_TO_TUNE = [
        ("Roll P", "roll_p", "0.1"), ("Roll I", "roll_i", "0.01"), ("Roll D", "roll_d", "0.05"),
        ("Pitch P", "pitch_p", "0.1"), ("Pitch I", "pitch_i", "0.01"), ("Pitch D", "pitch_d", "0.05"),
        ("Yaw P", "yaw_p", "0.2"), ("Yaw I", "yaw_i", "0.0"), ("Yaw D", "yaw_d", "0.0"),
        ("Altitude P", "alt_p", "0.1"), ("Altitude I", "alt_i", "0.02"), ("Altitude D", "alt_d", "0.05"),
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
        ("上昇時ピッチオフセット (度)", "ascending_pitch_offset", "5.0"),
        ("上昇スロットル基準", "ascending_throttle_base", "800.0"),
        ("LiDAR高度補正有効", "lidar_bank_correction_enable", "1"),
        ("LiDAR最大補正角度 (度)", "lidar_max_correction_angle", "45.0"),
        ("自動スロットル基準", "autopilot_throttle_base", "700.0"),
        ("高度制御スロットルゲイン", "altitude_throttle_gain", "20.0"),
        ("スロットル最小値", "throttle_min", "400.0"),
        ("スロットル最大値", "throttle_max", "1000.0"),
        ("エルロン→ラダーミキシング", "aileron_rudder_mix", "0.3"),
        ("右旋回ラダートリム", "rudder_trim_right", "0.1"),
        ("左旋回ラダートリム", "rudder_trim_left", "-0.1"),
    ]

    # 自動離着陸パラメータ
    AUTO_LANDING_PARAMS = [
        ("離陸スロットル", "takeoff_throttle", "1000.0"),
        ("投下前標準スロットル", "pre_drop_throttle", "700.0"),
        ("投下後標準スロットル", "post_drop_throttle", "650.0"),
        ("定常飛行高度 (m)", "steady_altitude", "1.5"),
        ("高度維持ゲイン", "altitude_gain", "50.0"),
        ("ラダー制御ゲイン", "rudder_gain", "0.5"),
        ("離陸フェーズ距離閾値 (m)", "takeoff_distance_threshold", "30.0"),
        ("投下フェーズ距離閾値 (m)", "drop_distance_threshold", "20.0"),
        ("定常フェーズ距離閾値 (m)", "steady_distance_threshold", "10.0"),
        ("着陸フェーズ距離閾値 (m)", "landing_distance_threshold", "5.0"),
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
        # 新しいヨー角追跡システム
        self.mission_total_rotation = 0.0  # ミッション開始からの総回転角（右回り正、左回り負）
        self.previous_yaw = 0.0  # 前回のヨー角
        # デバッグ用カウンター（定期的にyaw情報を出力するため）
        self.yaw_debug_counter = 0
        self.last_autopilot_commands = None
        self.previous_aux_values = [0, 0, 0, 0]
        self.latest_attitude = {'roll': 0, 'pitch': 0, 'yaw': 0, 'alt': 0}

        # --- ArUco Marker Data ---
        self.aruco_markers = {
            1: {'size': 0, 'id': 0, 'image_x': 0, 'image_y': 0, 'x': -3.0, 'y': 6.0, 'z': 0.0, 'detected': False},
            2: {'size': 0, 'id': 0, 'image_x': 0, 'image_y': 0, 'x': 0.0, 'y': 0.0, 'z': 0.0, 'detected': False},
            3: {'size': 0, 'id': 0, 'image_x': 0, 'image_y': 0, 'x': 3.0, 'y': 6.0, 'z': 0.0, 'detected': False}
        }

        # --- ArUco Marker Calibration Data ---
        self.marker_calibrations = {
            1: {'offset_angle': 0.0},
            2: {'offset_angle': 0.0},
            3: {'offset_angle': 0.0}
        }

        # --- ArUco Marker Distance Calibration Data ---
        self.marker_distance_calibrations = {}  # Initialize empty, will be loaded from file

        # --- Figure-8 Mission State ---
        self.figure8_phase = 0  # 0: 右旋回フェーズ, 1: 左旋回フェーズ
        self.figure8_completed = False
        self.figure8_phase_start_rotation = 0  # 左旋回フェーズ開始時の総回転角

        # --- Ascending Turn Mission State ---
        self.ascending_phase = 0  # 0: 2.5m左旋回2回, 1: 上昇中旋回, 2: 5m左旋回2回
        self.ascending_completed = False
        self.ascending_phase_start_rotation = 0  # フェーズ開始時の総回転角

        # --- Propeller Input Log System ---
        self.input_log_recording = False
        self.input_log_data = []  # {'timestamp': time, 'ail': val, 'elev': val, 'thro': val, 'rudd': val}
        self.input_log_start_time = 0
        self.input_log_last_record_time = 0  # 最後に記録した時刻（間隔制御用）
        self.input_log_replaying = False
        self.input_log_replay_start_time = 0
        self.input_log_replay_index = 0
        self.loaded_input_log = []
        self.input_log_interval = 0.1  # 記録・再現間隔（秒）- 調整可能

        # --- Auto Landing Log System ---
        self.auto_landing_log_recording = False
        self.auto_landing_log_data = []  # {'timestamp': time, 'altitude': val, 'yaw': val, 'throttle': val, 'aux1': val}
        self.auto_landing_log_start_time = 0
        self.auto_landing_log_last_record_time = 0
        self.auto_landing_log_replaying = False
        self.auto_landing_log_replay_start_time = 0
        self.auto_landing_log_replay_index = 0
        self.loaded_auto_landing_log = []
        self.auto_landing_log_interval = 0.1  # 自動離着陸ログ間隔（秒）

        # --- Current Flight State Values (Auto Landing Log Data) ---
        self.current_altitude = 0.0      # 高度 (mm)
        self.current_yaw = 0.0           # ヨー角 (度)
        self.current_throttle = 0.0      # スロットル量
        self.current_aux1 = 0.0          # 物資投下タイミング (AUX1)

        # --- Legacy Propeller Input Values (for compatibility) ---
        self.current_ail = 1500
        self.current_elev = 1500
        self.current_thro = 1000
        self.current_rudd = 1500

        # --- Auto Landing State ---
        self.auto_landing_enabled = False
        self.auto_landing_phase = 0  # 0: Manual, 1: Takeoff, 2: Drop, 3: Steady, 4: Landing
        self.estimated_distance = 0.0
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}  # Real world coordinates
        self.auto_landing_params = {}
        self.calibration_data = {
            'point1': {'distance': 5.0, 'size': 100.0},
            'point2': {'distance': 10.0, 'size': 50.0},
            'point3': {'distance': 15.0, 'size': 33.3}
        }

        self._setup_ui()
        self.load_pid_gains() # Also initializes PID controllers
        self.load_autopilot_params() # Load autopilot parameters
        self.load_auto_landing_params() # Load auto landing parameters
        self.load_marker_calibrations() # Load marker calibrations (angle + distance)
        self.load_auto_landing_ui_params() # Load parameters into UI
        self._setup_timers()
        self.update_com_ports()

        # 初期化完了後に距離キャリブレーション状況を確認・報告
        self.validate_distance_calibrations()

    def validate_distance_calibrations(self):
        """距離キャリブレーションデータの読み込み状況を検証・報告"""
        print("\n=== 距離キャリブレーション検証 ===")

        if not hasattr(self, 'marker_distance_calibrations'):
            print("ERROR: marker_distance_calibrations属性が存在しません")
            return

        if not self.marker_distance_calibrations:
            print("WARNING: marker_distance_calibrationsが空です")
            # 再読み込み試行
            print("距離キャリブレーション再読み込み試行中...")
            self.load_marker_distance_calibrations()

        print(f"距離キャリブレーション読み込み状況:")
        for marker_id in [1, 2, 3]:
            if marker_id in self.marker_distance_calibrations:
                count = len(self.marker_distance_calibrations[marker_id])
                print(f"  マーカー {marker_id}: {count} ポイント")
                if count > 0:
                    for i, calib in enumerate(self.marker_distance_calibrations[marker_id]):
                        print(f"    ポイント{i+1}: 距離={calib.get('distance')}m, サイズ={calib.get('size')}px")
            else:
                print(f"  マーカー {marker_id}: データなし")

        print("=====================================\n")

    def get_corrected_altitude(self, raw_altitude_mm, current_roll_deg):
        """
        LiDAR高度をバンク角で補正して真の高度を計算する

        Args:
            raw_altitude_mm (float): LiDARからの生高度（mm）
            current_roll_deg (float): 現在のバンク角（度）

        Returns:
            float: 補正後の高度（mm）
        """
        # 補正が無効の場合はそのまま返す
        correction_enabled = float(self.current_autopilot_params.get('lidar_bank_correction_enable', 1))
        if correction_enabled == 0:
            return raw_altitude_mm

        # 最大補正角度を超える場合は補正しない（測定値の信頼性が低いため）
        max_correction_angle = float(self.current_autopilot_params.get('lidar_max_correction_angle', 45.0))
        if abs(current_roll_deg) > max_correction_angle:
            return raw_altitude_mm

        # コサイン補正：真の高度 = LiDAR測定値 / cos(バンク角)
        import math
        roll_rad = math.radians(abs(current_roll_deg))
        cos_roll = math.cos(roll_rad)

        # ゼロ除算を避ける
        if cos_roll < 0.001:  # 約89.94度以上
            return raw_altitude_mm

        corrected_altitude = raw_altitude_mm / cos_roll

        return corrected_altitude

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
        # Create the main tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create GCS tab (existing functionality)
        self._create_gcs_tab()

        # Create Auto Landing tab (new functionality)
        self._create_auto_landing_tab()

        # Create Input Replay tab (manual piloting recording/replay system)
        self._create_input_replay_tab()

    def _create_gcs_tab(self):
        gcs_widget = QWidget()
        main_layout = QHBoxLayout(gcs_widget)

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

        # Add GCS tab to tab widget
        self.tab_widget.addTab(gcs_widget, "GCS")

    def _create_auto_landing_tab(self):
        auto_landing_widget = QWidget()
        main_layout = QHBoxLayout(auto_landing_widget)

        # Left panel: Control inputs
        left_panel = self._create_auto_landing_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center panel: Position and attitude visualization
        center_panel = self._create_auto_landing_center_panel()
        main_layout.addWidget(center_panel, 1)  # Reduced from 2 to 1 for more graph space

        # Right panel: Parameter panels
        right_panel = self._create_auto_landing_right_panel()
        main_layout.addWidget(right_panel, 2)  # Increased from 1 to 2 for more graph space

        # Add Auto Landing tab to tab widget
        self.tab_widget.addTab(auto_landing_widget, "自動離着陸")

        # Create calibration graph tab
        self._create_calibration_graph_tab()

    def _create_input_replay_tab(self):
        """Create the Input Replay tab for manual piloting recording/replay system"""
        input_replay_widget = QWidget()
        main_layout = QHBoxLayout(input_replay_widget)

        # Left panel: Input Log System controls
        left_panel = self._create_input_replay_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Center panel: Real-time data display
        center_panel = self._create_input_replay_center_panel()
        main_layout.addWidget(center_panel, 1)  # Reduced from 2 to 1 for more graph space

        # Right panel: Log data visualization
        right_panel = self._create_input_replay_right_panel()
        main_layout.addWidget(right_panel, 2)  # Increased from 1 to 2 for more graph space

        # Add Input Replay tab to tab widget
        self.tab_widget.addTab(input_replay_widget, "手動操縦記録・再現")

    def _create_input_replay_left_panel(self):
        """Create left panel for input replay controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Input Log System group
        log_group = QGroupBox("手動操縦ログシステム")
        log_layout = QVBoxLayout(log_group)

        # Recording controls
        record_layout = QHBoxLayout()
        self.record_button = QPushButton("操縦記録開始")
        self.record_button.setCheckable(True)
        self.record_button.clicked.connect(self.toggle_input_recording)

        self.record_status_label = QLabel("待機中")
        self.record_status_label.setStyleSheet("color: #888; font-weight: bold;")

        record_layout.addWidget(self.record_button)
        record_layout.addWidget(self.record_status_label)
        log_layout.addLayout(record_layout)

        # Load/Save controls
        file_layout = QHBoxLayout()
        self.save_log_button = QPushButton("操縦ログ保存")
        self.load_log_button = QPushButton("操縦ログ選択・読み込み")
        self.save_log_button.clicked.connect(self.save_input_log)
        self.load_log_button.clicked.connect(self.load_input_log)

        file_layout.addWidget(self.save_log_button)
        file_layout.addWidget(self.load_log_button)
        log_layout.addLayout(file_layout)

        # Replay controls
        replay_layout = QHBoxLayout()
        self.replay_button = QPushButton("操縦再現飛行開始")
        self.replay_button.clicked.connect(self.toggle_input_replay)
        self.replay_button.setEnabled(False)  # AUX5が有効な時のみ使用可能

        self.replay_status_label = QLabel("操縦ログ未読み込み")
        self.replay_status_label.setStyleSheet("color: #888; font-weight: bold;")

        replay_layout.addWidget(self.replay_button)
        replay_layout.addWidget(self.replay_status_label)
        log_layout.addLayout(replay_layout)

        # Log info
        self.log_info_label = QLabel("操縦記録データ: なし")
        self.log_info_label.setStyleSheet("color: #666; font-size: 10px;")
        log_layout.addWidget(self.log_info_label)

        layout.addWidget(log_group)

        # Settings group
        settings_group = QGroupBox("記録・再現設定")
        settings_layout = QFormLayout(settings_group)

        # Log interval setting
        self.log_interval_input = QLineEdit("0.1")
        self.log_interval_input.setMaximumWidth(80)
        self.log_interval_input.textChanged.connect(self.update_log_interval)
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(self.log_interval_input)
        interval_layout.addWidget(QLabel("秒"))
        interval_layout.addStretch()
        settings_layout.addRow("記録・再現間隔:", interval_layout)

        # Interval info
        interval_info = QLabel("※ 0.05-1.0秒の範囲で設定してください")
        interval_info.setStyleSheet("color: #666; font-size: 10px;")
        settings_layout.addWidget(interval_info)

        layout.addWidget(settings_group)

        # AUX5 Status group
        aux5_group = QGroupBox("AUX5状態 (自動離着陸スイッチ)")
        aux5_layout = QVBoxLayout(aux5_group)

        self.aux5_status_label = QLabel("AUX5: 無効")
        self.aux5_status_label.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 14px;")
        aux5_layout.addWidget(self.aux5_status_label)

        aux5_note = QLabel("※ ログ再現にはAUX5が有効である必要があります")
        aux5_note.setStyleSheet("color: #666; font-size: 10px;")
        aux5_layout.addWidget(aux5_note)

        layout.addWidget(aux5_group)

        layout.addStretch()
        return panel

    def _create_input_replay_center_panel(self):
        """Create center panel for real-time data display"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Current Input Values group
        input_group = QGroupBox("現在の機体状態量")
        input_layout = QGridLayout(input_group)

        # Create labels for current flight state values
        self.current_altitude_label = QLabel("高度: ---")
        self.current_yaw_label = QLabel("ヨー角: ---")
        self.current_throttle_label = QLabel("スロットル: ---")
        self.current_aux1_label = QLabel("物資投下: ---")

        input_layout.addWidget(QLabel("高度(mm):"), 0, 0)
        input_layout.addWidget(self.current_altitude_label, 0, 1)
        input_layout.addWidget(QLabel("ヨー角(°):"), 1, 0)
        input_layout.addWidget(self.current_yaw_label, 1, 1)
        input_layout.addWidget(QLabel("スロットル:"), 2, 0)
        input_layout.addWidget(self.current_throttle_label, 2, 1)
        input_layout.addWidget(QLabel("AUX1:"), 3, 0)
        input_layout.addWidget(self.current_aux1_label, 3, 1)

        layout.addWidget(input_group)

        # Manual piloting stick displays (autopilot servo control visualization)
        manual_control_group = QGroupBox("自動操縦時操縦量")
        manual_control_layout = QGridLayout(manual_control_group)

        self.manual_left_stick = StickWidget("ラダー", "エレベーター")
        self.manual_right_stick = StickWidget("エルロン", "スロットル")
        self.manual_left_stick_label = QLabel("R: 0, E: 0")
        self.manual_right_stick_label = QLabel("A: 0, T: 0")

        manual_control_layout.addWidget(self.manual_left_stick, 0, 0)
        manual_control_layout.addWidget(self.manual_right_stick, 0, 1)
        manual_control_layout.addWidget(self.manual_left_stick_label, 1, 0, Qt.AlignCenter)
        manual_control_layout.addWidget(self.manual_right_stick_label, 1, 1, Qt.AlignCenter)

        layout.addWidget(manual_control_group)

        # Recording Progress group
        progress_group = QGroupBox("記録進行状況")
        progress_layout = QVBoxLayout(progress_group)

        self.recording_time_label = QLabel("記録時間: 0.0秒")
        self.recording_points_label = QLabel("記録点数: 0点")
        self.recording_rate_label = QLabel("記録レート: 0Hz")

        progress_layout.addWidget(self.recording_time_label)
        progress_layout.addWidget(self.recording_points_label)
        progress_layout.addWidget(self.recording_rate_label)

        layout.addWidget(progress_group)

        # Loaded Log Info group
        log_info_group = QGroupBox("読み込み済みログ情報")
        log_info_layout = QVBoxLayout(log_info_group)

        self.loaded_log_info_label = QLabel("ログ未読み込み")
        self.loaded_log_duration_label = QLabel("継続時間: ---")
        self.loaded_log_points_label = QLabel("データ点数: ---")
        self.loaded_log_interval_label = QLabel("記録間隔: ---")

        log_info_layout.addWidget(self.loaded_log_info_label)
        log_info_layout.addWidget(self.loaded_log_duration_label)
        log_info_layout.addWidget(self.loaded_log_points_label)
        log_info_layout.addWidget(self.loaded_log_interval_label)

        layout.addWidget(log_info_group)

        # Replay Progress group
        replay_progress_group = QGroupBox("再現進行状況")
        replay_progress_layout = QVBoxLayout(replay_progress_group)

        self.replay_time_label = QLabel("再現時間: 0.0秒")
        self.replay_progress_label = QLabel("進行率: 0%")
        self.replay_current_values_label = QLabel("現在の再現値: ---")

        replay_progress_layout.addWidget(self.replay_time_label)
        replay_progress_layout.addWidget(self.replay_progress_label)
        replay_progress_layout.addWidget(self.replay_current_values_label)

        layout.addWidget(replay_progress_group)

        layout.addStretch()
        return panel

    def _create_input_replay_right_panel(self):
        """Create right panel for log data visualization"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Flight State Graph group
        graph_group = QGroupBox("飛行状態グラフ")
        graph_layout = QVBoxLayout(graph_group)

        # Create and add the flight state graph widget
        self.flight_state_graph = FlightStateGraphWidget()
        graph_layout.addWidget(self.flight_state_graph)

        layout.addWidget(graph_group)

        return panel

    def _create_auto_landing_left_panel(self):
        """Create left panel for auto landing tab - Control enable, inputs display and parameters"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Auto Landing Enable button (moved from right panel)
        enable_group = QGroupBox("自動離着陸制御")
        enable_layout = QVBoxLayout(enable_group)

        self.auto_landing_enable_button = QPushButton("自動離着陸有効化")
        self.auto_landing_enable_button.setCheckable(True)
        self.auto_landing_enable_button.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; }"
            "QPushButton:checked { background-color: #28a745; }"
        )
        self.auto_landing_enable_button.clicked.connect(self.toggle_auto_landing)
        enable_layout.addWidget(self.auto_landing_enable_button)

        # Phase display
        self.phase_label = QLabel("フェーズ: 手動")
        self.phase_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFD700;")
        enable_layout.addWidget(self.phase_label)

        # Distance display
        self.distance_label = QLabel("推定距離: -- m")
        enable_layout.addWidget(self.distance_label)

        # Altitude debug display
        self.altitude_debug_label = QLabel("高度: -- m (-- mm)")
        enable_layout.addWidget(self.altitude_debug_label)

        layout.addWidget(enable_group)

        # Auto landing stick displays (reuse existing StickWidget)
        control_group = QGroupBox("操縦量")
        control_layout = QGridLayout(control_group)

        self.auto_left_stick = StickWidget("ラダー", "エレベーター")
        self.auto_right_stick = StickWidget("エルロン", "スロットル")
        self.auto_left_stick_label = QLabel("R: 0, E: 0")
        self.auto_right_stick_label = QLabel("A: 0, T: 0")

        control_layout.addWidget(self.auto_left_stick, 0, 0)
        control_layout.addWidget(self.auto_right_stick, 0, 1)
        control_layout.addWidget(self.auto_left_stick_label, 1, 0, Qt.AlignCenter)
        control_layout.addWidget(self.auto_right_stick_label, 1, 1, Qt.AlignCenter)

        layout.addWidget(control_group)

        # Parameters panel (moved from right panel)
        params_group = QGroupBox("制御パラメータ")
        params_layout = QFormLayout(params_group)

        # Create parameter input fields
        self.auto_landing_param_edits = {}
        for label, key, default_value in self.AUTO_LANDING_PARAMS:
            edit = QLineEdit(default_value)
            self.auto_landing_param_edits[key] = edit
            params_layout.addRow(label + ":", edit)

        # Update button
        update_params_button = QPushButton("パラメータ更新")
        update_params_button.clicked.connect(self.update_and_save_auto_landing_params)
        params_layout.addRow(update_params_button)

        layout.addWidget(params_group)

        layout.addStretch()

        return panel

    def _create_auto_landing_center_panel(self):
        """Create center panel for auto landing tab - Position and attitude visualization"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Position visualization
        position_group = QGroupBox("位置表示")
        position_layout = QHBoxLayout(position_group)

        self.xy_position_widget = PositionVisualizationWidget("XY")
        self.zy_position_widget = PositionVisualizationWidget("ZY")

        position_layout.addWidget(self.xy_position_widget)
        position_layout.addWidget(self.zy_position_widget)

        # 2D Attitude display for auto landing (Pitch and Yaw only)
        attitude_group = QGroupBox("姿勢表示")
        attitude_layout = QHBoxLayout(attitude_group)

        self.auto_pitch_widget = Attitude2DWidget("2NAa_pitch.png")
        self.auto_yaw_widget = Attitude2DWidget("2NAa_yaw.png")

        attitude_layout.addWidget(self.auto_pitch_widget)
        attitude_layout.addWidget(self.auto_yaw_widget)

        layout.addWidget(position_group, 2)
        layout.addWidget(attitude_group, 1)

        return panel

    def _create_auto_landing_right_panel(self):
        """Create right panel for auto landing tab - Marker calibration and distance estimation"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Marker Position Configuration
        position_group = QGroupBox("マーカー位置設定 (フィールドレイアウト)")
        position_layout = QVBoxLayout(position_group)

        # Position input controls
        self.marker_position_controls = {}
        for marker_id in [1, 2, 3]:
            marker_widget = QWidget()
            marker_layout = QFormLayout(marker_widget)

            # Get current positions from position widget or use defaults
            default_positions = {1: {'x': -3.0, 'y': 6.0, 'z': 0.0},
                               2: {'x': 0.0, 'y': 0.0, 'z': 0.0},
                               3: {'x': 3.0, 'y': 6.0, 'z': 0.0}}
            default_pos = default_positions[marker_id]

            x_input = QLineEdit(f"{default_pos['x']:.1f}")
            y_input = QLineEdit(f"{default_pos['y']:.1f}")
            z_input = QLineEdit(f"{default_pos.get('z', 0.0):.1f}")

            marker_layout.addRow(f"マーカー {marker_id} X (m):", x_input)
            marker_layout.addRow(f"マーカー {marker_id} Y (m):", y_input)
            marker_layout.addRow(f"マーカー {marker_id} Z (m):", z_input)

            # Store references
            self.marker_position_controls[marker_id] = {
                'x': x_input,
                'y': y_input,
                'z': z_input
            }

            # Connect real-time update
            x_input.textChanged.connect(lambda text, mid=marker_id: self.update_marker_position_realtime(mid))
            y_input.textChanged.connect(lambda text, mid=marker_id: self.update_marker_position_realtime(mid))
            z_input.textChanged.connect(lambda text, mid=marker_id: self.update_marker_position_realtime(mid))

            position_layout.addWidget(marker_widget)

        # Position control buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)

        save_positions_button = QPushButton("位置保存")
        save_positions_button.clicked.connect(self.save_marker_positions)
        load_positions_button = QPushButton("位置読み込み")
        load_positions_button.clicked.connect(self.load_marker_positions_ui)
        reset_positions_button = QPushButton("デフォルト位置")
        reset_positions_button.clicked.connect(self.reset_marker_positions)

        button_layout.addWidget(save_positions_button)
        button_layout.addWidget(load_positions_button)
        button_layout.addWidget(reset_positions_button)

        position_layout.addWidget(button_widget)
        layout.addWidget(position_group)

        layout.addStretch()

        return panel

    def _create_calibration_graph_tab(self):
        """Create calibration graph tab for visualizing distance calibration data and individual marker calibration"""
        calib_widget = QWidget()
        main_layout = QHBoxLayout(calib_widget)

        # Left Panel: Individual Marker Calibration
        left_panel = self._create_calibration_left_panel()
        main_layout.addWidget(left_panel, 1)

        # Right Panel: Graph display and data management
        right_panel = self._create_calibration_right_panel()
        main_layout.addWidget(right_panel, 2)

        # Add calibration graph tab to tab widget
        self.tab_widget.addTab(calib_widget, "キャリブレーション")

    def _create_calibration_left_panel(self):
        """Create left panel for calibration tab - Individual marker calibration"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ArUco Marker Individual Calibration
        marker_calib_group = QGroupBox("ArUcoマーカー個別キャリブレーション")
        marker_calib_layout = QVBoxLayout(marker_calib_group)

        # Create individual calibration controls for each marker
        self.marker_calib_controls = {}
        for marker_id in [1, 2, 3]:
            marker_widget = QWidget()
            marker_layout = QVBoxLayout(marker_widget)

            # Marker header
            header_label = QLabel(f"マーカー {marker_id}")
            header_label.setStyleSheet("font-weight: bold; color: #FFD700;")
            marker_layout.addWidget(header_label)

            # Angle offset only (X,Y offsets removed as they are different from landing control coordinates)
            offset_angle_input = QLineEdit("0.0")

            # Real-time data display with units
            realtime_label = QLabel(f"リアルタイム: サイズ=0px, 画像X=0px, 画像Y=0px")
            realtime_label.setStyleSheet("color: #87CEEB; font-size: 10px;")

            # Calibration buttons
            set_current_button = QPushButton("現在値をキャリブレーション値に設定")
            set_current_button.clicked.connect(lambda checked, mid=marker_id: self.set_current_as_calibration(mid))
            set_current_button.setEnabled(False)  # Initially disabled

            # Distance calibration for this marker
            distance_input = QLineEdit("5.0")
            record_distance_button = QPushButton("距離キャリブレーション記録")
            record_distance_button.clicked.connect(lambda checked, mid=marker_id: self.record_marker_distance_calibration(mid))
            record_distance_button.setEnabled(False)  # Initially disabled

            marker_form = QFormLayout()
            marker_form.addRow("角度 オフセット (度):", offset_angle_input)
            marker_form.addRow("リアルタイムデータ:", realtime_label)
            marker_form.addRow(set_current_button)
            marker_form.addRow("基準距離 (m):", distance_input)
            marker_form.addRow(record_distance_button)

            marker_layout.addLayout(marker_form)

            # Store references (X,Y offset removed as they are different from landing control coordinates)
            self.marker_calib_controls[marker_id] = {
                'offset_angle': offset_angle_input,
                'realtime': realtime_label,
                'distance_input': distance_input,
                'set_current_button': set_current_button,
                'record_distance_button': record_distance_button
            }

            marker_calib_layout.addWidget(marker_widget)

        layout.addWidget(marker_calib_group)
        layout.addStretch()

        return panel

    def _create_calibration_right_panel(self):
        """Create right panel for calibration tab - Graph display and data management"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)

        # Title
        title_label = QLabel("距離キャリブレーション・グラフ表示")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFD700;")
        main_layout.addWidget(title_label)

        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Marker selection
        marker_select_label = QLabel("表示マーカー:")
        self.graph_marker_combo = QComboBox()
        self.graph_marker_combo.addItems(["マーカー 1", "マーカー 2", "マーカー 3", "全マーカー"])
        self.graph_marker_combo.setCurrentText("全マーカー")
        self.graph_marker_combo.currentTextChanged.connect(self.update_calibration_graph)

        # Refresh button
        refresh_button = QPushButton("グラフ更新")
        refresh_button.clicked.connect(self.update_calibration_graph)

        # Backup calibration data button
        backup_calib_button = QPushButton("キャリブレーションデータバックアップ")
        backup_calib_button.clicked.connect(self.backup_calibration_data)

        # Restore calibration data button
        restore_calib_button = QPushButton("キャリブレーションデータ復元")
        restore_calib_button.clicked.connect(self.restore_calibration_data)

        control_layout.addWidget(marker_select_label)
        control_layout.addWidget(self.graph_marker_combo)
        control_layout.addWidget(refresh_button)
        control_layout.addWidget(backup_calib_button)
        control_layout.addWidget(restore_calib_button)
        control_layout.addStretch()

        main_layout.addWidget(control_panel)

        # Graph display area
        self.calibration_graph_widget = CalibrationGraphWidget()
        main_layout.addWidget(self.calibration_graph_widget, 1)

        # Data display area
        data_group = QGroupBox("キャリブレーションデータ")
        data_layout = QVBoxLayout(data_group)

        self.calibration_data_display = QLabel("キャリブレーションデータなし")
        self.calibration_data_display.setStyleSheet("font-family: monospace; color: #FFFFFF;")
        self.calibration_data_display.setWordWrap(True)

        data_layout.addWidget(self.calibration_data_display)
        main_layout.addWidget(data_group)

        return panel

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



    def toggle_auto_landing(self):
        """Toggle auto landing enable/disable"""
        self.auto_landing_enabled = self.auto_landing_enable_button.isChecked()

        if self.auto_landing_enabled:
            print("Auto landing enabled")
            # Reset PID controllers
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.alt_pid.reset()
            self.yaw_pid.reset()

            # Reset phase
            self.auto_landing_phase = 0

            # Update button text
            self.auto_landing_enable_button.setText("自動離着陸無効化")
        else:
            print("Auto landing disabled")
            self.auto_landing_phase = 0

            # Clear autopilot displays
            if hasattr(self, 'auto_left_stick') and hasattr(self, 'auto_right_stick'):
                self.auto_left_stick.set_autopilot_position(None, None)
                self.auto_left_stick.set_autopilot_active(False)
                self.auto_right_stick.set_autopilot_position(None, None)
                self.auto_right_stick.set_autopilot_active(False)

            # Reset manual piloting tab stick displays as well
            if hasattr(self, 'manual_left_stick') and hasattr(self, 'manual_right_stick'):
                self.manual_left_stick.set_autopilot_position(None, None)
                self.manual_left_stick.set_autopilot_active(False)
                self.manual_right_stick.set_autopilot_position(None, None)
                self.manual_right_stick.set_autopilot_active(False)

            # Reset stick labels for both auto landing and manual piloting tabs
            if hasattr(self, 'auto_left_stick_label'):
                self.auto_left_stick_label.setText("R: 0, E: 0")
            if hasattr(self, 'auto_right_stick_label'):
                self.auto_right_stick_label.setText("A: 0, T: 0")
            if hasattr(self, 'manual_left_stick_label'):
                self.manual_left_stick_label.setText("R: 0, E: 0")
            if hasattr(self, 'manual_right_stick_label'):
                self.manual_right_stick_label.setText("A: 0, T: 0")

            # Update button text
            self.auto_landing_enable_button.setText("自動離着陸有効化")

    def record_calibration_point(self):
        """Record current marker sizes at specified distance for calibration"""
        try:
            distance = float(self.calib_distance_input.text())
            if distance <= 0:
                print("Invalid distance for calibration")
                return

            # Calculate average size from visible markers
            valid_sizes = []
            for marker_id, marker_data in self.aruco_markers.items():
                if marker_data['size'] > 0:
                    valid_sizes.append(marker_data['size'])

            if not valid_sizes:
                print("No visible markers for calibration")
                return

            # Calculate weighted average (larger markers get more weight)
            if len(valid_sizes) == 2:
                # 5:5 ratio
                avg_size = sum(valid_sizes) / len(valid_sizes)
            else:
                # 3:3:4 ratio with largest marker getting more weight
                valid_sizes.sort(reverse=True)  # Largest first
                if len(valid_sizes) == 3:
                    avg_size = (valid_sizes[0] * 4 + valid_sizes[1] * 3 + valid_sizes[2] * 3) / 10
                else:
                    avg_size = sum(valid_sizes) / len(valid_sizes)

            # Determine which calibration point to update (closest distance)
            point_key = 'point1'
            min_diff = abs(distance - self.calibration_data[point_key]['distance'])

            for key in ['point2', 'point3']:
                diff = abs(distance - self.calibration_data[key]['distance'])
                if diff < min_diff:
                    min_diff = diff
                    point_key = key

            # Update calibration data
            self.calibration_data[point_key] = {
                'distance': distance,
                'size': avg_size
            }

            # Save to file
            self.save_auto_landing_params()

            # Update display
            self.update_calibration_display()

            print(f"Recorded calibration point: {distance}m -> {avg_size} pixels")

        except ValueError:
            print("Invalid distance value for calibration")

    def update_calibration_display(self):
        """Update the calibration display"""
        if hasattr(self, 'calib_display'):
            calib_text = "キャリブレーション点:\n"
            for point_name, point_data in self.calibration_data.items():
                dist = point_data['distance']
                size = point_data['size']
                calib_text += f"{point_name}: {dist:.1f}m -> {size:.1f}px\n"
            self.calib_display.setText(calib_text)

    def set_current_as_calibration(self, marker_id):
        """Set current marker data as calibration baseline"""
        if marker_id not in self.aruco_markers:
            print(f"Marker {marker_id} not found")
            return

        marker_data = self.aruco_markers[marker_id]
        if marker_data['size'] <= 0:
            print(f"Marker {marker_id} not currently visible")
            return

        # Set angle calibration only (X,Y offset removed as they are different from landing control coordinates)
        controls = self.marker_calib_controls[marker_id]
        controls['offset_angle'].setText("0.0")

        # Update calibration data
        self.marker_calibrations[marker_id]['offset_angle'] = 0.0

        print(f"Set marker {marker_id} angle calibration to 0.0 degrees")
        self.save_marker_calibrations()

    def toggle_input_recording(self):
        """Toggle propeller input recording"""
        if not self.input_log_recording:
            # Start recording
            self.input_log_recording = True
            self.input_log_data = []
            self.input_log_start_time = 0
            self.input_log_last_record_time = 0

            # 手動操縦記録開始時の基準ヨー角を設定
            self.mission_start_yaw = self.latest_attitude.get('yaw', 0.0)
            print(f"手動操縦記録開始: 基準ヨー角={self.mission_start_yaw:.1f}°")

            self.record_button.setText("操縦記録停止")
            self.record_button.setStyleSheet("background-color: #dc3545; color: white;")
            self.record_status_label.setText("記録中...")
            self.record_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")

            print("プロポ入力記録を開始しました")
        else:
            # Stop recording
            self.input_log_recording = False

            self.record_button.setText("操縦記録開始")
            self.record_button.setStyleSheet("")
            self.record_status_label.setText(f"記録完了 ({len(self.input_log_data)}点)")
            self.record_status_label.setStyleSheet("color: #28a745; font-weight: bold;")

            duration = self.input_log_data[-1]['timestamp'] if self.input_log_data else 0
            self.log_info_label.setText(f"記録データ: {len(self.input_log_data)}点, {duration:.1f}秒, 間隔:{self.input_log_interval:.3f}秒")

            print(f"プロポ入力記録を停止しました ({len(self.input_log_data)}点, {duration:.1f}秒, 間隔:{self.input_log_interval:.3f}秒)")

    def update_log_interval(self):
        """Update log recording/replay interval"""
        try:
            interval = float(self.log_interval_input.text())
            if 0.05 <= interval <= 1.0:
                self.input_log_interval = interval
                print(f"ログ間隔を{interval}秒に設定しました")
            else:
                print(f"警告: ログ間隔は0.05-1.0秒の範囲で設定してください (現在値: {interval})")
        except (ValueError, AttributeError):
            print("警告: 無効なログ間隔の値です")

    def save_input_log(self):
        """Save recorded manual piloting flight state log to file"""
        if not self.input_log_data:
            print("保存する手動操縦ログデータがありません")
            return

        try:
            # Create logs directory if it doesn't exist
            if not os.path.exists('logs'):
                os.makedirs('logs')

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/manual_piloting_log_{timestamp}.json"

            # Save flight state data directly as array (same format as auto landing log)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.input_log_data, f, indent=2, ensure_ascii=False)

            duration = self.input_log_data[-1]['elapsed_time'] if self.input_log_data else 0
            print(f"手動操縦ログを保存しました: {filename} ({len(self.input_log_data)}点, {duration:.1f}秒)")

        except Exception as e:
            print(f"手動操縦ログ保存エラー: {e}")

    def load_input_log(self):
        """Load manual piloting flight state log from file with dialog"""
        try:
            # Open file dialog to select manual piloting log file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "手動操縦ログを選択",
                "logs/",
                "JSON Files (*.json);;All Files (*)"
            )

            if not file_path:
                print("ファイル選択がキャンセルされました")
                return

            # Verify it's a valid manual piloting log file
            if not (file_path.endswith('.json') and ('manual_piloting_log_' in file_path or 'manual_piloting_dummy_log' in file_path)):
                print("選択されたファイルは手動操縦ログファイルではありません")
                self.replay_status_label.setText("無効なファイル")
                return

            latest_file = file_path

            with open(latest_file, 'r', encoding='utf-8') as f:
                self.loaded_input_log = json.load(f)

            # Calculate duration and interval from flight state data
            if self.loaded_input_log:
                duration = self.loaded_input_log[-1]['elapsed_time']
                data_points = len(self.loaded_input_log)
                # Calculate average interval from elapsed time
                loaded_interval = duration / max(1, data_points - 1) if data_points > 1 else 0.1
            else:
                duration = 0
                data_points = 0
                loaded_interval = 0.1

            # 読み込んだログの間隔を表示用に記録（現在の設定は変更しない）
            self.loaded_log_interval = loaded_interval

            self.replay_status_label.setText(f"手動操縦ログ読み込み完了 ({data_points}点)")
            self.replay_status_label.setStyleSheet("color: #28a745; font-weight: bold;")

            # Update loaded log info in input replay tab
            if hasattr(self, 'loaded_log_info_label'):
                self.loaded_log_info_label.setText(f"ファイル: {os.path.basename(latest_file)}")
                self.loaded_log_duration_label.setText(f"継続時間: {duration:.1f}秒")
                self.loaded_log_points_label.setText(f"データ点数: {data_points}点")
                self.loaded_log_interval_label.setText(f"記録間隔: {loaded_interval:.3f}秒")

            # Update replay button state based on AUX5
            self.update_replay_button_state()

            # Update graph with loaded log data
            if hasattr(self, 'flight_state_graph') and self.flight_state_graph:
                self.flight_state_graph.update_plots(self.loaded_input_log)
                print("手動操縦ログのグラフを更新しました")

            print(f"ログを読み込みました: {latest_file} ({data_points}点, {duration:.1f}秒, 間隔:{loaded_interval:.3f}秒)")

        except Exception as e:
            print(f"ログ読み込みエラー: {e}")
            self.replay_status_label.setText("ログ読み込みエラー")

    def toggle_input_replay(self):
        """Toggle input log replay"""
        if not self.input_log_replaying:
            # Start replay
            if not self.loaded_input_log:
                print("再現するログがありません")
                return

            if not self.auto_landing_enabled:
                print("AUX5が無効のため、ログ再現を開始できません")
                return

            self.input_log_replaying = True
            self.input_log_replay_start_time = time.time()
            self.input_log_replay_index = 0

            # 再現開始時の基準ヨー角を設定（記録されたデータは変位角のため）
            self.replay_start_yaw = self.latest_attitude.get('yaw', 0.0)
            print(f"ログ再現開始: 基準ヨー角={self.replay_start_yaw:.1f}°")

            # ログ再現飛行用のPIDコントローラーをリセット
            if hasattr(self, 'roll_pid'):
                self.roll_pid.reset()
            if hasattr(self, 'yaw_pid'):
                self.yaw_pid.reset()
            if hasattr(self, 'pitch_pid'):
                self.pitch_pid.reset()
            if hasattr(self, 'alt_pid'):
                self.alt_pid.reset()
            print("ログ再現用PIDコントローラーをリセットしました")

            # Start mission mode 4 for input replay
            self.start_mission(4)

            self.replay_button.setText("操縦再現停止")
            self.replay_button.setStyleSheet("background-color: #dc3545; color: white;")
            self.replay_status_label.setText("再現中...")
            self.replay_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")

            print(f"ログ再現飛行を開始しました (ミッションモード4) ({len(self.loaded_input_log)}点)")
        else:
            # Stop replay
            self.input_log_replaying = False

            # Stop mission when replay is stopped
            self.stop_mission()

            # 即座にミッションモード0（手動）を送信
            if self.is_connected and hasattr(self, 'latest_attitude'):
                # 現在のプロポ入力値を取得して手動コマンドとして送信
                current_commands = {
                    'ail': getattr(self, 'current_ail', 1500),
                    'elev': getattr(self, 'current_elev', 1500),
                    'thro': getattr(self, 'current_thro', 1000),
                    'rudd': getattr(self, 'current_rudd', 1500),
                    'aux1': getattr(self, 'current_aux1', 1500)  # AUX1も含める
                }
                self.send_serial_command(current_commands)
                print("ミッションモード0（手動）に即座に切り替えました")

            self.replay_button.setText("ログ再現飛行開始")
            self.replay_button.setStyleSheet("")
            self.replay_status_label.setText("操縦再現停止")
            self.replay_status_label.setStyleSheet("color: #6c757d; font-weight: bold;")

            print("ログ再現飛行を手動停止しました (ミッション停止)")

    def update_replay_button_state(self):
        """Update replay button enabled state based on AUX5 and loaded log"""
        can_replay = self.auto_landing_enabled and len(self.loaded_input_log) > 0
        self.replay_button.setEnabled(can_replay)

        if can_replay:
            self.replay_button.setText("ログ再現飛行開始")
            self.replay_button.setStyleSheet("")

        # Update AUX5 status in input replay tab
        if hasattr(self, 'aux5_status_label'):
            if self.auto_landing_enabled:
                self.aux5_status_label.setText("AUX5: 有効")
                self.aux5_status_label.setStyleSheet("color: #28a745; font-weight: bold; font-size: 14px;")
            else:
                self.aux5_status_label.setText("AUX5: 無効")
                self.aux5_status_label.setStyleSheet("color: #dc3545; font-weight: bold; font-size: 14px;")

    def update_input_replay_displays(self, ail, elev, thro, rudd, aux1):
        """Update real-time displays in input replay tab"""
        # Update current flight state values
        if hasattr(self, 'current_altitude_label'):
            self.current_altitude_label.setText(f"高度: {self.current_altitude:.1f}mm")
            self.current_yaw_label.setText(f"ヨー角: {self.current_yaw:.1f}°")
            self.current_throttle_label.setText(f"スロットル: {self.current_throttle:.0f}")
            self.current_aux1_label.setText(f"物資投下: {self.current_aux1:.0f}")

        # Update auto landing log recording progress if recording
        if self.auto_landing_log_recording and hasattr(self, 'auto_landing_log_status_label'):
            current_time = time.time()
            if self.auto_landing_log_start_time > 0:
                recording_time = current_time - self.auto_landing_log_start_time
                recording_points = len(self.auto_landing_log_data)
                recording_rate = recording_points / recording_time if recording_time > 0 else 0

                status_text = f"記録中: {recording_time:.1f}秒, {recording_points}点, {recording_rate:.1f}Hz"
                self.auto_landing_log_status_label.setText(status_text)

        # Update auto landing log replay progress if replaying
        if self.auto_landing_log_replaying and hasattr(self, 'auto_landing_log_status_label'):
            current_time = time.time() - self.auto_landing_log_replay_start_time
            total_duration = self.loaded_auto_landing_log[-1]['elapsed_time'] if self.loaded_auto_landing_log else 0
            progress_percent = (current_time / total_duration * 100) if total_duration > 0 else 0
            progress_percent = min(100, progress_percent)

            status_text = f"再現中: {current_time:.1f}秒, 進行率: {progress_percent:.1f}%"
            self.auto_landing_log_status_label.setText(status_text)

            # Show current replay values if available
            if self.input_log_replay_index < len(self.loaded_input_log):
                current_replay = self.loaded_input_log[self.input_log_replay_index]
                aux1_value = current_replay.get('aux1', 'N/A')
                self.replay_current_values_label.setText(
                    f"A:{current_replay['ail']}, E:{current_replay['elev']}, "
                    f"T:{current_replay['thro']}, R:{current_replay['rudd']}, AUX1:{aux1_value}"
                )

    def update_marker_realtime_display(self):
        """Update real-time marker data display"""
        for marker_id in [1, 2, 3]:
            if marker_id in self.marker_calib_controls:
                marker_data = self.aruco_markers.get(marker_id, {'size': 0, 'x': 0, 'y': 0})

                # Check if marker is visible
                is_visible = marker_data['size'] > 0

                # Enable/disable buttons based on visibility
                self.marker_calib_controls[marker_id]['set_current_button'].setEnabled(is_visible)
                self.marker_calib_controls[marker_id]['record_distance_button'].setEnabled(is_visible)

                # Estimate distance if calibration data is available
                try:
                    estimated_distance = self.estimate_distance_from_marker(marker_id)
                    distance_text = f", 推定距離={estimated_distance:.2f}m" if estimated_distance is not None else ""
                except Exception as e:
                    print(f"Error estimating distance for marker {marker_id}: {e}")
                    distance_text = ""

                realtime_text = f"リアルタイム: サイズ={marker_data['size']}px, 画像X={marker_data.get('image_x', 0):.0f}px, 画像Y={marker_data.get('image_y', 0):.0f}px{distance_text}"

                # Color coding based on visibility
                if is_visible:
                    self.marker_calib_controls[marker_id]['realtime'].setStyleSheet("color: #00FF00; font-size: 10px;")  # Green
                else:
                    self.marker_calib_controls[marker_id]['realtime'].setStyleSheet("color: #FF0000; font-size: 10px;")  # Red

                self.marker_calib_controls[marker_id]['realtime'].setText(realtime_text)

    def get_calibrated_marker_position(self, marker_id):
        """Get calibrated marker position with offsets applied"""
        if marker_id not in self.aruco_markers or marker_id not in self.marker_calibrations:
            return None

        marker_data = self.aruco_markers[marker_id]
        calib_data = self.marker_calibrations[marker_id]

        if marker_data['size'] <= 0 or not calib_data['enabled']:
            return None

        # Apply calibration offsets
        calibrated_x = float(marker_data['x']) - calib_data['offset_x']
        calibrated_y = float(marker_data['y']) - calib_data['offset_y']
        calibrated_angle = calib_data['offset_angle']  # Future use for angle correction

        return {
            'x': calibrated_x,
            'y': calibrated_y,
            'angle': calibrated_angle,
            'size': marker_data['size'],
            'raw_x': marker_data['x'],
            'raw_y': marker_data['y']
        }

    def save_marker_calibrations(self):
        """Save all marker calibrations (angle + distance) to unified JSON file"""
        try:
            calib_file = 'marker_calibrations.json'

            # Prepare unified calibration data
            unified_data = {
                'last_updated': datetime.now().isoformat(),
                'version': '2.0',
                'angle_calibrations': self.marker_calibrations,
                'distance_calibrations': getattr(self, 'marker_distance_calibrations', {})
            }

            with open(calib_file, 'w', encoding='utf-8') as f:
                json.dump(unified_data, f, indent=2, ensure_ascii=False)

            print("All marker calibrations saved to unified JSON file")
        except Exception as e:
            print(f"Failed to save marker calibrations: {e}")

    def load_marker_calibrations(self):
        """Load all marker calibrations (angle + distance) from unified JSON file"""
        try:
            calib_file = 'marker_calibrations.json'
            print(f"Loading marker calibrations from {calib_file}")

            if os.path.exists(calib_file):
                with open(calib_file, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                    print(f"JSON loaded successfully: {import_data.keys()}")

                # Check if this is the new unified format
                if 'angle_calibrations' in import_data:
                    # Initialize marker_calibrations if not exists
                    if not hasattr(self, 'marker_calibrations'):
                        self.marker_calibrations = {1: {'offset_angle': 0.0}, 2: {'offset_angle': 0.0}, 3: {'offset_angle': 0.0}}

                    # Load angle calibrations
                    for marker_id_str, calib in import_data['angle_calibrations'].items():
                        marker_id = int(marker_id_str)
                        if marker_id not in self.marker_calibrations:
                            self.marker_calibrations[marker_id] = {}
                        self.marker_calibrations[marker_id].update(calib)

                        # Update UI input field if UI is initialized
                        if hasattr(self, 'marker_calib_controls') and marker_id in self.marker_calib_controls:
                            try:
                                self.marker_calib_controls[marker_id]['offset_angle'].setText(str(calib.get('offset_angle', 0.0)))
                            except Exception as ui_error:
                                print(f"UI update failed for marker {marker_id}: {ui_error}")

                    # Load distance calibrations
                    if 'distance_calibrations' in import_data:
                        if not hasattr(self, 'marker_distance_calibrations'):
                            self.marker_distance_calibrations = {}

                        for marker_id_str, calib_list in import_data['distance_calibrations'].items():
                            marker_id = int(marker_id_str)

                            # Filter out invalid entries and ensure complete data
                            valid_calibrations = []
                            for calib in calib_list:
                                if (isinstance(calib, dict) and
                                    all(key in calib and calib[key] is not None
                                        for key in ['distance', 'size', 'x', 'y'])):
                                    valid_calibrations.append(calib)

                            if valid_calibrations:
                                self.marker_distance_calibrations[marker_id] = valid_calibrations
                                print(f"距離キャリブレーション - マーカー {marker_id}: {len(valid_calibrations)} ポイント読み込み")

                    print(f"All marker calibrations loaded from unified JSON file")
                    print(f"Angle calibrations: {self.marker_calibrations}")
                    print(f"Distance calibrations loaded for markers: {list(self.marker_distance_calibrations.keys()) if hasattr(self, 'marker_distance_calibrations') else 'None'}")

                    # 距離キャリブレーションの詳細デバッグ情報
                    if hasattr(self, 'marker_distance_calibrations') and self.marker_distance_calibrations:
                        for marker_id, calibs in self.marker_distance_calibrations.items():
                            print(f"マーカー {marker_id} 距離キャリブレーション: {len(calibs)} ポイント")
                            for i, calib in enumerate(calibs):
                                print(f"  ポイント {i+1}: 距離={calib.get('distance', 'N/A')}m, サイズ={calib.get('size', 'N/A')}px")

                elif 'markers' in import_data:
                    # This is old distance-only format, load distance calibrations only
                    if not hasattr(self, 'marker_distance_calibrations'):
                        self.marker_distance_calibrations = {}

                    for marker_id_str, calib_list in import_data['markers'].items():
                        marker_id = int(marker_id_str)

                        # Filter out invalid entries and ensure complete data
                        valid_calibrations = []
                        for calib in calib_list:
                            if (isinstance(calib, dict) and
                                all(key in calib and calib[key] is not None
                                    for key in ['distance', 'size', 'x', 'y'])):
                                valid_calibrations.append(calib)

                        if valid_calibrations:
                            self.marker_distance_calibrations[marker_id] = valid_calibrations
                            print(f"旧形式 - マーカー {marker_id}: {len(valid_calibrations)} ポイント読み込み")

                    print("Distance calibrations loaded from old JSON format")

            else:
                # Fallback: try loading from old text format for angle calibrations
                old_calib_file = 'marker_calibrations.txt'
                if os.path.exists(old_calib_file):
                    with open(old_calib_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '=' in line:
                                key, value = line.split('=', 1)
                                if key.startswith('marker_') and '_' in key:
                                    parts = key.split('_')
                                    if len(parts) >= 3:
                                        marker_id = int(parts[1])
                                        param = '_'.join(parts[2:])

                                        if marker_id in self.marker_calibrations:
                                            if param == 'offset_angle':
                                                self.marker_calibrations[marker_id][param] = float(value)
                                                # Update UI input field
                                                if marker_id in self.marker_calib_controls:
                                                    self.marker_calib_controls[marker_id][param].setText(str(value))
                    print("Angle calibrations loaded from old text format")
                else:
                    print(f"No unified calibration file found: {calib_file}")

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON in marker calibrations file: {e}")
        except FileNotFoundError as e:
            print(f"Marker calibrations file not found: {e}")
        except Exception as e:
            print(f"Failed to load marker calibrations: {e}")
            import traceback
            traceback.print_exc()

    def record_marker_distance_calibration(self, marker_id):
        """Record distance calibration for a specific marker"""
        try:
            if marker_id not in self.marker_calib_controls:
                print(f"Marker {marker_id} controls not found")
                return

            # Get distance from input field
            distance_input = self.marker_calib_controls[marker_id]['distance_input']
            distance = float(distance_input.text())

            if distance <= 0:
                print(f"Invalid distance for marker {marker_id} calibration")
                return

            # Check if marker is visible
            if marker_id not in self.aruco_markers or self.aruco_markers[marker_id]['size'] <= 0:
                print(f"Marker {marker_id} not currently visible")
                return

            marker_size = self.aruco_markers[marker_id]['size']

            # Store calibration data for this specific marker
            if not hasattr(self, 'marker_distance_calibrations'):
                self.marker_distance_calibrations = {}

            if marker_id not in self.marker_distance_calibrations:
                self.marker_distance_calibrations[marker_id] = []

            # Add new calibration point (using image coordinates from camera)
            calibration_point = {
                'distance': distance,
                'size': marker_size,
                'x': self.aruco_markers[marker_id]['image_x'],  # Image coordinates (pixels)
                'y': self.aruco_markers[marker_id]['image_y']   # Image coordinates (pixels)
            }

            self.marker_distance_calibrations[marker_id].append(calibration_point)

            # Keep only the last 3 calibration points per marker
            if len(self.marker_distance_calibrations[marker_id]) > 3:
                self.marker_distance_calibrations[marker_id] = self.marker_distance_calibrations[marker_id][-3:]

            # Save to file
            self.save_marker_distance_calibrations()

            # Update calibration graph if it exists
            if hasattr(self, 'calibration_graph_widget'):
                self.update_calibration_graph()

            print(f"Recorded distance calibration for marker {marker_id}: {distance}m -> {marker_size} pixels")

        except ValueError:
            print(f"Invalid distance value for marker {marker_id} calibration")
        except Exception as e:
            print(f"Error recording marker {marker_id} distance calibration: {e}")

    def save_marker_distance_calibrations(self):
        """Save all marker calibrations (unified method - calls save_marker_calibrations)"""
        # This method now redirects to the unified save method
        self.save_marker_calibrations()

    def load_marker_distance_calibrations(self):
        """Load distance calibrations - redirects to unified method if not already loaded"""
        # If marker calibrations haven't been loaded yet, call the unified load method
        if not hasattr(self, 'marker_distance_calibrations') or not self.marker_distance_calibrations:
            print("距離キャリブレーションが未読み込み - 統合読み込みメソッドを呼び出し")
            self.load_marker_calibrations()

        # If still empty, try fallback methods
        if not hasattr(self, 'marker_distance_calibrations') or not self.marker_distance_calibrations:
            try:
                # Fallback: try to load from old text format and convert
                old_calib_file = 'marker_distance_calibrations.txt'
                if os.path.exists(old_calib_file):
                    print("Loading from old text format and converting...")
                    self.marker_distance_calibrations = {}

                    with open(old_calib_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if '=' in line:
                                key, value = line.split('=', 1)

                                if key.startswith('marker_') and '_point_' in key:
                                    parts = key.split('_')
                                    if len(parts) >= 5:
                                        marker_id = int(parts[1])
                                        point_idx = int(parts[3])
                                        param = parts[4]

                                        if marker_id not in self.marker_distance_calibrations:
                                            self.marker_distance_calibrations[marker_id] = []

                                        # Ensure we have enough calibration points
                                        while len(self.marker_distance_calibrations[marker_id]) <= point_idx:
                                            self.marker_distance_calibrations[marker_id].append({})

                                        self.marker_distance_calibrations[marker_id][point_idx][param] = float(value)

                    # Clean up incomplete data
                    for marker_id in list(self.marker_distance_calibrations.keys()):
                        valid_points = []
                        for point in self.marker_distance_calibrations[marker_id]:
                            if isinstance(point, dict) and all(key in point for key in ['distance', 'size', 'x', 'y']):
                                valid_points.append(point)
                        self.marker_distance_calibrations[marker_id] = valid_points

                    # Save in new unified format
                    self.save_marker_calibrations()
                    print("Converted old calibration data to unified JSON format")

            except Exception as e:
                print(f"Failed to load legacy distance calibrations: {e}")

    def estimate_distance_from_marker(self, marker_id):
        """Estimate distance to marker based on its size and calibration data"""
        if not hasattr(self, 'marker_distance_calibrations'):
            return None

        if marker_id not in self.marker_distance_calibrations:
            return None

        if marker_id not in self.aruco_markers or self.aruco_markers[marker_id]['size'] <= 0:
            return None

        current_size = self.aruco_markers[marker_id]['size']
        calibrations = self.marker_distance_calibrations[marker_id]

        if len(calibrations) == 0:
            return None

        # Validate calibration data - filter out invalid entries
        valid_calibrations = []
        for calib in calibrations:
            if isinstance(calib, dict) and 'size' in calib and 'distance' in calib:
                if calib['size'] > 0 and calib['distance'] > 0:
                    valid_calibrations.append(calib)

        if len(valid_calibrations) == 0:
            return None

        # Use linear interpolation or the closest calibration point
        if len(valid_calibrations) == 1:
            # Single point: use simple inverse relationship
            calib = valid_calibrations[0]
            estimated_distance = calib['distance'] * calib['size'] / current_size
            return estimated_distance
        else:
            # Multiple points: find the two closest by size
            calibrations = sorted(valid_calibrations, key=lambda x: x['size'])

            # Find the best interpolation range
            if current_size <= calibrations[0]['size']:
                # Extrapolate using first two points
                p1, p2 = calibrations[0], calibrations[1] if len(calibrations) > 1 else calibrations[0]
            elif current_size >= calibrations[-1]['size']:
                # Extrapolate using last two points
                p1, p2 = calibrations[-2] if len(calibrations) > 1 else calibrations[-1], calibrations[-1]
            else:
                # Interpolate between closest points
                p1 = calibrations[0]
                p2 = calibrations[-1]
                for i in range(len(calibrations) - 1):
                    if calibrations[i]['size'] <= current_size <= calibrations[i + 1]['size']:
                        p1, p2 = calibrations[i], calibrations[i + 1]
                        break

            # Linear interpolation
            if p1['size'] == p2['size']:
                return p1['distance']

            # Interpolate distance based on size ratio
            size_ratio = (current_size - p1['size']) / (p2['size'] - p1['size'])
            estimated_distance = p1['distance'] + size_ratio * (p2['distance'] - p1['distance'])

            return estimated_distance

    def update_calibration_graph(self):
        """Update calibration graph display from marker_calibrations.json"""
        if hasattr(self, 'calibration_graph_widget'):
            # Use existing marker_distance_calibrations data if available, otherwise load from JSON
            calibration_data = {}

            if hasattr(self, 'marker_distance_calibrations') and self.marker_distance_calibrations:
                calibration_data = self.marker_distance_calibrations
                print(f"キャリブレーションデータを内部データから取得: {len(calibration_data)} マーカー")
            else:
                # Fallback: load from JSON file
                calibration_data = self.load_calibration_data_from_json()
                print(f"キャリブレーションデータをJSONから取得: {len(calibration_data)} マーカー")

            # Get selected marker
            selected_text = self.graph_marker_combo.currentText()
            self.calibration_graph_widget.set_selected_marker(selected_text)

            # Pass calibration data
            if calibration_data:
                self.calibration_graph_widget.set_calibration_data(calibration_data)
                print(f"キャリブレーショングラフを更新: {len(calibration_data)} マーカーのデータを表示")
            else:
                print("キャリブレーションデータが見つかりません")

            # Update text display
            self.update_calibration_data_display()

    def load_calibration_data_from_json(self):
        """Load calibration data directly from marker_calibrations.json"""
        try:
            calib_file = 'marker_calibrations.json'
            if not os.path.exists(calib_file):
                print("marker_calibrations.jsonファイルが存在しません")
                return {}

            with open(calib_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Check for unified format (version 2.0)
            if 'distance_calibrations' in import_data:
                calibration_data = {}
                print(f"JSON内のdistance_calibrations: {import_data['distance_calibrations']}")

                for marker_id_str, calib_list in import_data['distance_calibrations'].items():
                    marker_id = int(marker_id_str)
                    print(f"マーカー {marker_id}: {calib_list}")

                    # Filter out empty dictionaries and ensure complete data
                    valid_calibrations = []
                    for calib in calib_list:
                        print(f"キャリブレーションエントリ検査: {calib}")

                        if isinstance(calib, dict):
                            missing_keys = [key for key in ['distance', 'size', 'x', 'y'] if key not in calib or calib[key] is None]
                            if missing_keys:
                                print(f"不完全なデータ - 欠損キー: {missing_keys}")
                            else:
                                valid_calibrations.append(calib)
                                print(f"有効なキャリブレーションデータ: {calib}")
                        else:
                            print(f"無効なデータ型: {type(calib)}")

                    if valid_calibrations:
                        calibration_data[marker_id] = valid_calibrations
                        print(f"マーカー {marker_id}: {len(valid_calibrations)} 個の有効なキャリブレーション")

                print(f"統合形式からキャリブレーションデータを読み込み: {len(calibration_data)} マーカー")
                return calibration_data

            # Check for old format (version 1.0)
            elif 'markers' in import_data:
                calibration_data = {}
                for marker_id_str, calib_list in import_data['markers'].items():
                    marker_id = int(marker_id_str)
                    # Filter out empty dictionaries and ensure complete data
                    valid_calibrations = []
                    for calib in calib_list:
                        if (isinstance(calib, dict) and
                            all(key in calib and calib[key] is not None for key in ['distance', 'size', 'x', 'y'])):
                            valid_calibrations.append(calib)
                    if valid_calibrations:
                        calibration_data[marker_id] = valid_calibrations

                print(f"旧形式からキャリブレーションデータを読み込み: {len(calibration_data)} マーカー")
                return calibration_data

            else:
                print("不明なJSONファイル形式です")
                return {}

        except json.JSONDecodeError as e:
            print(f"JSONファイルの解析エラー: {e}")
            return {}
        except Exception as e:
            print(f"キャリブレーションデータの読み込みエラー: {e}")
            return {}

    def update_calibration_data_display(self):
        """Update calibration data text display"""
        if not hasattr(self, 'calibration_data_display'):
            return

        if not hasattr(self, 'marker_distance_calibrations') or not self.marker_distance_calibrations:
            self.calibration_data_display.setText("キャリブレーションデータなし")
            return

        display_text = "キャリブレーションデータ:\n\n"

        for marker_id in [1, 2, 3]:
            if marker_id in self.marker_distance_calibrations:
                display_text += f"マーカー {marker_id}:\n"
                for i, point in enumerate(self.marker_distance_calibrations[marker_id]):
                    if 'distance' in point and 'size' in point:
                        display_text += f"  ポイント{i+1}: 距離={point['distance']:.2f}m, サイズ={point['size']:.1f}px\n"
                display_text += "\n"

        self.calibration_data_display.setText(display_text)

    def backup_calibration_data(self):
        """Create a timestamped backup of current calibration data"""
        try:
            if not hasattr(self, 'marker_distance_calibrations'):
                print("No calibration data to backup")
                return

            # Prepare data for backup
            backup_data = {
                'backup_time': datetime.now().isoformat(),
                'version': '1.0',
                'markers': self.marker_distance_calibrations
            }

            # Save backup file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f'marker_calibrations_backup_{timestamp}.json'

            with open(backup_filename, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)

            print(f"Calibration data backed up to {backup_filename}")

        except Exception as e:
            print(f"Failed to backup calibration data: {e}")

    def restore_calibration_data(self):
        """Restore calibration data from most recent backup"""
        try:
            import glob

            # Find the most recent backup file
            backup_files = glob.glob('marker_calibrations_backup_*.json')
            if not backup_files:
                print("No calibration backup files found")
                return

            # Use the most recent backup file
            latest_backup = max(backup_files)

            with open(latest_backup, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Load the data
            if 'markers' in backup_data:
                # Convert string keys to integers for marker IDs
                self.marker_distance_calibrations = {}
                for marker_id_str, calibrations in backup_data['markers'].items():
                    marker_id = int(marker_id_str)
                    self.marker_distance_calibrations[marker_id] = calibrations

                # Save to main calibration file
                self.save_marker_distance_calibrations()
                self.update_calibration_graph()
                print(f"Calibration data restored from {latest_backup}")
            else:
                print("Invalid backup data format")

        except Exception as e:
            print(f"Failed to restore calibration data: {e}")

    def update_and_save_auto_landing_params(self):
        """Update and save auto landing parameters"""
        try:
            for key, edit in self.auto_landing_param_edits.items():
                value = float(edit.text())
                self.auto_landing_params[key] = value

            self.save_auto_landing_params()
            print("Auto landing parameters updated and saved")

        except ValueError as e:
            print(f"自動離着陸パラメータの更新中にエラーが発生しました: {e}")
        except Exception as e:
            print(f"パラメータ保存中にエラーが発生しました: {e}")

    def load_auto_landing_ui_params(self):
        """Load auto landing parameters into UI elements"""
        # Update parameter input fields
        for key, edit in self.auto_landing_param_edits.items():
            if key in self.auto_landing_params:
                edit.setText(str(self.auto_landing_params[key]))

        # Update calibration display
        self.update_calibration_display()

        # Load all marker calibrations (unified method)
        self.load_marker_calibrations()

        # The unified method now handles both angle and distance calibrations
        # No need to call load_marker_distance_calibrations separately

    def _init_pid_controllers(self):
        gains = self.current_pid_gains
        self.roll_pid = PIDController(gains.get('roll_p', 0), gains.get('roll_i', 0), gains.get('roll_d', 0))
        self.pitch_pid = PIDController(gains.get('pitch_p', 0), gains.get('pitch_i', 0), gains.get('pitch_d', 0))
        # Altitude PID gains now adjustable from UI
        self.alt_pid = PIDController(
            Kp=gains.get('alt_p', 0.1),
            Ki=gains.get('alt_i', 0.02),
            Kd=gains.get('alt_d', 0.05),
            output_limits=(-15, 15)
        ) # Output is target pitch angle
        self.yaw_pid = PIDController(gains.get('yaw_p', 0), gains.get('yaw_i', 0), gains.get('yaw_d', 0))
        print("PID controllers initialized/updated.")

    def update_and_save_pid_gains(self):
        try:
            # 現在のPIDゲインを更新
            for _, key, _ in self.GAINS_TO_TUNE:
                value_str = self.pid_gain_edits[key].text()
                self.current_pid_gains[key] = float(value_str)

            # 既存のcoef.txtデータを読み込み
            existing_data = {}
            try:
                with open("coef.txt", "r") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                pass

            # PIDゲインを既存データに追加/更新
            for _, key, _ in self.GAINS_TO_TUNE:
                existing_data[key] = self.pid_gain_edits[key].text()

            # ファイルに保存
            with open("coef.txt", "w") as f:
                json.dump(existing_data, f, indent=4)

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
        try:
            # 現在の自動操縦パラメータを更新
            for _, key, _ in self.AUTOPILOT_PARAMS:
                value_str = self.autopilot_param_edits[key].text()
                self.current_autopilot_params[key] = float(value_str)

            # 既存のcoef.txtデータを読み込み
            existing_data = {}
            try:
                with open("coef.txt", "r") as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                pass

            # 自動操縦パラメータを既存データに追加/更新
            for _, key, _ in self.AUTOPILOT_PARAMS:
                existing_data[key] = self.autopilot_param_edits[key].text()

            # ファイルに保存
            with open("coef.txt", "w") as f:
                json.dump(existing_data, f, indent=4)

            print(f"Autopilot parameters updated and saved: {self.current_autopilot_params}")
        except ValueError as e:
            print(f"自動操縦パラメータの値が不正です: {e}")
        except Exception as e:
            print(f"自動操縦パラメータの保存中にエラーが発生しました: {e}")
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

    def load_auto_landing_params(self):
        """Load auto landing parameters from autoGoCoef.txt"""
        try:
            with open("autoGoCoef.txt", "r") as f:
                data = json.load(f)

            # Load control parameters
            params = data.get("parameters", {})
            for key, value in params.items():
                self.auto_landing_params[key] = float(value)

            # Load calibration data
            calib = data.get("calibration", {})
            if calib:
                self.calibration_data = calib

            print(f"Loaded auto landing parameters from autoGoCoef.txt")

        except FileNotFoundError:
            print("autoGoCoef.txt not found, using default auto landing parameters.")
            # Set default values
            for _, key, default_value in self.AUTO_LANDING_PARAMS:
                self.auto_landing_params[key] = float(default_value)
            print(f"Using default auto landing parameters: {self.auto_landing_params}")
        except Exception as e:
            print(f"自動離着陸パラメータの読み込み中にエラーが発生しました: {e}")
            # Set default values on error
            for _, key, default_value in self.AUTO_LANDING_PARAMS:
                self.auto_landing_params[key] = float(default_value)

    def save_auto_landing_params(self):
        """Save auto landing parameters to autoGoCoef.txt"""
        try:
            data = {
                "parameters": self.auto_landing_params,
                "calibration": self.calibration_data
            }
            with open("autoGoCoef.txt", "w") as f:
                json.dump(data, f, indent=2)
            print("Auto landing parameters saved to autoGoCoef.txt")
        except Exception as e:
            print(f"自動離着陸パラメータの保存中にエラーが発生しました: {e}")

    def estimate_distance_from_markers(self):
        """Estimate distance using ArUco marker data with individual marker calibrations"""
        valid_estimates = []

        # Check each marker and get individual distance estimates
        for marker_id, marker_data in self.aruco_markers.items():
            if marker_data['size'] > 0:  # Marker is detected
                try:
                    # Use individual marker distance estimation
                    distance_est = self.estimate_distance_from_marker(marker_id)
                    if distance_est is not None and distance_est > 0:
                        weight = marker_data['size']  # Larger markers get more weight
                        valid_estimates.append((distance_est, weight))
                        print(f"Marker {marker_id}: distance {distance_est:.2f}m, weight {weight}")
                except Exception as e:
                    print(f"Error estimating distance for marker {marker_id}: {e}")
                    continue

        if not valid_estimates:
            return 0.0

        # Calculate weighted average distance
        total_weighted_distance = sum(dist * weight for dist, weight in valid_estimates)
        total_weight = sum(weight for _, weight in valid_estimates)

        if total_weight > 0:
            weighted_average = total_weighted_distance / total_weight
            print(f"Distance estimate: {len(valid_estimates)} markers, result: {weighted_average:.2f}m")
            return weighted_average

        return 0.0

    def interpolate_distance_from_size(self, size):
        """Linear interpolation using 3 calibration points"""
        calib = self.calibration_data
        points = []

        # Get calibration points
        for point_name in ['point1', 'point2', 'point3']:
            if point_name in calib:
                points.append((calib[point_name]['distance'], calib[point_name]['size']))

        if len(points) < 2:
            # Fallback to simple inverse relationship
            return 100.0 / size if size > 0 else 0.0

        # Sort by distance
        points.sort()

        # Find the right interval for interpolation
        for i in range(len(points) - 1):
            dist1, size1 = points[i]
            dist2, size2 = points[i + 1]

            if size2 <= size <= size1:  # Larger size = smaller distance
                # Linear interpolation
                if size1 != size2:
                    ratio = (size - size2) / (size1 - size2)
                    distance = dist2 + ratio * (dist1 - dist2)
                    return distance

        # Extrapolation
        if len(points) >= 2:
            if size > points[0][1]:  # Closer than closest calibration point
                dist1, size1 = points[0]
                dist2, size2 = points[1]
            else:  # Farther than farthest calibration point
                dist1, size1 = points[-2]
                dist2, size2 = points[-1]

            if size1 != size2:
                ratio = (size - size2) / (size1 - size2)
                distance = dist2 + ratio * (dist1 - dist2)
                return max(0.0, distance)  # Don't allow negative distances

        return 0.0

    def calculate_aircraft_position(self):
        """Calculate aircraft position from ArUco marker data and estimated distance"""
        distance = self.estimate_distance_from_markers()

        if distance <= 0:
            return

        self.estimated_distance = distance

        # For now, use simple positioning based on visible markers
        # This is a simplified implementation - in reality you'd need more sophisticated
        # camera calibration and pose estimation

        valid_markers = []
        for marker_id, marker_data in self.aruco_markers.items():
            if marker_data['size'] > 0:
                valid_markers.append(marker_id)

        if not valid_markers:
            return

        # Simple positioning logic based on camera coordinates
        # Camera image is 400x300, center at (200, 150)
        camera_center_x = 200
        camera_center_y = 150

        # Use marker ID 2 (center marker) for primary positioning if available
        if 2 in valid_markers:
            marker2 = self.aruco_markers[2]
            # Convert camera pixel coordinates to real world offset
            # This is a simplified conversion - real implementation would use camera calibration
            dx = (marker2['x'] - camera_center_x) * distance * 0.01  # Scale factor
            dy = (marker2['y'] - camera_center_y) * distance * 0.01

            # Aircraft position relative to marker ID 2 (which is at origin)
            self.current_position['x'] = -dx  # Negative because camera view is inverted
            self.current_position['y'] = distance  # Distance from runway

            # 高度データの取得とバンク角補正
            raw_alt = self.latest_attitude.get('alt', 0)
            current_roll = self.latest_attitude.get('roll', 0)
            corrected_alt = self.get_corrected_altitude(raw_alt, current_roll)
            self.current_position['z'] = corrected_alt / 1000.0  # Convert mm to m

        else:
            # Use other markers for positioning (simplified)
            self.current_position['y'] = distance
            raw_alt = self.latest_attitude.get('alt', 0)
            current_roll = self.latest_attitude.get('roll', 0)
            corrected_alt = self.get_corrected_altitude(raw_alt, current_roll)
            self.current_position['z'] = corrected_alt / 1000.0

        # Update position visualization
        if hasattr(self, 'xy_position_widget') and hasattr(self, 'zy_position_widget'):
            self.xy_position_widget.set_aircraft_position(
                self.current_position['x'],
                self.current_position['y'],
                self.current_position['z']
            )
            self.zy_position_widget.set_aircraft_position(
                self.current_position['x'],
                self.current_position['y'],
                self.current_position['z']
            )

            # Update marker detection status
            for marker_id, marker_data in self.aruco_markers.items():
                detected = marker_data['size'] > 0
                self.xy_position_widget.set_marker_detection(marker_id, detected)
                self.zy_position_widget.set_marker_detection(marker_id, detected)

    def update_auto_landing_phase(self):
        """Update auto landing phase based on estimated distance"""
        distance = self.estimated_distance

        # Always update distance and altitude display regardless of auto-landing status
        if hasattr(self, 'distance_label'):
            self.distance_label.setText(f"推定距離: {distance:.1f} m")

        # Update altitude debug info
        if hasattr(self, 'altitude_debug_label'):
            current_alt_mm_raw = self.latest_attitude.get('alt', 0)
            current_roll = self.latest_attitude.get('roll', 0)
            current_alt_mm = self.get_corrected_altitude(current_alt_mm_raw, current_roll)
            current_alt_m = current_alt_mm / 1000.0
            self.altitude_debug_label.setText(f"高度: {current_alt_m:.2f} m ({current_alt_mm:.0f} mm補正済)")

        if not self.auto_landing_enabled:
            self.auto_landing_phase = 0  # Manual
            # Update phase display for manual mode
            if hasattr(self, 'phase_label'):
                self.phase_label.setText("フェーズ: 手動")
            return

        # Phase transition logic based on distance thresholds
        takeoff_threshold = self.auto_landing_params.get('takeoff_distance_threshold', 30.0)
        drop_threshold = self.auto_landing_params.get('drop_distance_threshold', 20.0)
        steady_threshold = self.auto_landing_params.get('steady_distance_threshold', 10.0)
        landing_threshold = self.auto_landing_params.get('landing_distance_threshold', 5.0)

        if distance > takeoff_threshold:
            new_phase = 1  # Takeoff
        elif distance > drop_threshold:
            new_phase = 2  # Drop
        elif distance > landing_threshold:
            new_phase = 3  # Steady
        else:
            new_phase = 4  # Landing

        if new_phase != self.auto_landing_phase:
            print(f"Auto landing phase changed from {self.auto_landing_phase} to {new_phase}")
            self.auto_landing_phase = new_phase

        # Update phase display for auto-landing modes
        phase_names = {0: "手動", 1: "離陸", 2: "投下", 3: "定常", 4: "着陸"}
        if hasattr(self, 'phase_label'):
            self.phase_label.setText(f"フェーズ: {phase_names.get(self.auto_landing_phase, '不明')}")

    def run_auto_landing_control(self):
        """Execute auto landing control logic based on current phase"""
        if not self.auto_landing_enabled or not self.is_connected:
            return

        # Calculate position and distance
        self.calculate_aircraft_position()

        # Update phase based on distance
        self.update_auto_landing_phase()

        raw_alt = self.latest_attitude.get('alt', 0)
        current_roll = self.latest_attitude.get('roll', 0)
        corrected_alt = self.get_corrected_altitude(raw_alt, current_roll)
        current_alt = corrected_alt / 1000.0  # Convert mm to m
        current_pitch = self.latest_attitude.get('pitch', 0)
        current_yaw = self.latest_attitude.get('yaw', 0)

        # Default control values
        target_roll = 0.0  # Always keep wings level
        target_pitch = 0.0
        target_throttle = self.auto_landing_params.get('pre_drop_throttle', 700.0)
        target_rudder = 0.0

        if self.auto_landing_phase == 1:  # Takeoff phase
            target_throttle = self.auto_landing_params.get('takeoff_throttle', 1000.0)
            # Climb to steady altitude
            steady_alt = self.auto_landing_params.get('steady_altitude', 1.5)
            if current_alt < steady_alt:
                target_pitch = 5.0  # Nose up for climb

        elif self.auto_landing_phase == 2:  # Drop phase
            target_throttle = self.auto_landing_params.get('pre_drop_throttle', 700.0)
            # Maintain steady altitude
            steady_alt = self.auto_landing_params.get('steady_altitude', 1.5)
            altitude_error = steady_alt - current_alt
            alt_gain = self.auto_landing_params.get('altitude_gain', 50.0)
            target_throttle += altitude_error * alt_gain

        elif self.auto_landing_phase == 3:  # Steady phase
            target_throttle = self.auto_landing_params.get('post_drop_throttle', 650.0)
            # Maintain altitude and center on runway (X=0)
            steady_alt = self.auto_landing_params.get('steady_altitude', 1.5)
            altitude_error = steady_alt - current_alt
            alt_gain = self.auto_landing_params.get('altitude_gain', 50.0)
            target_throttle += altitude_error * alt_gain

            # Use rudder to center on runway (marker ID 2 should be at camera center)
            if 2 in self.aruco_markers and self.aruco_markers[2]['size'] > 0:
                marker_x = self.aruco_markers[2]['x']
                camera_center_x = 200  # Camera center
                x_error = marker_x - camera_center_x
                rudder_gain = self.auto_landing_params.get('rudder_gain', 0.5)
                target_rudder = -x_error * rudder_gain / 200.0  # Normalize to [-1, 1]
                target_rudder = max(-1.0, min(1.0, target_rudder))

        elif self.auto_landing_phase == 4:  # Landing phase
            target_throttle = self.auto_landing_params.get('post_drop_throttle', 650.0) * 0.8  # Reduce throttle
            # Gentle descent
            target_pitch = -2.0  # Slight nose down

            # Continue centering on runway
            if 2 in self.aruco_markers and self.aruco_markers[2]['size'] > 0:
                marker_x = self.aruco_markers[2]['x']
                camera_center_x = 200
                x_error = marker_x - camera_center_x
                rudder_gain = self.auto_landing_params.get('rudder_gain', 0.5)
                target_rudder = -x_error * rudder_gain / 200.0
                target_rudder = max(-1.0, min(1.0, target_rudder))

        # Apply control limits
        target_throttle = max(400, min(1000, int(target_throttle)))
        target_roll = max(-30, min(30, target_roll))
        target_pitch = max(-15, min(15, target_pitch))

        # Convert to RC commands using existing PID controllers
        dt = 0.05
        self.roll_pid.setpoint = target_roll
        self.pitch_pid.setpoint = target_pitch

        aileron_cmd = self.roll_pid.update(current_roll, dt)
        elevator_cmd = self.pitch_pid.update(current_pitch, dt)

        # Convert normalized commands to RC values
        aileron_rc = self.denormalize_symmetrical(aileron_cmd, 'ail')
        elevator_rc = self.denormalize_symmetrical(elevator_cmd, 'elev')
        rudder_rc = self.denormalize_symmetrical(target_rudder, 'rudd')
        throttle_rc = int(target_throttle)

        # Send commands
        commands = {
            'ail': aileron_rc,
            'elev': elevator_rc,
            'rudd': rudder_rc,
            'thro': throttle_rc
        }
        self.send_serial_command(commands)

        # Update auto landing stick displays
        if hasattr(self, 'auto_left_stick') and hasattr(self, 'auto_right_stick'):
            self.auto_left_stick.set_autopilot_position(target_rudder, elevator_cmd)
            self.auto_left_stick.set_autopilot_active(True)

            throttle_norm = (target_throttle - 400) / 600.0 - 1.0  # Normalize throttle
            self.auto_right_stick.set_autopilot_position(aileron_cmd, throttle_norm)
            self.auto_right_stick.set_autopilot_active(True)

        # Update manual piloting tab stick displays (same values as auto landing)
        if hasattr(self, 'manual_left_stick') and hasattr(self, 'manual_right_stick'):
            self.manual_left_stick.set_autopilot_position(target_rudder, elevator_cmd)
            self.manual_left_stick.set_autopilot_active(True)

            throttle_norm = (target_throttle - 400) / 600.0 - 1.0  # Normalize throttle
            self.manual_right_stick.set_autopilot_position(aileron_cmd, throttle_norm)
            self.manual_right_stick.set_autopilot_active(True)

        # Update stick labels
        if hasattr(self, 'auto_left_stick_label'):
            self.auto_left_stick_label.setText(f"R: {rudder_rc}, E: {elevator_rc}")
        if hasattr(self, 'auto_right_stick_label'):
            self.auto_right_stick_label.setText(f"A: {aileron_rc}, T: {throttle_rc}")

        # Update manual piloting tab stick labels
        if hasattr(self, 'manual_left_stick_label'):
            self.manual_left_stick_label.setText(f"R: {rudder_rc}, E: {elevator_rc}")
        if hasattr(self, 'manual_right_stick_label'):
            self.manual_right_stick_label.setText(f"A: {aileron_rc}, T: {throttle_rc}")

        # Update auto landing attitude displays
        if hasattr(self, 'auto_pitch_widget') and hasattr(self, 'auto_yaw_widget'):
            roll = self.latest_attitude.get('roll', 0)
            pitch = self.latest_attitude.get('pitch', 0)
            yaw = self.latest_attitude.get('yaw', 0)

            self.auto_pitch_widget.set_angle(pitch)
            self.auto_yaw_widget.set_angle(yaw)
            self.auto_pitch_widget.set_autopilot_active(self.auto_landing_enabled)
            self.auto_yaw_widget.set_autopilot_active(self.auto_landing_enabled)

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
        for i in range(1, 6):  # AUX1からAUX5まで（5つ）
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
            # 自動操縦時は開始角度 + 総回転角で表示
            display_yaw = self.mission_start_yaw + self.mission_total_rotation
            self.yaw_widget.set_angle(display_yaw)
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
            self.video_toggle_button.setText("FPV")
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
            if len(parts) == 25:  # 25パラメータに対応
                # パラメータ1-4: 姿勢・高度データ
                roll, pitch, yaw, alt = parts[0:4]

                # パラメータ5-8: プロポ入力データ（送信側順序に対応）
                ail, elev, thro, rudd = parts[4:8]

                # 現在のプロポ入力値を保存（停止時の手動復帰用）
                self.current_ail = float(ail)
                self.current_elev = float(elev)
                self.current_thro = float(thro)
                self.current_rudd = float(rudd)
                if len(parts) > 8:
                    self.current_aux1 = float(parts[8])

                # Handle manual piloting flight state replay - must be before regular processing
                if self.input_log_replaying and self.loaded_input_log:
                    current_time = time.time() - self.input_log_replay_start_time

                    # Find the current data point to apply (設定間隔データの適用)
                    current_data = None
                    for i, data in enumerate(self.loaded_input_log):
                        if data['elapsed_time'] <= current_time:
                            current_data = data
                            self.input_log_replay_index = i + 1
                        else:
                            break

                    # Apply current replay data if available - send target flight state instead of propeller inputs
                    if current_data:
                        # 手動操縦ログ再現では飛行状態目標値を設定（自動制御で達成）
                        target_altitude = float(current_data['altitude'])
                        target_yaw_displacement = float(current_data['yaw'])  # 記録された変位角
                        target_throttle = float(current_data['throttle'])
                        target_aux1 = float(current_data['aux1'])

                        # 変位角を再現開始時の基準角度に対する絶対角度に変換
                        target_yaw_absolute = getattr(self, 'replay_start_yaw', 0.0) + target_yaw_displacement

                        # Set flight state targets for manual piloting replay system
                        self.target_altitude = target_altitude
                        self.target_yaw_angle = target_yaw_absolute

                        # ログ再現飛行専用制御: ロール角は常に0度目標、ヨー角はラダーで制御
                        self.replay_target_roll = 0.0  # ロール角は常に0度
                        self.replay_target_yaw = target_yaw_absolute  # ヨー角目標

                        # 制御コマンドを計算してpropsに設定
                        dt = 0.02  # 50Hz制御ループ想定
                        current_roll = self.latest_attitude.get('roll', 0.0)
                        current_yaw = self.latest_attitude.get('yaw', 0.0)

                        # ロール角制御（水平旋回パラメータを使用）
                        self.roll_pid.setpoint = self.replay_target_roll
                        ail_command = self.roll_pid.update(current_roll, dt)

                        # ヨー角制御（ラダーで直接制御、水平旋回パラメータを使用）
                        yaw_error = target_yaw_absolute - current_yaw
                        # ヨー角誤差を-180~+180度に正規化
                        while yaw_error > 180:
                            yaw_error -= 360
                        while yaw_error < -180:
                            yaw_error += 360

                        self.yaw_pid.setpoint = 0.0  # 誤差を0にするように制御
                        rudd_command = self.yaw_pid.update(-yaw_error, dt)  # 負号で方向調整

                        # 高度制御（エレベータ）
                        current_alt = self.get_corrected_altitude(
                            self.latest_attitude.get('alt', self.mission_start_altitude),
                            current_roll
                        )
                        altitude_error = target_altitude - current_alt
                        target_pitch_from_alt = self.alt_pid.update(current_alt, dt)
                        self.pitch_pid.setpoint = target_pitch_from_alt
                        elev_command = self.pitch_pid.update(self.latest_attitude.get('pitch', 0.0), dt)

                        # 制御出力を他の自動ミッションと同じ方式で変換（RC_RANGES使用）
                        ail_pwm = int(self.denormalize_symmetrical(ail_command, 'ail'))
                        elev_pwm = int(self.denormalize_symmetrical(elev_command, 'elev'))
                        rudd_pwm = int(self.denormalize_symmetrical(rudd_command, 'rudd'))
                        thro_pwm = int(target_throttle)  # スロットルはログデータをそのまま使用

                        # PWM値を実測範囲内に制限（他の自動ミッションと同じ制限）
                        ail_pwm = max(self.RC_RANGES['ail']['min_in'], min(self.RC_RANGES['ail']['max_in'], ail_pwm))
                        elev_pwm = max(self.RC_RANGES['elev']['min_in'], min(self.RC_RANGES['elev']['max_in'], elev_pwm))
                        rudd_pwm = max(self.RC_RANGES['rudd']['min_in'], min(self.RC_RANGES['rudd']['max_in'], rudd_pwm))
                        # スロットルは範囲制限なし（ログデータをそのまま使用）

                        # partsを数値として更新（文字列変換は行わない）
                        parts[4] = float(ail_pwm)   # AIL
                        parts[5] = float(elev_pwm)  # ELEV
                        parts[6] = float(thro_pwm)  # THRO
                        parts[7] = float(rudd_pwm)  # RUDD

                        # Override AUX1 directly
                        if len(parts) > 8:
                            parts[8] = float(int(target_aux1))  # AUX1

                        # 制御コマンドを即座に送信（ミッションモード4で送信）
                        replay_commands = {
                            'ail': ail_pwm,
                            'elev': elev_pwm,
                            'rudd': rudd_pwm,
                            'thro': thro_pwm,
                            'aux1': int(target_aux1)  # AUX1（物資投下）も再現
                        }

                        # ミッションモードを一時的に4に設定して送信
                        original_mission_mode = self.active_mission_mode
                        self.active_mission_mode = 4

                        self.send_serial_command(replay_commands)

                        # ミッションモードを元に戻す
                        self.active_mission_mode = original_mission_mode

                        # デバッグ出力は設定間隔ごと（データが更新された時のみ）
                        if not hasattr(self, '_last_replay_data') or self._last_replay_data != current_data:
                            print(f"Manual Piloting Replay: Alt:{target_altitude:.1f}mm, Yaw:{target_yaw_absolute:.1f}°(Δ{target_yaw_displacement:.1f}°), Roll:0.0°, Controls: A{ail_pwm} E{elev_pwm} T{thro_pwm} R{rudd_pwm} AUX1:{int(target_aux1)}, Time:{current_time:.1f}s")
                            self._last_replay_data = current_data

                        # ログ再現飛行中は制御コマンドを生成したので、後続の処理は継続
                        # return を削除して通常の処理を継続

                    # Check if replay is complete (time-based)
                    if (self.loaded_input_log and
                        current_time >= self.loaded_input_log[-1]['elapsed_time']):
                        self.input_log_replaying = False

                        # Stop mission when replay is complete
                        self.stop_mission()

                        # 即座にミッションモード0（手動）を送信
                        if self.is_connected:
                            current_commands = {
                                'ail': self.current_ail,
                                'elev': self.current_elev,
                                'thro': self.current_thro,
                                'rudd': self.current_rudd,
                                'aux1': getattr(self, 'current_aux1', 1500)  # AUX1も含める
                            }
                            self.send_serial_command(current_commands)
                            print("ミッションモード0（手動）に即座に切り替えました")

                        self.replay_button.setText("ログ再現飛行開始")
                        self.replay_button.setStyleSheet("")
                        self.replay_status_label.setText("再現完了")
                        self.replay_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
                        print("ログ再現飛行が完了しました (ミッション停止)")

                # 手動操縦ログ記録（飛行状態データ記録、設定可能間隔）
                if self.input_log_recording:
                    current_time = time.time()
                    if self.input_log_start_time == 0:
                        self.input_log_start_time = current_time
                        self.input_log_last_record_time = current_time

                    # 設定した間隔経過した場合のみ記録
                    if current_time - self.input_log_last_record_time >= self.input_log_interval:
                        elapsed_time = current_time - self.input_log_start_time
                        # ヨー角の変位（ミッション開始時からの相対角度）を計算
                        current_yaw = self.latest_attitude.get('yaw', 0.0)
                        yaw_displacement = current_yaw - self.mission_start_yaw
                        # 角度を-180~+180度の範囲に正規化
                        while yaw_displacement > 180:
                            yaw_displacement -= 360
                        while yaw_displacement < -180:
                            yaw_displacement += 360

                        log_entry = {
                            'timestamp': current_time,
                            'elapsed_time': elapsed_time,
                            'altitude': self.latest_attitude.get('alt', 0),
                            'yaw': yaw_displacement,  # 変位角を記録
                            'throttle': int(float(thro)),
                            'aux1': int(float(parts[8])) if len(parts) > 8 else 1000
                        }
                        self.input_log_data.append(log_entry)
                        self.input_log_last_record_time = current_time

                # パラメータ9-13: AUXスイッチ（AUX5追加）
                aux1, aux2, aux3, aux4, aux5 = parts[8:13]

                # パラメータ14-25: ArUcoマーカー情報
                aruco1_size, aruco1_id, aruco1_x, aruco1_y = parts[13:17]
                aruco2_size, aruco2_id, aruco2_x, aruco2_y = parts[17:21]
                aruco3_size, aruco3_id, aruco3_x, aruco3_y = parts[21:25]

                # ArUcoマーカー情報を保存
                self.aruco_markers[1] = {'size': aruco1_size, 'id': int(aruco1_id), 'x': aruco1_x, 'y': aruco1_y}
                self.aruco_markers[2] = {'size': aruco2_size, 'id': int(aruco2_id), 'x': aruco2_x, 'y': aruco2_y}
                self.aruco_markers[3] = {'size': aruco3_size, 'id': int(aruco3_id), 'x': aruco3_x, 'y': aruco3_y}

                # リアルタイムマーカーデータ更新
                self.update_marker_realtime_display()

                # Position visualization widgets にマーカーデータを更新
                self.xy_position_widget.update_marker_data(1, aruco1_size, aruco1_x, aruco1_y)
                self.xy_position_widget.update_marker_data(2, aruco2_size, aruco2_x, aruco2_y)
                self.xy_position_widget.update_marker_data(3, aruco3_size, aruco3_x, aruco3_y)

                self.zy_position_widget.update_marker_data(1, aruco1_size, aruco1_x, aruco1_y)
                self.zy_position_widget.update_marker_data(2, aruco2_size, aruco2_x, aruco2_y)
                self.zy_position_widget.update_marker_data(3, aruco3_size, aruco3_x, aruco3_y)

                # 前回値を取得
                prev = self.latest_attitude
                prev_roll = prev.get('roll', 0.0)
                prev_pitch = prev.get('pitch', 0.0)
                prev_yaw = prev.get('yaw', 0.0)
                prev_alt = prev.get('alt', 0.0)

                # 姿勢データのフィルタリング（角度エラーの可能性がある場合は前回値を使用）
                # Pitch: -0.2 < pitch < 0.2 の場合は前回値を使用
                if -0.2 < pitch < 0.2:
                    pitch = prev_pitch

                # Roll: -0.2 < roll < 0.2 の場合は前回値を使用
                if -0.2 < roll < 0.2:
                    roll = prev_roll

                # Yaw: 359.8 < yaw または yaw < 0.2 の場合は前回値を使用
                if yaw > 359.8 or yaw < 0.2:
                    yaw = prev_yaw

                # 高度データの処理（0.0の場合のみ前回値を使用、それ以外は送信データをそのまま使用）
                if alt != 0.0:
                    filtered_alt = alt
                    if abs(alt - prev_alt) > 1000.0 and prev_alt != 0.0:  # 1m以上変化した場合にログ出力
                        print(f"高度変化: {prev_alt:.0f}mm -> {alt:.0f}mm (差: {alt-prev_alt:.0f}mm)")
                else:
                    filtered_alt = prev_alt
                    print(f"高度データが0.0のため前回値({prev_alt:.0f}mm)を使用")

                self.latest_attitude = {'roll': roll, 'pitch': pitch, 'yaw': yaw, 'alt': filtered_alt}

                # 高度計には補正済み高度を表示
                corrected_alt = self.get_corrected_altitude(filtered_alt, roll)
                self.adi_widget.set_attitude(roll, pitch)
                self.altimeter_widget.set_altitude(corrected_alt)
                self.heading_label.setText(f"方位: {yaw:.1f} °")
                self.update_aux_switches([aux1, aux2, aux3, aux4, aux5])

                # Store current flight state data for auto landing log
                self.current_altitude = filtered_alt
                self.current_yaw = yaw
                self.current_throttle = float(thro)
                self.current_aux1 = float(aux1)

                # Update input replay tab displays
                self.update_input_replay_displays(float(ail), float(elev), float(thro), float(rudd), float(aux1))

                rud_norm = self.normalize_symmetrical(float(rudd), **self.RC_RANGES['rudd'])
                ele_norm = self.normalize_symmetrical(float(elev), **self.RC_RANGES['elev'])
                rud_norm = -rud_norm
                ele_norm = -ele_norm
                self.left_stick.set_position(rud_norm, ele_norm)
                self.left_stick_label.setText(f"R: {int(float(rudd))}, E: {int(float(elev))}")

                ail_norm = self.normalize_symmetrical(float(ail), **self.RC_RANGES['ail'])
                thr_norm = self.normalize_value(float(thro), **self.RC_RANGES['thro'])
                self.right_stick.set_position(ail_norm, thr_norm)
                self.right_stick_label.setText(f"A: {int(float(ail))}, T: {int(float(thro))}")

                # スティックウィジェットに自動操縦状態を設定
                self.left_stick.set_autopilot_active(self.autopilot_active)
                self.right_stick.set_autopilot_active(self.autopilot_active)
                self.check_mission_triggers([aux1, aux2, aux3, aux4, aux5])
            else:
                print(f"データ形式エラー: 25パラメータが必要ですが、{len(parts)}パラメータを受信しました - {line}")
        except (ValueError, IndexError) as e:
            print(f"データ解析エラー: {e} - {line}")

    def update_aux_switches(self, aux_values):
        on_style = "background-color: #28a745; color: white; padding: 5px; border-radius: 3px;"
        off_style = "background-color: #555; color: white; padding: 5px; border-radius: 3px;"

        mission_text = "なし"
        missions = { 2: "水平旋回", 3: "上昇旋回", 4: "八の字旋回", 5: "自動離着陸" } # AUX switch number to mission name

        for i, value in enumerate(aux_values):
            if i < len(self.aux_labels):  # 既存のAUXラベル数に制限
                label = self.aux_labels[i]
                is_on = float(value) > 1100
                aux_number = i + 1  # AUX1, AUX2, AUX3, AUX4, AUX5

                if is_on:
                    label.setText(f"AUX{aux_number}: ON")
                    label.setStyleSheet(on_style)
                    # Check if this AUX is assigned to a mission
                    if aux_number in missions:
                        mission_text = missions[aux_number]
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
        # AUX5: 自動離着陸（Auto Landing）
        mission_map = {
            2: 1, # AUX2 -> Mission 1 (水平旋回)
            3: 2, # AUX3 -> Mission 2 (上昇旋回)
            4: 3  # AUX4 -> Mission 3 (八の字旋回)
        }

        # AUX5の自動離着陸チェック（最優先）
        if len(aux_values) >= 5 and float(aux_values[4]) > 1100:  # AUX5がON
            if not self.auto_landing_enabled:
                # 自動離着陸を有効化
                self.auto_landing_enabled = True
                if hasattr(self, 'auto_landing_enable_button'):
                    self.auto_landing_enable_button.setChecked(True)
                    self.auto_landing_enable_button.setText("自動離着陸無効化")
                print("AUX5により自動離着陸が有効化されました")
                # Update replay button state when AUX5 is enabled
                self.update_replay_button_state()

            # 自動離着陸が有効な場合のミッションモード制御
            # 操縦量記録中は手動（ミッションモード0）、再現飛行中のみミッションモード4
            if self.input_log_replaying or self.auto_landing_log_replaying:
                # 再現飛行中はミッションモード4
                if not self.autopilot_active or self.active_mission_mode != 4:
                    self.start_mission(4)  # 再現飛行用ミッションモード4を開始
            else:
                # 操縦量記録中または通常の自動離着陸時は手動（ミッションモード0）
                if self.autopilot_active and self.active_mission_mode != 0:
                    self.stop_mission()  # 他のミッションを停止して手動に戻す

                # 飛行状態記録処理
                if self.auto_landing_log_recording:
                    self.record_auto_landing_log_data()
        else:
            if self.auto_landing_enabled:
                # 自動離着陸を無効化
                self.auto_landing_enabled = False
                if hasattr(self, 'auto_landing_enable_button'):
                    self.auto_landing_enable_button.setChecked(False)
                    self.auto_landing_enable_button.setText("自動離着陸有効化")
                print("AUX5により自動離着陸が無効化されました")
                # Update replay button state when AUX5 is disabled
                self.update_replay_button_state()

                # AUX5無効時に即座に手動操縦に戻る処理
                self.emergency_return_to_manual()

                # 自動離着陸が無効になった場合、アクティブなミッションを停止する
                if self.autopilot_active:
                    self.stop_mission()

        # 既存のミッション処理（AUX2-4）
        # 再現飛行中のみ他のミッションを無視、記録中や通常時は他のミッションも有効
        if not (self.auto_landing_enabled and self.input_log_replaying):
            mission_found = False
            # Prioritize lower AUX numbers if multiple are on
            for aux_number in sorted(mission_map.keys()):
                aux_index = aux_number - 1
                if aux_index < len(aux_values) and float(aux_values[aux_index]) > 1100: # If this mission switch is ON
                    mission_found = True
                    prev_val = self.previous_aux_values[aux_index] if aux_index < len(self.previous_aux_values) else 0
                    if float(prev_val) < 1100: # And it was previously OFF (rising edge)
                        # 新ミッション開始時に回転角をリセット
                        self.mission_total_rotation = 0.0
                        self.start_mission(mission_map[aux_number])
                    break # Only handle one mission at a time

            if not mission_found:
                self.stop_mission()

        # previous_aux_valuesをaux_valuesの長さに合わせて更新（数値として保存）
        self.previous_aux_values = [float(val) for val in aux_values]

    def start_mission(self, mission_mode):
        if self.autopilot_active and self.active_mission_mode == mission_mode:
            return
        print(f"Starting Mission: {mission_mode}")
        self.autopilot_active = True
        self.active_mission_mode = mission_mode
        self.mission_start_yaw = self.latest_attitude.get('yaw', 0)
        raw_alt = self.latest_attitude.get('alt', 0)
        current_roll = self.latest_attitude.get('roll', 0)
        self.mission_start_altitude = self.get_corrected_altitude(raw_alt, current_roll)
        # 新しいヨー角追跡システムの初期化
        self.previous_yaw = self.mission_start_yaw
        self.mission_total_rotation = 0.0

        print(f"ミッション開始: 開始ヨー角={self.mission_start_yaw:.1f}°, 開始高度={self.mission_start_altitude/1000.0:.2f}m")

        # 八の字旋回の状態をリセット
        self.figure8_phase = 0  # 右旋回から開始
        self.figure8_completed = False
        self.figure8_phase_start_rotation = 0.0  # 左旋回フェーズ開始時の総回転角

        # 上昇旋回の状態をリセット
        self.ascending_phase = 0  # 0: 2.5m左旋回2回, 1: 上昇中旋回, 2: 5m左旋回2回
        self.ascending_completed = False
        self.ascending_phase_start_rotation = 0.0  # フェーズ開始時の総回転角

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

    def emergency_return_to_manual(self):
        """AUX5無効時に即座に手動操縦に戻る緊急処理"""
        print("緊急手動復帰: AUX5が無効になりました")

        # 進行中の全ての自動操作を停止

        # 1. ログ再現飛行を停止
        if self.input_log_replaying:
            self.input_log_replaying = False
            self.replay_button.setText("ログ再現飛行開始")
            self.replay_button.setStyleSheet("")
            self.replay_status_label.setText("緊急停止: AUX5無効")
            self.replay_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            print("手動操縦ログ再現を緊急停止しました")

        # 2. 自動離着陸ログ再現を停止
        if hasattr(self, 'auto_landing_log_replaying') and self.auto_landing_log_replaying:
            self.auto_landing_log_replaying = False
            print("自動離着陸ログ再現を緊急停止しました")

        # 3. 操縦記録を停止
        if self.input_log_recording:
            self.input_log_recording = False
            self.record_button.setChecked(False)
            self.record_button.setText("操縦記録開始")
            self.record_status_label.setText("緊急停止: AUX5無効")
            self.record_status_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            print("手動操縦記録を緊急停止しました")

        # 4. 自動離着陸ログ記録を停止
        if hasattr(self, 'auto_landing_log_recording') and self.auto_landing_log_recording:
            self.auto_landing_log_recording = False
            print("自動離着陸ログ記録を緊急停止しました")

        # 5. 即座に現在のプロポ入力値で手動操縦コマンドを送信
        if self.is_connected:
            manual_commands = {
                'ail': int(self.current_ail),
                'elev': int(self.current_elev),
                'thro': int(self.current_thro),
                'rudd': int(self.current_rudd),
                'aux1': int(getattr(self, 'current_aux1', 1500))  # AUX1も含める
            }

            # ミッションモードを0（手動）に設定して送信
            original_mission_mode = self.active_mission_mode
            self.active_mission_mode = 0

            self.send_serial_command(manual_commands)
            print(f"緊急手動復帰コマンド送信: A{manual_commands['ail']} E{manual_commands['elev']} T{manual_commands['thro']} R{manual_commands['rudd']} AUX1{manual_commands['aux1']}")

        # 6. UI状態をリセット
        if hasattr(self, 'auto_left_stick') and hasattr(self, 'auto_right_stick'):
            self.auto_left_stick.set_autopilot_position(None, None)
            self.auto_left_stick.set_autopilot_active(False)
            self.auto_right_stick.set_autopilot_position(None, None)
            self.auto_right_stick.set_autopilot_active(False)

        if hasattr(self, 'manual_left_stick') and hasattr(self, 'manual_right_stick'):
            self.manual_left_stick.set_autopilot_position(None, None)
            self.manual_left_stick.set_autopilot_active(False)
            self.manual_right_stick.set_autopilot_position(None, None)
            self.manual_right_stick.set_autopilot_active(False)

        # 7. ラベル表示をリセット
        if hasattr(self, 'auto_left_stick_label'):
            self.auto_left_stick_label.setText("R: 0, E: 0")
        if hasattr(self, 'auto_right_stick_label'):
            self.auto_right_stick_label.setText("A: 0, T: 0")
        if hasattr(self, 'manual_left_stick_label'):
            self.manual_left_stick_label.setText("R: 0, E: 0")
        if hasattr(self, 'manual_right_stick_label'):
            self.manual_right_stick_label.setText("A: 0, T: 0")

    def run_autopilot_cycle(self):
        if not self.is_connected:
            return

        # Process auto landing log replay
        self.process_auto_landing_log_replay()

        if not self.autopilot_active:
            if self.last_autopilot_commands:
                self.send_serial_command(self.last_autopilot_commands)
            return

        raw_alt = self.latest_attitude.get('alt', self.mission_start_altitude)
        current_roll = self.latest_attitude.get('roll', 0)
        corrected_alt = self.get_corrected_altitude(raw_alt, current_roll)
        current_alt = corrected_alt
        current_roll = self.latest_attitude.get('roll', 0)
        current_pitch = self.latest_attitude.get('pitch', 0)
        current_yaw = self.latest_attitude.get('yaw', 0)
        dt = 0.05 # 50ms interval

        # 新しいヨー角追跡システム
        current_yaw = self.latest_attitude.get('yaw', 0)

        # ヨー角の変化量を計算（0-359度システム対応）
        yaw_change = current_yaw - self.previous_yaw

        # 360度境界を跨ぐ場合の補正（最短角度で計算）
        # 大きな値の飛びは実際の回転ではなく境界クロッシングとして処理
        if yaw_change > 180:
            yaw_change -= 360  # 例: 350° → 10°の変化 = 20°（右回り）を -340° → 20°に補正
        elif yaw_change < -180:
            yaw_change += 360  # 例: 10° → 350°の変化 = 340°（左回り）を -20°に補正

        # 総回転角に加算（右回り正、左回り負）
        self.mission_total_rotation += yaw_change
        self.previous_yaw = current_yaw

        # デバッグ出力（自動操縦時のみ、1秒間隔で出力）
        if self.autopilot_active:
            self.yaw_debug_counter += 1
            if self.yaw_debug_counter >= 20:  # 50ms * 20 = 1秒間隔
                print(f"YAW DEBUG: current={current_yaw:.1f}°, change={yaw_change:.1f}°, total_rotation={self.mission_total_rotation:.1f}°, start={self.mission_start_yaw:.1f}°")
                self.yaw_debug_counter = 0

        target_roll = 0
        target_pitch = 0

        # スロットル制御：高度誤差に基づく調整
        base_throttle = self.current_autopilot_params.get('autopilot_throttle_base', 700)
        altitude_gain = self.current_autopilot_params.get('altitude_throttle_gain', 20.0)
        throttle_min = self.current_autopilot_params.get('throttle_min', 400)
        throttle_max = self.current_autopilot_params.get('throttle_max', 1000)

        if self.active_mission_mode == 1: # Horizontal Turn (右旋回)
            target_roll = self.current_autopilot_params.get('bank_angle', 20)
            self.alt_pid.setpoint = self.mission_start_altitude
            # 水平旋回：ミッション開始時の高度を維持
            target_altitude = self.mission_start_altitude
            horizontal_target = self.current_autopilot_params.get('horizontal_turn_target', 760)

            # 水平旋回のデバッグ出力（5秒間隔）
            if self.yaw_debug_counter % 100 == 0:  # 50ms * 100 = 5秒間隔
                current_alt_m = current_alt / 1000.0
                print(f"水平旋回: 高度={current_alt_m:.2f}m, 総回転={self.mission_total_rotation:.1f}°, 目標={horizontal_target}°")

            # 右旋回なので、総回転角が目標角度以上になったら成功
            if self.mission_total_rotation >= horizontal_target:
                print(f"水平旋回成功: 開始角={self.mission_start_yaw:.1f}°, 現在角={current_yaw:.1f}°, 総回転角={self.mission_total_rotation:.1f}°")
                self.mission_status_label.setText("ミッション: 水平旋回 成功")
        elif self.active_mission_mode == 2: # Ascending Turn (4段階フェーズ制御)
            # 上昇旋回時は負のバンク角（左旋回）
            target_roll = -self.current_autopilot_params.get('bank_angle', 20)
            current_alt_m = current_alt / 1000.0  # mm -> m

            if not self.ascending_completed:
                if self.ascending_phase == 0:  # フェーズ0: 2.5m高度で左旋回2回（-720度）
                    target_altitude = 2.5  # 2.5m維持
                    phase_rotation = self.mission_total_rotation - self.ascending_phase_start_rotation

                    # フェーズ0のデバッグ出力（5秒間隔）
                    if self.yaw_debug_counter % 100 == 0:  # 50ms * 100 = 5秒間隔
                        print(f"上昇旋回フェーズ0: 高度={current_alt_m:.2f}m, フェーズ回転={phase_rotation:.1f}°, 目標=-720°")

                    if phase_rotation <= -720:  # 2回転完了
                        self.ascending_phase = 1  # 上昇中旋回フェーズに移行
                        self.ascending_phase_start_rotation = self.mission_total_rotation
                        print(f"上昇旋回: 2.5m水平旋回完了→上昇中旋回開始 (総回転角={self.mission_total_rotation:.1f}°)")

                elif self.ascending_phase == 1:  # フェーズ1: 旋回しながら5mまで上昇
                    # フェーズ1のデバッグ出力（5秒間隔）
                    if self.yaw_debug_counter % 100 == 0:  # 50ms * 100 = 5秒間隔
                        print(f"上昇旋回フェーズ1: 高度={current_alt_m:.2f}m, 総回転={self.mission_total_rotation:.1f}°, 目標高度=7.5m")

                    # 高度に応じて目標を調整（2.5m→7.5m）
                    altitude_high = float(self.current_autopilot_params.get('altitude_high', 7.5))
                    if current_alt_m < altitude_high:
                        target_altitude = altitude_high  # 上昇目標
                    else:
                        # 7.5m到達で次フェーズに移行
                        self.ascending_phase = 2
                        self.ascending_phase_start_rotation = self.mission_total_rotation
                        print(f"上昇旋回: 7.5m到達→7.5m水平旋回開始 (総回転角={self.mission_total_rotation:.1f}°)")
                        target_altitude = altitude_high

                elif self.ascending_phase == 2:  # フェーズ2: 7.5m高度で左旋回2回（-720度）
                    altitude_high = float(self.current_autopilot_params.get('altitude_high', 7.5))
                    target_altitude = altitude_high  # 高度維持
                    phase_rotation = self.mission_total_rotation - self.ascending_phase_start_rotation

                    # フェーズ2のデバッグ出力（5秒間隔）
                    if self.yaw_debug_counter % 100 == 0:  # 50ms * 100 = 5秒間隔
                        print(f"上昇旋回フェーズ2: 高度={current_alt_m:.2f}m, フェーズ回転={phase_rotation:.1f}°, 目標=-720°")

                    if phase_rotation <= -720:  # 2回転完了
                        self.ascending_completed = True
                        print(f"上昇旋回完了: 開始角={self.mission_start_yaw:.1f}°, 現在角={current_yaw:.1f}°, 総回転角={self.mission_total_rotation:.1f}°")
                        self.mission_status_label.setText("ミッション: 上昇旋回 成功")
            else:
                # ミッション完了後は8m高度維持
                target_altitude = 8.0
                target_roll = 0
        elif self.active_mission_mode == 3: # Figure-8 Turn (右旋回→左旋回)
            # 八の字旋回：右旋回から入り、目標角到達で左バンクに切り替え
            self.alt_pid.setpoint = self.mission_start_altitude
            # 八の字旋回：ミッション開始時の高度を維持
            target_altitude = self.mission_start_altitude

            right_target = self.current_autopilot_params.get('figure8_right_target', 300)  # 右旋回目標（正の値）
            left_target = abs(self.current_autopilot_params.get('figure8_left_target', -320))  # 左旋回目標（絶対値で使用）
            bank_angle = self.current_autopilot_params.get('bank_angle', 20)

            if not self.figure8_completed:
                if self.figure8_phase == 0:  # 右旋回フェーズ
                    target_roll = bank_angle  # 正のバンク角（右バンク）

                    # 八の字旋回 右フェーズのデバッグ出力（5秒間隔）
                    if self.yaw_debug_counter % 100 == 0:
                        print(f"八の字旋回右フェーズ: 総回転={self.mission_total_rotation:.1f}°, 目標={right_target}°")

                    # 右旋回なので、総回転角が目標角度以上になったら次フェーズ
                    if self.mission_total_rotation >= right_target:  # 右旋回目標角に到達
                        self.figure8_phase = 1  # 左旋回フェーズに切り替え
                        self.figure8_phase_start_rotation = self.mission_total_rotation  # 左旋回開始時点の総回転角を記録
                        print(f"八の字旋回: 右旋回完了→左旋回開始 (開始角={self.mission_start_yaw:.1f}°, 現在角={current_yaw:.1f}°, 総回転角={self.mission_total_rotation:.1f}°)")

                elif self.figure8_phase == 1:  # 左旋回フェーズ
                    target_roll = -bank_angle  # 負のバンク角（左バンク）
                    # 左旋回の進行度を計算：右旋回完了時点からの相対角度（負の値）
                    left_turn_progress = self.mission_total_rotation - self.figure8_phase_start_rotation

                    # 八の字旋回 左フェーズのデバッグ出力（5秒間隔）
                    if self.yaw_debug_counter % 100 == 0:
                        print(f"八の字旋回左フェーズ: 総回転={self.mission_total_rotation:.1f}°, 左進行={left_turn_progress:.1f}°, 目標=-{left_target}°")

                    # 左旋回でleft_target度以上回転したら完了（left_turn_progressが-left_target以下）
                    if left_turn_progress <= -left_target:  # 左旋回目標角に到達
                        self.figure8_completed = True
                        print(f"八の字旋回完了: 開始角={self.mission_start_yaw:.1f}°, 現在角={current_yaw:.1f}°, 総回転角={self.mission_total_rotation:.1f}°, 左旋回進行={left_turn_progress:.1f}°")
                        self.mission_status_label.setText("ミッション: 八の字旋回 成功")
            else:
                # ミッション完了後は水平飛行
                target_roll = 0
        elif self.active_mission_mode == 4:  # 再現飛行ミッション
            # 再現飛行中は入力再生処理で直接シリアル送信済みなので、ここでは何もしない
            if self.input_log_replaying:
                # 入力再生処理で既にシリアル送信済み、追加の制御は行わない
                return
            else:
                # 再現飛行でない場合は手動操縦として処理
                target_altitude = self.mission_start_altitude
        else:
            # 手動操縦時：ミッション開始時の高度を基準とする
            target_altitude = self.mission_start_altitude

        # 高度誤差に基づくスロットル制御
        current_alt_m = current_alt / 1000.0  # 高度をmm -> m単位に変換

        if self.active_mission_mode == 2:  # 上昇旋回の新しい4段階制御
            # target_altitudeは上記のフェーズ制御で設定済み
            if self.ascending_phase == 0:  # フェーズ0: 2.5m維持
                if abs(current_alt_m - 2.5) < 0.1:  # 2.5m付近では一定スロットル
                    target_throttle = base_throttle
                else:
                    altitude_error_m = 2.5 - current_alt_m
                    throttle_adjustment = altitude_error_m * altitude_gain
                    target_throttle = base_throttle + throttle_adjustment
                    target_throttle = max(throttle_min, min(target_throttle, throttle_max))
                    target_throttle = int(target_throttle)
            elif self.ascending_phase == 1:  # フェーズ1: 上昇中
                # 上昇時は専用のスロットル基準を使用
                ascending_throttle_base = float(self.current_autopilot_params.get('ascending_throttle_base', 800.0))
                altitude_high = float(self.current_autopilot_params.get('altitude_high', 7.5))
                altitude_error_m = altitude_high - current_alt_m
                throttle_adjustment = altitude_error_m * altitude_gain
                target_throttle = ascending_throttle_base + throttle_adjustment
                target_throttle = max(throttle_min, min(target_throttle, throttle_max))
                target_throttle = int(target_throttle)

                # 上昇時のピッチオフセットを適用（頭上げ）
                ascending_pitch_offset = self.current_autopilot_params.get('ascending_pitch_offset', 5.0)
                target_pitch += ascending_pitch_offset
            else:  # フェーズ2: 高度維持, またはミッション完了
                altitude_high = float(self.current_autopilot_params.get('altitude_high', 7.5))
                if abs(current_alt_m - altitude_high) < 0.1:  # 目標高度付近では一定スロットル
                    target_throttle = base_throttle
                else:
                    altitude_error_m = altitude_high - current_alt_m
                    throttle_adjustment = altitude_error_m * altitude_gain
                    target_throttle = base_throttle + throttle_adjustment
                    target_throttle = max(throttle_min, min(target_throttle, throttle_max))
                    target_throttle = int(target_throttle)
        elif self.active_mission_mode == 3:  # 八の字旋回
            # 八の字旋回は水平旋回と同じスロットル制御
            target_throttle = base_throttle
        else:
            # その他のミッション：ミッション開始時の高度との差分でスロットル制御
            altitude_error_m = self.mission_start_altitude / 1000.0 - current_alt_m
            throttle_adjustment = altitude_error_m * altitude_gain
            target_throttle = base_throttle + throttle_adjustment
            # スロットル値を制限
            target_throttle = max(throttle_min, min(target_throttle, throttle_max))
            target_throttle = int(target_throttle)

        target_pitch_from_alt = self.alt_pid.update(current_alt, dt)
        final_target_pitch = target_pitch_from_alt + target_pitch
        self.pitch_pid.setpoint = final_target_pitch
        elev_out = self.pitch_pid.update(current_pitch, dt)

        self.roll_pid.setpoint = target_roll
        ail_out = self.roll_pid.update(current_roll, dt)

        # エルロン→ラダーミキシング
        aileron_rudder_mix_coef = self.current_autopilot_params.get('aileron_rudder_mix', 0.3)
        rudd_out = ail_out * aileron_rudder_mix_coef

        # 自動旋回系ミッションでラダートリムを適用
        if self.active_mission_mode in [1, 2, 3]:  # 水平旋回、上昇旋回、八の字旋回
            if self.active_mission_mode == 1:  # 水平旋回（右旋回）
                rudder_trim = self.current_autopilot_params.get('rudder_trim_right', 0.1)
            elif self.active_mission_mode == 2:  # 上昇旋回（左旋回）
                rudder_trim = self.current_autopilot_params.get('rudder_trim_left', -0.1)
            elif self.active_mission_mode == 3:  # 八の字旋回
                if self.figure8_phase == 0:  # 右旋回フェーズ
                    rudder_trim = self.current_autopilot_params.get('rudder_trim_right', 0.1)
                else:  # 左旋回フェーズ
                    rudder_trim = self.current_autopilot_params.get('rudder_trim_left', -0.1)

            rudd_out += rudder_trim

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
        elif self.active_mission_mode == 2:  # 上昇旋回（新しい4段階制御）
            # 各フェーズの目標角度を表示
            if self.ascending_phase == 0:  # フェーズ0: -720度目標
                target_yaw_display = self.mission_start_yaw + self.ascending_phase_start_rotation - 720
            elif self.ascending_phase == 1:  # フェーズ1: 上昇中（現在の回転角）
                target_yaw_display = self.mission_start_yaw + self.mission_total_rotation
            else:  # フェーズ2: さらに-720度目標
                target_yaw_display = self.mission_start_yaw + self.ascending_phase_start_rotation - 720
        elif self.active_mission_mode == 3:  # 八の字旋回
            if self.figure8_phase == 0:  # 右旋回フェーズ
                figure8_target = self.current_autopilot_params.get('figure8_right_target', 300)
                target_yaw_display = self.mission_start_yaw + figure8_target
            else:  # 左旋回フェーズ
                figure8_target = self.current_autopilot_params.get('figure8_left_target', -320)
                target_yaw_display = self.mission_start_yaw + figure8_target
        else:
            # 手動操縦時は現在の総回転角
            target_yaw_display = self.mission_start_yaw + self.mission_total_rotation

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
            # AUX1チャンネルも含める（物資投下用）
            aux1_value = commands.get('aux1', 1500)  # デフォルトは1500（中立）
            command_str = f"{int(commands['ail'])},{int(commands['elev'])},{int(commands['rudd'])},{int(commands['thro'])},{self.active_mission_mode},{int(aux1_value)}\n"
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

    def update_marker_position_realtime(self, marker_id):
        """Update marker position in real-time as user types"""
        try:
            if marker_id not in self.marker_position_controls:
                return

            controls = self.marker_position_controls[marker_id]
            x = float(controls['x'].text())
            y = float(controls['y'].text())
            z = float(controls['z'].text())

            # Update position widgets
            if hasattr(self, 'xy_position_widget'):
                self.xy_position_widget.set_marker_position(marker_id, x, y, z)
            if hasattr(self, 'zy_position_widget'):
                self.zy_position_widget.set_marker_position(marker_id, x, y, z)

        except (ValueError, KeyError):
            # Ignore invalid input during typing
            pass

    def save_marker_positions(self):
        """Save marker positions from UI inputs"""
        try:
            success = False
            if hasattr(self, 'xy_position_widget'):
                success = self.xy_position_widget.save_marker_positions()
            if success:
                print("マーカー位置を保存しました")
            else:
                print("マーカー位置保存に失敗しました")
        except Exception as e:
            print(f"マーカー位置保存エラー: {e}")

    def load_marker_positions_ui(self):
        """Load marker positions and update UI"""
        try:
            if hasattr(self, 'xy_position_widget'):
                self.xy_position_widget.load_marker_positions()
                if hasattr(self, 'zy_position_widget'):
                    self.zy_position_widget.load_marker_positions()

                # Update UI controls with loaded positions
                for marker_id in [1, 2, 3]:
                    if marker_id in self.marker_position_controls and marker_id in self.xy_position_widget.aruco_markers:
                        marker_pos = self.xy_position_widget.aruco_markers[marker_id]
                        controls = self.marker_position_controls[marker_id]
                        controls['x'].setText(f"{marker_pos['x']:.1f}")
                        controls['y'].setText(f"{marker_pos['y']:.1f}")
                        controls['z'].setText(f"{marker_pos['z']:.1f}")

                print("マーカー位置を読み込みました")
        except Exception as e:
            print(f"マーカー位置読み込みエラー: {e}")

    def reset_marker_positions(self):
        """Reset marker positions to default values"""
        default_positions = {
            1: {'x': -3.0, 'y': 6.0, 'z': 0.0},
            2: {'x': 0.0, 'y': 0.0, 'z': 0.0},
            3: {'x': 3.0, 'y': 6.0, 'z': 0.0}
        }

        try:
            # Update position widgets
            for marker_id, pos in default_positions.items():
                if hasattr(self, 'xy_position_widget'):
                    self.xy_position_widget.set_marker_position(marker_id, pos['x'], pos['y'], pos['z'])
                if hasattr(self, 'zy_position_widget'):
                    self.zy_position_widget.set_marker_position(marker_id, pos['x'], pos['y'], pos['z'])

                # Update UI controls
                if marker_id in self.marker_position_controls:
                    controls = self.marker_position_controls[marker_id]
                    controls['x'].setText(f"{pos['x']:.1f}")
                    controls['y'].setText(f"{pos['y']:.1f}")
                    controls['z'].setText(f"{pos['z']:.1f}")

            print("マーカー位置をデフォルトにリセットしました")
        except Exception as e:
            print(f"マーカー位置リセットエラー: {e}")

    # --- Auto Landing Log System Methods ---

    def toggle_auto_landing_log_recording(self):
        """Toggle auto landing log recording"""
        if self.auto_landing_log_recording:
            self.stop_auto_landing_log_recording()
        else:
            self.start_auto_landing_log_recording()

    def start_auto_landing_log_recording(self):
        """Start recording flight state data for auto landing log"""
        if not self.auto_landing_enabled:
            print("自動離着陸が有効化されていません")
            return

        self.auto_landing_log_recording = True
        self.auto_landing_log_data = []
        self.auto_landing_log_start_time = time.time()
        self.auto_landing_log_last_record_time = 0

        # Update UI
        self.auto_landing_log_record_button.setText("記録中...")
        self.auto_landing_log_record_button.setStyleSheet("background-color: #dc3545; color: white;")
        self.auto_landing_log_stop_record_button.setEnabled(True)
        self.auto_landing_log_status_label.setText("状態: 飛行状態記録中")

        print("飛行状態記録開始")

    def stop_auto_landing_log_recording(self):
        """Stop recording and save flight state data"""
        if not self.auto_landing_log_recording:
            return

        self.auto_landing_log_recording = False

        # Save log data to file with dialog
        if self.auto_landing_log_data:
            self.save_auto_landing_log_with_dialog()

        # Update UI
        self.auto_landing_log_record_button.setText("飛行状態記録開始")
        self.auto_landing_log_record_button.setStyleSheet("background-color: #28a745; color: white;")
        self.auto_landing_log_stop_record_button.setEnabled(False)
        self.auto_landing_log_status_label.setText("状態: 待機中")

        # Update flight state graph with recorded data
        if self.auto_landing_log_data and hasattr(self, 'flight_state_graph'):
            self.flight_state_graph.update_plots(self.auto_landing_log_data)
            print("記録データでグラフを更新しました")

        print("飛行状態記録停止")

    def save_auto_landing_log_with_dialog(self):
        """Save auto landing log data with file dialog"""
        try:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"auto_landing_log_{timestamp}.json"

            # Open save dialog
            os.makedirs("logs", exist_ok=True)
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "自動離着陸ログを保存",
                os.path.join("logs", default_filename),
                "JSON files (*.json);;All files (*.*)"
            )

            if not file_path:
                print("ファイル保存がキャンセルされました")
                return

            # Save the data
            with open(file_path, 'w') as f:
                json.dump(self.auto_landing_log_data, f, indent=2)

            print(f"飛行状態ログを保存しました: {file_path}")
            print(f"記録データ点数: {len(self.auto_landing_log_data)}")

        except Exception as e:
            print(f"飛行状態ログの保存中にエラーが発生しました: {e}")

    def record_auto_landing_log_data(self):
        """Record current flight state data"""
        if not self.auto_landing_log_recording:
            return

        current_time = time.time()

        # Check if enough time has passed since last recording
        if current_time - self.auto_landing_log_last_record_time < self.auto_landing_log_interval:
            return

        self.auto_landing_log_last_record_time = current_time
        elapsed_time = current_time - self.auto_landing_log_start_time

        # Get current flight state data
        altitude = getattr(self, 'current_altitude', 0.0)  # Get current altitude from telemetry
        yaw_angle = getattr(self, 'current_yaw', 0.0)      # Get current yaw angle from telemetry
        throttle = getattr(self, 'current_throttle', 0.0)  # Get current throttle from propeller inputs
        aux1 = getattr(self, 'current_aux1', 0.0)          # Get current AUX1 (material drop timing)

        # ヨー角の変位（ミッション開始時からの相対角度）を計算
        yaw_displacement = yaw_angle - getattr(self, 'mission_start_yaw', 0.0)
        # 角度を-180~+180度の範囲に正規化
        while yaw_displacement > 180:
            yaw_displacement -= 360
        while yaw_displacement < -180:
            yaw_displacement += 360

        # Create log entry
        log_entry = {
            'timestamp': current_time,
            'elapsed_time': elapsed_time,
            'altitude': altitude,
            'yaw': yaw_displacement,  # 変位角を記録
            'throttle': throttle,
            'aux1': aux1
        }

        self.auto_landing_log_data.append(log_entry)

        # Optional: Print periodic updates
        if len(self.auto_landing_log_data) % 10 == 0:
            print(f"飛行状態記録中: {len(self.auto_landing_log_data)} データ点")

    def load_auto_landing_log(self):
        """Load auto landing log file with dialog"""
        try:
            # Open file dialog to select log file
            os.makedirs("logs", exist_ok=True)
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "自動離着陸ログファイルを選択",
                "logs",
                "JSON files (*.json);;All files (*.*)"
            )

            if not file_path:
                print("ファイル選択がキャンセルされました")
                return

            with open(file_path, 'r') as f:
                self.loaded_auto_landing_log = json.load(f)

            print(f"自動離着陸ログ読み込み完了: {file_path}")
            print(f"データ点数: {len(self.loaded_auto_landing_log)}")

            # Enable replay button
            self.auto_landing_log_replay_button.setEnabled(True)
            self.auto_landing_log_status_label.setText(f"状態: ログ読み込み完了 ({len(self.loaded_auto_landing_log)} データ点)")

            # Update flight state graph with loaded data
            if hasattr(self, 'flight_state_graph'):
                self.flight_state_graph.update_plots(self.loaded_auto_landing_log)
                print("飛行状態グラフを更新しました")

        except Exception as e:
            print(f"自動離着陸ログの読み込み中にエラーが発生しました: {e}")

    def start_auto_landing_log_replay(self):
        """Start auto landing log replay"""
        if not self.loaded_auto_landing_log:
            print("自動離着陸ログが読み込まれていません")
            return

        if not self.auto_landing_enabled:
            print("自動離着陸が有効化されていません")
            return

        self.auto_landing_log_replaying = True
        self.auto_landing_log_replay_start_time = time.time()
        self.auto_landing_log_replay_index = 0

        # 自動着陸ログ再現開始時の基準ヨー角を設定（記録されたデータは変位角のため）
        self.auto_landing_replay_start_yaw = self.latest_attitude.get('yaw', 0.0)
        print(f"自動着陸ログ再現開始: 基準ヨー角={self.auto_landing_replay_start_yaw:.1f}°")

        # Update UI
        self.auto_landing_log_replay_button.setEnabled(False)
        self.auto_landing_log_stop_replay_button.setEnabled(True)
        self.auto_landing_log_status_label.setText("状態: ログ再現飛行中")

        print(f"自動離着陸ログ再現飛行開始: {len(self.loaded_auto_landing_log)} データ点")

    def stop_auto_landing_log_replay(self):
        """Stop auto landing log replay"""
        if not self.auto_landing_log_replaying:
            return

        self.auto_landing_log_replaying = False

        # ログ再現飛行停止が押されたら即座にミッションモード4から抜け出す
        if self.autopilot_active and self.active_mission_mode == 4:
            self.stop_mission()

        # Update UI
        self.auto_landing_log_replay_button.setEnabled(True)
        self.auto_landing_log_stop_replay_button.setEnabled(False)
        self.auto_landing_log_status_label.setText("状態: 再現停止")

        print("自動離着陸ログ再現飛行停止")

    def process_auto_landing_log_replay(self):
        """Process auto landing log replay - sends flight state commands"""
        if not self.auto_landing_log_replaying or not self.loaded_auto_landing_log:
            return

        current_time = time.time()
        elapsed_time = current_time - self.auto_landing_log_replay_start_time

        # Find the appropriate data point based on elapsed time
        target_index = None
        for i, data_point in enumerate(self.loaded_auto_landing_log):
            if data_point['elapsed_time'] <= elapsed_time:
                target_index = i
            else:
                break

        if target_index is not None and target_index != self.auto_landing_log_replay_index:
            self.auto_landing_log_replay_index = target_index
            data_point = self.loaded_auto_landing_log[target_index]

            # Use flight state data to generate control commands
            # This would typically involve converting altitude, yaw, throttle, aux1
            # into appropriate control surface commands
            altitude_target = data_point['altitude']
            yaw_displacement = data_point['yaw']  # 記録された変位角
            throttle_target = data_point['throttle']
            aux1_target = data_point['aux1']

            # 変位角を再現開始時の基準角度に対する絶対角度に変換
            yaw_target = getattr(self, 'auto_landing_replay_start_yaw', 0.0) + yaw_displacement

            # 飛行状態から制御コマンドを生成（簡易版）
            # 実際の実装では、目標値に対してPID制御等を適用する必要があります

            # 基本的な制御コマンド（とりあえずの実装）
            commands = {
                'ail': 1500,  # 中立値（後でPID制御で計算）
                'elev': 1500, # 中立値（後で高度制御で計算）
                'rudd': 1500, # 中立値（後でヨー制御で計算）
                'thro': int(throttle_target),  # 記録されたスロットル値をそのまま使用
                'aux1': int(aux1_target)  # AUX1（物資投下）も再現
            }

            # 正しいシリアル送信関数を使用（ミッションモード4で送信）
            # ミッションモードを一時的に4に設定
            original_mission_mode = self.active_mission_mode
            self.active_mission_mode = 4

            self.send_serial_command(commands)

            # ミッションモードを元に戻す
            self.active_mission_mode = original_mission_mode

            print(f"Auto landing replay: Alt={altitude_target}, Yaw={yaw_target:.1f}°, Thro={throttle_target}, AUX1={aux1_target}")

    def update_auto_landing_log_interval(self, value):
        """Update auto landing log recording interval"""
        self.auto_landing_log_interval = value
        print(f"自動離着陸ログ記録間隔を {value} 秒に設定しました")

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
