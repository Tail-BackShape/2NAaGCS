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

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QPushButton, QLabel, QGridLayout, QLineEdit, QCheckBox,
    QTabWidget
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

# --- Position Visualization Widgets for Auto Landing ---
class PositionVisualizationWidget(QWidget):
    def __init__(self, view_type="XY", parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.view_type = view_type  # "XY" or "ZY"
        self.aircraft_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # Default marker positions (can be modified)
        self.aruco_markers = {
            1: {'x': -3.0, 'y': 6.0, 'z': 0.0, 'detected': False},
            2: {'x': 0.0, 'y': 0.0, 'z': 0.0, 'detected': False},
            3: {'x': 3.0, 'y': 6.0, 'z': 0.0, 'detected': False}
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
        """Update marker data with real-time values"""
        if marker_id in self.aruco_markers:
            self.aruco_markers[marker_id]['size'] = size
            self.aruco_markers[marker_id]['x'] = x
            self.aruco_markers[marker_id]['y'] = y
            self.aruco_markers[marker_id]['detected'] = size > 0
        self.update()

    def set_marker_position(self, marker_id, x, y, z=0.0):
        """Set marker position (for field setup)"""
        if marker_id in self.aruco_markers:
            self.aruco_markers[marker_id]['x'] = x
            self.aruco_markers[marker_id]['y'] = y
            self.aruco_markers[marker_id]['z'] = z
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
            runway_end_y = center_y - 33 * self.scale
            painter.drawLine(center_x - 10, runway_start_y, center_x + 10, runway_start_y)
            painter.drawLine(center_x - 10, runway_end_y, center_x + 10, runway_end_y)
            painter.drawLine(center_x, runway_start_y, center_x, runway_end_y)

        # Draw ArUco markers (both configured positions and real-time detections)
        for marker_id, marker in self.aruco_markers.items():
            if self.view_type == "XY":
                # Use configured field positions for markers
                marker_x = center_x + float(marker['x']) * self.scale
                marker_y = center_y - float(marker['y']) * self.scale
            else:  # ZY view
                marker_x = center_x + float(marker['y']) * self.scale
                marker_y = center_y - float(marker.get('z', 0)) * self.scale

            # Color based on detection status
            is_detected = marker.get('detected', False) or marker.get('size', 0) > 0
            color = QColor("lime") if is_detected else QColor("orange")

            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))

            # Draw marker as square (more representative of ArUco markers)
            marker_size = 12
            marker_rect = QRectF(marker_x - marker_size/2, marker_y - marker_size/2, marker_size, marker_size)
            painter.drawRect(marker_rect)

            # Draw marker ID and position info
            painter.setPen(QPen(QColor("white"), 1))
            if self.view_type == "XY":
                marker_info = f"ID{marker_id}\n({marker['x']:.1f}, {marker['y']:.1f})"
                if is_detected:
                    marker_info += f"\nSize: {marker.get('size', 0):.0f}"
            else:
                marker_info = f"ID{marker_id}\n({marker['y']:.1f}, {marker.get('z', 0):.1f})"
                if is_detected:
                    marker_info += f"\nSize: {marker.get('size', 0):.0f}"

            painter.drawText(marker_x + marker_size/2 + 5, marker_y - marker_size/2, marker_info)

        # Draw aircraft position
        if self.view_type == "XY":
            aircraft_x = center_x + self.aircraft_pos['x'] * self.scale
            aircraft_y = center_y - self.aircraft_pos['y'] * self.scale
        else:  # ZY view
            aircraft_x = center_x + self.aircraft_pos['y'] * self.scale
            aircraft_y = center_y - self.aircraft_pos['z'] * self.scale

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
            painter.drawText(center_x + 10, 15, "Y")
        else:
            painter.drawText(self.width() - 30, center_y - 10, "Y")
            painter.drawText(center_x + 10, 15, "Z")

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#2b2b2b"))

        # Draw graph
        self._draw_graph(painter)

    def _draw_graph(self, painter):
        """Draw calibration graph"""
        if not self.calibration_data:
            # No data message
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawText(self.rect().center() + QPointF(-100, 0), "キャリブレーションデータがありません")
            return

        # Graph area
        margin = 50
        graph_rect = QRectF(margin, margin, self.width() - 2 * margin, self.height() - 2 * margin)

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

                    # Draw point
                    painter.drawEllipse(QPointF(x, y), 4, 4)

            # Draw connecting lines
            if len(points) > 1:
                painter.setPen(QPen(color, 1, Qt.DashLine))
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
            1: {'size': 0, 'id': 0, 'x': 0, 'y': 0},
            2: {'size': 0, 'id': 0, 'x': 0, 'y': 0},
            3: {'size': 0, 'id': 0, 'x': 0, 'y': 0}
        }

        # --- ArUco Marker Calibration Data ---
        self.marker_calibrations = {
            1: {'offset_x': 0.0, 'offset_y': 0.0, 'offset_angle': 0.0},
            2: {'offset_x': 0.0, 'offset_y': 0.0, 'offset_angle': 0.0},
            3: {'offset_x': 0.0, 'offset_y': 0.0, 'offset_angle': 0.0}
        }

        # --- Figure-8 Mission State ---
        self.figure8_phase = 0  # 0: 右旋回フェーズ, 1: 左旋回フェーズ
        self.figure8_completed = False
        self.figure8_phase_start_rotation = 0  # 左旋回フェーズ開始時の総回転角

        # --- Ascending Turn Mission State ---
        self.ascending_phase = 0  # 0: 2.5m左旋回2回, 1: 上昇中旋回, 2: 5m左旋回2回
        self.ascending_completed = False
        self.ascending_phase_start_rotation = 0  # フェーズ開始時の総回転角

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
        self.load_auto_landing_ui_params() # Load parameters into UI
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
        # Create the main tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create GCS tab (existing functionality)
        self._create_gcs_tab()

        # Create Auto Landing tab (new functionality)
        self._create_auto_landing_tab()

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
        main_layout.addWidget(center_panel, 2)

        # Right panel: Parameter panels
        right_panel = self._create_auto_landing_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Add Auto Landing tab to tab widget
        self.tab_widget.addTab(auto_landing_widget, "自動離着陸")

        # Create calibration graph tab
        self._create_calibration_graph_tab()

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

            # Position offsets
            offset_x_input = QLineEdit("0.0")
            offset_y_input = QLineEdit("0.0")
            offset_angle_input = QLineEdit("0.0")

            # Real-time data display
            realtime_label = QLabel(f"リアルタイム: サイズ=0, X=0, Y=0")
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
            marker_form.addRow("X オフセット (m):", offset_x_input)
            marker_form.addRow("Y オフセット (m):", offset_y_input)
            marker_form.addRow("角度 オフセット (度):", offset_angle_input)
            marker_form.addRow("リアルタイムデータ:", realtime_label)
            marker_form.addRow(set_current_button)
            marker_form.addRow("基準距離 (m):", distance_input)
            marker_form.addRow(record_distance_button)

            marker_layout.addLayout(marker_form)

            # Store references
            self.marker_calib_controls[marker_id] = {
                'offset_x': offset_x_input,
                'offset_y': offset_y_input,
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

        # Save calibration data button
        save_calib_button = QPushButton("キャリブレーションデータ保存")
        save_calib_button.clicked.connect(self.export_calibration_data)

        # Load calibration data button
        load_calib_button = QPushButton("キャリブレーションデータ読み込み")
        load_calib_button.clicked.connect(self.import_calibration_data)

        control_layout.addWidget(marker_select_label)
        control_layout.addWidget(self.graph_marker_combo)
        control_layout.addWidget(refresh_button)
        control_layout.addWidget(save_calib_button)
        control_layout.addWidget(load_calib_button)
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

        # Auto landing timer
        self.auto_landing_timer = QTimer(self)
        self.auto_landing_timer.timeout.connect(self.auto_landing_update_cycle)
        self.auto_landing_timer.start(100) # 10 Hz

    def auto_landing_update_cycle(self):
        """Auto landing update cycle - runs at 10Hz"""
        if self.is_connected:
            # Calculate position and distance
            self.calculate_aircraft_position()

            # Update phase based on distance
            self.update_auto_landing_phase()

            # Run control logic
            self.run_auto_landing_control()

            # Update auto landing attitude displays
            if hasattr(self, 'auto_pitch_widget') and hasattr(self, 'auto_yaw_widget'):
                roll = self.latest_attitude.get('roll', 0)
                pitch = self.latest_attitude.get('pitch', 0)
                yaw = self.latest_attitude.get('yaw', 0)

                self.auto_pitch_widget.set_angle(pitch)
                self.auto_yaw_widget.set_angle(yaw)
                self.auto_pitch_widget.set_autopilot_active(self.auto_landing_enabled)
                self.auto_yaw_widget.set_autopilot_active(self.auto_landing_enabled)

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

        # Set current position as reference (zero offset)
        controls = self.marker_calib_controls[marker_id]
        controls['offset_x'].setText(str(float(marker_data['x'])))
        controls['offset_y'].setText(str(float(marker_data['y'])))
        controls['offset_angle'].setText("0.0")

        # Update calibration data
        self.marker_calibrations[marker_id]['offset_x'] = float(marker_data['x'])
        self.marker_calibrations[marker_id]['offset_y'] = float(marker_data['y'])
        self.marker_calibrations[marker_id]['offset_angle'] = 0.0

        print(f"Set marker {marker_id} calibration: X={marker_data['x']}, Y={marker_data['y']}")
        self.save_marker_calibrations()

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
                estimated_distance = self.estimate_distance_from_marker(marker_id)
                distance_text = f", 推定距離={estimated_distance:.2f}m" if estimated_distance is not None else ""

                realtime_text = f"リアルタイム: サイズ={marker_data['size']}, X={marker_data['x']}, Y={marker_data['y']}{distance_text}"

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
        """Save marker calibrations to file"""
        try:
            calib_file = 'marker_calibrations.txt'
            with open(calib_file, 'w') as f:
                for marker_id, calib in self.marker_calibrations.items():
                    f.write(f"marker_{marker_id}_offset_x={calib['offset_x']}\n")
                    f.write(f"marker_{marker_id}_offset_y={calib['offset_y']}\n")
                    f.write(f"marker_{marker_id}_offset_angle={calib['offset_angle']}\n")
            print("Marker calibrations saved")
        except Exception as e:
            print(f"Failed to save marker calibrations: {e}")

    def load_marker_calibrations(self):
        """Load marker calibrations from file"""
        try:
            calib_file = 'marker_calibrations.txt'
            if not os.path.exists(calib_file):
                return

            with open(calib_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)

                        # Parse marker calibration settings
                        if key.startswith('marker_') and '_' in key:
                            parts = key.split('_')
                            if len(parts) >= 3:
                                marker_id = int(parts[1])
                                param = '_'.join(parts[2:])

                                if marker_id in self.marker_calibrations:
                                    if param in ['offset_x', 'offset_y', 'offset_angle']:
                                        self.marker_calibrations[marker_id][param] = float(value)
                                        # Update UI input field
                                        if marker_id in self.marker_calib_controls:
                                            self.marker_calib_controls[marker_id][param].setText(str(value))

            print("Marker calibrations loaded")
        except Exception as e:
            print(f"Failed to load marker calibrations: {e}")

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

            # Add new calibration point
            calibration_point = {
                'distance': distance,
                'size': marker_size,
                'x': self.aruco_markers[marker_id]['x'],
                'y': self.aruco_markers[marker_id]['y']
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
        """Save marker distance calibrations to file"""
        try:
            calib_file = 'marker_distance_calibrations.txt'
            if not hasattr(self, 'marker_distance_calibrations'):
                return

            with open(calib_file, 'w') as f:
                for marker_id, calibrations in self.marker_distance_calibrations.items():
                    for i, calib in enumerate(calibrations):
                        f.write(f"marker_{marker_id}_point_{i}_distance={calib['distance']}\n")
                        f.write(f"marker_{marker_id}_point_{i}_size={calib['size']}\n")
                        f.write(f"marker_{marker_id}_point_{i}_x={calib['x']}\n")
                        f.write(f"marker_{marker_id}_point_{i}_y={calib['y']}\n")

            print("Marker distance calibrations saved")
        except Exception as e:
            print(f"Failed to save marker distance calibrations: {e}")

    def load_marker_distance_calibrations(self):
        """Load marker distance calibrations from file"""
        try:
            calib_file = 'marker_distance_calibrations.txt'
            if not os.path.exists(calib_file):
                return

            self.marker_distance_calibrations = {}

            with open(calib_file, 'r') as f:
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

            print("Marker distance calibrations loaded")
        except Exception as e:
            print(f"Failed to load marker distance calibrations: {e}")

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

        # Use linear interpolation or the closest calibration point
        if len(calibrations) == 1:
            # Single point: use simple inverse relationship
            calib = calibrations[0]
            estimated_distance = calib['distance'] * calib['size'] / current_size
            return estimated_distance
        else:
            # Multiple points: find the two closest by size
            calibrations = sorted(calibrations, key=lambda x: x['size'])

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
        """Update calibration graph display"""
        if hasattr(self, 'calibration_graph_widget'):
            # Get selected marker
            selected_text = self.graph_marker_combo.currentText()
            self.calibration_graph_widget.set_selected_marker(selected_text)

            # Pass calibration data
            if hasattr(self, 'marker_distance_calibrations'):
                self.calibration_graph_widget.set_calibration_data(self.marker_distance_calibrations)

            # Update text display
            self.update_calibration_data_display()

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

    def export_calibration_data(self):
        """Export calibration data to JSON file"""
        try:
            import json
            from datetime import datetime

            if not hasattr(self, 'marker_distance_calibrations'):
                print("No calibration data to export")
                return

            # Prepare data for export
            export_data = {
                'export_time': datetime.now().isoformat(),
                'markers': self.marker_distance_calibrations
            }

            # Save to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'calibration_export_{timestamp}.json'

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"Calibration data exported to {filename}")

        except Exception as e:
            print(f"Failed to export calibration data: {e}")

    def import_calibration_data(self):
        """Import calibration data from JSON file"""
        try:
            import json
            import glob

            # Find the most recent export file
            export_files = glob.glob('calibration_export_*.json')
            if not export_files:
                print("No calibration export files found")
                return

            # Use the most recent file
            latest_file = max(export_files)

            with open(latest_file, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Load the data
            if 'markers' in import_data:
                self.marker_distance_calibrations = import_data['markers']
                self.save_marker_distance_calibrations()  # Save to regular format too
                self.update_calibration_graph()
                print(f"Calibration data imported from {latest_file}")
            else:
                print("Invalid calibration data format")

        except Exception as e:
            print(f"Failed to import calibration data: {e}")

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

        # Load marker calibrations
        self.load_marker_calibrations()

        # Load marker distance calibrations
        self.load_marker_distance_calibrations()

    def _init_pid_controllers(self):
        gains = self.current_pid_gains
        self.roll_pid = PIDController(gains.get('roll_p', 0), gains.get('roll_i', 0), gains.get('roll_d', 0))
        self.pitch_pid = PIDController(gains.get('pitch_p', 0), gains.get('pitch_i', 0), gains.get('pitch_d', 0))
        # TODO: Make alt PID gains adjustable
        self.alt_pid = PIDController(Kp=0.1, Ki=0.02, Kd=0.05, output_limits=(-15, 15)) # Output is target pitch angle
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
        """Estimate distance using ArUco marker data with 3-point linear interpolation"""
        valid_markers = []

        # Check which markers have valid data (size > 0)
        for marker_id, marker_data in self.aruco_markers.items():
            if marker_data['size'] > 0:
                valid_markers.append((marker_id, marker_data['size']))

        if not valid_markers:
            return 0.0

        # Calculate weighted distance estimates
        distance_estimates = []
        total_weight = 0.0

        for marker_id, size in valid_markers:
            distance_est = self.interpolate_distance_from_size(size)
            if distance_est > 0:
                weight = size  # Larger markers get more weight
                distance_estimates.append((distance_est, weight))
                total_weight += weight

        if not distance_estimates:
            return 0.0

        # Calculate weighted average
        weighted_distance = sum(dist * weight for dist, weight in distance_estimates) / total_weight
        return weighted_distance

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

            # 高度データの取得とデバッグ
            raw_alt = self.latest_attitude.get('alt', 0)
            self.current_position['z'] = raw_alt / 1000.0  # Convert mm to m

        else:
            # Use other markers for positioning (simplified)
            self.current_position['y'] = distance
            raw_alt = self.latest_attitude.get('alt', 0)
            self.current_position['z'] = raw_alt / 1000.0

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
        if not self.auto_landing_enabled:
            self.auto_landing_phase = 0  # Manual
            return

        distance = self.estimated_distance

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

        # Update phase display
        phase_names = {0: "手動", 1: "離陸", 2: "投下", 3: "定常", 4: "着陸"}
        if hasattr(self, 'phase_label'):
            self.phase_label.setText(f"フェーズ: {phase_names.get(self.auto_landing_phase, '不明')}")
        if hasattr(self, 'distance_label'):
            self.distance_label.setText(f"推定距離: {distance:.1f} m")

        # Update altitude debug info
        if hasattr(self, 'altitude_debug_label'):
            current_alt_mm = self.latest_attitude.get('alt', 0)
            current_alt_m = current_alt_mm / 1000.0
            self.altitude_debug_label.setText(f"高度: {current_alt_m:.2f} m ({current_alt_mm:.0f} mm)")

    def run_auto_landing_control(self):
        """Execute auto landing control logic based on current phase"""
        if not self.auto_landing_enabled or not self.is_connected:
            return

        current_alt = self.latest_attitude.get('alt', 0) / 1000.0  # Convert mm to m
        current_roll = self.latest_attitude.get('roll', 0)
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
        commands = [aileron_rc, elevator_rc, rudder_rc, throttle_rc]
        self.send_serial_command(commands)

        # Update auto landing stick displays
        if hasattr(self, 'auto_left_stick') and hasattr(self, 'auto_right_stick'):
            self.auto_left_stick.set_autopilot_position(target_rudder, elevator_cmd)
            self.auto_left_stick.set_autopilot_active(True)

            throttle_norm = (target_throttle - 400) / 600.0 - 1.0  # Normalize throttle
            self.auto_right_stick.set_autopilot_position(aileron_cmd, throttle_norm)
            self.auto_right_stick.set_autopilot_active(True)

        # Update stick labels
        if hasattr(self, 'auto_left_stick_label'):
            self.auto_left_stick_label.setText(f"R: {rudder_rc}, E: {elevator_rc}")
        if hasattr(self, 'auto_right_stick_label'):
            self.auto_right_stick_label.setText(f"A: {aileron_rc}, T: {throttle_rc}")

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
            if len(parts) == 24:  # 24パラメータに対応
                # パラメータ1-12: 既存のテレメトリデータ
                roll, pitch, yaw, alt, ail, elev, thro, rudd, aux1, aux2, aux3, aux4 = parts[0:12]

                # パラメータ13-24: ArUcoマーカー情報
                aruco1_size, aruco1_id, aruco1_x, aruco1_y = parts[12:16]
                aruco2_size, aruco2_id, aruco2_x, aruco2_y = parts[16:20]
                aruco3_size, aruco3_id, aruco3_x, aruco3_y = parts[20:24]

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

                # 姿勢データのフィルタリング（0.0の場合または前回と10度以上違う場合は前回値を使用）
                roll = roll if (roll != 0.0 and abs(roll - prev_roll) < 10.0) else prev_roll
                pitch = pitch if (pitch != 0.0 and abs(pitch - prev_pitch) < 10.0) else prev_pitch
                yaw = yaw if (yaw != 0.0 and abs(yaw - prev_yaw) < 10.0) else prev_yaw

                # 高度データの処理（0.0の場合のみ前回値を使用、それ以外は送信データをそのまま使用）
                if alt != 0.0:
                    filtered_alt = alt
                    if abs(alt - prev_alt) > 1000.0 and prev_alt != 0.0:  # 1m以上変化した場合にログ出力
                        print(f"高度変化: {prev_alt:.0f}mm -> {alt:.0f}mm (差: {alt-prev_alt:.0f}mm)")
                else:
                    filtered_alt = prev_alt
                    print(f"高度データが0.0のため前回値({prev_alt:.0f}mm)を使用")

                self.latest_attitude = {'roll': roll, 'pitch': pitch, 'yaw': yaw, 'alt': filtered_alt}

                self.adi_widget.set_attitude(roll, pitch)
                self.altimeter_widget.set_altitude(filtered_alt)
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
            else:
                print(f"データ形式エラー: 24パラメータが必要ですが、{len(parts)}パラメータを受信しました - {line}")
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
                    # 新ミッション開始時に回転角をリセット
                    self.mission_total_rotation = 0.0
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
                        print(f"上昇旋回フェーズ1: 高度={current_alt_m:.2f}m, 総回転={self.mission_total_rotation:.1f}°, 目標高度=5.0m")

                    # 高度に応じて目標を調整（2.5m→5m）
                    if current_alt_m < 5.0:
                        target_altitude = 5.0  # 上昇目標
                    else:
                        # 5m到達で次フェーズに移行
                        self.ascending_phase = 2
                        self.ascending_phase_start_rotation = self.mission_total_rotation
                        print(f"上昇旋回: 5m到達→5m水平旋回開始 (総回転角={self.mission_total_rotation:.1f}°)")
                        target_altitude = 5.0

                elif self.ascending_phase == 2:  # フェーズ2: 5m高度で左旋回2回（-720度）
                    target_altitude = 5.0  # 5m維持
                    phase_rotation = self.mission_total_rotation - self.ascending_phase_start_rotation

                    # フェーズ2のデバッグ出力（5秒間隔）
                    if self.yaw_debug_counter % 100 == 0:  # 50ms * 100 = 5秒間隔
                        print(f"上昇旋回フェーズ2: 高度={current_alt_m:.2f}m, フェーズ回転={phase_rotation:.1f}°, 目標=-720°")

                    if phase_rotation <= -720:  # 2回転完了
                        self.ascending_completed = True
                        print(f"上昇旋回完了: 開始角={self.mission_start_yaw:.1f}°, 現在角={current_yaw:.1f}°, 総回転角={self.mission_total_rotation:.1f}°")
                        self.mission_status_label.setText("ミッション: 上昇旋回 成功")
            else:
                # ミッション完了後は5m高度維持
                target_altitude = 5.0
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
                altitude_error_m = 5.0 - current_alt_m
                throttle_adjustment = altitude_error_m * altitude_gain
                target_throttle = base_throttle + throttle_adjustment
                target_throttle = max(throttle_min, min(target_throttle, throttle_max))
                target_throttle = int(target_throttle)
            else:  # フェーズ2: 5m維持, またはミッション完了
                if abs(current_alt_m - 5.0) < 0.1:  # 5m付近では一定スロットル
                    target_throttle = base_throttle
                else:
                    altitude_error_m = 5.0 - current_alt_m
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
                self.status_label.setText("マーカー位置を保存しました")
                print("マーカー位置保存完了")
            else:
                self.status_label.setText("マーカー位置保存に失敗しました")
        except Exception as e:
            print(f"マーカー位置保存エラー: {e}")
            self.status_label.setText(f"保存エラー: {e}")

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

                self.status_label.setText("マーカー位置を読み込みました")
                print("マーカー位置読み込み完了")
        except Exception as e:
            print(f"マーカー位置読み込みエラー: {e}")
            self.status_label.setText(f"読み込みエラー: {e}")

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

            self.status_label.setText("マーカー位置をデフォルトにリセットしました")
            print("マーカー位置リセット完了")
        except Exception as e:
            print(f"マーカー位置リセットエラー: {e}")
            self.status_label.setText(f"リセットエラー: {e}")

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
