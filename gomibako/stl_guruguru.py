#!/usr/bin/env python3
"""
STL回転プログラム (stl_guruguru.py)
STLファイルを指定した軸周りに回転させるツール
"""

import argparse
import numpy as np
import pyvista as pv
from pathlib import Path
import sys
import os
import time
import threading

# GUI用のインポート（オプショナル）
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class STLRotator:
    """STLファイル回転クラス"""

    def __init__(self, input_file):
        """
        初期化
        Args:
            input_file (str): 入力STLファイルのパス
        """
        self.input_file = Path(input_file)
        self.mesh = None
        self.original_mesh = None

        if not self.input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

        # STLファイルを読み込み
        self.load_mesh()

    def load_mesh(self):
        """STLファイルを読み込み"""
        try:
            print(f"STLファイルを読み込み中: {self.input_file}")
            self.mesh = pv.read(str(self.input_file))
            self.original_mesh = self.mesh.copy()
            print(f"メッシュ情報:")
            print(f"  - 頂点数: {self.mesh.n_points:,}")
            print(f"  - 面数: {self.mesh.n_cells:,}")
            print(f"  - バウンディングボックス: {self.mesh.bounds}")
        except Exception as e:
            raise RuntimeError(f"STLファイルの読み込みに失敗: {e}")

    def get_mesh_info(self):
        """現在のメッシュ情報を取得"""
        if self.mesh is None:
            return None

        return {
            'points': self.mesh.n_points,
            'faces': self.mesh.n_cells,
            'bounds': self.mesh.bounds,
            'center': self.mesh.center,
            'volume': self.mesh.volume if hasattr(self.mesh, 'volume') else 'N/A'
        }

    def rotate_x(self, angle_degrees):
        """
        X軸周りに回転
        Args:
            angle_degrees (float): 回転角度（度）
        """
        print(f"X軸周りに{angle_degrees}度回転中...")
        self.mesh = self.mesh.rotate_x(angle_degrees, inplace=False)
        print("X軸回転完了")

    def rotate_y(self, angle_degrees):
        """
        Y軸周りに回転
        Args:
            angle_degrees (float): 回転角度（度）
        """
        print(f"Y軸周りに{angle_degrees}度回転中...")
        self.mesh = self.mesh.rotate_y(angle_degrees, inplace=False)
        print("Y軸回転完了")

    def rotate_z(self, angle_degrees):
        """
        Z軸周りに回転
        Args:
            angle_degrees (float): 回転角度（度）
        """
        print(f"Z軸周りに{angle_degrees}度回転中...")
        self.mesh = self.mesh.rotate_z(angle_degrees, inplace=False)
        print("Z軸回転完了")

    def rotate_custom_axis(self, axis, angle_degrees, point=None):
        """
        指定した軸周りに回転
        Args:
            axis (list): 回転軸ベクトル [x, y, z]
            angle_degrees (float): 回転角度（度）
            point (list, optional): 回転中心点 [x, y, z]。Noneの場合はメッシュの中心
        """
        if point is None:
            point = self.mesh.center

        print(f"カスタム軸 {axis} 周りに{angle_degrees}度回転中（中心点: {point}）...")
        self.mesh = self.mesh.rotate_vector(axis, angle_degrees, point=point, inplace=False)
        print("カスタム軸回転完了")

    def reset_rotation(self):
        """回転をリセットして元の状態に戻す"""
        print("回転をリセット中...")
        self.mesh = self.original_mesh.copy()
        print("リセット完了")

    def multiple_rotations(self, rotations):
        """
        複数の回転を順次適用
        Args:
            rotations (list): 回転のリスト。各要素は ('axis', angle) のタプル
                            axis: 'x', 'y', 'z', または [x, y, z] のベクトル
        """
        print("複数回転を実行中...")
        for i, (axis, angle) in enumerate(rotations):
            print(f"  {i+1}. ", end="")
            if axis == 'x':
                self.rotate_x(angle)
            elif axis == 'y':
                self.rotate_y(angle)
            elif axis == 'z':
                self.rotate_z(angle)
            elif isinstance(axis, (list, tuple)) and len(axis) == 3:
                self.rotate_custom_axis(axis, angle)
            else:
                print(f"警告: 不正な軸指定をスキップ: {axis}")
        print("複数回転完了")

    def save_mesh(self, output_file, binary=True):
        """
        メッシュを保存
        Args:
            output_file (str): 出力ファイルパス
            binary (bool): バイナリ形式で保存するかどうか
        """
        if self.mesh is None:
            raise RuntimeError("保存するメッシュがありません")

        try:
            print(f"\nSTLファイルを保存中: {output_file}")
            self.mesh.save(output_file, binary=binary)

            # ファイルサイズを表示
            input_size = self.input_file.stat().st_size
            output_size = Path(output_file).stat().st_size

            print(f"保存完了!")
            print(f"ファイルサイズ: {input_size:,} → {output_size:,} bytes")

        except Exception as e:
            raise RuntimeError(f"STLファイルの保存に失敗: {e}")

    def show_rotation_views(self, axis='z', num_views=8, window_size=(1200, 800)):
        """
        回転した複数視点を同時表示
        Args:
            axis (str): 回転軸 ('x', 'y', 'z')
            num_views (int): 表示する視点数
            window_size (tuple): ウィンドウサイズ (幅, 高さ)
        """
        if self.original_mesh is None:
            raise RuntimeError("表示するメッシュがありません")

        print(f"\n{axis.upper()}軸回転の{num_views}視点表示を開始します...")

        try:
            # サブプロットを作成
            plotter = pv.Plotter(shape=(2, 4), window_size=window_size)

            # 各角度でのメッシュを作成・表示
            angles = np.linspace(0, 360, num_views, endpoint=False)

            for i, angle in enumerate(angles):
                # サブプロット位置を計算
                row = i // 4
                col = i % 4
                plotter.subplot(row, col)

                # メッシュを回転
                rotated_mesh = self.original_mesh.copy()
                if axis.lower() == 'x':
                    rotated_mesh = rotated_mesh.rotate_x(angle, inplace=False)
                elif axis.lower() == 'y':
                    rotated_mesh = rotated_mesh.rotate_y(angle, inplace=False)
                elif axis.lower() == 'z':
                    rotated_mesh = rotated_mesh.rotate_z(angle, inplace=False)

                # メッシュを追加
                plotter.add_mesh(rotated_mesh, color='lightblue', show_edges=True)
                plotter.add_title(f"{axis.upper()}軸 {angle:.0f}°", font_size=10)
                plotter.camera_position = 'iso'

                if i == 0:
                    # 最初のサブプロットにのみ軸を表示
                    plotter.add_axes()

            # 全体のタイトル
            plotter.add_text(f"STL {axis.upper()}軸回転表示 ({num_views}視点)", position='upper_edge')

            plotter.show()
            print("表示終了")

        except Exception as e:
            raise RuntimeError(f"回転視点表示に失敗: {e}")

    def show_simple_rotation(self, axis='z', angle=45):
        """
        シンプルな回転表示（元のメッシュと回転後を比較）
        Args:
            axis (str): 回転軸 ('x', 'y', 'z')
            angle (float): 回転角度
        """
        if self.original_mesh is None:
            raise RuntimeError("表示するメッシュがありません")

        print(f"\n{axis.upper()}軸{angle}度回転の比較表示...")

        try:
            # 回転後のメッシュを作成
            rotated_mesh = self.original_mesh.copy()
            if axis.lower() == 'x':
                rotated_mesh = rotated_mesh.rotate_x(angle, inplace=False)
            elif axis.lower() == 'y':
                rotated_mesh = rotated_mesh.rotate_y(angle, inplace=False)
            elif axis.lower() == 'z':
                rotated_mesh = rotated_mesh.rotate_z(angle, inplace=False)

            # サイドバイサイド表示
            plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 600))

            # 元のメッシュ
            plotter.subplot(0, 0)
            plotter.add_mesh(self.original_mesh, color='lightblue', show_edges=True)
            plotter.add_title("元のメッシュ", font_size=12)
            plotter.add_axes()
            plotter.camera_position = 'iso'

            # 回転後のメッシュ
            plotter.subplot(0, 1)
            plotter.add_mesh(rotated_mesh, color='lightcoral', show_edges=True)
            plotter.add_title(f"{axis.upper()}軸 {angle}°回転後", font_size=12)
            plotter.add_axes()
            plotter.camera_position = 'iso'

            # 情報表示
            info = self.get_mesh_info()
            info_text = f"頂点数: {info['points']:,} | 面数: {info['faces']:,}"
            plotter.add_text(info_text, position='upper_edge')

            plotter.show()
            print("比較表示終了")

        except Exception as e:
            raise RuntimeError(f"回転比較表示に失敗: {e}")


class STLRotatorGUI:
    """STL回転ツールのGUIアプリケーション"""

    def __init__(self, root):
        self.root = root
        self.root.title("STL回転ツール (guruguru)")
        self.root.geometry("700x800")

        self.rotator = None
        self.input_file = None
        self.setup_gui()

    def setup_gui(self):
        """GUIの設定"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ファイル選択セクション
        file_frame = ttk.LabelFrame(main_frame, text="ファイル選択", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(file_frame, text="STLファイルを選択",
                  command=self.select_input_file).grid(row=0, column=0, sticky=tk.W)

        self.input_file_label = ttk.Label(file_frame, text="ファイルが選択されていません")
        self.input_file_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))

        # メッシュ情報セクション
        info_frame = ttk.LabelFrame(main_frame, text="メッシュ情報", padding="5")
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.info_text = ScrolledText(info_frame, height=6, width=70)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 回転オプションセクション
        rotation_frame = ttk.LabelFrame(main_frame, text="回転オプション", padding="5")
        rotation_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # 基本回転（X, Y, Z軸）
        basic_frame = ttk.LabelFrame(rotation_frame, text="基本回転", padding="5")
        basic_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # X軸回転
        ttk.Label(basic_frame, text="X軸回転:").grid(row=0, column=0, sticky=tk.W)
        self.x_angle = tk.DoubleVar()
        ttk.Scale(basic_frame, from_=-180, to=180, orient=tk.HORIZONTAL, variable=self.x_angle).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        self.x_label = ttk.Label(basic_frame, text="0°")
        self.x_label.grid(row=0, column=2)
        self.x_angle.trace('w', lambda *args: self.x_label.config(text=f"{self.x_angle.get():.0f}°"))

        # Y軸回転
        ttk.Label(basic_frame, text="Y軸回転:").grid(row=1, column=0, sticky=tk.W)
        self.y_angle = tk.DoubleVar()
        ttk.Scale(basic_frame, from_=-180, to=180, orient=tk.HORIZONTAL, variable=self.y_angle).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        self.y_label = ttk.Label(basic_frame, text="0°")
        self.y_label.grid(row=1, column=2)
        self.y_angle.trace('w', lambda *args: self.y_label.config(text=f"{self.y_angle.get():.0f}°"))

        # Z軸回転
        ttk.Label(basic_frame, text="Z軸回転:").grid(row=2, column=0, sticky=tk.W)
        self.z_angle = tk.DoubleVar()
        ttk.Scale(basic_frame, from_=-180, to=180, orient=tk.HORIZONTAL, variable=self.z_angle).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 5))
        self.z_label = ttk.Label(basic_frame, text="0°")
        self.z_label.grid(row=2, column=2)
        self.z_angle.trace('w', lambda *args: self.z_label.config(text=f"{self.z_angle.get():.0f}°"))

        basic_frame.columnconfigure(1, weight=1)

        # カスタム軸回転
        custom_frame = ttk.LabelFrame(rotation_frame, text="カスタム軸回転", padding="5")
        custom_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(custom_frame, text="回転軸 (x, y, z):").grid(row=0, column=0, sticky=tk.W)
        axis_frame = ttk.Frame(custom_frame)
        axis_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self.axis_x = tk.DoubleVar(value=1.0)
        self.axis_y = tk.DoubleVar(value=0.0)
        self.axis_z = tk.DoubleVar(value=0.0)

        ttk.Entry(axis_frame, textvariable=self.axis_x, width=8).grid(row=0, column=0, padx=(0, 2))
        ttk.Entry(axis_frame, textvariable=self.axis_y, width=8).grid(row=0, column=1, padx=(0, 2))
        ttk.Entry(axis_frame, textvariable=self.axis_z, width=8).grid(row=0, column=2)

        ttk.Label(custom_frame, text="角度:").grid(row=1, column=0, sticky=tk.W)
        self.custom_angle = tk.DoubleVar()
        ttk.Scale(custom_frame, from_=-180, to=180, orient=tk.HORIZONTAL, variable=self.custom_angle).grid(row=1, column=1, sticky=(tk.W, tk.E))

        custom_frame.columnconfigure(1, weight=1)

        # ボタンセクション
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(0, 10))

        self.apply_button = ttk.Button(button_frame, text="回転適用",
                                     command=self.apply_rotation, state=tk.DISABLED)
        self.apply_button.grid(row=0, column=0, padx=(0, 5))

        self.reset_button = ttk.Button(button_frame, text="リセット",
                                     command=self.reset_rotation, state=tk.DISABLED)
        self.reset_button.grid(row=0, column=1, padx=(0, 5))

        self.save_button = ttk.Button(button_frame, text="名前を付けて保存",
                                    command=self.save_as_file, state=tk.DISABLED)
        self.save_button.grid(row=0, column=2)

        # プリセット回転ボタン
        preset_frame = ttk.LabelFrame(main_frame, text="プリセット回転", padding="5")
        preset_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(preset_frame, text="90°回転(X)", command=lambda: self.preset_rotation('x', 90)).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(preset_frame, text="90°回転(Y)", command=lambda: self.preset_rotation('y', 90)).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(preset_frame, text="90°回転(Z)", command=lambda: self.preset_rotation('z', 90)).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(preset_frame, text="180°回転(Z)", command=lambda: self.preset_rotation('z', 180)).grid(row=0, column=3)

        # アニメーション表示ボタン
        animation_frame = ttk.LabelFrame(main_frame, text="アニメーション表示", padding="5")
        animation_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Button(animation_frame, text="回転視点表示", command=lambda: self.show_rotation_views('z')).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(animation_frame, text="回転比較(X軸)", command=lambda: self.show_simple_rotation('x')).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(animation_frame, text="回転比較(Y軸)", command=lambda: self.show_simple_rotation('y')).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(animation_frame, text="回転比較(Z軸)", command=lambda: self.show_simple_rotation('z')).grid(row=0, column=3)

        # ステータスバー
        self.status_label = ttk.Label(main_frame, text="STLファイルを選択してください")
        self.status_label.grid(row=6, column=0, sticky=tk.W)

        # グリッドの重み設定
        main_frame.columnconfigure(0, weight=1)
        rotation_frame.columnconfigure(0, weight=1)

    def select_input_file(self):
        """入力ファイルの選択"""
        filename = filedialog.askopenfilename(
            title="STLファイルを選択",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.input_file = filename
                self.rotator = STLRotator(filename)

                # ファイル名を表示
                self.input_file_label.config(text=f"選択済み: {os.path.basename(filename)}")

                # メッシュ情報を表示
                self.display_mesh_info()

                # ボタンを有効化
                self.apply_button.config(state=tk.NORMAL)
                self.reset_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.NORMAL)

                self.status_label.config(text="ファイルが読み込まれました")

            except Exception as e:
                messagebox.showerror("エラー", f"ファイルの読み込みに失敗しました:\n{e}")

    def display_mesh_info(self):
        """メッシュ情報の表示"""
        if self.rotator:
            self.info_text.delete(1.0, tk.END)
            info = self.rotator.get_mesh_info()

            info_text = f"ファイル: {os.path.basename(self.input_file)}\n"
            info_text += f"頂点数: {info['points']:,}\n"
            info_text += f"面数: {info['faces']:,}\n"
            info_text += f"中心点: ({info['center'][0]:.2f}, {info['center'][1]:.2f}, {info['center'][2]:.2f})\n"
            info_text += f"バウンディングボックス:\n"
            info_text += f"  X: {info['bounds'][0]:.2f} ~ {info['bounds'][1]:.2f}\n"
            info_text += f"  Y: {info['bounds'][2]:.2f} ~ {info['bounds'][3]:.2f}\n"
            info_text += f"  Z: {info['bounds'][4]:.2f} ~ {info['bounds'][5]:.2f}\n"

            self.info_text.insert(tk.END, info_text)

    def apply_rotation(self):
        """回転の適用"""
        if not self.rotator:
            messagebox.showerror("エラー", "ファイルが選択されていません")
            return

        try:
            rotations = []

            # 基本回転を収集
            if self.x_angle.get() != 0:
                rotations.append(('x', self.x_angle.get()))
            if self.y_angle.get() != 0:
                rotations.append(('y', self.y_angle.get()))
            if self.z_angle.get() != 0:
                rotations.append(('z', self.z_angle.get()))

            # カスタム軸回転
            if self.custom_angle.get() != 0:
                axis = [self.axis_x.get(), self.axis_y.get(), self.axis_z.get()]
                rotations.append((axis, self.custom_angle.get()))

            if not rotations:
                messagebox.showwarning("警告", "回転角度が設定されていません")
                return

            # 回転を適用
            self.rotator.multiple_rotations(rotations)

            # 情報を更新
            self.display_mesh_info()

            self.status_label.config(text="回転が適用されました")
            messagebox.showinfo("完了", "回転処理が完了しました")

        except Exception as e:
            messagebox.showerror("エラー", f"回転処理中にエラーが発生しました:\n{e}")

    def reset_rotation(self):
        """回転のリセット"""
        if self.rotator:
            self.rotator.reset_rotation()
            self.display_mesh_info()

            # スライダーをリセット
            self.x_angle.set(0)
            self.y_angle.set(0)
            self.z_angle.set(0)
            self.custom_angle.set(0)

            self.status_label.config(text="回転がリセットされました")

    def preset_rotation(self, axis, angle):
        """プリセット回転"""
        if not self.rotator:
            return

        try:
            if axis == 'x':
                self.rotator.rotate_x(angle)
            elif axis == 'y':
                self.rotator.rotate_y(angle)
            elif axis == 'z':
                self.rotator.rotate_z(angle)

            self.display_mesh_info()
            self.status_label.config(text=f"{axis.upper()}軸{angle}度回転完了")

        except Exception as e:
            messagebox.showerror("エラー", f"回転エラー:\n{e}")

    def save_as_file(self):
        """ファイルの保存"""
        if not self.rotator:
            messagebox.showerror("エラー", "保存するデータがありません")
            return

        filename = filedialog.asksaveasfilename(
            title="STLファイルを保存",
            defaultextension=".stl",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.rotator.save_mesh(filename)
                messagebox.showinfo("完了", f"ファイルが保存されました:\n{filename}")
                self.status_label.config(text="ファイル保存完了")
            except Exception as e:
                messagebox.showerror("エラー", f"ファイルの保存に失敗しました:\n{e}")

    def show_rotation_views(self, axis):
        """回転視点表示"""
        if not self.rotator:
            messagebox.showerror("エラー", "STLファイルが読み込まれていません")
            return

        try:
            # 別スレッドで表示を実行
            display_thread = threading.Thread(
                target=self._run_rotation_views,
                args=(axis,)
            )
            display_thread.daemon = True
            display_thread.start()

            self.status_label.config(text=f"{axis.upper()}軸回転視点表示開始")

        except Exception as e:
            messagebox.showerror("エラー", f"表示開始エラー:\n{e}")

    def show_simple_rotation(self, axis):
        """回転比較表示"""
        if not self.rotator:
            messagebox.showerror("エラー", "STLファイルが読み込まれていません")
            return

        try:
            # 別スレッドで表示を実行
            display_thread = threading.Thread(
                target=self._run_simple_rotation,
                args=(axis,)
            )
            display_thread.daemon = True
            display_thread.start()

            self.status_label.config(text=f"{axis.upper()}軸回転比較表示開始")

        except Exception as e:
            messagebox.showerror("エラー", f"表示開始エラー:\n{e}")

    def _run_rotation_views(self, axis):
        """回転視点表示実行（別スレッド用）"""
        try:
            self.rotator.show_rotation_views(axis)
        except Exception as e:
            # メインスレッドでエラー表示
            self.root.after(0, lambda: messagebox.showerror("エラー", f"回転視点表示エラー:\n{e}"))

    def _run_simple_rotation(self, axis):
        """回転比較表示実行（別スレッド用）"""
        try:
            self.rotator.show_simple_rotation(axis, 90)  # 90度回転で比較
        except Exception as e:
            # メインスレッドでエラー表示
            self.root.after(0, lambda: messagebox.showerror("エラー", f"回転比較表示エラー:\n{e}"))
class AnimationSettingsDialog:
    """アニメーション設定ダイアログ"""

    def __init__(self, parent, axis_type):
        self.result = None
        self.axis_type = axis_type

        # ダイアログウィンドウを作成
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("アニメーション設定")
        self.dialog.geometry("300x200")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()

        # 親ウィンドウの中央に配置
        self.dialog.transient(parent)
        parent.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - 150
        y = parent.winfo_y() + (parent.winfo_height() // 2) - 100
        self.dialog.geometry(f"+{x}+{y}")

        self.setup_dialog()

        # モーダルダイアログとして実行
        self.dialog.wait_window()

    def setup_dialog(self):
        """ダイアログの設定"""
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # タイトル
        if self.axis_type == "multi":
            title = "複数軸回転アニメーション設定"
        else:
            title = f"{self.axis_type.upper()}軸回転アニメーション設定"
        ttk.Label(main_frame, text=title, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 15))

        # 回転速度設定
        ttk.Label(main_frame, text="回転速度:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.speed_var = tk.DoubleVar(value=45.0)
        speed_frame = ttk.Frame(main_frame)
        speed_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Scale(speed_frame, from_=5.0, to=180.0, orient=tk.HORIZONTAL, variable=self.speed_var).grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.speed_label = ttk.Label(speed_frame, text="45.0°/s")
        self.speed_label.grid(row=0, column=1, padx=(5, 0))
        self.speed_var.trace('w', self.update_speed_label)

        # 継続時間設定
        ttk.Label(main_frame, text="継続時間:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.duration_var = tk.DoubleVar(value=10.0)
        duration_frame = ttk.Frame(main_frame)
        duration_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Scale(duration_frame, from_=2.0, to=30.0, orient=tk.HORIZONTAL, variable=self.duration_var).grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.duration_label = ttk.Label(duration_frame, text="10.0s")
        self.duration_label.grid(row=0, column=1, padx=(5, 0))
        self.duration_var.trace('w', self.update_duration_label)

        # ボタン
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="開始", command=self.start_animation).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="キャンセル", command=self.cancel).grid(row=0, column=1)

        # グリッド設定
        main_frame.columnconfigure(1, weight=1)
        speed_frame.columnconfigure(0, weight=1)
        duration_frame.columnconfigure(0, weight=1)

    def update_speed_label(self, *args):
        """速度ラベル更新"""
        self.speed_label.config(text=f"{self.speed_var.get():.1f}°/s")

    def update_duration_label(self, *args):
        """継続時間ラベル更新"""
        self.duration_label.config(text=f"{self.duration_var.get():.1f}s")

    def start_animation(self):
        """アニメーション開始"""
        self.result = {
            'speed': self.speed_var.get(),
            'duration': self.duration_var.get()
        }
        self.dialog.destroy()

    def cancel(self):
        """キャンセル"""
        self.result = None
        self.dialog.destroy()


def run_gui():
    """GUI版の実行"""
    if not GUI_AVAILABLE:
        print("エラー: tkinterが利用できません。GUI版を実行できません。", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    app = STLRotatorGUI(root)
    root.mainloop()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="STLファイルを回転させるツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な回転
  python stl_guruguru.py input.stl -o output.stl --x 90      # X軸周りに90度回転
  python stl_guruguru.py input.stl -o output.stl --y 45      # Y軸周りに45度回転
  python stl_guruguru.py input.stl -o output.stl --z 180     # Z軸周りに180度回転

  # 複数軸の回転
  python stl_guruguru.py input.stl -o output.stl --x 90 --y 45 --z 30

  # カスタム軸回転
  python stl_guruguru.py input.stl -o output.stl --axis 1,1,0 --angle 45

  # 回転表示
  python stl_guruguru.py input.stl --animate views            # 多視点回転表示
  python stl_guruguru.py input.stl --animate compare          # Z軸回転比較
  python stl_guruguru.py input.stl --animate x                # X軸回転比較

  # GUI版を起動
  python stl_guruguru.py --gui
        """
    )

    parser.add_argument('input', nargs='?', help='入力STLファイル')
    parser.add_argument('-o', '--output', help='出力STLファイル（省略時は入力ファイル名_rotated.stl）')

    # 基本回転オプション
    parser.add_argument('--x', type=float, metavar='ANGLE', help='X軸周りの回転角度（度）')
    parser.add_argument('--y', type=float, metavar='ANGLE', help='Y軸周りの回転角度（度）')
    parser.add_argument('--z', type=float, metavar='ANGLE', help='Z軸周りの回転角度（度）')

    # カスタム軸回転
    parser.add_argument('--axis', help='カスタム回転軸（例: 1,0,1）')
    parser.add_argument('--angle', type=float, help='カスタム軸周りの回転角度（度）')

    parser.add_argument('--ascii', action='store_true', help='ASCII形式で保存（デフォルトはバイナリ形式）')
    parser.add_argument('--info', action='store_true', help='メッシュ情報のみ表示（処理は行わない）')
    parser.add_argument('--gui', action='store_true', help='GUI版を起動')

    # 表示オプション
    parser.add_argument('--animate', choices=['x', 'y', 'z', 'views', 'compare'], help='回転表示（x, y, z: 軸比較, views: 多視点, compare: Z軸比較）')

    args = parser.parse_args()

    # GUI版の起動
    if args.gui:
        run_gui()
        return

    # 入力ファイルのチェック
    if not args.input:
        parser.error("入力ファイルが指定されていません（--guiオプションを使用してGUI版を起動することもできます）")

    try:
        # STLローテーターを初期化
        rotator = STLRotator(args.input)

        # 情報表示のみの場合
        if args.info:
            info = rotator.get_mesh_info()
            print(f"\n入力STLファイル情報:")
            print(f"  - 頂点数: {info['points']:,}")
            print(f"  - 面数: {info['faces']:,}")
            print(f"  - 中心点: {info['center']}")
            print(f"  - バウンディングボックス: {info['bounds']}")
            return

        # 表示処理の場合
        if args.animate:
            if args.animate == 'views':
                rotator.show_rotation_views('z', num_views=8)
            elif args.animate == 'compare':
                rotator.show_simple_rotation('z', 90)
            elif args.animate in ['x', 'y', 'z']:
                rotator.show_simple_rotation(args.animate, 90)
            return

        # 出力ファイル名を決定
        if args.output:
            output_file = args.output
        else:
            output_file = rotator.input_file.stem + "_rotated" + rotator.input_file.suffix

        # 回転処理
        rotations = []

        # 基本回転を収集
        if args.x is not None:
            rotations.append(('x', args.x))
        if args.y is not None:
            rotations.append(('y', args.y))
        if args.z is not None:
            rotations.append(('z', args.z))

        # カスタム軸回転
        if args.axis is not None and args.angle is not None:
            try:
                axis = [float(x.strip()) for x in args.axis.split(',')]
                if len(axis) != 3:
                    raise ValueError("軸は x,y,z の3つの値で指定してください")
                rotations.append((axis, args.angle))
            except ValueError as e:
                print(f"エラー: 軸の指定が不正です: {e}", file=sys.stderr)
                sys.exit(1)

        if not rotations:
            print("エラー: 回転パラメータが指定されていません", file=sys.stderr)
            sys.exit(1)

        # 回転を適用
        rotator.multiple_rotations(rotations)

        # ファイルを保存
        rotator.save_mesh(output_file, binary=not args.ascii)

        # 最終的な情報表示
        info = rotator.get_mesh_info()
        print(f"\n処理後のメッシュ情報:")
        print(f"  - 中心点: ({info['center'][0]:.2f}, {info['center'][1]:.2f}, {info['center'][2]:.2f})")
        print(f"  - バウンディングボックス: {info['bounds']}")

    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
