#!/usr/bin/env python3
"""
STL軽量化プログラム
STLファイルのメッシュを軽量化（ポリゴン数削減、スムージング）するツール
"""

import argparse
import os
import sys
import pyvista as pv
from pathlib import Path

# GUI用のインポート（オプショナル）
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter.scrolledtext import ScrolledText
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class STLOptimizer:
    """STLファイル軽量化クラス"""

    def __init__(self, input_file):
        """
        初期化
        Args:
            input_file (str): 入力STLファイルのパス
        """
        self.input_file = Path(input_file)
        self.mesh = None
        self.original_points = 0
        self.original_faces = 0

        if not self.input_file.exists():
            raise FileNotFoundError(f"入力ファイルが見つかりません: {input_file}")

        # STLファイルを読み込み
        self.load_mesh()

    def load_mesh(self):
        """STLファイルを読み込み"""
        try:
            print(f"STLファイルを読み込み中: {self.input_file}")
            self.mesh = pv.read(str(self.input_file))
            self.original_points = self.mesh.n_points
            self.original_faces = self.mesh.n_cells
            print(f"元のメッシュ情報:")
            print(f"  - 頂点数: {self.original_points:,}")
            print(f"  - 面数: {self.original_faces:,}")
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
            'volume': self.mesh.volume if hasattr(self.mesh, 'volume') else 'N/A'
        }

    def print_mesh_info(self, title="メッシュ情報"):
        """メッシュ情報を表示"""
        info = self.get_mesh_info()
        if info:
            print(f"\n{title}:")
            print(f"  - 頂点数: {info['points']:,}")
            print(f"  - 面数: {info['faces']:,}")
            if info['volume'] != 'N/A':
                print(f"  - 体積: {info['volume']:.3f}")

    def decimate_mesh(self, reduction_ratio):
        """
        メッシュの簡略化（ポリゴン数削減）
        Args:
            reduction_ratio (float): 削減率 (0.0-1.0, 小さいほど軽量化)
        """
        if self.mesh is None:
            raise RuntimeError("処理するメッシュがありません")

        if not (0.0 <= reduction_ratio <= 1.0):
            raise ValueError("削減率は0.0から1.0の間で指定してください")

        try:
            print(f"\nメッシュ簡略化を実行中（削減率: {reduction_ratio}）...")

            # Decimateフィルタを適用
            decimated = self.mesh.decimate(reduction_ratio)

            # 結果を表示
            original_faces = self.mesh.n_cells
            new_faces = decimated.n_cells
            actual_reduction = (original_faces - new_faces) / original_faces

            print(f"簡略化完了:")
            print(f"  - 面数: {original_faces:,} → {new_faces:,}")
            print(f"  - 実際の削減率: {actual_reduction:.1%}")

            self.mesh = decimated
            return decimated

        except Exception as e:
            raise RuntimeError(f"メッシュ簡略化に失敗: {e}")

    def smooth_mesh(self, iterations=10, relaxation_factor=0.01):
        """
        メッシュのスムージング
        Args:
            iterations (int): スムージングの反復回数
            relaxation_factor (float): 緩和係数
        """
        if self.mesh is None:
            raise RuntimeError("処理するメッシュがありません")

        if iterations < 1:
            raise ValueError("反復回数は1以上で指定してください")

        try:
            print(f"\nメッシュスムージングを実行中（反復回数: {iterations}）...")

            # Smoothフィルタを適用
            smoothed = self.mesh.smooth(n_iter=iterations, relaxation_factor=relaxation_factor)

            print(f"スムージング完了:")
            print(f"  - 反復回数: {iterations}")
            print(f"  - 緩和係数: {relaxation_factor}")

            self.mesh = smoothed
            return smoothed

        except Exception as e:
            raise RuntimeError(f"メッシュスムージングに失敗: {e}")

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
            size_reduction = (1 - output_size / input_size) * 100

            print(f"保存完了!")
            print(f"ファイルサイズ: {input_size:,} → {output_size:,} bytes")
            print(f"サイズ削減率: {size_reduction:.1f}%")

        except Exception as e:
            raise RuntimeError(f"STLファイルの保存に失敗: {e}")


class STLOptimizerGUI:
    """STL軽量化ツールのGUIアプリケーション"""

    def __init__(self, root):
        self.root = root
        self.root.title("STL軽量化ツール")
        self.root.geometry("800x700")

        self.optimizer = None
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

        self.info_text = ScrolledText(info_frame, height=8, width=80)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        # 軽量化オプションセクション
        options_frame = ttk.LabelFrame(main_frame, text="軽量化オプション", padding="5")
        options_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # メッシュ簡略化
        self.decimate_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="メッシュ簡略化を適用",
                       variable=self.decimate_var).grid(row=0, column=0, sticky=tk.W)

        ttk.Label(options_frame, text="削減率:").grid(row=1, column=0, sticky=tk.W, padx=(20, 5))
        self.decimate_scale = ttk.Scale(options_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL)
        self.decimate_scale.set(0.5)
        self.decimate_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.decimate_value_label = ttk.Label(options_frame, text="0.5")
        self.decimate_value_label.grid(row=1, column=2, sticky=tk.W)
        self.decimate_scale.configure(command=self.update_decimate_label)

        # スムージング
        self.smooth_var = tk.BooleanVar()
        ttk.Checkbutton(options_frame, text="スムージングを適用",
                       variable=self.smooth_var).grid(row=2, column=0, sticky=tk.W, pady=(10, 0))

        ttk.Label(options_frame, text="反復回数:").grid(row=3, column=0, sticky=tk.W, padx=(20, 5))
        self.smooth_scale = ttk.Scale(options_frame, from_=1, to=20, orient=tk.HORIZONTAL)
        self.smooth_scale.set(10)
        self.smooth_scale.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        self.smooth_value_label = ttk.Label(options_frame, text="10")
        self.smooth_value_label.grid(row=3, column=2, sticky=tk.W)
        self.smooth_scale.configure(command=self.update_smooth_label)

        # 実行ボタンセクション
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(0, 10))

        self.process_button = ttk.Button(button_frame, text="軽量化実行",
                                       command=self.process_file, state=tk.DISABLED)
        self.process_button.grid(row=0, column=0, padx=(0, 10))

        ttk.Button(button_frame, text="名前を付けて保存",
                  command=self.save_as_file, state=tk.DISABLED).grid(row=0, column=1)

        # プログレスバー
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # ステータスバー
        self.status_label = ttk.Label(main_frame, text="STLファイルを選択してください")
        self.status_label.grid(row=5, column=0, sticky=tk.W)

        # グリッドの重み設定
        main_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)

    def update_decimate_label(self, value):
        """削減率ラベルの更新"""
        self.decimate_value_label.config(text=f"{float(value):.2f}")

    def update_smooth_label(self, value):
        """スムージング反復回数ラベルの更新"""
        self.smooth_value_label.config(text=f"{int(float(value))}")

    def select_input_file(self):
        """入力ファイルの選択"""
        filename = filedialog.askopenfilename(
            title="STLファイルを選択",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.input_file = filename
                self.optimizer = STLOptimizer(filename)

                # ファイル名を表示
                self.input_file_label.config(text=f"選択済み: {os.path.basename(filename)}")

                # メッシュ情報を表示
                self.display_mesh_info()

                # ボタンを有効化
                self.process_button.config(state=tk.NORMAL)

                self.status_label.config(text="ファイルが読み込まれました")

            except Exception as e:
                messagebox.showerror("エラー", f"ファイルの読み込みに失敗しました:\n{e}")

    def display_mesh_info(self):
        """メッシュ情報の表示"""
        if self.optimizer:
            self.info_text.delete(1.0, tk.END)
            info = self.optimizer.get_mesh_info()

            info_text = f"ファイル: {os.path.basename(self.input_file)}\n"
            info_text += f"頂点数: {info['points']:,}\n"
            info_text += f"面数: {info['faces']:,}\n"
            info_text += f"バウンディングボックス:\n"
            info_text += f"  X: {info['bounds'][0]:.3f} ~ {info['bounds'][1]:.3f}\n"
            info_text += f"  Y: {info['bounds'][2]:.3f} ~ {info['bounds'][3]:.3f}\n"
            info_text += f"  Z: {info['bounds'][4]:.3f} ~ {info['bounds'][5]:.3f}\n"
            if info['volume'] != 'N/A':
                info_text += f"体積: {info['volume']:.3f}\n"

            file_size = Path(self.input_file).stat().st_size
            info_text += f"ファイルサイズ: {file_size:,} bytes ({file_size/1024:.1f} KB)\n"

            self.info_text.insert(tk.END, info_text)

    def process_file(self):
        """ファイルの処理"""
        if not self.optimizer:
            messagebox.showerror("エラー", "ファイルが選択されていません")
            return

        try:
            self.progress.start(10)
            self.status_label.config(text="処理中...")
            self.root.update()

            processed = False

            # メッシュ簡略化
            if self.decimate_var.get():
                reduction_ratio = self.decimate_scale.get()
                self.optimizer.decimate_mesh(reduction_ratio)
                processed = True

            # スムージング
            if self.smooth_var.get():
                iterations = int(self.smooth_scale.get())
                self.optimizer.smooth_mesh(iterations)
                processed = True

            if not processed:
                messagebox.showwarning("警告", "軽量化オプションが選択されていません")
                return

            # 処理結果を表示
            self.display_processed_info()

            self.progress.stop()
            self.status_label.config(text="処理完了")
            messagebox.showinfo("完了", "軽量化処理が完了しました")

        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="処理エラー")
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{e}")

    def display_processed_info(self):
        """処理後の情報表示"""
        if self.optimizer:
            self.info_text.delete(1.0, tk.END)
            info = self.optimizer.get_mesh_info()

            info_text = f"処理後のメッシュ情報:\n\n"
            info_text += f"頂点数: {self.optimizer.original_points:,} → {info['points']:,}\n"
            info_text += f"面数: {self.optimizer.original_faces:,} → {info['faces']:,}\n"

            point_reduction = (self.optimizer.original_points - info['points']) / self.optimizer.original_points * 100
            face_reduction = (self.optimizer.original_faces - info['faces']) / self.optimizer.original_faces * 100

            info_text += f"頂点削減率: {point_reduction:.1f}%\n"
            info_text += f"面削減率: {face_reduction:.1f}%\n"

            self.info_text.insert(tk.END, info_text)

    def save_as_file(self):
        """ファイルの保存"""
        if not self.optimizer:
            messagebox.showerror("エラー", "保存するデータがありません")
            return

        filename = filedialog.asksaveasfilename(
            title="STLファイルを保存",
            defaultextension=".stl",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.optimizer.save_mesh(filename)
                messagebox.showinfo("完了", f"ファイルが保存されました:\n{filename}")
                self.status_label.config(text="ファイル保存完了")
            except Exception as e:
                messagebox.showerror("エラー", f"ファイルの保存に失敗しました:\n{e}")


def run_gui():
    """GUI版の実行"""
    if not GUI_AVAILABLE:
        print("エラー: tkinterが利用できません。GUI版を実行できません。", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    app = STLOptimizerGUI(root)
    root.mainloop()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="STLファイルを軽量化するツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python stl2light.py input.stl -o output.stl

  # 削減率50%でメッシュを簡略化
  python stl2light.py input.stl -o output.stl --decimate 0.5

  # スムージングを適用
  python stl2light.py input.stl -o output.stl --smooth 10

  # 両方を適用
  python stl2light.py input.stl -o output.stl --decimate 0.3 --smooth 5

  # GUI版を起動
  python stl2light.py --gui
        """
    )

    parser.add_argument('input', nargs='?', help='入力STLファイル')
    parser.add_argument('-o', '--output', help='出力STLファイル（省略時は入力ファイル名_light.stl）')
    parser.add_argument('--decimate', type=float, metavar='RATIO',
                       help='メッシュ簡略化率 (0.0-1.0, 小さいほど軽量化)')
    parser.add_argument('--smooth', type=int, metavar='ITERATIONS',
                       help='スムージング反復回数 (1-50)')
    parser.add_argument('--ascii', action='store_true',
                       help='ASCII形式で保存（デフォルトはバイナリ形式）')
    parser.add_argument('--info', action='store_true',
                       help='メッシュ情報のみ表示（処理は行わない）')
    parser.add_argument('--gui', action='store_true',
                       help='GUI版を起動')

    args = parser.parse_args()

    # GUI版の起動
    if args.gui:
        run_gui()
        return

    # 入力ファイルのチェック
    if not args.input:
        parser.error("入力ファイルが指定されていません（--guiオプションを使用してGUI版を起動することもできます）")

    try:
        # STLオプティマイザーを初期化
        optimizer = STLOptimizer(args.input)

        # 情報表示のみの場合
        if args.info:
            optimizer.print_mesh_info("入力STLファイル情報")
            return

        # 処理前の情報表示
        optimizer.print_mesh_info("処理前のメッシュ情報")

        # 出力ファイル名を決定
        if args.output:
            output_file = args.output
        else:
            output_file = optimizer.input_file.stem + "_light" + optimizer.input_file.suffix

        # 処理を実行
        processed = False

        # メッシュ簡略化
        if args.decimate is not None:
            if not (0.0 <= args.decimate <= 1.0):
                print("エラー: 削減率は0.0から1.0の間で指定してください", file=sys.stderr)
                sys.exit(1)
            optimizer.decimate_mesh(args.decimate)
            processed = True

        # スムージング
        if args.smooth is not None:
            if args.smooth < 1 or args.smooth > 50:
                print("エラー: スムージング反復回数は1から50の間で指定してください", file=sys.stderr)
                sys.exit(1)
            optimizer.smooth_mesh(args.smooth)
            processed = True

        # 何も処理しない場合は元のメッシュをコピー保存
        if not processed:
            print("\n軽量化オプションが指定されていません。元のメッシュをコピー保存します。")

        # ファイルを保存
        optimizer.save_mesh(output_file, binary=not args.ascii)

        # 処理後の情報表示
        optimizer.print_mesh_info("処理後のメッシュ情報")

    except Exception as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
