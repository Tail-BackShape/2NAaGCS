from flask import Flask, Response
from picamera2 import Picamera2
import cv2

app = Flask(__name__)

# --- 画像サイズ設定 ---
CAPTURE_SIZE = (1024, 768)  # カメラ取得サイズ（高解像度でArUco検出精度向上）
STREAM_SIZE = (320, 240)   # 配信サイズ（軽量化でストリーミング高速化）

# --- Picamera2 セットアップ ---
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": CAPTURE_SIZE, "format": "BGR888"}
)
picam2.configure(config)
picam2.start()

# --- ArUco 検出用の設定 ---
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters_create()

def generate():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


        # --- ArUco マーカー検出 ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

        if ids is not None:
            # マーカーを枠で囲み、ID を描画
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, corner in enumerate(corners):
                c = corner[0]
                # ID の位置にテキストを描画
                cv2.putText(frame, f"ID:{ids[i][0]}", (int(c[0][0]), int(c[0][1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # --- 配信用に低解像度にリサイズ ---
        stream_frame = cv2.resize(frame, STREAM_SIZE)

        # --- JPEG にエンコードして配信 ---
        ret, jpeg = cv2.imencode('.jpg', stream_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# トップページからアクセスした場合に自動で映像を埋め込む
@app.route('/')
def index():
    return '<h1>Raspberry Pi Zero Flask ArUco Stream</h1><img src="/video">'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
