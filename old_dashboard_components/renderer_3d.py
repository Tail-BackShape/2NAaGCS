from flask import Flask, Response
import threading
import socket
import open3d as o3d
import numpy as np
import cv2
import time
import math

# --- Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 12345
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001

# --- Global State for Telemetry ---
telemetry_lock = threading.Lock()
# Default attitude: roll, pitch, yaw in degrees
latest_attitude = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}

# --- UDP Listener Thread ---
def udp_listener():
    """Listens for UDP packets and updates the global attitude."""
    global latest_attitude
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Add this line
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[*] UDP listener started on {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            message = data.decode('utf-8')
            parts = [float(p) for p in message.split(',')]
            if len(parts) >= 3:
                roll, pitch, yaw = parts[0], parts[1], parts[2]
                with telemetry_lock:
                    latest_attitude["roll"] = roll
                    latest_attitude["pitch"] = pitch
                    latest_attitude["yaw"] = yaw
        except (ValueError, IndexError, UnicodeDecodeError):
            # Silently ignore malformed packets
            pass

# --- Flask Application ---
app = Flask(__name__)

def generate_frames():
    """Sets up Open3D and generates frames for the MJPEG stream."""
    print("[*] Initializing 3D renderer...")

    # --- Open3D Setup ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=False) # Off-screen rendering

    # Load mesh
    try:
        # Assuming the model is oriented with +X forward, +Z up
        mesh = o3d.io.read_triangle_mesh("plane.stl")
        print("[+] Loaded plane.stl")
    except Exception:
        print("[!] plane.stl not found, using placeholder cube.")
        mesh = o3d.geometry.TriangleMesh.create_box()
    
    mesh.compute_vertex_normals()
    original_mesh = o3d.geometry.TriangleMesh(mesh) # Keep an unmodified copy

    # Add a coordinate system axis for reference
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])

    # View control and render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15]) # Dark grey background
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_zoom(0.9)
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([0, -1, -0.2]) # Camera view angle
    ctr.set_up([0, 0, 1])
    
    print("[+] Open3D renderer initialized successfully.")

    while True:
        with telemetry_lock:
            roll = latest_attitude["roll"]
            pitch = latest_attitude["pitch"]
            yaw = latest_attitude["yaw"]

        # --- Rotation Logic ---
        rotated_mesh = o3d.geometry.TriangleMesh(original_mesh)

        # Convert degrees to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        # Build rotation matrix based on drone conventions (Yaw, Pitch, Roll)
        # This assumes a specific model orientation and may need tuning.
        R_yaw = rotated_mesh.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * yaw_rad)
        R_pitch = rotated_mesh.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * pitch_rad)
        R_roll = rotated_mesh.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * roll_rad)
        
        # Apply rotations in Yaw -> Pitch -> Roll order
        R = R_yaw @ R_pitch @ R_roll
        rotated_mesh.rotate(R, center=(0, 0, 0))
        
        # --- Update Visualizer and Render Frame ---
        vis.clear_geometries()
        vis.add_geometry(rotated_mesh)
        vis.add_geometry(axes)
        vis.poll_events()
        vis.update_renderer()

        img = vis.capture_screen_image_to_numpy(do_render=True)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', img_bgr)
        if not ret:
            continue
        
        frame = buffer.tobytes()

        # Yield the frame for the MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Small delay to aim for ~30 FPS and reduce CPU load
        time.sleep(0.03)

@app.route('/')
def stream():
    """Main video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the UDP listener in a background thread
    udp_thread = threading.Thread(target=udp_listener, daemon=True)
    udp_thread.start()

    # Start the Flask web server
    print(f"[*] Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
    print(f"[*] MJPEG stream available at: http://127.0.0.1:{FLASK_PORT}/")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
