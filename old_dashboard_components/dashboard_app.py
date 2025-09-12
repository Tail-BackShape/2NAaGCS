
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import socket
import json
import time
import subprocess
import sys
import os
import atexit
import serial.tools.list_ports # Import serial.tools.list_ports

# --- Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 12345 # Same as telemetry_gateway.py
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000 # Default Flask port

# --- Subprocess Management ---
telemetry_gateway_process = None
renderer_3d_process = None

script_dir = os.path.dirname(os.path.abspath(__file__))
telemetry_gateway_path = os.path.join(script_dir, "telemetry_gateway.py")
renderer_3d_path = os.path.join(script_dir, "renderer_3d.py")

def start_telemetry_gateway(com_port=None):
    global telemetry_gateway_process
    if telemetry_gateway_process and telemetry_gateway_process.poll() is None:
        print("[*] Terminating existing telemetry_gateway.py process...")
        telemetry_gateway_process.terminate()
        telemetry_gateway_process.wait(timeout=5)
        telemetry_gateway_process = None

    print(f"[*] Starting telemetry_gateway.py on port {com_port if com_port else 'auto-detect'}...")
    cmd = [sys.executable, telemetry_gateway_path]
    if com_port:
        cmd.extend(["--port", com_port])
    
    telemetry_gateway_process = subprocess.Popen(cmd)
    print(f"[*] telemetry_gateway.py started with PID: {telemetry_gateway_process.pid}")
    return True

def start_renderer_3d():
    global renderer_3d_process
    if renderer_3d_process and renderer_3d_process.poll() is None:
        print("[*] Terminating existing renderer_3d.py process...")
        renderer_3d_process.terminate()
        renderer_3d_process.wait(timeout=5)
        renderer_3d_process = None

    print("[*] Starting renderer_3d.py...")
    renderer_3d_process = subprocess.Popen([sys.executable, renderer_3d_path])
    print(f"[*] renderer_3d.py started with PID: {renderer_3d_process.pid}")
    return True

def stop_subprocesses():
    global telemetry_gateway_process, renderer_3d_process
    print("[*] Shutting down subprocesses...")
    if telemetry_gateway_process and telemetry_gateway_process.poll() is None:
        telemetry_gateway_process.terminate()
        telemetry_gateway_process.wait(timeout=5)
        print("[*] telemetry_gateway.py terminated.")
    if renderer_3d_process and renderer_3d_process.poll() is None:
        renderer_3d_process.terminate()
        renderer_3d_process.wait(timeout=5)
        print("[*] renderer_3d.py terminated.")

# Register the cleanup function to run on exit
atexit.register(stop_subprocesses)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this in production
socketio = SocketIO(app, cors_allowed_origins="*") # Allow all origins for development

# --- UDP Listener Thread ---
def udp_listener():
    """Listens for UDP packets and emits them via SocketIO."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Add this line
    sock.bind((UDP_IP, UDP_PORT))
    print(f"[*] Dashboard UDP listener started on {UDP_IP}:{UDP_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            message = data.decode('utf-8')
            
            # Parse the message (same format as before)
            parts = [float(p) for p in message.split(',')]
            if len(parts) >= 18: # Ensure we have enough data
                telemetry_data = {
                    "roll": parts[0],
                    "pitch": parts[1],
                    "yaw": parts[2],
                    "altitude": parts[3],
                    "aileron": parts[4],
                    "elevator": parts[5],
                    "throttle": parts[6],
                    "rudder": parts[7],
                    "aux1": parts[8],
                    "aux2": parts[9],
                    "aux3": parts[10],
                    "aux4": parts[11],
                    # ArUco data is not used in this dashboard for now
                }
                # Emit data to all connected SocketIO clients
                socketio.emit('telemetry_update', telemetry_data)
            # print(f"Received UDP: {message}") # For debugging
        except (ValueError, IndexError, UnicodeDecodeError):
            # print(f"Could not parse UDP packet: {data}")
            pass
        except Exception as e:
            print(f"Error in UDP listener: {e}")
            time.sleep(1) # Prevent busy-loop on persistent errors

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main dashboard HTML page."""
    return render_template('index.html')

# --- SocketIO Events ---
@socketio.on('list_com_ports')
def handle_list_com_ports():
    ports = []
    for port in serial.tools.list_ports.comports():
        ports.append({"device": port.device, "description": port.description})
    emit('com_ports_list', ports)
    print(f"[*] Emitted COM ports list: {ports}")

@socketio.on('connect_com_port')
def handle_connect_com_port(data):
    com_port = data.get('port')
    if com_port:
        try:
            success = start_telemetry_gateway(com_port)
            emit('com_port_status', {'success': success, 'message': f'Connected to {com_port}'})
        except Exception as e:
            emit('com_port_status', {'success': False, 'message': f'Failed to connect to {com_port}: {e}'})
    else:
        emit('com_port_status', {'success': False, 'message': 'No COM port specified.'})

# --- Main Execution ---
if __name__ == '__main__':
    # Start renderer_3d.py automatically
    start_renderer_3d()
    # telemetry_gateway.py will be started via SocketIO event from frontend

    # Start the UDP listener in a background thread
    udp_thread = threading.Thread(target=udp_listener, daemon=True)
    udp_thread.start()

    # Start the Flask-SocketIO server
    print(f"[*] Starting Dashboard Flask server on {FLASK_HOST}:{FLASK_PORT}")
    print(f"[*] Access dashboard at: http://127.0.0.1:{FLASK_PORT}/")
    socketio.run(app, host=FLASK_HOST, port=FLASK_PORT, allow_unsafe_werkzeug=True)
