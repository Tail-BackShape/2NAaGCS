import serial
import serial.tools.list_ports
import socket
import time
import sys
import argparse # Import argparse

# --- Configuration ---
BAUD_RATE = 115200
UDP_BROADCAST_IP = '<broadcast>'
UDP_PORT = 12345
RECONNECT_DELAY_S = 5  # seconds

def find_best_serial_port():
    """Finds the first available serial port."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        return None
    
    print("--- Available Serial Ports ---")
    for i, port in enumerate(ports):
        print(f"  {i}: {port.device} - {port.description}")
    print("------------------------------")
    
    # For now, returning the first port found.
    # A more advanced version could take user input or look for a specific description.
    selected_port = ports[0].device
    print(f"Selected port {selected_port} for connection.")
    return selected_port

def main(com_port=None): # Add com_port argument
    """Main function to run the telemetry gateway."""
    print("--- Telemetry Gateway Starting ---")

    # Setup UDP socket for broadcasting
    try:
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        broadcast_address = (UDP_BROADCAST_IP, UDP_PORT)
        print(f"[*] Ready to broadcast UDP packets to {broadcast_address}")
    except Exception as e:
        print(f"[!] Failed to create UDP socket: {e}")
        sys.exit(1)

    serial_connection = None

    while True:
        try:
            # --- Connection Phase ---
            if serial_connection is None or not serial_connection.is_open:
                port_to_connect = com_port # Use provided com_port
                if port_to_connect is None: # If no com_port provided, find best
                    port_to_connect = find_best_serial_port()

                if not port_to_connect:
                    print(f"[!] No serial ports found. Retrying in {RECONNECT_DELAY_S} seconds...")
                    time.sleep(RECONNECT_DELAY_S)
                    continue
                
                print(f"[*] Attempting to connect to {port_to_connect} at {BAUD_RATE} bps...")
                serial_connection = serial.Serial(port_to_connect, BAUD_RATE, timeout=1)
                print(f"[+] Successfully connected to {port_to_connect}.")

            # --- Reading and Broadcasting Phase ---
            line = serial_connection.readline().decode('utf-8').strip()
            if line:
                # Forward the raw line via UDP
                udp_socket.sendto(line.encode('utf-8'), broadcast_address)
                # print(f"  -> Sent UDP: {line}") # Commented out for less verbose output

        except serial.SerialException as e:
            print(f"[!] Serial error: {e}")
            if serial_connection and serial_connection.is_open:
                serial_connection.close()
            serial_connection = None
            print(f"[*] Connection lost. Retrying in {RECONNECT_DELAY_S} seconds...")
            time.sleep(RECONNECT_DELAY_S)
        except UnicodeDecodeError:
            # Silently ignore lines that can't be decoded
            pass
        except KeyboardInterrupt:
            print("\n[!] Keyboard interrupt detected. Shutting down.")
            break
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}")
            break

    # --- Cleanup ---
    if serial_connection and serial_connection.is_open:
        serial_connection.close()
    udp_socket.close()
    print("--- Gateway Stopped ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telemetry Gateway for RC Drone.")
    parser.add_argument("--port", type=str, help="Specify the COM port to connect to (e.g., COM3 or /dev/ttyUSB0).")
    args = parser.parse_args()
    
    main(com_port=args.port)

