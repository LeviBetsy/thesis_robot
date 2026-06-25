import socket
import threading
import json

class PoseStream:
    def __init__(self, port=5000):
        self.port = port
        # Set up a standard TCP socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Allow immediate port reuse if the script crashes
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('127.0.0.1', self.port))
        self.server_socket.listen(1)
        
        self.client_socket = None
        self.running = True

        # Start listening for connections in the background
        self.thread = threading.Thread(target=self._accept_connections, daemon=True)
        self.thread.start()
        print(f"Telemetry server listening on tcp://127.0.0.1:{self.port}")

    def _accept_connections(self):
        while self.running:
            try:
                # Use a timeout so the thread can exit cleanly if self.running becomes False
                self.server_socket.settimeout(1.0)
                client, addr = self.server_socket.accept()
                print(f"Data client connected from {addr}")
                self.client_socket = client
            except socket.timeout:
                continue
            except Exception as e:
                break

    def send_data(self, data_dict):
        # Only attempt to send if a client (your laptop) is actually connected
        if self.client_socket:
            try:
                # Convert the dictionary to a JSON string and append the newline delimiter
                message = json.dumps(data_dict) + "\n"
                self.client_socket.sendall(message.encode('utf-8'))
            except Exception as e:
                print("Client disconnected.")
                self.client_socket.close()
                self.client_socket = None

    def stop(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()