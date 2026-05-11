import socket
import threading

class RobotController:
    def __init__(self, command_callback, port=8080):
        self.command_callback = command_callback
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # TCP
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('127.0.0.1', self.port))
        self.running = True

    def start(self):
        self.sock.listen(1)
        print(f"TCP Server listening on port {self.port}...")
        
        while self.running:
            try:
                conn, addr = self.sock.accept()
                print(f"Connected by {addr}")
                with conn:
                    while self.running:
                        data = conn.recv(1024)
                        if not data:
                            break 
                        
                        msg = data.decode('utf-8')
                        # Handle multiple commands if they arrive stuck together
                        for cmd_str in msg.strip().split('\n'):
                            if cmd_str:
                                parts = [int(x) for x in cmd_str.split(',')]
                                if len(parts) == 3:
                                    self.command_callback(parts[0], parts[1], parts[2])
            except Exception as e:
                print(f"Connection lost or error: {e}")
                self.command_callback(0, 0, 0) # Safety stop

    def stop(self):
        self.running = False
        self.sock.close()