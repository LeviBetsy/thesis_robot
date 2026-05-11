# """
# keyboard_controller_ssh.py

# Overview:
# This module provides a TCP server (RobotController) designed to receive control 
# signals from a remote client, typically forwarded over an SSH tunnel. It 
# decouples the network listening logic from the robot's execution logic using 
# a callback mechanism and multi-threading.

# Technical Details:
# - Protocol: TCP (SOCK_STREAM) bound to 127.0.0.1.
# - Concurrency: The 'connect' loop runs in a daemonized background thread to 
#   prevent blocking the main execution loop.
# - Data Format: Expects UTF-8 encoded strings in the format "int,int,int\n". 
# - Command Handling: Supports handling multiple commands in a single packet 
#   by splitting on newline characters.

# Usage:
# 1. Define a callback function that accepts three integers: 
#    def my_callback(val1, val2, val3): ... (presumably this call back function is the function to send DIR,LDUTY,RDUTY over UART)
# 2. Instantiate RobotController(command_callback=my_callback, port=8080).
# 3. Call .start() to begin listening for incoming connections.
# 4. On connection loss, the controller automatically triggers a safety stop (0,0,0).
# """


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
        keyboard_thread = threading.Thread(target=self.connect, daemon=True)
        keyboard_thread.start()

    def connect(self):
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