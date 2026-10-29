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
# - Data Format: Expects UTF-8 encoded strings in the format "left_cps,right_cps\n",
#   signed encoder counts per second (negative = reverse).
# - Command Handling: Supports handling multiple commands in a single packet
#   by splitting on newline characters.

# Usage:
# 1. Define a callback function that accepts two signed integers:
#    def my_callback(left_cps, right_cps): ... (presumably this call back function sends the speeds over UART)
# 2. Instantiate RobotController(command_callback=my_callback, port=8080).
# 3. Call .start() to begin listening for incoming connections.
# 4. On connection loss, the controller automatically triggers a safety stop (0,0).
# """


import socket
import threading
import time

# The last command is re-sent over UART this often (seconds), so a single packet the MSP432
# drops doesn't leave the robot ignoring a key
RESEND_PERIOD = 0.1

# keyboard_stream.py re-sends its command every 0.1 s. Hearing nothing for this long (seconds)
# means the WiFi link stalled, so the robot is stopped instead of driving on the last command
COMMAND_TIMEOUT = 0.5

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
                conn.settimeout(RESEND_PERIOD) # recv wakes up to resend and check the watchdog
                current_cmd = (0, 0)
                last_heard = time.monotonic()
                buffer = ""
                with conn:
                    while self.running:
                        try:
                            data = conn.recv(1024)
                            if not data:
                                break
                            # TCP is a byte stream, a command can be split across two recvs, so keep the unfinished tail
                            buffer += data.decode('utf-8')
                            *lines, buffer = buffer.split('\n')
                            for cmd_str in lines:
                                parts = [int(x) for x in cmd_str.split(',')] if cmd_str else []
                                if len(parts) == 2:
                                    current_cmd = (parts[0], parts[1])
                                    last_heard = time.monotonic()
                        except socket.timeout:
                            pass
                        if time.monotonic() - last_heard > COMMAND_TIMEOUT and current_cmd != (0, 0):
                            print("No command from the laptop in time, stopping")
                            current_cmd = (0, 0)
                        self.command_callback(*current_cmd) # new command, or resend of the current one
                print("Laptop disconnected")
                self.command_callback(0, 0) # Safety stop
            except Exception as e:
                print(f"Connection lost or error: {e}")
                self.command_callback(0, 0) # Safety stop

    def stop(self):
        self.running = False
        self.sock.close()