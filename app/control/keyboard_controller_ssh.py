import socket
import threading

class RobotController: #a listener to listen to the ssh-ing laptop for commands to send to the MSP432
    def __init__(self, command_callback, port=5005):
        self.command_callback = command_callback
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        self.sock.bind(('', self.port))
        self.running = True

    def start(self):
        print(f"Listening for laptop commands on UDP port {self.port}...")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                # Data format: "INST,LEFT,RIGHT"
                msg = data.decode('utf-8')
                parts = [int(x) for x in msg.split(',')]
                if len(parts) == 3:
                    self.command_callback(parts[0], parts[1], parts[2])
            except Exception as e:
                print(f"Receiver Error: {e}")

    def stop(self):
        self.running = False
        self.sock.close()