import zmq
import threading
import json
import time

class PoseStreamer:
    def __init__(self, port=5000):
        self.port = port
        
        # Initialize ZeroMQ context and PUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # Bind to all interfaces using wildcard (*) so external machines can connect
        self.socket.bind(f"tcp://*:{self.port}")
        
        print(f"ZeroMQ telemetry publisher bound to tcp://*:{self.port}")

    def send_data(self, data_dict):
        # ZeroMQ automatically handles JSON serialization
        # If no subscribers are connected, ZMQ quietly drops the messages
        try:
            self.socket.send_json(data_dict)
        except Exception as e:
            print(f"Error sending data: {e}")

    def stop(self):
        # Clean up the socket and context
        self.socket.close()
        self.context.term()

class PoseReceiver:
    def __init__(self, host='127.0.0.1', port=5000, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.running = True
        
        # Initialize ZeroMQ context and SUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # Subscribe to all topics (empty string means accept everything)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Set a receive timeout (in milliseconds) so the background thread 
        # doesn't block indefinitely and can check the self.running flag
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        # ZMQ handles connection and automatic background reconnection
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"ZeroMQ subscriber connecting to tcp://{self.host}:{self.port}")
        
        # Start the listening thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # ZeroMQ receives the fully framed JSON payload and deserializes it
                robot_pose = self.socket.recv_json()
                if self.callback:
                    self.callback(robot_pose)
            except zmq.error.Again:
                # This exception is raised when the RCVTIMEO duration is reached.
                # It is expected behavior when no data is flowing.
                continue
            except Exception as e:
                if self.running:
                    print(f"ZeroMQ receiver error: {e}")
                    time.sleep(1)

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.socket.close()
        self.context.term()