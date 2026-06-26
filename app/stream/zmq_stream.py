'''
ZMQ Stream for both pose and video using multipart stream
'''

import zmq
import threading
import time
import json
import cv2
import numpy as np

class PoseVideoStreamer:
    def __init__(self, port=5003, fps = 10):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ streamer bound to tcp://*:{self.port}")
        self.fps = fps

    def send_data(self, pose_dict, frame):
        try:
            # 1. Serialize the pose dictionary to a JSON byte string
            pose_bytes = json.dumps(pose_dict).encode('utf-8')
            
            # 2. Compress the frame to JPEG to save massive network bandwidth
            # 'encode_param' controls the quality (0-100). 90 is a good balance.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                print("Failed to encode frame")
                return
                
            # Convert the encoded image to raw bytes
            frame_bytes = encoded_image.tobytes()

            # 3. Send as a multipart message
            # The receiver will get exactly these two parts in order
            self.socket.send_multipart([pose_bytes, frame_bytes])
            
        except Exception as e:
            print(f"Error sending data: {e}")

    def stop(self):
        self.socket.close()
        self.context.term()


class PoseVideoReceiver:
    def __init__(self, host='127.0.0.1', port=5003, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.running = True
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"ZeroMQ subscriber connecting to tcp://{self.host}:{self.port}")
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # 1. Receive the multipart message
                parts = self.socket.recv_multipart()
                
                # Ensure we received exactly two parts (pose + frame)
                if len(parts) == 2:
                    pose_bytes, frame_bytes = parts
                    
                    # 2. Deserialize the JSON pose data
                    robot_pose = json.loads(pose_bytes.decode('utf-8'))
                    
                    # 3. Decode the JPEG bytes back into a standard OpenCV/NumPy array
                    # np.frombuffer reads the bytes into a 1D array, imdecode reconstructs the 2D/3D image
                    frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                    
                    # 4. Pass both pieces of synchronized data to the callback
                    if self.callback:
                        self.callback(robot_pose, frame)
                        
            except zmq.error.Again:
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