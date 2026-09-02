'''
ZMQ Stream for both pose and video using multipart stream
'''

import zmq
import threading
import time
import json
import cv2
import numpy as np

class VideoStreamer:
    def __init__(self, fps, port=5003):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ video streamer bound to tcp://*:{self.port}")
        self.fps = fps

    def send_frame(self, frame):
        try:
            # Compress the frame to JPEG to save network bandwidth
            # 'encode_param' controls the quality (0-100). 90 is a good balance.
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                print("Failed to encode frame")
                return
                
            # Convert the encoded image to raw bytes
            frame_bytes = encoded_image.tobytes()

            # Send the frame as a single message
            self.socket.send(frame_bytes)
            
        except Exception as e:
            print(f"Error sending frame: {e}")

    def stop(self):
        self.socket.close()
        self.context.term()


class VideoReceiver:
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
        print(f"ZeroMQ video subscriber connecting to tcp://{self.host}:{self.port}")
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # 1. Receive the message payload
                frame_bytes = self.socket.recv()
                
                # 2. Decode the JPEG bytes back into a standard OpenCV/NumPy array
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                
                # 3. Pass the video frame to the callback
                if self.callback:
                    self.callback(frame)
                        
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

'''
    Class for Streaming a numpy array of (N,2) points. The Z dimension must be squished
'''
class PointStreamer:
    def __init__(self, fps, port=5004):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ array streamer bound to tcp://*:{self.port}")
        self.fps = fps

    def send_point(self, arr):
        try:
            # Ensure the array is a consistent data type and contiguous in memory
            # float32 is standard for coordinate data, change to float64 or int32 if needed
            arr_contiguous = np.ascontiguousarray(arr, dtype=np.float32)
            
            # Convert directly to raw bytes
            array_bytes = arr_contiguous.tobytes()

            # Send the byte payload
            self.socket.send(array_bytes)
            
        except Exception as e:
            print(f"Error sending array: {e}")

    def stop(self):
        self.socket.close()
        self.context.term()

'''
    Class for Receiving a numpy array of (N,2) points. The Z dimension is squished
'''
class PointReceiver:
    def __init__(self, host='127.0.0.1', port=5004, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.running = True
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"ZeroMQ array subscriber connecting to tcp://{self.host}:{self.port}")
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # 1. Receive the raw byte payload
                array_bytes = self.socket.recv()
                
                # 2. Decode bytes back to float32, then reshape to (N, 2)
                # Using -1 lets NumPy automatically calculate N based on byte length
                arr = np.frombuffer(array_bytes, dtype=np.float32).reshape(-1, 2)
                
                # 3. Pass the array to the callback
                if self.callback:
                    self.callback(arr)
                        
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
