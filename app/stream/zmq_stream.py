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
    def __init__(self, port=5003):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ video streamer bound to tcp://*:{self.port}")

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
    Class for Streaming a numpy array of (N,3) particle poses: (x, y, theta) per row.

    Owns its own background thread (started in __init__, like the Receiver classes below),
    which periodically snapshots an MCLLocalization's particle cloud and publishes it, so
    a caller only needs to construct this once and let it run.

    estimate is an optional callable returning one (x, y, theta) pose, e.g. mcl.estimate_mean_pose
    or mcl.estimate_best_particle, sent alongside the cloud so the visualizer can highlight it.
    Sent as a two-part message [particles, estimate], the estimate part empty when there is none.
'''
class ParticleStreamer:
    def __init__(self, particle_set, fps=5, port=5004, estimate=None):
        self.particle_set = particle_set
        self.estimate = estimate
        self.fps = fps
        self.port = port
        self.running = True

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ particle streamer bound to tcp://*:{self.port}")

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            loop_start = time.time()

            # ParticleSet snapshots under its own lock, so a concurrent resample() on the
            # range receiver thread can't hand us a half-swapped set.
            positions = self.particle_set.poses()
            if positions.shape[0] > 0:
                estimate = None if self.estimate is None else self.estimate()
                self.send_point(positions, estimate)

            processing_time = time.time() - loop_start
            sleep_time = (1.0 / self.fps) - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def send_point(self, arr, estimate=None):
        try:
            # Ensure the array is a consistent data type and contiguous in memory
            # float32 is standard for coordinate data, change to float64 or int32 if needed
            arr_contiguous = np.ascontiguousarray(arr, dtype=np.float32)

            # Convert directly to raw bytes
            array_bytes = arr_contiguous.tobytes()
            estimate_bytes = b"" if estimate is None else np.array(estimate, dtype=np.float32).tobytes()

            # Send the two-part payload [particles, estimate]
            self.socket.send_multipart([array_bytes, estimate_bytes])

        except Exception as e:
            print(f"Error sending array: {e}")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.socket.close()
        self.context.term()

'''
    Class for Receiving a numpy array of (N,3) particle poses: (x, y, theta) per row.
'''
class ParticleReceiver:
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
                # 1. Receive the two-part payload [particles, estimate]
                array_bytes, estimate_bytes = self.socket.recv_multipart()

                # 2. Decode bytes back to float32, then reshape to (N, 3): x, y, theta
                # Using -1 lets NumPy automatically calculate N based on byte length
                arr = np.frombuffer(array_bytes, dtype=np.float32).reshape(-1, 3)
                estimate = np.frombuffer(estimate_bytes, dtype=np.float32) if estimate_bytes else None

                # 3. Pass the array and the (3,) estimate pose (None if not streamed) to the callback
                if self.callback:
                    self.callback(arr, estimate)

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
    Class for Streaming a numpy array of ray-cast ranges, shape (N,)
'''
class RangeStreamer:
    def __init__(self, port=5005):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ range streamer bound to tcp://*:{self.port}")

    def send_range(self, arr: np.ndarray):
        try:
            # Ensure the array is a consistent data type and contiguous in memory
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
    Class for Streaming one nav step's plan: the A* path, shape (M,2), and the lookahead aim, shape (2,),
    both world (x, y) meters.

    No thread: the plan only changes once per nav step, so the nav loop calls send_nav right after
    choosing the aim. Sent as a two-part message [aim, path] so the arrays never need a header to split.
'''
class NavStreamer:
    def __init__(self, port=5006):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ nav streamer bound to tcp://*:{self.port}")

    def send_nav(self, path, aim):
        try:
            aim_bytes = np.ascontiguousarray(aim, dtype=np.float32).tobytes()
            path_bytes = np.ascontiguousarray(path, dtype=np.float32).tobytes()
            self.socket.send_multipart([aim_bytes, path_bytes])

        except Exception as e:
            print(f"Error sending nav plan: {e}")

    def stop(self):
        self.socket.close()
        self.context.term()

'''
    Class for Streaming a nav goal, world (x, y) meters, shape (2,). Runs on the laptop.

    No thread: a goal only changes when one is picked in the map visualizer, which calls send_goal.
'''
class GoalStreamer:
    def __init__(self, port=5007):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 2)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"ZeroMQ goal streamer bound to tcp://*:{self.port}")

    def send_goal(self, x, y):
        try:
            self.socket.send(np.array([x, y], dtype=np.float32).tobytes())

        except Exception as e:
            print(f"Error sending goal: {e}")

    def stop(self):
        self.socket.close()
        self.context.term()

'''
    Class for Receiving a numpy array of ray-cast ranges, shape (N,)
'''
class RangeReceiver:
    def __init__(self, host='127.0.0.1', port=5005, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.running = True

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"ZeroMQ range subscriber connecting to tcp://{self.host}:{self.port}")

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # 1. Receive the raw byte payload
                array_bytes = self.socket.recv()

                # 2. Decode bytes back to float32, shape (N,)
                arr = np.frombuffer(array_bytes, dtype=np.float32)

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

'''
    Class for Receiving one nav step's plan: the A* path, shape (M,2), and the lookahead aim, shape (2,)
'''
class NavReceiver:
    def __init__(self, host='127.0.0.1', port=5006, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.running = True

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"ZeroMQ nav subscriber connecting to tcp://{self.host}:{self.port}")

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # 1. Receive the two-part payload [aim, path]
                aim_bytes, path_bytes = self.socket.recv_multipart()

                # 2. Decode bytes back to float32: aim (2,), path (M, 2)
                aim = np.frombuffer(aim_bytes, dtype=np.float32)
                path = np.frombuffer(path_bytes, dtype=np.float32).reshape(-1, 2)

                # 3. Pass the plan to the callback
                if self.callback:
                    self.callback(path, aim)

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
    Class for Receiving a nav goal, world (x, y) meters. Runs on the Pi.
'''
class GoalReceiver:
    def __init__(self, host='127.0.0.1', port=5007, callback=None):
        self.host = host
        self.port = port
        self.callback = callback
        self.running = True

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        self.socket.connect(f"tcp://{self.host}:{self.port}")
        print(f"ZeroMQ goal subscriber connecting to tcp://{self.host}:{self.port}")

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                # 1. Receive the raw byte payload, decode to float32 (2,): x, y
                x, y = np.frombuffer(self.socket.recv(), dtype=np.float32)

                # 2. Pass the goal to the callback
                if self.callback:
                    self.callback(float(x), float(y))

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

