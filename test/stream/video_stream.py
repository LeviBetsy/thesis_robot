import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import cv2

from app.robot_module.robot import Robot
from app.stream.zmq_stream import VideoStreamer
from app.util.config import Config
#********************************************** IMPORTS **********************************************

'''
Stream-only test: camera -> undistort -> VideoStreamer -> laptop. Nothing is received back.
Pair with laptop/mde_projection/main.py real_time_visualize() to view the point cloud.
'''

#Robot
config = Config.load()
robot = Robot(config)
camera = robot.camera

# ZeroMQ publisher for camera frames
streamer = VideoStreamer(port=config.stream.video_port)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

try:
    while True:
        # Free-running: cap.read() blocks until the next camera frame, so the camera paces this loop
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cant capture video frame")
        frame = camera.undistort_fisheye(frame=frame)
        streamer.send_frame(frame)
except KeyboardInterrupt:
    print("stopping")
except RuntimeError as e:
    print(f"Something went wrong, closing program: {e}")
finally:
    cap.release()
    streamer.stop()
