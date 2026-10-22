import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import zmq
import json
import time
import cv2

from app.robot_module.robot import Robot
from app.robot_module.uart import MSP432Uart
from app.control.keyboard_controller_ssh import RobotController
from app.robot_module.odometry import OdometryLocalization
from app.stream.zmq_stream import VideoStreamer, RangeReceiver
#********************************************** IMPORTS **********************************************


#UART
msp432_uart = MSP432Uart()
msp432_uart.start_receiving() #THREAD 1: to listen to odometry data from MSP432 and fill buffer

#Robot
robot = Robot("fisheye_calib.npz")
camera_module = robot.camera

# #Localization
loc = OdometryLocalization(robot=robot)
loc.init_odometry_thread(msp432_uart) #THREAD 2: start thread to change localization using UART buffer

# #Keyboard Controller
# controller = RobotController(msp432_uart.send_command)
# controller.start() #THREAD 3: start thread to listen for keyboard and sending command to msp432

# ZeroMQ publisher for camera stream process
streamer = VideoStreamer() #THREAD 4: thread to stream pose data
#Main loop
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

# TODO: account for latency from returning laptop, might not be where we are, actually keep track of an accumulated motion for that

def callback_new_pcd(pcd):
    pass

ray_receiver = RangeReceiver(callback=callback_new_pcd) #Thread 5: receiver to receive pcd data everytime MDE output

try:
    while True:
        # Free-running: cap.read() blocks until the next camera frame, so the camera paces this loop
        ret, frame = cap.read()
        if not ret:
            print("Can't capture video frame")
            raise RuntimeError("Cant capture video frame")
        frame = camera.undistort_fisheye(frame=frame)
        streamer.send_frame(frame)
except KeyboardInterrupt:
    print("stopping")
except RuntimeError:
    print("Something went wrong, closing program")
finally:
    streamer.stop()
    # msp432_uart.close()
    # controller.stop()
