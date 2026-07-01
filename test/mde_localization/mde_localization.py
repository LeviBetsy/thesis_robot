import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import zmq
import json
import time
import cv2

from app.module.robot import Robot
from app.module.uart import MSP432Uart
from app.control.keyboard_controller_ssh import RobotController
from app.mapping.localization import OdometryLocalization
# from app.stream.zmq_pose_stream import PoseStreamer
from app.stream.zmq_stream import PoseVideoStreamer
#********************************************** IMPORTS **********************************************


#UART
msp432_uart = MSP432Uart()
msp432_uart.start_receiving() #THREAD 1: to listen to odometry data from MSP432 and fill buffer

#Robot
robot = Robot("fisheye_calib.npz")

# #Localization
loc = OdometryLocalization(robot=robot)
loc.init_odometry_thread(msp432_uart) #THREAD 2: start thread to change localization using UART buffer

#Keyboard Controller
controller = RobotController(msp432_uart.send_command)
controller.start() #THREAD 3: start thread to listen for keyboard and sending command to msp432

# ZeroMQ publisher for camera stream process
# pose_streamer = PoseStreamer() #THREAD 4: thread to stream pose data
streamer = PoseVideoStreamer()
pose = {"x": 0, "y": 0, "theta": 0}
#Main loop
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)
# cap.set(cv2.CAP_PROP_FPS, streamer.fps)
try:
    while True:
        loop_start = time.time()

        with robot.mutex_lock:
            pose = {"x": robot.x, "y": robot.y, "theta": robot.theta}
        ret, frame = cap.read()
        if not ret:
            print("Can't capture video frame")
            raise RuntimeError("Cant capture video frame")
        streamer.send_data(pose, frame)

        # Making sure sending rate match intended fps
        processing_time = time.time() - loop_start
        sleep_time = (1.0 / streamer.fps) - processing_time
        if sleep_time > 0:
            time.sleep(sleep_time)
except KeyboardInterrupt:
    print("stopping")
except RuntimeError:
    print("Something went wrong, closing program")
finally:
    streamer.stop()
    msp432_uart.close()
    controller.stop()
