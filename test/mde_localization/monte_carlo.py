import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import math
import time
import cv2

from app.robot_module.robot import Robot
from app.robot_module.uart import MSP432Uart
from app.control.keyboard_controller_ssh import RobotController
from app.localization.map import OccupancyGrid
from app.localization.mcl import MCLLocalization
from app.stream.zmq_stream import VideoStreamer, ParticleStreamer, RangeReceiver
#********************************************** IMPORTS **********************************************


#UART
msp432_uart = MSP432Uart()
msp432_uart.start_receiving() #THREAD 1: to listen to odometry data from MSP432 and fill buffer

#Robot
robot = Robot("fisheye_calib.npz")

# #Keyboard Controller
# controller = RobotController(msp432_uart.send_command)
# controller.start() #THREAD 2: start thread to listen for keyboard and sending command to msp432

# ZeroMQ publisher for camera stream process
streamer = VideoStreamer(fps=3) #THREAD 3: thread to stream video
#Main loop
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

# TODO: account for latency from returning laptop, might not be where we are, actually keep track of an accumulated motion for that

# Monte Carlo Localization against the known map.
grid = OccupancyGrid.from_json("config/map/map0.json", default_value=0)
mcl = MCLLocalization(robot=robot, grid=grid, n_particles=100)

mcl.init_particles_gaussian(0.535, 0.89, math.pi/2, sigma_xy=0.1, sigma_theta=0.5)

ray_receiver = RangeReceiver(callback=mcl.on_scan) #Thread 5: receiver to run one MCL measurement update every time MDE outputs a scan

particle_streamer = ParticleStreamer(mcl, fps=3) #Thread 6: streams the particle cloud, owns its own thread

try:
    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        frame = camera.undistort_fisheye(frame=frame)
        if not ret:
            print("Can't capture video frame")
            raise RuntimeError("Cant capture video frame")
        streamer.send_frame(frame)
        # Making sure sending rate match intended fps
        processing_time = time.time() - loop_start
        sleep_time = (1.0 / streamer.fps) - processing_time
        #TODO: set streamer.fps so they only send over frame once it finishes processing for max latency
        if sleep_time > 0:
            time.sleep(sleep_time)
except KeyboardInterrupt:
    print("stopping")
except RuntimeError:
    print("Something went wrong, closing program")
finally:
    streamer.stop()
    particle_streamer.stop()
    ray_receiver.stop()
    # msp432_uart.close()
    # controller.stop()
