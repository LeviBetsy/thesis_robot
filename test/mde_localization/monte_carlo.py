import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import zmq
import json
import time
import threading
import numpy as np
import cv2

from app.robot_module.robot import Robot
from app.robot_module.uart import MSP432Uart
from app.control.keyboard_controller_ssh import RobotController
from app.localization.odometry import OdometryLocalization
from app.localization.map import OccupancyGrid
from app.localization.mcl import MCLLocalization
from app.stream.zmq_stream import VideoStreamer, ParticleStreamer, RangeReceiver
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
streamer = VideoStreamer(fps=3) #THREAD 4: thread to stream pose data
#Main loop
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

# TODO: account for latency from returning laptop, might not be where we are, actually keep track of an accumulated motion for that

# Monte Carlo Localization against the known map.
grid = OccupancyGrid.from_json("config/map/map0.json", default_value=0)
mcl = MCLLocalization(robot=robot, grid=grid, n_particles=500)
# Robot always powers on at (0,0,0) (see Robot.__init__), so this assumes the operator
# physically places the robot at the map's origin corner before start. If the start pose
# isn't known, swap for mcl.init_particles_uniform() to run global localization instead.
mcl.init_particles_gaussian(0.0, 0.0, 0.0, sigma_xy=0.05, sigma_theta=0.1)

ray_receiver = RangeReceiver(callback=mcl.on_scan) #Thread 5: receiver to run one MCL measurement update every time MDE outputs a scan

# ZeroMQ publisher for the particle cloud, so the laptop can visualize where MCL thinks the robot is
PARTICLE_STREAM_FPS = 5
particle_streamer = ParticleStreamer(fps=PARTICLE_STREAM_FPS)

def particle_stream_loop(): #THREAD 6: thread to stream the particle cloud
    while True:
        loop_start = time.time()

        # mcl.particles is reassigned wholesale by init_particles_*/resample rather than
        # mutated in place, so lock just long enough to snapshot the list reference.
        with mcl.mutex_lock:
            particles = mcl.particles
        positions = np.array([[p.x, p.y, p.theta] for p in particles], dtype=np.float32)
        particle_streamer.send_point(positions)

        processing_time = time.time() - loop_start
        sleep_time = (1.0 / particle_streamer.fps) - processing_time
        if sleep_time > 0:
            time.sleep(sleep_time)

particle_stream_thread = threading.Thread(target=particle_stream_loop, daemon=True)
particle_stream_thread.start()

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
