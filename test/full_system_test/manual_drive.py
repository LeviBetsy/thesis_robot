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
from app.robot_module.odometry import WheelOdometry
from app.control.keyboard_controller_ssh import RobotController
from app.localization.map import OccupancyGrid
from app.localization.mcl import MCLLocalization
from app.stream.zmq_stream import VideoStreamer, ParticleStreamer, RangeReceiver
from app.util.config import Config
#********************************************** IMPORTS **********************************************

'''
Full system test: drive the real robot by keyboard while MCL tracks it on the known map.

Pi (this file):
    camera -> undistort -> VideoStreamer          -> laptop (MDE)
    laptop range fan -> RangeReceiver -> mcl.on_scan (motion from WheelOdometry, then measurement, resample)
    mcl particles -> ParticleStreamer             -> laptop (visualizer)
    laptop keyboard (TCP over ssh tunnel) -> RobotController -> UART -> MSP432

Laptop, in separate terminals:
    python laptop/mde_projection/main.py                 # frames in, range fan out
    python laptop/mde_projection/visualize_map_main.py   # particle cloud on the map
    python laptop/keyboard_control/keyboard_stream.py    # arrow keys -> left_cps,right_cps
'''

MAP_PATH = "config/map/real_map0.json"

# The test scenario, not a tunable: where the robot is physically placed on map0 at start,
# meters / radians CCW from +x. config.mcl.init_pose is a placeholder inside the perimeter wall.
START_POSE = (0.395, 0.555, math.pi / 2)

POSE_PRINT_PERIOD = 1.0 # seconds between pose estimate printouts

#Robot
config = Config.load()
robot = Robot(config)

#UART
msp432_uart = MSP432Uart(port=config.uart.port, baudrate=config.uart.baudrate, timeout=config.uart.timeout)
msp432_uart.start_receiving() #THREAD 1: listen to cumulative encoder counts from MSP432 and fill data_queue

#Keyboard Controller
controller = RobotController(msp432_uart.send_command)
controller.start() #THREAD 2: listen for left_cps,right_cps from the laptop and forward to msp432

# Monte Carlo Localization against the known map, motion stage fed by the wheel encoders
odometry = WheelOdometry(robot)
grid = OccupancyGrid.from_json(MAP_PATH, config, default_value=0)
mcl = MCLLocalization(robot=robot, grid=grid, config=config, odometry=odometry)
mcl.init_particles_gaussian(*START_POSE,
                            sigma_x=config.mcl.init_sigma_x,
                            sigma_y=config.mcl.init_sigma_y,
                            sigma_theta=config.mcl.init_sigma_theta,
                            max_attempts=config.mcl.init_max_attempts)

# Started after seeding so motion from before the cloud existed is never applied to it
odometry.init_odometry_thread(msp432_uart) #THREAD 3: fold encoder deltas into the motion increment

# ZeroMQ streams, all hosts / ports from config.stream
streamer = VideoStreamer(port=config.stream.video_port) #frames to the laptop's MDE, sent from the main loop
ray_receiver = RangeReceiver(host=config.stream.laptop_host, port=config.stream.range_port,
                             callback=mcl.on_scan) #THREAD 4: one MCL step per range fan from the laptop
particle_streamer = ParticleStreamer(mcl, fps=config.stream.particle_fps,
                                     port=config.stream.particle_port) #THREAD 5: particle cloud to the laptop

#Camera
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

# TODO: account for latency from returning laptop, the scan describes where the robot was at capture (see mcl.on_scan)

if __name__ == "__main__":
    print("Manual drive MCL test running. Drive with keyboard_stream.py on the laptop. Ctrl+C to stop.")
    last_print = time.monotonic()
    try:
        while True:
            # Free-running: cap.read() blocks until the next camera frame, so the camera paces this loop
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cant capture video frame")
            frame = camera.undistort_fisheye(frame=frame)
            streamer.send_frame(frame)

            now = time.monotonic()
            if now - last_print >= POSE_PRINT_PERIOD:
                x, y, theta = mcl.estimate_mean_pose()
                print(f"MCL mean pose: x={x:.3f} m  y={y:.3f} m  theta={math.degrees(theta):.1f} deg")
                last_print = now

    except KeyboardInterrupt:
        print("\nShutting down...")
    except RuntimeError as e:
        print(f"Something went wrong, closing program: {e}")
    finally:
        msp432_uart.send_command(0, 0) # Safety stop
        controller.stop()
        odometry.stop()
        ray_receiver.stop()
        particle_streamer.stop()
        streamer.stop()
        cap.release()
        msp432_uart.close()
