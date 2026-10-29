import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import math
import time
import threading
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
    laptop arrow keys -> ssh tunnel (port 8080) -> RobotController -> UART -> MSP432

Laptop, in separate terminals:
    python laptop/mde_projection/main.py                 # frames in, range fan out
    python laptop/mde_projection/visualize_map_main.py   # particle cloud on the map
    python laptop/keyboard_control/keyboard_stream.py    # arrow keys out, over the ssh tunnel
'''

MAP_PATH = "config/map/real_map0.json"

# The test scenario, not a tunable: where the robot is physically placed on map0 at start,
# meters / radians CCW from +x. config.mcl.init_pose is a placeholder inside the perimeter wall.
START_POSE = (0.395, 0.555, math.pi / 2)

#Robot
config = Config.load()
robot = Robot(config)

#UART
msp432_uart = MSP432Uart(port=config.uart.port, baudrate=config.uart.baudrate, timeout=config.uart.timeout)
msp432_uart.start_receiving() #THREAD 1: listen to cumulative encoder counts from MSP432 and fill data_queue

#Keyboard Controller
controller = RobotController(msp432_uart.send_command)
controller.start() #THREAD 2: listen for left_cps,right_cps from the laptop over the ssh tunnel and forward to msp432

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

# Lockstep with the laptop: one frame in flight at a time, so the range fan that comes back
# is known to describe THAT frame. The odometry consumed at its capture is exactly the motion
# up to the moment z describes; motion while it is at the laptop accumulates for the next frame.
in_flight = [None] #odometry consumed at the in-flight frame's capture, None when nothing is in flight
in_flight_lock = threading.Lock() #held across the whole filter step, so a timeout can't apply the same motion twice

def on_range(z):
    with in_flight_lock:
        if in_flight[0] is None:
            return #its frame already timed out and the motion went in without it
        mcl.on_scan(z, motion=in_flight[0])
        in_flight[0] = None

# ZeroMQ streams, all hosts / ports from config.stream
streamer = VideoStreamer(port=config.stream.video_port) #frames to the laptop's MDE, sent from the main loop
ray_receiver = RangeReceiver(host=config.stream.laptop_host, port=config.stream.range_port,
                             callback=on_range) #THREAD 4: one MCL step per range fan from the laptop
particle_streamer = ParticleStreamer(mcl.particle_set, fps=config.stream.particle_fps,
                                     port=config.stream.particle_port) #THREAD 5: particle cloud to the laptop

#Camera
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

if __name__ == "__main__":
    print("Manual drive MCL test running. Drive with keyboard_stream.py on the laptop. Ctrl+C to stop.")
    last_print = time.monotonic()
    sent_at = 0.0
    try:
        while True:
            # Read every frame even while one is in flight: left undrained, the V4L2 buffer
            # fills and the next read hands back a frame from before the wait
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cant capture video frame")

            with in_flight_lock:
                if in_flight[0] is not None:
                    if time.monotonic() - sent_at < config.stream.scan_timeout:
                        continue #previous frame still at the laptop, drop this one
                    # PUB/SUB drops silently (laptop not up yet, lost message). The motion
                    # still happened, so push the cloud through it without a measurement.
                    print("No range fan back in time, sending a fresh frame")
                    mcl.motion_update(*in_flight[0])
                in_flight[0] = odometry.consume() #the capture moment this frame's z will describe
            sent_at = time.monotonic()

            frame = camera.undistort_fisheye(frame=frame)
            streamer.send_frame(frame)


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
