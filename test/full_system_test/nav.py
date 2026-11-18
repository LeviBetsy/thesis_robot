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
from app.localization.map import OccupancyGrid
from app.localization.mcl import MCLLocalization
from app.navigation.astar import astar, lookahead_point
from app.navigation.step_controller import StepController
from app.stream.zmq_stream import VideoStreamer, ParticleStreamer, RangeReceiver, NavStreamer, GoalReceiver
from app.util.config import Config
#********************************************** IMPORTS **********************************************

'''
Full system test: drive the real robot on its own to goals clicked on the laptop's map while MCL tracks it.

Pi (this file):
    camera -> undistort -> VideoStreamer          -> laptop (MDE)
    laptop range fan -> RangeReceiver -> mcl.on_scan (motion from WheelOdometry, then measurement, resample)
    mcl particles -> ParticleStreamer             -> laptop (visualizer)
    laptop picked cell -> GoalReceiver -> goal[0]
    nav thread: mean pose -> A* -> lookahead point -> StepController turn + drive -> UART -> MSP432

Each nav step: wait for one range fan captured after the last stop, read the mean pose,
replan to the current goal, turn toward the lookahead point, drive up to step_dist, stop.
Within goal_tol (or no path), idle until a new goal is picked. The goal starts at START_POSE,
so the robot holds still until the first pick.

Laptop, in separate terminals:
    python laptop/mde_projection/main.py                 # frames in, range fan out
    python laptop/mde_projection/visualize_map_main.py   # particle cloud on the map, click a cell to send a goal
'''

MAP_PATH = "config/map/real_map0.json"

# The test scenario, not a tunable: where the robot is physically placed at start,
# (x, y, theta) meters / radians CCW from +x. Goals come from the laptop's map visualizer.
START_POSE = (0.395, 0.555, math.pi / 2)

# Current goal (x, y) meters, replaced whole by the goal receiver thread. Seeded with the start
# so the robot sits "at its goal" until one is picked, no separate no-goal case needed.
goal = [START_POSE[:2]]

#Robot
config = Config.load()
robot = Robot(config)

#UART
msp432_uart = MSP432Uart(port=config.uart.port, baudrate=config.uart.baudrate, timeout=config.uart.timeout)
msp432_uart.start_receiving() #THREAD 1: listen to cumulative encoder counts from MSP432 and fill data_queue

# Monte Carlo Localization against the known map, motion stage fed by the wheel encoders
odometry = WheelOdometry(robot)
grid = OccupancyGrid.from_json(MAP_PATH, config, default_value=0)
mcl = MCLLocalization(robot=robot, grid=grid, config=config, odometry=odometry)

# The pose nav steers from, also streamed to the visualizer so the highlighted pose is the one in use.
# Swap to mcl.estimate_best_particle to test the best particle instead of the weighted mean.
POSE_ESTIMATE = mcl.estimate_mean_pose

mcl.init_particles_gaussian(*START_POSE,
                            sigma_x=config.mcl.init_sigma_x,
                            sigma_y=config.mcl.init_sigma_y,
                            sigma_theta=config.mcl.init_sigma_theta,
                            max_attempts=config.mcl.init_max_attempts)

# Started after seeding so motion from before the cloud existed is never applied to it
odometry.init_odometry_thread(msp432_uart) #THREAD 2: fold encoder deltas into both motion accumulators

# Turns and drives closed on odometry.nav_odometry, the nav thread's own accumulator
step_controller = StepController(msp432_uart, odometry, config)

# Lockstep with the laptop, same as manual_drive: one frame in flight at a time, so the range
# fan that comes back is known to describe THAT frame.
in_flight = [None] #(odometry consumed at capture, capture time) of the in-flight frame, None when nothing is in flight
in_flight_lock = threading.Lock() #held across the whole filter step, so a timeout can't apply the same motion twice
last_fan_capture = [0.0] #capture time of the frame behind the last fan MCL processed, read by the nav thread

def on_range(z):
    with in_flight_lock:
        if in_flight[0] is None:
            return #its frame already timed out and the motion went in without it
        motion, captured_at = in_flight[0]
        mcl.on_scan(z, motion=motion)
        last_fan_capture[0] = captured_at
        in_flight[0] = None

# ZeroMQ streams, all hosts / ports from config.stream
streamer = VideoStreamer(port=config.stream.video_port) #frames to the laptop's MDE, sent from the main loop
ray_receiver = RangeReceiver(host=config.stream.laptop_host, port=config.stream.range_port,
                             callback=on_range) #THREAD 3: one MCL step per range fan from the laptop
particle_streamer = ParticleStreamer(mcl.particle_set, fps=config.stream.particle_fps,
                                     port=config.stream.particle_port,
                                     estimate=POSE_ESTIMATE) #THREAD 4: particle cloud + pose estimate to the laptop
nav_streamer = NavStreamer(port=config.stream.nav_port) #A* path + aim to the laptop, sent from the nav thread

def on_goal(x, y):
    goal[0] = (x, y) #one tuple swap, so the nav thread never reads a half-updated goal
    print(f"New goal ({x:.3f}, {y:.3f})")

goal_receiver = GoalReceiver(host=config.stream.laptop_host, port=config.stream.goal_port,
                             callback=on_goal) #THREAD 5: goals picked in the laptop's map visualizer

#Camera
camera = robot.camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

FAN_POLL = 0.01 #seconds between checks for a fresh fan
GOAL_POLL = 0.1 #seconds between checks for a new goal while idle

def wait_for_fan_after(t):
    """
    Blocks until MCL has processed a range fan whose frame was captured after t.

    Until then the in-flight frame's motion and the motion since its capture have not
    reached the cloud, so the mean pose lags the robot by up to a round trip. The first
    fan captured after a stop applies all of it, plus a measurement taken standing still.
    """
    while last_fan_capture[0] <= t:
        time.sleep(FAN_POLL)

nav_done = threading.Event() #set by the nav thread when it finishes, the main loop exits on it

def nav_loop():
    """
    THREAD 6: stop-turn-go toward goal[0], replanned every step so a goal picked mid-run takes
    over at the next step. Once the goal is reached or unreachable, idles until a different one arrives.
    """
    nav = config.nav
    stopped_at = time.monotonic()
    idle_for = None #goal already reached or found unreachable, skipped until goal[0] changes
    try:
        while True:
            goal_xy = goal[0] #read once, the receiver may swap it mid-step
            if goal_xy == idle_for:
                time.sleep(GOAL_POLL)
                continue

            wait_for_fan_after(stopped_at)
            x, y, theta = POSE_ESTIMATE()
            if math.hypot(goal_xy[0] - x, goal_xy[1] - y) <= nav.goal_tol:
                print(f"Goal reached at x={x:.3f} y={y:.3f}, waiting for a new goal")
                idle_for = goal_xy
                continue

            path = astar(grid, (x, y), goal_xy)
            if path is None:
                print(f"No path from x={x:.3f} y={y:.3f} to ({goal_xy[0]:.3f}, {goal_xy[1]:.3f}), waiting for a new goal")
                idle_for = goal_xy
                continue
            aim = lookahead_point(grid, path, (x, y), nav.lookahead)
            nav_streamer.send_nav(path, aim)

            dx, dy = aim[0] - x, aim[1] - y
            dtheta = math.atan2(dy, dx) - theta
            dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta)) #short way round, so heading_tol compares |turn| <= pi
            if abs(dtheta) > nav.heading_tol:
                step_controller.rotate_by(dtheta)
            step_controller.drive_by(min(nav.step_dist, math.hypot(dx, dy)))
            stopped_at = time.monotonic()
            print(f"Step: pose x={x:.3f} y={y:.3f} th={math.degrees(theta):.1f}  "
                  f"aim ({aim[0]:.3f}, {aim[1]:.3f})  turned {math.degrees(dtheta):+.1f} deg")
    finally:
        step_controller.stop()
        nav_done.set()


if __name__ == "__main__":
    print(f"Autonomous nav test from {START_POSE}. Pick goals in the map visualizer. Ctrl+C to stop.")
    threading.Thread(target=nav_loop, daemon=True).start()
    sent_at = 0.0
    try:
        while not nav_done.is_set():
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
                    mcl.motion_update(*in_flight[0][0])
                in_flight[0] = (odometry.consume(), time.monotonic()) #the capture moment this frame's z will describe
            sent_at = time.monotonic()

            frame = camera.undistort_fisheye(frame=frame)
            streamer.send_frame(frame)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except RuntimeError as e:
        print(f"Something went wrong, closing program: {e}")
    finally:
        msp432_uart.send_command(0, 0) # Safety stop
        odometry.stop()
        ray_receiver.stop()
        particle_streamer.stop()
        nav_streamer.stop()
        goal_receiver.stop()
        streamer.stop()
        cap.release()
        msp432_uart.close()
