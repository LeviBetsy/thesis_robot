''' Verified not tested'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import math
import threading
import time

import cv2
import numpy as np

from app.localization.map import OccupancyGrid
from app.localization.ray_caster import RayCaster
from app.localization.visualize_map import MapVisualizer
from app.robot_module.robot import Robot
from app.stream.zmq_stream import VideoStreamer, RangeReceiver
from app.util.config import Config

'''
Runs on the Pi with the robot standing still at a KNOWN pose on the map. Streams the camera
to the laptop (laptop/mde_projection/main.py real_time_stream), receives the n_rays MDE
ranges back, and compares them beam-by-beam with what RayCaster predicts for a particle
placed at that same pose.

If the two agree on the contract (bin layout, left/right sign, ranges from the camera) the
rows should match to within MDE noise. A constant offset points at scale calibration; a
mirrored pattern points at a flipped bearing sign.

Run on the Pi:
    python test/localization/raycast_vs_mde.py
    python test/localization/raycast_vs_mde.py --pose 0.5 0.5 90
'''

CONFIG_PATH = "./config/robot_config.json"
MAP_PATH = os.path.join(os.path.dirname(__file__), "../../config/map/map1.json")

# Where the robot is physically placed for the test: meters / degrees CCW from +x.
# Arbitrary for now; measure the real placement and pass it with --pose.
TEST_POSE = (0.60, 0.40, 90.0)


def print_comparison(z_pred, z_mde, no_return):
    """
    One scan as aligned rows, index 0 is the leftmost ray.

    Args:
        z_pred (np.ndarray): shape (n_rays,), RayCaster ranges from the camera, meters.
        z_mde (np.ndarray): shape (n_rays,), MDE ranges from the camera, meters.
        no_return (float): a range at or above this is a no-return (empty bin / nothing hit).
    """
    #only compare beams where BOTH sides hit something, a no-return has no real distance
    valid = (z_pred < no_return) & (z_mde < no_return) #shape (n_rays,)
    err = z_mde - z_pred

    fmt = lambda arr: " ".join(f"{v:6.3f}" for v in arr)
    print("  ray : " + " ".join(f"{k:6d}" for k in range(z_pred.shape[0])))
    print("  pred: " + fmt(z_pred))
    print("  mde : " + fmt(z_mde))
    print("  diff: " + " ".join(f"{d:+6.3f}" if ok else "     -" for d, ok in zip(err, valid)))

    if valid.any():
        e = err[valid]
        print(f"  {valid.sum()}/{valid.size} beams both hit | bias {e.mean():+.3f}  "
              f"mae {np.abs(e).mean():.3f}  rmse {np.sqrt((e ** 2).mean()):.3f} m")
    else:
        print("  no beam where both sides hit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", type=float, nargs=3, default=TEST_POSE, metavar=("X", "Y", "THETA_DEG"),
                        help="where the robot is standing, meters / degrees CCW from +x")
    parser.add_argument("--map", default=MAP_PATH)
    args = parser.parse_args()

    config = Config.load(CONFIG_PATH)
    robot = Robot(config)
    camera = robot.camera
    n_rays = config.ray_cast.n_rays
    no_return = config.ray_cast.max_range - config.measurement_model.max_eps

    #Map and the particle's predicted scan, fixed for the whole run since the robot doesn't move
    grid = OccupancyGrid.from_json(args.map, config, default_value=0)
    caster = RayCaster(grid, camera, config)
    pose = np.array([[args.pose[0], args.pose[1], math.radians(args.pose[2])]]) #shape (1, 3)

    z_pred = caster.predict(pose)[0]        #shape (n_rays,), meters from the camera
    origins, headings = caster.rays(pose)   #shape (1, n_rays, 2) and (1, n_rays)
    origins, headings = origins[0], headings[0]

    print(f"particle x={pose[0, 0]:.3f} y={pose[0, 1]:.3f} theta={args.pose[2]:.1f}deg")
    print("predicted:", np.round(z_pred, 3))

    #Every MDE scan the laptop sends back, kept so the run can be summarised at the end
    scans = []
    scans_lock = threading.Lock()

    def callback_new_range(z_mde):
        if z_mde.shape[0] != n_rays:
            print(f"laptop sent {z_mde.shape[0]} rays, config.ray_cast.n_rays is {n_rays}, skipping")
            return
        z_mde = z_mde.copy() #frombuffer arrays are read-only and share the zmq buffer
        with scans_lock:
            scans.append(z_mde)
            i = len(scans)
        print(f"\n[scan {i}]")
        print_comparison(z_pred, z_mde, no_return)

    streamer = VideoStreamer(port=config.stream.video_port)
    range_receiver = RangeReceiver(host=config.stream.laptop_host, port=config.stream.range_port,
                                   callback=callback_new_range)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

    try:
        while True:
            # Free-running: cap.read() blocks until the next camera frame, so the camera paces this loop
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Can't capture video frame")
            streamer.send_frame(camera.undistort_fisheye(frame=frame))
    except KeyboardInterrupt:
        print("\nstopping")
    except RuntimeError as e:
        print(f"{e}, closing program")
    finally:
        cap.release()
        range_receiver.stop()
        streamer.stop()

        with scans_lock:
            if scans:
                stack = np.stack(scans) #shape (n_scans, n_rays)
                print(f"\nsummary over {stack.shape[0]} scans (mean mde vs pred):")
                print_comparison(z_pred, stack.mean(axis=0), no_return)
                print("  std : " + " ".join(f"{v:6.3f}" for v in stack.std(axis=0)))
            else:
                print("no MDE scans received, is laptop/mde_projection/main.py running?")
