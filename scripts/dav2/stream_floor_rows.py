import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import cv2
import numpy as np

from app.robot_module.robot import Robot
from app.stream.zmq_stream import VideoStreamer
from app.util.config import Config
from scripts.dav2.annotate_floor_depth import annotate_depth
#********************************************** IMPORTS **********************************************

'''
Stream test: camera -> undistort -> draw the 2 closest floor-sample rows -> VideoStreamer -> laptop.
The npz grid is stored column-major: 6 columns x 16 depth steps (index = col*16 + step),
so the closest row is step 0 and the next row is step 1.
'''

NPZ_NAME = "z_real_fix_checker_size.npz"
N_COLS, N_STEPS = 6, 16
N_ROWS_KEEP = 2


def load_closest_rows(npz_name=NPZ_NAME, n_rows=N_ROWS_KEEP):
    data = np.load(os.path.join(root_path, "config", npz_name))
    corners = data["cornersOrg"].reshape(N_COLS, N_STEPS, 2)[:, :n_rows].reshape(-1, 2)
    z_real = data["z_real"].reshape(N_COLS, N_STEPS)[:, :n_rows].reshape(-1)
    return corners, z_real


if __name__ == "__main__":
    config = Config.load()
    robot = Robot(config)
    camera = robot.camera

    corners, z_real = load_closest_rows()
    print(f"Loaded {len(corners)} floor points, z = {sorted(set(np.round(z_real, 3)))}")

    streamer = VideoStreamer(port=config.stream.video_port)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Cant capture video frame")
            frame = camera.undistort_fisheye(frame=frame)
            frame = annotate_depth(frame, corners, z_real, font_scale=0.4)
            streamer.send_frame(frame)
    except KeyboardInterrupt:
        print("stopping")
    except RuntimeError as e:
        print(f"Something went wrong, closing program: {e}")
    finally:
        cap.release()
        streamer.stop()
