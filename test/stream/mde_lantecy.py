import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import threading

import cv2
import numpy as np

from app.robot_module.robot import Robot
from app.stream.zmq_stream import VideoStreamer, RangeReceiver
from app.util.config import Config

'''
Runs on the Pi. Streams one camera frame to the laptop (laptop/mde_projection/main.py
real_time_stream) at a time and waits for its matching MDE range scan before sending the
next, measuring the round-trip latency per frame.

This is closed-loop (ping-style) on purpose: if frames were sent at a fixed fps regardless
of how fast the laptop can process them, and the laptop is slower than that fps, sends
would pile up faster than scans drain them, and a simple FIFO match against send timestamps
would report a latency that keeps growing with the backlog instead of the real round trip.

Run on the Pi:
    python test/stream/mde_lantecy.py
'''

CONFIG_PATH = "./config/robot_config.json"

config = Config.load(CONFIG_PATH)
robot = Robot(config)
camera = robot.camera
n_rays = config.ray_cast.n_rays

#Set by callback_new_range to signal the main loop that the scan for the frame currently
#in flight has arrived. sent_at / latest_latency are only ever touched by one thread at a
#time since the main loop waits for the event before sending the next frame.
scan_event = threading.Event()
sent_at = None
latest_latency = None

latencies = []


def callback_new_range(z_mde):
    global latest_latency
    if z_mde.shape[0] != n_rays:
        print(f"laptop sent {z_mde.shape[0]} rays, config.ray_cast.n_rays is {n_rays}, skipping")
        return

    latest_latency = time.time() - sent_at
    scan_event.set()


if __name__ == "__main__":
    streamer = VideoStreamer(fps=config.stream.video_fps, port=config.stream.video_port)
    range_receiver = RangeReceiver(host=config.stream.laptop_host, port=config.stream.range_port,
                                   callback=callback_new_range)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.h)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Can't capture video frame")

            scan_event.clear()
            sent_at = time.time()
            streamer.send_frame(camera.undistort_fisheye(frame=frame))

            if scan_event.wait(timeout=5.0):
                latencies.append(latest_latency)
                print(f"[scan {len(latencies)}] latency: {latest_latency * 1000:.1f} ms")
            else:
                print("timed out waiting for scan, skipping")
    except KeyboardInterrupt:
        print("\nstopping")
    except RuntimeError as e:
        print(f"{e}, closing program")
    finally:
        cap.release()
        range_receiver.stop()
        streamer.stop()

        if latencies:
            arr = np.array(latencies)
            print(f"\nsummary over {arr.size} scans:")
            print(f"  mean: {arr.mean() * 1000:.1f} ms")
            print(f"  std : {arr.std() * 1000:.1f} ms")
            print(f"  min : {arr.min() * 1000:.1f} ms")
            print(f"  max : {arr.max() * 1000:.1f} ms")
        else:
            print("no MDE scans received, is laptop/mde_projection/main.py running?")
