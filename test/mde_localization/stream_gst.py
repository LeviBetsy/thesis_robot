import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import sys
import time
import cv2
import json
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst
import zmq

from app.stream.video_stream import GIVideoStreamer
from app.module.camera import Camera

def main():
    # Interprocess communication for latest robot pose
    cam = Camera("fisheye_calib.npz")
    video_streamer = GIVideoStreamer(camera=cam) # Camera Steamer
    timestamp = 0

    cap = cv2.VideoCapture(0)
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            
            # Video Stream
            video_streamer.stream_frame(timestamp, frame, True)
            timestamp += video_streamer.duration
            
            # Sleep to match the target framerate (accounting for processing time)
            processing_time = time.time() - loop_start
            sleep_time = (1.0 / video_streamer.fps) - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        # Signal the end of the stream and clean up
        video_streamer.release()

if __name__ == '__main__':
    main()