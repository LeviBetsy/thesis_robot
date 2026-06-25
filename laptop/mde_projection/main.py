import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
import queue
import cv2
import time
import open3d as o3d
import numpy as np
# Import GStreamer bindings
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

from scripts.mde.DAV2_pth import DepthAnythingPredictor
from laptop.mde_projection.scale_calibration_floor import FloorScaleCorrection
from app.module.robot import Robot
from app.mapping.point_cloud import PointCloudProcessor
from app.stream.pose_stream import PoseReceiver
from app.stream.video_stream import GIVideoReceiver

# **********************************************************************

# ***************************** Initialize *****************************

# Initialize GStreamer
Gst.init(None)
display_queue = queue.Queue(maxsize=10) # Queue to pass synchronized frames to the main thread for calculation
# Dictionaries to hold arriving buffers until their matching partner arrives
video_cache = {}
pose_cache = {}

# Robot
robot = Robot("fisheye_calib.npz")
camera = robot.camera
w, h = camera.w, camera.h

# **********************************************************************

# # **************************** Callbacks *********************

def queue_sync(pts):
    """Checks if we have BOTH the video and data for a given timestamp.
        And then push it to a queue for process """
    pass
    # if pts in video_cache and pts in data_cache:
    #     # We have a match! Extract them and remove from caches.
    #     frame = video_cache.pop(pts)
    #     state = data_cache.pop(pts)
        
    #     # Clean up old orphaned frames in case packets were dropped over the network
    #     for old_pts in list(video_cache.keys()):
    #         if old_pts < pts: del video_cache[old_pts]
    #     for old_pts in list(data_cache.keys()):
    #         if old_pts < pts: del data_cache[old_pts]

    #     # Push the synchronized pair to the main thread
    #     if not display_queue.full():
    #         display_queue.put((frame, state))

def callback_new_video(frame, pts):
    video_cache[pts] = frame
    queue_sync(pts)
    cv2.imshow("Robot Stream", frame)

def callback_new_pose(pose):
    """Callback triggered when a new telemetry JSON string arrives."""
    pts = pose["pts"]
    pose_cache[pts] = pose
    queue_sync(pts)

# Receivers
pose_receiver = PoseReceiver(callback=callback_new_pose)
video_receiver = GIVideoReceiver(callback=callback_new_video, camera=camera)
# **********************************************************************

def main():
    try:
        while True:
            # Block and wait for a synchronized pair from the GStreamer callbacks
            frame, state = display_queue.get()
            
            # Extract your synchronized variables
            robot_x = state.get("x", 0.0)
            robot_y = state.get("y", 0.0)
            robot_theta = state.get("theta", 0.0)
            
            # Overlay the data onto the synced frame
            text = f"Synced -> X: {robot_x:.2f} | Y: {robot_y:.2f} | T: {robot_theta:.2f}"
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Robot Stream (Laptop End)", frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
