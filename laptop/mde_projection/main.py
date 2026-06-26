import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import json
import cv2
import time
# import open3d as o3d
import numpy as np
from pathlib import Path
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
if not Gst.is_initialized():
    Gst.init(sys.argv)

# Robot
robot = Robot("fisheye_calib.npz")
camera = robot.camera
w, h = camera.w, camera.h
latest_pose = [{"x": 0, "y": 0, "theta": 0}]
display_frame = [np.random.rand(480, 640, 3)]

predictor = DepthAnythingPredictor()
fsc = FloorScaleCorrection("z_real")
pcd_processor = PointCloudProcessor(robot)
# pcd = o3d.geometry.PointCloud() #Visualizer
pcd_collection = []

# **********************************************************************

# **************************** Receiver Callbacks *********************
def callback_new_video(frame, pts):
    # if not display_queue.full():
    #     display_queue.put((frame, latest_pose))
    display_frame[0] = frame

def callback_new_pose(pose):
    """Callback triggered when a new telemetry JSON string arrives."""
    latest_pose[0] = pose

# Receivers
pose_receiver = PoseReceiver(callback=callback_new_pose)
video_receiver = GIVideoReceiver(callback=callback_new_video, camera=camera)
# **********************************************************************

def main(save_point_cloud=False, save_file_path="example.npz"):
    boot = True
    try:
        while True:
            if boot:
                print("Warming up video receiver")
                time.sleep(10)
                boot = False
            # Block and wait for a synchronized pair from the GStreamer callbacks
            frame = display_frame[0]
            pose = latest_pose[0]
            
            
            with robot.mutex_lock:
                robot.set_robot_pose(pose["x"], pose["y"], pose["theta"])
            
            rel_depth = predictor.infer(frame)
            fsc.scale_correction(rel_depth)
            metric_depth = fsc.relative_to_metric(rel_depth)

            pcd_cc = pcd_processor.proj_pcd_cc(metric_depth)
            pcd_wc = pcd_processor.pcd_camera_to_world(pcd_cc)
            if save_point_cloud:
                pcd_collection.append(pcd_wc)
            
            # pcd.points = o3d.utility.Vector3dVector(pcd_wc)
            # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([pcd, axes])


            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
        if (save_point_cloud):
            script_path = Path(__file__).resolve()
            project_root = script_path.parents[2]  # Goes up two levels from scripts/
            output = project_root / "data" / "point_cloud" / save_file_path
            combined_pcd = np.vstack(pcd_collection) #combine into (M, 3) nparray
            unique_pcd = np.unique(combined_pcd, axis=0)
            np.savez_compressed(output, points=unique_pcd)

    finally:
        # pipeline.set_state(Gst.State.NULL)
        pose_receiver.stop()
        video_receiver.release()

if __name__ == "__main__":
    main()
