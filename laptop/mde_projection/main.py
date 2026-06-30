import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import time
import numpy as np
import math
from pathlib import Path

from scripts.mde.DAV2_pth import DepthAnythingPredictor
from app.mde.scale_calibration_floor import FloorScaleCorrection
from app.module.robot import Robot
from app.mapping.point_cloud import PointCloudProcessor
from app.stream.zmq_stream import PoseVideoReceiver

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

def callback_new_pose_video(pose, frame):
    latest_pose[0] = pose
    display_frame[0] = frame

# Receivers
receiver = PoseVideoReceiver(callback=callback_new_pose_video) # THREAD 1: start thread to listen for video and pose
# **********************************************************************

def main(process_pcd = True, save_pcd=False, save_file_path="example.npz", cv_imshow = False):
    if cv_imshow:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 255, 0)      # Green color in BGR format
        thickness = 2
    boot = True
    try:
        while True:
            if boot:
                wait = 10
                print(f"Warming up video receiver, please wait {wait} second")
                print("************************************************")
                time.sleep(wait)
                print("Program ready")
                boot = False
            frame = display_frame[0]
            pose = latest_pose[0]
            
            
            with robot.mutex_lock:
                robot.set_robot_pose(pose["x"], pose["y"], pose["theta"])
            
            # ****************** Point Cloud MDE Inference ***************
            if process_pcd:
                rel_depth = predictor.infer(frame)
                fsc.scale_correction(rel_depth)
                metric_depth = fsc.relative_to_metric(rel_depth)

                pcd_cc = pcd_processor.proj_pcd_cc(metric_depth)
                pcd_wc = pcd_processor.pcd_camera_to_world(pcd_cc)
                if save_pcd:
                    pcd_collection.append(pcd_wc)
                
                # pcd.points = o3d.utility.Vector3dVector(pcd_wc)
                # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([pcd, axes])
            # ************************************************************

            #CV Imshow
            if cv_imshow:
                text_x = f"X: {pose['x']:.2f}"
                text_y = f"Y: {pose['y']:.2f}"
                text_theta = f"Theta: {math.degrees(pose['theta']):.2f} degree"
                cv2.putText(frame, text_x, (20, 40), font, font_scale, color, thickness)
                cv2.putText(frame, text_y, (20, 70), font, font_scale, color, thickness)
                cv2.putText(frame, text_theta, (20, 100), font, font_scale, color, thickness)
                cv2.imshow("Pose Telemetry", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if (save_pcd):
            script_path = Path(__file__).resolve()
            project_root = script_path.parents[2]  # Goes up two levels from scripts/
            output = project_root / "data" / "point_cloud" / save_file_path
            combined_pcd = np.vstack(pcd_collection) #combine into (M, 3) nparray
            unique_pcd = np.unique(combined_pcd, axis=0)
            np.savez_compressed(output, points=unique_pcd)
            print(f"Saved to {output}")
        # pose_receiver.stop()
        # video_receiver.release()
        receiver.stop()
        if (cv_imshow): cv2.destroyAllWindows

if __name__ == "__main__":
    main(process_pcd=True, save_pcd=True, save_file_path="pcd3.npz", cv_imshow=False)
