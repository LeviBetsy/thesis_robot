import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import time
import numpy as np

from app.perception.mde_depth import MDE_Depth
from app.perception.point_cloud_visualizer import PointCloudVisualizer
from app.stream.zmq_stream import VideoReceiver


def real_time_pcd():
    mde_depth = MDE_Depth(calib_file="fisheye_calib.npz", ref_file="z_real_16group.npz")
    visualizer = PointCloudVisualizer()

    display_frame = [np.random.rand(480, 640, 3)]
    def callback_new_video(frame):
        display_frame[0] = frame
    receiver = VideoReceiver(callback=callback_new_video)
    time.sleep(5)
    try:
        while True:
            frame = display_frame[0]

            pcd : np.ndarray = mde_depth.frame_to_pointcloud(frame)
            visualizer.update(pcd)
            
            #CV Imshow
            anno_frame = mde_depth.annotate_floor(frame)
            cv2.imshow("Video Feed", anno_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        receiver.stop()
        visualizer.close()
        cv2.destroyAllWindows

def static_inspection():
    mde_depth = MDE_Depth(calib_file="fisheye_calib.npz", ref_file="z_real_16group.npz")
    visualizer = PointCloudVisualizer()
    frame = cv2.imread("./data/leg_problem/ref20.jpg")

    pcd = mde_depth.frame_to_pointcloud(frame)
    rel_depth = mde_depth.rel_depth
    print(f"min calibrated rel is: {mde_depth.fsc.min_calibrated_rel}")
    visualizer.update(pcd)
    def click_event(event, x, y, flags, param):
        # Check if the event was a left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param  # Extract the image passed in
            
            # OpenCV passes (x, y), but numpy arrays are indexed as [row, col] -> [y, x]
            pixel_value = rel_depth[y, x] 
            
            print(f"Clicked at (x={x}, y={y}) | Drel Value: {pixel_value} | ")

    window_name = "Debug Window"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event, frame)
    
    while True:
        visualizer.update(pcd)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    visualizer.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_pcd()
    # static_inspection()