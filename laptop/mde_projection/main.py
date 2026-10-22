import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import time
import threading
import numpy as np

from app.perception.mde_depth import MDE_Depth
from app.perception.point_cloud_visualizer import PointCloudVisualizer
from app.stream.zmq_stream import VideoReceiver, RangeStreamer
from app.util.config import Config

# Same file the Pi loads, so the ray fan (n_rays, max_range) is binned identically on both sides
config = Config.load("./config/robot_config.json")

def real_time_stream(cv_show=False, visualize=False):
    mde_depth = MDE_Depth(config)

    display_frame = [None]
    new_frame = threading.Event() #set by the receiver thread, so MDE only runs once per fresh frame
    def callback_new_video(frame):
        display_frame[0] = frame
        new_frame.set()
    video_receiver = VideoReceiver(host=config.stream.pi_host, port=config.stream.video_port,
                                   callback=callback_new_video)
    # pcd_streamer = PointStreamer(fps= 3)
    range_streamer = RangeStreamer(port=config.stream.range_port)
    if visualize:
        visualizer = PointCloudVisualizer()
    time.sleep(5)
    try:
        while True:
            # Timeout keeps the loop responsive to Ctrl+C when the Pi isn't sending
            if not new_frame.wait(timeout=1.0):
                continue
            new_frame.clear()
            frame = display_frame[0] #newest frame; any that arrived during the last MDE pass are skipped

            pcd = mde_depth.frame_to_pcd(frame)
            if visualize: visualizer.update(pcd)
            ray_cast = mde_depth.pcd_to_ray_casting(pcd, n_rays=config.ray_cast.n_rays,
                                                    max_range=config.ray_cast.max_range)
            range_streamer.send_range(ray_cast)

            #CV Imshow
            if cv_show:
                cv2.imshow("Video Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        video_receiver.stop()
        range_streamer.stop()
        if visualize: visualizer.close()
        cv2.destroyAllWindows()

def static_inspection():
    mde_depth = MDE_Depth(config)
    visualizer = PointCloudVisualizer()
    frame = cv2.imread("./data/leg_problem/ref20.jpg")

    pcd = mde_depth.frame_to_pcd(frame)
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

def static_process(path):
    mde_depth = MDE_Depth(config)
    frame = cv2.imread(path)
    visualizer = PointCloudVisualizer()

    pcd = mde_depth.frame_to_pcd(frame)
    ray_cast = mde_depth.pcd_to_ray_casting(pcd, n_rays=config.ray_cast.n_rays,
                                            max_range=config.ray_cast.max_range)
    print("Ray Casting result:")
    print(ray_cast)

    while True:
        visualizer.update(pcd)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    visualizer.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # static_inspection()
    # real_time_stream(cv_show=True, visualize=True)
    real_time_stream()
    # static_process("./data/ref23.jpg")