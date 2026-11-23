import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import time
import threading
import numpy as np

from app.perception.mde_depth import MDE_Depth
from app.perception.dav2_pth import DepthAnythingPredictor
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

def real_time_visualize(cv_show=False):
    """Receives Pi frames and visualizes the full pcd (ground kept); nothing is streamed back to the Pi."""
    mde_depth = MDE_Depth(config)

    display_frame = [None]
    new_frame = threading.Event()
    def callback_new_video(frame):
        display_frame[0] = frame
        new_frame.set()
    video_receiver = VideoReceiver(host=config.stream.pi_host, port=config.stream.video_port,
                                   callback=callback_new_video)
    visualizer = PointCloudVisualizer()
    try:
        while True:
            if not new_frame.wait(timeout=1.0):
                continue
            new_frame.clear()
            frame = display_frame[0]

            pcd = mde_depth.frame_to_pcd(frame, delete_ground=False)
            visualizer.update(pcd)

            if cv_show:
                cv2.imshow("Video Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        video_receiver.stop()
        visualizer.close()
        cv2.destroyAllWindows()

def static_inspection():
    mde_depth = MDE_Depth(config)
    visualizer = PointCloudVisualizer()
    frame = cv2.imread("./data/leg_problem/ref20.jpg")

    pcd = mde_depth.frame_to_pcd(frame)
    rel_depth = mde_depth.rel_depth
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

def static_calibration_plot(path):
    """Runs MDE on a single image, fits the floor scale calibration, and saves the fsc plot to data/plot/."""
    mde_depth = MDE_Depth(config)
    frame = cv2.imread(path)

    rel_depth = mde_depth.process_rel(frame)
    plot_file = f"fsc_{os.path.splitext(os.path.basename(path))[0]}.png"
    mde_depth.fsc.scale_calibration(rel_depth, plot=True, plot_file=plot_file, plot_line=True)

def benchmark_latency(path, n_iters=50, n_warmup=5):
    """Feeds the same frame through the real_time_stream per-frame pipeline n_iters times and prints per-stage latency."""
    mde_depth = MDE_Depth(config)
    frame = cv2.imread(path)

    def run_once():
        t0 = time.perf_counter()
        pcd = mde_depth.frame_to_pcd(frame)
        t1 = time.perf_counter()
        mde_depth.pcd_to_ray_casting(pcd, n_rays=config.ray_cast.n_rays,
                                     max_range=config.ray_cast.max_range)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    for _ in range(n_warmup): #first passes include MPS kernel compile / allocation, not representative
        run_once()

    times = np.array([run_once() for _ in range(n_iters)]) * 1000 #shape (n_iters, 2), ms
    stages = {"frame_to_pcd": times[:, 0], "ray_casting": times[:, 1], "total": times.sum(axis=1)}
    print(f"Latency over {n_iters} iters ({n_warmup} warmup), frame {frame.shape[1]}x{frame.shape[0]}:")
    for name, t in stages.items():
        print(f"  {name:13s} mean {t.mean():7.2f} ms | median {np.median(t):7.2f} ms | max {t.max():7.2f} ms")
    print(f"  throughput    {1000 / stages['total'].mean():.2f} FPS")


if __name__ == "__main__":
    # static_inspection()
    real_time_stream(cv_show=False, visualize=False)
    # real_time_stream()
    # real_time_visualize(cv_show=True)
    # real_time_dav2_visualize(cv_show=True)
    # static_process("./data/ref23.jpg")
    # static_calibration_plot("./data/checkerboard/ref13.jpg")
    # benchmark_latency("./data/checkerboard/ref13.jpg")