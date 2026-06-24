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

# **********************************************************************

# # ***************************** Initialize *****************************

# # Initialize GStreamer
# Gst.init(None)
# display_queue = queue.Queue(maxsize=10) # Queue to pass synchronized frames to the main thread for calculation
# # Dictionaries to hold arriving buffers until their matching partner arrives
# video_cache = {}
# data_cache = {}

# # Robot
# robot = Robot("fisheye_calib.npz")
# w, h = robot.camera.w, robot.camera.h

# # **********************************************************************

# # ***************************** GStreamer Callback *********************

# def check_and_queue_sync(pts):
#     """Checks if we have BOTH the video and data for a given timestamp.
#         And then push it to a queue for process """
#     if pts in video_cache and pts in data_cache:
#         # We have a match! Extract them and remove from caches.
#         frame = video_cache.pop(pts)
#         state = data_cache.pop(pts)
        
#         # Clean up old orphaned frames in case packets were dropped over the network
#         for old_pts in list(video_cache.keys()):
#             if old_pts < pts: del video_cache[old_pts]
#         for old_pts in list(data_cache.keys()):
#             if old_pts < pts: del data_cache[old_pts]

#         # Push the synchronized pair to the main thread
#         if not display_queue.full():
#             display_queue.put((frame, state))

# def callback_new_video(sink):
#     print("got new video")
#     """Callback triggered when a new video frame arrives from the network."""
#     sample = sink.emit("pull-sample")
#     if not sample:
#         return Gst.FlowReturn.ERROR

#     buf = sample.get_buffer()
#     pts = buf.pts # The synchronization key

#     result, map_info = buf.map(Gst.MapFlags.READ)
#     if result:
#         frame = np.ndarray((h, w, 3), buffer=map_info.data, dtype=np.uint8).copy() # .copy() to prevent memory corruption when GStreamer unmaps the buffer.
#         buf.unmap(map_info)
        
#         video_cache[pts] = frame
#         check_and_queue_sync(pts)

#     return Gst.FlowReturn.OK

# def callback_new_data(sink):
#     """Callback triggered when a new telemetry JSON string arrives."""
#     sample = sink.emit("pull-sample")
#     if not sample:
#         return Gst.FlowReturn.ERROR

#     buf = sample.get_buffer()
#     pts = buf.pts # The synchronization key

#     result, map_info = buf.map(Gst.MapFlags.READ)
#     if result:
#         # Decode the UTF-8 bytes back into a Python dictionary
#         text_data = map_info.data.decode('utf-8')
#         # Remove the null terminator that GStreamer sometimes appends to text caps
#         text_data = text_data.strip('\x00') 
#         state = json.loads(text_data)
#         buf.unmap(map_info)
        
#         data_cache[pts] = state
#         check_and_queue_sync(pts)

#     return Gst.FlowReturn.OK

# # **********************************************************************

# def main():
#     # **************** GSTREAMER ****************
#     pipeline_str = (
#         "tcpclientsrc host=127.0.0.1 port=5002 ! matroskademux name=demux "
        
#         # Video Branch
#         "demux. ! queue ! h264parse ! decodebin ! videoconvert ! "
#         "video/x-raw, format=BGR ! appsink name=videosink emit-signals=true sync=true "
        
#         # # Data Branch
#         # "demux. ! queue ! text/x-raw ! appsink name=datasink emit-signals=true sync=true"
#     )

#     print("Initializing GStreamer Pipeline...")
#     pipeline = Gst.parse_launch(pipeline_str)

#     videosink = pipeline.get_by_name("videosink")
#     videosink.connect("new-sample", callback_new_video)

#     # datasink = pipeline.get_by_name("datasink")
#     # datasink.connect("new-sample", callback_new_data)

#     pipeline.set_state(Gst.State.PLAYING) # Start the pipeline
#     print("Listening for multiplexed stream on port 5002...")

#     # ******************************************

#     try:
#         while True:
#             # Block and wait for a synchronized pair from the GStreamer callbacks
#             frame, state = display_queue.get()
            
#             # Extract your synchronized variables
#             robot_x = state.get("x", 0.0)
#             robot_y = state.get("y", 0.0)
#             robot_theta = state.get("theta", 0.0)
            
#             # Overlay the data onto the synced frame
#             text = f"Synced -> X: {robot_x:.2f} | Y: {robot_y:.2f} | T: {robot_theta:.2f}"
#             cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
#             cv2.imshow("Robot Stream (Laptop End)", frame)
            
#             # Press 'q' to quit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         pipeline.set_state(Gst.State.NULL)
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

def main(imshow=False):
    # Initialize the depth model (will auto-select mps, cuda, or cpu)
    print("Loading Depth Anything V2 model...")
    predictor = DepthAnythingPredictor()

    # #*********************** Initialize GStreamer ***********************

    # # Define the GStreamer pipeline string for OpenCV.
    # gst_pipeline = (
    #     "tcpclientsrc host=127.0.0.1 port=5002 ! "
    #     "matroskademux ! "
    #     "h264parse ! "
    #     "decodebin ! "
    #     "videoconvert ! "
    #     "video/x-raw, format=BGR ! "
    #     "appsink drop=true max-buffers=1"
    # )

    # print("Opening GStreamer pipeline stream...")
    # # Open the video capture using the GStreamer backend explicitly
    # cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    # if not cap.isOpened():
    #     print("Error: Could not open GStreamer pipeline. Ensure your SSH tunnel is active and streaming on port 5002.")
    #     return
    # print("Stream established. Processing frames (Press 'q' to quit)...")
    # #*********************************************************************

    # Initialize Robot
    robot = Robot("fisheye_calib.npz")
    fsc = FloorScaleCorrection("z_real")
    pcd_processor = PointCloudProcessor(robot)


    pcd = o3d.geometry.PointCloud() #Visualizer

    prev_time = time.perf_counter()
    try:
        while True:
            # # ***************** Gstreamer ***************
            # ret, frame = cap.read()
            # if not ret or frame is None:
            #     print("Warning: Empty frame received or stream disconnected.")
            #     time.sleep(0.1)
            #     continue
            # # *******************************************
            frame = cv2.imread("./data/test/und_ref8.png")

            rel_depth_map = predictor.model.infer_image(frame)

            # ***************** Visualize ******************
            if imshow:
                color_depth = predictor.colorize(rel_depth_map)
                cv2.imshow("Robot Live Camera Feed - Depth Map", color_depth)
                # Check for 'q' key press to break loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # ***********************************************
            fsc.scale_correction(rel_depth_map, True, "error.png")
            metric_map = fsc.relative_to_metric(rel_depth_map)

            point_cloud_cc = pcd_processor.proj_pcd_cc(metric_map)
            point_cloud_rc = pcd_processor.pcd_camera_to_robot(point_cloud_cc)
            z_avg = pcd_processor.average_floor_z(point_cloud_rc)
            print(f"average z: {z_avg}")

            pcd.points = o3d.utility.Vector3dVector(point_cloud_rc)
            # print(pcd.colors.shape)
            


            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([pcd, axes])


            # ***************** Debug ***********************
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            print(f"FPS: {1/elapsed:.2f} | Latency: {elapsed * 1000:.1f}ms")
            prev_time = current_time
            # ***********************************************
            break

    finally:
        # Graceful cleanup
        # cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")

if __name__ == "__main__":
    main(imshow=True)