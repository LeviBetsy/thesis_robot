import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from scripts.mde.DAV2_pth import DepthAnythingPredictor
from scale_calibration_floor import FloorScaleCorrection

import cv2
import time

def main(imshow=False):
    # Initialize the depth model (will auto-select mps, cuda, or cpu)
    print("Loading Depth Anything V2 model...")
    predictor = DepthAnythingPredictor()

    #*********************** Initialize GStreamer ***********************

    # Define the GStreamer pipeline string for OpenCV.
    gst_pipeline = (
        "tcpclientsrc host=127.0.0.1 port=5002 ! "
        "matroskademux ! "
        "h264parse ! "
        "decodebin ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1"
    )

    print("Opening GStreamer pipeline stream...")
    # Open the video capture using the GStreamer backend explicitly
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Could not open GStreamer pipeline. Ensure your SSH tunnel is active and streaming on port 5002.")
        return
    print("Stream established. Processing frames (Press 'q' to quit)...")
    #*********************************************************************



    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Empty frame received or stream disconnected.")
                time.sleep(0.1)
                continue

            depth_map = predictor.model.infer_image(frame)

            # ***************** Visualize ******************
            if imshow:
                color_depth = predictor.colorize(depth_map)
                cv2.imshow("Robot Live Camera Feed - Depth Map", color_depth)
                # Check for 'q' key press to break loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            # ***********************************************



            # ***************** Debug ***********************
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            print(f"FPS: {1/elapsed:.2f} | Latency: {elapsed * 1000:.1f}ms")
            prev_time = current_time
            # ***********************************************
            

    finally:
        # Graceful cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline stopped and windows closed.")

if __name__ == "__main__":
    main()
    # pass