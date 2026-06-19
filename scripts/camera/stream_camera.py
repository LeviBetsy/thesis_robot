import sys
import os
# Adds the project root (two levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import cv2 as cv
import numpy as np
from app.camera.undistorter import ImageUndistorter

class CameraStreamer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5002, width: int = 640, height: int = 480, fps: int = 30, undistorter=None):
        """
        Initializes the GStreamer pipeline for streaming processed frames.
        Uses tcpclientsink to ensure compatibility with standard SSH tunnels.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.undistorter = undistorter
        
        # Pipeline configured for TCP streaming (compatible with SSH port forwarding)
        # using hardware-accelerated H.264 encoding if available, otherwise x264enc.
        # Note: On a Pi, you may replace 'x264enc' with 'v4l2h264enc' for hardware encoding.
        # self.pipeline = (
        #     "appsrc ! "
        #     "video/x-raw, format=BGR ! "
        #     "videoconvert ! "
        #     "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
        #     "rtph264pay config-interval=1 pt=96 ! "  # Added config-interval
        #     "rtpstreampay ! "                        # Commits RTP packets to a stream format
        #     "tcpserversink host=127.0.0.1 port=5002"
        # )

        self.pipeline = (
            "appsrc ! "
            "video/x-raw, format=BGR ! "
            "videoconvert ! "
            "video/x-raw, format=I420 ! "
            "x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast ! "
            "matroskamux streamable=true ! "
            "tcpserversink host=127.0.0.1 port=5002 sync=false"
        )

        self.writer = cv.VideoWriter(
            self.pipeline,
            cv.CAP_GSTREAMER,
            0, # 0 means fourcc is ignored for GStreamer pipelines
            self.fps,
            (self.width, self.height),
            True
        )
        
        if not self.writer.isOpened():
            raise RuntimeError("Failed to open GStreamer VideoWriter. Ensure OpenCV has GStreamer support enabled.")

    def stream_frame(self, frame: np.ndarray, do_undistort: bool = False):
        """
        Conditionally undistorts the image and pushes it to the GStreamer pipeline.
        
        :param frame: The raw input image (NumPy array)
        :param do_undistort: Boolean flag to apply fisheye undistortion
        """
        if frame is None:
            return
            
        # Apply undistortion if requested and the undistorter instance exists
        if do_undistort and self.undistorter:
            frame = self.undistorter.undistort_fisheye(frame)
            
        # Ensure frame dimensions match the pipeline expectations
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv.resize(frame, (self.width, self.height))
            
        # Push the frame into the GStreamer pipeline
        self.writer.write(frame)

    def release(self):
        """Closes the GStreamer pipeline."""
        self.writer.release()


# Example Usage Integration:
if __name__ == "__main__":
    undistorter_instance = ImageUndistorter("fisheye_camera_calibration.npz")
    
    # host="127.0.0.1" assumes you are using an SSH local forward (ssh -L)
    streamer = CameraStreamer(host="127.0.0.1", port=5002, undistorter=undistorter_instance)
    
    cap = cv.VideoCapture(0)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    print(f"Camera Resolution: {int(width)}x{int(height)}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Pass the frame and the boolean flag
            streamer.stream_frame(frame, do_undistort=True)
            
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        streamer.release()