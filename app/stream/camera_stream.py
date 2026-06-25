import sys
import os
# Adds the project root (two levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import cv2
import numpy as np
from app.module.camera import Camera

class CameraStreamer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5002, fps: int = 30, camera: Camera = None):
        """
        Initializes the GStreamer pipeline for streaming processed frames.
        Uses tcpclientsink to ensure compatibility with standard SSH tunnels.
        """
        self.camera = camera
        self.width = camera.w
        self.height = camera.h
        self.fps = fps
        
        
        # Pipeline configured for TCP streaming (compatible with SSH port forwarding)
        self.pipeline = (
            "appsrc ! "
            "video/x-raw, format=BGR ! "
            "videoconvert ! "
            "video/x-raw, format=I420 ! "
            "x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast ! "
            "matroskamux streamable=true ! "
            "tcpserversink host=127.0.0.1 port=5002 sync=true"
        )

        self.writer = cv2.VideoWriter(
            self.pipeline,
            cv2.CAP_GSTREAMER,
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
            
        if do_undistort and self.camera:
            frame = self.camera.undistort_fisheye(frame)
            
        # Ensure frame dimensions match the pipeline expectations
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))
            
        # Push the frame into the GStreamer pipeline
        self.writer.write(frame)

    def release(self):
        """Closes the GStreamer pipeline."""
        self.writer.release()


if __name__ == "__main__":
    cam = Camera("new_calib.npz")
    stream = CameraStreamer(camera=cam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
    else: 
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
                stream.stream_frame(frame, True)
        except Exception:
            raise Exception("something")
        finally:
            cap.release()
            stream.release()