import sys
import os
# Adds the project root (two levels up from this file) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst
from app.module.camera import Camera

if not Gst.is_initialized():
    Gst.init(sys.argv)

class GIVideoStreamer:
    def __init__(self, host: str = "127.0.0.1", port: int = 5002, fps: int = 30, camera = None):
        """
        Initializes the GStreamer pipeline via gi for streaming processed frames.
        """
        self.camera = camera
        self.width = camera.w if camera else 640
        self.height = camera.h if camera else 480
        self.fps = fps

        pipeline_str = (
            f"appsrc name=source is-live=true format=GST_FORMAT_TIME ! "
            f"video/x-raw, format=BGR, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            f"videoconvert ! "
            f"video/x-raw, format=I420 ! "
            f"x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast ! "
            f"matroskamux streamable=true ! "
            f"tcpserversink host={host} port={port} sync=false"
        )
        
        # Parse the string into a functioning pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name('source')
        if not self.appsrc:
            raise RuntimeError("Failed to get appsrc from pipeline.")
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # Timing variables
        self.duration = Gst.util_uint64_scale(1, Gst.SECOND, self.fps)

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
            
        # Convert the numpy array to bytes
        data = frame.tobytes()
        
        # Allocate a GStreamer buffer and copy the data into it
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        
        # Set the Presentation Timestamp (PTS) and duration
        buf.duration = self.duration
        
        # Push the buffer into the appsrc
        retval = self.appsrc.emit('push-buffer', buf)
        
        if retval != Gst.FlowReturn.OK:
            print(f"Failed to push buffer to appsrc. Return code: {retval}")
            

    def release(self):
        """Closes the GStreamer pipeline cleanly."""
        if self.appsrc:
            self.appsrc.emit('end-of-stream')
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

class GIVideoReceiver:
    def __init__(self, host: str = "127.0.0.1", port: int = 5002, callback=None, camera: Camera = None):
        """
        Initializes the GStreamer pipeline to receive and decode the TCP video stream.
        
        :param callback: A function that takes (frame: np.ndarray, pts: float)
        """
        self.host = host
        self.port = port
        self.callback = callback
        self.camera = camera
        
        pipeline_str = (
            f"tcpclientsrc host={self.host} port={self.port} ! "
            f"matroskademux ! "
            f"h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink name=sink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name('sink')
        if not self.appsink:
            raise RuntimeError("Failed to get appsink from pipeline.")
        self.appsink.connect("new-sample", self._new_sample_handler) #attach process callback for whenever receive frame
            
        self.pipeline.set_state(Gst.State.PLAYING)
        print(f"Receiver attempting to connect to tcp://{self.host}:{self.port}...")

    def _new_sample_handler(self, sink):
        """
        Internal callback triggered by GStreamer. Extracts the data and routes it
        to the user-defined callback.
        """
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
            
        buffer = sample.get_buffer()
        
        pts = buffer.pts
        
        # Map Memory
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if success:
            try:
                frame = np.ndarray(
                    shape=(self.camera.h, self.camera.w, 3),
                    dtype=np.uint8,
                    buffer=map_info.data
                )
                frame_copy = frame.copy()
                
                # TRIGGER THE USER'S CALLBACK
                self.callback(frame_copy, pts)
                    
            finally:
                buffer.unmap(map_info)
                
        return Gst.FlowReturn.OK

    def release(self):
        """Closes the receiver pipeline cleanly."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

class OpenCVCameraStreamer:
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
    stream = OpenCVCameraStreamer(camera=cam)
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