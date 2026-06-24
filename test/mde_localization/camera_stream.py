import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)


import json
import time
import cv2
import zmq

# Import GStreamer bindings
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

def main():
    # # 1. Initialize ZeroMQ Subscriber to receive local updates from robot process
    # context = zmq.Context()
    # zmq_socket = context.socket(zmq.SUB)
    # zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "state")
    # zmq_socket.setsockopt(zmq.CONFLATE, 1)  # Drop old messages, only keep latest
    # zmq_socket.connect("tcp://127.0.0.1:5556")

    # 2. Initialize Camera Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0: 
        fps = 30  # Fallback if camera doesn't report frame rate

    # Calculate frame duration in nanoseconds (GStreamer uses nanoseconds for clocking)
    frame_duration = int(1e9 / fps)

    pipeline_str = (
        f"appsrc name=videosrc emit-signals=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        f"videoconvert ! x264enc tune=zerolatency bitrate=1000 speed-preset=superfast ! "
        f"matroskamux name=mux streamable=true ! tcpserversink host=127.0.0.1 port=5002 "
        f"appsrc name=datasrc emit-signals=true format=time caps=text/x-raw,format=utf8 ! "
        f"queue max-size-buffers=3 leaky=downstream ! mux."
    )

    print("Launching GStreamer multiplexer pipeline...")
    pipeline = Gst.parse_launch(pipeline_str)
    
    # Get handles to the injection points
    videosrc = pipeline.get_by_name("videosrc")
    datasrc = pipeline.get_by_name("datasrc")

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)

    frame_count = 0
    latest_state = {"x": 0.0, "y": 0.0, "theta": 0.0}

    try:
        while True:
            start_time = time.time()
            
            # Capture the frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # # Check for new telemetry data from the localization process
            # try:
            #     msg = zmq_socket.recv_string(flags=zmq.NOBLOCK)
            #     topic, json_data = msg.split(" ", 1)
            #     latest_state = json.loads(json_data)
            # except zmq.Again:
            #     # No new position packet; reuse the last known position
            #     pass
            

            # Compute the synchronized timestamp for this loop iteration (in nanoseconds)
            current_pts = frame_count * frame_duration

            # --- A. Package and Push Video Frame ---
            video_bytes = frame.tobytes()
            video_buf = Gst.Buffer.new_allocate(None, len(video_bytes), None)
            video_buf.fill(0, video_bytes)
            video_buf.pts = current_pts
            video_buf.duration = frame_duration
            
            videosrc.emit("push-buffer", video_buf)

            # --- B. Package and Push Matching Telemetry Data ---
            data_bytes = json.dumps(latest_state).encode('utf-8')
            data_buf = Gst.Buffer.new_allocate(None, len(data_bytes), None)
            data_buf.fill(0, data_bytes)
            data_buf.pts = current_pts
            data_buf.duration = frame_duration
            
            datasrc.emit("push-buffer", data_buf)

            frame_count += 1

            # Throttle the loop to match the target camera frame rate
            elapsed = time.time() - start_time
            sleep_time = (1.0 / fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        # Clean up resources
        pipeline.set_state(Gst.State.NULL)
        cap.release()
        # zmq_socket.close()
        # context.term()

if __name__ == "__main__":
    main()