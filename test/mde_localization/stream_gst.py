import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import sys
import time
import cv2
import json
import numpy as np

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst
import zmq

from app.stream.pose_stream import PoseStream

def main():
    # ***************** Initalize Pose Stream ****************
    # Interprocess communication for latest robot pose
    context = zmq.Context()
    zmq_socket = context.socket(zmq.SUB)
    zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "state")
    zmq_socket.setsockopt(zmq.CONFLATE, 1)  # Drop old messages, only keep latest
    zmq_socket.connect("tcp://127.0.0.1:5556")

    pose_stream = PoseStream(port=5000)
    latest_pose = {"x": 0.0, "y": 0.0, "theta": 0.0, "pts": 0}
    # ********************************************************

    # ****************** Intialize GStreamer *****************
    Gst.init(sys.argv)
    
    # Define the pipeline string
    pipeline_str = (
        "appsrc name=source is-live=true format=GST_FORMAT_TIME ! "
        "video/x-raw, format=BGR, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=I420 ! "
        "x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast ! "
        "matroskamux streamable=true ! "
        "tcpserversink host=127.0.0.1 port=5002 sync=false"
    )
    
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name('source')
    pipeline.set_state(Gst.State.PLAYING)
    print("Pipeline running. Listening on tcp://127.0.0.1:5002...")
    
    # Timing variables
    timestamp = 0
    fps = 30
    duration = Gst.util_uint64_scale(1, Gst.SECOND, fps)

    # *******************************************************

    cap = cv2.VideoCapture(0)
    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            # ********** Pose Stream **********
            try:
                msg = zmq_socket.recv_string(flags=zmq.NOBLOCK)
                topic, json_data = msg.split(" ", 1)
                latest_pose = json.loads(json_data)
                latest_pose["pts"] = timestamp
            except zmq.Again:
                # No new position packet; reuse the last known position
                pass

            print(latest_pose)
            pose_stream.send_data(latest_pose)
            # ********************************
            
            # *********** GST Stream **********
            # Convert the numpy array to bytes
            data = frame.tobytes()
            
            # Allocate a GStreamer buffer and copy the data into it
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            
            # Set the Presentation Timestamp (PTS) and duration
            buf.pts = timestamp
            buf.dts = timestamp
            buf.duration = duration
            
            # Push the buffer into the appsrc
            retval = appsrc.emit('push-buffer', buf)
            
            if retval != Gst.FlowReturn.OK:
                print(f"Failed to push buffer to appsrc. Return code: {retval}")
                break
            # *********************************

                
            # Increment the timestamp for the next frame
            timestamp += duration
            
            # Sleep to match the target framerate (accounting for processing time)
            processing_time = time.time() - loop_start
            sleep_time = (1.0 / fps) - processing_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        # Signal the end of the stream and clean up
        appsrc.emit('end-of-stream')
        pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()