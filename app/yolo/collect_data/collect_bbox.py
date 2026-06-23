import os
import sys

# Get the absolute path of the directory two levels up (project root)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import cv2
from ultralytics import YOLO
from app.yolo.detection import *

# 1. Path to your ONNX model
model_path = "./app/models/best.onnx"
output_file = "./app/data/distance_log_cube.txt"

# Load the ONNX model
model = YOLO(model_path, task="detect")

class_id = find_class_id(model, "cube")

cam = cv2.VideoCapture(0)

# To keep track of entries within the current session
num = 1

print(f"Logged data will be saved to: {output_file}")

while True:
    ret, image = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference using the ONNX model
    results = model(image)

    pixel_width = 0
    # Use your custom filter logic
    box_mouse = get_bounds(results, class_id) 
    
    if box_mouse:
        # Calculate width: x_max - x_min
        pixel_width = abs(box_mouse[0][0] - box_mouse[0][2])
        
    # Draw results for visual confirmation
    annotated_frame = results[0].plot()
    cv2.imshow("Detection Loop", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        # Use 'a' mode to append to the end of the file
        with open(output_file, "a") as f:
            f.write(f"entry_{num} pixel_width: {pixel_width}\n")
            print(f"Logged entry {num}: {pixel_width}px")
            num += 1

cam.release()
cv2.destroyAllWindows()