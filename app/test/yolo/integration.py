import os
import sys
# Get the absolute path of the directory two levels up (project root)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import cv2
import math
import time
from ultralytics import YOLO
from app.yolo.detector import *
from app.yolo.geometry import *
from app.localization.localization import *
from app.control.keyboard_controller_ssh import *
#********************************************** IMPORTS **********************************************


#UART
msp432_uart = MSP432Uart()
msp432_uart.start_receiving() #THREAD 1: to listen to odometry data from MSP432 and fill buffer

#Localization
loc = Localization(width=23, length=37, cell_size=50) #23 columns, 37 rows, cell size is 50x50mm
loc.stream_occupancy_grid() #THREAD 2: to stream odometry data to Flask Server
loc.init_odometry_thread(msp432_uart) #THREAD 3: start thread to change localization using UART buffer

#Keyboard Controller
controller = RobotController(msp432_uart.send_command)
controller.start() #start thread to listen for keyboard

#YOLO Model
model = YOLO("./app/models/best.onnx", task="detect")

#Obstacle Definition and setup spatial resolver
cube_id = find_class_id(model, "cube")
cone_id = find_class_id(model, "cone")
cubeResolver = ObjectResolver(28949.08, -1.21) #Inverse Regression parameter for the cube
coneResolver = ObjectResolver(27649.16, -42.82) #Inverse Regression parameter for the cone

#Camera setup
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Optional: Set resolution to improve FPS if needed
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#Main loop
inference_interval = 2.0 
last_inference_time = 0


while True:
    ret, image = cam.read()
    if not ret:
        break

    current_time = time.time()


    # Only run inference if the interval has passed
    if current_time - last_inference_time >= inference_interval:
        results = model(image, conf=0.25)

        #Process Cube(s)
        cube_boxes = get_bounds(results, cube_id)
        for bbox in cube_boxes:
            pixel_width = (abs(bbox[0] - bbox[2])).item()
            angle_from_center = cubeResolver.calculate_theta(bbox[0].item(), bbox[2].item())
            distance = cubeResolver.calculate_distance(pixel_width) 
            cubeResolver.resolve_coor(distance, angle_from_center, loc) #edit the occupancy grid with cone position
        
        # Process Cone(s)
        cone_boxes = get_bounds(results, cone_id)
        for bbox in cone_boxes:
            pixel_width = (abs(bbox[0] - bbox[2])).item()
            angle_from_center = coneResolver.calculate_theta(bbox[0].item(), bbox[2].item())
            distance = coneResolver.calculate_distance(pixel_width) 
            coneResolver.resolve_coor(distance, angle_from_center, loc) #edit the occupancy grid with cone position

        # Update the timestamp
        last_inference_time = current_time
        # annotated_frame = results[0].plot()
    else: 
        # annotated_frame = image
        pass

    # cv2.imshow("YOLOv26 ONNX Inference", annotated_frame)

    # Smallest possible delay to catch 'q' and refresh UI
    if cv2.waitKey(1) == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()