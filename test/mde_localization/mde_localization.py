import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

import zmq
import json
import time
from app.module.robot import Robot
from app.module.uart import MSP432Uart
from app.control.keyboard_controller_ssh import RobotController
from app.mapping.localization import OdometryLocalization
#********************************************** IMPORTS **********************************************


#UART
msp432_uart = MSP432Uart()
msp432_uart.start_receiving() #THREAD 1: to listen to odometry data from MSP432 and fill buffer

#Robot
robot = Robot("new_calib.npz")

# #Localization
loc = OdometryLocalization(robot=robot)
loc.init_odometry_thread(msp432_uart) #THREAD 2: start thread to change localization using UART buffer

#Keyboard Controller
controller = RobotController(msp432_uart.send_command)
controller.start() #THREAD 3: start thread to listen for keyboard and sending command to msp432

# ZeroMQ publisher for camera stream process
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5556")
state = {"x": 0, "y": 0, "theta": 0}
#Main loop
while True:
    #update publisher for camera stream
    # state = {"x": robot.x, "y": robot.y, "theta": robot.theta}
    with robot.mutex_lock:
        state = {"x": robot.x, "y": robot.y, "theta": robot.theta}
    socket.send_string(f"state {json.dumps(state)}")
    time.sleep(0.02)