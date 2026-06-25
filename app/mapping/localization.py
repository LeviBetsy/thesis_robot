import numpy as np
import cv2
import threading
import math
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from app.module.uart import *
from app.module.robot import Robot

class OdometryLocalization:
    def __init__(self, robot: Robot):
        self.robot = robot

    def init_odometry_thread(self, msp432_uart: MSP432Uart): #Start thread to change localization data using UART buffer
        def odom_loop():
            while True:
                # Retrieve the latest counts from the UART interface
                # Note: parentheses added assuming get_data is a method call
                Lcount, Rcount = msp432_uart.get_data()
                
                # Update the internal odometry state
                self.update_odom_coordinate(Lcount, Rcount)
        self.odometry_thread = threading.Thread(target=odom_loop, daemon=True)
        self.odometry_thread.start()


    def update_odom_coordinate(self, LCount, RCount):
        """
        Updates the robot's coordinate and angle based on wheel encoder counts.
        """
        dl = LCount*self.robot.c/self.robot.n
        dr = RCount*self.robot.c/self.robot.n
        d = (dl + dr)/2 #distance traveled by the middle point
        delta_theta = (dr - dl)/self.robot.w

        with self.robot.mutex_lock:
            new_x = self.robot.x + (d*math.cos(self.robot.theta + (delta_theta/2)))
            new_y = self.robot.y - (d*math.sin(self.robot.theta + (delta_theta/2)))
            new_theta = self.robot.theta + delta_theta
            self.robot.set_robot_pose(new_x, new_y, new_theta)
