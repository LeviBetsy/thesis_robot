import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.module.camera import Camera
import numpy as np
import math
import threading


class Robot:
    def __init__(self, camera_calib_fpath):
        """
        Initializes the Robot with its position, orientation, and camera matrices.
            camera_transform_matrix (np.ndarray, optional): A 4x4 homogeneous transformation matrix 
                                                            for the camera. Defaults to a 4x4 identity matrix.
        """
        self.x = 0
        self.y = 0
        self.theta = 0

        angle = np.radians(-30) # Initilize the camera to tilt 30 degree down from the mounting position
        self.cam_R = np.array([ # Rotation matrix to get camera coordinate to robot coordinate
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        self.cam_t = np.array([0, 0.07, 0.11]) # Translation vector to get camera to robot coorindate, 7 cm down, 11 cm forward
        self.camera = Camera(camera_calib_fpath)

        #Robot description in meters
        self.n = 360 #number of slots/rotation
        self.d = 0.07 #70 mm diameter
        self.w = 0.122 #122mm distance between wheels
        self.c = math.pi*self.d #circumfrence of wheel

        self.mutex_lock = threading.Lock() #mutex lock so only 1 thread access at once
    def set_robot_pose(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta