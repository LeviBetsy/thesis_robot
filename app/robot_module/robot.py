import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from app.robot_module.camera import Camera
from app.util.config import Config
import numpy as np
import math
import threading


class Robot:
    def __init__(self, config: Config):
        """
        Initializes the Robot with its position, orientation, and camera matrices.
            camera_transform_matrix (np.ndarray, optional): A 4x4 homogeneous transformation matrix 
                                                            for the camera. Defaults to a 4x4 identity matrix.
            Coordinate system is based on unit wheel (CCW, +N -W -S +E) and 0 theta points East
        """
        rconfig = config.robot
        angle = np.radians(rconfig.camera_tilt_deg) # Initilize the camera to tilt 30 degree down from the mounting position
        self.cam_R = np.array([ # Rotation matrix to get camera coordinate to robot coordinate
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        self.cam_t = np.array(rconfig.cam_t) # Translation vector to get camera to robot coorindate, 7 cm down, 11 cm forward
        #Robot description in meters
        self.n = rconfig.encoder_slots
        self.w = rconfig.wheel_base
        self.d = rconfig.wheel_diameter
        self.c = math.pi*self.d

        self.camera = Camera(config.paths.camera_calib)