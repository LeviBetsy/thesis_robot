import numpy as np

class Robot:
    def __init__(self):
        """
        Initializes the Robot with its position, orientation, and camera matrices.
            camera_transform_matrix (np.ndarray, optional): A 4x4 homogeneous transformation matrix 
                                                            for the camera. Defaults to a 4x4 identity matrix.
        """
        self.x = 0
        self.y = 0
        self.theta = 0

        angle = np.radians(30) #Initilize the camera to tilt 30 degree down from the mounting position
        self.cam_R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])