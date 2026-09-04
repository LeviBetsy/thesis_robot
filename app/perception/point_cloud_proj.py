import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from app.robot_module.robot import Robot

class PointCloudProjection:
    def __init__(self, robot: Robot):
        self.robot = robot
        camera = self.robot.camera
        
        # Pre-compute and cache the full 2D pixel coordinate grids
        self.u, self.v = np.meshgrid(np.arange(camera.w), np.arange(camera.h))

        self.fx = camera.fx
        self.fy = camera.fy
        self.cx = camera.cx
        self.cy = camera.cy

    def proj_pcd_cc(self, Z):
        """
        Converts a 2D NumPy array of metric depth (Z) into a point cloud in CAMERA COORDINATE
        
        Args:
            Z (np.ndarray): A 2D array of metric depths (shape: H x W).
        Returns:
            point_cloud (np.ndarray): Array of (X, Y, Z) coordinates in meters.
        """
        valid_mask = Z != -1 #masking. Generate a 480,640 of True False depending on Z value there
        Z_valid = Z[valid_mask] # If valid_mask apply then take the value otherwise discard it while also flattening into (N)
        u_valid = self.u[valid_mask]
        v_valid = self.v[valid_mask]

        X = (u_valid - self.cx) * Z_valid / self.fx
        Y = -(v_valid - self.cy) * Z_valid / self.fy #negative because camera space grow downward while pointcloud grow up
        
        ret = np.stack((X, Z_valid, Y), axis=-1) # camera's z is robot coordinate y and vice versa
        return ret

    def pcd_camera_to_robot(self, pcd_cc: np.ndarray) -> np.ndarray:
        pcd_rc = (pcd_cc @ self.robot.cam_R.T) + self.robot.cam_t
        return pcd_rc
    
    def average_floor_z(self, pcd_rc: np.ndarray) -> float:
        return np.mean(pcd_rc[:, 2])