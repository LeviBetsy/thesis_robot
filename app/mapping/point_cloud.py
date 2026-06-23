import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from pathlib import Path
from app.module.camera import Camera

class PointCloudProcessor:
    def __init__(self, camera: Camera):
        self.camera = camera

        # Pre-compute and cache the full 2D pixel coordinate grids
        self.u, self.v = np.meshgrid(np.arange(camera.w), np.arange(camera.h))

        # ---------------------------------------------------------
        # K = np.array([[fx,  0, cx],
        #                    [ 0, fy, cy],
        #                    [ 0,  0,  1]])
        # ---------------------------------------------------------
        
        # Extract focal lengths (in pixels)
        intrinsic = self.camera.K
        self.fx = intrinsic[0, 0]
        self.fy = intrinsic[1, 1]
        
        # Extract principal point (optical center)
        self.cx = intrinsic[0, 2]
        self.cy = intrinsic[1, 2]
    

    def proj_point_cloud_cc(self, Z):
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
        
        ret = np.stack((X, Z_valid, Y), axis=-1)
        return ret
    
    
