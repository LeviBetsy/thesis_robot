import numpy as np
from pathlib import Path

class InversePerspectiveProjection:
    def __init__(self, height, width, cam_calib_fname):
        self.h = height
        self.w = width

        # Pre-compute and cache the full 2D pixel coordinate grids
        self.u, self.v = np.meshgrid(np.arange(self.w), np.arange(self.h))

        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Adjust levels to match your directory structure
        config_dir = self.project_root / "config"
        config_path = str(config_dir / cam_calib_fname)
        
        # Load calibration data once during initialization
        try:
            with np.load(config_path) as calib_data:
                # Adjust string keys if your variables were saved under different names
                self.K = np.array(calib_data['camera_matrix'], dtype=np.float32)
                self.D = np.array(calib_data['distortion_coefficients'], dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration configuration file not found at: {config_path}")
        except KeyError as e:
            raise KeyError(f"Could not find expected key {e} in the .npz file. Check your calibration script keys.")

        # ---------------------------------------------------------
        # self.K = np.array([[fx,  0, cx],
        #                    [ 0, fy, cy],
        #                    [ 0,  0,  1]])
        # ---------------------------------------------------------
        
        # Extract focal lengths (in pixels)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        
        # Extract principal point (optical center)
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
    

    def proj_point_cloud_cc(self, Z, flatten=True):
        """
        Converts a 2D NumPy array of metric depth (Z) into a point cloud in CAMERA COORDINATE
        
        Args:
            Z (np.ndarray): A 2D array of metric depths (shape: H x W).
            flatten (bool): If True, returns a flat list of 3D points (shape: N x 3).
                            If False, returns a spatial 3D array (shape: H x W x 3).
            
        Returns:
            point_cloud (np.ndarray): Array of (X, Y, Z) coordinates in meters.
        """
        # Calculate X and Y coordinates (Z is already provided)
        Z = np.full((480, 640), 0.5)
        X = (self.u - self.cx) * Z / self.fx
        Y = (self.v - self.cy) * Z / self.fy
        
        # Stack X, Y, and Z along the last axis to create an (H, W, 3) array
        point_cloud = np.stack((X, Y, Z), axis=-1)
        print(point_cloud.shape)
        
        # Commonly, point clouds are processed as an (N, 3) array rather than a 2D image grid
        if flatten:
            point_cloud = point_cloud.reshape(-1, 3)
            
        return point_cloud
    
    
