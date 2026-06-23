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
    
    
