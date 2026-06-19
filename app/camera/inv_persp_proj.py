import numpy as np

class InversePerspectiveProjection:
    def __init__(self, height, width):
        self.h = height
        self.w = width
        # Pre-compute and cache the full 2D pixel coordinate grids
        self.u, self.v = np.meshgrid(np.arange(self.w), np.arange(self.h))

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

    def compute_X(self, Z):
        """
        Converts a 2D NumPy array of metric depth (Z) into a 2D array 
        of X coordinates in the camera coordinate space.
        
        Args:
            Z (np.ndarray): A 2D array of metric depths.
            
        Returns:
            X (np.ndarray): A 2D array of X coordinates in meters.
        """
        # Vectorized calculation for the entire array
        X = (self.u - self.cx) * Z / self.fx
        
        return X