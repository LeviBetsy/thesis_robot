import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
import time
import cv2 
import math
from scipy.optimize import curve_fit

class FloorScaleCorrection:
    def __init__(self, gt_z_file_path):
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Goes up two levels from scripts/
        floor_pixels, z_real = self.read_gt_floor_z(gt_z_file_path)
        inv_z_points = 1.0 / np.array(z_real) # inverse groudtruth metric (==relative) depth of samppled floor pixels
        stacked_floor = np.column_stack((floor_pixels, inv_z_points))

        self.group_n = 8
        sort_idx = np.argsort(stacked_floor[:, 1])
        sorted_floor_px = stacked_floor[sort_idx]
        #floor_lst is list len(8) of np.array of shape (6,3) for each floor pixels
        # floor_lst[0] is np.array(6,(x,y,inv_z))
        # floor_lst[0] shows the row closest to the camera and opposite for floor_lst[7] 
        self.floor_lst = np.split(sorted_floor_px, self.group_n, axis=0)[::-1] 

        #filter relative depth smaller than min_calibrated_rel
        self.min_calibrated_rel = 0 #default min_calibrated_rel, further away points == smaller relative distance
        self.max_calibrated_rel = 100

        self.fits = [(0,0)]*(self.group_n - 1) # definition for linear fit for each segment
        self.segment_mins = [0]*(self.group_n - 1) # the smallest d_rel bound of each segment

    def read_gt_floor_z(self, gt_z_file_path) -> tuple[np.ndarray, np.ndarray]:
        # Reading ground truth z for points on the floor 
        config_dir = self.project_root / "config"
        z_file = Path(str(config_dir / gt_z_file_path))
        data = np.load(z_file)
        return data['cornersOrg'], data['z_real'].squeeze() #squeze zreal because it is shape (N, 1)

    def plot_scale_correction(self, plot_file, A, b):
        plot_dir = self.project_root / "data" / "plot"
        plot_path = Path(str(plot_dir / f"{plot_file}.jpg"))
        # Create figure and axis using subplots for a clean layout
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot for the raw data points only
        ax.scatter(A, b, alpha=0.6, color='blue', edgecolors='none', s=20)
        x_line = np.linspace(A.min(), A.max(), 100)

        # y_line = self.s1 * x_line + self.s2
        y_line = self.a * math.e**(self.b * (x_line - self.min_calibrated_rel)) + self.c
        # y_line = self.a * (x_line**2) + self.b * x_line + self.c        

        # Plotting
        ax.plot(x_line, y_line, color='red', linewidth=2)
        # Format axes using LaTeX notation
        ax.set_xlabel(r'Relative Depth ($d_{rel}$)', fontsize=12)
        ax.set_ylabel(r'Inverse Real Depth ($z^{-1}$)', fontsize=12)
        ax.set_title(r'Data Points: Inverse Depth $z^{-1}$ vs Relative Depth $d_{rel}$', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save the visualization to the specified file path
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot successfully saved to {plot_path}")

    def scale_calibration(self, d_rel, plot=False, plot_file=""): 
        # Perform grouping of cluster close to each other for segmeneted regression
        groupings = []
        for i in range(self.group_n): #group_n groups ==> group_n - 1 linear line
            x_coords = self.floor_lst[i][:, 0].astype(int)
            y_coords = self.floor_lst[i][:, 1].astype(int)
            inv_z = self.floor_lst[i][:, 2] #shape (6,)
            rel_depth = d_rel[y_coords, x_coords] #shape (6,)
            rel_inv = np.column_stack((rel_depth, inv_z)) #shape (6,2)
            if i < self.group_n - 1:
                self.segment_mins[i] = rel_depth.min()
                groupings.append(rel_inv) # prevent appending trailing lonely group
            if i > 0:
                groupings[i-1] = np.vstack((groupings[i-1], rel_inv)) # this group is stacked with previous group
        groupings = [(i, groupings[i]) for i in range(self.group_n - 1)]

        # Intial Filter to make sure only floor pixels are used for calculation
        # ..... filters ..... TODO
        # for each of the 7 segment there is a tuple of (i, nparray).
        # np array of shape (12,2)
        # after the filtering, there might be N segments N <= 7
        # each with tuple (i,nparray) have an np array of (M,2) with M <= 12
        # but we would get min_calibrated_rel for later mask
        # thus self.segment_mins though not updated, does not matter
        # because we will filter it before even looking up with self.segment_mins
        # ********************************************
        self.min_calibrated_rel = groupings[0][1][:, 0].max() #grouping 0 is closest points => max d_rel
        self.max_calibrated_rel = groupings[-1][1][:, 0].min() #grouping n is furthest points => min d_rel

        for i in range(len(groupings)):
            X = groupings[i][1][:, 0].reshape(-1, 1)
            y = groupings[i][1][:, 1]
            ridge_model = Ridge(alpha=1.0) # alpha=0 is standard linear regression. Higher alpha = more penalty.
            ridge_model.fit(X, y)
            self.fits[groupings[i][0]] = ridge_model.coef_[0], ridge_model.intercept_
        
        # #Plot
        # if plot:
        #     self.plot_scale_correction(plot_file, valid_drel, self.inv_z_points)

    
    def relative_to_metric(self, d_rel: np.array) -> np.array:
        # TODO: not done
        valid_mask = (d_rel >= self.min_calibrated_rel) & (d_rel <= self.max_calibrated_rel)
        valid_vals = d_rel[valid_mask]

        matching_fit = np.searchsorted(self.segments, valid_vals, side='right') - 1
    
        result = np.full_like(d_rel, -1.0, dtype=float)
        
        valid_mask = (d_rel >= self.min_calibrated_rel) & (d_rel <= self.max_calibrated_rel)
        valid_vals = d_rel[valid_mask]
        result[valid_mask] = 1.0 / (self.a* math.e**(self.b*(valid_vals - self.min_calibrated_rel)) + self.c)
        
        return result
    
if __name__ == "__main__":
    fsc = FloorScaleCorrection("z_real.npz")
