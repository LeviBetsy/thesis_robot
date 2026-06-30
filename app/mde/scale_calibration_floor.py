import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
import piecewise_regression
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

        stacked_floor = np.column_stack((floor_pixels, inv_z_points)) #shape (48,3)
        sort_idx = np.argsort(stacked_floor[:, 1])
        sorted_floor_px = stacked_floor[sort_idx]
        # floor_lst[0] shows the row furthest to the camera and opposite for floor_lst[7] 
        self.group_n = 8
        self.pixel_blocks = np.vsplit(sorted_floor_px, self.group_n) #returns a lst

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

    def plot_scale_calibration(self, plot_file, points):
        plot_dir = self.project_root / "data" / "plot"
        # Ensure the directory exists
        plot_dir.mkdir(parents=True, exist_ok=True) 
        plot_path = Path(str(plot_dir / f"{plot_file}.jpg"))
        
        # Extract x and y from the (N, 2) numpy array
        x_data = points[:, 0]
        y_data = points[:, 1]
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot for the raw data points
        ax.scatter(x_data, y_data, alpha=0.6, color='blue', edgecolors='none', s=20)

        # Plot piecewise linear segments
        for i, (a, b) in enumerate(self.fits):
            x_start = self.segment_mins[i]
            
            # The last segment ends at max_calibrated_rel, others end at the next segment's min
            if i < len(self.fits) - 1:
                x_end = self.segment_mins[i + 1]
            else:
                x_end = self.max_calibrated_rel
                
            # Since it's a straight line, we only need the start and end coordinates
            x_line = np.array([x_start, x_end])
            y_line = a * x_line + b
            
            # Plot the segment
            ax.plot(x_line, y_line, color='red', linewidth=2)

        # Format axes using LaTeX notation
        ax.set_xlabel(r'Relative Depth ($d_{rel}$)', fontsize=12)
        ax.set_ylabel(r'Inverse Real Depth ($z^{-1}$)', fontsize=12)
        ax.set_title(r'Data Points: Inverse Depth $z^{-1}$ vs Relative Depth $d_{rel}$', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save and close the figure to prevent memory leaks
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close(fig) 
        
        print(f"Plot successfully saved to {plot_path}")

    def scale_calibration(self, d_rel, plot=False, plot_file=""): 
        # Extracting drel, turning self.pixel_blocks lst(nparray(6,3)) into rel_blocks lst(nparray(6,2))
        rel_blocks = []
        for i, coords in enumerate(self.pixel_blocks):
            x_coords, y_coords = coords[:, 0].astype(int), coords[:, 1].astype(int)
            inv_z = coords[:, 2] #shape (6,)
            rel_depth = d_rel[y_coords, x_coords] #shape (6,)
            if i < self.group_n - 1:
                self.segment_mins[i] = rel_depth.min()
            rel_blocks.append(np.column_stack((rel_depth, inv_z)))

        # Intial Filter to make sure only floor pixels are used for calculation
        # ..... filters ..... TODO
        # for each of the 7 segment there is a tuple of (i, nparray).
        # np array of shape (12,2)
        # after the filtering, there might be N segments N <= 7
        # each with tuple (i,nparray) have an np array of (M,2) with M <= 12
        # but we would get min_calibrated_rel for later mask
        # thus self.segment_mins though not updated, does not matter
        # because we will filter it before even looking up with self.segment_mins
        # TODO: filter less than 2 groupings
        # ********************************************
        filtered_blocks = rel_blocks #TODO: 
        

        groupings = [
            (i, np.vstack((curr_block, next_block)))
            for i, (curr_block, next_block) in enumerate(zip(filtered_blocks[:-1], filtered_blocks[1:]))
        ] #grouping first block with next block
        groupings = [(i, group) for (i,group) in groupings if group.shape[0] >= 2] #keep groups with > 1 datapoints for regression
        self.min_calibrated_rel = groupings[0][1][:, 0].min() #grouping 0 is furthest points => min d_rel
        self.max_calibrated_rel = groupings[-1][1][:, 0].max() #grouping n is closest points => max d_rel

        
        for i in range(len(groupings)):
            idx, data_points = groupings[i][0], groupings[i][1]
            X = data_points[:, 0].reshape(-1, 1)
            y = data_points[:, 1]
            ridge_model = Ridge(alpha=0.001) # alpha=0 is standard linear regression. Higher alpha = more penalty.
            ridge_model.fit(X, y)
            self.fits[idx] = ridge_model.coef_[0], ridge_model.intercept_
        
        #Plot
        if plot:
            self.plot_scale_calibration(plot_file, np.vstack(rel_blocks))

    
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
