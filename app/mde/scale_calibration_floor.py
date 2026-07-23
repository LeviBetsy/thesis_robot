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
    def __init__(self, gt_z_file_path, group_n):
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Goes up two levels from scripts/
        self.floor_pixels, z_real = self.read_gt_floor_z(gt_z_file_path)
        inv_z_points = 1.0 / np.array(z_real) # inverse groudtruth metric (==relative) depth of samppled floor pixels

        stacked_floor = np.column_stack((self.floor_pixels, inv_z_points)) #shape (48,3)
        sort_idx = np.argsort(stacked_floor[:, 1])
        sorted_floor_px = stacked_floor[sort_idx]
        # floor_lst[0] shows the row furthest to the camera and opposite for floor_lst[7] 
        self.group_n = group_n
        self.pixel_blocks = np.vsplit(sorted_floor_px, self.group_n) #returns a lst of 7 nparray

        #filter relative depth smaller than min_calibrated_rel
        self.min_calibrated_rel = 0 #default min_calibrated_rel, further away points == smaller relative distance
        self.max_calibrated_rel = 100

    def read_gt_floor_z(self, gt_z_file_path) -> tuple[np.ndarray, np.ndarray]:
        # Reading ground truth z for points on the floor 
        config_dir = self.project_root / "config"
        z_file = Path(str(config_dir / gt_z_file_path))
        data = np.load(z_file)
        return data['cornersOrg'], data['z_real'].squeeze() #squeze zreal because it is shape (N, 1)

    def plot_scale_calibration(self, plot_fpath, blocks):
        plot_dir = self.project_root / "data" / "plot"
        # Ensure the directory exists
        plot_dir.mkdir(parents=True, exist_ok=True) 
        plot_path = Path(str(plot_dir / plot_fpath))
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract x and y from the (N, 2) numpy array
        num_blocks = len(blocks)
        # 'turbo' or 'gist_rainbow' offer a highly vibrant, full-spectrum range of colors
        cmap = plt.get_cmap('turbo')
        for i, points in enumerate(blocks):
            x_data = points[:, 0]
            y_data = points[:, 1]

            block_color = cmap(i / max(1, num_blocks - 1))
        
            # Scatter plot for the raw data points
            ax.scatter(x_data, y_data, alpha=0.9, color=block_color, edgecolors='none', s=20)

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
            ax.plot(x_line, y_line, color='red', linewidth=2, alpha = 0.5)

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
            rel_blocks.append(np.column_stack((rel_depth, inv_z)))
            #OH FUCK TODO
            #segment satisfying min_calibrated_depth means you can still get into self.segment_mins

        # Filter to make sure only floor pixels are used for calculation
        filtered_blocks = rel_blocks.copy()
        for i in range(len(filtered_blocks) - 2, -1, -1): #loop from 2nd closest row filtered_blocks[n-2] to furthest row filtered_blocks[0]
            keep_mask = filtered_blocks[i][:,0] < filtered_blocks[i+1][:,0] #further row must have rel_depth less than closer row
            if not keep_mask.all(): #if a single point is not a floor pixel, filter all further point to be invalid (inclusive)
                for k in range(i + 1):
                    filtered_blocks[k] = filtered_blocks[k][keep_mask]
        

        # groupings are tuple of (i, group_i) where group_i is block_i U block_i+1
        groupings = [np.vstack((curr_block, next_block))
            for (curr_block, next_block) in zip(filtered_blocks[:-1], filtered_blocks[1:])
            if len(curr_block) > 0 and len(next_block) > 0
        ] #grouping first block with next block, groupingmust be made of 2 blocks that are not empty or it is not valid
        n_segment = len(groupings)
        if n_segment > 0:
            self.min_calibrated_rel = groupings[0][:, 0].min() #grouping 0 is furthest points => min d_rel
            self.max_calibrated_rel = groupings[-1][:, 0].max() #grouping n is closest points => max d_rel
        else: #all calibration is invalid (likely because too close to obstacle)
            print("all calibration invalid (likely due to obstacle being too close)")
            self.min_calibrated_rel = 100
            self.max_calibrated_rel = 0
        
        self.fits = np.zeros((n_segment, 2))
        self.segment_mins = [0]*(n_segment)
        
        for i in range(n_segment):
            data_points = groupings[i]
            self.segment_mins[i] = data_points[:,0].min()
            X = data_points[:, 0].reshape(-1, 1)
            y = data_points[:, 1]
            ridge_model = Ridge(alpha=0.001) 
            ridge_model.fit(X, y)
            self.fits[i] = [ridge_model.coef_[0], ridge_model.intercept_] # change the fit
        
        #Plot
        if plot:
            self.plot_scale_calibration(plot_file, filtered_blocks)

    
    def relative_to_metric(self, d_rel: np.array, extrapolate= False) -> np.array:
        if extrapolate:
            valid_mask = (d_rel >= self.min_calibrated_rel)
        else: 
            valid_mask = (d_rel >= self.min_calibrated_rel) & (d_rel <= self.max_calibrated_rel)
        valid_vals = d_rel[valid_mask] # np array of points within calibration

        fit_idx = np.searchsorted(self.segment_mins, valid_vals, side='right') - 1
        slopes = self.fits[fit_idx, 0]
        intercepts = self.fits[fit_idx, 1]

        result = np.full_like(d_rel, -1.0, dtype=float)
        result[valid_mask] = 1 / (slopes * valid_vals + intercepts)
        return result
    
    def annotate_floor_pixels(self, frame: np.ndarray, out_fpath: str, puttext=True):
        ''' Takes in an image and annotate the floor pixels on that image 
        and save the annotated image to data/floor_verificaiton/out_fpath'''
        save_dir = self.project_root / "data" / "floor_verification"
        # Ensure the directory exists
        save_path = Path(str(save_dir / out_fpath))

        annotated_img = frame.copy()
    
        # Define cosmetic parameters for the markers
        circle_color = (0, 0, 255)     # Red for the circle
        text_color = (0, 255, 0)       # Green for the text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3
        thickness = 1
        
        for block in self.pixel_blocks:
            for x, y, z in block:
                center = (int(x), int(y))
                cv2.circle(annotated_img, center, radius=5, color=circle_color, thickness=-1)
                if puttext:
                    text = f"{(1/z):.2f}"
                    text_position = (center[0] + 10, center[1] + 5)
                    cv2.putText(annotated_img, text, text_position, font, font_scale, text_color, thickness, cv2.LINE_AA)
            
        success = cv2.imwrite(save_path, annotated_img)
        print(save_path)
        return annotated_img

    
if __name__ == "__main__":
    fsc = FloorScaleCorrection("z_real.npz")
