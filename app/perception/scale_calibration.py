import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import cv2 

class FloorScaleCalibration:
    def __init__(self, gt_z_file_path, group_n):
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Goes up two levels from scripts/
        self.floor_pixels, z_real = self.read_gt_floor_z(gt_z_file_path)
        inv_z_points = 1.0 / np.array(z_real) # inverse groudtruth metric (==relative) depth of samppled floor pixels

        stacked_floor = np.column_stack((self.floor_pixels, inv_z_points)) #shape (N,3)
        sort_idx = np.argsort(stacked_floor[:, 1])
        sorted_floor_px = stacked_floor[sort_idx]
        # pixel_blocks[0] is the row furthest from the camera and opposite for pixel_blocks[N-1]
        # Only the two closest rows (pixel_blocks[-2:]) are used for calibration
        self.group_n = group_n # number of checkerboard rows
        self.pixel_blocks = np.vsplit(sorted_floor_px, self.group_n) #returns a lst of N nparray

        self.reset_calibration()

        

    def reset_calibration(self):
        self.slope = None # single linear fit: z^-1 = slope * d_rel + intercept
        self.intercept = None

    def read_gt_floor_z(self, gt_z_file_path) -> tuple[np.ndarray, np.ndarray]:
        # Reading ground truth z for points on the floor 
        config_dir = self.project_root / "config"
        z_file = Path(str(config_dir / gt_z_file_path))
        data = np.load(z_file)
        return data['cornersOrg'], data['z_real'].squeeze() #squeze zreal because it is shape (N, 1)

    def plot_scale_calibration(self, plot_fpath, blocks, plot_line=True):
        plot_dir = self.project_root / "data" / "plot"
        # Ensure the directory exists
        plot_dir.mkdir(parents=True, exist_ok=True) 
        plot_path = Path(str(plot_dir / plot_fpath))
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Extract x and y from the (N, 2) numpy array
        num_blocks = len(blocks)
        cmap = plt.get_cmap('turbo')
        for i, points in enumerate(blocks):
            x_data = points[:, 0]
            y_data = points[:, 1]

            block_color = cmap(i / max(1, num_blocks - 1))
        
            # Scatter plot for the raw data points
            ax.scatter(x_data, y_data, alpha=0.9, color=block_color, edgecolors='none', s=20)

        # Plot the single linear fit across the calibrated d_rel range
        if plot_line and self.slope is not None:
            all_rel = np.vstack(blocks)[:, 0]
            x_line = np.array([all_rel.min(), all_rel.max()])
            y_line = self.slope * x_line + self.intercept
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

    def scale_calibration(self, d_rel, plot=False, plot_file="", plot_line=True):
        self.reset_calibration()


        # Extracting drel for the two closest rows only, turning each pixel block nparray(6,3) into a rel block nparray(6,2)
        rel_blocks = []
        for coords in self.pixel_blocks[-2:]:
            x_coords, y_coords = coords[:, 0].astype(int), coords[:, 1].astype(int)
            inv_z = coords[:, 2] #shape (6,)
            rel_depth = d_rel[y_coords, x_coords] #shape (6,)
            rel_blocks.append(np.column_stack((rel_depth, inv_z)))
        far_block, near_block = rel_blocks # near_block is the closest row, far_block is the row behind it

        # Obstacle check: a far-row point is floor only if it is further (smaller d_rel) than the near-row point
        keep_mask = far_block[:, 0] < near_block[:, 0]
        filtered_blocks = [far_block[keep_mask], near_block]


        # Single linear fit over the surviving points of the two closest rows
        data_points = np.vstack(filtered_blocks) #shape (M, 2): (d_rel, z^-1)
        if len(filtered_blocks[0]) > 0: # far row needs a surviving point, near row alone spans ~one depth
            ridge_model = Ridge(alpha=0.001)
            ridge_model.fit(data_points[:, 0].reshape(-1, 1), data_points[:, 1])
            self.slope = ridge_model.coef_[0]
            self.intercept = ridge_model.intercept_
        else: #all calibration is invalid (likely because too close to obstacle)
            print("all calibration invalid (likely due to obstacle being too close)")
        
        #Plot
        if plot:
            self.plot_scale_calibration(plot_file, filtered_blocks, plot_line=plot_line)

    
    def relative_to_metric(self, d_rel: np.array) -> np.array:
        if self.slope is None: # no valid calibration this frame
            return np.full_like(d_rel, -1.0, dtype=float)
        return 1 / (self.slope * d_rel + self.intercept) # every pixel, no calibrated-range mask
    

