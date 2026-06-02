import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
import time

class FloorScaleCorrection:
    def __init__(self, gt_z_file):
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Goes up two levels from scripts/
        self.floor_pixels, self.z_real = self.read_gt_floor_z(gt_z_file)

    def read_depthfile(self, depth_map_file):
        # Reading relative depth map from .bin file
        depth_map_dir = self.project_root / "data" / "depth_map"
        depth_map = Path(str(depth_map_dir / depth_map_file))

        if depth_map.is_file():
            width, height = 640,480
            depth_data = np.fromfile(depth_map, dtype=np.float32).reshape((height, width))
            return depth_data
            # Safe to read/process the file here
        else:
            raise Exception(f"File {depth_map} does not exist")

    def read_gt_floor_z(self, gt_z_file):
        # Reading ground truth z for points on the floor 
        config_dir = self.project_root / "config"
        z_file = Path(str(config_dir / f"{gt_z_file}.npz"))
        data = np.load(z_file)
        #Extract the arrays using the keys they were saved with
        return data['cornersOrg'], data['z_real']

    def plot_scale_correction(self, plot_file, s1, s2, A, b):
        plot_dir = self.project_root / "data" / "plot"
        plot_path = Path(str(plot_dir / f"{plot_file}.png"))
        # Create figure and axis using subplots for a clean layout
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot for the raw data points only
        ax.scatter(A[:,0], b, alpha=0.6, color='blue', edgecolors='none', s=20)

        # Linear fit line
        x_line = np.linspace(A[:, 0].min(), A[:, 0].max(), 100)
        y_line = s1 * x_line + s2
        ax.plot(x_line, y_line, color='red', linewidth=2, label=rf'Fit: $z^{{-1}} = {s1:.4f} \cdot d_{{rel}} + {s2:.4f}$')
        
        # Format axes using LaTeX notation
        ax.set_xlabel(r'Relative Depth ($d_{rel}$)', fontsize=12)
        ax.set_ylabel(r'Inverse Real Depth ($z^{-1}$)', fontsize=12)
        ax.set_title(r'Data Points: Inverse Depth $z^{-1}$ vs Relative Depth $d_{rel}$', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save the visualization to the specified file path
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot successfully saved to {plot_path}")

    '''
    output the s1, s2 arguments for the linear fit to convert relative depth to absolute depth
    depth_map_file is most recent relative reading of the camera
    '''
    def scale_correction_ridge_reg(self, depthmap_file, plot=False, plot_file="") -> tuple[float, float]: 
        # Reading Data File
        start_time = time.perf_counter()
        rel_depth_data = self.read_depthfile(f"{depthmap_file}.bin")
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"--- Execution Time For Loading Files: {elapsed_time:.6f} seconds ---")

        # *********************** Data Collection **********************
        floor_pixels_arr = np.array(self.floor_pixels)
        x_coords = np.round(floor_pixels_arr[:, 0]).astype(int)
        y_coords = np.round(floor_pixels_arr[:, 1]).astype(int)

        drel_points = rel_depth_data[y_coords, x_coords]
        inv_z_points = 1.0 / np.array(self.z_real)
        #**************************************************************

        #********************** Ridge Regression **********************
        start_time = time.perf_counter()

        #Following CeRlp paper for notation
        ones_column = np.ones_like(drel_points)
        A = np.column_stack((drel_points, ones_column))
        b = inv_z_points

        # # With cross validation
        # lambdas_to_test = np.logspace(-6, 6, 13) 
        # # cv=5 means 5-fold cross-validation
        # ridge_cv = RidgeCV(alphas=lambdas_to_test, fit_intercept=False, cv=5)
        # ridge_cv.fit(A, b)
        # optimal_lambda = ridge_cv.alpha_
        # x = ridge_cv.coef_
        # print(f"Optimal Lambda: {optimal_lambda}")

        # Without cross validation
        ridge = Ridge(alpha=0.001, fit_intercept=False) #without cross_validation
        ridge.fit(A,b)
        x=ridge.coef_
    
        s1, s2 = x[0],x[1]
        print(f"Coefficients (s1, s2): {x}")
        print(f"--- Execution Time For Ridge Regression: {elapsed_time:.6f} seconds ---")
        #**************************************************************

        #Plot
        if plot:
            self.plot_scale_correction(plot_file, s1, s2, A, b)
        return s1, s2
        
if __name__=="__main__":
    fsc = FloorScaleCorrection("z_real_ref6")
    s1, s2 = fsc.scale_correction_ridge_reg("DAV2_cube60_depthmap", False, "DAV2_cube_60cm_plot")