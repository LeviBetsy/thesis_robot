import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
import time
import cv2 as cv
import math
from scipy.optimize import curve_fit

class FloorScaleCorrection:
    def __init__(self, gt_z_file):
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Goes up two levels from scripts/
        self.floor_pixels, self.z_real = self.read_gt_floor_z(gt_z_file)
        self.a, self.b, self.c = 0, 0, 0 

        floor_pixels_arr = np.array(self.floor_pixels)
        self.x_coords = np.round(floor_pixels_arr[:, 0]).astype(int) #x coordinates of sampled floor pixels
        self.y_coords = np.round(floor_pixels_arr[:, 1]).astype(int) #y coordinates of sampled floor pixles
        self.inv_z_points = 1.0 / np.array(self.z_real) # inverse groudtruth metric (==relative) depth of samppled floor pixels

        #filter relative depth smaller than min_calibrated_rel
        self.min_calibrated_rel = 0 #default min_calibrated_rel, further away points == smaller relative distance
    

    def read_gt_floor_z(self, gt_z_file):
        # Reading ground truth z for points on the floor 
        config_dir = self.project_root / "config"
        z_file = Path(str(config_dir / f"{gt_z_file}.npz"))
        data = np.load(z_file)

        #Extract the arrays using the keys they were saved with
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

    '''
    depth_map_file is most recent relative reading of the camera
    '''
    def scale_correction(self, d_rel, plot=False, plot_file=""): 
        drel_points = d_rel[self.y_coords, self.x_coords]
        self.min_calibrated_rel = drel_points.min()
        self.max_calibrated_rel = drel_points.max()


        # #****************** Ridge Regression ****************
        # ones_column = np.ones_like(drel_points)
        # A = np.column_stack((drel_points, ones_column))
        # b = self.inv_z_points

        # # # With cross validation
        # # lambdas_to_test = np.logspace(-6, 6, 13) 
        # # # cv=5 means 5-fold cross-validation
        # # ridge_cv = RidgeCV(alphas=lambdas_to_test, fit_intercept=False, cv=5)
        # # ridge_cv.fit(A, b)
        # # optimal_lambda = ridge_cv.alpha_
        # # x = ridge_cv.coef_
        # # print(f"Optimal Lambda: {optimal_lambda}")

        # # Without cross validation
        # ridge = Ridge(alpha=0.001, fit_intercept=False) #without cross_validation
        # ridge.fit(A,b)
        # x=ridge.coef_
        # self.s1, self.s2 = float(x[0].item()), float(x[1].item())
        
        #********************** Exponential *****************
        def exp_model(x, a, b, c):
            """
            Equation: y = a * e^(b * (x - offset)) + c
            We use an x_offset to prevent numerical overflow. 
            Calculating e^20 directly can cause optimization to fail.
            """
            x_offset = np.min(x) 
            return a * np.exp(b * (x - x_offset)) + c
        
        coefficients, covariance = curve_fit(
            exp_model, 
            drel_points, 
            self.inv_z_points, 
            maxfev=5000 # Increases maximum iterations in case it struggles to converge
        )

        self.a, self.b, self.c = coefficients
        # #********************** Polynomial Regression *****************
        # # drel_points is x, inv_z_points is y, 2 is the polynomial degree
        # coefficients = np.polyfit(drel_points, self.inv_z_points, 2)
        
        # # polyfit returns coefficients in descending order of power: [a, b, c]
        # self.a = float(coefficients[0])
        # self.b = float(coefficients[1])
        # self.c = float(coefficients[2])
        # print(f"[Polynomial Params] a: {self.a:.6f} | b: {self.b:.6f} | c: {self.c:.6f}")
        # #**************************************************************
        #Plot
        if plot:
            self.plot_scale_correction(plot_file, drel_points, self.inv_z_points)

    
    def relative_to_metric(self, d_rel: np.array) -> np.array:
        result = np.full_like(d_rel, -1.0, dtype=float)
        
        valid_maskk = valid_mask = d_rel >= self.min_calibrated_rel
        valid_vals = d_rel[valid_mask]


        result[valid_mask] = 1.0 / (self.a* math.e**(self.b*(valid_vals - self.min_calibrated_rel)) + self.c)


        # result[self.y_coords, self.x_coords] = 1.0/ (self.a*math.e**(self.b*(d_rel[self.y_coords, self.x_coords]-self.min_calibrated_rel)) + self.c)
        
        return result
    
if __name__ == "__main__":
    fsc = FloorScaleCorrection("z_real_undistort_ref6")
