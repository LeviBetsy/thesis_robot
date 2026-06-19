import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Ridge
import time
import cv2 as cv
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
        return data['cornersOrg'], data['z_real'].squeeze()

    def plot_scale_correction(self, plot_file, A, b):
        plot_dir = self.project_root / "data" / "plot"
        plot_path = Path(str(plot_dir / f"{plot_file}.jpg"))
        # Create figure and axis using subplots for a clean layout
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot for the raw data points only
        ax.scatter(A, b, alpha=0.6, color='blue', edgecolors='none', s=20)
        x_line = np.linspace(A.min(), A.max(), 100)
        y_line = self.a * (x_line**2) + self.b * x_line + self.c        

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

        #********************** Polynomial Regression *****************
        # drel_points is x, inv_z_points is y, 2 is the polynomial degree
        coefficients = np.polyfit(drel_points, self.inv_z_points, 2)
        
        # polyfit returns coefficients in descending order of power: [a, b, c]
        self.a = float(coefficients[0])
        self.b = float(coefficients[1])
        self.c = float(coefficients[2])
        #**************************************************************
        #Plot
        if plot:
            self.plot_scale_correction(plot_file, drel_points, self.inv_z_points)
    
    '''
    pixel must be (x,y) where x, y are ints
    return the metrix depth of a pixel in float
    '''
    def predict_metric(self, pixel_rel) -> float:
        return 1.0/(self.a * (pixel_rel ** 2) + self.b * pixel_rel + self.c) #using a polynomial fit now
    
    def annotate_floor_pixels(self, image_file, plot_file="annotated_floor", fit_style="linear"):
        """
        Loads an image, iterates through floor pixels, annotates them with circles, 
        and labels each with the output of predict_metric(x, y).
        """
        ref_dir = self.project_root / "data" / "test"
        cb_path = ref_dir / f"{image_file}.jpg" # adjust extension if using .jpg
        
        if not cb_path.is_file():
            raise FileNotFoundError(f"Image file {cb_path} does not exist.")
            
        # Load image using OpenCV
        img = cv.imread(str(cb_path))
        if img is None:
            raise FileNotFoundError("no file")
        
        floor_pixels_arr = np.array(self.floor_pixels)
        new_points = np.array([[354, 100],[354,150], [354,125], [354,63]])
        floor_pixels_arr = np.vstack((floor_pixels_arr, new_points))
        
        for pixel in floor_pixels_arr:
            # Round coordinates to the nearest integer for pixel mapping
            x = int(np.round(pixel[0]))
            y = int(np.round(pixel[1]))
            
            # Ensure the coordinates actually fall within your image boundaries to prevent crashing
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                # Get the prediction for this specific pixel
                metric_val = self.predict_metric(None, (x, y), fit_style)
                
                # Format the text to 3 decimal places
                text = f"{metric_val:.3f}"
                
                # Draw a small solid dot/circle at the rounded (x, y) location
                # cv uses (B, G, R) color format. (0, 255, 0) is pure green.
                cv.circle(img, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                
                # Annotate the numerical text slightly offset (+5 pixels) from the dot
                cv.putText(
                    img, 
                    text, 
                    (x + 5, y - 5), 
                    fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.3, 
                    color=(0, 0, 255), # Red text
                    thickness=1,
                    lineType=cv.LINE_AA
                )

        # Save the annotated image out to your plots/output directory
        output_dir = self.project_root / "data" / "output"
        output_path = output_dir / f"{plot_file}.jpg"
        
        cv.imwrite(str(output_path), img)
        print(f"Annotated image successfully saved to {output_path}")
    
    def relative_to_metric(self, d_rel) -> np.array:
        return self.predict_metric(d_rel)
