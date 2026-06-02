import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def read_depthfile(depth_map_file):
    # Intialize project_root
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/

    # Reading relative depth map from .bin file
    depth_map_dir = project_root / "data" / "depth_map"
    depth_map = Path(str(depth_map_dir / depth_map_file))

    if depth_map.is_file():
        width, height = 640,480
        depth_data = np.fromfile(depth_map, dtype=np.float32).reshape((height, width))
        return depth_data
        # Safe to read/process the file here
    else:
        raise Exception(f"File {depth_map} does not exist")

def read_ground_truth_z(gt_z_file):
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/

    # Reading ground truth z for points on the floor 
    config_dir = project_root / "config"
    z_file = Path(str(config_dir / gt_z_file))
    data = np.load(z_file)

# 2. Extract the arrays using the keys they were saved with
    return data['cornersOrg'], data['z_real']
    return None, None

'''
correct the scale for mde result
relative -> metric depth conversion
depth_map_file is most recent relative reading of the camera
'''
def relative_to_metric(depth_map_file, gt_z_file, plot=False, plot_file=""):
    # Intialize project_root
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts

    rel_depth_data = read_depthfile(depth_map_file)
    floor_pixels, z_real = read_ground_truth_z(gt_z_file)

    # *********************** Data Collection **********************
    # Lists to accumulate all data points
    drel_points = []
    inv_z_points = []
    for i, (x, y) in enumerate(floor_pixels): 
        real_depth = z_real[i]
        drel = rel_depth_data[round(y)][round(x)]
        
        # Append the calculated values for plotting
        drel_points.append(drel)
        inv_z_points.append(1.0 / real_depth)
        
        if i < 2:
            print(f"x: {x}, y: {y}")
            print(f"z^-1: {1/real_depth}")
            print(f"drel: {drel}")
    #**************************************************************

    #*************************** Plot *****************************
    if plot:
        plot_dir = project_root / "data" / "plot"
        plot_path = Path(str(plot_dir / plot_file))
        # Create figure and axis using subplots for a clean layout
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot for the raw data points only
        ax.scatter(drel_points, inv_z_points, alpha=0.6, color='blue', edgecolors='none', s=20)
        
        # Format axes using LaTeX notation
        ax.set_xlabel(r'Relative Depth ($d_{rel}$)', fontsize=12)
        ax.set_ylabel(r'Inverse Real Depth ($z^{-1}$)', fontsize=12)
        ax.set_title(r'Data Points: Inverse Depth $z^{-1}$ vs Relative Depth $d_{rel}$', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save the visualization to the specified file path
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"Plot successfully saved to {plot_path}")

    #**************************************************************
    
    #********************** Ridge Regression **********************

    #**************************************************************
    
if __name__=="__main__":
    relative_to_metric("new_depth_map.bin", "z_real_ref6.npz", True, "scdepth_cube_60cm_plot.png")