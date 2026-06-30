import numpy as np
import open3d as o3d
import sys
from pathlib import Path

def visualize_point_cloud(file_path):
    # 1. Load the .npz file
    try:
        data = np.load(file_path)
        # Extract the N,3 array from the 'points' key
        points = data['points'] 
        print(f"Successfully loaded {points.shape[0]} points from {file_path}.")
    except FileNotFoundError:
        print(f"Error: Could not find file '{file_path}'")
        sys.exit(1)
    except KeyError:
        print(f"Error: The file '{file_path}' does not contain an array named 'points'.")
        print(f"Available keys are: {data.files}")
        sys.exit(1)

    # 2. Initialize the Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()

    # 3. Assign the NumPy array to the PointCloud
    pcd.points = o3d.utility.Vector3dVector(points)

    # 4. Create a coordinate frame for reference (Optional but helpful)
    # X axis = Red, Y axis = Green, Z axis = Blue
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # 5. Launch the Open3D visualization window
    print("Opening Open3D visualizer. Press 'Q' or 'Esc' in the window to close.")
    o3d.visualization.draw_geometries([pcd, axes])

if __name__ == "__main__":
    # Specify the path to your saved point cloud here
    TARGET_FILE = "pcd2.npz"
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    pcd_file = project_root / "data" / "point_cloud" / TARGET_FILE
    
    
    visualize_point_cloud(pcd_file)