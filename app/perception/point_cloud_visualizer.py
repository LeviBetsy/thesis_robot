import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

class PointCloudVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Real-Time Point Cloud", width=1200, height=1000)
        self.pcd = o3d.geometry.PointCloud()
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.axes)
        self.cmap = plt.get_cmap("turbo")

    def update(self, pcd: np.ndarray):
        if pcd.size > 0:
            z_values = pcd[:, 1]
            z_min, z_max = np.min(z_values), np.max(z_values)
            z_norm = (z_values - z_min) / (z_max - z_min + 1e-6)
            
            pcd_colors = self.cmap(z_norm)[:, :3]
            self.pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
            self.pcd.points = o3d.utility.Vector3dVector(pcd)
            
            self.vis.update_geometry(self.pcd)
            
        self.vis.poll_events()
        self.vis.update_renderer()
        
    def close(self):
        self.vis.destroy_window()