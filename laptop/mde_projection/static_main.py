import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import numpy as np
import open3d as o3d

from scripts.mde.DAV2_pth import DepthAnythingPredictor
from app.mde.scale_calibration_floor import FloorScaleCorrection
from app.module.robot import Robot
from app.mapping.point_cloud import PointCloudProcessor


def main(imshow=False):
    # Initialize the depth model (will auto-select mps, cuda, or cpu)
    print("Loading Depth Anything V2 model...")
    predictor = DepthAnythingPredictor()

    # Initialize Robot
    robot = Robot("fisheye_calib.npz")
    # fsc = FloorScaleCorrection("z_real.npz")
    fsc = FloorScaleCorrection("z_real_ref13.npz", group_n=16)
    pcd_processor = PointCloudProcessor(robot)
    pcd = o3d.geometry.PointCloud()

    # pcd = o3d.geometry.PointCloud() #Visualizer
    frame = cv2.imread("./data/test/ref13.jpg")
    fsc.annotate_floor_pixels(frame, "ref13_anno.jpg")
    

    # rel_depth_map = predictor.infer(frame)
    # # predictor.infer_image_save(frame)
    # # print(rel_depth_map.max())
    # # print(rel_depth_map.min())

    # # colored_image = predictor.colorize(rel_depth_map)
    # # fsc.annotate_floor_pixels(colored_image, "unblocked_DAV2.png")

    # # # rel_depth_map = np.load("data/test/rel_depth_test.npz")['infer']


    # fsc.scale_calibration(rel_depth_map, False)
    # metric_map = fsc.relative_to_metric(rel_depth_map)

    # point_cloud_cc = pcd_processor.proj_pcd_cc(metric_map, delete_ground=False)
    # point_cloud_rc = pcd_processor.pcd_camera_to_robot(point_cloud_cc)

    # pcd.points = o3d.utility.Vector3dVector(point_cloud_rc)
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([pcd, axes])
    # pcd.points = o3d.utility.Vector3dVector(point_cloud_rc)
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # # 2. Use VisualizerWithEditing instead of standard draw_geometries
    # vis = o3d.visualization.VisualizerWithEditing()
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.add_geometry(axes)
    
    # print("Instructions: Shift + Left-Click on a point to select it. Close the window when done.")
    # vis.run()  # The script will pause here while the window is open
    # vis.destroy_window()

    # # 3. Retrieve the indices of the points you clicked
    # picked_indices = vis.get_picked_points()

    # # 4. Print the actual coordinates (including Z) of the selected points
    # points_array = np.asarray(pcd.points)
    # for idx in picked_indices:
    #     x, y, z = points_array[idx]
    #     print(f"Selected Point Index {idx}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")

if __name__ == "__main__":
    main(imshow=True)