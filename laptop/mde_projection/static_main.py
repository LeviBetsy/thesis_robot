import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# import cv2
import numpy as np
import open3d as o3d

# from scripts.mde.DAV2_pth import DepthAnythingPredictor
from app.mde.scale_calibration_floor import FloorScaleCorrection
from app.module.robot import Robot
from app.mapping.point_cloud import PointCloudProcessor


def main(imshow=False):
    # Initialize the depth model (will auto-select mps, cuda, or cpu)
    print("Loading Depth Anything V2 model...")
    # predictor = DepthAnythingPredictor()

    # Initialize Robot
    robot = Robot("fisheye_calib.npz")
    fsc = FloorScaleCorrection("z_real.npz")
    pcd_processor = PointCloudProcessor(robot)
    pcd = o3d.geometry.PointCloud()

    # pcd = o3d.geometry.PointCloud() #Visualizer
    # frame = cv2.imread("./data/test/test_point_cloud.png")
    

    # rel_depth_map = predictor.model.infer_image(frame)

    rel_depth_map = np.load("data/test/rel_depth_test.npz")['infer']


    fsc.scale_calibration(rel_depth_map, False)
    metric_map = fsc.relative_to_metric(rel_depth_map)

    point_cloud_cc = pcd_processor.proj_pcd_cc(metric_map, delete_ground=False)
    point_cloud_rc = pcd_processor.pcd_camera_to_robot(point_cloud_cc)

    pcd.points = o3d.utility.Vector3dVector(point_cloud_rc)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axes])

if __name__ == "__main__":
    main(imshow=True)