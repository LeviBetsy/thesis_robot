import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2

from app.perception.mde_depth import MDE_Depth
from app.perception.point_cloud_visualizer import PointCloudVisualizer
from app.util.config import Config

config = Config.load("./config/robot_config.json")

def calibration_pcd_inspect(path, delete_ground=False):
    """Runs MDE on one image, saves + shows the floor scale calibration plot, and visualizes the resulting pcd."""
    mde_depth = MDE_Depth(config)
    frame = cv2.imread(path)

    # Single inference: the same calibration fit drives both the plot and the pcd
    rel_depth = mde_depth.process_rel(frame)
    plot_file = f"fsc_{os.path.splitext(os.path.basename(path))[0]}.png"
    mde_depth.fsc.scale_calibration(rel_depth, plot=True, plot_file=plot_file, plot_line=True)
    if mde_depth.fsc.slope is None:
        print("Scale calibration failed on this image (no valid floor fit); pcd will be invalid.")
    print(f"fsc slope {mde_depth.fsc.slope} | intercept {mde_depth.fsc.intercept}")

    # Same steps as MDE_Depth.frame_to_pcd, minus the second inference
    metric_depth = mde_depth.fsc.relative_to_metric(rel_depth)
    pcd_cc = mde_depth.pcd_processor.proj_pcd_cc(metric_depth)
    pcd = mde_depth.pcd_processor.pcd_camera_to_robot(pcd_cc)
    if delete_ground:
        pcd = pcd[pcd[:, 2] >= mde_depth.ground_Z]

    visualizer = PointCloudVisualizer()
    try:
        while True: #Ctrl+C to exit
            visualizer.update(pcd)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        visualizer.close()


if __name__ == "__main__":
    calibration_pcd_inspect("./data/ref27.jpg")
    # calibration_pcd_inspect("./data/ref23.jpg", delete_ground=True)
