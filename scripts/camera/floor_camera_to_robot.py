'''script used to take the metric floor points found by sample_floor.py (camera frame,
x_real/y_real/z_real in the standard OpenCV camera convention: x right, y down, z forward)
and convert them into robot frame coordinates using PointCloudProjection.pcd_camera_to_robot,
then annotate the result back onto the reference image.
'''

import os
import sys
import cv2
import numpy as np
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

from app.util.config import Config
from app.robot_module.robot import Robot
from app.perception.point_cloud_proj import PointCloudProjection


'''
z_real_file is the .npz file saved by sample_floor.py, holding cornersOrg (pixel coords)
  and x_real/y_real/z_real (camera frame, meters). It must be in project_root/config
ref_img is the file name (with extension) of the reference image the points were sampled
  from. It must be in project_root/data/references
'''
def floor_points_to_robot(z_real_file, ref_img, config_file=None, out_file=""):
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    config = Config.load(config_file)
    robot = Robot(config)
    point_cloud_proj = PointCloudProjection(robot)

    configDir = project_root / "config"
    data = np.load(str(configDir / z_real_file))
    cornersOrg = data['cornersOrg']        # (N, 2) pixel coords
    x_real = data['x_real'].squeeze(-1)    # (N,) camera frame, right
    y_real = data['y_real'].squeeze(-1)    # (N,) camera frame, down
    z_real = data['z_real'].squeeze(-1)    # (N,) camera frame, forward/depth

    # proj_pcd_cc stacks points as (X_right, Z_depth, Y_up), with Y flipped positive-up,
    # before handing them to pcd_camera_to_robot. Match that convention here.
    pcd_cc = np.column_stack((x_real, z_real, -y_real))
    pcd_rc = point_cloud_proj.pcd_camera_to_robot(pcd_cc)  # (N, 3) robot frame: right, forward, up

    referenceDir = project_root / "data" / "references"
    img_path = str(referenceDir / ref_img)
    imgBGR = cv2.imread(img_path)
    if imgBGR is None:
        raise FileNotFoundError(f"Could not read reference image at: {img_path}")

    return cornersOrg, pcd_rc, imgBGR, project_root, out_file


def annotate_robot_coords(cornersOrg, pcd_rc, imgBGR, project_root, out_file=""):
    imgPlot = imgBGR.copy()
    for i in range(len(cornersOrg)):
        px_x = int(cornersOrg[i][0])
        px_y = int(cornersOrg[i][1])

        x, y, z = pcd_rc[i]
        cv2.circle(imgPlot, (px_x, px_y), radius=2, color=(0, 255, 0), thickness=-1)

        text = f"{x:.2f},{y:.2f},{z:.2f}"
        cv2.putText(imgPlot, text, (px_x - 15, px_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)

    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_save_path = str(output_dir / f"robot_coords_{out_file}.jpg")
    cv2.imwrite(img_save_path, imgPlot)
    print(f"Successfully saved robot coordinate map to: {img_save_path}")


if __name__ == "__main__":
    cornersOrg, pcd_rc, imgBGR, project_root, out_file = floor_points_to_robot(
        "z_real_16_rows_xyz.npz", "ref13.jpg", out_file="16_rows_xyz"
    )
    annotate_robot_coords(cornersOrg, pcd_rc, imgBGR, project_root, out_file)
