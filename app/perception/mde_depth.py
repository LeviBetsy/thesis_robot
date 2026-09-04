import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np

from scripts.mde.DAV2_pth import DepthAnythingPredictor
from app.robot_module.robot import Robot
from app.perception.scale_calibration import FloorScaleCalibration
from app.perception.point_cloud_proj import PointCloudProjection

class MDE_Depth:
    def __init__(self, calib_file="fisheye_calib.npz", ref_file="z_real_16group.npz", ground_Z=0.01):
        self.robot = Robot(calib_file)
        self.predictor = DepthAnythingPredictor()
        self.fsc = FloorScaleCalibration(ref_file, group_n=16)
        self.pcd_processor = PointCloudProjection(self.robot)

        self.ground_Z = ground_Z #in meters

    def process_rel(self,frame):
        return self.predictor.infer(frame)

    def process_metric(self, frame):
        self.rel_depth = self.predictor.infer(frame)
        self.fsc.scale_calibration(self.rel_depth)
        metric_depth = self.fsc.relative_to_metric(self.rel_depth, extrapolate=False)
        return metric_depth


    def frame_to_pcd(self, frame, delete_ground=True) -> np.ndarray :
        """Processes a single frame and returns the pointcloud."""
        metric_depth = self.process_metric(frame)
        pcd_cc = self.pcd_processor.proj_pcd_cc(metric_depth)
        pcd_rc = self.pcd_processor.pcd_camera_to_robot(pcd_cc)

        if delete_ground:
            mask = pcd_rc[:, 2] >= self.ground_Z
            pcd_rc = pcd_rc[mask]
        return pcd_rc


    def squish_pcd(self, pcd: np.ndarray) -> np.ndarray:
        """Drops the height column, projecting a robot-frame pointcloud to 2D (x, y)."""
        return pcd[:, :2]


    def pcd_to_ray_casting(self, pcd: np.ndarray, n_rays=16, max_range=0.6) -> np.ndarray:
        """
        Casts a fan of n_rays evenly spaced across the camera's horizontal FOV
        (each ray covering an angular width of fov_x / n_rays), and returns for
        each ray the distance to the nearest point in the squished (x, y) point cloud.
        Rays with no point within max_range report max_range.

        Angle 0 is straight ahead; positive angles are to the robot's right.

        Returns:
            ranges (np.ndarray): shape (n_rays,), distance in meters per ray.
        """
        squished_pcd = self.squish_pcd(pcd)
        ranges = np.full(n_rays, max_range, dtype=np.float32)
        if squished_pcd.shape[0] == 0:
            return ranges

        x, y = squished_pcd[:, 0], squished_pcd[:, 1]

        # squished_pcd is in the robot frame (rotated AND translated by cam_t), but
        # fov_x is a camera-intrinsic quantity centered on the camera, not the robot's
        # origin. Undo just the translation (cam_t[0] is always 0 - camera sits on the
        # robot's centerline) to get the bearing as seen from the camera, so bin
        # boundaries line up with the camera's true angular FOV.
        cam_t_y = self.robot.cam_t[1]
        dir_y = y - cam_t_y
        dir_range = np.hypot(x, dir_y) #camera-to-point distance, shape (N,)
        angles = np.arctan2(x, dir_y) #bearing from the camera, 0 is straight ahead, positive is right, shape (N,)

        fov_x = self.robot.camera.fov_x
        ray_w = fov_x / n_rays # calculate the ray width

        # +fov_x/2 makes range from [-fov/2,fov/2] to [0,fov]
        # fit each angle from angles to each bin_idx, shape (N,)
        bin_idx = np.floor((angles + fov_x / 2) / ray_w).astype(int)

        # Camera sits cam_t_y ahead of the robot's center along its forward axis, so
        # dir_range is measured from the wrong origin. Recover the true distance from
        # the robot's center via law of cosines on the (robot center, camera, point)
        # triangle, using the known offset cam_t_y and the camera bearing `angles`.
        point_ranges = np.sqrt(dir_range**2 + cam_t_y**2 + 2 * dir_range * cam_t_y * np.cos(angles)) #math confirmed

        valid = (bin_idx >= 0) & (bin_idx < n_rays) & (point_ranges <= max_range)
        #compare elements of ranges (n_rays,) at indices bin_idx[valid]
        #with point_ranges[valid] at those indices
        np.minimum.at(ranges, bin_idx[valid], point_ranges[valid])
        return ranges

    def frame_to_ray_casting(self, frame, n_rays=16, max_range=0.6, delete_ground=True) -> np.ndarray:
        """Processes a single frame end-to-end into ray-casted ranges. See pcd_to_ray_casting for details."""
        pcd = self.frame_to_pcd(frame, delete_ground=delete_ground)
        return self.pcd_to_ray_casting(pcd, n_rays=n_rays, max_range=max_range)

    def annotate_floor(self, frame):
        return self.fsc.annotate_floor(frame)