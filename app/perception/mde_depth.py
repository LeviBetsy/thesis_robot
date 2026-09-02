import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.mde.DAV2_pth import DepthAnythingPredictor
from app.robot_module.robot import Robot
from app.perception.scale_calibration import FloorScaleCalibration
from app.perception.point_cloud_proj import PointCloudProjection

class MDE_Depth:
    def __init__(self, calib_file="fisheye_calib.npz", ref_file="z_real_16group.npz"):
        self.robot = Robot(calib_file)
        self.predictor = DepthAnythingPredictor()
        self.fsc = FloorScaleCalibration(ref_file, group_n=16)
        self.pcd_processor = PointCloudProjection(self.robot)

    def process_rel(self,frame):
        return self.predictor.infer(frame)

    def process_metric(self, frame):
        self.rel_depth = self.predictor.infer(frame)
        self.fsc.scale_calibration(self.rel_depth)
        metric_depth = self.fsc.relative_to_metric(self.rel_depth, extrapolate=False)
        return metric_depth


    def frame_to_pointcloud(self, frame):
        """Processes a single frame and returns the pointcloud."""
        metric_depth = self.process_metric(frame)
        pcd_cc = self.pcd_processor.proj_pcd_cc(metric_depth, delete_ground=True)
        pcd_rc = self.pcd_processor.pcd_camera_to_robot(pcd_cc)
        
        return pcd_rc

    def frame_to_depth_scan(self, frame):
        pass

    def annotate_floor(self, frame):
        return self.fsc.annotate_floor(frame)