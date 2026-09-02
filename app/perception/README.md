# Perception Module: Monocular Depth Estimation

This perception module utilizes Monocular Depth Estimation (MDE) to generate a dense range-finder scan of the robot's environment. The system extracts structural depth data directly from a single RGB camera feed. Module isolate perception task, no localization.

## Core Architecture

The module is segmented into four distinct stages, handling the pipeline from raw image ingestion to 3D visualization.

*   **`mde_depth.py`**
    The primary pipeline.
*   **`scale_calibration.py`**
    The physical grounding layer. It analyzes expected floor geometries within the camera's field of view to convert the relative MDE output into an accurate metric depth map. 
*   **`point_cloud_proj.py`**
    The 3D projection node. It uses the camera's intrinsic parameters to project the 2D metric depth map into a physical 3D point cloud array relative to the camera's coordinate frame.
*   **`point_cloud_visualizer.py`**
    The rendering interface. It ingests the generated 3D point cloud arrays and renders them for structural debugging and visual verification.
