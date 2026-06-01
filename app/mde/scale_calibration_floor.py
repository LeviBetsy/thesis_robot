import numpy as np
from pathlib import Path

def read_depthfile(depth_map_file):
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    dmDir = project_root / "data" / "depth_map"
    depth_map = Path(str(dmDir / depth_map_file))

    if depth_map.is_file():
        width, height = 640,480
        depth_data = np.fromfile(depth_map, dtype=np.float32).reshape((height, width))
        print(depth_data)
        # Safe to read/process the file here
    else:
        raise Exception(f"File {depth_map} does not exist")