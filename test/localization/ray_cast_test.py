import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import math

import cv2
import numpy as np

from app.localization.map import OccupancyGrid
from app.localization.particle import Particle
from app.localization.ray_caster import RayCaster
from app.localization.visualize_map import MapVisualizer
from app.util.config import Config
from app.robot_module.robot import Robot

'''
Visual check of RayCaster.predict against config/map/map1.json.

Every ray starts at the CAMERA and predict returns the distance from the camera, so the
hit point is simply camera + z * (cos heading, sin heading). If the caster is right, every
coloured dot lands on the face of a wall.

Drawn per test pose, one colour each:
    filled circle   robot centre, with its heading line
    coloured line   camera -> hit point, dot at the hit
    gray line       nothing hit within max_range
    black ring      the leftmost ray (index 0), so you can check left/right isn't flipped

Run from anywhere:
    python test/ray_cast_test.py              # serve on http://localhost:8001
    python test/ray_cast_test.py --save out.png
'''

config = Config.load("./config/robot_config.json")
robot = Robot(config)
MAP_PATH = "./config/map/map_wall.json"
# poses in meters / radians CCW from +x, each with what it should show
TEST_POSES = [
    (Particle(0.20, 0.20, -math.pi / 2),      "facing the y=0 perimeter"),
    (Particle(0.20, 0.20, 0.0),               "facing +x, RIGHT (high index) rays hit the y=0 perimeter"),
    (Particle(0.20, 0.55, -math.pi / 2),      "facing down at the y=0.37 wall"),
    (Particle(0.55, 0.45, math.pi / 2),       "facing up at the y=0.73 wall"),
    (Particle(0.70, 0.55, 0.0),               "facing the x=0.90 wall"),
    (Particle(0.85, 1.30, math.radians(135)), "diagonal, mixed hits and misses"),
    (Particle(0.55, 0.20, math.radians(135)), "diagonal toward the y=0.37 wall"),
    # (Particle(0.55, 0.45, math.pi / 2), "something")
]

# BGR, one per pose
COLORS = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (160, 160, 0), (0, 200, 200)]


class RayCastVisualizer(MapVisualizer):
    """MapVisualizer that also draws the ray fan of every test pose."""

    def set_rays(self, poses, origins, hits, z):
        with self.mutex_lock:
            self.ray_data = (poses, origins, hits, z)

    def render(self):
        image = super().render()
        if not hasattr(self, "ray_data"):
            return image

        with self.mutex_lock:
            poses, origins, hits, z = self.ray_data

        rows = self.grid.data.shape[0]
        scale = max(1, int(self.max_size / max(self.grid.data.shape)))
        to_px = lambda x, y: tuple(int(round(v)) for v in self._world_to_pixel(x, y, rows, scale))

        for i in range(poses.shape[0]):
            color = COLORS[i % len(COLORS)]
            x, y, theta = poses[i]

            for k in range(z.shape[1]):
                start = to_px(*origins[i, k])
                end = to_px(*hits[i, k])
                missed = z[i, k] >= config.ray_cast.max_range - 1e-6
                cv2.line(image, start, end, (170, 170, 170) if missed else color, 1, lineType=cv2.LINE_AA)
                if not missed:
                    cv2.circle(image, end, 3, color, -1, lineType=cv2.LINE_AA)

            cv2.circle(image, to_px(*hits[i, 0]), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            centre = to_px(x, y)
            cv2.circle(image, centre, 5, color, -1, lineType=cv2.LINE_AA)
            cv2.line(image, centre, to_px(*origins[i, 0]), color, 2, lineType=cv2.LINE_AA)
            cv2.putText(image, str(i), (centre[0] + 6, centre[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="write the render to this PNG instead of serving it")
    args = parser.parse_args()

    grid = OccupancyGrid.from_json(MAP_PATH, default_value=0)
    caster = RayCaster(grid, robot.camera, config)
    poses = np.array([p.pose() for p, _ in TEST_POSES]) #shape (N, 3)

    z = caster.predict(poses)                #shape (N, n_rays), meters from the camera
    origins, headings = caster.rays(poses)   #shape (N, n_rays, 2) and (N, n_rays)
    hits = origins + z[..., None] * np.stack([np.cos(headings), np.sin(headings)], axis=-1)

    for i, (particle, note) in enumerate(TEST_POSES):
        print(f"[{i}] {particle}  ({note})")
        print("    z:", np.round(z[i], 3))

    vis = RayCastVisualizer(grid)
    vis.set_rays(poses, origins, hits, z)
    if args.save:
        print(f"saved {vis.save(args.save)}")
    else:
        vis.serve_forever()
