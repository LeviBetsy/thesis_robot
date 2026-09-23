import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math

from app.localization.map import OccupancyGrid
from app.robot_module.camera import Camera
from app.util.config import Config

'''
Turns a hypothesised pose into the n_rays ranges the MDE pipeline would report if the
robot really were standing there.

This MUST mirror MDE_Depth.pcd_to_ray_casting on the laptop exactly, or every weight the
filter computes is biased. The contract it defines:
  - n_rays evenly spaced bins across robot.camera.fov_x
  - bearing 0 is straight ahead, POSITIVE IS THE ROBOT'S RIGHT (clockwise), which is the
    opposite sign to world theta (CCW from +x). Index 0 is the leftmost ray.
  - bearings are measured from the CAMERA
  - the returned range is also from the CAMERA, not the robot centre
'''
#VERIFIED and TESTED
class RayCaster:
    def __init__(self, grid: OccupancyGrid, camera: Camera, config: Config):
        self.grid = grid
        self.fov_x = camera.fov_x
        self.n_rays = int(config.ray_cast.n_rays)
        self.max_range = float(config.ray_cast.max_range)
        self.cam_forward = float(config.robot.cam_t[1]) #camera offset ahead of robot centre, meters
        self.occ_threshold = float(config.ray_cast.occ_threshold) #cells strictly above this block a ray

        # Centre bearing of each ray bin, matching pcd_to_ray_casting's binning. That function
        # assigns a point to bin floor((angle + fov_x/2) / ray_w), so bin k spans
        # [-fov_x/2 + k*ray_w, -fov_x/2 + (k+1)*ray_w) and its centre sits half a bin further in.
        ray_w = self.fov_x / self.n_rays
        self.bearings = -self.fov_x / 2.0 + (np.arange(self.n_rays) + 0.5) * ray_w #shape (n_rays,)

    # verified and TESTED
    def cast(self, origins, headings) -> np.ndarray:
        """
        Marches M rays through the grid and returns the distance to the first occupied cell.

        Fixed-step marching rather than a true DDA: every ray takes the same number of
        steps, so the whole fan is one numpy loop instead of M variable-length traversals.
        The step is cell_size/3 so a one-cell-thick wall cannot be stepped over even when
        crossed diagonally; the cost is that ranges are quantized to that step.

        Args:
            origins (np.ndarray): ray start points in meters, shape (M, 2).
            headings (np.ndarray): world-frame ray directions in radians, CCW from +x, shape (M,).
        Returns:
            ranges (np.ndarray): shape (M,), float32, distance in meters, max_range if nothing hit.
        """
        grid = self.grid
        max_range = self.max_range
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
        headings = np.asarray(headings, dtype=np.float64).reshape(-1)

        step = grid.cell_size / 3.0
        n_steps = int(math.ceil(max_range / step))

        ox, oy = origins[:, 0], origins[:, 1] #shape (M,)
        dx, dy = np.cos(headings) * step, np.sin(headings) * step #per-step delta, shape (M,)

        ranges = np.full(headings.shape[0], max_range, dtype=np.float32)
        alive = np.ones(headings.shape[0], dtype=bool) #rays that have not hit anything yet

        for i in range(1, n_steps + 1):
            #sample the point i steps along every ray at once
            px = ox + dx * i
            py = oy + dy * i

            row, col = grid.world_to_grid(px, py)

            outside = (col < 0) | (col >= grid.cols) | (row < 0) | (row >= grid.rows)
            blocked = grid.data[np.clip(row, 0, grid.rows - 1), np.clip(col, 0, grid.cols - 1)] > self.occ_threshold

            #a ray leaving the grid is treated as a hit
            hit = alive & (outside | blocked)
            ranges[hit] = min(i * step, max_range)
            alive &= ~hit

            if not alive.any():
                break

        return ranges

    def rays(self, poses):
        """
        Builds the ray fan for every pose: where each ray starts and which way it points.

        Args:
            poses (np.ndarray): shape (N, 3) of (x, y, theta) in meters / radians CCW from +x.
        Returns:
            origins (np.ndarray): shape (N, n_rays, 2), camera position in meters.
            headings (np.ndarray): shape (N, n_rays), world-frame ray direction, radians CCW from +x.
        """
        #keep this reshape, if correct shape it wont run
        poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        n, n_rays = poses.shape[0], self.n_rays

        px, py, ptheta = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3] #shape (N,1) to broadcast over rays

        # Rays leave the camera, not the robot centre, so the bin edges line up with the camera's
        # true angular FOV the same way they do on the sensor side.
        cx = px + self.cam_forward * np.cos(ptheta) #shape (N,1)
        cy = py + self.cam_forward * np.sin(ptheta)

        # Explicitly expand both operands to the full (N, n_rays) grid before combining them,
        theta_grid = np.broadcast_to(ptheta, (n, n_rays))                          #each particle's theta, repeated across rays
        bearing_grid = np.broadcast_to(self.bearings.reshape(1, -1), (n, n_rays))  #the n_rays bearings, repeated across particles

        # Bearing is clockwise but theta is counter-clockwise, hence the minus.
        headings = theta_grid - bearing_grid #shape (N, n_rays)

        origins = np.stack([np.broadcast_to(cx, (n, n_rays)),
                            np.broadcast_to(cy, (n, n_rays))], axis=-1) #shape (N, n_rays, 2) of N number of particles, each has n_rays with 2 origin
        return origins, headings

    # verified, have not tested
    def predict(self, poses) -> np.ndarray:
        """
        Casts the full ray fan for every pose against the map.

        Args:
            poses (np.ndarray): shape (N, 3) of (x, y, theta) in meters / radians CCW from +x.
        Returns:
            z_pred (np.ndarray): shape (N, n_rays), float32, range in meters from the CAMERA.
        """
        origins, headings = self.rays(poses)
        d_cam = self.cast(origins.reshape(-1, 2), headings.reshape(-1))
        return d_cam.reshape(headings.shape).astype(np.float32)


if __name__ == "__main__":
    # Geometry sanity check against config/map/map1.json, which has a perimeter wall at y=0
    # and a horizontal wall running from x=0.0 to x=0.365 at y=0.37.
    grid = OccupancyGrid.from_json("config/map/map1.json", default_value=0)
    caster = RayCaster(grid, fov_x=1.2228, n_rays=16, max_range=0.6, cam_forward=0.07)

    print("bearings (deg, index 0 is leftmost):")
    print(np.round(np.degrees(caster.bearings), 2))

    # Facing -y from (0.20, 0.20) at the perimeter wall. Its cell spans y in [0, 0.025), so
    # the occupied face is at y=0.025. The camera sits 0.07 ahead at y=0.13, so the centre
    # beams must read ~0.105 to within one march step (cell_size/3 = 8 mm).
    z = caster.predict(np.array([(0.20, 0.20, -math.pi / 2)]))[0]
    print("\nfacing the perimeter wall (-y) from (0.20, 0.20), expect centre ~0.105:")
    print(np.round(z, 3))

    # Facing +y at the y=0.37 wall, whose cell starts at y=0.35; camera at y=0.27.
    z2 = caster.predict(np.array([(0.20, 0.20, math.pi / 2)]))[0]
    print("\nfacing the y=0.37 wall (+y) from (0.20, 0.20), expect centre ~0.080:")
    print(np.round(z2, 3))

    # Facing +x along open floor. This one pins down the BEARING SIGN, which is the easiest
    # thing to get backwards: the robot is 0.20 m above the bottom perimeter wall, which
    # sits on its RIGHT when facing +x. So the high indices (the right-hand beams) must be
    # the short ones. If the low indices shorten instead, the theta/bearing sign is flipped.
    z3 = caster.predict(np.array([(0.20, 0.20, 0.0)]))[0]
    print("\nfacing open floor (+x) from (0.20, 0.20), expect the RIGHT beams to shorten:")
    print(np.round(z3, 3))
