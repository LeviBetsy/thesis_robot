import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math

from app.localization.map import OccupancyGrid

'''
Geometry shared by every particle: turning a hypothesised pose into the 16 ranges the
MDE pipeline would report if the robot really were standing there.

This MUST mirror MDE_Depth.pcd_to_ray_casting on the laptop exactly, or every weight the
filter computes is biased. The contract it defines:
  - n_rays evenly spaced bins across robot.camera.fov_x
  - bearing 0 is straight ahead, POSITIVE IS THE ROBOT'S RIGHT (clockwise), which is the
    opposite sign to world theta (CCW from +x). Index 0 is the leftmost ray.
  - bearings are measured from the CAMERA, which sits cam_t[1] = 7 cm forward of the centre
  - but the returned range is from the ROBOT CENTRE, recovered by law of cosines
'''

# verified, have not tested
def beam_bearings(fov_x, n_rays) -> np.ndarray:
    """
    Returns the centre bearing of each ray bin, matching pcd_to_ray_casting's binning.

    That function assigns a point to bin floor((angle + fov_x/2) / ray_w), so bin k spans
    [-fov_x/2 + k*ray_w, -fov_x/2 + (k+1)*ray_w) and its centre sits half a bin further in.

    Args:
        fov_x (float): horizontal field of view in radians.
        n_rays (int): number of bins.
    Returns:
        bearings (np.ndarray): shape (n_rays,), radians, 0 straight ahead, positive to the right.
    """
    ray_w = fov_x / n_rays
    return -fov_x / 2.0 + (np.arange(n_rays) + 0.5) * ray_w


# verified, have not tested
def predict_ranges(grid: OccupancyGrid, poses, bearings, cam_forward, max_range) -> np.ndarray:
    """
    Casts the full ray fan for every pose against the map.

    Args:
        grid (OccupancyGrid): the known map.
        poses (np.ndarray): shape (N, 3) of (x, y, theta) in meters / radians CCW from +x.
        bearings (np.ndarray): shape (n_rays,) from beam_bearings.
        cam_forward (float): camera offset ahead of the robot centre in meters (robot.cam_t[1]).
        max_range (float): the sensor's reporting cap in meters.
    Returns:
        z_pred (np.ndarray): shape (N, n_rays), float32, range in meters from the ROBOT CENTRE.
    """
    #keep this reshape, if correct shape it wont run
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3) 
    bearings = np.asarray(bearings, dtype=np.float64).reshape(-1)
    n, n_rays = poses.shape[0], bearings.shape[0]

    px, py, ptheta = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3] #shape (N,1) to broadcast over rays

    # Rays leave the camera, not the robot centre, so the bin edges line up with the camera's
    # true angular FOV the same way they do on the sensor side.
    cx = px + cam_forward * np.cos(ptheta) #shape (N,1)
    cy = py + cam_forward * np.sin(ptheta)

    # Explicitly expand both operands to the full (N, n_rays) grid before combining them,
    theta_grid = np.broadcast_to(ptheta, (n, n_rays))                     #each particle's theta, repeated across rays
    bearing_grid = np.broadcast_to(bearings.reshape(1, -1), (n, n_rays))  #the n_rays bearings, repeated across particles
    # print("LOOK")
    # print(bearing_grid)

    # Bearing is clockwise but theta is counter-clockwise, hence the minus.
    headings = theta_grid - bearing_grid #shape (N, n_rays)

    origins = np.stack([np.broadcast_to(cx, (n, n_rays)),
                        np.broadcast_to(cy, (n, n_rays))], axis=-1) #shape (N, n_rays, 2) of N number of particles, each has n_rays with 2 origin

    # March past max_range: a robot-centre range of max_range can sit up to cam_forward
    # further out along the camera ray, and clipping before the conversion would lose that.
    d_cam = grid.ray_cast_batch(origins.reshape(-1, 2),
                                headings.reshape(-1),
                                max_range + cam_forward)
    d_cam = d_cam.reshape(n, n_rays).astype(np.float64)

    # Same law of cosines pcd_to_ray_casting uses to move a camera-relative distance back
    # onto the robot centre, over the (robot centre, camera, hit point) triangle.
    ct = cam_forward
    #calculate depth from camera to robot center
    z_pred = np.sqrt(d_cam**2 + ct**2 + 2 * d_cam * ct * np.cos(bearing_grid)) 
    return np.minimum(z_pred, max_range).astype(np.float32)


'''
    One weighted pose hypothesis. The filter keeps a list of these; the vectorised path in
    MCLLocalization packs them into an (N,3) array, so ray_cast here is the single-particle
    handle for debugging and visualisation rather than the hot path.
'''
class Particle:
    def __init__(self, x, y, theta, weight=1.0):
        self.x = float(x) #in meter
        self.y = float(y)
        self.theta = float(theta) % (2 * math.pi) #match Robot.set_robot_pose's wrapping
        self.weight = float(weight)

    def pose(self) -> tuple:
        """Returns this particle's (x, y, theta)."""
        return self.x, self.y, self.theta


    def __repr__(self):
        return f"Particle(x={self.x:.3f}, y={self.y:.3f}, theta={math.degrees(self.theta):.1f}deg, w={self.weight:.4g})"


if __name__ == "__main__":
    # Geometry sanity check against config/map/map0.json, which has a horizontal wall
    # running from x=0.0 to x=0.34 at y=0.35.
    grid = OccupancyGrid.from_json("config/map/map0.json", default_value=0)

    bearings = beam_bearings(1.2228, 16)
    print("bearings (deg, index 0 is leftmost):")
    print(np.round(np.degrees(bearings), 2))

    # Facing -y from (0.20, 0.20) at the perimeter wall, whose inner face is exactly y=0.
    # This is the cleanest check available: it exercises the camera offset and the law of
    # cosines with no rasterisation ambiguity, so the centre beams must read the true
    # centre-to-wall distance of 0.200 to within one march step (cell_size/3 = 8 mm).
    p = Particle(0.20, 0.20, -math.pi / 2)
    z = predict_ranges(grid, np.array([p.pose()]), bearings, cam_forward=0.07, max_range=0.6)[0]
    print("\nfacing the perimeter wall (-y) from (0.20, 0.20), expect centre ~0.200:")
    print(np.round(z, 3))

    # Facing +y at the wall the JSON calls y=0.35. NOTE the expected value is 0.125, not
    # 0.150: world_to_grid maps y=0.35 to row 14 (spanning [0.325, 0.35]) rather than row
    # 15, because 0.025 is not exact in binary and 0.375 // 0.025 == 14.0. So every wall on
    # an exact cell multiple rasterises one cell short. That is pre-existing map behaviour,
    # not a ray casting error, but it biases the whole filter and is worth fixing in map.py.
    p2 = Particle(0.20, 0.20, math.pi / 2)
    z2 = predict_ranges(grid, np.array([p2.pose()]), bearings, cam_forward=0.07, max_range=0.6)[0]
    print("\nfacing the y=0.35 wall (+y) from (0.20, 0.20), expect centre ~0.125:")
    print(np.round(z2, 3))

    # Facing +x along open floor. This one pins down the BEARING SIGN, which is the easiest
    # thing to get backwards: the robot is 0.20 m above the bottom perimeter wall, which
    # sits on its RIGHT when facing +x. So the high indices (the right-hand beams) must be
    # the short ones. If the low indices shorten instead, the theta/bearing sign is flipped.
    p3 = Particle(0.20, 0.20, 0.0)
    z3 = predict_ranges(grid, np.array([p3.pose()]), bearings, cam_forward=0.07, max_range=0.6)[0]
    print("\nfacing open floor (+x) from (0.20, 0.20), expect the RIGHT beams to shorten:")
    print(np.round(z3, 3))
