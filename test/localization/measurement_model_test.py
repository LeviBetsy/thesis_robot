import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import argparse
import math
from types import SimpleNamespace

import cv2
import numpy as np

from app.localization.map import OccupancyGrid
from app.localization.mcl import MCLLocalization
from app.localization.visualize_map import MapVisualizer

'''
Visual check of the measurement stage of MCLLocalization against config/map/map1.json.

No robot, no laptop, no ZMQ: the "real" scan is faked by ray casting from a known TRUE_POSE
and adding gaussian noise, which is exactly what the MDE fan is supposed to deliver. A wide
gaussian cloud of particles is scattered around that pose and scored against it, so if the
beam model is right the cloud's colour should peak on / near the true pose.

Drawn:
    green circle + line      the true pose and its heading
    faint green rays         the noisy scan fed to measurement_update (from the camera)
    blue -> red circles      particles, red and larger = higher weight. The ramp is
                             normalised in LOG space (see color_strength), so it shows
                             the ordering of the cloud, not absolute confidence.
    black ring               the highest weight particle

Run from anywhere:
    python test/localization/measurement_model_test.py           # serve on http://localhost:8001
    python test/localization/measurement_model_test.py --save out.png
    python test/localization/measurement_model_test.py --seed 7
'''

MAP_PATH = os.path.join(os.path.dirname(__file__), "../../config/map/map1.json")

FOV_X = 1.2228      # robot.camera.fov_x, hardcoded so this test does not need the camera / calibration
N_RAYS = 16
CAM_FORWARD = 0.07  # robot.cam_t[1]
MAX_RANGE = 0.6

N_PARTICLES = 30
SIGMA_XY = 0.22     # deliberately wide, the point is to see the weights separate
SIGMA_THETA = math.radians(45)

# meters / radians CCW from +x. Facing +y at the y=0.73 wall, so the scan has real structure
# in it (wall dead ahead, open floor off to the sides) rather than 16 identical max ranges.
TRUE_POSE = (0.55, 0.45, math.pi / 2)

SCAN_NOISE = 0.02   # meters, stands in for monocular depth error on the faked scan

TRUE_COLOR = (0, 160, 0)
WEAK_COLOR = np.array([230, 190, 120], dtype=np.float64)  # BGR, lowest weight
STRONG_COLOR = np.array([0, 0, 255], dtype=np.float64)    # BGR, highest weight
HEADING_LEN = 0.05   # meters, length of the heading stub drawn on each particle
LOG_SPAN = 20.0      # a particle e^-20 times as likely as the best one draws fully pale


def color_strength(weights):
    """
    Maps weights to [0, 1] for colouring: LOG_SPAN log units below the best particle is
    fully pale, the best particle is fully red.

    All 16 beams come out of one image, so the likelihoods are heavily correlated and the
    normalised weights routinely collapse to one particle at ~1.0 and the rest at ~1e-20
    (see BeamSensorModel's alpha note). Ramping on the raw weights paints one dot red and
    leaves every other identically pale, which hides the shape of the posterior; stretching
    the log range end to end has the opposite problem, since one hopeless particle sets the
    floor and everything else bunches at the top. A fixed window below the best is stable
    across runs and keeps the ordering exact.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0:
        return weights
    logw = np.log(np.maximum(weights, 1e-300))
    return 1.0 - np.clip((logw.max() - logw) / LOG_SPAN, 0.0, 1.0)


def stub_robot(fov_x, cam_forward):
    """MCLLocalization only reads robot.camera.fov_x and robot.cam_t[1], so a real Robot
    (and its calibration npz) is not needed to exercise the filter."""
    return SimpleNamespace(camera=SimpleNamespace(fov_x=fov_x),
                           cam_t=np.array([0.0, cam_forward, 0.11]))


class ParticleWeightVisualizer(MapVisualizer):
    """MapVisualizer that draws the weighted particle cloud and the true pose it was scored against."""

    def set_cloud(self, poses, weights, true_pose, origins=None, hits=None, missed=None):
        with self.mutex_lock:
            self.cloud = (np.asarray(poses), np.asarray(weights), tuple(true_pose),
                          origins, hits, missed)

    def render(self):
        image = super().render()
        if not hasattr(self, "cloud"):
            return image

        with self.mutex_lock:
            poses, weights, true_pose, origins, hits, missed = self.cloud

        rows = self.grid.data.shape[0]
        scale = max(1, int(self.max_size / max(self.grid.data.shape)))
        to_px = lambda x, y: tuple(int(round(v)) for v in self._world_to_pixel(x, y, rows, scale))

        # the scan that was actually handed to measurement_update
        if origins is not None:
            for k in range(origins.shape[0]):
                color = (170, 170, 170) if missed[k] else (150, 220, 150)
                cv2.line(image, to_px(*origins[k]), to_px(*hits[k]), color, 1, lineType=cv2.LINE_AA)
                if not missed[k]:
                    cv2.circle(image, to_px(*hits[k]), 2, TRUE_COLOR, -1, lineType=cv2.LINE_AA)

        strength = color_strength(weights)

        # weakest first, so the particles worth looking at end up on top of the pile
        for i in np.argsort(strength):
            t = float(strength[i])
            color = tuple(int(v) for v in (WEAK_COLOR + (STRONG_COLOR - WEAK_COLOR) * t))
            radius = int(round(4 + 5 * t))

            x, y, theta = poses[i]
            centre = to_px(x, y)
            cv2.circle(image, centre, radius, color, -1, lineType=cv2.LINE_AA)
            cv2.line(image, centre,
                     to_px(x + HEADING_LEN * math.cos(theta), y + HEADING_LEN * math.sin(theta)),
                     color, 2, lineType=cv2.LINE_AA)

        if strength.size:
            best = to_px(*poses[int(np.argmax(strength))][:2])
            cv2.circle(image, best, 11, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        tx, ty, ttheta = true_pose
        centre = to_px(tx, ty)
        cv2.circle(image, centre, 6, TRUE_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.line(image, centre,
                 to_px(tx + 0.08 * math.cos(ttheta), ty + 0.08 * math.sin(ttheta)),
                 TRUE_COLOR, 2, lineType=cv2.LINE_AA)

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="write the render to this PNG instead of serving it")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for the cloud and the scan noise")
    args = parser.parse_args()

    np.random.seed(args.seed)

    grid = OccupancyGrid.from_json(MAP_PATH, default_value=0)
    mcl = MCLLocalization(stub_robot(FOV_X, CAM_FORWARD), grid,
                          n_particles=N_PARTICLES, n_rays=N_RAYS, max_range=MAX_RANGE)

    # the scan the MDE pipeline would have returned from the true pose, plus sensor noise
    true_pose = np.array([TRUE_POSE])
    z = mcl.caster.predict(true_pose)[0].astype(np.float64)
    z = np.clip(z + np.random.normal(0.0, SCAN_NOISE, z.shape[0]), 0.0, MAX_RANGE)

    mcl.init_particles_gaussian(*TRUE_POSE, sigma_xy=SIGMA_XY, sigma_theta=SIGMA_THETA)
    mcl.measurement_update(z)

    poses, weights = mcl.particle_set.snapshot()

    print(f"true pose: x={TRUE_POSE[0]:.3f} y={TRUE_POSE[1]:.3f} theta={math.degrees(TRUE_POSE[2]):.1f}deg")
    print(f"scan (m): {np.round(z, 3)}")
    print(f"{N_PARTICLES} particles, sigma_xy={SIGMA_XY} m, sigma_theta={math.degrees(SIGMA_THETA):.0f}deg")
    print(f"ESS: {mcl.effective_sample_size():.2f} / {N_PARTICLES}")
    print(f"weighted mean pose: {tuple(round(v, 3) for v in mcl.estimate_mean_pose())}")
    print(f"best particle:      {tuple(round(v, 3) for v in mcl.estimate_map())}")

    print("\n rank  weight     x      y    theta   err_xy")
    for rank, i in enumerate(np.argsort(weights)[::-1]):
        x, y, theta = poses[i]
        err = math.hypot(x - TRUE_POSE[0], y - TRUE_POSE[1])
        print(f"  {rank:3d}  {weights[i]:.4f}  {x:.3f}  {y:.3f}  {math.degrees(theta):6.1f}  {err:.3f}")

    origins, headings = mcl.caster.rays(true_pose)
    hits = origins[0] + z[:, None] * np.stack([np.cos(headings[0]), np.sin(headings[0])], axis=-1)
    missed = z >= MAX_RANGE - 1e-6

    vis = ParticleWeightVisualizer(grid)
    vis.set_cloud(poses, weights, TRUE_POSE, origins[0], hits, missed)
    if args.save:
        print(f"\nsaved {vis.save(args.save)}")
    else:
        vis.serve_forever()
