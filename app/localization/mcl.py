import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math
import threading

from app.localization.map import OccupancyGrid
from app.localization.particle import Particle, beam_bearings, predict_ranges
from app.localization.measurement_model import BeamSensorModel
from app.robot_module.robot import Robot

'''
    Monte Carlo Localization against the known OccupancyGrid, driven by the 16-ray fan the
    laptop returns from monocular depth estimation.

    on_scan is written to drop straight into RangeReceiver(callback=...), so it runs on the
    ZMQ receiver thread, concurrently with the odometry thread. self.mutex_lock guards the
    particle list.

    STATUS: only the measurement stage is implemented. motion_update and resample are
    placeholders, so estimate() is not yet meaningful and this deliberately does NOT write
    back to Robot's pose.
'''

#TODO: VERIFY
class MCLLocalization:
    def __init__(self, robot: Robot, grid: OccupancyGrid, n_particles=500,
                 n_rays=16, max_range=0.6, sensor_model=None):
        self.robot = robot
        self.grid = grid
        self.n_particles = int(n_particles)
        self.n_rays = int(n_rays)
        self.max_range = float(max_range)
        self.sensor_model = sensor_model if sensor_model is not None else BeamSensorModel(max_range=max_range)

        # Fixed for the life of the filter, so compute them once rather than per scan.
        self.bearings = beam_bearings(robot.camera.fov_x, self.n_rays) #shape (n_rays,)
        self.cam_forward = float(robot.cam_t[1]) #camera sits this far ahead of the robot centre

        self.particles = []
        self.mutex_lock = threading.Lock() #guards self.particles

    # ****** INITIALISATION ******

    def init_particles_uniform(self):
        """Scatters particles over the map's free cells with uniform heading (global localization)."""
        rows, cols = np.nonzero(self.grid.data <= 0.5)
        if rows.shape[0] == 0:
            raise RuntimeError("Map has no free cells to place particles in")

        pick = np.random.randint(0, rows.shape[0], size=self.n_particles)
        half = self.grid.cell_size / 2.0

        #cell centre plus jitter within the cell, so particles are not stuck on a lattice
        x = (cols[pick] * self.grid.cell_size) - half + np.random.uniform(-half, half, self.n_particles)
        y = (rows[pick] * self.grid.cell_size) - half + np.random.uniform(-half, half, self.n_particles)
        theta = np.random.uniform(0.0, 2 * math.pi, self.n_particles)

        w = 1.0 / self.n_particles
        with self.mutex_lock:
            self.particles = [Particle(x[i], y[i], theta[i], w) for i in range(self.n_particles)]

    def init_particles_gaussian(self, x, y, theta, sigma_xy=0.05, sigma_theta=0.1):
        """Scatters particles around a known starting pose (tracking rather than global localization)."""
        px = np.random.normal(x, sigma_xy, self.n_particles)
        py = np.random.normal(y, sigma_xy, self.n_particles)
        pt = np.random.normal(theta, sigma_theta, self.n_particles)

        w = 1.0 / self.n_particles
        with self.mutex_lock:
            self.particles = [Particle(px[i], py[i], pt[i], w) for i in range(self.n_particles)]

    # ****** FILTER STAGES ******

    def motion_update(self, dx, dy, dtheta):
        """
        PLACEHOLDER. Should push every particle through the odometry increment measured
        since the last update, with noise sampled per particle (Thrun's odometry motion
        model: decompose into rotate / translate / rotate and perturb each), so the cloud
        spreads to reflect encoder drift.
        """
        pass

    def measurement_update(self, z):
        """
        Reweights every particle by how well the map explains the actual scan.

        For each particle: ray cast its 16 beams against the known map, then score the
        real reading against those predictions with the beam sensor model.

        Args:
            z (np.ndarray): the measured ranges, shape (n_rays,), meters from the robot centre.
        """
        with self.mutex_lock:
            if not self.particles:
                return
            #pack once, ~0.3 ms at N=500, negligible next to the cast
            poses = np.array([[p.x, p.y, p.theta] for p in self.particles]) #shape (N, 3)

        z_pred = predict_ranges(self.grid, poses, self.bearings,
                                cam_forward=self.cam_forward,
                                max_range=self.max_range) #shape (N, n_rays)

        w = self.sensor_model.weights(z, z_pred) #shape (N,)

        # A particle sitting inside a wall or off the map is not a valid hypothesis no
        # matter what its beams say.
        w = np.where(self.grid.is_free(poses[:, 0], poses[:, 1]), w, 0.0)

        total = w.sum()
        #if every particle is invalid, fall back to uniform rather than emitting NaNs
        w = w / total if total > 0.0 else np.full(w.shape[0], 1.0 / w.shape[0])

        with self.mutex_lock:
            for particle, weight in zip(self.particles, w):
                particle.weight = float(weight)

    def resample(self):
        """
        PLACEHOLDER. Should draw a new particle set proportional to the weights (low
        variance / systematic resampling), reset every weight to 1/N, and only fire when
        the effective sample size drops below a threshold so the cloud is not needlessly
        collapsed on every scan.
        """
        pass

    # ****** OUTPUT ******

    def estimate(self) -> tuple:
        """
        Weighted mean pose of the particle set.

        theta uses a circular mean, since headings wrap at 2*pi and a plain average of
        0.01 and 6.27 rad would point backwards.

        Returns:
            pose (tuple): (x, y, theta), meters and radians in [0, 2*pi).
        """
        with self.mutex_lock:
            if not self.particles:
                return 0.0, 0.0, 0.0
            poses = np.array([[p.x, p.y, p.theta] for p in self.particles])
            w = np.array([p.weight for p in self.particles])

        total = w.sum()
        w = w / total if total > 0.0 else np.full(w.shape[0], 1.0 / w.shape[0])

        x = float(np.sum(w * poses[:, 0]))
        y = float(np.sum(w * poses[:, 1]))
        theta = float(math.atan2(np.sum(w * np.sin(poses[:, 2])),
                                 np.sum(w * np.cos(poses[:, 2]))) % (2 * math.pi))
        return x, y, theta

    def effective_sample_size(self) -> float:
        """1 / sum(w^2). Falls toward 1 as the weights collapse onto a single particle."""
        with self.mutex_lock:
            w = np.array([p.weight for p in self.particles])
        total = w.sum()
        if total <= 0.0:
            return 0.0
        w = w / total
        return float(1.0 / np.sum(w**2))

    # ****** CALLBACK ******

    def on_scan(self, z):
        """
        RangeReceiver callback: one full filter step per scan from the laptop.

        Args:
            z (np.ndarray): float32, shape (n_rays,), meters. Comes in on the ZMQ receiver thread.
        """
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape[0] != self.n_rays:
            print(f"MCL: expected {self.n_rays} ranges, got {z.shape[0]}, dropping scan")
            return

        # TODO: this scan describes where the robot was when the frame was captured, not
        # where it is now. Rewind by the odometry accumulated since then before updating.
        self.motion_update(0.0, 0.0, 0.0) #PLACEHOLDER
        self.measurement_update(z)
        self.resample() #PLACEHOLDER

        # Deliberately not writing self.robot's pose yet: with resampling stubbed the
        # estimate is not meaningful, and writing it would race the odometry thread.


if __name__ == "__main__":
    import time

    grid = OccupancyGrid.from_json("config/map/map0.json", default_value=0)

    robot = Robot("fisheye_calib.npz")
    truth = (0.30, 0.60, math.radians(90))

    mcl = MCLLocalization(robot=robot, grid=grid, n_particles=500)
    z = predict_ranges(grid, np.array([truth]), mcl.bearings,
                       cam_forward=mcl.cam_forward, max_range=mcl.max_range)[0]
    print("ground truth  :", tuple(round(v, 3) for v in truth))
    print("synthesised z :", np.round(z, 3))

    # ****** 1. does the true pose actually outscore a random one? ******
    # Scatter uniformly, then plant the true pose in the cloud. 500 particles over this map
    # will not land near the truth by chance, so planting it is the only way to ask the
    # question this test is actually about: given the true pose, does the model rank it top?
    mcl.init_particles_uniform()
    mcl.particles[0] = Particle(*truth)

    start = time.perf_counter()
    mcl.measurement_update(z)
    elapsed = (time.perf_counter() - start) * 1000

    ranked = sorted(mcl.particles, key=lambda p: p.weight, reverse=True)
    rank = next(i for i, p in enumerate(ranked) if p.pose() == mcl.particles[0].pose())
    print(f"\nmeasurement_update: {elapsed:.1f} ms at N={mcl.n_particles}")
    print(f"true pose ranked {rank + 1} of {mcl.n_particles}, "
          f"weight {mcl.particles[0].weight:.4g} vs uniform {1.0 / mcl.n_particles:.4g}")
    print(f"effective sample size: {mcl.effective_sample_size():.1f} / {mcl.n_particles}")

    # ****** 2. tracking: does the cloud's mean sit on the truth? ******
    # The realistic case once the motion model exists, and the one where the weighted mean
    # is meaningful at all (a uniform cloud's mean is just the map centroid, multi-modal by
    # construction, so it says nothing about whether the model works).
    mcl.init_particles_gaussian(truth[0], truth[1], truth[2], sigma_xy=0.08, sigma_theta=0.3)
    mcl.measurement_update(z)
    est = mcl.estimate()
    err = math.hypot(est[0] - truth[0], est[1] - truth[1])
    print(f"\ntracking estimate: {tuple(round(v, 3) for v in est)}")
    print(f"  position error {err * 100:.1f} cm, "
          f"heading error {math.degrees(abs(est[2] - truth[2])):.1f} deg")
    print(f"  effective sample size: {mcl.effective_sample_size():.1f} / {mcl.n_particles}")

    # ****** 3. is the tempering knob doing anything? ******
    for alpha in (1.0, 0.25):
        mcl.sensor_model.alpha = alpha
        mcl.init_particles_gaussian(truth[0], truth[1], truth[2], sigma_xy=0.08, sigma_theta=0.3)
        mcl.measurement_update(z)
        print(f"\nalpha={alpha}: effective sample size {mcl.effective_sample_size():.1f} / {mcl.n_particles}")
