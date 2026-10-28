import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math

from app.localization.map import OccupancyGrid
from app.localization.particle import ParticleSet
from app.localization.ray_caster import RayCaster
from app.localization.measurement_model import BeamSensorModel
from app.localization.motion_model import OdometryMotionModel
from app.robot_module.robot import Robot
from app.robot_module.odometry import Odometry
from app.util.config import Config

'''
    Monte Carlo Localization against the known OccupancyGrid, driven by the 16-ray fan the
    laptop returns from monocular depth estimation.

    on_scan is written to drop straight into RangeReceiver(callback=...), so it runs on the
    ZMQ receiver thread, concurrently with the odometry thread. self.particle_set owns the
    lock guarding the particles/weights it holds.

    The motion stage pulls from an injected Odometry rather than owning one, so the same
    filter runs off MSP432 tachometer packets (WheelOdometry) or off a virtual rover with
    no change beyond the one constructor argument. See set_odometry for the contract,
    including the thread-safety it demands.

    STATUS: motion, measurement and resample stages are implemented. This still
    deliberately does NOT write back to Robot's pose.
'''

#TODO: VERIFY
class MCLLocalization:
    def __init__(self, robot: Robot, grid: OccupancyGrid, config: Config, sensor_model=None,
                 odometry=None, motion_model=None):
        self.robot = robot
        self.grid = grid
        self.n_particles = int(config.mcl.n_particles)
        self.n_rays = int(config.ray_cast.n_rays)
        self.sensor_model = sensor_model if sensor_model is not None else BeamSensorModel(config=config)
        self.motion_model = motion_model if motion_model is not None else OdometryMotionModel(config=config)

        # Where on_scan gets its motion from, see set_odometry for the contract. A bare
        # Odometry that nothing ever feeds consumes zeros, so "no odometry wired up" is an
        # inert source rather than a special case in on_scan, and the filter stays
        # runnable measurement-only, which is how the existing tests drive it.
        self.odometry = odometry if odometry is not None else Odometry()

        # Fixed for the life of the filter, so build the ray fan once rather than per scan.
        self.caster = RayCaster(grid, robot.camera, config=config)
        self.particle_set = ParticleSet()

    # ****** WIRING ******

    def set_odometry(self, odometry):
        """
        Swaps in where the motion stage gets its increment, for when the source does not
        exist yet at construction (the receiver and the filter are usually built first,
        and a virtual rover may want a reference to this filter before it can be built).

        CONTRACT, both halves of which the filter uses:
          consume() -> (dx, dy, dtheta), the increment accumulated since ITS OWN last
                       call, in the robot frame as of that last call, meters and radians,
                       AND clears the accumulator so no motion is ever applied twice.
          clear()   -> drops whatever has accumulated, called on re-seeding the cloud.

        Subclassing Odometry gets both for free, along with the body-frame composition,
        but nothing here checks the type: anything exposing those two methods drops in.

        consume() is called on whichever thread runs on_scan (the ZMQ receiver thread)
        while the source is being written on its own thread, so the read-and-clear must be
        atomic. Odometry guards it with a lock; a replacement has to do the same.

        Args:
            odometry (Odometry): the source. None installs an inert one, i.e. the filter
                                 runs measurement-only.
        """
        self.odometry = odometry if odometry is not None else Odometry()

    # ****** INITIALISATION ******

    # verified not tested
    def init_particles_gaussian(self, x, y, theta, sigma_x=0.05, sigma_y=0.05, sigma_theta=0.1, max_attempts=50):
        """
        Scatters particles around a known starting pose (tracking rather than global
        localization). Rejection-sampled against the map: any draw landing outside the
        grid or inside a wall is redrawn, so every particle starts on a valid hypothesis.

        Clears the odometry accumulator on the way out: motion from before the cloud
        existed means nothing to it, and would otherwise be applied on the first scan.
        """
        px = np.empty(self.n_particles)
        py = np.empty(self.n_particles)
        pt = np.empty(self.n_particles)
        remaining = np.arange(self.n_particles) #indices still needing a valid draw

        attempts = 0
        while remaining.size > 0:
            if attempts >= max_attempts:
                raise RuntimeError(f"init_particles_gaussian: {remaining.size} particles still "
                                   f"invalid after {max_attempts} attempts, is ({x}, {y}) actually free?")
            attempts += 1

            #only redraw the indices that failed last round, not the whole batch
            sample_x = np.random.normal(x, sigma_x, remaining.size)
            sample_y = np.random.normal(y, sigma_y, remaining.size)
            sample_theta = np.random.normal(theta, sigma_theta, remaining.size)

            valid = self.grid.is_placeable(sample_x, sample_y)
            accepted = remaining[valid]
            px[accepted], py[accepted], pt[accepted] = sample_x[valid], sample_y[valid], sample_theta[valid]
            remaining = remaining[~valid]

        self.particle_set.replace(np.column_stack((px, py, pt)),
                                  np.full(self.n_particles, 1.0 / self.n_particles))
        self.odometry.clear()

    # ****** FILTER STAGES ******

    def motion_update(self, dx, dy, dtheta):
        """
        Pushes every particle through the odometry increment measured since the last
        update. The sampling itself lives in OdometryMotionModel.sample, see there for
        the frame (dx, dy) must be in and the noise model.

        Args:
            dx (float): travel along the starting heading, meters.
            dy (float): travel to the left of the starting heading, meters.
            dtheta (float): heading change over the step, radians.
        """
        poses, w = self.particle_set.snapshot()
        n = poses.shape[0]
        if n == 0:
            return

        new_poses = self.motion_model.sample(poses, dx, dy, dtheta) #shape (N, 3)

        # Particles are pushed blind: one that walks into a wall keeps its weight until
        # measurement_update masks it, which is where that check belongs. Weights carry
        # over untouched, motion does not change how well a hypothesis explained the scan.
        # replace() wraps theta into [0, 2pi).
        self.particle_set.replace(new_poses, w)

    def measurement_update(self, z):
        """
        Reweights every particle by how well the map explains the actual scan.

        For each particle: ray cast its 16 beams against the known map, then score the
        real reading against those predictions with the beam sensor model.

        Args:
            z (np.ndarray): the measured ranges, shape (n_rays,), meters from the camera.
        """
        poses = self.particle_set.poses() #shape (N, 3)
        if poses.shape[0] == 0:
            return

        z_pred = self.caster.predict(poses) #shape (N, n_rays)

        w = self.sensor_model.weights(z, z_pred) #shape (N,)

        # A particle sitting inside a wall or off the map is not a valid hypothesis no
        # matter what its beams say.
        w = np.where(self.grid.is_placeable(poses[:, 0], poses[:, 1]), w, 0.0)

        # weights() normalised, but zeroing the masked particles afterwards drops the total
        # below 1, so renormalise here to keep the stored weights summing to 1. If EVERY
        # particle got masked there is nothing to normalise against, so fall back to uniform
        # rather than emitting NaNs.
        #
        # The zero case also covers the surviving weights underflowing to 0 when the
        # best-scoring particle is the one that got masked (weights() takes its max-subtract
        # over ALL particles, including that one). That cannot currently happen: p_rand
        # floors every in-range beam at z_rand/max_range, which caps the log-likelihood
        # spread at ~22 nats over 16 beams, nowhere near float64's limit. Set z_rand to 0 or
        # raise n_rays a long way and it stops being true.
        total = w.sum()
        w = w / total if total > 0.0 else np.full(w.shape[0], 1.0 / w.shape[0])

        self.particle_set.set_weights(w)

    def resample(self):
        """
        Importance resampling: draws a fresh set of exactly n_particles in proportion to
        the weights measurement_update left, then resets every weight to 1/N. Good
        hypotheses come back as several copies, poor ones die out; the next motion_update
        gives each copy its own noise draw, which is what pulls duplicates apart again.

        Low variance / systematic (Thrun Table 4.4): one random offset in [0, 1/N), then
        N evenly spaced pointers walked through the weight CDF. A particle with weight w
        gets floor or ceil of N*w copies, where multinomial draws would give it
        Binomial(N, w), e.g. 5 +- 2.2 copies at N=500, w=0.01.

        """
        poses, w = self.particle_set.snapshot()
        if poses.shape[0] == 0:
            return

        total = w.sum()
        w = w / total if total > 0.0 else np.full(w.shape[0], 1.0 / w.shape[0])

        # Output size comes from config, not the current set, so the count is held at
        # n_particles no matter what the set was handed.

        n = self.n_particles
        pointers = np.random.uniform(0.0, 1.0 / n) + np.arange(n) / n
        cdf = np.cumsum(w)
        cdf[-1] = 1.0 #float drift can leave the sum at 0.9999..., past which searchsorted runs off the end
        idx = np.searchsorted(cdf, pointers, side="right")

        self.particle_set.replace(poses[idx], np.full(n, 1.0 / n))

    # ****** OUTPUT ******

    def estimate_mean_pose(self) -> tuple:
        """
        Weighted mean pose of the particle set.

        theta uses a circular mean, since headings wrap at 2*pi and a plain average of
        0.01 and 6.27 rad would point backwards.

        Returns:
            pose (tuple): (x, y, theta), meters and radians in [0, 2*pi).
        """
        poses, w = self.particle_set.snapshot()
        if poses.shape[0] == 0:
            return 0.0, 0.0, 0.0

        total = w.sum()
        w = w / total if total > 0.0 else np.full(w.shape[0], 1.0 / w.shape[0])

        x = float(np.sum(w * poses[:, 0]))
        y = float(np.sum(w * poses[:, 1]))
        theta = float(math.atan2(np.sum(w * np.sin(poses[:, 2])),
                                 np.sum(w * np.cos(poses[:, 2]))) % (2 * math.pi))
        return x, y, theta

    def estimate_best_particle(self) -> tuple:
        """
        Pose of the single highest-weight particle, rather than estimate_mean_pose()'s
        weighted mean. Prefer this when the cloud is multi-modal or not yet collapsed
        by resample() - the mean of scattered clusters can land on a pose nothing
        actually supports (even inside a wall), while the best particle is always one
        the model scored well.

        on_scan resamples at the end of every scan, which resets every weight to 1/N, so
        called after on_scan this returns an arbitrary particle. It only means something
        between measurement_update and resample.

        Returns:
            pose (tuple): (x, y, theta), or (0.0, 0.0, 0.0) if the set is empty.
        """
        pose, _ = self.particle_set.best()
        if pose is None:
            return 0.0, 0.0, 0.0
        return pose

    def effective_sample_size(self) -> float:
        """1 / sum(w^2). Falls toward 1 as the weights collapse onto a single particle."""
        w = self.particle_set.get_weights()
        total = w.sum()
        if total <= 0.0:
            return 0.0
        w = w / total
        return float(1.0 / np.sum(w**2))

    # ****** CALLBACK ******

    def on_scan(self, z, do_resample=True, motion=None):
        """
        RangeReceiver callback: one full filter step per scan from the laptop.

        z describes where the robot was when its frame was CAPTURED, not where it is now.
        Consuming odometry here would also take the motion made while the frame was at the
        laptop, pushing the cloud ahead of what z describes. So a caller that knows the
        capture moment consumes the odometry THEN and passes it in as motion; the motion
        after the capture stays in the accumulator for the next scan.

        Args:
            z (np.ndarray): float32, shape (n_rays,), meters. Comes in on the ZMQ receiver thread.
            motion (tuple): (dx, dy, dtheta) consumed at the frame's capture, see
                            Odometry.consume. None consumes the odometry now instead.
        """
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.shape[0] != self.n_rays:
            print(f"MCL: expected {self.n_rays} ranges, got {z.shape[0]}, dropping scan")
            if motion is not None:
                self.motion_update(*motion) #already consumed by the caller, so it is applied now or never
            return

        dx, dy, dtheta = motion if motion is not None else self.odometry.consume()
        self.motion_update(dx, dy, dtheta)
        self.measurement_update(z)
        if do_resample:
            self.resample()