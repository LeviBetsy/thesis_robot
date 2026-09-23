import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math

from app.localization.map import OccupancyGrid
from app.localization.particle import Particle, ParticleSet
from app.localization.ray_caster import RayCaster
from app.localization.measurement_model import BeamSensorModel
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

    STATUS: motion and measurement stages are implemented, resample is still a
    placeholder, so estimate() is not yet meaningful and this deliberately does NOT write
    back to Robot's pose.
'''

#TODO: VERIFY
class MCLLocalization:
    def __init__(self, robot: Robot, grid: OccupancyGrid, config: Config, sensor_model=None,
                 odometry=None):
        self.robot = robot
        self.grid = grid
        self.n_particles = int(config.mcl.n_particles)
        self.n_rays = int(config.ray_cast.n_rays)
        self.sensor_model = sensor_model if sensor_model is not None else BeamSensorModel(config=config)

        # Where on_scan gets its motion from, see set_odometry for the contract. A bare
        # Odometry that nothing ever feeds consumes zeros, so "no odometry wired up" is an
        # inert source rather than a special case in on_scan, and the filter stays
        # runnable measurement-only, which is how the existing tests drive it.
        self.odometry = odometry if odometry is not None else Odometry()

        # Odometry motion model noise (Thrun Table 5.6), see motion_update.
        self.alpha1 = float(config.mcl.motion_alpha1)
        self.alpha2 = float(config.mcl.motion_alpha2)
        self.alpha3 = float(config.mcl.motion_alpha3)
        self.alpha4 = float(config.mcl.motion_alpha4)
        self.min_trans = float(config.mcl.motion_min_trans)

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

    # have never verified
    def init_particles_uniform(self):
        """Scatters particles over the map's free cells with uniform heading (global
        localization).

        Clears the odometry accumulator on the way out: motion from before the cloud
        existed means nothing to it, and would otherwise be applied on the first scan."""
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
        particles = [Particle(x[i], y[i], theta[i]) for i in range(self.n_particles)]
        self.particle_set.replace(particles, [w] * self.n_particles)
        self.odometry.clear()

    # verified not tested
    def init_particles_gaussian(self, x, y, theta, sigma_x=0.05, sigma_y=0.05, sigma_theta=0.1, max_attempts=50):
        """
        Scatters particles around a known starting pose (tracking rather than global
        localization). Rejection-sampled against the map: any draw landing outside the
        grid or inside a wall is redrawn, so every particle starts on a valid hypothesis.

        Clears the odometry accumulator on the way out, same reason as
        init_particles_uniform: pre-seed motion does not belong to this cloud.
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

        w = 1.0 / self.n_particles
        particles = [Particle(px[i], py[i], pt[i]) for i in range(self.n_particles)]
        self.particle_set.replace(particles, [w] * self.n_particles)
        self.odometry.clear()

    # ****** FILTER STAGES ******

    def motion_update(self, dx, dy, dtheta):
        """
        Pushes every particle through the odometry increment measured since the last
        update, with noise drawn independently per particle, so the cloud spreads to
        reflect encoder drift.

        Thrun's odometry motion model (Probabilistic Robotics, Table 5.6): the step is
        decomposed into rotate / translate / rotate, each of the three is perturbed, and
        the perturbed triple is replayed from each particle's own pose.

        FRAME: (dx, dy) are the odometry increment measured RELATIVE to where the robot
        was at the start of the scan, x along the heading it had then. That is the book's
        (x_bar' - x_bar, y_bar' - y_bar) with theta_bar already taken out, so
        delta_rot1 = atan2(dy, dx) needs no further subtraction. Every particle then
        applies delta_rot1 relative to its OWN theta, which is the point of the
        decomposition: particles facing different ways must move in different world
        directions. For a differential drive this increment is
        dx = d*cos(dtheta/2), dy = d*sin(dtheta/2) with d the mid-point arc length.

        Args:
            dx (float): travel along the starting heading, meters.
            dy (float): travel to the left of the starting heading, meters.
            dtheta (float): heading change over the step, radians.
        """
        poses, w = self.particle_set.snapshot()
        n = poses.shape[0]
        if n == 0:
            return

        d_trans = math.hypot(dx, dy)
        if d_trans < self.min_trans:
            # Pure rotation: atan2(dy, dx) on a sub-millimeter translation is reading a
            # bearing off encoder noise, which injects a random rot1/rot2 pair that
            # cancels in the mean but not in the spread. Attribute the whole turn to rot2.
            d_rot1 = 0.0
            d_rot2 = float(dtheta)
        else:
            d_rot1 = math.atan2(dy, dx)
            d_rot2 = float(dtheta) - d_rot1

        # Wrap to [-pi, pi) before the noise scales are computed: a -350 deg rotation is a
        # +10 deg one, and feeding |-6.1| rad into a1*|d_rot| would claim 35x the noise.
        d_rot1 = (d_rot1 + math.pi) % (2 * math.pi) - math.pi
        d_rot2 = (d_rot2 + math.pi) % (2 * math.pi) - math.pi

        # sample(b) is read as zero-mean Gaussian with STANDARD DEVIATION b, not variance
        # b: the book writes sample(b^2) for the variance form, and the table's argument
        # a1*d_rot1 + a2*d_trans is dimensionally a std (rad in, rad out). Read as a
        # variance, a 3 cm step would spread the cloud by sqrt(0.1*0.03) = 5.5 cm, two map
        # cells per step. abs() on the rotations because a right turn is a negative
        # d_rot and a negative sigma is not a thing.
        sigma_rot1 = self.alpha1 * abs(d_rot1) + self.alpha2 * d_trans
        sigma_trans = self.alpha3 * d_trans + self.alpha4 * (abs(d_rot1) + abs(d_rot2))
        sigma_rot2 = self.alpha1 * abs(d_rot2) + self.alpha2 * d_trans

        # Sign of the perturbation is irrelevant for a symmetric Gaussian, but the minus
        # is kept so this reads against the table line for line.
        rot1 = d_rot1 - np.random.normal(0.0, sigma_rot1, n) #TODO: fact check this sigma_rot1
        trans = d_trans - np.random.normal(0.0, sigma_trans, n)
        rot2 = d_rot2 - np.random.normal(0.0, sigma_rot2, n)

        heading = poses[:, 2] + rot1
        x = poses[:, 0] + trans * np.cos(heading)
        y = poses[:, 1] + trans * np.sin(heading)
        theta = heading + rot2 #Particle.__init__ wraps this into [0, 2pi)

        # Particles are pushed blind: one that walks into a wall keeps its weight until
        # measurement_update masks it, which is where that check belongs. Weights carry
        # over untouched, motion does not change how well a hypothesis explained the scan.
        particles = [Particle(x[i], y[i], theta[i]) for i in range(n)]
        self.particle_set.replace(particles, w.tolist())

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
        PLACEHOLDER. Should draw a new particle set proportional to the weights (low
        variance / systematic resampling), reset every weight to 1/N, and only fire when
        the effective sample size drops below a threshold so the cloud is not needlessly
        collapsed on every scan.
        """
        pass

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

        Returns:
            pose (tuple): (x, y, theta), or (0.0, 0.0, 0.0) if the set is empty.
        """
        particle, _ = self.particle_set.best()
        if particle is None:
            return 0.0, 0.0, 0.0
        return particle.pose()

    def effective_sample_size(self) -> float:
        """1 / sum(w^2). Falls toward 1 as the weights collapse onto a single particle."""
        w = self.particle_set.get_weights()
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
        # where it is now. The increment below covers everything since the last scan,
        # INCLUDING the motion that happened after the capture, so the cloud is pushed
        # slightly ahead of what z actually describes. Rewind by the odometry accumulated
        # since the capture before updating.
        dx, dy, dtheta = self.odometry.consume()
        self.motion_update(dx, dy, dtheta)
        self.measurement_update(z)
        self.resample() #PLACEHOLDER