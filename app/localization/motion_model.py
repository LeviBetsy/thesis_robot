import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math
from app.util.config import Config




'''
    Odometry motion model: samples x_t ~ p(x_t | u_t, x_{t-1}) for every particle, with
    noise drawn independently per particle, so the cloud spreads to reflect encoder drift
    (Thrun, Probabilistic Robotics, Table 5.6).

    The step is decomposed into rotate / translate / rotate, each of the three is
    perturbed, and the perturbed triple is replayed from each particle's own pose.

    Pure array math, like BeamSensorModel: it never touches the ParticleSet or its lock,
    MCLLocalization.motion_update owns the snapshot / replace around it.
'''
class OdometryMotionModel:
    def __init__(self, config: Config):
        mconfig = config.mcl
        self.alpha1 = float(mconfig.motion_alpha1)
        self.alpha2 = float(mconfig.motion_alpha2)
        self.alpha3 = float(mconfig.motion_alpha3)
        self.alpha4 = float(mconfig.motion_alpha4)
        self.min_trans = float(mconfig.motion_min_trans)

    def sample(self, poses, dx, dy, dtheta) -> np.ndarray:
        """
        Pushes every pose through the odometry increment measured since the last update.

        FRAME: (dx, dy) are the odometry increment measured RELATIVE to where the robot
        was at the start of the scan, x along the heading it had then. That is the book's
        (x_bar' - x_bar, y_bar' - y_bar) with theta_bar already taken out, so
        delta_rot1 = atan2(dy, dx) needs no further subtraction. Every particle then
        applies delta_rot1 relative to its OWN theta, which is the point of the
        decomposition: particles facing different ways must move in different world
        directions. For a differential drive this increment is
        dx = d*cos(dtheta/2), dy = d*sin(dtheta/2) with d the mid-point arc length.

        Args:
            poses (np.ndarray): shape (N, 3), (x, y, theta) per particle, meters and radians.
            dx (float): travel along the starting heading, meters.
            dy (float): travel to the left of the starting heading, meters.
            dtheta (float): heading change over the step, radians.
        Returns:
            new_poses (np.ndarray): shape (N, 3), float64. theta is NOT wrapped,
                                    Particle.__init__ wraps it into [0, 2pi).
        """
        poses = np.asarray(poses, dtype=np.float64)
        n = poses.shape[0]

        d_trans = math.hypot(dx, dy)
        if d_trans < self.min_trans:
            # Pure rotation: atan2(dy, dx) on a sub-millimeter translation is reading a
            # bearing off encoder noise, which injects a random rot1/rot2 pair that
            # cancels in the mean but not in the spread. Attribute the whole turn to rot2.
            d_rot1 = 0.0
            d_rot2 = float(dtheta)
        else:
            d_rot1 = math.atan2(dy, dx)
            if abs(d_rot1) > math.pi / 2:
                # The translation vector points backward relative to the starting
                # heading (e.g. dx<0, dy~=0 from driving straight in reverse). Left as
                # atan2, this reads as "spin ~180 deg, walk forward, spin back", which
                # is correct in the mean but feeds alpha1*|d_rot1| and
                # alpha4*(|d_rot1|+|d_rot2|) a phantom ~pi rotation that a straight
                # reverse never made - a 3cm backward step then gets ~20x the heading
                # noise and ~40x the translation noise of the same step forward (see
                # motion_alpha1..4 in config/robot_config.json). Flip the sign of the
                # distance instead so reverse is "walk backward, no turn".
                d_trans = -d_trans
                d_rot1 = d_rot1 - math.pi if d_rot1 >= 0 else d_rot1 + math.pi
            d_rot2 = float(dtheta) - d_rot1

        # Wrap to [-pi, pi) before the noise scales are computed: a -350 deg rotation is a
        # +10 deg one, and feeding |-6.1| rad into a1*|d_rot| would claim 35x the noise.
        d_rot1 = (d_rot1 + math.pi) % (2 * math.pi) - math.pi
        d_rot2 = (d_rot2 + math.pi) % (2 * math.pi) - math.pi

        # sample(b) is read as zero-mean Gaussian with STANDARD DEVIATION b, not variance
        # b: the book writes sample(b^2) for the variance form, and the table's argument
        # a1*d_rot1 + a2*d_trans is dimensionally a std (rad in, rad out). Read as a
        # variance, a 3 cm step would spread the cloud by sqrt(0.1*0.03) = 5.5 cm, two map
        # cells per step. abs() everywhere here because a right turn is a negative d_rot,
        # and reverse (above) is a negative d_trans, and a negative sigma is not a thing.
        d_trans_abs = abs(d_trans)
        sigma_rot1 = self.alpha1 * abs(d_rot1) + self.alpha2 * d_trans_abs
        sigma_trans = self.alpha3 * d_trans_abs + self.alpha4 * (abs(d_rot1) + abs(d_rot2))
        sigma_rot2 = self.alpha1 * abs(d_rot2) + self.alpha2 * d_trans_abs

        # Sign of the perturbation is irrelevant for a symmetric Gaussian, but the minus
        # is kept so this reads against the table line for line.
        rot1 = d_rot1 - np.random.normal(0.0, sigma_rot1, n) #TODO: fact check this sigma_rot1
        trans = d_trans - np.random.normal(0.0, sigma_trans, n)
        rot2 = d_rot2 - np.random.normal(0.0, sigma_rot2, n)

        heading = poses[:, 2] + rot1
        x = poses[:, 0] + trans * np.cos(heading)
        y = poses[:, 1] + trans * np.sin(heading)
        theta = heading + rot2

        return np.stack([x, y, theta], axis=1)
