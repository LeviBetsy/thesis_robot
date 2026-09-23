import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math
from scipy.stats import norm
from app.util.config import Config




'''
    Beam sensor model for the 16-ray MDE fan: p(z | x, map) per beam, as the standard
    four-component mixture (Thrun, Probabilistic Robotics ch. 6.3).

    Two properties of THIS sensor drove the defaults:

    1. pcd_to_ray_casting pre-fills bins with no point in them with max_range, so
       "wall at 0.4 m" and "saw nothing" are the same float. The p_max spike is what
       keeps a no-return beam from being scored as a confident wall hit (which would
       otherwise drag the whole particle set out into open space). z_max is set high
       for that reason.
    2. sigma_hit reflects monocular-depth error, not lidar error, so it is large.

    All 16 beams come from one image through one network, so they are heavily correlated
    and a straight product of their likelihoods collapses the posterior onto a single
    particle within a couple of updates. `alpha` tempers that: it is the knob to turn
    when particle depletion shows up in testing.
'''
# VERIFIED, HAVE NOT TESTED
class BeamSensorModel:
    def __init__(self, config: Config):
        mconfig = config.measurement_model
        self.max_range = float(config.ray_cast.max_range)
        self.sigma_hit = float(mconfig.sigma_hit)
        self.lambda_short = float(mconfig.lambda_short)
        self.alpha = float(mconfig.alpha)

        total = mconfig.z_hit + mconfig.z_short + mconfig.z_max + mconfig.z_rand
        self.z_hit = mconfig.z_hit / total
        self.z_short = mconfig.z_short / total
        self.z_max = mconfig.z_max / total
        self.z_rand = mconfig.z_rand / total

        # A reading within this of max_range is treated as "no return" by the p_max spike.
        # pcd_to_ray_casting fills empty bins with EXACTLY max_range (np.full) and a real
        # hit lands on a computed value, so this only has to survive the float32 round trip
        # through ZMQ. Keeping it small stops genuine near-max hits being read as no-returns.
        self.max_eps = 0.005

        # TODO: monocular depth error grows with distance, so sigma = sigma_hit + sigma_scale*z_pred
        # is the natural extension here if the residuals turn out range-dependent in testing.

    # VERIFIED, DID NOT TEST
    def log_likelihood(self, z, z_pred) -> np.ndarray:
        """
        Per-particle log p(z | x, map), summed over beams.

        Args:
            z (np.ndarray): the actual reading, shape (n_rays,), meters.
            z_pred (np.ndarray): map-predicted ranges, shape (N, n_rays), meters.
        Returns:
            logp (np.ndarray): shape (N,), float64, unnormalised and untempered.
        """
        z = np.asarray(z, dtype=np.float64).reshape(1, -1) #broadcast over particles
        z_pred = np.asarray(z_pred, dtype=np.float64)

        sigma = self.sigma_hit
        in_range = (z >= 0.0) & (z <= self.max_range)

        # p_hit: gaussian around the predicted range, renormalised over [0, max_range] so
        # particles predicting a range near either end are not penalised for the tail that
        # falls off the interval.

        #eta is actually 1/eta in the book
        eta = norm.cdf(self.max_range, loc=z_pred, scale=sigma) - norm.cdf(0.0, loc=z_pred, scale=sigma)
        eta = np.maximum(eta, 1e-9)
        p_hit = np.exp(-0.5 * ((z - z_pred) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi) * eta)
        p_hit = np.where(in_range, p_hit, 0.0)

        # p_short: something unmapped in the way (a chair leg, a person). Only explains
        # readings SHORTER than the prediction.
        lam = self.lambda_short
        norm_short = np.maximum(1.0 - np.exp(-lam * z_pred), 1e-9)
        p_short = np.where((z >= 0.0) & (z <= z_pred), lam * np.exp(-lam * z) / norm_short, 0.0)

        # p_max: the no-return spike. Carried as an indicator rather than a density, which
        # is the conventional implementation even though it mixes a mass with densities.
        p_max = np.where(z >= self.max_range - self.max_eps, 1.0, 0.0)

        # p_rand: uniform floor so one bad beam cannot zero an otherwise good particle.
        p_rand = np.where(in_range, 1.0 / self.max_range, 0.0)

        p = (self.z_hit * p_hit + self.z_short * p_short
             + self.z_max * p_max + self.z_rand * p_rand) #shape (N, n_rays)

        return np.log(np.maximum(p, 1e-12)).sum(axis=1)

    # VERIFIED, NOT TESTED
    def weights(self, z, z_pred) -> np.ndarray:
        """
        Normalised particle weights for one scan.

        Args:
            z (np.ndarray): the actual reading, shape (n_rays,), meters.
            z_pred (np.ndarray): map-predicted ranges, shape (N, n_rays), meters.
        Returns:
            w (np.ndarray): shape (N,), float64, sums to 1 (all-equal if every particle
                            scored equally badly).
        """
        # logp = self.alpha * self.log_likelihood(z, z_pred) #temper the correlated beams
        #TODO: times self.alpha if test result turns out too sharp or something
        logp = self.log_likelihood(z, z_pred)

        ''' if we naively exponential the log sum, we might get 0 for sum weight
        since w.sum() would be a very very small negative number ~-100
        have each log subtract logp.max shifts the values
        shifting does not affect final answer because /total means cancelling logp.max'''
        w = np.exp(logp - logp.max()) #max-subtract before exp, the raw logs underflow
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            print("ERROR in weights??!?!")
            return np.full(logp.shape[0], 1.0 / logp.shape[0])
        return w / total