import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import math

#TODO: VERIFY

def _norm_cdf(x) -> np.ndarray:
    """
    Standard normal CDF, vectorised. Uses the Abramowitz & Stegun 7.1.26 erf approximation
    (max error 1.5e-7) so this works on the Pi without pulling in scipy.
    """
    # erf(z) for z >= 0, mirrored for negatives
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


'''
    Beam sensor model for the 16-ray MDE fan: p(z | x, map) per beam, as the standard
    four-component mixture (Thrun, Probabilistic Robotics ch. 6.3).

    Two properties of THIS sensor drove the defaults:

    1. pcd_to_ray_casting pre-fills bins with no point in them with max_range, so
       "wall at 0.6 m" and "saw nothing" are the same float. The p_max spike is what
       keeps a no-return beam from being scored as a confident wall hit (which would
       otherwise drag the whole particle set out into open space). z_max is set high
       for that reason.
    2. sigma_hit reflects monocular-depth error, not lidar error, so it is large.

    All 16 beams come from one image through one network, so they are heavily correlated
    and a straight product of their likelihoods collapses the posterior onto a single
    particle within a couple of updates. `alpha` tempers that: it is the knob to turn
    when particle depletion shows up in testing.
'''
class BeamSensorModel:
    def __init__(self, max_range=0.6, sigma_hit=0.08, lambda_short=2.0,
                 z_hit=0.65, z_short=0.05, z_max=0.20, z_rand=0.10, alpha=0.25):
        self.max_range = float(max_range)
        self.sigma_hit = float(sigma_hit)
        self.lambda_short = float(lambda_short)
        self.alpha = float(alpha)

        total = z_hit + z_short + z_max + z_rand
        self.z_hit = z_hit / total
        self.z_short = z_short / total
        self.z_max = z_max / total
        self.z_rand = z_rand / total

        # A reading within this of max_range is treated as "no return" by the p_max spike.
        # pcd_to_ray_casting fills empty bins with EXACTLY max_range (np.full) and a real
        # hit lands on a computed value, so this only has to survive the float32 round trip
        # through ZMQ. Keeping it small stops genuine near-max hits being read as no-returns.
        self.max_eps = 0.005

        # TODO: monocular depth error grows with distance, so sigma = sigma_hit + sigma_scale*z_pred
        # is the natural extension here if the residuals turn out range-dependent in testing.

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
        eta = _norm_cdf((self.max_range - z_pred) / sigma) - _norm_cdf((0.0 - z_pred) / sigma)
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
        logp = self.alpha * self.log_likelihood(z, z_pred) #temper the correlated beams

        w = np.exp(logp - logp.max()) #max-subtract before exp, the raw logs underflow
        total = w.sum()
        if total <= 0.0 or not np.isfinite(total):
            return np.full(logp.shape[0], 1.0 / logp.shape[0])
        return w / total


if __name__ == "__main__":
    model = BeamSensorModel()
    z_pred = np.full((1, 16), 0.30)

    # A reading right on the prediction should score far above one that is well off it.
    print("exact match      :", model.log_likelihood(np.full(16, 0.30), z_pred)[0].round(3))
    print("off by 5 cm      :", model.log_likelihood(np.full(16, 0.35), z_pred)[0].round(3))
    print("off by 25 cm     :", model.log_likelihood(np.full(16, 0.55), z_pred)[0].round(3))

    # The no-return spike, the reason the full beam model is worth its extra parameters:
    # reading exactly max_range while the map predicts a wall at 0.30 must be tolerated far
    # better than an in-range reading that is no further from the prediction. Without p_max
    # these two would score alike and every empty bin would shove particles into open space.
    print("no-return (0.60) :", model.log_likelihood(np.full(16, 0.60), z_pred)[0].round(3))
    print("in-range at 0.55 :", model.log_likelihood(np.full(16, 0.55), z_pred)[0].round(3))

    # And a short reading (something unmapped in the way) must beat a long one by the same
    # margin the p_short component buys it.
    print("short at 0.15    :", model.log_likelihood(np.full(16, 0.15), z_pred)[0].round(3))
    print("long  at 0.45    :", model.log_likelihood(np.full(16, 0.45), z_pred)[0].round(3))
