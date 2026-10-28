import math
import threading

import numpy as np


class ParticleSet:
    """
    N weighted pose hypotheses: an (N, 3) array of (x, y, theta) and an (N,) array of
    weights, guarded by a lock.

    Stored as arrays rather than a list of Particles because every filter stage (motion
    model, ray caster, resampler) works on the (N, 3) array anyway, so this skips a
    per-particle Python loop in and out of every stage.

    Owns the lock (rather than the filter that uses it) because the set is what's
    actually shared between the measurement/resample thread and any thread reading
    poses() / get_weights() concurrently. Every read returns a copy, so a caller never
    holds a view the next replace() could change under it.
    """

    def __init__(self, poses=None, weights=None):
        if poses is None and weights is None:
            poses, weights = np.empty((0, 3)), np.empty(0)
        self.particle_poses, self.weights = self._as_arrays(poses, weights)
        self.lock = threading.Lock()

    @staticmethod
    def _as_arrays(poses, weights):
        """Copies into float64 arrays, checks the lengths match, wraps theta."""
        poses = np.array(poses, dtype=np.float64).reshape(-1, 3)
        weights = np.array(weights, dtype=np.float64).reshape(-1)
        if poses.shape[0] != weights.shape[0]:
            raise ValueError("poses and weights must be the same length")
        poses[:, 2] %= 2 * math.pi #match Robot.set_robot_pose's wrapping
        return poses, weights

    def __len__(self):
        with self.lock:
            return self.particle_poses.shape[0]

    def replace(self, poses, weights):
        """Atomically swaps in an entirely new pose/weight set (init, motion, resample).

        Args:
            poses (np.ndarray): shape (N, 3), theta is wrapped into [0, 2*pi) here.
            weights (np.ndarray): shape (N,).
        """
        poses, weights = self._as_arrays(poses, weights)
        with self.lock:
            self.particle_poses = poses
            self.weights = weights

    def poses(self) -> np.ndarray:
        """Snapshot of (x, y, theta) for every particle, shape (N, 3)."""
        with self.lock:
            return self.particle_poses.copy()

    def get_weights(self) -> np.ndarray:
        with self.lock:
            return self.weights.copy()

    def set_weights(self, weights):
        weights = np.array(weights, dtype=np.float64).reshape(-1)
        with self.lock:
            if weights.shape[0] != self.particle_poses.shape[0]:
                raise ValueError("weights must match particle count")
            self.weights = weights

    def snapshot(self):
        """Poses and weights read together under one lock, so a concurrent replace()
        can't be interleaved between two separate reads and leave them mismatched."""
        with self.lock:
            return self.particle_poses.copy(), self.weights.copy()

    def best(self):
        """Returns ((x, y, theta), weight) for the highest-weight particle, or
        (None, 0.0) if empty."""
        with self.lock:
            if self.particle_poses.shape[0] == 0:
                return None, 0.0
            i = int(np.argmax(self.weights))
            return tuple(float(v) for v in self.particle_poses[i]), float(self.weights[i])
