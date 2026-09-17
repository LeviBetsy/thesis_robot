import math
import threading

import numpy as np


'''
    One weighted pose hypothesis. The filter keeps a list of these; the vectorised path in
    MCLLocalization packs them into an (N,3) array and casts them with RayCaster.predict.
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


class ParticleSet:
    """
    A list of Particles paired with a same-length list of weights, guarded by a lock.

    Owns the lock (rather than the filter that uses it) because the set is what's
    actually shared between the measurement/resample thread and any thread reading
    poses() / get_weights() (e.g. estimate()) concurrently.
    """

    def __init__(self, particles=None, weights=None):
        self.particles = list(particles) if particles is not None else []
        self.weights = list(weights) if weights is not None else []
        if len(self.particles) != len(self.weights):
            raise ValueError("particles and weights must be the same length")
        self.lock = threading.Lock()

    def __len__(self):
        with self.lock:
            return len(self.particles)

    def add(self, particle, weight=1.0):
        with self.lock:
            self.particles.append(particle)
            self.weights.append(weight)

    def remove(self, i):
        with self.lock:
            del self.particles[i]
            del self.weights[i]

    def replace(self, particles, weights):
        """Atomically swaps in an entirely new particle/weight set (init, resample)."""
        particles, weights = list(particles), list(weights)
        if len(particles) != len(weights):
            raise ValueError("particles and weights must be the same length")
        with self.lock:
            self.particles = particles
            self.weights = weights

    def poses(self) -> np.ndarray:
        """Snapshot of (x, y, theta) for every particle, shape (N, 3)."""
        with self.lock:
            return np.array([[p.x, p.y, p.theta] for p in self.particles])

    def get_weights(self) -> np.ndarray:
        with self.lock:
            return np.array(self.weights)

    def set_weights(self, weights):
        with self.lock:
            if len(weights) != len(self.particles):
                raise ValueError("weights must match particle count")
            self.weights = list(weights)

    def snapshot(self):
        """Poses and weights read together under one lock, so a concurrent replace()
        can't be interleaved between two separate reads and leave them mismatched."""
        with self.lock:
            poses = np.array([[p.x, p.y, p.theta] for p in self.particles])
            weights = np.array(self.weights)
            return poses, weights

    def best(self):
        """Returns the (particle, weight) pair with the highest weight, or (None, 0.0)
        if empty."""
        with self.lock:
            if not self.particles:
                return None, 0.0
            i = int(np.argmax(self.weights))
            return self.particles[i], self.weights[i]
