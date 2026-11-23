import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(ROOT)

import math

import cv2
import matplotlib
matplotlib.use("Agg") #only saves figures, so it also runs without a display
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from app.localization.map import OccupancyGrid
from app.localization.measurement_model import BeamSensorModel
from app.localization.ray_caster import RayCaster
from app.localization.visualize_map import MapVisualizer
from app.robot_module.robot import Robot
from app.util.config import Config

'''
Thesis Section 5.4.4: tests the beam measurement model on its own, with no motion model,
resampling, or MDE. A noiseless scan is cast from the true pose x*, then particles on a
regular pose grid are scored against it.

Color value (Eq. 5.16), in orders of magnitude below the best particle:
    c = max(C_MIN, (logp - logp_max) / ln 10)
log_likelihood already returns ln w, so no second log is taken.

Run from anywhere:
    python scripts/thesis/measurement_model_test.py
'''

CONFIG_PATH = os.path.join(ROOT, "config/robot_config.json")
MAP_PATH = os.path.join(ROOT, "config/map/empty_enclosure.json")
OUTPUT_DIR = os.path.join(ROOT, "data/thesis")

# The test scenario, not tunables: meters / radians CCW from +x. Faces a corner because a
# pose in the middle of the enclosure would see no walls within max_range.
TRUE_POSE = (0.725, 0.725, math.pi / 4)
N_HEADINGS = 24         # candidate headings 0 to 23*pi/12, in steps of pi/12

C_MIN = -11.0           # color floor, just below the no-wall plateau at c = -10.85
REGION_C = -0.5         # a high-weight region is a connected set with c >= this
TIE_TOL = 1e-9          # poses within this of c = 0 count as tied for the maximum
BATCH_SIZE = 4000       # particles ray cast at once, keeps memory bounded


def setup():
    """
    Builds the test's components from the same config the Pi and laptop run with.

    Returns:
        grid (OccupancyGrid): the empty enclosure.
        caster (RayCaster): casts the n_rays fan using the calibrated camera fov_x.
        model (BeamSensorModel): the beam model with the measurement parameters.
    """
    config = Config.load(CONFIG_PATH)
    robot = Robot(config)
    grid = OccupancyGrid.from_json(MAP_PATH, config, default_value=0, wall_buffer=False)
    caster = RayCaster(grid, robot.camera, config)
    model = BeamSensorModel(config)
    return grid, caster, model


def reference_scan(grid, caster, headings):
    """
    Snaps TRUE_POSE onto the test grid and casts the noiseless reference scan from it.

    Args:
        grid (OccupancyGrid): the map.
        caster (RayCaster): the ray caster.
        headings (np.ndarray): candidate headings in radians, shape (N_HEADINGS,).
    Returns:
        true_idx (tuple): (row, col, heading index) of x* in the test grid.
        true_pose (np.ndarray): snapped (x, y, theta), shape (3,).
        z (np.ndarray): reference scan in meters, shape (n_rays,).
    """
    row, col = grid.world_to_grid(np.array(TRUE_POSE[0]), np.array(TRUE_POSE[1]))
    row, col = int(row), int(col)
    k = int(round(TRUE_POSE[2] / (2 * math.pi / N_HEADINGS))) % N_HEADINGS

    x, y = grid.grid_to_world(row, col)
    true_pose = np.array([x, y, headings[k]])
    z = caster.predict(true_pose[None, :])[0]
    return (row, col, k), true_pose, z


def score_volume(grid, caster, model, z, headings):
    """
    Scores every free cell at every heading against the reference scan.

    Args:
        grid (OccupancyGrid): the map.
        caster (RayCaster): the ray caster.
        model (BeamSensorModel): the beam model.
        z (np.ndarray): reference scan in meters, shape (n_rays,).
        headings (np.ndarray): candidate headings in radians, shape (N_HEADINGS,).
    Returns:
        c (np.ndarray): log10(w / w_max), shape (rows, cols, N_HEADINGS). Unclipped, and
                        NaN on wall cells (the map is loaded with no wall buffer).
    """
    rows, cols = np.nonzero(grid.free_mask())
    xs, ys = grid.grid_to_world(rows, cols)

    logp = np.full((grid.rows, grid.cols, headings.size), np.nan)
    for k, theta in enumerate(headings):
        poses = np.column_stack([xs, ys, np.full(xs.size, theta)])
        scores = [model.log_likelihood(z, caster.predict(poses[i:i + BATCH_SIZE]))
                  for i in range(0, poses.shape[0], BATCH_SIZE)]
        logp[rows, cols, k] = np.concatenate(scores)

    return (logp - np.nanmax(logp)) / math.log(10)


def define_sweeps(true_idx):
    """
    Describes each sweep as a slice of the (rows, cols, N_HEADINGS) score volume, holding
    the components it does not vary at their values in x*.

    Args:
        true_idx (tuple): (row, col, heading index) of x*.
    Returns:
        sweeps (list[dict]): "name", "index" (numpy index into the volume), and
                             "wrap_axis" (the heading axis of the slice, None if fixed).
    """
    row, col, k = true_idx
    every = slice(None)
    return [
        {"name": "x, y", "index": (every, every, k), "wrap_axis": None},
        {"name": "x, theta", "index": (row, every, every), "wrap_axis": 1},
        {"name": "y, theta", "index": (every, col, every), "wrap_axis": 1},
    ]


def count_regions(values, wrap_axis):
    """
    Counts connected regions with c >= REGION_C, wrapping around the heading axis.

    Args:
        values (np.ndarray): c values of one sweep, 2D or 3D. NaN cells never count.
        wrap_axis (int | None): the heading axis, or None if the heading is fixed.
    Returns:
        n_regions (int): number of separate high-weight regions.
    """
    labels, n = ndimage.label(values >= REGION_C)
    if wrap_axis is None or n == 0:
        return n

    parent = list(range(n + 1))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    # the first and last headings are adjacent, so join labels that meet across the seam
    first = np.take(labels, 0, axis=wrap_axis).ravel()
    last = np.take(labels, -1, axis=wrap_axis).ravel()
    for a, b in zip(first, last):
        if a and b:
            parent[find(a)] = find(b)

    return len({find(i) for i in range(1, n + 1)})


def summarize(grid, c, sweeps, true_pose, headings):
    """
    Prints every pose tied for the maximum in each sweep, its error from x*, and the
    number of high-weight regions.

    Args:
        grid (OccupancyGrid): the map.
        c (np.ndarray): score volume from score_volume.
        sweeps (list[dict]): sweeps from define_sweeps.
        true_pose (np.ndarray): snapped x*, shape (3,).
        headings (np.ndarray): candidate headings in radians, shape (N_HEADINGS,).
    """
    for sweep in sweeps:
        in_sweep = np.zeros(c.shape, dtype=bool)
        in_sweep[sweep["index"]] = True
        ties = np.argwhere(in_sweep & (c >= -TIE_TOL))
        n_regions = count_regions(c[sweep["index"]], sweep["wrap_axis"])

        print(f"\nsweep ({sweep['name']}): {n_regions} region(s) at c >= {REGION_C}, "
              f"{len(ties)} tied maximum pose(s)")
        for row, col, k in ties:
            x, y = grid.grid_to_world(row, col)
            pos_err = math.hypot(x - true_pose[0], y - true_pose[1])
            # wrapped to [-pi, pi) so 350 deg vs 10 deg is a 20 deg error
            theta_err = (headings[k] - true_pose[2] + math.pi) % (2 * math.pi) - math.pi
            print(f"  x={x:.4f} m  y={y:.4f} m  theta={math.degrees(headings[k]):5.1f} deg  "
                  f"pos_err={pos_err:.3f} m  theta_err={abs(math.degrees(theta_err)):.1f} deg")


def render_true_rays(grid, caster, true_pose, z):
    """
    Panel 1: the reference scan from x*, drawn in the style of ray_cast_test.py.

    Args:
        grid (OccupancyGrid): the map.
        caster (RayCaster): the ray caster.
        true_pose (np.ndarray): snapped x*, shape (3,).
        z (np.ndarray): reference scan in meters, shape (n_rays,).
    Returns:
        image (np.ndarray): RGB image of the whole map.
    """
    vis = MapVisualizer(grid)
    image = vis.render()
    rows = grid.data.shape[0]
    scale = max(1, int(vis.max_size / max(grid.data.shape)))
    to_px = lambda x, y: tuple(int(round(v)) for v in vis._world_to_pixel(x, y, rows, scale))

    origins, headings = caster.rays(true_pose[None, :])
    origins, headings = origins[0], headings[0] #shape (n_rays, 2) and (n_rays,)
    hits = origins + z[:, None] * np.stack([np.cos(headings), np.sin(headings)], axis=-1)

    color = (0, 0, 255) #BGR
    for k in range(z.shape[0]):
        missed = z[k] >= caster.max_range - 1e-6
        cv2.line(image, to_px(*origins[k]), to_px(*hits[k]),
                 (170, 170, 170) if missed else color, 1, lineType=cv2.LINE_AA)
        if not missed:
            cv2.circle(image, to_px(*hits[k]), 3, color, -1, lineType=cv2.LINE_AA)

    cv2.circle(image, to_px(*hits[0]), 6, (0, 0, 0), 1, lineType=cv2.LINE_AA) #leftmost ray

    centre = to_px(true_pose[0], true_pose[1])
    cv2.circle(image, centre, 5, color, -1, lineType=cv2.LINE_AA)
    cv2.line(image, centre, to_px(*origins[0]), color, 2, lineType=cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_particles(grid, true_pose):
    """
    Panel 2: one test particle in every free cell, all at the true heading.

    Args:
        grid (OccupancyGrid): the map.
        true_pose (np.ndarray): snapped x*, shape (3,), highlighted over the particles.
    Returns:
        image (np.ndarray): RGB image of the whole map.
    """
    rows, cols = np.nonzero(grid.free_mask())
    xs, ys = grid.grid_to_world(rows, cols)
    particles = np.column_stack([xs, ys, np.full(xs.size, true_pose[2])])

    vis = MapVisualizer(grid)
    vis.update_particles(particles, estimate=true_pose)
    return cv2.cvtColor(vis.render(), cv2.COLOR_BGR2RGB)


def plot_xy_test(grid, caster, c, sweep, true_pose, z):
    """
    Saves the x-y test as three side-by-side panels on the same meter axes.

    Args:
        grid (OccupancyGrid): the map.
        caster (RayCaster): the ray caster.
        c (np.ndarray): score volume from score_volume.
        sweep (dict): the x-y sweep from define_sweeps.
        true_pose (np.ndarray): snapped x*, shape (3,).
        z (np.ndarray): reference scan in meters, shape (n_rays,).
    Returns:
        paths (list[str]): the saved PDF and PNG.
    """
    cmap = plt.get_cmap("coolwarm").copy() #blue = poor, red = close, the two ends stay distinct
    cmap.set_bad("black") #walls, matching the map panels

    extent = (0, grid.cols * grid.cell_size, 0, grid.rows * grid.cell_size)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)

    axes[0].imshow(render_true_rays(grid, caster, true_pose, z), extent=extent)
    axes[1].imshow(render_particles(grid, true_pose), extent=extent)
    im = axes[2].imshow(np.maximum(c[sweep["index"]], C_MIN), origin="lower", extent=extent,
                        cmap=cmap, vmin=C_MIN, vmax=0.0, interpolation="nearest")
    axes[2].plot(true_pose[0], true_pose[1], marker="*", markersize=13, markerfacecolor="white",
                 markeredgecolor="black", linestyle="none")

    titles = ("(a) Reference scan from true pose", "(b) Test particles at true heading",
              "(c) Particle weight")
    for ax, title in zip(axes, titles):
        ax.set(xlabel="x (m)", ylabel="y (m)", title=title)

    fig.colorbar(im, ax=axes[2], shrink=0.85, extend="min", label=r"$c = \log_{10}(w / w_{\max})$")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = [os.path.join(OUTPUT_DIR, f"measurement_model_xy.{ext}") for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return paths


def plot_pose_sweeps(grid, c, sweeps, true_pose):
    """
    Saves the x-y, x-theta, and y-theta sweeps side by side on one color scale (Eq. 5.16).

    Args:
        grid (OccupancyGrid): the map.
        c (np.ndarray): score volume from score_volume.
        sweeps (list[dict]): the three sweeps, in define_sweeps order.
        true_pose (np.ndarray): snapped x*, shape (3,).
    Returns:
        paths (list[str]): the saved PDF and PNG.
    """
    cmap = plt.get_cmap("coolwarm").copy() #blue = poor, red = close, the two ends stay distinct
    cmap.set_bad("black") #walls

    x_true, y_true, theta_true = true_pose
    width, length = grid.cols * grid.cell_size, grid.rows * grid.cell_size
    half_step = math.pi / N_HEADINGS #each heading row is centered on its value
    theta_lim = (-half_step, 2 * math.pi - half_step)
    theta_label = r"$\theta$ (rad)"

    # image, extent, x* in panel axes, x label, y label, title
    panels = [
        (c[sweeps[0]["index"]], (0, width, 0, length), (x_true, y_true), "x (m)", "y (m)", "(a) Vary x and y"),
        (c[sweeps[1]["index"]].T, (0, width, *theta_lim), (x_true, theta_true), "x (m)", theta_label, r"(b) Vary x and $\theta$"),
        (c[sweeps[2]["index"]].T, (0, length, *theta_lim), (y_true, theta_true), "y (m)", theta_label, r"(c) Vary y and $\theta$"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for ax, (image, extent, marker, xlabel, ylabel, title) in zip(axes, panels):
        im = ax.imshow(np.maximum(image, C_MIN), origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=C_MIN, vmax=0.0, interpolation="nearest")
        ax.set_box_aspect(1)
        ax.plot(*marker, marker="*", markersize=13, markerfacecolor="white",
                markeredgecolor="black", linestyle="none")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if ylabel == theta_label:
            ax.set_yticks(np.arange(4) * math.pi / 2, ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])

    fig.colorbar(im, ax=axes, shrink=0.85, extend="min", label=r"$c = \log_{10}(w / w_{\max})$")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = [os.path.join(OUTPUT_DIR, f"measurement_model_sweeps.{ext}") for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    grid, caster, model = setup()
    headings = np.arange(N_HEADINGS) * 2 * math.pi / N_HEADINGS

    true_idx, true_pose, z = reference_scan(grid, caster, headings)
    c = score_volume(grid, caster, model, z, headings)
    sweeps = define_sweeps(true_idx)

    print(f"true pose: x={true_pose[0]:.4f} m  y={true_pose[1]:.4f} m  "
          f"theta={math.degrees(true_pose[2]):.1f} deg  (c = {c[true_idx]:.3f})")
    print(f"fov_x: {math.degrees(caster.fov_x):.1f} deg, {caster.n_rays} rays, "
          f"max_range {caster.max_range} m, {int(grid.free_mask().sum())} free cells")
    print(f"reference scan (m): {np.round(z, 3)}")

    summarize(grid, c, sweeps, true_pose, headings)

    for path in plot_xy_test(grid, caster, c, sweeps[0], true_pose, z) + plot_pose_sweeps(grid, c, sweeps, true_pose):
        print(f"saved {path}")
