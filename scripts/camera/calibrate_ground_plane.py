'''Offline calibration of the ground plane and the MDE disparity scale.

Two fits are produced, and they are independent of each other:

1. The ground-plane fit  1/Z_c = m*v + b0
   For a planar floor, inverse depth is EXACTLY affine in the image row v.
   This depends only on the camera intrinsics and how the camera is mounted,
   so it is a fixed property of the rig. It is what makes floor-boundary
   ranging scale-free: an image row maps to a metric distance without any
   MDE quantity entering.

2. The disparity fit  1/Z = a*logit(d_rel/max_depth) + b
   DepthAnythingV2's head is Sigmoid()*max_depth (see dpt.py), and the
   relative checkpoint is loaded into it, so the raw d_rel we get back is a
   squashed version of the affine-invariant disparity. Undoing the squash
   makes the relation to inverse depth a single straight line. These are the
   seed values for the per-frame scale calibration.

Writes config/ground_plane.npz.
'''

import os
import sys
import numpy as np
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

MAX_DEPTH = 20.0  # DepthAnythingV2 dpt.py: depth_head(...) * self.max_depth


def to_logit(d_rel, max_depth=MAX_DEPTH, eps=1e-6):
    """Undo DAV2's Sigmoid()*max_depth to recover the affine-invariant disparity."""
    p = np.clip(np.asarray(d_rel, dtype=np.float64) / max_depth, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def from_logit(y, max_depth=MAX_DEPTH):
    """Inverse of to_logit."""
    return max_depth / (1.0 + np.exp(-np.asarray(y, dtype=np.float64)))


def _line_fit(x, y):
    """Least squares y = s*x + c. Returns (s, c, residuals)."""
    A = np.column_stack((x, np.ones_like(x)))
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1], y - A @ coef


def _load_floor_samples(gt_z_file):
    """Returns (u, v, Z_cam) for the labelled floor points."""
    data = np.load(Path(project_root) / "config" / gt_z_file)
    px = data['cornersOrg'].reshape(-1, 2).astype(np.float64)
    Z = data['z_real'].squeeze().astype(np.float64)
    return px[:, 0], px[:, 1], Z


def fit_ground_plane(gt_z_file="z_real_16group.npz", camera_calib="fisheye_calib.npz"):
    """Least-squares fit of 1/Z_c = m*v + b0 over the labelled floor samples.

    Recovering tilt and height from (m, b0):
        1/Z_c(v) = ( sin(t) + cos(t)*(v - cy)/fy ) / h
    so     m  = cos(t) / (h*fy)
           b0 = ( sin(t) - cos(t)*cy/fy ) / h
    giving tan(t) = (b0 + m*cy) / (m*fy)  and  h = cos(t) / (m*fy).

    NOTE: only (m, b0) is observable from a single plane -- the tilt and the
    principal point cy are exactly degenerate. Derive (tilt, height) from the
    fit purely to populate Robot.cam_R / cam_t; never fit them independently.
    """
    from app.robot_module.camera import Camera

    camera = Camera(camera_calib)
    _, v, Z = _load_floor_samples(gt_z_file)
    m, b0, resid = _line_fit(v, 1.0 / Z)

    # residual in 1/Z -> depth error via |dZ| = Z^2 * |d(1/Z)|
    depth_err = np.abs(resid) * Z ** 2

    tilt = np.arctan((b0 + m * camera.cy) / (m * camera.fy))
    height = np.cos(tilt) / (m * camera.fy)

    return {
        "m": m,
        "b0": b0,
        "v_horizon": -b0 / m,
        "tilt_rad": tilt,
        "cam_height": height,
        "rms_mm": float(np.sqrt((depth_err ** 2).mean()) * 1000.0),
        "max_mm": float(depth_err.max() * 1000.0),
        "n_samples": int(len(v)),
    }


def fit_disparity_affine(rel_depth, gt_z_file="z_real_16group.npz", max_depth=MAX_DEPTH):
    """Fit 1/Z = a*logit(d_rel/max_depth) + b on one clean reference frame.

    A single global line in logit space is accurate to a few mm across the
    whole calibrated range, which is why the piecewise-segment fit is not
    needed once the squash is undone.
    """
    rel_depth = np.asarray(rel_depth)
    u, v, Z = _load_floor_samples(gt_z_file)
    d = rel_depth[np.round(v).astype(int), np.round(u).astype(int)]
    y = to_logit(d, max_depth)

    a, b, resid = _line_fit(y, 1.0 / Z)
    depth_err = np.abs(resid) * Z ** 2

    return {
        "a_seed": a,
        "b_seed": b,
        "rms_mm": float(np.sqrt((depth_err ** 2).mean()) * 1000.0),
        "max_mm": float(depth_err.max() * 1000.0),
    }


def save(plane, disparity, out="ground_plane.npz", max_depth=MAX_DEPTH):
    out_path = Path(project_root) / "config" / out
    np.savez(
        out_path,
        m=plane["m"],
        b0=plane["b0"],
        tilt_rad=plane["tilt_rad"],
        cam_height=plane["cam_height"],
        a_seed=disparity["a_seed"],
        b_seed=disparity["b_seed"],
        max_depth=max_depth,
    )
    print(f"wrote {out_path}")
    return out_path


def main(rel_depth_file="data/test/rel_depth_test.npz", gt_z_file="z_real_16group.npz"):
    plane = fit_ground_plane(gt_z_file)
    print("Ground plane fit  1/Z = %.7f*v + %.7f" % (plane["m"], plane["b0"]))
    print("  horizon row      : v = %.2f" % plane["v_horizon"])
    print("  implied tilt     : %.3f deg" % np.degrees(plane["tilt_rad"]))
    print("  implied height   : %.4f m" % plane["cam_height"])
    print("  depth residual   : rms %.2f mm, max %.2f mm (%d samples)"
          % (plane["rms_mm"], plane["max_mm"], plane["n_samples"]))

    npz = np.load(Path(project_root) / rel_depth_file)
    rel_depth = npz[npz.files[0]]
    disparity = fit_disparity_affine(rel_depth, gt_z_file)
    print("\nDisparity fit  1/Z = %.6f*logit(d_rel/%.0f) + %.6f"
          % (disparity["a_seed"], MAX_DEPTH, disparity["b_seed"]))
    print("  depth residual   : rms %.2f mm, max %.2f mm"
          % (disparity["rms_mm"], disparity["max_mm"]))

    save(plane, disparity)


if __name__ == "__main__":
    main()
