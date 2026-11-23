'''Annotate the floor sample points saved by sample_floor.py onto an image.
Loads the pixel coordinates (cornersOrg) and the depth (z_real) from the .npz
and draws each point with its depth z next to it.
'''

import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]


def load_floor_samples(img_rel_path, npz_name):
    '''img_rel_path is relative to project_root/data, npz_name is in project_root/config'''
    img_path = project_root / "data" / img_rel_path
    npz_path = project_root / "config" / npz_name

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"could not read image: {img_path}")

    data = np.load(npz_path)
    corners = data["cornersOrg"].reshape(-1, 2)  # N x 2 pixel (u, v)
    z_real = data["z_real"].reshape(-1)          # N depths in meters
    return img, corners, z_real


def annotate_depth(img, corners, z_real, font_scale=0.25):
    out = img.copy()
    for (u, v), z in zip(corners, z_real):
        px = (int(round(u)), int(round(v)))
        cv2.circle(out, px, radius=2, color=(0, 255, 0), thickness=-1)
        cv2.putText(out, f"{z:.2f}", (px[0] + 3, px[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    img, corners, z_real = load_floor_samples("test/ref13.jpg", "z_real_fix_checker_size.npz")
    annotated = annotate_depth(img, corners, z_real)

    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "annotated_depth_ref13.jpg"
    cv2.imwrite(str(save_path), annotated)
    print(f"Saved annotated image ({len(corners)} points) to: {save_path}")

    cv2.imshow("Floor depth", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
