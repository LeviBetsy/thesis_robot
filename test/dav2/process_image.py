import sys
from pathlib import Path
import cv2

# Project root is two levels up (test/dav2/ -> thesis_robot/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.perception.dav2_pth import DepthAnythingPredictor


def process_image(image_name, dav2=None):
    """Loads data/<image_name>, runs DepthAnythingV2, saves the turbo-colorized depth to data/output/dav2_<stem>.jpg."""
    dav2 = dav2 or DepthAnythingPredictor()

    image_path = PROJECT_ROOT / "data" / image_name
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    depth = dav2.infer(frame)
    colored = dav2.colorize(depth)  # BGR, so cv2.imwrite keeps colors correct

    output_dir = PROJECT_ROOT / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dav2_{Path(image_name).stem}.jpg"
    cv2.imwrite(str(output_path), colored)
    print(f"Saved colorized depth to {output_path}")
    return depth


if __name__ == "__main__":
    process_image("ref27.jpg")
    # process_image("checkerboard/ref13.jpg")
