import cv2
import torch
import os
import sys
import numpy as np
import matplotlib
import time
from pathlib import Path
import matplotlib.pyplot as plt

# Get the absolute path of the directory two levels up (project root)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

from app.models.DAV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2



class DepthAnythingPredictor:
    def __init__(self, encoder="vits", device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        model_configs = {
            "vits": {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            "vitb": {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            "vitl": {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        if encoder not in model_configs:
            raise ValueError(f"Invalid encoder: {encoder}")
        
        encoder = 'vits' # or 'vits', 'vitb'
        dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 20 # 20 for indoor model, 80 for outdoor model

        # Load model
        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        self.model.load_state_dict(torch.load(f'app/models/DAV2_checkpoint/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location="cpu"))
        self.model = self.model.to(self.device).eval()

        # Default colormap
        self.cmap = matplotlib.colormaps["turbo"]

        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]

    def infer_image(self, full_img_name):
        """Run depth estimation on a single image in data/test."""

        test_dir = self.project_root / "data" / "test"
        img_path = test_dir / str(full_img_name)


        image = cv2.imread(img_path)

        depth = self.model.infer_image(image)  # float32 depth tensor
        return depth
    
    def infer_image_save(self, full_img_name):
        depth_array = self.infer_image(full_img_name)

        output_dir = self.project_root / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_img_path = output_dir / f"DAV2_{full_img_name}"
        colored_image = self.colorize(depth_array)
        plt.imsave(output_img_path, colored_image)

        return depth_array

    def save_depth_bin(self, depth_array, full_bin_name):
        depth_map_dir = self.project_root / "data" / "depth_map"
        depth_map_file_path = depth_map_dir / f"{full_bin_name}"
        depth_array.tofile(depth_map_file_path)


    def colorize(self, depth):
        """Convert depth map to turbo colormap."""
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        colormap = self.cmap(depth_norm)[:, :, :3]  # RGB float (0-1)
        colormap = (colormap * 255).astype(np.uint8)
        return cv2.cvtColor(colormap, cv2.COLOR_RGB2BGR)

    def infer_video(self, video_path):
        """Run depth estimation on video and save colored depth video."""
        cap = cv2.VideoCapture(video_path)

        prev_time = time.perf_counter()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current_time = time.perf_counter()
            elapsed = current_time - prev_time
            print(f"Iteration took: {elapsed:.4f} seconds ({1/elapsed:.2f} FPS)")

            prev_time = current_time
            depth = self.infer_image(frame)

            color = self.colorize(depth)
            # Optional live preview
            cv2.imshow("DepthAnythingV2", color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    depth_model = DepthAnythingPredictor(encoder="vits")

    

    #Inference on image
    depth = depth_model.infer_image_save("cube_60cm.jpg")
    depth_model.save_depth_bin(depth, "DAV2_cube60_depthmap.bin")



    # # Inference on video
    # depth_model.infer_video(0)