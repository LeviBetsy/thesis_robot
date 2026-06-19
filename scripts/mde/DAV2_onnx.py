import cv2
import os
import sys
import numpy as np
import matplotlib
import time
from pathlib import Path
import matplotlib.pyplot as plt
import onnxruntime as ort  # Swapped torch for onnxruntime

# Get the absolute path of the directory two levels up (project root)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)


class DepthAnythingONNXPredictor:
    def __init__(self, onnx_model_path, device=None):
        """
        Initialize the ONNX runtime session.
        onnx_model_path: Path to your 'model.onnx' file
        device: 'cuda', 'cpu', etc. (Will auto-detect if None)
        """
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]


        # Determine execution providers based on hardware availability
        # if device == "mps" or device == "coreml" or (device is None and "CoreMLExecutionProvider" in ort.get_available_providers()):
        #     providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda" or (device is None and "CUDAExecutionProvider" in ort.get_available_providers()):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        print(f"Initializing ONNX Runtime with providers: {providers}")
        self.session = ort.InferenceSession(self.project_root/onnx_model_path, providers=providers)
        
        # Get model input names and shapes
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # Usually [1, 3, H, W]
        
        # Extract target height and width from the ONNX model input profile
        self.input_height = 518
        self.input_width = 518

        # Image normalization constants (ImageNet defaults used by Depth Anything)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # Default colormap
        self.cmap = matplotlib.colormaps["turbo"]

        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]

    def _preprocess(self, image):
        """
        Prepare the raw BGR image for the ONNX model matching PyTorch standards.
        """
        # 1. Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to the exact dimensions the ONNX model expects
        img_resized = cv2.resize(img_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        
        # 3. Scale to [0, 1] and Normalize
        img_normalized = (img_resized.astype(np.float32) / 255.0 - self.mean) / self.std
        
        # 4. HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
        img_chw = img_normalized.transpose(2, 0, 1)
        
        # 5. Add Batch dimension -> [1, 3, H, W]
        img_batch = np.expand_dims(img_chw, axis=0).astype(np.float32)
        return img_batch

    def infer_image(self, full_img_name):
        """Run depth estimation on a single image in data/test using ONNX."""
        test_dir = self.project_root / "data" / "test"
        img_path = test_dir / str(full_img_name)

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image from {img_path}")
            
        return self.infer_frame(image)

    def infer_frame(self, frame):
        """Process an in-memory BGR frame (shared by image and video pipelines)"""
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess frame
        blob = self._preprocess(frame)
        
        # Run inference via ONNX Runtime
        onnx_outputs = self.session.run(None, {self.input_name: blob})
        
        # Extract depth map tensor and squeeze batch/channel dimensions
        depth = np.squeeze(onnx_outputs[0])
        
        # Resize depth map back to the original image dimensions
        depth_resized = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth_resized
    
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
        """Run depth estimation on video or camera stream and save colored depth video."""
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
            
            # Use unified frame inference logic
            depth = self.infer_frame(frame) 
            color = self.colorize(depth)
            
            # Live preview
            cv2.imshow("DepthAnythingV2_ONNX", color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path to your downloaded standard FP32 ONNX model file
    # Replace this with your actual local path to the file
    onnx_path = "app/models/DAV2_onnx/model.onnx"

    # Initialize the predictor with the ONNX file
    depth_model = DepthAnythingONNXPredictor(onnx_model_path=onnx_path, device="mps")

    # # Inference on image
    # depth = depth_model.infer_image_save("cube_60cm.jpg")
    # depth_model.save_depth_bin(depth, "DAV2_cube60_depthmap.bin")

    # Inference on live video / camera
    depth_model.infer_video(0)