import numpy as np
import cv2 as cv
from pathlib import Path

class ImageUndistorter:
    def __init__(self, config_file_name: str):
        """
        Initializes the undistorter by loading the camera matrix and 
        distortion coefficients from a specified .npz file.
        
        Assumes the file is located in: project_root/config/
        """
        script_path = Path(__file__).resolve()
        self.project_root = script_path.parents[2]  # Adjust levels to match your directory structure
        config_dir = self.project_root / "config"
        config_path = str(config_dir / config_file_name)
        
        # Load calibration data once during initialization
        try:
            with np.load(config_path) as calib_data:
                # Adjust string keys if your variables were saved under different names
                self.K = np.array(calib_data['camera_matrix'], dtype=np.float32)
                print(self.K)
                self.D = np.array(calib_data['distortion_coefficients'], dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Calibration configuration file not found at: {config_path}")
        except KeyError as e:
            raise KeyError(f"Could not find expected key {e} in the .npz file. Check your calibration script keys.")

    def undistort_fisheye(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the stored camera matrix and distortion coefficients 
        to rectify a raw image frame.

        For use with fisheye cameraonl
        
        :param frame: The raw input image (NumPy array)
        :return: The geometrically corrected image
        """
        if frame is None:
            raise ValueError("Input frame is None. Verify the image or camera stream source.")
        Knew = self.K.copy()
        Knew[(0,1), (0,1)] = 1 * Knew[(0,1), (0,1)] #use knew to scale the image, required so code no error
        return cv.fisheye.undistortImage(frame, self.K, self.D, None, Knew)
    
    def undistort_fisheye_save(self, frame: np.ndarray, file_name: str, relative_output_dir: str = "output") -> str:
        """
        undistort and save an image frame to a specified directory relative to the project root.
        Creates the directory automatically if it doesn't exist.
        
        :param frame: The image array to save
        :param file_name: The target file name (e.g., 'corrected_ref2.jpg')
        :param relative_output_dir: The target folder path relative to the project root
        :return: The absolute string path where the file was saved
        """
        if frame is None:
            raise ValueError("Cannot save an empty or None frame.")

        # Resolve the full target directory path
        output_dir = self.project_root / "data" / relative_output_dir
        
        # Create the directory structure if it is not already present
        output_dir.mkdir(parents=True, exist_ok=True)
        
        full_output_path = output_dir / file_name

        ud_frame = self.undistort_fisheye(frame)
        
        # Save using OpenCV
        success = cv.imwrite(str(full_output_path), ud_frame)
        if not success:
            raise IOError(f"Failed to write image to {full_output_path}. Check file permissions or extension.")
        print(f"saved to {full_output_path}")
        return str(full_output_path)


# Example Usage:
if __name__ == "__main__":
    undistorter = ImageUndistorter("fisheye_camera_calibration.npz")
