import os
import glob
import cv2
import numpy as np
from pathlib import Path

def calibrate_fisheye():
    # Directory Structure
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    calibrationDir = project_root / "data" / "callibration"
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    CHECKERBOARD = (8,6)
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    worldPtsCur = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    img_size = []
    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv2.imread(curImgPath)
        if imgBGR is None:
            raise FileNotFoundError(f"Could not load image at {curImgPath}")
        imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
        if not img_size:
            img_size = imgGray.shape[::-1]
            print(f"image size {img_size}")
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        cornersFound, cornersOrg =cv2.findChessboardCorners(imgGray, CHECKERBOARD, flags)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur.reshape(-1, 1, 3).astype(np.float32))
            cornersRefined = cv2.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
        else:
            raise Exception(f"cant detect corner for img {curImgPath}, exiting program")
    N_OK = len(worldPtsList)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            worldPtsList,
            imgPtsList,
            img_size,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")


    #****************** Save intrinsic matrix and distortion coeff *********

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    save_path = config_dir / "fisheye_camera_calibration.npz"

    width, height = img_size

    # Save the arrays and the float together
    np.savez(
        save_path,
        camera_matrix=K,
        distortion_coefficients=D,
        width = width,
        height = height
    )

    print(f"Saved calibration data successfully to: {save_path}")

    #*****************************************************************

if __name__ == "__main__":
    calibrate_fisheye()