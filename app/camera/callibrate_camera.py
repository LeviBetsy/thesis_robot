import os
import glob
import cv2 as cv
import numpy as np
from pathlib import Path

def calibrate_fisheye():
    # Directory Structure
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    calibrationDir = project_root / "data" / "callibration"
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize
    calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv.fisheye.CALIB_CHECK_COND+cv.fisheye.CALIB_FIX_SKEW
    CHECKERBOARD = (8,6)
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    worldPtsCur = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    img_size = []
    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        if not img_size:
            img_size = imgGray.shape[::-1]
            print(f"image size {img_size}")
        
        cornersFound, cornersOrg =cv.findChessboardCorners(imgGray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH+cv.CALIB_CB_FAST_CHECK+cv.CALIB_CB_NORMALIZE_IMAGE)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur.reshape(-1, 1, 3).astype(np.float32))
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
        else:
            raise Exception(f"cant detect corner for img {curImgPath}, exiting program")
    N_OK = len(worldPtsList)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    print("WORLD PTS LIST")
    print(len(worldPtsList))
    print(len(worldPtsList[0]))
    print("IMG PT LIST")
    print(len(imgPtsList))
    print(len(imgPtsList[0]))
    rms, _, _, _, _ = \
        cv.fisheye.calibrate(
            worldPtsList,
            imgPtsList,
            img_size,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print("Retavl={retval}")


    #***************************** save callibration matrix and distortion coeff *********

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    save_path = config_dir / "fisheye_camera_calibration.npz"

    # Save the arrays and the float together
    np.savez(
        save_path,
        camera_matrix=K,
        distortion_coefficients=D
    )

    print(f"Saved calibration data successfully to: {save_path}")

    #*****************************************************************

def calibrate_pinhole():
    pass

if __name__ == "__main__":
    calibrate_fisheye()