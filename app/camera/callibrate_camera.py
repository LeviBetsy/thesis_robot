import os
import glob
import cv2 as cv
import numpy as np
from pathlib import Path

def calibrate(showPics=True):

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    calibrationDir = project_root / "data" / "callibration"
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg'))

    # Initialize
    nRows = 8
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

        if cornersFound == True:
            print("true")
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)
            
            cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
            cv.imshow('Chessboard', imgBGR)
            cv.waitKey(500)
        else:
            print(f"cant find corners {curImgPath}")

    cv.destroyAllWindows()

    # Calibrate
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.fisheye.calibrate(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    print('Camera Matrix:\n', camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    save_path = config_dir / "fisheye_camera_calibration.npz"

    # Save the arrays and the float together
    np.savez(
        save_path,
        rep_error=repError,
        camera_matrix=camMatrix,
        distortion_coefficients=distCoeff
    )

    print(f"Saved calibration data successfully to: {save_path}")

if __name__ == "__main__":
    calibrate(True)