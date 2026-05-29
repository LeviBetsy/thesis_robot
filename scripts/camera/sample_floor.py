'''script used to find the metric distances of points on the ground
points on the ground are established with checkerboard print
on the ground that you know the distance of each cell for
'''

import os
import glob
import cv2 as cv
import numpy as np
from pathlib import Path
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

# Now you can use a clean absolute import
from app.camera.undistorter import ImageUndistorter

'''
d_ref_img is the file name for an image of the checkerboard on the floor.
  the image must be in project_root/data/references
  !!!!d_ref_img must be unprocessed and distorted!!!!
config_file is the .npz file storing the camera intrinsic matrix and distortion coefficients
  the config_file must be in project_root/config
square_size is in meter
'''
def find_checker_metric(d_ref_img, config_file, square_size, showPics=False):
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    referenceDir = project_root / "data" / "references"
    cboard_path = str(referenceDir / d_ref_img)

    # Initialize
    nRows = 8
    nCols = 6
    
    
    #**** P_obj definition starting from (0,0,0) to (w*square_size, h*square_size, 9)
    #Note that the vertical coordinate (row) is first index
    #horizontal coordinate ()
    P_obj_list = []
    for _ in range(nRows*nCols):
        P_obj_list.append([0.0, 0.0, 0.0])
    idx = 0
    for row in range(nRows):
        for col in range(nCols):
            P_obj_list[idx][0] = row
            P_obj_list[idx][1] = col
            idx += 1

    
    P_obj = np.array(P_obj_list, dtype=np.float32)
    # Scale the grid by the physical size of your printed squares
    P_obj = P_obj * square_size

    #********************************************************************************

    # Find Corners
    imgBGR = cv.imread(cboard_path)
    imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
    
    #cornersFound is true/false
    cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), None)

    if (cornersFound):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersOrg = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), criteria)
        if (showPics):
            cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersOrg, cornersFound)
            cv.imshow('Undistorted Checkerboard Corners', imgBGR)
            cv.waitKey(0)
        
        IU = ImageUndistorter(config_file)

        sucess, rvec, tvec = cv.solvePnP(P_obj, cornersOrg, IU.camera_matrix, IU.dist_coeffs)
        tvec = tvec.flatten()
        R, _ = cv.Rodrigues(rvec)
        P_cam = np.zeros((nCols*nRows, 3), dtype=np.float32)
        for i in range(nCols*nRows):
            P_cam[i] = (R @ P_obj[i]) + tvec

        print(P_cam)
        

        
    else:
        raise Exception("cant find corners") 

    cv.destroyAllWindows()

if __name__ == "__main__":
  find_checker_metric("ref4.jpg", "camera_calibration.npz", 0.0285, False)