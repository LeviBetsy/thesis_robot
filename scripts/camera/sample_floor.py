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
import math

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
    pattern_x = 8
    pattern_y = 6
    
    #**** P_obj definition starting from (0,0,0) ******************
    #Note that the vertical coordinate (row) is first index
    #horizontal coordinate ()
    P_obj_list = []
    for y in range(pattern_y):
        for x in range(pattern_x):
            # Scale immediately during generation
            P_obj_list.append([x * square_size, y * square_size, 0.0])
    P_obj = np.array(P_obj_list, dtype=np.float32)

    #********************************************************************************

    # Find Corners
    imgBGR = cv.imread(cboard_path)
    imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
    
    #cornersFound is true/false
    cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (pattern_x, pattern_y), None)

    if (cornersFound):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersOrg = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), criteria)
        if (showPics):
            cv.drawChessboardCorners(imgBGR, (pattern_x, pattern_y), cornersOrg, cornersFound)
            cv.imshow('Checkerboard Corners', imgBGR)
            cv.waitKey(0)
        
        #************************** Solve Pnp *************************************
        IU = ImageUndistorter(config_file)
        undistorted = cv.fisheye.undistortPoints(cornersOrg, IU.K, IU.D)
        sucess, rvec, tvec = cv.solvePnP(P_obj, undistorted, np.eye(3), None)
        tvec = tvec.flatten()
        R, _ = cv.Rodrigues(rvec)
        P_cam = np.zeros((pattern_y*pattern_x, 3), dtype=np.float32)
        for i in range(pattern_y*pattern_x):
            P_cam[i] = (R @ P_obj[i]) + tvec
        #**************************************************************************


        #******** plotting depth to each point on the image for output image******
        imgPlot = imgBGR.copy()
        for i in range(len(cornersOrg)):
            # cornersOrg shape is typically (N, 1, 2)
            px_x = int(cornersOrg[i][0][0])
            px_y = int(cornersOrg[i][0][1])
            
            depth = P_cam[i][2]
            # depth = math.sqrt(P_cam[i][0]**2 + P_cam[i][1]**2 + P_cam[i][2]**2)
            cv.circle(imgPlot, (px_x, px_y), radius=4, color=(0, 255, 0), thickness=-1)
            
            text = f"{depth:.2f}m"
            
            # 5. Draw the text slightly above the corner point
            # Parameters: image, text, bottom-left corner of text, font, scale, color, thickness
            cv.putText(imgPlot, text, (px_x - 15, px_y - 10), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv.LINE_AA)
        
        #Saving Image
        output_dir = project_root / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        img_save_path = str(output_dir / f"mapped_fisheye_{d_ref_img}")
        cv.imwrite(img_save_path, imgPlot)
        print(f"Successfully saved mapped image to: {img_save_path}")

        #Saving npz file
        configDir = project_root / "config"
        z_real_file = str(configDir / f"z_real_{d_ref_img}.npz")
        cornersOrg_flat = cornersOrg.reshape(-1, 2) #cornersOrg_flatis Nx2
        z_real = P_cam[:, 2:3] # Using [:, 2:3] slices the 3rd column while keeping it 2D
        np.savez(z_real_file, cornersOrg=cornersOrg_flat, z_real=z_real)
        print(f"Successfully saved zreal map to: {z_real_file}")
        #**************************************************  
    else:
        raise Exception("cant find corners") 

    cv.destroyAllWindows()

if __name__ == "__main__":
  find_checker_metric("ref6.jpg", "fisheye_camera_calibration.npz", 0.0285, False)