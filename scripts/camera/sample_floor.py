'''script used to find the metric distances of points on the ground
points on the ground are established with checkerboard print
on the ground that you know the distance of each cell for
'''

import os
import glob
import cv2
import numpy as np
from pathlib import Path
import sys
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

if project_root not in sys.path:
    sys.path.append(project_root)

from app.module.camera import Camera

'''
d_ref_img is the file name for an image of the checkerboard on the floor.
  the image must be in project_root/data/references
  !!!!d_ref_img must be undistorted!!!!
config_file is the .npz file storing the camera intrinsic matrix and distortion coefficients
  the config_file must be in project_root/config
square_size is in meter
'''
def find_checker_metric(d_ref_img, config_file, square_size, showPics=False):
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]  # Goes up two levels from scripts/
    referenceDir = project_root / "data" / "references"
    cboard_path = str(referenceDir / d_ref_img) + ".jpg"
    
    pattern_x = 8
    pattern_y = 6

    # Find Corners
    imgBGR = cv2.imread(cboard_path)
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    
    #cornersFound is true/false
    cornersFound, cornersOrg = cv2.findChessboardCorners(imgGray, (pattern_x, pattern_y), None)
    cornersOrg[7, 0, 0] = 228
    cornersOrg[7, 0, 1] = 186

    manualCorners = [[236, 171], [243, 158], [249, 147], [255,136], [261,127], [265,120], [269, 113], [272, 106],
    [275, 171], [280, 158], [284, 147], [287.5, 136], [290, 128], [293, 119], [295.5, 113], [298.5, 106],
    [314, 171], [316, 158], [317,146], [319,136], [320,128], [321,120], [322,112], [323, 106],
    [353,171], [352,158], [351,146], [351, 136], [350, 128], [349,120],[349,113], [348,105],
    [392,170], [389,157], [385.5,146], [382,136], [380,128], [378,120], [376, 113], [374,105],
    [431, 169], [424,157], [419,146], [414, 136], [410, 128], [406,120], [402, 112], [399, 106]
    ]
    manualOrg = np.array(manualCorners, dtype= np.float32).reshape(-1, 1, 2) # N,1,2

    orig_chunks = cornersOrg.reshape(6, 8, -1)
    new_chunks = manualOrg.reshape(6, 8, -1)
    combined = np.concatenate((orig_chunks, new_chunks), axis=1)
    final_array = combined.reshape(96, 1, 2)

    # Initialize
    pattern_x = 16
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


    if (cornersFound):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # cornersOrg = cv2.cornerSubPix(imgGray, final_array, (11, 11), (-1, -1), criteria)
        cornersOrg = final_array
        if (showPics):
            cv2.drawChessboardCorners(imgBGR, (pattern_x, pattern_y), cornersOrg, cornersFound)
            cv2.imshow('Checkerboard Corners', imgBGR)
            cv2.waitKey(0)
        
        #************************** Solve Pnp *************************************
        camera = Camera(config_file)
        sucess, rvec, tvec = cv2.solvePnP(P_obj, cornersOrg, camera.K, None)
        tvec = tvec.flatten()
        R, _ = cv2.Rodrigues(rvec)
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
            cv2.circle(imgPlot, (px_x, px_y), radius=2, color=(0, 255, 0), thickness=-1)
            
            # text = f"{depth:.2f}m"
            
            # # 5. Draw the text slightly above the corner point
            # # Parameters: image, text, bottom-left corner of text, font, scale, color, thickness
            # cv2.putText(imgPlot, text, (px_x - 15, px_y - 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
        
        #Saving Image
        output_dir = project_root / "data" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        img_save_path = str(output_dir / f"mapped_fisheye_{d_ref_img}.jpg")
        cv2.imwrite(img_save_path, imgPlot)
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


if __name__ == "__main__":
  find_checker_metric("ref13", "fisheye_calib.npz", 0.0285, False)