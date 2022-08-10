

"""

Usage:

To get the unique K and D array values for a lense
1) collect images of a black and white checkerboard at multiple angles
   with lense. Make sure that the checkerboard is flat while you are 
   collecting images

checkerboard printed from: http://clipart-library.com/clipart/6ipokG9AT.htm

2) put all images into working directory
3) import this .py file and run:
    calibration.calibrate(corners_height, corners_width)
    if you leave the parameters blank the default is 6,8 wich are the 
    correct parameters for the checkerbaord given above
4) replace the DIM K and D values in this file with the ones that calibration.calibrate()
   gives you for your lense

5) to dewarp a single frame or image call:
   calibration.dewarp(img)


Calibrate Function:

Make sure that the images of the checkerbaord are present in 
the working directory

the calibrate function takes the number of corners of our test 
checkerboard's height and width as parameters.

the calibrate function will return the array values of K and D 
these arrays are unique to the lense and can be used to undistort 
any image that the lense creates


dewarp function:

Make sure that you have updated the K D and DIM values to their 
proper dimensions in this file

Parameter: an image or frame to dewarp
Returns: a dewarped image


"""

import cv2
import numpy as np
import os
import glob


#these are example dimensions for a GOPRO hero 3 that should be replaced after 
#calibration.calibrate() is done on your own lense 
DIM=(1920, 1080)
K=np.array([[1209.4636416230867, 0.0, 1294.5846820755305], [0.0, 1208.7544583141064, 978.4039362891001], [0.0, 0.0, 1.0]])
D=np.array([[0.02087909969265743], [-0.08509281826480271], [0.19554990995705712], [-0.12830059111751368]])



def dewarp(img):

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img




def calibrate(num_corners_checkerboard_height=6, num_corners_checkerboard_width=8):
    
    CHECKERBOARD = (num_corners_checkerboard_height,num_corners_checkerboard_width)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.JPG')

    #print(len(images))

    if len(images) > 0:

        for fname in images:
            #read in image and record shape (make sure all are the same size)
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            #turn the image into a grey only color scale (will make it easier for cv function to identify board)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 


            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)



        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ =             cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )


        #print out the results

        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")







