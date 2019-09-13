import numpy as np
import cv2

# setting for subplot
FIGURE_SIZE = (12.8, 7.2)
W_SPACE = 0.01
H_SPACE = 0.01

# chessboard object point
N_X_POINT = 9
N_Y_POINT = 6

# cv2.cornerSubPix parameters
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
