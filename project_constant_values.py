import cv2
import numpy as np

# movie path
MOVIE_PATH = 'project_video.mp4'

# perspective src point (left up, right up, left down, right down)
SOURCE_POINT = np.float32([[567.5, 470], [720.5, 470], [275, 670], [1034.5, 670]])

# color threshold
S_COLOR_THRESH = (180, 253)

# gradient threshold
SX_THRESH = (25, 80)

# setting for subplot
FIGURE_SIZE = (12.8, 7.2)
W_SPACE = 0.05
H_SPACE = 0.05

# chessboard object point
N_X_POINT = 9
N_Y_POINT = 6

# perspective transformed image's offset
OFFSET = 75
MARGIN = 300

# cv2.cornerSubPix parameters
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
