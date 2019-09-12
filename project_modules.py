from project_constant_values import *
import cv2
import matplotlib.pyplot as plt


def correct_distortion(img):
    extract_obj_img_points(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    is_found, obj_p, img_p = extract_obj_img_points(img)
    if is_found:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, (gray.shape[0], gray.shape[1]), None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        return undist
    else:
        return img


def extract_obj_img_points(img, do_plot=False):
    objp = np.zeros((N_X_POINT*N_Y_POINT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:N_X_POINT, 0:N_Y_POINT].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    is_found, corners = cv2.findChessboardCorners(gray, (N_X_POINT, N_Y_POINT), None)

    if is_found:
        objpoints.append(objp)
        imgpoints.append(corners)

    if do_plot:
        cv2.drawChessboardCorners(img, (N_X_POINT, N_Y_POINT), corners, is_found)
        return img
    else:
        return is_found, objpoints, imgpoints
