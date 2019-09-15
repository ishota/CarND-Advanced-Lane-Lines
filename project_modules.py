from project_constant_values import *
import cv2
import numpy as np


def apply_threshold_gradient(scaled_sobel, s_thresh=SX_THRESH):
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= s_thresh[0]) & (scaled_sobel <= s_thresh[1])] = 1
    return sxbinary


def apply_sobel_filter(s_channel_img):
    sobel = cv2.Sobel(s_channel_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobelx / np.max(abs_sobelx))
    return scaled_sobel


def get_s_channel_image(img):
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel_image = hls_image[:, :, 2]
    return s_channel_image


def correct_distortion_and_transform(img):
    is_found, obj_p, img_p = extract_obj_img_points(img)
    if is_found:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, gray.shape[::-1], None, None)
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])
        undist = cv2.undistort(img, mtx, dist, None, new_mtx)
        undist_gray = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
        is_found, corners = cv2.findChessboardCorners(undist_gray, (N_X_POINT, N_Y_POINT), None)

        if is_found:
            cv2.drawChessboardCorners(undist, (N_X_POINT, N_Y_POINT), corners, is_found)
            p1 = corners[0, 0, :]
            p2 = corners[N_X_POINT - 1, 0, :]
            p3 = corners[N_X_POINT * (N_Y_POINT - 1), 0, :]
            p4 = corners[N_X_POINT * N_Y_POINT - 1, 0, :]
            src = np.float32([p1, p2, p3, p4])
            offset = 75
            dst = np.float32([[offset, offset], [undist.shape[1] - offset, offset], [offset, undist.shape[0] - offset],
                              [undist.shape[1] - offset, undist.shape[0] - offset]])
            trans_mtx = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(undist, trans_mtx, (undist.shape[1], undist.shape[0]), flags=cv2.INTER_LINEAR)
            return warped
        else:
            return img
    else:
        return img


def correct_distortion(img):
    is_found, obj_p, img_p = extract_obj_img_points(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if is_found:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, gray.shape[::-1], None, None)
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, gray.shape[::-1], 1, gray.shape[::-1])
        undist = cv2.undistort(img, mtx, dist, None, new_mtx)
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
        corners = cv2.cornerSubPix(gray, corners, (10, 10), (-1, -1), CRITERIA)
        imgpoints.append(corners)

    if do_plot:
        corners_img = np.ndarray.copy(img)
        cv2.drawChessboardCorners(corners_img, (N_X_POINT, N_Y_POINT), corners, is_found)
        return corners_img
    else:
        return is_found, objpoints, imgpoints
