from project_constant_values import *
from numpy.random import *
import cv2
import numpy as np


def find_movie_lane(img, pre_left_fit, pre_right_fit, left_dots, right_dots):
    birds_eye_img = create_binary_b_img(img)
    polynomial_fit_img, left_c, right_c, left_line, right_line, left_f, right_f, left_dots, right_dots, position \
        = fit_polynomial(birds_eye_img, True, pre_left_fit, pre_right_fit, left_dots, right_dots)
    info_img = put_lane_information(img, polynomial_fit_img, (left_c, right_c), position)
    return info_img, left_f, right_f, left_dots, right_dots


def find_lane(img):
    birds_eye_img = create_binary_b_img(img)
    polynomial_fit_img, left_c, right_c, left_line, right_line, position = fit_polynomial(birds_eye_img)
    info_img = put_lane_information(img, polynomial_fit_img, (left_c, right_c), position)
    return info_img


def create_binary_b_img(img):
    b_img = birds_eye_view(img)
    s_channel_img = get_s_channel_image(b_img)
    color_threshold_img = apply_color_threshold(s_channel_img)
    sobel_filtered_img = apply_sobel_filter(s_channel_img)
    gradient_threshold_img = apply_gradient_threshold(sobel_filtered_img)
    marge_img = marge_color_gradient_image(color_threshold_img, gradient_threshold_img, True)
    return marge_img


def put_lane_information(img, polynomial_fit_img, curvatures, position):
    info_img = np.copy(img)
    org_b_img = birds_eye_view(img)
    destination_point = np.float32([[MARGIN, 0], [img.shape[1] - MARGIN, 0],
                                    [MARGIN, img.shape[0]], [img.shape[1] - MARGIN, img.shape[0]]])
    inverse_t_mtx = cv2.getPerspectiveTransform(destination_point, SOURCE_POINT)
    original_view_line = cv2.warpPerspective(polynomial_fit_img, inverse_t_mtx, (img.shape[1], img.shape[0]))
    info_img = cv2.addWeighted(info_img, 1, original_view_line, 0.5, 0)
    left_c = round(curvatures[0], 2)
    cv2.putText(info_img, 'Left Lane Curve', (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 4)
    cv2.putText(info_img, str(left_c) + ' m', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 4)
    right_c = round(curvatures[1], 2)
    cv2.putText(info_img, 'Right Lane Curve', (880, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 4)
    cv2.putText(info_img, str(right_c) + ' m', (880, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 4)
    position = round(position, 2)
    cv2.putText(info_img, 'Center Offset m', (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 4)
    cv2.putText(info_img, str(position) + ' m', (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 4)
    org_b_img = cv2.resize(org_b_img, (org_b_img.shape[1] // 2, org_b_img.shape[0] // 2))
    polynomial_fit_img = cv2.resize(polynomial_fit_img, (polynomial_fit_img.shape[1] // 2, polynomial_fit_img.shape[0] // 2))
    debug_info_img = cv2.hconcat([org_b_img, polynomial_fit_img])
    info_img = cv2.vconcat([debug_info_img, info_img])
    return info_img


def compute_rial_position(left_x, left_y, right_x, right_y):
    # calculates the curvature of  polynomial function in meters (rial!)
    left_fit_rial = np.polyfit(left_y*YM_PER_PIX, left_x*XM_PER_PIX, 2)
    right_fit_rial = np.polyfit(right_y*YM_PER_PIX, right_x*XM_PER_PIX, 2)
    left_c = ((1 + (2*left_fit_rial[0]*719*YM_PER_PIX + left_fit_rial[1])**2)**1.5) / np.absolute(2*left_fit_rial[0])
    right_c = ((1 + (2*right_fit_rial[0]*719*YM_PER_PIX + right_fit_rial[1])**2)**1.5) / np.absolute(2*right_fit_rial[0])

    # calculate position of car
    left_x = left_fit_rial[0] * 719 * YM_PER_PIX ** 2 + left_fit_rial[1] * 719 * YM_PER_PIX + left_fit_rial[2]
    right_x = right_fit_rial[0] * 719 * YM_PER_PIX ** 2 + right_fit_rial[1] * 719 * YM_PER_PIX + right_fit_rial[2]
    mid_x = (right_x + left_x) / 2
    lane_width = right_x - left_x
    position = (mid_x - 640 * XM_PER_PIX) * 3.7 / lane_width
    return position, left_c, right_c


def fit_polynomial(img, movie=False, pre_left_fit=None, pre_right_fit=None, left_dots=None, right_dots=None):
    if pre_left_fit is not None and pre_right_fit is not None:
        leftx, lefty, rightx, righty, left_dots, right_dots, out_img = \
            region_of_pre_fit(img, pre_left_fit, pre_right_fit, left_dots, right_dots)
    else:
        leftx, lefty, rightx, righty, out_img = region_of_interest(img)
        left_dots = np.stack([leftx, lefty])
        right_dots = np.stack([rightx, righty])

    if left_dots.shape[1] < MIN_PIX:
        leftx, lefty, _, _, out_img = region_of_interest(img)
        left_dots = np.stack([leftx, lefty])

    if right_dots.shape[1] < MIN_PIX:
        _, _, rightx, righty, out_img = region_of_interest(img)
        right_dots = np.stack([rightx, righty])

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    y = np.arange(img.shape[0], dtype=int)
    left_points = np.array([left_fitx, y]).T
    right_points = np.array([right_fitx, y]).T
    all_points = np.concatenate([left_points, right_points], 0)
    all_points = all_points[np.argsort(all_points[:, 1])]
    all_points = all_points.reshape(1, -1, 2)
    left_points = left_points.reshape(1, -1, 2)
    right_points = right_points.reshape(1, -1, 2)

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    cv2.polylines(out_img, left_points.astype(int), False, color=(255, 255, 0), thickness=10)
    cv2.polylines(out_img, right_points.astype(int), False, color=(255, 255, 0), thickness=10)
    cv2.polylines(out_img, all_points.astype(int), False, color=(187, 190, 43), thickness=1)

    position, left_c, right_c = compute_rial_position(leftx, lefty, rightx, righty)

    if movie:
        return out_img, left_c, right_c, \
               left_points, right_points, left_fit, right_fit, left_dots, right_dots, position

    return out_img, left_c, right_c, left_points, right_points, position


def drop_out_dots(left_dots, right_dots):

    left_size = left_dots.shape[1]
    right_size = right_dots.shape[1]

    left_array = np.arange(left_size)
    right_array = np.arange(right_size)

    left_del_index = choice(left_array, int(left_size/3), replace=False)
    right_del_index = choice(right_array, int(right_size/3), replace=False)

    left_dots = np.delete(left_dots, left_del_index, 1)
    right_dots = np.delete(right_dots, right_del_index, 1)

    return left_dots, right_dots


def region_of_pre_fit(img, pre_left_f, pre_right_f, left_dots, right_dots):

    left_dots, right_dots = drop_out_dots(left_dots, right_dots)

    out_img = np.dstack((img, img, img))
    nonzero = np.array(img.nonzero())
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    nonzerox_left = np.array(nonzerox[nonzerox < img.shape[1]/2])
    nonzeroy_left = np.array(nonzeroy[nonzerox < img.shape[1]/2])
    nonzerox_right = np.array(nonzerox[nonzerox >= img.shape[1]/2])
    nonzeroy_right = np.array(nonzeroy[nonzerox >= img.shape[1]/2])

    nonzerox_left = np.concatenate([nonzerox_left, np.array(left_dots[0])], 0)
    nonzeroy_left = np.concatenate([nonzeroy_left, np.array(left_dots[1])], 0)
    nonzerox_right = np.concatenate([nonzerox_right, np.array(right_dots[0])], 0)
    nonzeroy_right = np.concatenate([nonzeroy_right, np.array(right_dots[1])], 0)

    pre_left_func_val = pre_left_f[0] * nonzeroy_left ** 2 + pre_left_f[1] * nonzeroy_left + pre_left_f[2]
    pre_right_func_val = pre_right_f[0] * nonzeroy_right ** 2 + pre_right_f[1] * nonzeroy_right + pre_right_f[2]

    gap_left = np.abs(pre_left_func_val - nonzerox_left)
    gap_right = np.abs(pre_right_func_val - nonzerox_right)

    good_left_inds = np.array(gap_left < DOT_THRESHOLD)
    good_right_inds = np.array(gap_right < DOT_THRESHOLD)

    leftx = nonzerox_left[good_left_inds]
    lefty = nonzeroy_left[good_left_inds]
    rightx = nonzerox_right[good_right_inds]
    righty = nonzeroy_right[good_right_inds]

    left_dots = np.stack([leftx, lefty])
    right_dots = np.stack([rightx, righty])

    return leftx, lefty, rightx, righty, left_dots, right_dots, out_img


def region_of_interest(img):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    out_img = np.dstack((img, img, img))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(img.shape[0]//N_WINDOWS)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(N_WINDOWS):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - WINDOW_WIDTH
        win_xleft_high = leftx_current + WINDOW_WIDTH
        win_xright_low = rightx_current - WINDOW_WIDTH
        win_xright_high = rightx_current + WINDOW_WIDTH

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > MIN_PIX:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > MIN_PIX:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        pass

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def birds_eye_view(img):
    destination_point = np.float32([[MARGIN, 0], [img.shape[1] - MARGIN, 0],
                                    [MARGIN, img.shape[0]], [img.shape[1] - MARGIN, img.shape[0]]])
    transition_mtx = cv2.getPerspectiveTransform(SOURCE_POINT, destination_point)
    birds_eye_img = cv2.warpPerspective(img, transition_mtx, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return birds_eye_img


def marge_color_gradient_image(color_img, gradient_img, is_gray_scale=False):
    if is_gray_scale:
        marge_img = np.zeros_like(gradient_img)
        marge_img[(color_img == 1) | (gradient_img == 1)] = 1
        return marge_img
    else:
        return np.dstack((np.zeros_like(gradient_img), gradient_img, color_img)) * 255


def apply_gradient_threshold(scaled_sobel, sx_thresh=SX_THRESH):
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    return sxbinary


def apply_sobel_filter(s_channel_img):
    sobel = cv2.Sobel(s_channel_img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobelx / np.max(abs_sobelx))
    return scaled_sobel


def apply_color_threshold(s_channel_img, s_thresh=S_COLOR_THRESH):
    s_binary = np.zeros_like(s_channel_img)
    s_binary[(s_channel_img >= s_thresh[0]) & (s_channel_img <= s_thresh[1])] = 1
    return s_binary


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
            dst = np.float32([[OFFSET, OFFSET], [undist.shape[1] - OFFSET, OFFSET], [OFFSET, undist.shape[0] - OFFSET],
                              [undist.shape[1] - OFFSET, undist.shape[0] - OFFSET]])
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


def get_frame_list(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('no movie and exit this')
        exit()
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    image_list = []
    for n in range(int(count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            image_list.append(frame)
        else:
            return image_list


def convert_frame_to_video(image_list, name, result_dir_path):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    shape1 = image_list[0].shape[1]
    shape2 = image_list[0].shape[0]
    video = cv2.VideoWriter('{}{}.mp4'.format(result_dir_path, name), fourcc, 20.0, (shape1, shape2))

    for image in image_list:
        video.write(image)