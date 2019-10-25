# CarND-Advanced-Lane-Lines

This is a project for Udacity lesson of Self-driving car engineer.
The project is Advanced Lane Finding that detects lane line in picuture and put line area on image.

## Quick start guide

- `distortion_correction.py` -> `output_images/result_distortion_correction.jpg`
- `find_lane_in_image.py` -> `output_images/result_fine_lane.jpg`
- `find_lane_in_movie.py` -> `output_videos/project_video_result`

## Description

The project consists three steps below.

- Camera calibration : Calculate the correct camera matrix and distortion coefficients using the calibration chessboard images.
- Lane line detection (image) : Detect lane line in birds-eye view image and put line area on it. Transform birds-eye view image to original view image finally.
- Lane line detection (movie) : Cut a movie to images and applied the lane line detection algorithms to images. Detect line effectively by using previous frame line information.

## 1. Camera calibration

In order to detect the position of an object, it is necessary to know in advance the correspondence that "the object in this direction as seen from the camera is reflected here on the image".
You can get an image that shows process of camera calibration by using `distortion_correction.py`.

[camera_result]: ./output_images/result_distortion_correction.jpg
![alt_txt][camera_result]

Above result is, from top to bottom, original image, grayscaled image, image put extracted chess board corners, undistorted images, and front view image.
I used the OpenCV function of `findChessboardCorners` to find 54(=9x6) corners and `drawChessboardCorners` to color found corners.
Then I used `calibrateCamera` and `getOptimalNewCameraMatrix` to calculate a matrix that fix camera lens distortion.
Finally, an undistort image can be got by `undistort`.

The function of `calibrateCamera` can calculate a camera matrix and a distortion coefficients whose equations are shown below.

<img src=https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;f_x&space;&&space;0&space;&&space;c_x&space;\\&space;0&space;&&space;f_y&space;&&space;c_y&space;\\&space;0&space;&&space;0&space;&&space;1&space;\end{bmatrix} />

*f* are parameters that reflects the angle of view of the camera and the resolution of the image, and is called the focal length.
*c* is the coordinate of the pixel where the light passing through the camera's optical axis is reflected.
It is approximately the coordinates around the center of the image.

<img src=https://latex.codecogs.com/gif.latex?(k_1,&space;k_2,&space;p_1,&space;p_2[,&space;k_3[,&space;k_4,&space;k_5,&space;k_6[,s_1,s_2,s_3,s_4[,\tau_x,&space;\tau_y]]]]) />

This list is distortion parameter list.
The kind of distortion are radial distortion, tangential distortion and thin prism distortion.
If you know more information about camera distortion, please reference the document of [OpenCV](https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html).

## 2. Lane line detection (image)

You can get an image that shows process of lane line finding by using `find_lane_in_image.py`.
My pipeline consisted of 7 steps. 

1. Convert the images to HLS color scale.
2. Create binary image applied s value threshold.
3. Apply x direction sobel filter is applied to HLS color scale made in step 1.
4. Create binary image applied x direction gradient threshold.
5. Marge binary images crated in step 2 and 4.
6. Transform image perspective from original to birds eye view.
7. Put a line approximated by a quadratic function on birds eye view image, then transform image to original view.

[image_result]: ./output_images/result_fine_lane.jpg
![alt_txt][image_result]

Above result is, from top to bottom, original image, HLS color scaled image, binary image made in step 2, x direction sobel filtered image, binary image made in step 4, marged binary image, birds eye view image, and original image with color lane line.

## 3. Lane line detection (movie)

## Discussion
