# CarND-Advanced-Lane-Lines

This is a project for Udacity lesson of Self-driving car engineer.
The project is Advanced Lane Finding that detects lane line in picuture and put line area on image.

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

The lane pixels were identified using an image composed of two types of images.

The first image is the S color image of the HLS color map.
S color is a saturation.
The saturation is an element that represents the vividness and darkness of the color defined by the hue, and is often expressed in the range of 0% to 100%.
100% is the most vivid, the color fades as the saturation decreases, and becomes gray when it reaches 0%.
Roads are often gray.
Since lanes are often vivid colors, it is possible to identify lane pixels using saturation.

Another is an image obtained by filtering the s color image.
This filter is called a sobel filter and can highlight the boundaries between pixels in a specific direction.
In the case of lanes, the filter is used in the horizontal direction because it extends in the direction of travel.
As a result, the pixel at the boundary between the road and the lane line can be identified.

Next, I draw approximated line.
I calculate a number of pixel which identified by previous function to detect area near lane line left and right separately.
Then, I fit a quadratic function to pixel in the area.

Finally I put the radius of curvature of each lane line on image.
I use YM_PER_PIX 720 / 30 [pixel / m] and CURVE_POINT 50 to conversion in y from pixels space to meters.
And I calculate curvature in below function.
```
def compute_rial_curvature(coefficient):
    y_eval = CURVE_POINT*YM_PER_PIX
    curvature = ((1 + (2*coefficient[0]*y_eval + coefficient[1])**2)**1.5) / np.absolute(2*coefficient[0])
    return curvature
```
The radius of curvature is calculated for each of the left and right lanes, but the one with the smaller approximate error (RMS, etc.) of the approximate curve is adopted. This is because most lanes are symmetrical. Considering the middle of the photo as the center of the vehicle, the position of the vehicle on the road can be calculated by performing unit conversion [pixel / m] in the horizontal direction.

## 3. Lane line detection (movie)

A result of movies in output_videos directory.
A project_video_result_histogram.mp4 is movie file made by lane line detection algorithm at step 2.
But the algorithm sometimes cannot detect lane line.

I proposed two ideas and implemented algorithms.
The first idea is to identify only the pixels around the approximate line of the previous frame as lane line.
We can exclude pixcels detected outside the previous lane.
Another idea is to leave the pixels detected in the previous frame.
This idea improved approximation accuracy of the lane.
Here, instead of leaving the pixel forever, we removed it with a 30% chance to avoid being too influenced by the past.
Of the six images, the upper three images show before the idea is put in, and the lower three images show after the idea is put in.
My idea shows that the lane detection accuracy has improved.

[histo_1]: ./output_videos/histo_1.png
[histo_2]: ./output_videos/histo_2.png
[histo_3]: ./output_videos/histo_3.png
[propose_1]: ./output_videos/propose_1.png
[propose_2]: ./output_videos/propose_2.png
[propose_3]: ./output_videos/propose_3.png

| ![alt_txt][histo_1] | ![alt_txt][histo_2] | ![alt_txt][histo_3] |
| ---- | ---- | --- |
| ![alt_txt][propose_1] | ![alt_txt][propose_2] | ![alt_txt][propose_3] |

## Discussion

### Weak to lane that cannot be detected in HLS color space

My pipeline detects lane line in s color of HLS color space.
For example, in a scene where a lane with low saturation continues, the lane cannot be specified.
Therefore, I propose a more robust identification method by introducing an algorithm that estimates the road shape from past driving histories, an algorithm that estimates from the direction of the preceding vehicle.