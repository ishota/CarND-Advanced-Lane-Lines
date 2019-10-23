# CarND-Advanced-Lane-Lines

This is a project for Udacity lesson of Self-driving car engineer.
The project is Advanced Lane Finding that detects lane line in picuture and put line area on image.

[camera_result]: ./output_images/result_distortion_correction.jpg

## Description

The project consists three steps below.

- Camera calibration : Calculate the correct camera matrix and distortion coefficients using the calibration chessboard images.
- Lane line detection (image) : Detect lane line in birds-eye view image and put line area on it. Transform birds-eye view image to original view image finally.
- Lane line detection (movie) : Cut a movie to images and applied the lane line detection algorithms to images. Detect line effectively by using previous frame line information.

## 1. Camera calibration

You can get an image that shows process of camera calibration by using `distortion_correction.py`.

![alt_txt][camera_result]

The result is, from top to bottom, original image, grayscaled image, image put extracted chess board corners, undistorted images, and front view image.

## 2. Lane line detection (image)

## 3. Lane line detection (movie)
