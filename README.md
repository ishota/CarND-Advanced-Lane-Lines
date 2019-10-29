# CarND-Advanced-Lane-Lines

*Details of this project can be found in Writeup.md.*

This is a project for Udacity lesson of Self-driving car engineer.
The project is Advanced Lane Finding that detects lane line in picuture and put line area on image.

## Quick start guide

Below python files output image or video file after make directory `output_images` and `output_videos`. 

- `distortion_correction.py` -> `output_images/result_distortion_correction.jpg`
- `find_lane_in_image.py` -> `output_images/result_find_lane.jpg`
- `find_lane_in_movie.py` -> `output_videos/project_video_result`

`project_constant_value.py` contains the constants and parameters used in this project.

## Description

The project consists three steps below.

- Camera calibration : Calculate the correct camera matrix and distortion coefficients using the calibration chessboard images.
- Lane line detection (image) : Detect lane line in birds-eye view image and put line area on it. Transform birds-eye view image to original view image finally.
- Lane line detection (movie) : Cut a movie to images and applied the lane line detection algorithms to images. Detect line effectively by using previous frame line information.
