# -*- coding: utf-8 -*-

from project_modules import *
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt


def main():
    # initial setting
    images = glob.glob(os.path.join("./camera_cal/", "*.jpg"))
    num_images = len(images)
    num_kind = 5
    fig, axs = plt.subplots(num_kind, num_images, figsize=(FIGURE_SIZE[0]*num_images, FIGURE_SIZE[1]*num_kind))
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
    plot_count = 0

    # reading in images
    im_list = []
    for i in range(num_images):
        im_list.append(np.array(Image.open(images[i])))
        axs[plot_count, i].imshow(im_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # convert image to gray scale
    for i in range(num_images):
        axs[plot_count, i].imshow(cv2.cvtColor(im_list[i], cv2.COLOR_RGB2GRAY), cmap='gray')
        axs[plot_count, i].axis("off")
    plot_count += 1

    # extract chess board corners
    for i in range(num_images):
        axs[plot_count, i].imshow(extract_obj_img_points(im_list[i], True))
        axs[plot_count, i].axis("off")
    plot_count += 1

    # make undistorted images
    undist_list = []
    for i in range(num_images):
        undist_list.append(correct_distortion(im_list[i]))
        axs[plot_count, i].imshow(undist_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # make undistorted and transformed images
    for i in range(num_images):
        axs[plot_count, i].imshow(correct_distortion_and_transform(im_list[i]))
        axs[plot_count, i].axis("off")

    # save result at 'output_images' directory
    plt.savefig('output_images/result_distortion_correction.jpg')


if __name__ == '__main__':
    main()
    
