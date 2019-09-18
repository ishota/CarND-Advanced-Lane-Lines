# -*- coding: utf-8 -*-

from project_modules import *
from PIL import Image
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


def main():
    # initial setting
    images = glob.glob(os.path.join("./test_images/", "*.jpg"))
    num_images = len(images)
    num_kind = 8
    fig, axs = plt.subplots(num_kind, num_images, figsize=(FIGURE_SIZE[0]*num_images, FIGURE_SIZE[1]*num_kind))
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
    plot_count = 0

    # reading in images
    im_list = []
    for i in range(num_images):
        im_list.append(np.array(Image.open(images[i])))
        axs[plot_count, i].imshow(im_list[i])
    plot_count += 1

    # convert to HLS color space and separate the S channel
    s_channel_list = []
    for i in range(num_images):
        s_channel_list.append(get_s_channel_image(im_list[i]))
        axs[plot_count, i].imshow(s_channel_list[i])
    plot_count += 1

    # create binary image applied s threshold
    s_threshold_list = []
    for i in range(num_images):
        s_threshold_list.append(apply_color_threshold(s_channel_list[i]))
        axs[plot_count, i].imshow(s_threshold_list[i])
    plot_count += 1

    # create x direction sobel filter
    sobel_list = []
    for i in range(num_images):
        sobel_list.append(apply_sobel_filter(s_channel_list[i]))
        axs[plot_count, i].imshow(sobel_list[i])
    plot_count += 1

    # apply x direction gradient threshold
    gradient_list = []
    for i in range(num_images):
        gradient_list.append(apply_gradient_threshold(sobel_list[i]))
        axs[plot_count, i].imshow(gradient_list[i])
    plot_count += 1

    # marge color and gradient images
    color_marge_list = []
    gray_marge_list = []
    for i in range(num_images):
        color_marge_list.append(marge_color_gradient_image(s_threshold_list[i], gradient_list[i]))
        gray_marge_list.append(marge_color_gradient_image(s_threshold_list[i], gradient_list[i], True))
        axs[plot_count, i].imshow(marge_color_gradient_image(s_threshold_list[i], gradient_list[i]))
    plot_count += 1

    # birds eye view
    birds_eye_img_list = []
    for i in range(num_images):
        birds_eye_img_list.append(birds_eye_view(gray_marge_list[i]))
        axs[plot_count, i].imshow(birds_eye_view(color_marge_list[i]))
    plot_count += 1

    # detect lane area
    # region_of_interest(birds_eye_img_list[0])
    # for i in range(num_images):
    #     axs[plot_count, i].imshow(compute_histogram(birds_eye_img_list[i]))
    # plot_count += 1

    # check find lane module
    for i in range(num_images):
        axs[plot_count, i].imshow(find_lane(im_list[i]), cmap='gray')

    # save result at 'output_images' directory
    plt.savefig('output_images/result_fine_lane.jpg')


if __name__ == '__main__':
    main()
