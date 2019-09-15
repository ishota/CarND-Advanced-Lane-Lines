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
    num_kind = 6
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

    # convert to HLS color space and separate the S channel
    s_channel_list = []
    for i in range(num_images):
        s_channel_list.append(get_s_channel_image(im_list[i]))
        axs[plot_count, i].imshow(s_channel_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # create binary image applied s threshold
    s_threshold_list = []
    for i in range(num_images):
        s_threshold_list.append(apply_color_threshold(s_channel_list[i]))
        axs[plot_count, i].imshow(s_threshold_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # create x direction sobel filter
    sobel_list = []
    for i in range(num_images):
        sobel_list.append(apply_sobel_filter(s_channel_list[i]))
        axs[plot_count, i].imshow(sobel_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # apply x direction gradient threshold
    gradient_list = []
    for i in range(num_images):
        gradient_list.append(apply_threshold_gradient(sobel_list[i]))
        axs[plot_count, i].imshow(gradient_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # marge color and gradient images
    color_gradient_list = []
    for i in range(num_images):
        color_gradient_list.append(marge_color_gradient_image(s_threshold_list[i], gradient_list[i]))
        axs[plot_count, i].imshow(color_gradient_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # save result at 'output_images' directory
    plt.savefig('output_images/result_fine_lane.jpg')


if __name__ == '__main__':
    main()
