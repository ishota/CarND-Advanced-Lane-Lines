# -*- coding: utf-8 -*-

from project_constant_values import *
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt


def main():
    # reading in images
    image_list = []
    images = glob.glob(os.path.join("./camera_cal/", "*.jpg"))
    num_images = len(images)
    plot_count = 0
    fig, axs = plt.subplots(2, num_images, figsize=FIGURE_SIZE)
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
    for i in range(num_images):
        image_list.append(np.array(Image.open(images[i])))
        plt.imshow(image_list[i])
        axs[plot_count, i].imshow(image_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # save result at 'camera_cal_output' directory
    if not os.path.exists('camera_cal_output'):
        os.mkdir('camera_cal_output')
    plt.savefig('camera_cal_output/result.jpg')


if __name__ == '__main__':
    main()
