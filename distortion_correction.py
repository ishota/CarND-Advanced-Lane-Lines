# -*- coding: utf-8 -*-

from project_constant_values import *
from project_modules import *
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt



def main():
    # reading in images
    im_list = []
    images = glob.glob(os.path.join("./camera_cal/", "*.jpg"))
    num_images = len(images)
    plot_count = 0
    fig, axs = plt.subplots(2, num_images, figsize=FIGURE_SIZE)
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
    for i in range(num_images):
        im_list.append(np.array(Image.open(images[i])))
        axs[plot_count, i].imshow(im_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1


    # make undistorted images
    undist_list = []

    # save result at 'output_images' directory
    plt.savefig('output_images/result_distortion_correction.jpg')


if __name__ == '__main__':
    main()
