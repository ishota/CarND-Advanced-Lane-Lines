# -*- coding: utf-8 -*-

from project_constant_values import *
from project_modules import *


def main():

    # divide videos into img
    image_list = get_frame_list(MOVIE_PATH)

    # detect lane line
    for n, image in enumerate(image_list):
        image_list[n] = find_lane(image)

    # create video from img
    convert_frame_to_video(image_list, MOVIE_NAME, RESULT_PATH)


if __name__ == '__main__':
    main()
