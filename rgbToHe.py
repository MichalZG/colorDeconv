#!/usr/bin/env python

import warnings
from skimage import io
import numpy as np
from scipy import linalg
import glob
from skimage.util import dtype
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from multiprocessing import Pool
import os
import argparse

warnings.filterwarnings('ignore')


# H&E
M2 = np.array([[0.6443186, 0.7166757, 0.26688856],
               [0.09283128, 0.9545457, 0.28324],
               [0.63595444, 0.001, 0.7717266]])
"""
# H&E2
M2 = np.array([[0.49, 0.760, 0.41],
               [0.046, 0.84, 0.54],
               [0.76, 0.001, 0.64]])
"""


D = linalg.inv(M2)


def loadData(file_name):
    file_data = io.imread(file_name)
    return file_data


def makeDeconv(image):
    file_data = loadData(image)
    rgb = dtype.img_as_float(file_data, force_copy=True)
    rgb += 2
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), D)
    saveNewFile(np.reshape(stains, rgb.shape), os.path.basename(image))


def saveNewFile(new_file_data, name):
    if args.channel == 'h':
        new_data = rescale_intensity(new_file_data[:, :, 0], out_range=(0, 1))
    elif args.channel == 'e':
        new_data = rescale_intensity(new_file_data[:, :, 1], out_range=(0, 1))
    elif args.channel == 'hed':
        h = rescale_intensity(new_file_data[:, :, 0], out_range=(0, 1))
        e = rescale_intensity(new_file_data[:, :, 1], out_range=(0, 1))
        d = rescale_intensity(new_file_data[:, :, 2], out_range=(0, 1))
        new_data = np.dstack((h, e, d))

    new_data = img_as_ubyte(new_data)
    
    io.imsave(os.path.join(args.output, name), new_data)


if __name__ == '__main__':

    cur_path = os.getcwd()

    parser = argparse.ArgumentParser(description="color deconvolution")

    parser.add_argument('-c', '--channel', type=str, choices=['h', 'e', 'hed'],
                        nargs='?', default='h', help='channel for save: '
                                                     'h: hematoxylin, '
                                                     'e: eosin, '
                                                     'hed: h + e + dab. '
                                                     'Default: %(default)s')
    parser.add_argument('-r', '--regexp', type=str, nargs='?', default='*.png',
                        help='regulas expression for images name. '
                             'Default: %(default)s')
    parser.add_argument('-o', '--output', type=str,
                        nargs='?', default=os.path.join(
                            cur_path, 'output'), help='output dir. '
                                                      'Default: %(default)s')
    args = parser.parse_args()

    images = sorted(glob.glob(os.path.join(cur_path, args.regexp)))
    
    try:
        os.makedirs(args.output)
    except OSError:
        pass

    p = Pool(7)
    p.map(makeDeconv, images)
