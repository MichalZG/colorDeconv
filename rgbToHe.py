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
M2 = np.array([[0.49, 0.760, 0.41],
               [0.046, 0.84, 0.54],
               [0.76, 0.001, 0.64]])

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

    if args.channel == 'e':
        new_data = rescale_intensity(new_file_data[:, :, 0], out_range=(0, 1))
    elif args.channel == 'h':
        new_data = rescale_intensity(new_file_data[:, :, 1], out_range=(0, 1))
    elif args.channel == 'eh':
        e = rescale_intensity(new_file_data[:, :, 0], out_range=(0, 1))
        h = rescale_intensity(new_file_data[:, :, 1], out_range=(0, 1))
        new_data = np.dstack((np.zeros_like(e), e, h))

    new_data = img_as_ubyte(new_data)
    
    io.imsave(os.path.join(args.output, name), new_data)


if __name__ == '__main__':

    cur_path = os.getcwd()

    parser = argparse.ArgumentParser(description="color deconvolution")

    parser.add_argument('-c', '--channel', type=str, choices=['e', 'h', 'eh'],
                        nargs='?', default='e', help='channel for save: '
                                                     'e - eosin, '
                                                     'h - hematoxylin, '
                                                     'eh - e + h. '
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

    p = Pool(2)
    p.map(makeDeconv, images)
