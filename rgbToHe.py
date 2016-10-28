#!/usr/bin/env python

import warnings
from skimage import io
import numpy as np
from scipy import linalg
import glob
from skimage.util import dtype
from skimage.exposure import rescale_intensity
from skimage import img_as_uint
from multiprocessing import Pool
import os

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
    new_file_data = rescale_intensity(new_file_data[:, :, 0], out_range=(0, 1))
    new_file_data = img_as_uint(new_file_data)
    io.imsave(os.path.join(cur_path, 'hed', name), new_file_data)


if __name__ == '__main__':

    cur_path = os.getcwd()
    images = sorted(glob.glob(os.path.join(cur_path, '*.png')))

    try:
        os.mkdir('hed')
    except OSError:
        pass

    p = Pool(1)
    p.map(makeDeconv, images)
