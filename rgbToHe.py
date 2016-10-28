from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hed
from scipy import linalg
import sys
from skimage.util import dtype
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte, img_as_uint
from skimage.filters import threshold_otsu

M = np.array([[0.65, 0.70, 0.29],
              [0.07, 0.99, 0.11],
              [0.27, 0.57, 0.78]])


M2 = np.array([[0.49, 0.760, 0.41],
              [0.046, 0.84, 0.54],
	      [0.76, 0.001, 0.64]])

# M2[2, :] = np.cross(M[0, :], M[1, :])
print(M2)
D = linalg.inv(M2)


def loadData(file_name):
	file_data = io.imread(file_name)
	return file_data

def makeDeconv(file_data, conv_matrix):
	rgb = dtype.img_as_float(file_data, force_copy=True)
    	rgb += 2
    	stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)
    
	return np.reshape(stains, rgb.shape)

def saveNewFile(new_file_data, output_name):
	new_file_data = rescale_intensity(new_file_data[:, :, 0], out_range=(0, 1))
	print(new_file_data)
	new_file_data = img_as_uint(new_file_data)
	io.imsave(output_name, new_file_data)

if __name__ == '__main__':
	file_name = sys.argv[1]
	output_name = sys.argv[2]
	file_data = loadData(file_name)
	new_file_data = makeDeconv(file_data, D)
	saveNewFile(new_file_data, output_name)
