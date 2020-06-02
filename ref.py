import numpy as np
import scipy.ndimage as nd
from plt_helper import *
from util import *
import logging


def get_eroded_reference(img, threshold=200, eroding_width=3,
                         blur=True, blur_sigma=3):
    if blur:
        img = gaussian_blur(img, blur_sigma) 
    img_bin = (img > threshold)
    return nd.binary_erosion(img_bin, structure=np.ones((eroding_width, eroding_width))).astype(img.dtype)
