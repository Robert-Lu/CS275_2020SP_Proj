import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt 
from scipy.ndimage.filters import gaussian_filter
from plt_helper import *


def rgb2gray(rgb):
    """
        From: 
        https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
        Altered
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.333 * r + 0.333 * g + 0.333 * b
    return gray

def normalized(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v

def gaussian_blur(img, sigma):
    return gaussian_filter(img, sigma)

# [_min, _max) for int
def _between(_min, val, _max):
    if val >= _max:
        return _max - 1
    if val < _min:
        return _min
    return val

# [_min, _max] for float
def _fbetween(_min, val, _max):
    if val > _max:
        return _max
    if val < _min:
        return _min
    return val

def get_image_intensity(img, pos):
    x = int(round(pos[0]))
    y = int(round(pos[1]))

    img_h = img.shape[0]
    img_w = img.shape[1]

    return img[_between(0, y, img_h)][_between(0, x, img_w)]
