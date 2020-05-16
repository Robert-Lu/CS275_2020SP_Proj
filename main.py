import numpy as np
import scipy.ndimage as nd
from scipy.misc import imresize
import matplotlib.pyplot as plt
from time import sleep
from plt_helper import *
from util import *
from worm import *
import logging
logging.basicConfig(level=logging.INFO)

def image_preprocess(img, target_width):
    img_size = img.shape 
    if len(img_size) > 2: # multi channel
        img = rgb2gray(img)
    height, width = img_size[:2]
    if width != target_width:
        scale = 1.0 * target_width / width
        img = imresize(img, scale)
    return img


if __name__ == '__main__':
    img = plt.imread("./brain.jpg")
    img = image_preprocess(img, target_width=512)

    w = Worm(img, num_nodes=32, scale=1.0, thickness=2.0, 
            # orientation_angle=3.14/4,
            base_position=np.array([237, 164]))

    m = StretchMotor(w)
    i = 0

    ax_collection_to_remove = []
    f, ax = plt.subplots(figsize=grid_figure_size(1, 1, magnitude=2.75))
    plt.imshow(img, cmap="gray")

    w.draw(f, ax)

    def next_on_click(event):
        print("NEXT")
        m.apply(scale_length=0.5, scale_orientation=0.05)
        w.draw(f, ax)

    def auto_on_click(event):
        print("AUTO (not implemented)")

    def reset_on_click(event):
        print("RESET (not implemented)")

    def quit_on_click(event):
        print("QUIT")
        exit()

    btn_quit = Button(plt.axes([0.85, 0.10, 0.1, 0.04]), '--- QUIT ---')
    btn_reset = Button(plt.axes([0.85, 0.15, 0.1, 0.04]), '--- RESET ---')
    btn_auto = Button(plt.axes([0.85, 0.20, 0.1, 0.04]), '--- AUTO ---')
    btn_next = Button(plt.axes([0.85, 0.25, 0.1, 0.04]), '--- NEXT ---')
    btn_quit.on_clicked(quit_on_click)
    btn_reset.on_clicked(reset_on_click)
    btn_auto.on_clicked(auto_on_click)
    btn_next.on_clicked(next_on_click)

    plt.show()
