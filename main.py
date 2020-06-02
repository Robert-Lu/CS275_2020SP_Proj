import numpy as np
import scipy.ndimage as nd
from scipy.misc import imresize
import matplotlib.pyplot as plt
from time import sleep
from plt_helper import *
from util import *
from worm import *
from ref import *
import logging

logging.basicConfig(level=logging.INFO)


def image_preprocess(img, target_width):
    img_size = img.shape
    if len(img_size) > 2:  # multi channel
        img = rgb2gray(img)
    height, width = img_size[:2]
    if width != target_width:
        scale = 1.0 * target_width / width
        img = imresize(img, scale)
    return img


if __name__ == '__main__':
    img = plt.imread("./brain.jpg")
    img = image_preprocess(img, target_width=512)
    f, ax = plt.subplots(figsize=grid_figure_size(1, 1, magnitude=2.75))

    w = Worm(img, num_nodes=64, scale=.5, thickness=2.0,
             # orientation_angle=3.14/4,
             base_position=np.array([237, 163]))
    w.load("worm_data_20200601_164545")

    ref_image = get_eroded_reference(img, threshold=230, blur_sigma=2, eroding_width=3)
    plt.imshow(img, cmap="gray")
    # plt.imshow(img/2 + ref_image*128, cmap="gray")
    # show_2_image(ref_image, img /2 + ref_image*128, block=True)
    # exit()
    w.add_ref_image(ref_image)

    ms = StretchMotor(w, scale_length=0.2, scale_orientation=0.4, sensor_threshold=0.95)
    mt = ThickenMotor(w, scale_thickness=0.3, sensor_threshold=0.6)
    mn = NomalizationMotor(w, balance_threshold=0.5, balance_scale=0.02)

    w.draw(f, ax)

    def test_on_click(event):
        print("IDLE")
        # ms.apply()
        mn.apply()
        # mt.apply()
        w.draw(f, ax)

    def next_on_click(event):
        print("NEXT")
        w.apply_motors()
        w.draw(f, ax)

    def auto_on_click(event):
        print("AUTO (not implemented)")

    def reset_on_click(event):
        print("RESET")
        w.reset()
        w.draw(f, ax)

    def quit_on_click(event):
        print("QUIT")
        exit()

    def save_on_click(event):
        print("SAVE")
        w.save()

    btn_quit = Button(plt.axes([0.85, 0.10, 0.1, 0.04]), '--- QUIT ---')
    btn_reset = Button(plt.axes([0.85, 0.15, 0.1, 0.04]), '--- RESET ---')
    btn_auto = Button(plt.axes([0.85, 0.20, 0.1, 0.04]), '--- AUTO ---')
    btn_next = Button(plt.axes([0.85, 0.25, 0.1, 0.04]), '--- NEXT ---')
    btn_test = Button(plt.axes([0.85, 0.30, 0.1, 0.04]), '--- TEST ---')
    btn_save = Button(plt.axes([0.85, 0.35, 0.1, 0.04]), '--- SAVE ---')
    btn_quit.on_clicked(quit_on_click)
    btn_reset.on_clicked(reset_on_click)
    btn_auto.on_clicked(auto_on_click)
    btn_next.on_clicked(next_on_click)
    btn_test.on_clicked(test_on_click)
    btn_save.on_clicked(save_on_click)

    plt.show()
