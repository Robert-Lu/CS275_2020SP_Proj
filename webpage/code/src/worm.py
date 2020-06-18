import numpy as np
from math import sin, cos, exp
import scipy.ndimage as nd
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from time import sleep
from plt_helper import *
from util import normalized, gaussian_blur, get_image_intensity, _fbetween
from random import random
from time import time, strftime
import logging
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d
from sensor import *
from motor import *

PI = 3.1415926535
E = exp(1)


class Worm(object):
    def __init__(self, image,
                 orientation_angle=0 * (PI / 180),
                 base_position=np.array([10, 10], dtype=float),
                 scale=1.0,
                 thickness=1.0,
                 num_nodes=64):
        # background image
        self.image = image
        self.ref_image = None
        self.blur_image_cache = dict()
        # trandormation parameters:
        self.trans_params_orientation_angle = orientation_angle
        self.trans_params_base_position = base_position
        self.trans_params_scale = scale  # np.array([1, 1], dtype=float)
        self.num_nodes = num_nodes
        # record original parameters
        self.ori_trans_params_orientation_angle = orientation_angle
        self.ori_trans_params_base_position = base_position
        self.ori_trans_params_scale = scale  # np.array([1, 1], dtype=float)
        self.ori_num_nodes = num_nodes
        self.ori_thickness = thickness
        # initialization
        self.profile_init()

        # create nodes
        self.node_pos = [None for _ in range(self.num_nodes)]
        self.node_left = [None for _ in range(self.num_nodes)]
        self.node_right = [None for _ in range(self.num_nodes)]
        # motors
        self.motors = []
        self.sensors = []
        # update
        self.last_change = time()
        self.last_update = 0
        self.iter = 0
        self.update_node()
        # draw
        self.fig_handle = None
        self.ax_collection_to_remove = []
        self.mode = {"HeadReady": False, "TailReady": False, "Final": False}

    def add_ref_image(self, ref):
        self.ref_image = ref

    def profile_init(self):
        num_nodes = self.ori_num_nodes
        self.profile_length = np.ones([self.num_nodes], dtype=float)
        self.profile_orientation = np.zeros([self.num_nodes], dtype=float)
        self.profile_thick_left = np.ones(
            [self.num_nodes], dtype=float) * self.ori_thickness
        self.profile_thick_right = np.ones(
            [self.num_nodes], dtype=float) * self.ori_thickness
        # terminal shrink TODO?
        self.profile_thick_left[0] *= 0
        self.profile_thick_left[num_nodes - 1] *= 0
        self.profile_thick_right[0] *= 0
        self.profile_thick_right[num_nodes - 1] *= 0
        self.profile_thick_left[1] *= 0.5
        self.profile_thick_left[2] *= 0.85
        self.profile_thick_left[num_nodes - 2] *= 0.5
        self.profile_thick_left[num_nodes - 3] *= 0.85
        self.profile_thick_right[1] *= 0.5
        self.profile_thick_right[2] *= 0.85
        self.profile_thick_right[num_nodes - 2] *= 0.5
        self.profile_thick_right[num_nodes - 3] *= 0.85

    def reset(self):
        self.trans_params_orientation_angle = self.ori_trans_params_orientation_angle
        self.trans_params_base_position = self.ori_trans_params_base_position
        self.trans_params_scale = self.ori_trans_params_scale
        self.num_nodes = self.ori_num_nodes
        self.last_change = time()
        self.profile_init()
        self.update_node()

    def operate_profile_gaussion(self, profile_name, position, width, magnitude):
        profile = None
        if profile_name == 'L':
            profile = self.profile_length
        elif profile_name == 'O':
            profile = self.profile_orientation
        elif profile_name == 'TL':
            profile = self.profile_thick_left
        elif profile_name == 'TR':
            profile = self.profile_thick_right
        if profile is None:
            logging.warning(
                "Bad call: operate_profile_gaussion: profile_name=%r" % profile_name)
            return
        self.last_change = time()

        N = self.num_nodes
        position = position % N
        for i in range(position - 3 * width, position + 3 * width):
            # index range check
            if not (i >= 0 and i < N):
                continue
            profile[i] += magnitude * \
                norm.pdf(i, position, width) / \
                norm.pdf(position, position, width)

    def update_node(self):
        if self.last_update >= self.last_change:
            return
        self.last_update = time()

        N = self.num_nodes
        N_middle = N // 2
        theta = self.trans_params_orientation_angle
        p_0 = self.trans_params_base_position
        scale = self.trans_params_scale
        L = self.profile_length
        O = self.profile_orientation
        TL = self.profile_thick_left
        TR = self.profile_thick_right

        p_last = None
        angle_90 = 90 * (PI / 180)

        # front: the direction worn's nodes grow,
        #           for node index > N//2, same as p[i]-p[i-1]
        #           for node index < N//2, same as p[i]-p[i+1]
        # left/right: direction with respect to node index increasing
        for i in range(N_middle, N):
            unit_front = np.array(
                [cos(theta + O[i - 1]), sin(theta + O[i - 1])])
            if i == N_middle:
                this_pos = p_0
            else:
                this_pos = p_last + L[i - 1] * scale * unit_front

            unit_left = np.array(
                [cos(theta + O[i] - angle_90), sin(theta + O[i] - angle_90)])
            unit_right = np.array(
                [cos(theta + O[i] + angle_90), sin(theta + O[i] + angle_90)])
            this_left = this_pos + TL[i] * scale * unit_left
            this_right = this_pos + TR[i] * scale * unit_right

            p_last = self.node_pos[i] = this_pos
            self.node_left[i] = this_left
            self.node_right[i] = this_right

        p_last = self.node_pos[N_middle]
        for i in range(N_middle - 1, -1, -1):
            unit_front = - \
                np.array([cos(theta + O[i + 1]), sin(theta + O[i + 1])])
            this_pos = p_last + L[i + 1] * scale * unit_front

            unit_left = np.array(
                [cos(theta + O[i] - angle_90), sin(theta + O[i] - angle_90)])
            unit_right = np.array(
                [cos(theta + O[i] + angle_90), sin(theta + O[i] + angle_90)])
            this_left = this_pos + TL[i] * scale * unit_left
            this_right = this_pos + TR[i] * scale * unit_right

            p_last = self.node_pos[i] = this_pos
            self.node_left[i] = this_left
            self.node_right[i] = this_right

        self.node_left[N - 1] = self.node_pos[N - 1]
        self.node_right[N - 1] = self.node_pos[N - 1]
        self.node_left[0] = self.node_pos[0]
        self.node_right[0] = self.node_pos[0]

    def apply_motors(self):
        N = self.num_nodes
        if abs(self.profile_orientation[0]) > 90 * PI / 180:
            self.mode["HeadReady"] = True
        if abs(self.profile_orientation[N-1]) > 90 * PI / 180:
            self.mode["TailReady"] = True
        self.iter += 1
        if self.mode["Final"]:
            for m in self.motors:
                if m.name == "ThickenMotor":
                    m.sensor_sigma=0.5
                    m.apply()
        else:
            for m in self.motors:
                m.apply()

    def get_blur_image(self, sigma):
        if sigma not in self.blur_image_cache.keys():
            self.blur_image_cache[sigma] = gaussian_blur(self.image, sigma)
            logging.info(
                'Worm.get_blur_image: added blur image with sigma=%f to cache.' % sigma)
        return self.blur_image_cache[sigma]

    def draw(self, f, axs):
        self.update_node()

        ax, ax_L, ax_O, ax_TL, ax_TR = axs

        # handle the handle
        if f is None:
            return

        # clear previous
        for lc in self.ax_collection_to_remove:
            lc.remove()
        self.ax_collection_to_remove.clear()

        # draw worm
        N = self.num_nodes
        # draw the shape and middle line
        coords = np.empty([3, N, 2])
        for i in range(N):
            coords[0, i, :] = self.node_left[i]
            coords[1, i, :] = self.node_pos[i]
            coords[2, i, :] = self.node_right[i]
        colors = np.array(
            [(1, .5, .5, 1), (.5, .5, 1, 1), (0.35, .85, 0.5, 1)])
        lc = LineCollection(coords, colors=colors, linewidths=2)
        ax.add_collection(lc)
        self.ax_collection_to_remove.append(lc)
        # draw the nodes
        coords = np.empty([N - 2, 3, 2])
        colors = np.empty([N - 2, 4])
        for i in range(1, N - 1):
            coords[i - 1, 0, :] = self.node_left[i]
            coords[i - 1, 1, :] = self.node_pos[i]
            coords[i - 1, 2, :] = self.node_right[i]
            colors[i - 1, :] = np.array([.25, .75, .25, .5])
            if i < N / 5 and self.mode["HeadReady"]:
                colors[i - 1, :] = np.array([1, 0, 0, 1])
            if i > 4 * N / 5 and self.mode["TailReady"]:
                colors[i - 1, :] = np.array([1, 0, 0, 1])
        if not self.mode["Final"]:
            lc = LineCollection(coords, colors=colors, linewidths=1)
            ax.add_collection(lc)
            self.ax_collection_to_remove.append(lc)

        # draw plots
        ax_L.clear()
        ax_O.clear()
        ax_TL.clear()
        ax_TR.clear()
        x = np.arange(N)
        L = self.profile_length
        O = self.profile_orientation
        TL = self.profile_thick_left
        TR = self.profile_thick_right
        ax_L.plot(x, L)
        ax_O.plot(x, O)
        ax_TL.plot(x, TL)
        ax_TR.plot(x, TR)

        # draw end
        title_str = "iteration %d, " % self.iter
        if self.mode["Final"]:
            title_str += "Final Adjustment"
        else:
            title_str += "Head:"
            if self.mode["HeadReady"]:
                title_str += "Ready"
            else:
                title_str += "Not Ready"
            title_str += ", Tail:"
            if self.mode["TailReady"]:
                title_str += "Ready"
            else:
                title_str += "Not Ready"

        ax.set_title(title_str, fontsize=32)
        ax_L.set_title("L", fontsize=32)
        ax_O.set_title("O", fontsize=32)
        ax_TL.set_title("TL", fontsize=32)
        ax_TR.set_title("TR", fontsize=32)
        plt.draw()

    def save(self):
        filename = "worm_data" + strftime("_%Y%m%d_%H%M%S")
        fo = open(filename, "w")
        fo.write("%f \n" % self.trans_params_orientation_angle)
        fo.write("%f %f \n" % (self.trans_params_base_position[0], 
                               self.trans_params_base_position[1]))
        fo.write("%f \n" % self.trans_params_scale)
        N = self.num_nodes
        for i in range(N):
            fo.write("%f %f %f %f \n" % 
                    (self.profile_length[i],
                     self.profile_orientation[i],
                     self.profile_thick_left[i],
                     self.profile_thick_right[i],
                     ))
        fo.close()

    def load(self, filename):
        fi = open(filename, "r")

        nums = list(map(float, fi.read().split()))
        fi.close()

        # self.trans_params_orientation_angle = nums[0]
        # self.trans_params_base_position[0] = nums[1]
        # self.trans_params_base_position[1] = nums[2]
        # self.trans_params_scale = nums[3]

        idx = 4
        N = self.num_nodes
        for i in range(N):
            self.profile_length[i]      = nums[idx]   
            self.profile_orientation[i] = nums[idx+1]  
            self.profile_thick_left[i]  = nums[idx+2] 
            self.profile_thick_right[i] = nums[idx+3]
            idx += 4  

        self.last_change=time()
        self.update_node()

