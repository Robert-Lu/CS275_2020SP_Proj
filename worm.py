import numpy as np
from math import sin, cos, exp
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from time import sleep
from plt_helper import *
from util import normalized, gaussian_blur, get_image_intensity, _fbetween
from random import random
from time import time
import logging
from matplotlib.widgets import Button


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
        self.blur_image_cache = dict()
        # trandormation parameters:
        self.trans_params_orientation_angle = orientation_angle
        self.trans_params_base_position = base_position
        self.trans_params_scale = scale  # np.array([1, 1], dtype=float)
        self.num_nodes = num_nodes
        # initialization
        self.profile_length      = np.ones([self.num_nodes],  dtype=float)
        self.profile_orientation = np.zeros([self.num_nodes], dtype=float) 
        self.profile_thick_left  = np.ones([self.num_nodes],  dtype=float) * thickness
        self.profile_thick_right = np.ones([self.num_nodes],  dtype=float) * thickness
        # terminal shrink TODO?
        self.profile_thick_left[1] *= 0.5
        self.profile_thick_left[2] *= 0.85
        self.profile_thick_left[num_nodes-2] *= 0.5
        self.profile_thick_left[num_nodes-3] *= 0.85
        self.profile_thick_right[1] *= 0.5
        self.profile_thick_right[2] *= 0.85
        self.profile_thick_right[num_nodes-2] *= 0.5
        self.profile_thick_right[num_nodes-3] *= 0.85
        # # rand for test
        # for i in range(self.num_nodes):
        #     self.profile_length[i] += random() * 0.5
        #     self.profile_orientation[i] += random() * 0.3
        #     self.profile_thick_left[i] += random() * 0.5
        #     self.profile_thick_right[i] += random() * 0.5
        # create nodes
        self.node_pos   = [None for _ in range(self.num_nodes)]
        self.node_left  = [None for _ in range(self.num_nodes)]
        self.node_right = [None for _ in range(self.num_nodes)]
        # update
        self.last_change = time()
        self.last_update = 0
        self.update_node()
        # draw
        self.fig_handle = None
        self.ax_collection_to_remove = []

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
            unit_front = np.array([cos(theta + O[i-1]), sin(theta + O[i-1])])
            if i == N_middle:
                this_pos = p_0
            else:
                this_pos = p_last + L[i-1] * scale * unit_front
            
            unit_left = np.array([cos(theta + O[i] - angle_90), sin(theta + O[i] - angle_90)])
            unit_right = np.array([cos(theta + O[i] + angle_90), sin(theta + O[i] + angle_90)])
            this_left = this_pos + TL[i] * scale * unit_left
            this_right = this_pos + TR[i] * scale * unit_right

            p_last = self.node_pos[i] = this_pos
            self.node_left[i] = this_left
            self.node_right[i] = this_right

        p_last = self.node_pos[N_middle]
        for i in range(N_middle -1, -1, -1):
            unit_front = - np.array([cos(theta + O[i+1]), sin(theta + O[i+1])])
            this_pos = p_last + L[i+1] * scale * unit_front
            
            unit_left = np.array([cos(theta + O[i] - angle_90), sin(theta + O[i] - angle_90)])
            unit_right = np.array([cos(theta + O[i] + angle_90), sin(theta + O[i] + angle_90)])
            this_left = this_pos + TL[i] * scale * unit_left
            this_right = this_pos + TR[i] * scale * unit_right

            p_last = self.node_pos[i] = this_pos
            self.node_left[i] = this_left
            self.node_right[i] = this_right

        self.node_left[N-1] = self.node_pos[N-1]
        self.node_right[N-1] = self.node_pos[N-1]
        self.node_left[0] = self.node_pos[0]
        self.node_right[0] = self.node_pos[0]

    def get_blur_image(self, sigma):
        if sigma not in self.blur_image_cache.keys():
            self.blur_image_cache[sigma] = gaussian_blur(self.image, sigma)
            print("add", sigma)
        return self.blur_image_cache[sigma]

    def draw(self, f, ax):
        self.update_node()

        ## handle the handle
        if f is None:
            return

        ## clear previous
        for lc in self.ax_collection_to_remove:
            lc.remove()
        self.ax_collection_to_remove.clear()
        
        ## draw worm
        N = self.num_nodes
        # draw the shape and middle line
        coords = np.empty([3, N, 2])
        for i in range(N):
            coords[0, i, :] = self.node_left[i]
            coords[1, i, :] = self.node_pos[i]
            coords[2, i, :] = self.node_right[i]
        colors = np.array([(1, .5, .5, 1), (.5, .5, 1, 1), (0.35, .85, 0.5, 1)])
        lc = LineCollection(coords, colors=colors, linewidths=2)
        ax.add_collection(lc)
        self.ax_collection_to_remove.append(lc)
        # draw the nodes
        coords = np.empty([N-2, 3, 2])
        colors = np.empty([N-2, 4])
        for i in range(1, N-1):
            coords[i-1, 0, :] = self.node_left[i]
            coords[i-1, 1, :] = self.node_pos[i]
            coords[i-1, 2, :] = self.node_right[i]
            colors[i-1, :] = np.array([.25, .75, .25, .5])
        lc = LineCollection(coords, colors=colors, linewidths=1)
        ax.add_collection(lc)
        self.ax_collection_to_remove.append(lc)

        ## draw end
        plt.draw()


class Sensor(object):
    def __init__(self, worm):
        self.worm = worm

    def detect(self):
        pass

    def draw(self):
        pass


class NodeSensor(Sensor):
    def __init__(self, worm, node_index):
        super(NodeSensor, self).__init__(worm)
        self.node_index = node_index % self.worm.num_nodes


class TerminalSensor(Sensor):
    def __init__(self, worm, node_index):
        super(TerminalSensor, self).__init__(worm)
        N = self.worm.num_nodes
        self.node_index = node_index % N
        self.node_pos = self.worm.node_pos[self.node_index]

        # the head: front = p[0] - p[1]
        if self.node_index == 0: 
            self.unit_front = normalized(self.worm.node_pos[0] - self.worm.node_pos[1])
            self.unit_left = np.array([self.unit_front[1], -self.unit_front[0]])
            self.unit_right = np.array([-self.unit_front[1], self.unit_front[0]])

        # the tail: front = p[N-1] - p[N-2]
        elif self.node_index == N - 1:
            self.unit_front = normalized(self.worm.node_pos[N - 1] - self.worm.node_pos[N - 2])
            self.unit_left = np.array([self.unit_front[1], -self.unit_front[0]])
            self.unit_right = np.array([-self.unit_front[1], self.unit_front[0]])
        else:
            logging.warning('TerminalSensor.__init__: %d is not a terminal index.' % node_index)

        # logging.info('unit_front %r' % self.unit_front)
        # logging.info('unit_left %r' % self.unit_left)
        # logging.info('unit_right %r' % self.unit_right)

    def detect(self, threshold=0.8, scale=1.0, sigma=1.5):
        w = self.worm
        w.update_node()
        pos = w.node_pos[self.node_index]
        _front = self.unit_front * scale
        _left = self.unit_left * scale
        _right = self.unit_right * scale

        ib = w.get_blur_image(sigma=sigma)
        base_image_intensity = get_image_intensity(ib, pos)
        r_f   = ((get_image_intensity(ib, pos + _front)              / base_image_intensity) - threshold) / (1 - threshold)
        r_ff  = ((get_image_intensity(ib, pos + 2 * _front)          / base_image_intensity) - threshold) / (1 - threshold)
        r_fl  = ((get_image_intensity(ib, pos + _front + _left)      / base_image_intensity) - threshold) / (1 - threshold)
        r_fll = ((get_image_intensity(ib, pos + _front + 2 * _left)  / base_image_intensity) - threshold) / (1 - threshold)
        r_fr  = ((get_image_intensity(ib, pos + _front + _right)     / base_image_intensity) - threshold) / (1 - threshold)
        r_frr = ((get_image_intensity(ib, pos + _front + 2 * _right) / base_image_intensity) - threshold) / (1 - threshold)

        # can go foward? further?
        p_can_foward  = exp(r_f) / E
        p_can_further = exp(r_ff) / E
        # can go left/right?
        p_can_left  = _fbetween(0, (exp(r_fl) * 2 + exp(r_fll)) / 3 / E, 1.0)
        p_can_right = _fbetween(0, (exp(r_fr) * 2 + exp(r_frr)) / 3 / E, 1.0)
        # forward strength (0~1) & direction (-1(L)~+1(R))
        strength = p_can_foward * 0.5
        if strength > 0.333:
            strength += 0.5 * p_can_further
        direction = p_can_right - p_can_left

        return [strength, direction]


class Motor(object):
    def __init__(self, worm):
        self.worm = worm        

    def apply(self):
        logging.warning('abstract base class method called?')


class StretchMotor(Motor):
    def __init__(self, worm):
        super(StretchMotor, self).__init__(worm)
        self.head_sensor = TerminalSensor(worm, 0)
        self.tail_sensor = TerminalSensor(worm, -1)

    def apply(self, scale_length=1.0, scale_orientation=1.0):
        w = self.worm
        N = w.num_nodes
        logging.info("StretchMotor(%r) begins" % w)
        head_strength, head_direction = self.head_sensor.detect(scale=1.0)
        w.profile_length[0:5] += head_strength * scale_length
        w.profile_orientation[0:5] += head_direction * scale_orientation
        logging.info("Head terminal sensor: detect: stretch strength = %f" % head_strength)
        logging.info("Head terminal sensor: detect: stretch direction = %f (+right, -left)" % head_direction)
        tail_strength, tail_direction = self.tail_sensor.detect(scale=1.0)
        w.profile_length[N-5:N-1] += tail_strength * scale_length
        w.profile_orientation[N-5:N-1] += tail_direction * scale_orientation
        logging.info("Tail terminal sensor: detect: stretch strength = %f" % tail_strength)
        logging.info("Tail terminal sensor: detect: stretch direction = %f (+right, -left)" % tail_direction)

        w.last_change = time()
        logging.info("StretchMotor(%r) ends" % w)


class ThickenMotor(Motor):
    def __init__(self, worm):
        super(ThickenMotor, self).__init__(worm)


class NomalizationMotor(Motor):
    def __init__(self, worm):
        super(NomalizationMotor, self).__init__(worm)
