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
from time import time
import logging
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d


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
        self.update_node()
        # draw
        self.fig_handle = None
        self.ax_collection_to_remove = []

    def profile_init(self):
        num_nodes = self.ori_num_nodes
        self.profile_length = np.ones([self.num_nodes], dtype=float)
        self.profile_orientation = np.zeros([self.num_nodes], dtype=float)
        self.profile_thick_left = np.ones(
            [self.num_nodes], dtype=float) * self.ori_thickness
        self.profile_thick_right = np.ones(
            [self.num_nodes], dtype=float) * self.ori_thickness
        # terminal shrink TODO?
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

    def get_blur_image(self, sigma):
        if sigma not in self.blur_image_cache.keys():
            self.blur_image_cache[sigma] = gaussian_blur(self.image, sigma)
            logging.info(
                'Worm.get_blur_image: added blur image with sigma=%f to cache.' % sigma)
        return self.blur_image_cache[sigma]

    def draw(self, f, ax):
        self.update_node()

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
        lc = LineCollection(coords, colors=colors, linewidths=1)
        ax.add_collection(lc)
        self.ax_collection_to_remove.append(lc)

        # draw end
        plt.draw()


class Sensor(object):
    def __init__(self, worm):
        self.worm = worm
        worm.sensors.append(self)
        logging.info("Wrom[%r] add Sensor[%r]" % (worm, self))

    def detect(self):
        pass

    def draw(self):
        pass

class NodeSensor(Sensor):

    # For node sensor, left and right are consistant with TL and TR
    #                  front is the direction node index increasing.
    def __init__(self, worm, node_index):
        super(NodeSensor, self).__init__(worm)
        w = self.worm
        I = self.node_index = node_index % w.num_nodes
        self.node_pos = w.node_pos[self.node_index]
        N = w.num_nodes
        N_middle = N // 2
        theta = w.trans_params_orientation_angle
        O = w.profile_orientation

        angle_90 = 90 * (PI / 180)
        if I >= N_middle:
            self.unit_front = np.array(
                [cos(theta + O[I - 1]), sin(theta + O[I - 1])])
        else:
            self.unit_front = np.array(
                [cos(theta + O[I + 1]), sin(theta + O[I + 1])])
        self.unit_left = np.array(
            [cos(theta + O[I] - angle_90), sin(theta + O[I] - angle_90)])
        self.unit_right = np.array(
            [cos(theta + O[I] + angle_90), sin(theta + O[I] + angle_90)])

    def detect(self, threshold=0.9, scale=1.0, sigma=1.5, verbose=False):
        w = self.worm
        w.update_node()
        pos = w.node_pos[self.node_index]
        left_pos = w.node_left[self.node_index]
        right_pos = w.node_right[self.node_index]
        _left = self.unit_left * scale
        _right = self.unit_right * scale

        ib = w.get_blur_image(sigma=sigma)
        base_image_intensity = get_image_intensity(ib, pos)
        r_l = ((get_image_intensity(ib, left_pos + _left) /
                base_image_intensity) - threshold) / (1 - threshold)
        r_ll = ((get_image_intensity(ib, left_pos + 2 * _left) /
                 base_image_intensity) - threshold) / (1 - threshold)
        r_r = ((get_image_intensity(ib, right_pos + _right) /
                base_image_intensity) - threshold) / (1 - threshold)
        r_rr = ((get_image_intensity(ib, right_pos + 2 * _right) /
                 base_image_intensity) - threshold) / (1 - threshold)

        if verbose:
            logging.info('r_l=%f, r_ll=%f, r_r=%f, r_rr=%f' %
                         (r_l, r_ll, r_r, r_rr))

        # can go left? further?
        p_can_left = exp(r_l) / E
        p_can_left_further = exp(r_ll) / E
        # can go right? further?
        p_can_right = exp(r_r) / E
        p_can_right_further = exp(r_rr) / E
        if verbose:
            logging.info('p_can_left=%f, p_can_left_further=%f, p_can_right=%f, p_can_right_further=%f' % (
                p_can_left, p_can_left_further, p_can_right, p_can_right_further))

        left_strength = -0.15
        right_strength = -0.15
        if p_can_left > 0.667:
            left_strength += p_can_left * 0.5
            if p_can_left_further > 0.667:
                left_strength += p_can_left_further * 0.5
        if p_can_right > 0.667:
            right_strength += p_can_right * 0.5
            if p_can_right_further > 0.667:
                right_strength += p_can_right_further * 0.5

        return left_strength, right_strength


class TerminalSensor(Sensor):

    # For terminal sensor, left and right are with respect to the front direction
    #  and the front directions are different for head and tail node.
    def __init__(self, worm, node_index):
        super(TerminalSensor, self).__init__(worm)
        N = self.worm.num_nodes
        self.node_index = node_index % N
        self.node_pos = self.worm.node_pos[self.node_index]

        # the head: front = p[0] - p[1]
        if self.node_index == 0:
            self.unit_front = normalized(
                self.worm.node_pos[0] - self.worm.node_pos[1])
            self.unit_left = np.array(
                [self.unit_front[1], -self.unit_front[0]])
            self.unit_right = np.array(
                [-self.unit_front[1], self.unit_front[0]])

        # the tail: front = p[N-1] - p[N-2]
        elif self.node_index == N - 1:
            self.unit_front = normalized(
                self.worm.node_pos[N - 1] - self.worm.node_pos[N - 2])
            self.unit_left = np.array(
                [self.unit_front[1], -self.unit_front[0]])
            self.unit_right = np.array(
                [-self.unit_front[1], self.unit_front[0]])
        else:
            logging.warning(
                'TerminalSensor.__init__: %d is not a terminal index.' % node_index)

    def detect(self, threshold=0.9, scale=1.0, sigma=1.5, verbose=False):
        w = self.worm
        w.update_node()
        pos = w.node_pos[self.node_index]
        _front = self.unit_front * scale
        _left = self.unit_left * scale
        _right = self.unit_right * scale

        ib = w.get_blur_image(sigma=sigma)
        base_image_intensity = get_image_intensity(ib, pos)
        r_f = ((get_image_intensity(ib, pos + _front) /
                base_image_intensity) - threshold) / (1 - threshold)
        r_ff = ((get_image_intensity(ib, pos + 2 * _front) /
                 base_image_intensity) - threshold) / (1 - threshold)
        r_fl = ((get_image_intensity(ib, pos + _front + _left) /
                 base_image_intensity) - threshold) / (1 - threshold)
        r_fll = ((get_image_intensity(ib, pos + _front + 2 * _left) /
                  base_image_intensity) - threshold) / (1 - threshold)
        r_fr = ((get_image_intensity(ib, pos + _front + _right) /
                 base_image_intensity) - threshold) / (1 - threshold)
        r_frr = ((get_image_intensity(ib, pos + _front + 2 * _right) /
                  base_image_intensity) - threshold) / (1 - threshold)

        if verbose:
            logging.info('r_f=%f, r_ff=%f, r_fl=%f, r_fll=%f, r_fr=%f, r_frr=%f' % (
                r_f, r_ff, r_fl, r_fll, r_fr, r_frr))

        # can go foward? further?
        p_can_foward = exp(r_f) / E
        p_can_further = exp(r_ff) / E
        # can go left/right?
        p_can_left = _fbetween(0, (exp(r_fl) * 2 + exp(r_fll)) / 3 / E, 1.0)
        p_can_right = _fbetween(0, (exp(r_fr) * 2 + exp(r_frr)) / 3 / E, 1.0)
        # forward strength (0~1) & direction (-1(L)~+1(R))
        if verbose:
            logging.info('p_can_foward=%f, p_can_further=%f, p_can_left=%f, p_can_right=%f' % (
                p_can_foward, p_can_further, p_can_left, p_can_right))

        strength = 0
        if p_can_foward > 0.667:
            strength += p_can_foward * 0.5
            if p_can_further > 0.667:
                strength += p_can_further * 0.5

        direction = p_can_right - p_can_left

        return strength, direction


class Motor(object):
    def __init__(self, worm):
        self.worm = worm
        worm.motors.append(self)
        logging.info("Wrom[%r] add Motor[%r]" % (worm, self))

    def apply(self):
        logging.warning('abstract base class method called?')


class StretchMotor(Motor):
    def __init__(self, worm, scale_length=1.0, scale_orientation=1.0,
                 sensor_scale=1.0, sensor_sigma=1.5, sensor_threshold=0.9):
        super(StretchMotor, self).__init__(worm)
        self.head_sensor = TerminalSensor(worm, 0)
        self.tail_sensor = TerminalSensor(worm, -1)
        self.scale_length = scale_length
        self.scale_orientation = scale_orientation
        self.sensor_scale = sensor_scale
        self.sensor_sigma = sensor_sigma
        self.sensor_threshold = sensor_threshold

    def apply(self, verbose=True):
        w = self.worm
        N = w.num_nodes
        if verbose:
            logging.info("StretchMotor(%r) begins" % w)
        head_strength, head_direction = self.head_sensor.detect(
            scale=self.sensor_scale, threshold=self.sensor_threshold, sigma=self.sensor_sigma, verbose=verbose)
        w.operate_profile_gaussion(
            'L', 0, 3, head_strength * self.scale_length)
        w.operate_profile_gaussion(
            'O', 0, 3, head_direction * self.scale_orientation)
        if verbose:
            logging.info(
                "Head terminal sensor: detect: stretch strength = %f" % head_strength)
            logging.info(
                "Head terminal sensor: detect: stretch direction = %f (+right, -left)" % head_direction)
        tail_strength, tail_direction = self.tail_sensor.detect(
            scale=self.sensor_scale, threshold=self.sensor_threshold, sigma=self.sensor_sigma, verbose=verbose)
        w.operate_profile_gaussion(
            'L', -1, 3, tail_strength * self.scale_length)
        w.operate_profile_gaussion(
            'O', -1, 3, tail_direction * self.scale_orientation)
        if verbose:
            logging.info(
                "Tail terminal sensor: detect: stretch strength = %f" % tail_strength)
            logging.info(
                "Tail terminal sensor: detect: stretch direction = %f (+right, -left)" % tail_direction)

        w.last_change = time()
        if verbose:
            logging.info("StretchMotor(%r) ends" % w)


class ThickenMotor(Motor):
    def __init__(self, worm, scale_thickness=1.0,
                 sensor_scale=1.0, sensor_sigma=1.5, sensor_threshold=0.9):
        super(ThickenMotor, self).__init__(worm)
        w = self.worm
        N = w.num_nodes
        self.sensors = [None for _ in range(N)]
        for i in range(1, N - 1):
            self.sensors[i] = NodeSensor(worm, i)
        self.scale_thickness = scale_thickness
        self.sensor_scale = sensor_scale
        self.sensor_sigma = sensor_sigma
        self.sensor_threshold = sensor_threshold

    def apply(self, verbose=True):
        w = self.worm
        N = w.num_nodes
        if verbose:
            logging.info("ThickenMotor(%r) begins" % w)

        for i in range(1, N - 1):
            sensor = self.sensors[i]
            ls, rs = sensor.detect(
                scale=self.sensor_scale, threshold=self.sensor_threshold, sigma=self.sensor_sigma, verbose=verbose)
            if verbose:
                logging.info(
                    "Node terminal sensor[%d]: detect: left/right strength = %f, %f" % (i, ls, rs))
            w.operate_profile_gaussion('TL', i, 1, ls * self.scale_thickness)
            w.operate_profile_gaussion('TR', i, 1, rs * self.scale_thickness)

        w.last_change = time()
        if verbose:
            logging.info("ThickenMotor(%r) ends" % w)


class NomalizationMotor(Motor):
    def __init__(self, worm):
        super(NomalizationMotor, self).__init__(worm)

    def apply(self, verbose=True):
        # L = self.worm.profile_length
        # AL = np.average(L)
        self.worm.profile_length = gaussian_filter1d(self.worm.profile_length,
                                                     sigma=1, mode='nearest')
        self.worm.last_change = time()
