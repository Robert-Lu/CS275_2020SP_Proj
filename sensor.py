import numpy as np
from math import sin, cos, exp
import scipy.ndimage as nd
from scipy.stats import norm
from util import normalized, gaussian_blur, get_image_intensity, _fbetween
from random import random
from time import time
import logging
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d

PI = 3.1415926535
E = exp(1)


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
        self.node_index = node_index % w.num_nodes
        self.node_pos = w.node_pos[self.node_index]

        self.update_directions()

    def update_directions(self):
        w = self.worm
        N = w.num_nodes
        I = self.node_index
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
        self.update_directions()

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

        left_strength = -0.25
        right_strength = -0.25
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

        self.update_directions()

    def update_directions(self):
        N = self.worm.num_nodes

        # the head: front = p[0] - p[1]
        if self.node_index == 0:
            self.node_prev_index = 1
            self.unit_front = normalized(
                self.worm.node_pos[0] - self.worm.node_pos[1])
            self.unit_left = np.array(
                [self.unit_front[1], -self.unit_front[0]])
            self.unit_right = np.array(
                [-self.unit_front[1], self.unit_front[0]])

        # the tail: front = p[N-1] - p[N-2]
        elif self.node_index == N - 1:
            self.node_prev_index = N - 2
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
        self.update_directions()
        pos = w.node_pos[self.node_index]
        # _length = w.profile_length[self.node_prev_index] * w.trans_params_scale
        _front = self.unit_front * scale #* _length
        _left = self.unit_left * scale #* _length
        _right = self.unit_right * scale #* _length

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

        strength = -0.15
        if p_can_foward > 0.667:
            strength += p_can_foward * 0.5
            if p_can_further > 0.667:
                strength += p_can_further * 0.5
        else:
            strength += p_can_foward * 0.25

        direction = p_can_right - p_can_left

        return strength, direction
