import numpy as np
from math import sin, cos, exp
import scipy.ndimage as nd
from scipy.stats import norm
from util import normalized, gaussian_blur, get_image_intensity, _fbetween
from random import random
from time import time
from itertools import chain
import logging
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d
from sensor import *

class Motor(object):
    def __init__(self, worm):
        self.worm = worm
        self.name = "Motor"
        worm.motors.append(self)
        logging.info("Wrom[%r] add Motor[%r]" % (worm, self))

    def apply(self):
        logging.warning('abstract base class method called?')


class StretchMotor(Motor):
    def __init__(self, worm, scale_length=1.0, scale_orientation=1.0,
                 sensor_scale=1.0, sensor_sigma=1.5, sensor_threshold=0.9):
        super(StretchMotor, self).__init__(worm)
        self.name = "StretchMotor"
        self.head_sensor = TerminalSensor(worm, 0)
        self.tail_sensor = TerminalSensor(worm, -1)
        self.scale_length = scale_length
        self.scale_orientation = scale_orientation
        self.sensor_scale = sensor_scale
        self.sensor_sigma = sensor_sigma
        self.sensor_threshold = sensor_threshold
        self.final_index = 0

    def apply(self, verbose=False):
        w = self.worm
        N = w.num_nodes
        head_use_ref = w.mode["HeadReady"]
        tail_use_ref = w.mode["TailReady"]
        if verbose:
            logging.info("StretchMotor(%r) begins" % w)
        head_strength, head_direction = self.head_sensor.detect(
            scale=self.sensor_scale, threshold=self.sensor_threshold, sigma=self.sensor_sigma, verbose=verbose, use_ref=head_use_ref)
        w.operate_profile_gaussion(
            'L', 0, N // 10, head_strength * self.scale_length)
        w.operate_profile_gaussion(
            'O', 0, N // 20, head_direction * self.scale_orientation)
        if verbose:
            logging.info(
                "Head terminal sensor: detect: stretch strength = %f" % head_strength)
            logging.info(
                "Head terminal sensor: detect: stretch direction = %f (+right, -left)" % head_direction)
        if not tail_use_ref:
            tail_strength, tail_direction = self.tail_sensor.detect(
                scale=self.sensor_scale, threshold=self.sensor_threshold, sigma=self.sensor_sigma, verbose=verbose, use_ref=False)
        else:
            tail_strength, tail_direction = self.tail_sensor.detect(
                scale=0.25, threshold=self.sensor_threshold, sigma=self.sensor_sigma, verbose=verbose, use_ref=False)
            # tail_direction *= 0.2
        # 
        if head_use_ref and tail_strength < 0 and tail_use_ref and head_strength < 0:
            self.final_index += 1
            logging.info("F%d" % self.final_index)
            if self.final_index > 1:
                w.mode["Final"] = True
        w.operate_profile_gaussion(
            'L', -1, N // 10, tail_strength * self.scale_length)
        w.operate_profile_gaussion(
            'O', -1, N // 20, tail_direction * self.scale_orientation)
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
        self.name = "ThickenMotor"
        w = self.worm
        N = w.num_nodes
        self.sensors = [None for _ in range(N)]
        for i in range(1, N - 1):
            self.sensors[i] = NodeSensor(worm, i)
        self.scale_thickness = scale_thickness
        self.sensor_scale = sensor_scale
        self.sensor_sigma = sensor_sigma
        self.sensor_threshold = sensor_threshold

    def apply(self, verbose=False):
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
    def __init__(self, worm, balance_threshold=0.6, balance_scale=0.1):
        super(NomalizationMotor, self).__init__(worm)
        self.name = "NomalizationMotor"
        self.balance_threshold = balance_threshold
        self.balance_scale = balance_scale

    def apply(self, verbose=False):
        w = self.worm
        N = w.num_nodes

        ## Part 1: Balance the TL and TR profiles.
        # range N_middle-->tail: i_prev = i - 1
        # range head<--N_middle: i_prev = i + 1
        # on N_middle: special case
        N_middle = N // 2
        theta = w.trans_params_orientation_angle
        p_0 = w.trans_params_base_position
        scale = w.trans_params_scale
        L = w.profile_length
        O = w.profile_orientation
        TL = w.profile_thick_left
        TR = w.profile_thick_right

        indices = chain(range(N_middle-1, 0, -1), range(N_middle+1, N-1))
        for i in indices:
            if i == N_middle:
                pass  # special
            else:
                if i > N_middle:
                    i_prev = i - 1
                    orientation = +1
                else:
                    i_prev = i + 1
                    orientation = -1
                if TL[i] / TR[i] < self.balance_threshold or TR[i] / TL[i] < self.balance_threshold:
                    # if not balance
                    diff = TR[i] - TL[i] * scale # the left-right difference
                    delta_length = diff * self.balance_scale  # how much (length) to tune
                    length = L[i_prev] * scale  # the length between nodes
                    delta_angle = delta_length / length  # the angle offset (delta_theta)
                    O[i_prev] += delta_angle * orientation 
                    # O[i] += delta_angle * orientation / 2
                    TL[i] += delta_length
                    TR[i] -= delta_length
                    if verbose:
                        logging.info("Balance node %d: delta=(%f, %f, %f)" % (i, delta_length, length, delta_angle))

        ## Part 2: Smooth the profiles.
        w.profile_length = gaussian_filter1d(w.profile_length, sigma=2, mode='nearest')
        w.profile_orientation = gaussian_filter1d(w.profile_orientation, sigma=1, mode='nearest')
        w.profile_thick_left = gaussian_filter1d(w.profile_thick_left, sigma=1, mode='constant', cval=1)
        w.profile_thick_right = gaussian_filter1d(w.profile_thick_right, sigma=1, mode='constant', cval=1)

        ## Part 3: Normalize the length.
        average_L = np.average(w.profile_length)
        # w.profile_length = np.clip(w.profile_length, average_L / 2, average_L * 2)

        self.worm.last_change = time()