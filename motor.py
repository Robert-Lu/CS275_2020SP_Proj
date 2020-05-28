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
from sensor import *


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