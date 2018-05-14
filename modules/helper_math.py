""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

This module contains helper functions for mathematical operations.
"""


# external libraries
import numpy as np
from scipy.spatial.distance import pdist, squareform


def distance(x, y):
    """ This function returns the euclidean distance between two point in three dimensional space. """
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2) ** 0.5


def maximal_distance(positions):
    """ This function returns the maximal distance between coordinates in three dimensional space. """
    return np.amax(squareform(pdist(positions)))


def mean(data):
    """ This function returns the arithmetic mean. """
    return sum(data) / len(data)


def squared_deviations_from_mean(data):
    """ This function returns the squared deviations from mean. """
    c = mean(data)
    return sum((x - c) ** 2 for x in data)


def standard_deviation(data):
    """ This functions returns the population standard deviation. """
    return (squared_deviations_from_mean(data) / len(data)) ** 0.5


def angle(a, b, c):
    """ This function returns the angle in degrees between 3 coordinates with b in the center. """
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc)
    sine_angle = np.linalg.norm(np.cross(ba, bc))
    return np.degrees(np.arctan2(sine_angle, cosine_angle))


def maximal_angle(positions, center_position, origin_position=None):
    """ This function returns the maximal angle of all possible position combinations with the center_position as
    vertex of the angle and the indices of the involved positions. If origin_position is given, angles to be compared
    are defined by origin_position, center_position and each position, respectively. """
    angle_maximum = 0
    indices = [None, None]
    for index_1, position_1 in enumerate(positions):
        if origin_position is None:
            for index_2, position_2 in enumerate(positions):
                current_angle = angle(position_1, center_position, position_2)
                if current_angle > angle_maximum:
                    angle_maximum = current_angle
                    indices = [index_1, index_2]
        else:
            current_angle = angle(position_1, center_position, origin_position)
            if current_angle > angle_maximum:
                angle_maximum = current_angle
                indices[0] = index_1
    if origin_position is None:
        return [angle_maximum, indices[0], indices[1]]
    else:
        return [angle_maximum, indices[0]]
