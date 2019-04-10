""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module contains functions for mathematical operations.
"""


# external libraries
import math
import numpy as np
from scipy.spatial.distance import pdist


def distance(x, y):
    """ This function returns the euclidean distance between two point in three dimensional space. """
    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2) ** 0.5


def maximal_distance(positions):
    """ This function returns the maximal distance between coordinates in three dimensional space. """
    return np.amax(pdist(positions))


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


def vector(a, b):
    """ This function returns the vector in 3-dimensional space going from a to b. """
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]]


def dot_product(a, b):
    """ This function returns the dot product of 3-dimensional vectors a and b. """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross_product(a, b):
    """ This function returns the cross product of 3-dimensional vectors a and b. """
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def normal(a, b, c):
    """ This function returns the normal of a plane defined by 3 points with b as the origin of the normal. """
    return cross_product(vector(b, a), vector(b, c))


def norm(vector):
    """ This function returns the length of a vector. """
    return ((vector[0] ** 2) + (vector[1] ** 2) + (vector[2] ** 2)) ** 0.5


def angle(a, b, c):
    """ This function returns the angle in degrees between 3 coordinates with b in the center. """
    ba = vector(b, a)
    bc = vector(b, c)
    cosine_angle = dot_product(ba, bc)
    sine_angle = norm(cross_product(ba, bc))
    return math.degrees(math.atan2(sine_angle, cosine_angle))


def vector_angle(a, b):
    """ This function returns the angle in degrees between 2 3-dimensional vectors. """
    cosine_angle = dot_product(a, b)
    sine_angle = norm(cross_product(a, b))
    return math.degrees(math.atan2(sine_angle, cosine_angle))


def opposite(alpha, c):
    """ This function returns the length of opposite a in a rectangular triangle by using angle alpha and the length of
    hypotenuse c. """
    a = math.sin(math.radians(alpha)) * c
    return a


def adjacent(alpha, c):
    """ This function returns the length of adjacent b in a rectangular triangle by using angle alpha and the length of
        hypotenuse c. """
    b = math.cos(math.radians(alpha)) * c
    return b


def rotate_vector(vector, axis, angle):
    """ This function rotates a vector around a given axis for a given angle and returns a new vector scaled by given
    length. """
    angle = math.radians(angle)
    axis_length = norm(axis)
    axis = [axis[0] / axis_length, axis[1] / axis_length, axis[2] / axis_length]
    x = (vector[0] * (math.cos(angle) + ((axis[0] ** 2) * (1 - math.cos(angle))))) + \
        (vector[1] * ((axis[0] * axis[1] * (1 - math.cos(angle))) - (axis[2] * math.sin(angle)))) + \
        (vector[2] * ((axis[0] * axis[2] * (1 - math.cos(angle))) + (axis[1] * math.sin(angle))))
    y = (vector[0] * ((axis[1] * axis[0] * (1 - math.cos(angle))) + (axis[2] * math.sin(angle)))) + \
        (vector[1] * (math.cos(angle) + ((axis[1] ** 2) * (1 - math.cos(angle))))) + \
        (vector[2] * ((axis[1] * axis[2] * (1 - math.cos(angle))) - (axis[0] * math.sin(angle))))
    z = (vector[0] * ((axis[2] * axis[0] * (1 - math.cos(angle))) - (axis[1] * math.sin(angle)))) + \
        (vector[1] * ((axis[2] * axis[1] * (1 - math.cos(angle))) + (axis[0] * math.sin(angle)))) + \
        (vector[2] * (math.cos(angle) + ((axis[2] ** 2) * (1 - math.cos(angle)))))
    return [x, y, z]


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
