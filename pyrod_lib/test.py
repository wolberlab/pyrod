""" PyRod - dynamic molecular interaction fields (dMIFs), based on tracing water molecules in MD simulations.

Released under the GNU Public Licence v2.

This module is used to perform unit tests.
"""


# python standard libraries
import unittest
import numpy as np

# pyrod modules
try:
    from pyrod.pyrod_lib.math import distance, maximal_distance, mean, squared_deviations_from_mean, \
        standard_deviation, angle, maximal_angle
except ImportError:
    from pyrod_lib.math import distance, maximal_distance, mean, squared_deviations_from_mean, \
        standard_deviation, angle, maximal_angle


class TestHelperMath(unittest.TestCase):
    """ Class is used for testing function from helper_math module. """

    def test_distance(self):
        """ This function tests the distance function. """
        test_cases = [[[0, 0, 0], [0, 0, 0], 0], [[0, 0, 0], [-1.0, 1, 1.0], 1.7320508075688772],
                      [np.array([0, 0, 0]), np.array([1, 1.0, 1]), 1.7320508075688772]]
        for test_case in test_cases:
            self.assertAlmostEqual(distance(test_case[0], test_case[1]), test_case[2])

    def test_maximal_distance(self):
        """ This function tests the maximal_distance function. """
        test_cases = [[np.array([[-1.0, -2, -5], [-1, -5, -5], [-5,  5,  0], [5, -5, -4], [-5, -1, -2], [3, -2, -1],
                                 [0,  1, -3], [0,  3, -5], [2, -3,  5], [3,  1, -1]]), 14.696938456699069],
                      [[[-1.0, -2, -5], [-1, -5, -5], [-5, 5, 0], [5, -5, -4], [-5, -1, -2], [3, -2, -1], [0, 1, -3],
                        [0, 3, -5], [2, -3, 5], [3, 1, -1]], 14.696938456699069]
                      ]
        for test_case in test_cases:
            self.assertAlmostEqual(maximal_distance(test_case[0]), test_case[1])

    def test_mean(self):
        """ This function tests the mean function. """
        test_cases = [[np.array([-1.0, -2, -5, -1, -5, -5, -5, 5, 0, 5, -5, -4, -5, -1, -2, 3, -2, -1, 0, 1, -3, 0, 3,
                                 -5, 2, -3, 5, 3, 1, -1]), -0.9333333333333333],
                      [[-1.0, -2, -5, -1, -5, -5, -5, 5, 0, 5, -5, -4, -5, -1, -2, 3, -2, -1, 0, 1, -3, 0, 3, -5, 2,
                        -3, 5, 3, 1, -1], -0.9333333333333333]
                      ]
        for test_case in test_cases:
            self.assertAlmostEqual(mean(test_case[0]), test_case[1])

    def test_squared_deviations_from_mean(self):
        """ This function tests the squared_deviations_from_mean function. """
        test_cases = [[np.array([-1.0, -2, -5, -1, -5, -5, -5, 5, 0, 5, -5, -4, -5, -1, -2, 3, -2, -1, 0, 1, -3, 0, 3,
                                 -5, 2, -3, 5, 3, 1, -1]), 307.8666666666666],
                      [[-1.0, -2, -5, -1, -5, -5, -5, 5, 0, 5, -5, -4, -5, -1, -2, 3, -2, -1, 0, 1, -3, 0, 3, -5, 2,
                        -3, 5, 3, 1, -1], 307.8666666666666]
                      ]
        for test_case in test_cases:
            self.assertAlmostEqual(squared_deviations_from_mean(test_case[0]), test_case[1])

    def test_population_standard_deviation(self):
        """ This function tests the population_standard_deviation function. """
        test_cases = [[np.array([-1.0, -2, -5, -1, -5, -5, -5, 5, 0, 5, -5, -4, -5, -1, -2, 3, -2, -1, 0, 1, -3, 0, 3,
                                 -5, 2, -3, 5, 3, 1, -1]), 3.20347034046239],
                      [[-1.0, -2, -5, -1, -5, -5, -5, 5, 0, 5, -5, -4, -5, -1, -2, 3, -2, -1, 0, 1, -3, 0, 3, -5, 2,
                        -3, 5, 3, 1, -1], 3.20347034046239]
                      ]
        for test_case in test_cases:
            self.assertAlmostEqual(standard_deviation(test_case[0]), test_case[1])

    def test_angle(self):
        """ This function tests the angle function. """
        test_cases = [[np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0],
                      [np.array([-1.0, 0, 0]), np.array([0, 0, 0]), np.array([0, -1.0, 0]), 90.0],
                      [np.array([-1.0, 0, 0]), np.array([0, 0, 0]), np.array([0, 1, 0]), 90.0],
                      [np.array([-1.0, 0, 0]), np.array([0, 0, 0]), np.array([1, 0, 0]), 180.0]
                      ]
        for test_case in test_cases:
            self.assertAlmostEqual(angle(test_case[0], test_case[1], test_case[2]), test_case[3])




if __name__ == '__main__':
    unittest.main()
