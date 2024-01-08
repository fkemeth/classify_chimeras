"""
Copyright 2024 Felix P. Kemeth

This file is part of the program classify_chimeras.

classify_chimeras is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

classify_chimeras is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with classify_chimeras.  If not, see <http://www.gnu.org/licenses/>.

Classify coherence/incoherence with the discrete Laplacian.

For systems without a spatial extension, use pairwise distances.
For temporal correlation, use correlation coefficients.
"""

###############################################################################
#                                                                             #
# http://dx.doi.org/10.1063/1.4959804                                         #
#                                                                             #
# Jun 2022                                                                    #
# felix@kemeth.de                                                             #
#                                                                             #
###############################################################################

import unittest

import numpy as np

from classify_chimeras import globaldist, spatial, temporal


def get_one_dimensional_data(num_time_steps: int = 500,
                             num_grid_points: int = 500) -> np.ndarray:
    """
    Create data with one spatial dimension.
    """
    data = np.zeros((num_time_steps, num_grid_points))
    for grid_point in range(num_grid_points):
        data[:, grid_point] = np.sin(np.linspace(0, 10*np.pi, num_time_steps))
        data[:, int(num_grid_points/2+1):] += np.random.randn(
            num_time_steps, int(num_grid_points/2-1))-0.5
    return data


def get_two_dimensional_data(num_time_steps: int = 500,
                             num_grid_points_x: int = 50,
                             num_grid_points_y: int = 50) -> np.ndarray:
    """
    Create data with two spatial dimension.
    """
    data = np.zeros((num_time_steps, num_grid_points_x, num_grid_points_y))
    for grid_point_x in range(num_grid_points_x):
        for grid_point_y in range(num_grid_points_y):
            data[:, grid_point_x, grid_point_y] = np.sin(
                np.linspace(0, 10*np.pi, num_time_steps))
    data[:, int(num_grid_points_x/2):, int(num_grid_points_y/2):] += \
        np.random.rand(
            num_time_steps,
            int(num_grid_points_x/2),
            int(num_grid_points_y/2))-0.5
    return data


class TestOneDimensional(unittest.TestCase):
    """
    Test spatial correlation measures.
    """

    def test_spatial(self):
        """
        Test spatial correlation measures.
        """
        data = get_one_dimensional_data()
        g_zero = spatial(data, nbins=2000)
        self.assertEqual(round(np.mean(g_zero), 2), 0.5, 'Should be 0.5')

    def test_global(self):
        """
        Test global correlation measures.
        """
        data = get_one_dimensional_data()
        g_zero = globaldist(data, nbins=2000)
        self.assertEqual(round(np.mean(g_zero), 2), 0.5, 'Should be 0.5')

    def test_temporal(self):
        """
        Test temporal correlation measures.
        """
        data = get_one_dimensional_data()
        h_data = temporal(data, nbins=500)
        self.assertEqual(round(np.sqrt(h_data[-1]), 2), 0.5, 'Should be 0.5')


class TestTwoDimensional(unittest.TestCase):
    """
    Test spatial correlation measures.
    """

    def test_spatial(self):
        """
        Test spatial correlation measures.
        """
        data = get_two_dimensional_data(500, 200, 200)
        g_zero = spatial(data, nbins=2000)
        self.assertEqual(round(np.mean(g_zero), 2), 0.75, 'Should be 0.75')

    def test_global(self):
        """
        Test global correlation measures.
        """
        data = get_two_dimensional_data(500, 30, 30)
        g_zero = globaldist(data.reshape((data.shape[0], -1)), nbins=2000)
        self.assertEqual(round(np.mean(g_zero), 2), 0.75, 'Should be 0.75')

    def test_temporal(self):
        """
        Test temporal correlation measures.
        """
        data = get_two_dimensional_data(500, 30, 30)
        h_data = temporal(data, nbins=500)
        self.assertEqual(round(np.sqrt(h_data[-1]), 2), 0.75, 'Should be 0.75')


if __name__ == '__main__':
    unittest.main()
