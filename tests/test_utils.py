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

from classify_chimeras.utils import (compute_pairwise_correlation_coefficients,
                                     transform_phases)


def get_one_dimensional_data(num_time_steps: int = 500,
                             num_grid_points: int = 500) -> np.ndarray:
    """
    Create data with one spatial dimension.
    """
    data = np.zeros((num_time_steps, num_grid_points))
    for grid_point in range(num_grid_points):
        data[:, grid_point] = np.linspace(0, 100, num_time_steps)+2*np.pi*np.random.rand()
    return data


class CorrelationDimensional(unittest.TestCase):
    """
    Test spatial correlation measures.
    """

    def test_correlation(self):
        """
        Test temporal correlation measures.
        """
        data = get_one_dimensional_data(500, 20)
        data = transform_phases(data)
        correlation_coefficients = compute_pairwise_correlation_coefficients(data)
        self.assertTrue((np.round(np.abs(correlation_coefficients), 9)).all(),
                        'Absolute value of all correlation coefficients should be 1.')


if __name__ == '__main__':
    unittest.main()
