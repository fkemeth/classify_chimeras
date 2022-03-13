"""
Copyright 2022 Felix P. Kemeth

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

import numpy as np
from scipy.sparse import csr_matrix

from classify_chimeras import stencil_1d, stencil_2d


def transform_phases(data: np.ndarray) -> np.ndarray:
    """
    Transform phases into complex plane.

    :param data: data containing phases
    :returns: data with phases mapped into complex plane
    """
    return np.exp(1.0j * data)


def create_stencil(shape: list) -> csr_matrix:
    """
    Create finite difference stencil to calculate curvature for discrete data.

    :param shape: shape of the data
    :returns: stencil as csr matrix
    """
    if len(shape) == 2:
        (_, num_grid_points) = shape
        stencil = stencil_1d.create_stencil(num_grid_points)
    else:
        (_, num_grid_points_x, num_grid_points_y) = shape
        stencil = stencil_2d.create_stencil(num_grid_points_x, num_grid_points_y, 1)
    return stencil


def calculate_curvature(data: np.ndarray) -> np.ndarray:
    """
    Calculate absolute curvature at each point in space.

    :param data: array containing the data
    :param data_shape: dimension of original data
    :returns: array containing the curvatures
    """
    stencil = create_stencil(data.shape)

    if len(data.shape) == 3:
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))

    curvature_data = np.zeros_like(data, dtype="complex")
    for time_step in range(0, data.shape[0]):
        curvature_data[time_step, :] = stencil.dot(data[time_step])
    return np.abs(curvature_data)


def calculate_maximal_curvature(curvature_data: np.ndarray, shape: list, boundaries: str):
    """
    Calculate maximal absolute curvature.

    :param curvature_data: array containing the curvature data
    :param shape: dimension of original data
    :param boundaries: boundary conditions
    :returns: maximal curvature
    """
    if len(shape) == 2:
        if boundaries == "no-flux":
            max_curvature = np.max(curvature_data[:, 1:-1])
        elif boundaries == "periodic":
            max_curvature = np.max(curvature_data)
        else:
            raise ValueError(
                "Please select proper boundary conditions: no-flux or periodic."
            )
    if len(shape) == 3:
        (num_time_steps, num_grid_points_x, num_grid_points_y) = shape
        if boundaries == "no-flux":
            max_curvature = np.max(np.reshape(
                curvature_data,
                (num_time_steps, num_grid_points_x, num_grid_points_y))[:, 1:-1, 1:-1])
        elif boundaries == "periodic":
            max_curvature = np.max(curvature_data)
        else:
            raise ValueError(
                "Please select proper boundary conditions: no-flux or periodic."
            )
    return max_curvature
