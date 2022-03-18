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

from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

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


def compute_curvature(data: np.ndarray) -> np.ndarray:
    """
    Calculate absolute curvature at each point in space.

    :param data: array containing the data
    :param data_shape: dimension of original data
    :returns: array containing the curvatures
    """
    stencil = create_stencil(data.shape)

    curvature_data = np.zeros_like(data.reshape((data.shape[0], -1)), dtype='complex')
    for time_step in range(0, data.shape[0]):
        curvature_data[time_step, :] = stencil.dot(data[time_step].flatten())

    curvature_data = curvature_data.reshape(data.shape)
    return np.abs(curvature_data)


def compute_maximal_curvature(curvature_data: np.ndarray, boundaries: str):
    """
    Calculate maximal absolute curvature.

    :param curvature_data: array containing the curvature data
    :param boundaries: boundary conditions
    :returns: maximal curvature
    """
    if len(curvature_data.shape) == 2:
        if boundaries == 'no-flux':
            max_curvature = np.max(curvature_data[:, 1:-1])
        elif boundaries == 'periodic':
            max_curvature = np.max(curvature_data)
        else:
            raise ValueError(
                'Please select proper boundary conditions: no-flux or periodic.'
            )
    if len(curvature_data.shape) == 3:
        if boundaries == 'no-flux':
            max_curvature = np.max(curvature_data[:, 1:-1, 1:-1])
        elif boundaries == 'periodic':
            max_curvature = np.max(curvature_data)
        else:
            raise ValueError(
                'Please select proper boundary conditions: no-flux or periodic.'
            )
    return max_curvature


def compute_normalized_curvature_histograms(
    curvature_data: np.ndarray, max_curvature: float, nbins: int, boundaries: str
) -> np.ndarray:
    """
    Compute histograms of curvature data.

    :param curvature_data: array containing the curvature data
    :param max_curvature: maximum curvature
    :param nbins: number of histogram bins
    :param boundaries: boundary conditions
    :returns: normalized histograms of curvature data
    """
    num_time_steps = curvature_data.shape[0]

    histdat = np.zeros((num_time_steps, nbins))
    if len(curvature_data.shape) == 2:
        num_grid_points = curvature_data.shape[1]
        if boundaries == 'no-flux':
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step, 1:-1], nbins, range=(0, max_curvature)
                )[0]
            normalization = float(num_grid_points - 2)
        elif boundaries == 'periodic':
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step], nbins, range=(0, max_curvature)
                )[0]
            normalization = float(num_grid_points)
        else:
            raise ValueError(
                'Please select proper boundary conditions: no-flux or periodic.'
            )
    if len(curvature_data.shape) == 3:
        (_, num_grid_points_x, num_grid_points_y) = curvature_data.shape
        if boundaries == 'no-flux':
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step, 1:-1, 1:-1],
                    nbins,
                    range=(0, max_curvature),
                )[0]
            normalization = float((num_grid_points_x - 2) * (num_grid_points_y - 2))
        elif boundaries == 'periodic':
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step], nbins, range=(0, max_curvature)
                )[0]
            normalization = float(num_grid_points_x * num_grid_points_y)
        else:
            raise ValueError(
                'Please select proper boundary conditions: no-flux or periodic.'
            )
    return histdat / normalization


def coarse_grain_data(data: np.ndarray, num_coarse: int = 1500) -> np.ndarray:
    """
    Downsample data.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param num_coarse: maximum number of oscillators to consider
    :returns: numpy array with downsampled data
    """
    return data[:, np.random.choice(np.arange(data.shape[0]), num_coarse)]


def compute_distances(data: np.ndarray) -> np.ndarray:
    """
    Compute cityblock distance between entry of data matrix.

    :param data: numpy array containing the data
    :returns: numpy array containing distance values
    """
    return pdist(data[:, np.newaxis], metric='cityblock')


def compute_maximal_distance(data: np.ndarray) -> float:
    """
    Compute the maximal distance contained in the data.

    :param data: array containing the data
    :returns: maximal distance
    """
    (num_time_steps, _) = data.shape
    max_distance = 0.0
    for time_step in tqdm(range(num_time_steps)):
        distances = compute_distances(data[time_step])
        if np.max(distances) > max_distance:
            max_distance = np.max(distances)
    return max_distance


def compute_normalized_distance_histograms(
    data: np.ndarray, max_distance: float, nbins: int
) -> np.ndarray:
    """
    Compute histograms of distance data.

    :param data: array containing the data
    :param max_distance: maximum distance
    :param nbins: number of histogram bins
    :returns: normalized histograms of distance data
    """
    num_time_steps, num_grid_points = data.shape

    histdat = np.zeros((num_time_steps, nbins))
    for time_step in tqdm(range(num_time_steps)):
        distances = compute_distances(data[time_step])
        histdat[time_step, :] = np.histogram(distances, nbins, range=(0, max_distance))[
            0
        ]

    normalization = float(num_grid_points * (num_grid_points - 1) / 2)
    return histdat / normalization


def compute_pearson_coefficient(data_x: np.ndarray, data_y: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation coefficient for real or complex data.

    :param data_x: real or complex time series
    :param data_y: real or complex time series
    :returns: real or complex correlation coefficient between the two time series
    """
    assert len(data_x) == len(data_y), 'Time series should be of same length.'

    coefficient = np.mean(
        np.conjugate(data_x - np.mean(data_x)) * (data_y - np.mean(data_y))
    ) / (np.std(np.conjugate(data_x)) * np.std(data_y))
    return coefficient


def compute_pairwise_correlation_coefficients(data: np.ndarray) -> np.ndarray:
    """
    Compute all pariwise correlation coefficients.

    :param data: array containing the data
    :returns: pairwise correlation coefficients
    """
    (_, num_grid_points) = data.shape

    pairwise_correlations = np.zeros(
        int(num_grid_points * (num_grid_points - 1) / 2), dtype='complex'
    )
    idx = 0
    print(
        'Calculating all pairwise correlation coefficients. This may take a few seconds.'
    )

    for space_x, space_y in tqdm(combinations(np.arange(num_grid_points), 2)):
        pairwise_correlations[idx] = compute_pearson_coefficient(
            data[:, space_x], data[:, space_y]
        )
        idx += 1

    return pairwise_correlations


def compute_normalized_correlation_histogram(
    pairwise_correlations: np.ndarray, nbins: int
) -> np.ndarray:
    """
    Compute histogram of all pariwise correlation coefficients.

    :param data: array containing the data
    :param nbins: number of histogram bins
    :returns: histogram of pairwise correlation coefficients
    """
    normalization = len(pairwise_correlations)
    histogram = np.histogram(np.abs(pairwise_correlations)/np.max(
        np.abs(pairwise_correlations)),
        bins=nbins, range=(0.0, 1.0 + np.finfo(float).eps)
    )[0]
    return histogram / normalization
