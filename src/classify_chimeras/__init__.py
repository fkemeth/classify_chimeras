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

from classify_chimeras.utils import (coarse_grain_data, compute_curvature,
                                     compute_maximal_curvature,
                                     compute_maximal_distance,
                                     compute_normalized_correlation_histogram,
                                     compute_normalized_curvature_histograms,
                                     compute_normalized_distance_histograms,
                                     compute_pairwise_correlation_coefficients,
                                     transform_phases)


def spatial(
    data: np.ndarray,
    boundaries: str = 'no-flux',
    phases: bool = False,
    nbins: int = 100,
) -> np.ndarray:
    """
    Classify systems with spatial extension using the discrete Laplacian.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param boundaries: boundary conditions, either periodic or no-flux
    :param phases: if the data consist of phases
    :param nbins: number of bins in the histogram
    :returns: numpy array with histogram data
    """
    assert len(data.shape) in [2, 3], 'Please pass a TxN or TxN1xN2 numpy matrix.'
    assert boundaries in [
        'no-flux',
        'periodic',
    ], 'Please select proper boundary conditions: no-flux or periodic.'

    print('Computing spatial coherence measure.')

    # If data contains only phases, map it onto the complex plane.
    if phases is True:
        data = transform_phases(data)

    # Create matrix with local curvatures
    curvature_data = compute_curvature(data)

    # Get maximal curvature
    max_curvature = compute_maximal_curvature(curvature_data, boundaries)

    # Check if there is incoherence at all.
    if max_curvature < 1e-9:
        raise ValueError(
            'Largest curvature smaller than 1e-9. System may just contain coherence.'
        )

    # Compute the normalized histograms
    histograms = compute_normalized_curvature_histograms(
        curvature_data, max_curvature, nbins, boundaries
    )

    # Return first bin of the normalized histograms
    return histograms[:, 0]


def globaldist(
    data: np.ndarray, nbins: int = 100, phases: bool = False, num_coarse: int = 1500
) -> np.ndarray:
    """
    Classify coherence without a spatial extension using pairwise distances.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param nbins: number of bins in the histogram
    :param phases: if the data consist of phases
    :param num_coarse: maximum number of oscillators to consider
    :returns: numpy array with histogram data
    """
    assert len(data.shape) == 2, 'Please pass a TxN numpy matrix.'

    print('Computing spatial coherence measure.')

    # Downsample data if it contains too many grid points
    if data.shape[1] > num_coarse:
        print(
            f"""Too many grid points ({data.shape[1]}).
        Using {num_coarse} grid points instead."""
        )
        data = coarse_grain_data(data, num_coarse)

    # If data contains only phases, map it onto the complex plane.
    if phases is True:
        data = transform_phases(data)

    # Get maximal distance
    print('Computing the maximal distance. This may take a few seconds.')
    max_distance = compute_maximal_distance(data)

    # Compute the histograms
    print('Computing histograms. This may take a few seconds.')
    histograms = compute_normalized_distance_histograms(data, max_distance, nbins)
    return np.sqrt(histograms[:, 0])


def temporal(
    data: np.ndarray, nbins: int = 100, phases: bool = False, num_coarse: int = 1500
) -> np.ndarray:
    """
    Calculate temporal correlation coefficients.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param nbins: number of bins in the histogram
    :param phases: if the data consist of phases
    :param num_coarse: maximum number of oscillators to consider
    :returns: numpy array with histogram data
    """

    print('Computing temporal coherence measure.')

    assert len(data.shape) in [2, 3], 'Please pass a TxN or TxN1xN2 numpy matrix.'

    # Flatten data if it has two spatial dimensions
    data = data.reshape((data.shape[0], -1))

    # Downsample data if it contains too many grid points
    if data.shape[1] > num_coarse:
        print(
            f"""Too many grid points ({data.shape[1]}).
        Using {num_coarse} grid points instead."""
        )
        data = coarse_grain_data(data, num_coarse)

    # If data contains only phases, map it onto the complex plane.
    if phases:
        data = transform_phases(data)

    # Compute all pariwise correlation coefficients
    pairwise_correlations = compute_pairwise_correlation_coefficients(data)

    # Compute normalized histogram of the correlation coefficients
    histogram = compute_normalized_correlation_histogram(pairwise_correlations, nbins)

    return histogram
