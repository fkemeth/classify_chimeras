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

import sys
from time import time
import numpy as np

from tqdm.auto import tqdm

from classify_chimeras.utils import transform_phases, compute_curvature, \
    compute_maximal_curvature, compute_normalized_curvature_histograms, \
    coarse_grain_data, compute_distances, compute_normalized_distance_histograms


def spatial(data: np.ndarray,
            boundaries: str = "no-flux",
            phases: bool = False,
            nbins: int = 100) -> np.ndarray:
    """
    Classify systems with spatial extension using the discrete Laplacian.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param boundaries: boundary conditions, either periodic or no-flux
    :param phases: if the data consist of phases
    :param nbins: number of bins in the histogram
    :returns: numpy array with histogram data
    """
    assert len(data.shape) in [2, 3], "Please pass a TxN or TxN1xN2 numpy matrix."
    assert boundaries in ["no-flux", "periodic"], \
        "Please select proper boundary conditions: no-flux or periodic."

    print("Computing spatial coherence measure.")

    # If A contains only phases, map it onto the complex plane.
    if phases is True:
        data = transform_phases(data)

    # Create matrix with local curvatures
    curvature_data = compute_curvature(data)

    # Get maximal curvature
    max_curvature = compute_maximal_curvature(curvature_data, boundaries)

    # Check if there is incoherence at all.
    if max_curvature < 1e-9:
        raise ValueError(
            "Largest curvature smaller than 1e-9. System may just contain coherence."
        )

    # Compute the normalized histograms
    histograms = compute_normalized_curvature_histograms(curvature_data,
                                                         max_curvature,
                                                         nbins,
                                                         boundaries)

    # Return first bin of the normalized histograms
    return histograms[:, 0]


def globaldist(data: np.ndarray,
               nbins: int = 100,
               phases: bool = False,
               num_coarse: int = 1500) -> np.ndarray:
    """
    Classify coherence without a spatial extension using pairwise distances.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param nbins: number of bins in the histogram
    :param phases: if the data consist of phases
    :param num_coarse: maximum number of oscillators to consider
    :returns: numpy array with histogram data
    """
    assert len(data.shape) == 2, "Please pass a TxN numpy matrix."

    print("Computing spatial coherence measure.")

    if data.shape[1] > num_coarse:
        data = coarse_grain_data(data, num_coarse)

    # If A contains only phases, map it onto the complex plane.
    if phases is True:
        data = transform_phases(data)

    (num_time_steps, _) = data.shape

    # Get maximal distance
    print("Computing the maximal distance. This may take a few seconds.")
    max_distance = 0.0
    for time_step in tqdm(range(num_time_steps)):
        distances = compute_distances(data[time_step])
        if np.max(distances) > max_distance:
            max_distance = np.max(distances)

    # Compute the histograms
    print("Computing histograms. This may take a few seconds.")
    histograms = compute_normalized_distance_histograms(data, max_distance, nbins)
    return np.sqrt(histograms[:, 0])


def temporal(data: np.ndarray,
             nbins: int = 100,
             phases: bool = False,
             num_coarse: int = 1500) -> np.ndarray:
    """
    Calculate temporal correlation coefficients.

    :param data: numpy array containing the data with shape TxN or TxN1xN2
    :param nbins: number of bins in the histogram
    :param phases: if the data consist of phases
    :param num_coarse: maximum number of oscillators to consider
    :returns: numpy array with histogram data
    """

    print("Computing temporal coherence measure.")
    tstart = time()

    assert len(data.shape) in [2, 3], "Please pass a TxN or TxN1xN2 numpy matrix."

    if len(data.shape) == 2:
        (num_time_steps, num_grid_points) = data.shape
    elif len(data.shape) == 3:
        (num_time_steps, num_grid_points_x, num_grid_points_y) = data.shape
        data = np.reshape(data, (num_time_steps, num_grid_points_x * num_grid_points_y))
        num_grid_points = num_grid_points_x * num_grid_points_y

    while num_grid_points > num_coarse:
        print("Too many oscillators (N>1500). Taking half the data.")
        data = data[:, ::2]
        (num_time_steps, num_grid_points) = data.shape
    if phases:
        data = transform_phases(data)
    vacoma = np.zeros(int(num_grid_points * (num_grid_points - 1) / 2), dtype="complex")
    idx = 0
    print(
        "Calculating all pairwise correlation coefficients. This may take a few seconds."
    )
    for x_position in range(num_grid_points - 1):
        for y_position in range(x_position + 1, num_grid_points):
            vacoma[idx] = np.mean(
                np.conjugate(
                    data[:, x_position] - np.mean(data[:, x_position])) * (
                        data[:, y_position] - np.mean(data[:, y_position]))) / (
                            np.std(
                                np.conjugate(
                                    data[:, x_position])) * np.std(data[:, y_position]))
            idx += 1
            if (idx) % (
                    np.floor((num_grid_points * (num_grid_points - 1) / 2) / 100)) == 0:
                sys.stdout.write(
                    "\r %9.1f"
                    % round(
                        (time() - tstart) / (float(idx)) * (float(
                            num_grid_points * (num_grid_points - 1) / 2) - float(idx)),
                        1,
                    )
                    + " seconds left"
                )
                sys.stdout.flush()
    histdat = np.histogram(np.abs(vacoma), bins=nbins, range=(0.0, 1.01))[0] / float(
        num_grid_points * (num_grid_points - 1) / 2
    )
    print("\nDone!")
    return histdat
