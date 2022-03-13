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

from classify_chimeras.utils import transform_phases, calculate_curvature, \
    calculate_maximal_curvature


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

    if len(data.shape) == 2:
        (num_time_steps, num_grid_points) = data.shape
    else:
        (num_time_steps, num_grid_points_x, num_grid_points_y) = data.shape

    # If A contains only phases, map it onto the complex plane.
    if phases is True:
        data = transform_phases(data)

    # Create matrix with local curvatures
    curvature_data = calculate_curvature(data)

    # Get maximal curvature
    max_curvature = calculate_maximal_curvature(curvature_data, data.shape, boundaries)

    # Check if there is incoherence at all.
    if max_curvature < 1e-9:
        raise ValueError(
            "Largest curvature smaller than 1e-9. System may just contain coherence."
        )
    # Compute the histograms
    histdat = np.zeros((num_time_steps, nbins))
    if len(data.shape) == 2:
        if boundaries == "no-flux":
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step, 1:-1], nbins,
                    range=(0, max_curvature))[0] / float((num_grid_points - 2))
        elif boundaries == "periodic":
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step], nbins,
                    range=(0, max_curvature))[0] / float((num_grid_points))
        else:
            raise ValueError(
                "Please select proper boundary conditions: no-flux or periodic."
            )
    if len(data.shape) == 3:
        if boundaries == "no-flux":
            curvature_data = np.reshape(
                curvature_data,
                (num_time_steps, num_grid_points_x, num_grid_points_y))
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step, 1:-1, 1:-1], nbins,
                    range=(0, max_curvature))[0] / float(
                        (num_grid_points_x - 2) * (num_grid_points_y - 2))
        elif boundaries == "periodic":
            for time_step in range(0, num_time_steps):
                histdat[time_step, :] = np.histogram(
                    curvature_data[time_step], nbins,
                    range=(0, max_curvature))[0] / float(
                        (num_grid_points_x * num_grid_points_y))
        else:
            raise ValueError(
                "Please select proper boundary conditions: no-flux or periodic.")
    print("\nDone!")
    return histdat[:, 0]


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
    tstart = time()
    assert len(data.shape) == 2, "Please pass a TxN numpy matrix."

    (num_time_steps, num_grid_points) = data.shape
    while num_grid_points > num_coarse:
        print("Too many oscillatrs (N>1000). Coarse grained data is used.")
        data = data[:, ::2]
        (num_time_steps, num_grid_points) = data.shape
    if phases is True:
        data = transform_phases(data)

    # Get maximal distance
    print("Computing the maximal distance. This may take a few seconds.")
    max_curvature = 0.0
    for time_step in range(0, num_time_steps):
        mesh_x, mesh_y = np.meshgrid(data[time_step, :], data[time_step, :])
        # get the distance via the norm
        out = abs(mesh_x - mesh_y)
        out = np.delete(out, np.diag_indices_from(out))
        if np.max(out) > max_curvature:
            max_curvature = np.max(out)
        if (time_step) % (np.floor((num_time_steps) / 100)) == 0:
            sys.stdout.write(
                "\r %9.1f"
                % round((time() - tstart) / (float(time_step + 1)) * (
                    float(num_time_steps) - float(time_step)), 1)
                + " seconds left"
            )
            sys.stdout.flush()
    print("\n")

    # Compute the histograms
    tstart = time()
    print("Computing histograms. This may take a few seconds.")
    histdat = np.zeros((num_time_steps, nbins))
    for time_step in range(0, num_time_steps):
        mesh_x, mesh_y = np.meshgrid(data[time_step, :], data[time_step, :])
        # get the distance via the norm
        out = abs(mesh_x - mesh_y)
        # Remove diagonal entries
        out = np.delete(out, np.diag_indices_from(out))
        histdat[time_step, :] = np.histogram(
            out, nbins, range=(0, max_curvature))[0] / float(
                num_grid_points * (num_grid_points - 1))
        if (time_step) % (np.floor((num_time_steps) / 100)) == 0:
            sys.stdout.write(
                "\r %9.1f"
                % round((time() - tstart) / (float(time_step + 1)) * (
                    float(num_time_steps) - float(time_step)), 1)
                + " seconds left"
            )
            sys.stdout.flush()
    print("\nDone!")
    return np.sqrt(histdat[:, 0])


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
    print(temporal.__doc__)
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
