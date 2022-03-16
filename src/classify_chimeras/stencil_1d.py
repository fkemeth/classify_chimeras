"""
Copyright 2022 Felix P. Kemeth.

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


Create 1d discrete Laplacian stencil.
"""

######################################################################################
#                                                                                    #
# Jun 2022                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from scipy.sparse import csr_matrix


def create_stencil(num_grid_points: int) -> csr_matrix:
    """
    Create 2nd order finite difference stencil in one dimension.

    :param num_grid_points: integer with the number of grid points of
                            spatial axis
    :returns: finite difference stencil as csr matrix
    """
    stencil_data = np.empty(3 * num_grid_points)
    stencil_data[0:num_grid_points].fill(-2.0)
    idxs = np.empty(num_grid_points)
    idxs[0:num_grid_points] = np.arange(0, num_grid_points)
    idxscol = np.empty(num_grid_points)
    idxscol[0:num_grid_points] = np.arange(0, num_grid_points)
    idxscolright = np.empty(num_grid_points)
    idxscolright = np.arange(0, num_grid_points) + 1
    idxscolright[-1] = 0
    idxscolleft = np.empty(num_grid_points)
    idxscolleft = np.arange(0, num_grid_points) - 1
    idxscolleft[0] = num_grid_points - 1
    stencil_data[num_grid_points:3 * num_grid_points].fill(1.0)
    idxtmp = idxs
    for _ in range(0, 2):
        idxs = np.append(idxs, idxtmp)
    idxscol = np.append(idxscol, idxscolright)
    idxscol = np.append(idxscol, idxscolleft)
    stencil = csr_matrix(
        (stencil_data, (idxs, idxscol)),
        shape=(num_grid_points, num_grid_points),
        dtype=float,
    )
    return stencil
