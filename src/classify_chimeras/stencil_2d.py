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


Create 5pt or 9pt discrete Laplacian stencil.
"""

######################################################################################
#                                                                                    #
# Jun 2022                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from scipy.sparse import csr_matrix


def create_stencil(num_grid_point_x: int, num_grid_points_y: int, ninepoint: bool):
    """
    Create 2nd order finite difference stencil in two dimensions.

    :param num_grid_points_x: integer with the number of grid points along
                              the x axis
    :param num_grid_points_y: integer with the number of grid points along
                              the y axis
    :param ninepoint: if to use 9pt stencil instead of 5pt stencil
    :returns: finite difference stencil as csr matrix
    """
    stencil_data = np.empty(5 * num_grid_point_x * num_grid_points_y)
    stencil_data[0:num_grid_point_x * num_grid_points_y].fill(-4.0)
    idxs = np.empty(num_grid_point_x * num_grid_points_y)
    idxs[0:num_grid_point_x * num_grid_points_y] = np.arange(
        0, num_grid_point_x * num_grid_points_y
    )
    idxscol = np.empty(num_grid_point_x * num_grid_points_y)
    idxscol[0:num_grid_point_x * num_grid_points_y] = np.arange(
        0, num_grid_point_x * num_grid_points_y
    )
    stencil_data[
        num_grid_point_x * num_grid_points_y: 5 * num_grid_point_x * num_grid_points_y
    ].fill(1.0)
    idxscolright = np.empty(num_grid_point_x * num_grid_points_y)
    idxscolright = np.arange(0, num_grid_point_x * num_grid_points_y) + 1.0
    idxscolright[
        np.where(
            idxscolright / num_grid_points_y
            - np.trunc(idxscolright / num_grid_points_y)
            == 0
        )
    ] = (
        idxscolright[
            np.where(
                idxscolright / num_grid_points_y
                - np.trunc(idxscolright / num_grid_points_y)
                == 0
            )
        ]
        - num_grid_points_y
    )
    idxscolright[np.where(idxscolright == -num_grid_points_y)] = 0
    idxscolleft = np.empty(num_grid_point_x * num_grid_points_y)
    idxscolleft = np.arange(0, num_grid_point_x * num_grid_points_y) - 1
    idxscolleft[
        np.where(
            (idxscolleft + 1.0) / num_grid_points_y
            - np.trunc((idxscolleft + 1.0) / num_grid_points_y)
            == 0
        )
    ] = (
        idxscolleft[
            np.where(
                (idxscolleft + 1.0) / num_grid_points_y
                - np.trunc((idxscolleft + 1.0) / num_grid_points_y)
                == 0
            )
        ]
        + num_grid_points_y
    )
    idxscoltop = np.empty(num_grid_point_x * num_grid_points_y)
    idxscoltop = np.arange(0, num_grid_point_x * num_grid_points_y) - num_grid_points_y
    idxscoltop[np.where(idxscoltop < 0)] = (
        num_grid_point_x * num_grid_points_y + idxscoltop[np.where(idxscoltop < 0)]
    )
    idxscolbottom = np.empty(num_grid_point_x * num_grid_points_y)
    idxscolbottom = (
        np.arange(0, num_grid_point_x * num_grid_points_y) + num_grid_points_y
    )
    idxscolbottom[np.where(idxscolbottom >= num_grid_point_x * num_grid_points_y)] = (
        idxscolbottom[np.where(idxscolbottom >= num_grid_point_x * num_grid_points_y)]
        - num_grid_point_x * num_grid_points_y
    )
    idxtmp = idxs
    for _ in range(0, 4):
        idxs = np.append(idxs, idxtmp)
    idxscol = np.append(idxscol, idxscolright)
    idxscol = np.append(idxscol, idxscolleft)
    idxscol = np.append(idxscol, idxscoltop)
    idxscol = np.append(idxscol, idxscolbottom)
    if ninepoint:
        stencil_data = np.empty(9 * num_grid_point_x * num_grid_points_y)
        stencil_data[0: num_grid_point_x * num_grid_points_y].fill(-20.0)
        stencil_data[
            num_grid_point_x
            * num_grid_points_y: 5
            * num_grid_point_x
            * num_grid_points_y
        ].fill(4.0)
        stencil_data[
            5
            * num_grid_point_x
            * num_grid_points_y: 9
            * num_grid_point_x
            * num_grid_points_y
        ].fill(1.0)
        for _ in range(0, 4):
            idxs = np.append(idxs, idxtmp)
        idxscol9 = np.empty(num_grid_point_x * num_grid_points_y)
        idxscol9 = (
            np.arange(0, num_grid_point_x * num_grid_points_y) - num_grid_points_y - 1
        )  # left top
        idxscol9[np.where(idxscol9 < 0)] = (
            num_grid_point_x * num_grid_points_y + idxscol9[np.where(idxscol9 < 0)]
        )
        idxscol9[
            np.where(
                (idxscol9 + float(num_grid_points_y) + 1.0) / num_grid_points_y
                - np.trunc(
                    (idxscol9 + float(num_grid_points_y) + 1.0) / num_grid_points_y
                )
                == 0
            )
        ] = (
            idxscol9[
                np.where(
                    (idxscol9 + float(num_grid_points_y) + 1.0) / num_grid_points_y
                    - np.trunc(
                        (idxscol9 + float(num_grid_points_y) + 1.0) / num_grid_points_y
                    )
                    == 0
                )
            ]
            + num_grid_points_y
        )
        idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)] = (
            idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)]
            - num_grid_point_x * num_grid_points_y
        )
        idxscol9[0] = num_grid_point_x * num_grid_points_y - 1
        idxscol = np.append(idxscol, idxscol9)
        idxscol9 = (
            np.arange(0, num_grid_point_x * num_grid_points_y) - num_grid_points_y + 1
        )  # right top
        idxscol9[-1] = num_grid_point_x * num_grid_points_y - 2 * num_grid_points_y
        idxscol9[np.where(idxscol9 < 0)] = (
            num_grid_point_x * num_grid_points_y + idxscol9[np.where(idxscol9 < 0)]
        )
        idxscol9[
            np.where(
                (idxscol9 + float(num_grid_points_y)) / num_grid_points_y
                - np.trunc((idxscol9 + float(num_grid_points_y)) / num_grid_points_y)
                == 0
            )
        ] = (
            idxscol9[
                np.where(
                    (idxscol9 + float(num_grid_points_y)) / num_grid_points_y
                    - np.trunc(
                        (idxscol9 + float(num_grid_points_y)) / num_grid_points_y
                    )
                    == 0
                )
            ]
            - num_grid_points_y
        )
        idxscol9[np.where(idxscol9 < 0)] = (
            idxscol9[np.where(idxscol9 < 0)] + num_grid_point_x * num_grid_points_y
        )
        idxscol9[-1] = num_grid_point_x * num_grid_points_y - 2 * num_grid_points_y
        idxscol = np.append(idxscol, idxscol9)
        idxscol9 = (
            np.arange(0, num_grid_point_x * num_grid_points_y) + num_grid_points_y - 1
        )  # left bottom
        idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)] = (
            idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)]
            - num_grid_point_x * num_grid_points_y
        )
        idxscol9[
            np.where(
                (idxscol9 - float(num_grid_points_y) + 1.0) / num_grid_points_y
                - np.trunc(
                    (idxscol9 - float(num_grid_points_y) + 1.0) / num_grid_points_y
                )
                == 0
            )
        ] = (
            idxscol9[
                np.where(
                    (idxscol9 - float(num_grid_points_y) + 1.0) / num_grid_points_y
                    - np.trunc(
                        (idxscol9 - float(num_grid_points_y) + 1.0) / num_grid_points_y
                    )
                    == 0
                )
            ]
            + num_grid_points_y
        )
        idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)] = (
            idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)]
            - num_grid_point_x * num_grid_points_y
        )
        idxscol9[0] = 2 * num_grid_points_y - 1
        idxscol = np.append(idxscol, idxscol9)
        idxscol9 = (
            np.arange(0, num_grid_point_x * num_grid_points_y) + num_grid_points_y + 1
        )  # right bottom
        idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)] = (
            idxscol9[np.where(idxscol9 >= num_grid_point_x * num_grid_points_y)]
            - num_grid_point_x * num_grid_points_y
        )
        idxscol9[
            np.where(
                (idxscol9 - float(num_grid_points_y)) / num_grid_points_y
                - np.trunc((idxscol9 - float(num_grid_points_y)) / num_grid_points_y)
                == 0
            )
        ] = (
            idxscol9[
                np.where(
                    (idxscol9 - float(num_grid_points_y)) / num_grid_points_y
                    - np.trunc(
                        (idxscol9 - float(num_grid_points_y)) / num_grid_points_y
                    )
                    == 0
                )
            ]
            - num_grid_points_y
        )
        idxscol9[np.where(idxscol9 < 0)] = (
            idxscol9[np.where(idxscol9 < 0)] + num_grid_point_x * num_grid_points_y
        )
        idxscol9[-1] = 0
        idxscol = np.append(idxscol, idxscol9)
    stencil = csr_matrix(
        (stencil_data, (idxs, idxscol)),
        shape=(
            num_grid_point_x * num_grid_points_y,
            num_grid_point_x * num_grid_points_y,
        ),
        dtype=float,
    )
    if ninepoint:
        stencil = stencil / 6.0
    return stencil
