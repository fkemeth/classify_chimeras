"""
Copyright 2016 Felix P. Kemeth

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
# Jun 2016                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from scipy.sparse import csr_matrix


def create_stencil(N):
    """Create finite difference stencil."""
    stdata = np.empty(3 * N)
    stdata[0:N].fill(-2.0)
    idxs = np.empty(N)
    idxs[0:N] = np.arange(0, N)
    idxscol = np.empty(N)
    idxscol[0:N] = np.arange(0, N)
    idxscolrechts = np.empty(N)
    idxscolrechts = np.arange(0, N) + 1
    idxscolrechts[-1] = 0
    idxscollinks = np.empty(N)
    idxscollinks = np.arange(0, N) - 1
    idxscollinks[0] = N - 1
    stdata[N:3 * N].fill(1.0)
    idxtmp = idxs
    for x in range(0, 2):
        idxs = np.append(idxs, idxtmp)
    idxscol = np.append(idxscol, idxscolrechts)
    idxscol = np.append(idxscol, idxscollinks)
    stencil = csr_matrix((stdata, (idxs, idxscol)), shape=(N, N), dtype=float)
    return stencil
