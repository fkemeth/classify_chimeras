"""Create 5pt or 9pt discrete Laplacianstencil."""

######################################################################################
#                                                                                    #
# Jun 2016                                                                           #
# felix@kemeth.de                                                                    #
#                                                                                    #
######################################################################################

import numpy as np
from scipy.sparse import csr_matrix


def create_stencil(N1, N2, ninepoint):
    """Create 5 or 9pt stencil."""
    stdata = np.empty(5 * N1 * N2)
    stdata[0:N1 * N2].fill(-4.0)
    idxs = np.empty(N1 * N2)
    idxs[0:N1 * N2] = np.arange(0, N1 * N2)
    idxscol = np.empty(N1 * N2)
    idxscol[0:N1 * N2] = np.arange(0, N1 * N2)
    stdata[N1 * N2:5 * N1 * N2].fill(1.0)
    idxscolrechts = np.empty(N1 * N2)
    idxscolrechts = np.arange(0, N1 * N2) + 1.0
    idxscolrechts[np.where(idxscolrechts / N2 - np.trunc(idxscolrechts / N2) == 0)] = idxscolrechts[
        np.where(idxscolrechts / N2 - np.trunc(idxscolrechts / N2) == 0)] - N2
    idxscolrechts[np.where(idxscolrechts == -N2)] = 0
    idxscollinks = np.empty(N1 * N2)
    idxscollinks = np.arange(0, N1 * N2) - 1
    idxscollinks[np.where((idxscollinks + 1.0) / N2 - np.trunc((idxscollinks + 1.0) / N2) == 0)] = \
        idxscollinks[np.where((idxscollinks + 1.0) / N2 -
                              np.trunc((idxscollinks + 1.0) / N2) == 0)] + N2
    idxscoloben = np.empty(N1 * N2)
    idxscoloben = np.arange(0, N1 * N2) - N2
    idxscoloben[np.where(idxscoloben < 0)] = N1 * N2 + idxscoloben[np.where(idxscoloben < 0)]
    idxscolunten = np.empty(N1 * N2)
    idxscolunten = np.arange(0, N1 * N2) + N2
    idxscolunten[np.where(idxscolunten >= N1 * N2)
                 ] = idxscolunten[np.where(idxscolunten >= N1 * N2)] - N1 * N2
    idxtmp = idxs
    for x in range(0, 4):
        idxs = np.append(idxs, idxtmp)
    idxscol = np.append(idxscol, idxscolrechts)
    idxscol = np.append(idxscol, idxscollinks)
    idxscol = np.append(idxscol, idxscoloben)
    idxscol = np.append(idxscol, idxscolunten)
    if ninepoint:
        stdata = np.empty(9 * N1 * N2)
        stdata[0:N1 * N2].fill(-20.0)
        stdata[N1 * N2:5 * N1 * N2].fill(4.0)
        stdata[5 * N1 * N2:9 * N1 * N2].fill(1.0)
        for x in range(0, 4):
            idxs = np.append(idxs, idxtmp)
        idxscol9 = np.empty(N1 * N2)
        idxscol9 = np.arange(0, N1 * N2) - N2 - 1  # links oben
        idxscol9[np.where(idxscol9 < 0)] = N1 * N2 + idxscol9[np.where(idxscol9 < 0)]
        idxscol9[np.where((idxscol9 + float(N2) + 1.0) / N2 -
                          np.trunc((idxscol9 + float(N2) + 1.0) / N2) == 0)] = \
            idxscol9[np.where((idxscol9 + float(N2) + 1.0) / N2 -
                              np.trunc((idxscol9 +
                                        float(N2) + 1.0) / N2) == 0)] + N2
        idxscol9[np.where(idxscol9 >= N1 * N2)] = idxscol9[np.where(idxscol9 >= N1 * N2)] - N1 * N2
        idxscol9[0] = N1 * N2 - 1
        idxscol = np.append(idxscol, idxscol9)
        idxscol9 = np.arange(0, N1 * N2) - N2 + 1  # rechts oben
        idxscol9[-1] = N1 * N2 - 2 * N2
        idxscol9[np.where(idxscol9 < 0)] = N1 * N2 + idxscol9[np.where(idxscol9 < 0)]
        idxscol9[np.where((idxscol9 +
                           float(N2)) / N2 - np.trunc((idxscol9 +
                                                       float(N2)) / N2) == 0)] = idxscol9[
            np.where((idxscol9 + float(N2)) / N2 - np.trunc((idxscol9 + float(N2)) / N2) == 0)] - N2
        idxscol9[np.where(idxscol9 < 0)] = idxscol9[np.where(idxscol9 < 0)] + N1 * N2
        idxscol9[-1] = N1 * N2 - 2 * N2
        idxscol = np.append(idxscol, idxscol9)
        idxscol9 = np.arange(0, N1 * N2) + N2 - 1  # links unten
        idxscol9[np.where(idxscol9 >= N1 * N2)] = idxscol9[np.where(idxscol9 >= N1 * N2)] - N1 * N2
        idxscol9[np.where((idxscol9 -
                           float(N2) + 1.0) / N2 - np.trunc((idxscol9 -
                                                             float(N2) + 1.0) / N2) == 0)] = \
            idxscol9[
            np.where((idxscol9 - float(N2) + 1.0) / N2 -
                     np.trunc((idxscol9 - float(N2) + 1.0) / N2) == 0)] + N2
        idxscol9[np.where(idxscol9 >= N1 * N2)] = idxscol9[np.where(idxscol9 >= N1 * N2)] - N1 * N2
        idxscol9[0] = 2 * N2 - 1
        idxscol = np.append(idxscol, idxscol9)
        idxscol9 = np.arange(0, N1 * N2) + N2 + 1  # rechts unten
        idxscol9[np.where(idxscol9 >= N1 * N2)] = idxscol9[np.where(idxscol9 >= N1 * N2)] - N1 * N2
        idxscol9[np.where((idxscol9 - float(N2)) / N2 -
                          np.trunc((idxscol9 - float(N2)) / N2) == 0)] = idxscol9[
            np.where((idxscol9 - float(N2)) / N2 - np.trunc((idxscol9 - float(N2)) / N2) == 0)] - N2
        idxscol9[np.where(idxscol9 < 0)] = idxscol9[np.where(idxscol9 < 0)] + N1 * N2
        idxscol9[-1] = 0
        idxscol = np.append(idxscol, idxscol9)
    stencil = csr_matrix((stdata, (idxs, idxscol)), shape=(N1 * N2, N1 * N2), dtype=float)
    if ninepoint:
        stencil = stencil / 6.0
    return stencil
