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

import matplotlib.pyplot as plt
from kuramoto_chimera import integrate

from classify_chimeras import spatial, temporal


def kuramoto_example() -> None:
    """
    Use a system of nonlocally coupled phase oscillators as an example.
    """
    # Integrate Kuramoto phase oscillator system with nonlocal coupling.
    data_dict = integrate()

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.scatter(data_dict['xx'], data_dict['data'][-1])
    axes.set_xlabel('x')
    plt.show()

    # Calculate g0
    g_zero = spatial(data_dict['data'], boundaries='periodic', phases=True)

    # Obtain the fraction of spatially coherent oscillators
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(data_dict['t_eval'], g_zero)
    axes.set_xlabel('t')
    axes.set_ylim((0, 1.0))
    plt.show()

    # Obtain the fraction of temporarily correlated oscillators
    temporal_coherence = temporal(data_dict['data'], phases=True)

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(temporal_coherence)
    axes.set_ylim((0, 0.3))
    plt.show()


if __name__ == '__main__':
    kuramoto_example()
