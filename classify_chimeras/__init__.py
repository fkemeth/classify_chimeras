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

Classify coherence/incoherence with the discrete Laplacian.

For systems without a spatial extension, use pairwise distances.
For temporal correlation, use correlation coefficients.
"""

###############################################################################
#                                                                             #
# http://dx.doi.org/10.1063/1.4959804                                         #
#                                                                             #
# Jun 2016                                                                    #
# felix@kemeth.de                                                             #
#                                                                             #
###############################################################################

import numpy as np
from time import time
import sys


def spatial(A, boundaries='no-flux', phases=False, nbins=100):
    """Classify systems with spatial extension using the discrete Laplacian."""
    print spatial.__doc__
    if np.size(A.shape) == 2:
        import stencil_1d as stl
        (T, N) = A.shape
        stencil = stl.create_stencil(N)
        dim = 1
    elif np.size(A.shape) == 3:
        import stencil_2d as stl
        (T, N1, N2) = A.shape
        A = np.reshape(A, (T, N1 * N2))
        stencil = stl.create_stencil(N1, N2, 1)
        dim = 2
    else:
        raise ValueError('Data matrix should either be TxN or TxN1xN2!')
    # If A contains only phases, map it onto the complex plane.
    if phases is True:
        A = np.exp(1.0j * A)
    # Create matrix with local curvatures
    AS = np.zeros_like(A, dtype='complex')
    for x in range(0, T):
        AS[x, :] = stencil.dot(A[x])
    AS = np.abs(AS)
    # Get maximal curvature
    if dim == 1:
        if boundaries == 'no-flux':
            Dmax = np.max(AS[:, 1:-1])
        elif boundaries == 'periodic':
            Dmax = np.max(AS)
        else:
            raise ValueError('Please select proper boundary conditions: no-flux or periodic.')
    if dim == 2:
        if boundaries == 'no-flux':
            Dmax = np.max(np.reshape(AS, (T, N1, N2))[:, 1:-1, 1:-1])
        elif boundaries == 'periodic':
            Dmax = np.max(AS)
        else:
            raise ValueError('Please select proper boundary conditions: no-flux or periodic.')
    # Check if there is incoherence at all.
    if Dmax < 1e-9:
        raise ValueError('Largest curvature smaller than 1e-9. System may just contain coherence.')
    # Compute the histograms
    histdat = np.zeros((T, nbins))
    if dim == 1:
        if boundaries == 'no-flux':
            for x in range(0, T):
                histdat[x, :] = np.histogram(AS[x, 1:-1], nbins, range=(0, Dmax)
                                             )[0] / float((N - 2))
        elif boundaries == 'periodic':
            for x in range(0, T):
                histdat[x, :] = np.histogram(AS[x], nbins, range=(0, Dmax)
                                             )[0] / float((N))
        else:
            raise ValueError('Please select proper boundary conditions: no-flux or periodic.')
    if dim == 2:
        if boundaries == 'no-flux':
            AS = np.reshape(AS, (T, N1, N2))
            for x in range(0, T):
                histdat[x, :] = np.histogram(AS[x, 1:-1, 1:-1], nbins, range=(0, Dmax)
                                             )[0] / float((N1 - 2) * (N2 - 2))
        elif boundaries == 'periodic':
            for x in range(0, T):
                histdat[x, :] = np.histogram(AS[x], nbins, range=(0, Dmax)
                                             )[0] / float((N1 * N2))
        else:
            raise ValueError('Please select proper boundary conditions: no-flux or periodic.')
    print '\nDone!'
    return histdat[:, 0]


def globaldist(A, nbins=100, phases=False, Ncoarse=1500):
    """Classify coherence without a spatial extension using pairwise distances."""
    print globaldist.__doc__
    tstart = time()
    try:
        (T, N) = A.shape
    except:
        raise ValueError('Please pass a TxN numpy matrix.')
    while N > Ncoarse:
        print "Too many oscillatrs (N>1000). Coarse grained data is used."
        A = A[:, ::2]
        (T, N) = A.shape
    if phases is True:
        A = np.exp(1.0j * A)

    # Get maximal distance
    print "Computing the maximal distance. This may take a few seconds."
    Dmax = 0.0
    for x in range(0, T):
        m, n = np.meshgrid(A[x, :], A[x, :])
        # get the distance via the norm
        out = abs(m - n)
        out = np.delete(out, np.diag_indices_from(out))
        if np.max(out) > Dmax:
            Dmax = np.max(out)
        if (x) % (np.floor((T) / 100)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(x + 1)) *
                                                (float(T) - float(x)), 1) + ' seconds left')
            sys.stdout.flush()
    print '\n'

    # Compute the histograms
    tstart = time()
    print "Computing histograms. This may take a few seconds."
    histdat = np.zeros((T, nbins))
    for x in range(0, T):
        m, n = np.meshgrid(A[x, :], A[x, :])
        # get the distance via the norm
        out = abs(m - n)
        # Remove diagonal entries
        out = np.delete(out, np.diag_indices_from(out))
        histdat[x, :] = np.histogram(out, nbins, range=(0, Dmax))[0] / float(N * (N - 1))
        if (x) % (np.floor((T) / 100)) == 0:
            sys.stdout.write("\r %9.1f" % round((time() - tstart) / (float(x + 1)) *
                                                (float(T) - float(x)), 1) + ' seconds left')
            sys.stdout.flush()
    print '\nDone!'
    return np.sqrt(histdat[:, 0])


def temporal(A, nbins=100, phases=False, Ncoarse=1500):
    """Calculate temporal correlation coefficients."""
    print temporal.__doc__
    tstart = time()
    try:
        Ashape = len(A.shape)
        if Ashape == 2:
            (T, N) = A.shape
        elif Ashape == 3:
            (T, N1, N2) = A.shape
            A = np.reshape(A, (T, N1 * N2))
            N = N1 * N2
        else:
            raise ValueError('Please pass a TxN or TxN1xN2 numpy matrix.')
    except:
        raise ValueError('Please pass a TxN or TxN1xN2 numpy matrix.')
    while N > Ncoarse:
        print "Too many oscillators (N>1500). Taking half the data."
        A = A[:, ::2]
        (T, N) = A.shape
    if phases:
        A = np.exp(1.0j * A)
    vacoma = np.zeros(N * (N - 1) / 2, dtype='complex')
    idx = 0
    print "Calculating all pairwise correlation coefficients. This may take a few seconds."
    for x in range(N - 1):
        for y in range(x + 1, N):
            vacoma[idx] = np.mean(np.conjugate(A[:, x] - np.mean(A[:, x])) * (
                A[:, y] - np.mean(A[:, y]))) / (np.std(np.conjugate(A[:, x])) * np.std(A[:, y]))
            idx += 1
            if (idx) % (np.floor((N * (N - 1) / 2) / 100)) == 0:
                sys.stdout.write(
                    "\r %9.1f" % round((time() - tstart) /
                                       (float(idx)) *
                                       (float(N * (N - 1) / 2) - float(idx)), 1) + ' seconds left')
                sys.stdout.flush()
    histdat = np.histogram(np.abs(vacoma), bins=100, range=(0.0, 1.01))[0] / float(N * (N - 1) / 2)
    print '\nDone!'
    return histdat
