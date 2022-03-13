REQUIREMENTS
---------

Following packages are required:

- numpy v1.11.0 or newer
- scipy v0.17 or newer

INSTALLATION
---------

Via pip:

`(sudo) pip install classify_chimeras`

Via source

(https://github.com/fkemeth/classify_chimeras)

DOCUMENTATION
---------

This python package contains functions to classify chimera states,
non-linear hybrid states of coexisting coherence and incoherence.
In partical, this package offers three functions, following the paper

"A classification scheme for chimera states"
(http://dx.doi.org/10.1063/1.4959804)

- `spatial(A, boundaries='no-flux', phases=False, nbins=100)`
`A` must be a TxN or a TxN1xN2 numpy matrix (either real or complex).
The function `spatial()` applies the discrete Laplacian on the data, and returns the coherent
fraction at each time step. `boundaries` specifies the boundary conditions under which the data was
generated. Set `phases=True` if A contains phases only. `nbins` specifies the number of bins of the histograms
which are generated.
- `globaldist(A, nbins=100, phases=False, Ncoarse=1500)`
`A` must be a TxN numpy matrix.
The function `globaldist()` calculates all pariwise Euclidean distances between all data points at
each time step, and returns
the coherent fraction of A at each time step.
`nbins` specifies the number of bins of the histograms.
Set `phases=True` if `A` contains phases only.
`Ncoarse` is a threshold above which the data is coarsed due to memory limitations. This can be increased,
but may lead to long calculation times or memory errors.
- `temporal(A, nbins=100, phases=False, Ncoarse=1500)`
A must be a TxN or TxN1xN2 numpy matrix.
The function `temporal()` calculates all pairwise temporal correlation coefficients between
the T-long timeseries of A. It returns a hisogram, with the square root of the last bin indicating the
amount of temporarily correlated time series.
`nbins` specifies the number of bins of the histograms.
Set `phases=True` if `A` contains phases only.
`Ncoarse` is a threshold above which the data is coarsed due to memory limitations. This can be increased,
but may lead to long calculation times or memory errors.

ISSUES
---------

For questions, please contact (<felix@kemeth.de>), or visit [the GitHub repo](https://github.com/fkemeth/classify_chimeras).

EXAMPLE
---------

As an illustrative example, we use a chimer state observed by Kuramoto and Battogtokh in
"Coexistence of Coherence and Incoherence in Nonlocally Coupled Phase Oscillators" (2002),
in Nonlinear Phenom. Complex Syst. We suppose
that we have the phases of this chimera state in a numpy matrix A.

    import matplotlib.pyplot as plt
    
    from kuramoto_chimera import integrate
    from classify_chimeras import spatial, temporal
    
    # Integrate Kuramoto phase oscillator system with nonlocal coupling.
    Ad = integrate()
    
    # Plot a snapshot of the data matrix A

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Ad["xx"], Ad["data"][-1])
    ax.set_xlabel('x')
    plt.show()

![Snapshot of the phases](/images/kuramoto.jpg)

    # Obtain the fraction of spatially coherent oscillators
    g_zero = spatial(Ad["data"], boundaries='periodic', phases=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Ad["t_eval"], g_zero)
    ax.set_xlabel('t')
    ax.set_ylim((0, 1.0))
    plt.show()

![Fraction of spatially coherent oscillators](/images/kuramoto_g0.jpg)

    # Obtain the fraction of temporarily correlated oscillators
    temporal_coherence = temporal(Ad["data"], phases=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(temporal_coherence)
    ax.set_ylim((0, 0.3))
    plt.show()

![Distribution of temporal correlation coefficients](/images/kuramoto_h.jpg)


LICENCE
---------


This work is licenced under GNU General Public License v3.
This means you must cite

"A classification scheme for chimera states"
F.P. Kemeth et al.
(http://dx.doi.org/10.1063/1.4959804)

if you use this package for publications.
