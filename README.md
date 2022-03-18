INSTALLATION
---------

Via pip:

`pip install classify_chimeras`

Via source

    git clone https://github.com/fkemeth/classify_chimeras
    cd classify_chimeras
    pip install .

DOCUMENTATION
---------

This python package contains functions to classify chimera states,
non-linear hybrid states of coexisting coherence and incoherence.
In particular, this package offers three functions, following the paper

"A classification scheme for chimera states"
(http://dx.doi.org/10.1063/1.4959804)

- `spatial(data, boundaries='no-flux', phases=False, nbins=100)`
`data` must be a TxN or a TxN1xN2 numpy matrix (either real or complex).
The function `spatial()` applies the discrete Laplacian on the data, and returns the coherent
fraction at each time step. `boundaries` specifies the boundary conditions under which the data was
generated. Set `phases=True` if A contains phases only. `nbins` specifies the number of bins of the histograms
which are generated.
- `globaldist(data, nbins=100, phases=False, num_coarse=1500)`
`data` must be a TxN numpy matrix.
The function `globaldist()` calculates all pariwise Euclidean distances between all data points at
each time step, and returns
the coherent fraction of A at each time step.
`nbins` specifies the number of bins of the histograms.
Set `phases=True` if `data` contains phases only.
`num_coarse` is a threshold above which the data is coarsed due to memory limitations. This can be increased,
but may lead to long calculation times or memory errors.
- `temporal(data, nbins=100, phases=False, num_coarse=1500)`
A must be a TxN or TxN1xN2 numpy matrix.
The function `temporal()` calculates all pairwise temporal correlation coefficients between
the T-long timeseries of A. It returns a hisogram, with the square root of the last bin indicating the
amount of temporarily correlated time series.
`nbins` specifies the number of bins of the histograms.
Set `phases=True` if `data` contains phases only.
`num_coarse` is a threshold above which the data is coarsed due to memory limitations. This can be increased,
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
    data_dict = integrate()

    # Plot a snapshot of the data matrix A

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_dict["xx"], data_dict["data"][-1])
    ax.set_xlabel('x')
    plt.show()

![Snapshot of the phases](/images/kuramoto.jpg)

    # Obtain the fraction of spatially coherent oscillators
    g_zero = spatial(data_dict["data"], boundaries='periodic', phases=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data_dict["t_eval"], g_zero)
    ax.set_xlabel('t')
    ax.set_ylim((0, 1.0))
    plt.show()

![Fraction of spatially coherent oscillators](/images/kuramoto_g0.jpg)

    # Obtain the fraction of temporarily correlated oscillators
    temporal_coherence = temporal(data_dict["data"], phases=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(temporal_coherence)
    ax.set_ylim((0, 0.3))
    plt.show()

![Distribution of temporal correlation coefficients](/images/kuramoto_h.jpg)


Changelog v.2.0.0
---------

- Refactored code for the correlation measures.
- Restructured code to confirm to pypi package layout.
- Use random subset of grid points when coarse graining data.
- Adjusted upper bound in temporal correlation histogram to 1+epsilon.
- Included example using kuramoto_chimera package.
- Added notebook example.
- Added unit tests.


LICENCE
---------


This work is licenced under GNU General Public License v3.
Please cite

"A classification scheme for chimera states"
F.P. Kemeth et al.
(http://dx.doi.org/10.1063/1.4959804)

if you use this package for publications.
