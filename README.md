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
