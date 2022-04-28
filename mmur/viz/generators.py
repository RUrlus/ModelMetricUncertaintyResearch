import math
import numpy as np
from scipy.special import expit, logit
import matplotlib.pyplot as plt

from mmur.viz import _set_plot_style

COLORS = _set_plot_style()


def plot_logstic_dgp(N=500, figsize=None):
    """Plot example of DGP as used in mmur.generators.LogisticGenerator.

    Parameters
    ----------
    N : int
        number of points to generate in plot
    figsize : tuple, default=None
        figure passed to plt.subplots, default size is (12, 7)

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes._subplots.AxesSubplot

    """
    betas = np.array((0.5, 1.2))
    X = np.ones((N, 2))
    X[:, 1] = np.random.uniform(-10., 10.1, size=N)
    L = X.dot(betas)
    gt_proba = expit(L)
    proba_noisy = expit(L + np.random.normal(0, 0.5, size=N))
    y = np.random.binomial(1, proba_noisy)

    figsize = figsize or (12, 7)
    fig, ax = plt.subplots(figsize=figsize)
    sidx = np.argsort(X[:, 1])
    x = X[sidx, 1]
    ax.plot(x, gt_proba[sidx], label='true P', lw=2)
    ax.scatter(x, proba_noisy[sidx], c='grey', marker='x', label='noisy P')
    ax.scatter(x, y[sidx], c=COLORS[2], marker='x', s=50, label='y')
    ax.legend(fontsize=14)
    ax.set_ylabel('probability', fontsize=14)
    ax.set_xlabel('X', fontsize=14)
    ax.set_title('Logistic data generating process', fontsize=16)
    return fig, ax


def plot_probas(
    probas, ground_truth, n_sets=None, alt_label=None, axs=None
):
    """Plot sorted probabilities compared to ground truth probability.

    Parameters
    ---------
    probas : np.ndarray[float]
        the classifier probabilities of shape (holdout_samples, n_sets)
    ground_truth : np.ndarray[float]
        ground truth probabilities, 1d array
    n_sets : int, float, default=None
        number of columns in proba to plot. If int it is interpreted as the
        number of columns. If a float as a fraction of the columns. Default
        is max(0.1 * probas.shape[1], 30)
    alt_label : str, default=None
        label for the source of probabilities, default is 'holdout'
    axs : np.ndarray[matplotlib.axes._subplots.AxesSubplot], default=None
        an array containing the axes to plot on, must be 1d and of length >= 2

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        the figure is returned when ``axs`` is None
    axs : matplotlib.axes._subplots.AxesSubplot
        the created or passed axes object

    """
    if probas.ndim == 1:
        probas = probas[:, None]

    alt_label = alt_label or 'holdout'
    if axs is None:
        fig, axs = plt.subplots(figsize=(14, 7), nrows=1, ncols=2)
    else:
        fig = None

    n_cols = probas.shape[1]
    if isinstance(n_sets, int):
        n_sets = max(n_cols, n_sets)
    elif isinstance(n_sets, float):
        n_sets = max(math.floor(n_sets * n_cols), n_cols)
    else:
        n_sets = max(math.floor(0.1 * probas.shape[1]), min(30, n_cols))

    sorted_gt = np.sort(ground_truth)
    xvals = logit(sorted_gt)

    for i in range(n_sets - 1):
        sarr = np.sort(probas[:, i])
        axs[0].plot(xvals, sarr, c='grey', alpha=0.5)
        axs[1].plot(sorted_gt, sarr, c='grey', alpha=0.5)

    # plot outside loop for easier labelling
    sarr = np.sort(probas[:, -1])
    axs[0].plot(xvals, sarr, c='grey', alpha=0.5, label=alt_label)
    axs[1].plot(sorted_gt, sarr, c='grey', alpha=0.5, label=alt_label)

    # plot DGP
    axs[0].plot(
        xvals,
        sorted_gt,
        c='red',
        ls='--',
        lw=2,
        zorder=10,
        label='DGP',
    )
    axs[0].set_title('Probabilities', fontsize=18)
    axs[0].set_ylabel('proba', fontsize=18)
    axs[0].set_xlabel('DGP linear estimate', fontsize=18)
    axs[0].tick_params(labelsize=16)
    axs[0].legend(fontsize=18)

    # plot DGP
    axs[1].plot(
        ground_truth,
        ground_truth,
        c='red',
        ls='--',
        lw=2,
        zorder=10,
        label='DGP'
    )
    axs[1].set_title('Q-Q ', fontsize=18)
    axs[1].set_ylabel('proba -- ground truth', fontsize=18)
    axs[1].set_xlabel('proba -- draws', fontsize=18)
    axs[1].tick_params(labelsize=16)
    axs[1].legend(fontsize=18)

    if fig is not None:
        fig.tight_layout()
        return fig, axs
    return axs
