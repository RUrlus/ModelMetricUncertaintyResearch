import numpy as np
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from mmur.viz import _set_plot_style
from mmu.stats import compute_hdi

COLORS = _set_plot_style()


def plot_hdis_violin(
        hdi_estimates, holdout_metrics, prob=0.95, ax=None
    ):
    """Plot Highest Density Interval containing `prob` of the distribution
    against the HDI of the holdout_metrics.

    Parameters
    ----------
    hdi_estimates : pd.DataFrame
        dataframe containing the Highest Density Interval with probability equal
        to `prob`. The index of ``hdi_estimates`` should be equal to the columns
        of ``holdout_metrics``. The columns should contain at least: 'lb',
        'ub' and 'mu'.
    axs : nmatplotlib.axes._subplots.AxesSubplot, default=None
        axes object to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        the figure is returned when ``axs`` is None
    ax : matplotlib.axes._subplots.AxesSubplot
        the created or passed axes object

    """
    holdout_metrics_moments = pd.concat(
        (
            holdout_metrics.apply([np.min, np.max, np.mean]).T, # type: ignore
            compute_hdi(holdout_metrics, prob=prob)
        ), axis=1
    )
    target_metrics = holdout_metrics.columns.to_list()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig = None

    _ = sns.violinplot(
        data=holdout_metrics,
        saturation=0.1,
        ax=ax,
        color=COLORS[3],
        zorder=5,
        label='out-of-sample',
    )
    violin = mpatches.Patch(color=COLORS[3], label='out-of-sample')

    for i, idx in enumerate(hdi_estimates.index):
        mu = hdi_estimates.loc[idx, 'mu']
        lb = hdi_estimates.loc[idx, 'lb']
        ub = hdi_estimates.loc[idx, 'ub']
        err = np.abs(np.array([lb, ub])[:, None] - mu)
        ax.errorbar(
            x=i - 0.1, y=mu, yerr=err, capsize=10, fmt='none',
            color=COLORS[0], zorder=10, lw=2, label='HDI estimate'
        );
        ax.scatter(
            x=i - 0.1, y=mu, marker='d', s=100, color=COLORS[0],
            zorder=10, label='mean estimate'
        )

        mu = holdout_metrics_moments.loc[idx, 'mean']
        lb = holdout_metrics_moments.loc[idx, 'lb']
        ub = holdout_metrics_moments.loc[idx, 'ub']
        err = np.abs(np.array([lb, ub])[:, None] - mu)
        ax.errorbar(
            x=i + 0.1, y=mu, yerr=err, capsize=10,
            fmt='none', color=COLORS[1], label='HDI out-of-sample', zorder=10, lw=2
        );
        ax.scatter(
            x=i + 0.1, y=mu, marker='d', s=100, color=COLORS[1], zorder=10,
            label='mean out-of-sample'
        )

    _ = ax.set_xticks([i for i in range(len(target_metrics))])
    _ = ax.set_xticklabels(target_metrics)

    ax.set_title('Coverage metrics', fontsize=18)
    ax.set_ylabel('value', fontsize=18)
    ax.set_xlabel('metrics', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=18)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [violin, ] + handles
    labels = ['out-of-sample'] + labels
    by_label = dict(zip(labels, handles))
    _ = ax.legend(by_label.values(), by_label.keys())

    if fig is None:
        return ax
    return fig, ax


def plot_ci_violin(
        ci_estimates, holdout_metrics, alpha=0.95, ax=None
    ):
    """Plot confidence interval containing `prob` of the distribution
    against the HDI of the holdout_metrics.

    Parameters
    ----------
    ci_estimates : pd.DataFrame
        dataframe containing the ``alpha`` confidence interval. The index of
        ``ci_estimates`` should be equal to the columns
        of ``holdout_metrics``. The columns should contain at least: 'lb',
        'ub' and 'mu'.
    axs : nmatplotlib.axes._subplots.AxesSubplot, default=None
        axes object to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure
        the figure is returned when ``axs`` is None
    ax : matplotlib.axes._subplots.AxesSubplot
        the created or passed axes object

    """
    alpha_ = (1 - alpha) / 2
    q = (alpha_, 1 - alpha_)
    alpha_perc = round(alpha * 100, 2)

    holdout_metrics_moments = holdout_metrics.apply([np.mean, np.std]).T
    holdout_metrics_moments['lb'] = 0.0
    holdout_metrics_moments['ub'] = 0.0

    dist = sts.norm(
        holdout_metrics_moments.iloc[0, 0],
        holdout_metrics_moments.iloc[0, 1]
    )
    holdout_metrics_moments.iloc[0, 2:] = dist.ppf(q)
    dist = sts.norm(
        holdout_metrics_moments.iloc[1, 0],
        holdout_metrics_moments.iloc[1, 1]
    )
    holdout_metrics_moments.iloc[1, 2:] = dist.ppf(q)
    target_metrics = holdout_metrics.columns.to_list()

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    else:
        fig = None

    _ = sns.violinplot(
        data=holdout_metrics,
        saturation=0.1,
        ax=ax,
        color=COLORS[3],
        zorder=5,
        label='observed',
    )
    violin = mpatches.Patch(color=COLORS[3], label='observed')

    for i, idx in enumerate(ci_estimates.index):
        mu = ci_estimates.loc[idx, 'mu']
        lb = ci_estimates.loc[idx, 'lb']
        ub = ci_estimates.loc[idx, 'ub']
        err = np.abs(np.array([lb, ub])[:, None] - mu)
        ax.errorbar(
            x=i - 0.1, y=mu, yerr=err, capsize=10, fmt='none',
            color=COLORS[0], zorder=10, lw=2, label=f'{alpha_perc}% CI estimate'
        );
        ax.scatter(
            x=i - 0.1, y=mu, marker='d', s=100, color=COLORS[0],
            zorder=10, label='mean estimate'
        )

        mu = holdout_metrics_moments.loc[idx, 'mean']
        lb = holdout_metrics_moments.loc[idx, 'lb']
        ub = holdout_metrics_moments.loc[idx, 'ub']
        err = np.abs(np.array([lb, ub])[:, None] - mu)
        ax.errorbar(
            x=i + 0.1, y=mu, yerr=err, capsize=10,
            fmt='none', color=COLORS[1],
            label=f'{alpha_perc}% CI observed', zorder=10, lw=2
        );
        ax.scatter(
            x=i + 0.1, y=mu, marker='d', s=100, color=COLORS[1], zorder=10,
            label='mean observed'
        )

    _ = ax.set_xticks([i for i in range(len(target_metrics))])
    _ = ax.set_xticklabels(target_metrics)

    ax.set_title(f'estimated vs observed\n {alpha_perc}% Confidence intervals', fontsize=18)
    ax.set_ylabel('value', fontsize=18)
    ax.set_xlabel('metrics', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=18)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [violin, ] + handles
    labels = ['out-of-sample'] + labels
    by_label = dict(zip(labels, handles))
    _ = ax.legend(by_label.values(), by_label.keys())

    if fig is None:
        return ax
    return fig, ax
