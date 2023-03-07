import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def add_core_to_df_quantile(df, key, quantile=0.95):
    d = np.quantile(np.sort(abs(df[key])), quantile)     
    core_range = (-d, d)
    
    df["is_core"] = df[key].between(*core_range)
    
    return df


def plot_core_tail_residuals(df, key, coor_unit=None, fig_axes=None):    
    # core_share = sum(df["is_core"]) / len(df)
    # tail_share = 1. - core_share
    
    core_part = df[df["is_core"]]
    tail_part = df[np.logical_not(df["is_core"])]
    
    if fig_axes is None:
        fig, axes = plt.subplots(1,2)
        axes = axes.flatten()
    else:
        fig, axes = fig_axes
     
    all_label  = "all: $\mu$={:.2f}, rms={:.2f}".format(np.mean(df[key]), rms(df[key]))
    tail_label = "tail: $\mu$={:.2f}, rms={:.2f}".format(np.mean(tail_part[key]), rms(tail_part[key]))
    core_label = "core: $\mu$={:.2f}, rms={:.2f}".format(np.mean(core_part[key]), rms(core_part[key]))
    hist_options = dict(edgecolor='black', linewidth=1.2, stacked=True)
        
    # Whole plot
    _, bins = np.histogram(df[key], bins="rice")
    _, bins, _ = axes[0].hist([core_part[key], tail_part[key]], bins=bins,  label=[core_label, tail_label], color=["tab:orange", "tab:blue"], histtype='stepfilled', **hist_options)
    axes[0].hist(df[key], bins=bins, histtype='step', label=all_label, **hist_options)
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].set_title("Core & Tail")
    
    # Core plot
    axes[1].hist(core_part[key], bins="rice", histtype='stepfilled', **hist_options)
    axes[1].set_yscale('log')
    axes[1].set_title("Core part")
    
    if not coor_unit is None:
        name, unit = coor_unit
        for ax in axes:
            ax.set_xlabel("${{{}}}_{{fit}} - {{{}}}_{{true}} \quad [{}]$".format(name, name, unit))
    
    return fig, axes


def plot_core_tail_pulls(df, key, coor_unit=None, fig_axes=None):
    if fig_axes is None:
        fig, axes = plt.subplots(1,2)
        axes = axes.flatten()
    else:
        fig, axes = fig_axes
        
    core_part = df[df["is_core"]]
    
    for ax, pull_df, title, color in zip(axes, [df, core_part], ["Pull Core+Tail", "Pull Core"], ["grey", "tab:orange"]):
        pull = pull_df[key]
        pull = pull[ np.isfinite(pull) ]
        mu, sigma = norm.fit(pull)
        
        ax.hist(np.clip(pull, -5*sigma, 5*sigma), bins="rice", density=True, color=color)
        
        x = np.linspace(*ax.get_xlim(), 200)
        ax.plot(x, norm.pdf(x, mu, sigma), label="$\mu$={:.2f}, $\sigma$={:.2f}".format(mu, sigma), color='red', lw=2.0)
        
        ax.legend()
        ax.set_title(title)
        
    if not coor_unit is None:
        name, unit = coor_unit
        for ax in axes:
            ax.set_xlabel("$({{{}}}_{{fit}} - {{{}}}_{{true}}) / \sigma_{{{},fit}} \quad [{}]$".format(name, name, name, unit))
        
    return fig, axes
