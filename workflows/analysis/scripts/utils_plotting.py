import numpy as np
from tqdm import tqdm

from scipy.stats import bootstrap, binned_statistic

def plot_binned_errorbar(ax, binning_data, stat_data, bins, stat, **kwargs):
    print("binned statistics...")
    vals, bins, bin_idxs = binned_statistic(binning_data, stat_data, stat, bins)

    errs = []
    for i in tqdm(range(len(bins[:-1])), desc="bootstrapping..."):
        data = stat_data[ bin_idxs == i].to_numpy()
        if len(data) == 0:
            errs.append(np.nan)
        else:
            errs.append(bootstrap((data,), stat, n_resamples=1000).standard_error)
            
    errs = np.array(errs)

    shift = 0.5*np.diff(bins)
    ax.errorbar(
        x=bins[:-1]+shift, 
        y=vals,
        xerr=shift,
        yerr=errs,
        **kwargs
    )
    
    return vals, errs, bins
