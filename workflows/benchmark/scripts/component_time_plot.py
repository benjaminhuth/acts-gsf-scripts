import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

from utils import get_count_dict, group_dict, color_dict


component_count_dicts = {}
count_dict_keys = None

for f in snakemake.input:
    components = int(str(Path(f).parent)[4:])
    count_dict, _ = get_count_dict(f)
    
    count_dict_keys = count_dict.keys()
    component_count_dicts[components] = count_dict
    
    
fig, ax = plt.subplots()

cmps = sorted(component_count_dicts.keys())

total = {c : 0 for c in cmps}

print(count_dict_keys)

for key, group in group_dict.items():
    samples = np.zeros(len(cmps))
    
    for subkey in group:
        for i, c in enumerate(cmps):
            v = component_count_dicts[c][subkey]
            samples[i] += v
            total[c] += v
            
    if "reduction" in key:
        f = lambda x, a, b, c: a*x**2 + b*x + c
    else:
        f = lambda x, m, t: m*x + t
    
    popt, _ = curve_fit(f, cmps, samples)
    
    print(key, samples)
    
    x = np.linspace(min(cmps)-0.5, max(cmps)+0.5)
    
    ax.plot(x, f(x, *popt), alpha=0.5, color=color_dict[key])
    ax.scatter(cmps, samples, label=key, color=color_dict[key], marker="x")
    
ax.yaxis.set_major_locator(plt.NullLocator())
ax.set_ylabel("samples [a. u.]")
ax.set_xlabel("# components")
ax.set_xticks(cmps)

ax.set_title("Scaling behaviour for parts of GSF")
    
    
#ax.scatter(cmps, total.values(), label="total")
    
ax.legend()
plt.show()

fig.savefig(snakemake.output[0])
    





