import pandas as pd
import awkward as ak
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pathlib import Path

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

particles = pd.read_csv(snakemake.input[0])

def get_time_per_particle(f):
    df = pd.read_csv(f, sep="\t")
    ms = 1000.0 * df[ df.identifier == "Algorithm:TrackFittingAlgorithm" ].iloc[0].time_perevent_s
    return ms/len(particles)

time_kf = get_time_per_particle(snakemake.input[1])
times_gsf = []
components = []

for f in snakemake.input[2:]:
    path = Path(f)
    components.append(int(str(Path(f).parent)[4:]))
    times_gsf.append(get_time_per_particle(f))

fn = lambda x, a, b, c: a*x**2 + b*x + c
popt, _ = curve_fit(fn, components, times_gsf)

x = np.linspace(min(components)-0.5, max(components)+0.5, 200)

print(popt)

fig, ax = plt.subplots()

ax.hlines(xmin=1, xmax=32, y=time_kf, label="KF", color="tab:blue", ls="--")
ax.scatter(components, times_gsf, color="tab:orange", label="GSF", marker="x", s=100)
ax.plot(x, fn(x, *popt), alpha=0.5, color="tab:orange", ls="--", label="quadratic fit to GSF")
ax.set_xticks(components)
ax.set_yscale('log')
ax.set_ylim(0.3,19)
ax.set_xlim(0,33)
ax.set_yticks([0.3,1,3,10])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel("# components")
ax.set_ylabel("time per track [ms]")
ax.set_title("Timing comparison between KF and GSF")
ax.legend()

fig.tight_layout()
fig.savefig(snakemake.output[0])
plt.show()

