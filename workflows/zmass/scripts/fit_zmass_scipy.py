import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from utils import *


def relativistic_breit_wigner(x, resonance_mass, width, normalization):
    gamma = np.sqrt(resonance_mass ** 2 * (resonance_mass ** 2 + width ** 2))
    k = 2.0 * np.sqrt(2) * resonance_mass * width * gamma / (np.pi * np.sqrt(resonance_mass ** 2 + gamma))
    return normalization * k / ((x ** 2 - resonance_mass ** 2) ** 2 + resonance_mass ** 2 * width ** 2)


def fit(masses, p_start=[90, 10, 1000]):
    bin_contents, bin_edges = np.histogram(masses, bins=30, range=(60, 125))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    popt, pcov = curve_fit(
        f=relativistic_breit_wigner,
        xdata=bin_centers,
        ydata=bin_contents,
        p0=p_start,
        sigma=np.sqrt(bin_contents)
    )

    return popt



z0_masses_gsf = pd.read_csv(snakemake.input[0])
z0_masses_kf = pd.read_csv(snakemake.input[1])

gsf_optimized_pars = fit(z0_masses_gsf.mass_fit.to_numpy())
kf_optimized_pars = fit(z0_masses_kf.mass_fit.to_numpy(), p_start=[100, 10, 1000])
true_optimized_pars = fit(z0_masses_gsf.mass_true.to_numpy())

class F:
    def __init__(self, pars):
        self.pars = pars
    def __call__(self, x):
        return relativistic_breit_wigner(x, *self.pars) / self.pars[2]

plotter = MassSpectrumPlotter()
plotter.plot_zmass_spectrum(F(gsf_optimized_pars), z0_masses_gsf.mass_fit, gsf_optimized_pars[0], "GSF", "tab:orange")
plotter.plot_zmass_spectrum(F(kf_optimized_pars), z0_masses_kf.mass_fit, kf_optimized_pars[0], "KF", "tab:blue")
plotter.add_pdg_fit()

plotter.ax.legend()
plotter.fig.tight_layout()

plt.show()
plotter.fig.savefig(snakemake.output[0])
