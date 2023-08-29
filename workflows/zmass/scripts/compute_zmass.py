import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import acts
import uproot
import ROOT

from gsfanalysis.pandas_import import uproot_to_pandas

from utils import preprocess_tracksummary

u = acts.UnitConstants
m_e = 0.51099895000 * u.MeV


def make_p_and_E(qop, theta, phi, m):
    p = abs(1./qop)
    pT = p*np.sin(theta)
    pz = p*np.cos(theta)

    px = pT*np.cos(phi)
    py = pT*np.sin(phi)

    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    return np.array([px, py, pz]), E


def compute_mass_fit(x):
    qop_a, theta_a, phi_a = x[:3]
    qop_b, theta_b, phi_b = x[3:]

    p_a, E_a = make_p_and_E(qop_a, theta_a, phi_a, m_e)
    p_b, E_b = make_p_and_E(qop_b, theta_b, phi_b, m_e)

    E_a = np.sqrt(p_a.dot(p_a) + m_e**2)
    E_b = np.sqrt(p_b.dot(p_b) + m_e**2)

    p = p_a + p_b
    E = E_a + E_b

    return np.sqrt(E**2 - p.dot(p))


def compute_mass_true(df):
    assert df.shape[0] == 2

    keys = ["t_px", "t_py", "t_pz"]

    p_a = df[keys].to_numpy()[0]
    p_b = df[keys].to_numpy()[1]

    E_a = np.sqrt(p_a.dot(p_a) + m_e**2)
    E_b = np.sqrt(p_b.dot(p_b) + m_e**2)

    E = E_a + E_b
    p = p_a + p_b

    return np.sqrt(E**2 - p.dot(p))


def compute_mass_fitted_and_true(df):
    assert df.shape[0] == 2

    keys = ['eQOP_fit', 'eTHETA_fit', 'ePHI_fit',]
    #err_keys = ['err_eQOP_fit','err_eTHETA_fit', 'err_ePHI_fit']
    err_keys = ['res_eQOP_fit','res_eTHETA_fit', 'res_ePHI_fit']

    x = np.concatenate([df[keys].to_numpy()[0], df[keys].to_numpy()[1]])
    dx = abs(np.concatenate([df[err_keys].to_numpy()[0], df[err_keys].to_numpy()[1]]))

    mass = compute_mass_fit(x)
    #mass_err = np.sqrt(np.sum(np.power(self.jac(x)*dx,2)))
    mass_true = compute_mass_true(df)

    return pd.DataFrame({
        "mass_fit": [mass],
        #"mass_err": [mass_err],
        "mass_true": [mass_true]
    })


def compute_masses_and_save(df : pd.DataFrame, outputFile):
    z0_masses = df.groupby("event_nr").apply(compute_mass_fitted_and_true)
    z0_masses = z0_masses[ ~np.isnan(z0_masses) ]
    z0_masses["mass_res"] = z0_masses.mass_fit - z0_masses.mass_true
    z0_masses.to_csv(outputFile)

tracksummary_gsf = uproot_to_pandas(uproot.open(snakemake.input[0] + ":tracksummary"))
tracksummary_kf = uproot_to_pandas(uproot.open(snakemake.input[1] + ":tracksummary"))

tracksummary_gsf, tracksummary_kf = preprocess_tracksummary(tracksummary_gsf, tracksummary_kf)

compute_masses_and_save(tracksummary_gsf, snakemake.output[0])
compute_masses_and_save(tracksummary_kf, snakemake.output[1])

