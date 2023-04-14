import numpy as np
from contrast.box import covariance_matrix
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from getdist import plots, MCSamples
from fisher_ingredients import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['science.mplstyle', 'bright.mplstyle'])


def convergence_sigma_ncov():
    mpl.rcParams['xtick.labelsize'] = 15
    mpl.rcParams['ytick.labelsize'] = 15
    smin = 10
    smax = 150
    nderiv = 1500
    ells = (0, 2)
    correlation_type = 'full'
    quantiles = [1, 2, 4, 5]
    ncov_list = np.arange(1000, 7500, 500)
    samples = ["Om", "Ob2", "h", "s8", "ns", "Mnu"]

    fig, ax = plt.subplots()
    npar = len(samples)
    sigma = []

    for ncov in ncov_list:
        fisher_i = ds_fisher_matrix(samples=samples,  ncov=ncov,
            nderiv=nderiv, smin=smin, smax=smax, ells=ells,
            quantiles=quantiles, correlation_type=correlation_type)

        cov_i = np.linalg.inv(fisher_i)
        sigma_i = np.sqrt(np.diag(cov_i))
        sigma.append(sigma_i)
    sigma = np.asarray(sigma, dtype=float)

    for i, sample in enumerate(samples):
        label = cosmo_dict[sample]['label']
        ax.plot(ncov_list, sigma[:, i]/sigma[-1, i], label=r'${}$'.format(label), marker='o', ms=5.0)

    ax.plot(ncov_list, ncov_list/ncov_list[-1], color='dimgray', ls='--')
    ax.plot(ncov_list, np.ones(len(ncov_list)), ls='--', color='dimgray')
    ax.fill_between(ncov_list, 0.9, 1.1, color='grey', alpha=0.3)
    ax.set_xlabel(r'$N_{\rm cov}$', fontsize=17)
    ax.set_ylabel(r'$\sigma_\theta(N_{\rm cov})/\sigma_\theta(N_{\rm cov} = 7000)$', fontsize=17)
    ax.set_title(r'${\rm DS_{1+2+4+5}^{qq + qh}}$', fontsize=17)
    ax.legend()
    ax.set_xlim(1000, 7000)
    plt.tight_layout()

    plt.savefig('paper_figures/pdf/convergence_ncov_ds.pdf')
    plt.savefig('paper_figures/png/convergence_ncov_ds.png', dpi=300)

convergence_sigma_ncov()
