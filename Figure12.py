import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from fisher_ingredients import *
from cosmology import quijote_cosmology as cosmo_dict
plt.style.use(['science.mplstyle', 'bright.mplstyle'])



def errors_smin(nderiv=1500, ncov=7000, smin=0, smax=150):
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    smax = 150
    ells = (0, 2)
    fisher_list = []
    smin_list = np.arange(10, 110, 10)

    # samples to include in the Fisher matrix
    samples = ["Om", "Ob2", "h", "s8", "ns", "Mmin", "Mnu"]

    ds_1 = []
    ds_5 = []
    ds_1245 = []
    tpcf = []
    for smin in smin_list:
        fisher = tpcf_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
            smin=smin, smax=smax, ells=ells,)
        cov_theta = np.linalg.inv(fisher)
        std_theta = np.sqrt(np.diag(cov_theta))
        tpcf.append(std_theta)

        fisher = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
            smin=smin, smax=smax, split='z', ells=ells, quantiles=[1],
            correlation_type='full')
        cov_theta = np.linalg.inv(fisher)
        std_theta = np.sqrt(np.diag(cov_theta))
        ds_1.append(std_theta)

        fisher = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
            smin=smin, smax=smax, split='z', ells=ells, quantiles=[5],
            correlation_type='full')
        cov_theta = np.linalg.inv(fisher)
        std_theta = np.sqrt(np.diag(cov_theta))
        ds_5.append(std_theta)

        fisher = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
            smin=smin, smax=smax, split='z', ells=ells, quantiles=[1, 2, 4, 5],
            correlation_type='full')
        cov_theta = np.linalg.inv(fisher)
        std_theta = np.sqrt(np.diag(cov_theta))
        ds_1245.append(std_theta)

    tpcf = np.asarray(tpcf)
    ds_1 = np.asarray(ds_1)
    ds_5 = np.asarray(ds_5)
    ds_1245 = np.asarray(ds_1245)

    for i, sample in enumerate(samples):
        fig, ax = plt.subplots(figsize=(4, 4))

        ax.plot(smin_list, np.log10(tpcf[:, i]),
            label=r'${\rm 2PCF}$',
            ls='-', lw=2.0, marker='o', ms=7.0)

        ax.plot(smin_list, np.log10(ds_1[:, i]),
            label=r'${\rm DS_{1}^{\rm qq+qh}}$',
            ls='-', lw=2.0, marker='^', ms=7.0)

        ax.plot(smin_list, np.log10(ds_5[:, i]),
            label=r'${\rm DS_{5}^{\rm qq+qh}}$',
            ls='-', lw=2.0, marker='v', ms=7.0)

        ax.plot(smin_list, np.log10(ds_1245[:, i]),
            label=r'${\rm DS_{1+2+4+5}^{\rm qq+qh}}$',
            ls='-', lw=2.0, marker='d', ms=7.0)

        label = cosmo_dict[sample]['label']
        ax.set_xlabel(r'$s_{\rm min}\,\left[h^{-1}{\rm Mpc}\right]$', fontsize=16)
        ax.set_ylabel(r'$\log_{{10}}\left[\sigma({})\right]$'.format(label), fontsize=16)

        if i == 0:
            ax.set_ylim(-2.4, -0.85)
            ax.legend(ncol=2, loc='lower right', columnspacing=0.7)
        plt.tight_layout()

        plt.savefig(f'paper_figures/pdf/errors_smin_{sample}.pdf')
        plt.savefig(f'paper_figures/png/errors_smin_{sample}.png', dpi=300)

if __name__ == '__main__':

    errors_smin()


