import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
import sys
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from fisher_ingredients import *
plt.style.use(['science.mplstyle', 'bright.mplstyle'])


def plot_derivatives():
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13

    smin = 10
    quantiles = [1, 2, 3, 4, 5]

    samples = ['Om']
    nmocks = 1500

    for correlation_type in ['cross', 'auto']:
        for i, sample in enumerate(samples):
            fig, ax = plt.subplots(2, 2, sharex='all', sharey='row')
            pm_values = cosmo_dict[sample]['pm']
            delta = pm_values[0] - pm_values[1]
            label = cosmo_dict[sample]['label']

            # plot the z-split derivatives in units of the standard error
            r_c, xi0_fid = ds_fiducial_multipoles(nmocks=nmocks, sample='fiducial',
                ells=(0,), split='z', correlation_type=correlation_type, smin=smin,
                return_sep=True)
            r_c, xi2_fid = ds_fiducial_multipoles(nmocks=nmocks, sample='fiducial',
                ells=(2,), split='z', correlation_type=correlation_type, smin=smin,
                return_sep=True)

            for j, ds in enumerate(quantiles):
                xi0_cov = ds_covariance(nmocks=7000, ells=(0,), split='z',
                    correlation_type=correlation_type, smin=smin, quantiles=[ds]) 
                xi2_cov = ds_covariance(nmocks=7000, ells=(2,), split='z',
                    correlation_type=correlation_type, smin=smin, quantiles=[ds]) 
                xi0_std = np.sqrt(np.diag(xi0_cov))
                xi2_std = np.sqrt(np.diag(xi2_cov))

                r_c, dxi_xi0 = ds_numerical_derivatives(sample, nmocks=nmocks, ells=(0,),
                    smin=smin, estimator=3, split='z', return_sep=True, quantiles=[ds],
                    correlation_type=correlation_type)
                r_c, dxi_xi2 = ds_numerical_derivatives(sample, nmocks=nmocks, ells=(2,),
                    smin=smin, estimator=3, split='z', return_sep=True, quantiles=[ds],
                    correlation_type=correlation_type)

                fig.axes[0].plot(r_c, dxi_xi0/xi0_std, ls='-', label=rf'${{\rm DS}}_{ds}$')
                fig.axes[2].plot(r_c, dxi_xi2/xi2_std, ls='-', label=rf'${{\rm DS}}_{ds}$')


            # plot the r-split derivatives in units of the standard error
            r_c, xi0_fid = ds_fiducial_multipoles(nmocks=nmocks, sample='fiducial', ells=(0,),
                split='r', correlation_type=correlation_type, smin=smin, return_sep=True)
            r_c, xi2_fid = ds_fiducial_multipoles(nmocks=nmocks, sample='fiducial', ells=(2,),
                split='r', correlation_type=correlation_type, smin=smin, return_sep=True)

            for j, ds in enumerate(quantiles):
                xi0_cov = ds_covariance(nmocks=7000, ells=(0,), split='r',
                    correlation_type=correlation_type, smin=smin, quantiles=[ds]) 
                xi2_cov = ds_covariance(nmocks=7000, ells=(2,), split='r',
                    correlation_type=correlation_type, smin=smin, quantiles=[ds]) 
                xi0_std = np.sqrt(np.diag(xi0_cov))
                xi2_std = np.sqrt(np.diag(xi2_cov))

                r_c, dxi_xi0 = ds_numerical_derivatives(sample, nmocks=nmocks,
                    ells=(0,), smin=smin, estimator=3, split='r', return_sep=True,
                    correlation_type=correlation_type, quantiles=[ds])
                r_c, dxi_xi2 = ds_numerical_derivatives(sample, nmocks=nmocks,
                    ells=(2,), smin=smin, estimator=3, split='r', return_sep=True,
                    correlation_type=correlation_type, quantiles=[ds])

                fig.axes[1].plot(r_c, dxi_xi0/xi0_std, ls='-',
                    label=rf'${{\rm DS}}_{ds}$')
                fig.axes[3].plot(r_c, dxi_xi2/xi2_std, ls='-',
                    label=rf'${{\rm DS}}_{ds}$')

            if correlation_type == 'auto':
                fig.axes[0].set_ylabel(r'$\partial \xi_0^{{\rm qq}}/\partial {} / \sigma_{{\xi_0^{{\rm qq}}}}$'.format(label), fontsize=15)
                fig.axes[2].set_ylabel(r'$\partial \xi_2^{{\rm qq}}/\partial {} / \sigma_{{\xi_2^{{\rm qq}}}}$'.format(label), fontsize=15)
            else:
                fig.axes[0].set_ylabel(r'$\partial \xi_0^{{\rm qh}}/\partial {} / \sigma_{{\xi_0^{{\rm qh}}}}$'.format(label), fontsize=15)
                fig.axes[2].set_ylabel(r'$\partial \xi_2^{{\rm qh}}/\partial {} / \sigma_{{\xi_2^{{\rm qh}}}}$'.format(label), fontsize=15)
                if sample == 'Mnu': fig.axes[2].set_ylim(-3, 3)
            fig.axes[2].set_xlabel(r'$s\,\left[h^{-1}{\rm Mpc}\right]$', fontsize=15)
            fig.axes[3].set_xlabel(r'$s\,\left[h^{-1}{\rm Mpc}\right]$', fontsize=15)
            leg = fig.axes[0].legend(loc='best', ncol=2, fontsize=11,
                handlelength=0, columnspacing=0.25)
            # change the font colors to match the line colors:
            for line,text in zip(leg.get_lines(), leg.get_texts()):
                text.set_color(line.get_color())

            fig.axes[0].set_title('$z$''-split', fontsize=15)
            fig.axes[1].set_title('$r$''-split', fontsize=15)

            plt.setp(fig.axes[0].get_xticklabels(), visible=False)
            plt.setp(fig.axes[1].get_xticklabels(), visible=False)
            plt.setp(fig.axes[1].get_yticklabels(), visible=False)
            fig.axes[0].set_xlim(10, 150)
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.05)
            plt.subplots_adjust(wspace=0.05)
            plt.savefig(f'paper_figures/png/derivatives_{correlation_type}_{sample}_nderiv{nmocks}.png', dpi=300)


if __name__ == '__main__':
    plot_derivatives()

