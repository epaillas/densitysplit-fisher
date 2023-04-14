import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fisher_ingredients import *
plt.style.use(['science.mplstyle', 'bright.mplstyle'])


def plot_multipoles():
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']

    smin, smax = 10, 150
    nmocks = 1500
    quantiles = [1, 2, 3, 4, 5]

    for correlation_type in ['auto', 'cross']:
        fig, ax = plt.subplots(2, 2, sharex='all', sharey='row')

        r_c, xi0 = ds_fiducial_multipoles(nmocks=nmocks, ells=(0,), return_sep=True,
            split='z', correlation_type=correlation_type, smin=smin, smax=smax)
        r_c, xi2 = ds_fiducial_multipoles(nmocks=nmocks, ells=(2,), return_sep=True,
            split='z', correlation_type=correlation_type, smin=smin, smax=smax)

        xi0 = np.mean(xi0, axis=0)
        xi2 = np.mean(xi2, axis=0)

        for ids, ds in enumerate(quantiles):
            cov_xi0 = ds_covariance(nmocks=7000, ells=(0,), quantiles=[ds],
                split='z', correlation_type=correlation_type, smin=smin, smax=smax)
            cov_xi2 = ds_covariance(nmocks=7000, ells=(2,), quantiles=[ds],
                split='z', correlation_type=correlation_type, smin=smin, smax=smax)
            std_xi0 = np.sqrt(np.diag(cov_xi0))
            std_xi2 = np.sqrt(np.diag(cov_xi2))
            fig.axes[0].plot(r_c, r_c**2 * xi0[ids], label=rf'${{\rm DS}}_{ds}$')
            fig.axes[2].plot(r_c, r_c**2 * xi2[ids], label=rf'${{\rm DS}}_{ds}$')
            fig.axes[0].fill_between(r_c, r_c**2*(xi0[ids] - std_xi0),
                r_c**2*(xi0[ids] + std_xi0), color=colors[ids], alpha=0.2)
            fig.axes[2].fill_between(r_c, r_c**2*(xi2[ids] - std_xi2),
                r_c**2*(xi2[ids] + std_xi2), color=colors[ids], alpha=0.2)

        r_c, xi0 = ds_fiducial_multipoles(nmocks=nmocks, ells=(0,), return_sep=True,
            split='r', correlation_type=correlation_type, smin=smin, smax=smax)
        r_c, xi2 = ds_fiducial_multipoles(nmocks=nmocks, ells=(2,), return_sep=True,
            split='r', correlation_type=correlation_type, smin=smin, smax=smax)

        xi0 = np.mean(xi0, axis=0)
        xi2 = np.mean(xi2, axis=0)

        for ids, ds in enumerate(quantiles):
            cov_xi0 = ds_covariance(nmocks=7000, ells=(0,), quantiles=[ds],
                split='r', correlation_type=correlation_type, smin=smin, smax=smax)
            cov_xi2 = ds_covariance(nmocks=7000, ells=(2,), quantiles=[ds],
                split='r', correlation_type=correlation_type, smin=smin, smax=smax)
            std_xi0 = np.sqrt(np.diag(cov_xi0))
            std_xi2 = np.sqrt(np.diag(cov_xi2))
            fig.axes[1].plot(r_c, r_c**2 * xi0[ids], label=rf'${{\rm DS}}_{ds}$')
            fig.axes[3].plot(r_c, r_c**2 * xi2[ids], label=rf'${{\rm DS}}_{ds}$')
            fig.axes[1].fill_between(r_c, r_c**2*(xi0[ids] - std_xi0),
                r_c**2*(xi0[ids] + std_xi0), color=colors[ids], alpha=0.2)
            fig.axes[3].fill_between(r_c, r_c**2*(xi2[ids] - std_xi2),
                r_c**2*(xi2[ids] + std_xi2), color=colors[ids], alpha=0.2)

        if correlation_type == 'auto':
            fig.axes[0].set_ylabel(r'$s^2 \xi_{0}^{\rm qq}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)
            fig.axes[2].set_ylabel(r'$s^2 \xi_{2}^{\rm qq}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)
        else:
            fig.axes[0].set_ylabel(r'$s^2 \xi_{0}^{\rm qh}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)
            fig.axes[2].set_ylabel(r'$s^2 \xi_{2}^{\rm qh}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)

        fig.axes[2].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=14)
        fig.axes[3].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=14)
        fig.axes[0].set_xlim(10, 150)
        leg = fig.axes[0].legend(fontsize=10, loc='best', ncol=2,
        handlelength=0, columnspacing=0.25)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

        fig.axes[0].set_title(f'$z$''-split', fontsize=14)
        fig.axes[1].set_title(f'$r$''-split', fontsize=14)
        plt.setp(fig.axes[0].get_xticklabels(), visible=False)
        plt.setp(fig.axes[1].get_xticklabels(), visible=False)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(f'paper_figures/png/ds_{correlation_type}_multipoles_fiducial.png', dpi=300)
        plt.savefig(f'paper_figures/pdf/ds_{correlation_type}_multipoles_fiducial.pdf')


if __name__ == '__main__':
    plot_multipoles()
