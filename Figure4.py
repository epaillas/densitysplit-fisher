import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fisher_ingredients import *
plt.style.use(['science.mplstyle', 'bright.mplstyle'])


def plot_multipoles():
    nmocks = 499
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(2, 2, sharex='all', figsize=(6, 5))

    r_c, rsplit_xi = ds_fiducial_multipoles(nmocks=nmocks, split='r',
        correlation_type='cross', return_sep=True, concatenate_ells=False)
    r_c, reconsplit_xi = ds_fiducial_multipoles(nmocks=nmocks,
        split='recon', correlation_type='cross', return_sep=True, concatenate_ells=False)

    r_c, auto_rsplit_xi = ds_fiducial_multipoles(nmocks=nmocks, split='r',
        correlation_type='auto', return_sep=True, concatenate_ells=False)
    r_c, auto_reconsplit_xi = ds_fiducial_multipoles(nmocks=nmocks,
        split='recon', correlation_type='auto', return_sep=True, concatenate_ells=False)

    linestyles = ['-', '-']
    for ids, ds in enumerate([1, 2, 3, 4, 5]):
        cov_xi0 = ds_covariance(nmocks=7000, sample='fiducial', smin=0,
            smax=150, ells=(0,), correlation_type='cross', quantiles=[ds]) 
        cov_xi2 = ds_covariance(nmocks=7000, sample='fiducial', smin=0,
            smax=150, ells=(2,), correlation_type='cross', quantiles=[ds]) 
        std_xi0 = np.sqrt(np.diag(cov_xi0))
        std_xi2 = np.sqrt(np.diag(cov_xi2))
        for i, xi in enumerate([rsplit_xi, reconsplit_xi]):
            multipoles_mean = np.mean(xi, axis=0)
            if i == 0:
                fig.axes[0].plot(r_c, r_c**2*multipoles_mean[ids, 0], ls='',
                    marker='o', ms=2.0, zorder=1, color=colors[ids])
                fig.axes[2].plot(r_c, r_c**2*multipoles_mean[ids, 1], ls='',
                    marker='o', ms=2.0, zorder=1, color=colors[ids])
                fig.axes[0].fill_between(r_c, r_c**2*(multipoles_mean[ids, 0] - std_xi0),
                    r_c**2*(multipoles_mean[ids, 0] + std_xi0), color=colors[ids], alpha=0.2)
                fig.axes[2].fill_between(r_c, r_c**2*(multipoles_mean[ids, 1] - std_xi2),
                    r_c**2*(multipoles_mean[ids, 1] + std_xi2), color=colors[ids], alpha=0.2)
            else:
                fig.axes[0].plot(r_c, r_c**2*multipoles_mean[ids, 0], ls=linestyles[i], zorder=0, lw=1.0)
                fig.axes[2].plot(r_c, r_c**2*multipoles_mean[ids, 1], ls=linestyles[i], zorder=0, lw=1.0)


    for ids, ds in enumerate([1, 2, 3, 4, 5]):
        auto_cov_xi0 = ds_covariance(nmocks=7000, sample='fiducial', smin=0,
            smax=150, ells=(0,), correlation_type='auto', quantiles=[ds]) 
        auto_cov_xi2 = ds_covariance(nmocks=7000, sample='fiducial', smin=0,
            smax=150, ells=(2,), correlation_type='auto', quantiles=[ds]) 
        std_xi0 = np.sqrt(np.diag(auto_cov_xi0))
        std_xi2 = np.sqrt(np.diag(auto_cov_xi2))
        for i, xi in enumerate([auto_rsplit_xi, auto_reconsplit_xi]):
            multipoles_mean = np.mean(xi, axis=0)

            if i == 0:
                fig.axes[1].plot(r_c, r_c**2*multipoles_mean[ids, 0], ls='',
                    marker='o', ms=2.0, zorder=1, color=colors[ids])
                fig.axes[3].plot(r_c, r_c**2*multipoles_mean[ids, 1], ls='',
                    marker='o', ms=2.0, zorder=1, color=colors[ids])
                fig.axes[1].fill_between(r_c, r_c**2*(multipoles_mean[ids, 0] - std_xi0),
                    r_c**2*(multipoles_mean[ids, 0] + std_xi0), color=colors[ids], alpha=0.2)
                fig.axes[3].fill_between(r_c, r_c**2*(multipoles_mean[ids, 1] - std_xi2),
                    r_c**2*(multipoles_mean[ids, 1] + std_xi2), color=colors[ids], alpha=0.2)
            else:
                fig.axes[1].plot(r_c, r_c**2*multipoles_mean[ids, 0], ls=linestyles[i], zorder=0, lw=1.0,
                    label=r'${{\rm DS}}_{}$'.format(ds))
                fig.axes[3].plot(r_c, r_c**2*multipoles_mean[ids, 1], ls=linestyles[i], zorder=0, lw=1.0)

    fig.axes[0].plot(np.nan, np.nan, marker='o', ls='', color='k', label='$r$''-split')
    fig.axes[0].plot(np.nan, np.nan, ls='-', color='k', label=r'$\textit{recon}$''-split')
    fig.axes[0].legend(fontsize=12, handlelength=1)

    fig.axes[0].set_ylabel(r'$s^2 \xi_{0}^{\rm qh}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)
    fig.axes[2].set_ylabel(r'$s^2 \xi_{2}^{\rm qh}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)
    fig.axes[1].set_ylabel(r'$s^2 \xi_{0}^{\rm qq}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)
    fig.axes[3].set_ylabel(r'$s^2 \xi_{2}^{\rm qq}(s)\,[h^{-2}{\rm Mpc^2}]$', fontsize=14)

    fig.axes[2].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=14)
    fig.axes[3].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$', fontsize=14)
    leg = fig.axes[1].legend(fontsize=12, loc='best', ncol=2,
        handlelength=0, columnspacing=0.25)
    for line,text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    plt.setp(fig.axes[0].get_xticklabels(), visible=False)
    plt.setp(fig.axes[1].get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    plt.savefig(f'paper_figures/png/ds_multipoles_reconsplit.png', dpi=300)
    plt.savefig(f'paper_figures/pdf/ds_multipoles_reconsplit.pdf')


if __name__ == '__main__':
    plot_multipoles()
