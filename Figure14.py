import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from fisher_ingredients import *
from cosmology import quijote_cosmology as cosmo_dict
plt.style.use(['science.mplstyle', 'bright.mplstyle'])



def likelihood_full(nderiv=450, ncov=7000, smin=0, smax=150):
    smin = 10
    smax = 150
    ells = (0, 2)
    correlation_type = 'full'
    covariance_correction = 'percival'
    fisher_list = []

    # samples to include in the Fisher matrix
    samples = ["Om", "Ob2", "h", "s8", "ns", "Mmin", "Mnu"]

    # samples to plot
    # samples_plot = ["Om", "Ob2", "h", "s8", "ns", "Mnu"]
    samples_plot = ['Om', 's8', 'Mnu']

    # legend describing each dataset
    legend = [
        '$z$''-split',
        r'$\textit{recon}$''-split',
        '$r$''-split',
    ]


    zsplit = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, split='z', ells=ells, quantiles=[1, 2, 4, 5],
        correlation_type=correlation_type, covariance_correction=covariance_correction)

    reconsplit = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, split='recon', ells=ells, quantiles=[1, 2, 4, 5],
        correlation_type=correlation_type, covariance_correction=covariance_correction)

    rsplit = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, split='r', ells=ells, quantiles=[1, 2, 4, 5],
        correlation_type=correlation_type, covariance_correction=covariance_correction)

    fisher_list.append(zsplit)
    fisher_list.append(reconsplit)
    fisher_list.append(rsplit)

    # add fisher errors as MC samples
    mc_samples = []
    labels = [cosmo_dict[i]['label'] for i in samples]
    mean_theta = [cosmo_dict[i]['fiducial'] for i in samples]

    for i, fisher in enumerate(fisher_list):
        random_state = np.random.default_rng(10)
        cov_theta = np.linalg.inv(fisher)
        samp = random_state.multivariate_normal(mean_theta, cov_theta,
            size=1000000)
        mcs = MCSamples(samples=samp, names=samples, labels=labels)
        mc_samples.append(mcs)

        print(legend[i] + ':')
        for j, sample in enumerate(samples):
            print(f'{sample} = {mean_theta[j]} +- {np.sqrt(np.diag((cov_theta)))[j]:.5f}')
        print('')

    # plot results
    n = len(fisher_list)
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    g = plots.get_subplot_plotter()
    g.settings.line_styles = colors
    g.settings.axes_labelsize = 22
    g.settings.figure_legend_loc = 'upper right'
    # g.settings.alpha_filled_add = 0.7
    g.settings.axes_fontsize = 19
    g.settings.num_plot_contours = 2 
    g.settings.legend_fontsize = 20
    g.settings.legend_colored_text = False
    g.settings.figure_legend_frame = True
    g.settings.linewidth = 7.0
    g.settings.solid_contour_palefactor = 0.5

    param_limits = {
        'Om':(0.22, 0.40),
        'Ob2':(0.01, 0.087),
        'h':(0.20, 1.19),
        's8':(0.65, 1.05),
        'ns':(0.64, 1.31),
        'Mnu':(-0.65, 0.65)}


    g.triangle_plot(
        mc_samples, samples_plot,
        filled=True, title_limit=False,
        legend_labels=legend,
        contour_colors=colors, 
        # param_limits=param_limits
    )

    # plt.annotate(r'$s_{\mathrm{min}} =\ $' + r'${}$'.format(smin) +\
    #     r'$\,h^{-1}{\rm Mpc}$', xy=(0.69, 0.75), xycoords='figure fraction',
    #     bbox=dict(ec='0.5', fc='w', boxstyle='round'), fontsize=26)

    plt.savefig('paper_figures/pdf/likelihood_reconsplit.pdf')
    plt.savefig('paper_figures/png/likelihood_reconsplit.png', dpi=300)

if __name__ == '__main__':

    likelihood_full()


