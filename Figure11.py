import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from general_fisher import ds_fisher_matrix_full

from cosmology import quijote_cosmology as cosmo_dict
plt.style.use(['science.mplstyle', 'bright.mplstyle'])


def likelihood_bridge(nderiv=1500, ncov=7000, smin=0, smax=150):
    smin = 10
    smax = 150
    fisher_list = []

    # samples to include in the Fisher matrix
    samples = ["Om", "Ob2", "h", "s8", "ns", "Mmin", "Mnu"]

    # samples to plot
    samples_plot = ['Om', 's8', 'Mnu',]

    # calculate fisher matrices

    ds_cross = ds_fisher_matrix_full(samples=samples, ncov=ncov,
        nderiv=nderiv, smin=smin, smax=smax, split='z', ells=(0, 2),
        quantiles=[1, 2, 4, 5], corr_type=('cross',))

    ds_full = ds_fisher_matrix_full(samples=samples, ncov=ncov,
        nderiv=nderiv, smin=smin, smax=smax, split='z', ells=(0, 2),
        quantiles=[1, 2, 4, 5])

    
    ds_cross_r = ds_fisher_matrix_full(samples=samples, ncov=ncov,
        nderiv=nderiv, smin=smin, smax=smax, split='r', ells=(0, 2),
        quantiles=[1, 2, 4, 5], corr_type=('cross',))

    ds_full_r = ds_fisher_matrix_full(samples=samples, ncov=ncov,
        nderiv=nderiv, smin=smin, smax=smax, split='r', ells=(0, 2),
        quantiles=[1, 2, 4, 5])

    fisher_list.append(ds_cross)
    fisher_list.append(ds_cross_r)
    fisher_list.append(ds_full)
    fisher_list.append(ds_full_r)

    # add fisher errors as MC samples
    mc_samples = []
    labels = [cosmo_dict[i]['label'] for i in samples]

    for fisher in fisher_list:
        random_state = np.random.default_rng(10)
        cov_theta = np.linalg.inv(fisher)
        print('1-sigma errors: ', np.sqrt(np.diag((cov_theta))))

        mean_theta = [cosmo_dict[i]['fiducial'] for i in samples]
        samp = random_state.multivariate_normal(mean_theta, cov_theta,
            size=1000000)
        mcs = MCSamples(samples=samp, names=samples, labels=labels)
        #print(1./(np.linalg.det(mcs.cov())**0.5))
        mc_samples.append(mcs)




    n = len(fisher_list)
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors[6] = 'darkgrey'
    g = plots.get_subplot_plotter()
    g.settings.line_styles = colors
    g.settings.axes_labelsize = 23
    g.settings.figure_legend_loc = 'upper right'
    g.settings.axes_fontsize = 20
    g.settings.num_plot_contours = 1 
    g.settings.legend_fontsize = 18
    g.settings.legend_colored_text = True
    g.settings.figure_legend_frame = True
    g.settings.figure_legend_ncol = 1
    g.settings.linewidth = 5.0
    contour_ls = ['-', '-', '-', '-', '-', '-', '--']
    param_limits = {
        'Om':(0.28, 0.335),
        'Ob2':(0.023, 0.077),
        'h':(0.35, 1.0),
        's8':(0.76, 0.91),
        'ns':(0.84, 0.93),
        'Mnu':(-0.3, 0.3)}


    legend_labels = [r'${\rm DS^{\rm qh}}\, (z$''-split)'] \
        + [r'${\rm DS^{\rm qh}}\, (r$''-split)'] \
        + [r'${\rm DS^{\rm qq+qh}}\, (z$''-split)']  \
        + [r'${\rm DS^{\rm qq+qh}}\, (r$''-split)']

    g.triangle_plot(
        mc_samples, samples_plot,
        filled=False, title_limit=False,
        legend_labels=legend_labels,
        contour_colors=colors, 
        contour_ls=contour_ls,
        #contour_lws=[1,1.1,1.,1.1],
        param_limits=param_limits
    )
    plt.savefig('paper_figures/pdf/likelihood_bridge_gap.pdf')
    plt.savefig('paper_figures/png/likelihood_bridge_gap.png', dpi=300)

if __name__ == '__main__':

    likelihood_bridge()


