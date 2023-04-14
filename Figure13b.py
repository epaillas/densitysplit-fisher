import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from gaussian_fisher_ingredients import *
from cosmology import quijote_cosmology as cosmo_dict
plt.style.use(['science.mplstyle', 'bright.mplstyle'])



def likelihood_full(nderiv=200, ncov=1000, smin=0, smax=150):
    smin = 10
    smax = 150
    ells = '0+2'
    correction = 'percival'
    fisher_list = []

    # samples to include in the Fisher matrix
    samples = ["Om", "Ob2", "h", "s8", "ns"]

    # samples to plot
    samples_plot = ['Om', 's8', 'h', 'Ob2']

    # calculate fisher matrices
    tpcf = tpcf_fisher_matrix_multipoles(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, ells=(0,), correction_type=correction)

    ds_12 = ds_fisher_matrix_multipoles(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, split='r', ells=(0,), quantiles=[1, 2],
        corr_type='full', correction_type=correction)

    full_tpcf = tpcf_fisher_matrix_density_multipoles(samples=samples, ncov=ncov,
        nderiv=nderiv, smin=smin, smax=smax, ells=(0,), correction_type=correction)

    fisher_list.append(tpcf)
    fisher_list.append(full_tpcf)
    fisher_list.append(ds_12)

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
        mc_samples.append(mcs)

    # plot results
    n = len(fisher_list)
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors[1], colors[2] = colors[2], colors[1]
    # colors[1], colors[3] = colors[3], colors[1]
    g = plots.get_subplot_plotter()
    g.settings.line_styles = colors
    g.settings.axes_labelsize = 25
    g.settings.figure_legend_loc = 'upper right'
    # g.settings.alpha_filled_add = 0.7
    g.settings.axes_fontsize = 25
    g.settings.num_plot_contours = 2 
    g.settings.legend_fontsize = 25
    g.settings.legend_colored_text = False
    g.settings.figure_legend_frame = True
    g.settings.linewidth = 7.0
    g.settings.solid_contour_palefactor = 0.5

    param_limits = {
        'Om':(mean_theta[0] - 0.15, mean_theta[0] + 0.15), 
        'Ob2':(mean_theta[1] - 0.08, mean_theta[1] + 0.08), 
        'h':(mean_theta[2] - 0.8, mean_theta[2] + 0.8), 
        's8':(mean_theta[3] - 0.1, mean_theta[3] + 0.1), 
    }

    legend_labels = ['2PCF'] + ['2PCF + ' + '$\overline{\Delta}(R_s)$'] + [r'${\rm DS_{1+2}^{\rm qq+qh}}$'] 
    # legend_labels = ['2PCF'] + [r'${\rm DS_{1+2}^{\rm qq+qh}}$'] 


    g.triangle_plot(
        mc_samples, samples_plot,
        filled=True, title_limit=False,
        legend_labels=legend_labels,
        contour_colors=colors, 
        param_limits=param_limits
    )

    plt.annotate(r'$s_{\mathrm{min}} =\ $' + r'${}$'.format(smin) +\
        r'$\,h^{-1}{\rm Mpc}$', xy=(0.63, 0.75), xycoords='figure fraction',
        bbox=dict(ec='0.5', fc='w', boxstyle='round'), fontsize=19)

    plt.savefig(f'paper_figures/png/likelihood_full_gaussian_smin{smin}.png', dpi=300)
    plt.savefig(f'paper_figures/pdf/likelihood_full_gaussian_smin{smin}.pdf')

if __name__ == '__main__':

    likelihood_full()


