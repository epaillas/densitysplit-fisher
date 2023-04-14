import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from getdist import plots, MCSamples
from fisher_ingredients import *
#from general_fisher import ds_fisher_matrix_full #Add this to run with Carols code
from cosmology import quijote_cosmology as cosmo_dict
plt.style.use(['science.mplstyle', 'bright.mplstyle'])



def likelihood_quintiles(nderiv=1500, ncov=7000, smin=0, smax=150):
    smin = 10
    smax = 150
    ells = (0,2)
    split = 'z'
   
    # samples to include in the Fisher matrix
    samples = ["Om", "Ob2", "h", "s8", "ns", "Mmin", "Mnu"]

    # samples to plot
    samples_plot = ['Om', 's8', 'Mnu']

    # legend describing each dataset
    legend = [rf'${{\rm DS_{ds}^{{qq + qh}}}}$' for ds in range(1, 6)]\
        + [r'${\rm DS_{1+2+4+5}^{qq + qh}}$']\
        + [r'${\rm 2PCF}$']

    fisher_list = []
    for ds in range(1, 6):
        ds_i = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
            smin=smin, smax=smax, split=split, ells=ells, quantiles=[ds],
            correlation_type='full')
        fisher_list.append(ds_i)

    ds_1245 = ds_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, split=split, ells=ells, quantiles=[1, 2, 4, 5],
        correlation_type='full')
    fisher_list.append(ds_1245)

    tpcf = tpcf_fisher_matrix(samples=samples, ncov=ncov, nderiv=nderiv,
        smin=smin, smax=smax, ells=ells)
    fisher_list.append(tpcf)

    # add fisher errors as MC samples
    mc_samples = []
    std_theta = []
    labels = [cosmo_dict[i]['label'] for i in samples]

    for i, fisher in enumerate(fisher_list):
        random_state = np.random.default_rng(10)
        cov_theta = np.linalg.inv(fisher)
        mean_theta = [cosmo_dict[i]['fiducial'] for i in samples]
        samp = random_state.multivariate_normal(mean_theta, cov_theta,
            size=1000000)
        mcs = MCSamples(samples=samp, names=samples, labels=labels)
        mc_samples.append(mcs)

        print(legend[i] + ':')
        for j, sample in enumerate(samples):
            print(f'{sample} = {mean_theta[j]} +- {np.sqrt(np.diag((cov_theta)))[j]:.5f}')
        print('')


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
    g.settings.figure_legend_frame = False
    g.settings.figure_legend_ncol = 1
    g.settings.linewidth = 5.0
    contour_ls = ['-', '-', '-', '-', '-', '-', '--']

    param_limits = {
        'Om':(mean_theta[0] - 0.08, mean_theta[0] + 0.08), 
        'Ob2':(mean_theta[1] - 0.02, mean_theta[1] + 0.02), 
        'h':(mean_theta[2] - 0.1, mean_theta[2] + 0.1), 
        's8':(mean_theta[3] - 0.17, mean_theta[3] + 0.17), 
        'ns':(mean_theta[4] - 0.02, mean_theta[4] + 0.02), 
        'Mnu':(mean_theta[6] - 0.6, mean_theta[6] + 0.6), 
    }

    g.triangle_plot(
        mc_samples, samples_plot,
        filled=False, title_limit=False,
        legend_labels=legend,
        contour_colors=colors, 
        contour_ls=contour_ls,
        param_limits=param_limits
    )
    plt.legend(frameon=False)
    plt.savefig(f'paper_figures/pdf/likelihood_quintiles_{split}split.pdf')
    plt.savefig(f'paper_figures/png/likelihood_quintiles_{split}split.png', dpi=300)

if __name__ == '__main__':

    likelihood_quintiles()


