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


def plot_correlation_matrix():

    # ---- DENSITY SPLIT ----
    quantiles = [1, 2, 4, 5]
    smin = 10
    smax = 150
    ells = (0, 2)
    split = 'z'
    nmocks = 7000

    fig, ax = plt.subplots()
    corrmat = ds_covariance(nmocks=nmocks, smin=smin,
        smax=smax, ells=ells, norm=True, split=split,
        correlation_type='full', quantiles=quantiles)

    im = ax.imshow(corrmat, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)

    ax.annotate(r'${\rm DS_{1+2+4+5}^{qq + qh}}$', xy=(0.07, 0.91), xycoords='axes fraction',
        bbox=dict(ec='0.5', fc='w', alpha=1.0, boxstyle='round'))

    ax.set_xlabel('bin number')
    ax.set_ylabel('bin number')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r'$R_{ij} = C_{ij}/\sqrt{C_{ii} C_{jj}}$',
        rotation=0, labelpad=10)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    plt.savefig(f'paper_figures/pdf/correlation_matrix_ds1+2+4+5_full_{split}split_ncov{nmocks}.pdf')
    plt.savefig(f'paper_figures/png/correlation_matrix_ds1+2+4+5_full_{split}split_ncov{nmocks}.png', dpi=300)


    # ---- TPCF ----
    fig, ax = plt.subplots()

    corr = tpcf_covariance(nmocks=nmocks, smin=smin, smax=smax, ells=ells, norm=True)
    im = ax.imshow(corr, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, label=f'2PCF')

    ax.annotate(f'2PCF', xy=(0.07, 0.91), xycoords='axes fraction',
        bbox=dict(ec='0.5', fc='w', alpha=1.0, boxstyle='round'))
    ax.set_xlabel('bin number')
    ax.set_ylabel('bin number')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r'$R_{ij} = C_{ij}/\sqrt{C_{ii} C_{jj}}$',
        rotation=0, labelpad=10)
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    plt.tight_layout()

    plt.savefig(f'paper_figures/pdf/correlation_matrix_tpcf_ncov{nmocks}.pdf')
    plt.savefig(f'paper_figures/png/correlation_matrix_tpcf_ncov{nmocks}.png', dpi=300)


if __name__ == '__main__':
    plot_correlation_matrix()
