import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from fisher_ingredients import *
import sys
plt.style.use(['science.mplstyle', 'bright.mplstyle'])


def plot_average_density_quintiles():
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']

    nmocks=120
    samples = ['Om', 's8', 'Mnu']

    fig, ax = plt.subplots(3, 1)

    xaxis = [1, 2, 3, 4, 5]

    for i, sample in enumerate(samples):
        pm_density = ds_pm_density(sample=sample, nmocks=nmocks)
        pm_density = np.mean(pm_density, axis=1)
        for j, ds in enumerate(range(5)):
            label = cosmo_dict[sample]['label']

            if sample == 'Mnu':
                fiducial_density = ds_fiducial_density(sample='fiducial_ZA', nmocks=nmocks)
                print(np.shape(fiducial_density))
                error = np.std(fiducial_density, axis=0)
                fiducial_density = np.mean(fiducial_density, axis=0)
                if j == 0:
                    ax[i].plot(xaxis[j], (pm_density[0][j]-fiducial_density[j])/error[j],
                        color=colors[0], label=r'${}^+$'.format(label), lw=1.0, marker='^', ms=5.0, ls='')
                    ax[i].plot(xaxis[j], (pm_density[1][j]-fiducial_density[j])/error[j],
                        color=colors[1], label=r'${}^{{++}}$'.format(label), lw=1.0, marker='v', ms=5.0, ls='')
                else:
                    ax[i].plot(xaxis[j], (pm_density[0][j]-fiducial_density[j])/error[j],
                        color=colors[0], lw=1.0, marker='^', ms=5.0, ls='')
                    ax[i].plot(xaxis[j], (pm_density[1][j]-fiducial_density[j])/error[j],
                        color=colors[1], lw=1.0, marker='v', ms=5.0, ls='')

            else:
                fiducial_density = ds_fiducial_density(sample='fiducial', nmocks=93)
                error = np.std(fiducial_density, axis=0)
                fiducial_density = np.mean(fiducial_density, axis=0)
                if j == 0:
                    ax[i].plot(xaxis[j], (pm_density[0][j]-fiducial_density[j])/error[j],
                        color=colors[0], label=r'${}^+$'.format(label), lw=1.0, marker='^', ms=5.0, ls='')
                    ax[i].plot(xaxis[j], (pm_density[1][j]-fiducial_density[j])/error[j],
                        color=colors[1], label=r'${}^-$'.format(label), lw=1.0, marker='v', ms=5.0, ls='')
                else:
                    ax[i].plot(xaxis[j], (pm_density[0][j]-fiducial_density[j])/error[j],
                        color=colors[0], lw=1.0, marker='^', ms=5.0, ls='')
                    ax[i].plot(xaxis[j], (pm_density[1][j]-fiducial_density[j])/error[j],
                        color=colors[1], lw=1.0, marker='v', ms=5.0, ls='')

        ymin, ymax = ax[i].get_ylim()
        ax[i].hlines(0.0, 0, 5.2, color='grey', ls='--')
        ax[i].vlines(5.2, -10, 10, color='k', ls='-', lw=1.0)
        ax[i].set_xlim(0.8, 6.0)
        ax[i].set_ylim(ymin + ymin/5, ymax + ymax/5)
        ax[i].xaxis.set_ticks([])
        leg = ax[i].legend(loc='center right', ncol=1, fontsize=12,
            handlelength=0, columnspacing=0.25, handletextpad=0, markerscale=0)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())

    xlabels = [r'${\rm DS_1}$', r'${\rm DS_2}$', r'${\rm DS_3}$', r'${\rm DS_4}$', r'${\rm DS_5}$'] 
    plt.xticks(xaxis, xlabels, rotation='vertical')
    ax[2].tick_params(axis='x', which='minor', bottom=False, top=False)
    ax[2].tick_params(axis='x', which='major', top=False)

    ax[1].set_ylabel(r'$\left(\bar{\Delta}(R_s) - \bar{\Delta}(R_s)_{\rm fid}\right) / \sigma_{\bar{\Delta}(R_s)}$')
    plt.setp(fig.axes[0].get_xticklabels(), visible=False)
    plt.setp(fig.axes[1].get_xticklabels(), visible=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    plt.savefig('paper_figures/png/density_quintiles.png', dpi=300)
    plt.savefig('paper_figures/pdf/density_quintiles.pdf')


if __name__ == '__main__':
    plot_average_density_quintiles()
