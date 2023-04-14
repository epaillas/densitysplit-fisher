from density import readDensityData
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(['science.mplstyle', 'bright.mplstyle'])

# read projected DM distribution
data_fn = 'data/density/XY.a_den'
header, data = readDensityData(data_fn)
density = data.reshape(5000, 5000).T

# read quantiles
data_fn = 'data/quantiles/ds1_rsplit_tophat20_zlos_fiducial_ph0_z0.0.txt'
ds1 = np.genfromtxt(data_fn)
condition = (ds1[:, 2] > 50) & (ds1[:, 2] < 100)
ds1 = ds1[condition]

data_fn = 'data/quantiles/ds5_rsplit_tophat20_zlos_fiducial_ph0_z0.0.txt'
ds5 = np.genfromtxt(data_fn)
condition = (ds5[:, 2] > 50) & (ds5[:, 2] < 100)
ds5 = ds5[condition]

fig, ax = plt.subplots(1, 2, figsize=(5, 5))

for aa in ax:
    aa.imshow(np.log10(density), cmap='afmhot',
        origin='lower', extent=(0, 1000, 0, 1000),
        interpolation='gaussian', vmin=-0.9, vmax=2.0)

ax[0].plot(ds1[:, 0], ds1[:, 1], marker='o', ms=1.5, markerfacecolor="w",
    markeredgecolor='w', ls='', alpha=0.7, markeredgewidth=0.0)
ax[1].plot(ds5[:, 0], ds5[:, 1], marker='o', ms=1.5, markerfacecolor="w",
    markeredgecolor='w', ls='', alpha=0.7, markeredgewidth=0.0)

ax[0].set_xlabel(r'$x\,\left[h^{-1}{\rm Mpc}\right]$')
ax[1].set_xlabel(r'$x\,\left[h^{-1}{\rm Mpc}\right]$')
ax[0].set_ylabel(r'$y\,\left[h^{-1}{\rm Mpc}\right]$')

ax[0].annotate(r'${\rm DS_1}$', xy=(0.825, 0.89), xycoords='axes fraction',
    bbox=dict(ec='0.5', fc='w', alpha=0.7, boxstyle='round'))
ax[1].annotate(r'${\rm DS_5}$', xy=(0.825, 0.89), xycoords='axes fraction',
    bbox=dict(ec='0.5', fc='w', alpha=0.7, boxstyle='round'))

for aa in ax:
    aa.set_xlim(0, 500)
    aa.set_ylim(0, 500)

plt.setp(fig.axes[1].get_yticklabels(), visible=False)
plt.tight_layout()
plt.subplots_adjust(wspace=0.05)

output_fn = 'paper_figures/png/projected_density_afmhot.png'
plt.savefig(output_fn, format='png', dpi=500)
