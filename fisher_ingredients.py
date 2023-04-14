import numpy as np
import os
from pathlib import Path
from cosmology import quijote_cosmology as cosmo_dict


# directory where the data is stored
dir_multipoles = './data/multipoles/'
dir_density = './data/density/'

# order in which the parameters are stored in the multipole arrays
samples = ["Om", "Ob2", "h", "s8", "ns", "Mmin", "Mnu"]

verbose = False

# ----------- DS INGREDIENTS ------------- 

def ds_covariance(nmocks, sample='fiducial', save=True,
    smin=0, smax=150, ells=(0,2), norm=False, split='z', 
    quantiles=[1, 2, 3, 4, 5], correlation_type='full'):

    # use r-split covariance when running reconstruction
    if split == 'recon': split = 'r'

    if correlation_type == 'full':
        multipoles_cross = ds_fiducial_multipoles(
            nmocks=nmocks, sample='fiducial', smin=smin, smax=smax,
            ells=ells, correlation_type='cross', split=split,
            quantiles=quantiles, concatenate_quantiles=True,
        )
        multipoles_auto = ds_fiducial_multipoles(
            nmocks=nmocks, sample='fiducial', smin=smin, smax=smax,
            ells=ells, correlation_type='auto', split=split,
            quantiles=quantiles, concatenate_quantiles=True,
        )
        multipoles = np.concatenate([multipoles_cross, multipoles_auto], axis=1)
    else:
        multipoles = ds_fiducial_multipoles(
            nmocks=nmocks, sample='fiducial', smin=smin, smax=smax,
            ells=ells, correlation_type=correlation_type,
            split=split, quantiles=quantiles, concatenate_quantiles=True,
        )

    multipoles_cov = np.cov(multipoles, rowvar=False)
    multipoles_corr = np.corrcoef(multipoles, rowvar=False)

    if verbose:
        print(f'ds_covariance:')
        print(f'sample, shape pre-cov: {sample}, {np.shape(multipoles)}')
        print(f'sample, shape post-cov: {sample}, {np.shape(multipoles_cov)}')
    if norm:
        return multipoles_corr
    else:
        return multipoles_cov


def ds_pm_multipoles(nmocks, sample, cosmo_dict=None,
    ells=(0, 2), smin=0, smax=150, correlation_type='cross', split='z',
    quantiles=[1, 2, 3, 4, 5], return_sep=False, concatenate_quantiles=False,):

    quantiles = np.asarray(quantiles)
    r_c = np.linspace(0, 150, 30)
    idx = (r_c >= smin) & (r_c <= smax)
    phases = [i for i in range(nmocks)]
    if sample == 'Mnu':
        data_fn = Path(dir_multipoles,
            f'ds_{correlation_type}_multipoles_{split}split_neutrinos_30bins.npy')
    else:
        data_fn = Path(dir_multipoles,
            f'ds_{correlation_type}_multipoles_{split}split_30bins.npy')
    data = np.load(data_fn, allow_pickle=True)

    multipoles_pm = []
    pm_list = ['p', 'pp', 'ppp'] if sample == 'Mnu' else ['p', 'm']
    for ipm, pm in enumerate(pm_list):
        multipoles_phases = []
        for phase in phases:
            multipoles_ds = []
            for ids, ds in enumerate(range(1,6)):
                if sample == 'Mnu':
                    xi_0 = data[ipm][phase][ids][0]
                    xi_2 = data[ipm][phase][ids][1]
                else:
                    xi_0 = data[samples.index(sample)][ipm][phase][ids][0]
                    xi_2 = data[samples.index(sample)][ipm][phase][ids][1]

                xi_0 = xi_0[idx]
                xi_2 = xi_2[idx]
                if 0 in ells and 2 in ells:
                    multipoles = np.concatenate((xi_0, xi_2))
                elif 0 in ells:
                    multipoles = xi_0
                elif 2 in ells:
                    multipoles = xi_2

                multipoles_ds.append(multipoles)
            multipoles_phases.append(multipoles_ds)
        multipoles_phases = np.asarray(multipoles_phases)

        if concatenate_quantiles:
            multipoles_phases = np.concatenate(
                [multipoles_phases[:, ds - 1] for ds in quantiles],
                axis=1
            )
        multipoles_pm.append(multipoles_phases)

    if return_sep:
        return r_c[idx], multipoles_pm
    else:
        return multipoles_pm


def ds_fiducial_multipoles(nmocks, sample='fiducial', ells=(0, 2), smin=0, smax=150,
    correlation_type='cross', split='z', quantiles=[1, 2, 3, 4, 5],
    return_sep=False, concatenate_quantiles=False, concatenate_ells=True):

    quantiles = np.asarray(quantiles)
    r_c = np.linspace(0, 150, 30)
    idx = (r_c >= smin) & (r_c <= smax)
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_multipoles,
        f'ds_{correlation_type}_multipoles_{split}split_{sample}_30bins.npy')
    data = np.load(data_fn, allow_pickle=True)

    multipoles_phases = []
    for phase in phases:
        multipoles_ds = []
        for ids, ds in enumerate(range(1,6)):
            xi_0 = data[phase][ids][0]
            xi_2 = data[phase][ids][1]

            xi_0 = xi_0[idx]
            xi_2 = xi_2[idx]

            if 0 in ells and 2 in ells:
                if concatenate_ells:
                    multipoles = np.concatenate((xi_0, xi_2))
                else:
                    multipoles = [xi_0, xi_2]
            elif 0 in ells:
                multipoles = xi_0
            elif 2 in ells:
                multipoles = xi_2
                    
            multipoles_ds.append(multipoles)

        multipoles_phases.append(multipoles_ds)

    multipoles_phases = np.asarray(multipoles_phases)

    if concatenate_quantiles:
        multipoles_phases = np.concatenate(
            [multipoles_phases[:, ds - 1] for ds in quantiles],
            axis=1
        )

    if verbose:
        print(f'ds_fiducial_multipoles {correlation_type}:')
        print(f'sample, shape: {sample}, {np.shape(multipoles_phases)}')

    if return_sep:
        return r_c[idx], multipoles_phases
    else:
        return multipoles_phases


def ds_fiducial_density(nmocks, sample='fiducial', split='z'):
    density_phases = []
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_density, f'ds_average_density_{split}split_{sample}.npy')
    density_phases = np.load(data_fn, allow_pickle=True)
    return density_phases


def ds_pm_density(nmocks, sample, cosmo_dict=None,
    split='z'):
    phases = [i for i in range(nmocks)]
    if sample == 'Mnu':
        data_fn = Path(dir_density, f'ds_average_density_{split}split_neutrinos.npy')
        density_phases = np.load(data_fn, allow_pickle=True)
    else:
        data_fn = Path(dir_density, f'ds_average_density_{split}split_{sample}.npy')
        density_phases = np.load(data_fn, allow_pickle=True)
        # density_phases = density_phases[samples.index(sample)]
    return density_phases


def ds_numerical_derivatives(sample, nmocks, smin=0, smax=150, ells=(0, 2),
    correlation_type='cross', estimator=3, return_sep=False, split='z',
    quantiles=[1, 2, 3, 4, 5]):
    pm = cosmo_dict[sample]['pm']
    delta = pm[0] - pm[1]
    r_c, poles_pm = ds_pm_multipoles(nmocks=nmocks, sample=sample,
        smin=smin, smax=smax, ells=ells, correlation_type=correlation_type,
        split=split, quantiles=quantiles, return_sep=True,
        concatenate_quantiles=True)
    poles_pm = np.mean(poles_pm, axis=1)
    if sample == 'Mnu':
        r_c, poles_fid = ds_fiducial_multipoles(sample='fiducial_za',
            nmocks=nmocks, ells=ells, smin=smin, smax=smax, correlation_type=correlation_type,
            split=split, quantiles=quantiles, return_sep=True, concatenate_quantiles=True)
        poles_fid = np.mean(poles_fid, axis=0)
        if estimator == 1:
            dxi = (poles_pm[0] - poles_fid) / delta
        elif estimator == 2:
            dxi = (-poles_pm[1] + 4*poles_pm[0]
                - 3*poles_fid) / (2*delta)
        elif estimator == 3:
            dxi = (poles_pm[2] - 12*poles_pm[1] +
                32*poles_pm[0] - 21*poles_fid) / (12*delta)
    else:
        dxi = (poles_pm[0] - poles_pm[1]) / delta
    if return_sep:
        return r_c, dxi
    else:
        return dxi


def ds_fisher_matrix(samples, ncov=7000, nderiv=1500, smin=0, smax=150,
        ells=(0, 2), split='z', quantiles=[1, 2, 3, 4, 5],
        correlation_type='full', covariance_correction='percival'):

    covmat = ds_covariance(nmocks=ncov, smin=smin, smax=smax, ells=ells,
        split=split, quantiles=quantiles, correlation_type=correlation_type)

    nbins = len(covmat)
    if covariance_correction == 'hartlap':
        correction = (ncov - 1)/(ncov - nbins - 2)
        covmat *= correction
    elif covariance_correction == 'percival':
        ntheta = len(samples)
        correction = (ncov - 1)/(ncov - nbins + ntheta - 1)
        covmat *= correction

    precision_matrix = np.linalg.inv(covmat)

    derivatives = []
    for sample in samples:
        if correlation_type == 'full':
            derivatives_cross = ds_numerical_derivatives(sample, nmocks=nderiv,
                smin=smin, smax=smax, ells=ells, correlation_type='cross', split=split,
                quantiles=quantiles)
            derivatives_auto = ds_numerical_derivatives(sample, nmocks=nderiv,
                smin=smin, smax=smax, ells=ells, correlation_type='auto',
                split=split, quantiles=quantiles)
            _derivatives = np.concatenate((derivatives_cross, derivatives_auto))
        else:
            _derivatives = ds_numerical_derivatives(sample, nmocks=nderiv,
                smin=smin, smax=smax, ells=ells, split=split,
                quantiles=quantiles, correlation_type=correlation_type)
        derivatives.append(_derivatives)
    derivatives = np.asarray(derivatives)
    fisher = derivatives @ precision_matrix @ derivatives.T
    return fisher


# ----------- TPCF ingredients ------------- 

def tpcf_covariance(nmocks, sample='fiducial',
    smin=0, smax=150, ells=(0,), norm=False):
    r_c = np.linspace(0, 150, 30)
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_multipoles, f'tpcf_multipoles_fiducial_30bins.npy')
    data = np.load(data_fn, allow_pickle=True)

    multipoles_phases = []
    for phase in phases:
        xi_0 = data[phase][0]
        xi_2 = data[phase][1]
        idx = (r_c >= smin) & (r_c <= smax)
        xi_0 = xi_0[idx]
        xi_2 = xi_2[idx]
        if 0 in ells and 2 in ells:
            multipoles = np.concatenate((xi_0, xi_2))
        elif 0 in ells:
            multipoles = xi_0
        elif 2 in ells:
            multipoles = xi_2
        multipoles_phases.append(multipoles)

    multipoles_phases = np.asarray(multipoles_phases)

    multipoles_corr = np.corrcoef(multipoles_phases, rowvar=False)
    multipoles_cov = np.cov(multipoles_phases, rowvar=False)

    if verbose:
        print(f'tpcf_covariance:')
        print(f'sample, shape pre-cov: {sample}, {np.shape(multipoles_phases)}')
        print(f'sample, shape post-cov: {sample}, {np.shape(multipoles_cov)}')
    if norm:
        return multipoles_corr
    else:
        return multipoles_cov


def tpcf_pm_multipoles(nmocks, sample, ells=(0, 2), smin=0, smax=150,
    return_sep=False):

    r_c = np.linspace(0, 150, 30)
    idx = (r_c >= smin) & (r_c <= smax)
    phases = [i for i in range(nmocks)]
    if sample == 'Mnu':
        data_fn = Path(dir_multipoles,
            f'tpcf_multipoles_neutrinos_ncv_30bins.npy')
    else:
        data_fn = Path(dir_multipoles,
            f'tpcf_multipoles_ncv_30bins.npy')
    data = np.load(data_fn, allow_pickle=True)

    multipoles_pm = []
    pm_list = ['p', 'pp', 'ppp'] if sample == 'Mnu' else ['p', 'm']
    for ipm, pm in enumerate(pm_list):
        multipoles_phases = []
        for phase in phases:
            if sample == 'Mnu':
                xi_0 = data[ipm][phase][0]
                xi_2 = data[ipm][phase][1]
            else:
                xi_0 = data[samples.index(sample)][ipm][phase][0]
                xi_2 = data[samples.index(sample)][ipm][phase][1]

            xi_0 = xi_0[idx]
            xi_2 = xi_2[idx]
            if 0 in ells and 2 in ells:
                multipoles = np.concatenate((xi_0, xi_2))
            elif 0 in ells:
                multipoles = xi_0
            elif 2 in ells:
                multipoles = xi_2

            multipoles_phases.append(multipoles)

        multipoles_phases = np.asarray(multipoles_phases)
        multipoles_pm.append(multipoles_phases)

    if return_sep:
        return r_c[idx], multipoles_pm
    else:
        return multipoles_pm


def tpcf_fiducial_multipoles(nmocks, sample='fiducial', 
    ells=(0,), smin=0, smax=150, return_sep=False):
    r_c = np.linspace(0, 150, 30)
    idx = (r_c >= smin) & (r_c <= smax)
    phases = [i for i in range(nmocks)]
    if sample == 'fiducial_za':
        data_fn = Path(dir_multipoles,
            f'tpcf_multipoles_fiducial_za_ncv_30bins.npy')
    else:
        data_fn = Path(dir_multipoles,
            f'tpcf_multipoles_fiducial_ncv_30bins.npy')
    data = np.load(data_fn, allow_pickle=True)

    multipoles_phases = []
    for phase in phases:
        xi_0 = data[phase][0]
        xi_2 = data[phase][1]
        xi_0 = xi_0[idx]
        xi_2 = xi_2[idx]
        if 0 in ells and 2 in ells:
            multipoles = np.concatenate((xi_0, xi_2))
        elif 0 in ells:
            multipoles = xi_0
        elif 2 in ells:
            multipoles = xi_2
        multipoles_phases.append(multipoles)

    multipoles_phases = np.asarray(multipoles_phases)

    if verbose:
        print(f'tpcf_fiducial_multipoles:')
        print(f'sample, shape: {sample}, {np.shape(multipoles_phases)}')

    if return_sep:
        return r_c[idx], multipoles_phases
    else:
        return multipoles_phases


def tpcf_numerical_derivatives(sample, nmocks, smin=0, smax=210,
    ells=(0,), estimator=3):
    pm = cosmo_dict[sample]['pm']
    delta = pm[0] - pm[1]
    r_c, poles_pm = tpcf_pm_multipoles(nmocks=nmocks, sample=sample,
        smin=smin, smax=smax, ells=ells, return_sep=True)
    poles_pm = np.mean(poles_pm, axis=1)

    if sample == 'Mnu':
        r_c, poles_fid = tpcf_fiducial_multipoles(sample='fiducial_za',
            nmocks=nmocks, ells=ells, smin=smin, smax=smax, return_sep=True)
        poles_fid = np.mean(poles_fid, axis=0)
        if estimator == 1:
            dxi = (poles_pm[0] - poles_fid) / delta
        elif estimator == 2:
            dxi = (-poles_pm[1] + 4*poles_pm[0]
                - 3*poles_fid) / (2*delta)
        elif estimator == 3:
            dxi = (poles_pm[2] - 12*poles_pm[1] +
                32*poles_pm[0] - 21*poles_fid) / (12*delta)
    else:
        dxi = (poles_pm[0] - poles_pm[1]) / delta

    derivatives = dxi
    return derivatives


def tpcf_fisher_matrix(samples, ncov=7000, nderiv=500,
    smin=0, smax=150, ells=(0,), covariance_correction='percival'):

    covmat = tpcf_covariance(nmocks=ncov, smin=smin,
        smax=smax, ells=ells)

    nbins = len(covmat)
    if covariance_correction == 'hartlap':
        correction = (ncov - 1)/(ncov - nbins - 2)
        covmat *= correction
    elif covariance_correction == 'percival':
        ntheta = len(samples)
        correction = (ncov - 1)/(ncov - nbins + ntheta - 1)
        covmat *= correction

    precision_matrix = np.linalg.inv(covmat)

    derivatives = []
    for sample in samples:
        derivatives.append(tpcf_numerical_derivatives(
            sample, nmocks=nderiv, smin=smin, smax=smax, ells=ells))
    derivatives = np.asarray(derivatives)
    fisher = derivatives @ precision_matrix @ derivatives.T

    return fisher
