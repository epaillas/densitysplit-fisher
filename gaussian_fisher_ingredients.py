import numpy as np
import os
from pathlib import Path
from cosmology import quijote_cosmology as cosmo_dict


# directory where the multipoles are stored
dir_multipoles = '/Users/epaillas/code/ds-fisher/data/multipoles/gaussian/'
dir_density = '/Users/epaillas/code/ds-fisher/data/density/gaussian/'

# order in which the parameters are stored in the multipole arrays
global_samples = ["Om", "Ob2", "h", "s8", "ns"]

verbose = True


def ds_covariance_multipoles_full(nmocks, sample='fiducial', save=True,
    smin=0, smax=150, ells=(0,2), los='z', norm=False, split='z', 
    quantiles=[1, 2, 3, 4, 5]):

    multipoles_cross = ds_fiducial_multipoles(
        nmocks=nmocks, sample='fiducial', smin=smin, smax=smax,
        ells=ells, los='z', corr_type='cross', split=split,
        quantiles=quantiles, combine_quantiles=True
    )

    multipoles_auto = ds_fiducial_multipoles(
        nmocks=nmocks, sample='fiducial', smin=smin, smax=smax,
        ells=(0,), los='z', corr_type='auto', split=split,
        quantiles=quantiles, combine_quantiles=True
    )

    multipoles_full = np.concatenate([multipoles_cross, multipoles_auto], axis=1)
    multipoles_cov = np.cov(multipoles_full, rowvar=False)
    multipoles_corr = np.corrcoef(multipoles_full, rowvar=False)

    if verbose:
        print(f'ds_fiducial_covariance:')
        print(f'sample, shape pre-cov: {sample}, {np.shape(multipoles_full)}')
        print(f'sample, shape post-cov: {sample}, {np.shape(multipoles_cov)}')
    if norm:
        return multipoles_corr
    else:
        return multipoles_cov


def ds_covariance_multipoles(nmocks, sample='fiducial', save=True, smin=0, smax=150,
    ells=(0,2), los='z', norm=False, corr_type='cross', split='z', 
    quantiles=[1, 2, 3, 4, 5], combine_quantiles=False):

    multipoles = ds_fiducial_multipoles(
        nmocks=nmocks, sample='fiducial', smin=smin, smax=smax,
        ells=ells, los='z', corr_type=corr_type, split=split,
        quantiles=quantiles, combine_quantiles=combine_quantiles
    )

    multipoles_cov = np.cov(multipoles, rowvar=False)
    multipoles_corr = np.corrcoef(multipoles, rowvar=False)

    if verbose:
        print(f'ds_fiducial_covariance:')
        print(f'sample, shape pre-cov: {sample}, {np.shape(multipoles)}')
        print(f'sample, shape post-cov: {sample}, {np.shape(multipoles_cov)}')
    if norm:
        return multipoles_corr
    else:
        return multipoles_cov


def ds_pm_multipoles(nmocks, sample, cosmo_dict=None,
    ells=(0, 2), smin=0, smax=150, corr_type='cross', split='z',
    quantiles=[1, 2, 3, 4, 5], return_sep=False, combine_quantiles=False):

    quantiles = np.asarray(quantiles)
    r_c = np.linspace(0, 150, 30)
    idx = (r_c >= smin) & (r_c <= smax)
    phases = [i for i in range(nmocks)]
    if sample == 'Mnu':
        data_fn = Path(dir_multipoles,
            f'ds_rspace_{corr_type}_multipoles_{split}split_neutrinos_30bins.npy')
    else:
        data_fn = Path(dir_multipoles,
            f'ds_rspace_{corr_type}_multipoles_{split}split_30bins.npy')
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
                    xi_0 = data[global_samples.index(sample)][ipm][phase][ids][0]
                    xi_2 = data[global_samples.index(sample)][ipm][phase][ids][1]

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

        if combine_quantiles:
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
    corr_type='cross', los='z', split='z', quantiles=[1, 2, 3, 4, 5], 
    return_sep=False, combine_quantiles=False):

    quantiles = np.asarray(quantiles)
    r_c = np.linspace(0, 150, 30)
    idx = (r_c >= smin) & (r_c <= smax)
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_multipoles,
        f'ds_rspace_{corr_type}_multipoles_{split}split_{sample}_30bins.npy')
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
                multipoles = np.concatenate((xi_0, xi_2))
            elif 0 in ells:
                multipoles = xi_0
            elif 2 in ells:
                multipoles = xi_2

            multipoles_ds.append(multipoles)

        multipoles_phases.append(multipoles_ds)

    multipoles_phases = np.asarray(multipoles_phases)

    if combine_quantiles:
        multipoles_phases = np.concatenate(
            [multipoles_phases[:, ds - 1] for ds in quantiles],
            axis=1
        )

    if verbose:
        print(f'ds_fiducial_multipoles {corr_type}:')
        print(f'sample, shape: {sample}, {np.shape(multipoles_phases)}')

    if return_sep:
        return r_c[idx], multipoles_phases
    else:
        return multipoles_phases


def ds_derivatives_multipoles(sample, nmocks, smin=0, smax=150, ells=(0, 2),
    los=('z',), corr_type='cross', estimator=3, return_dist=False, split='z',
    quantiles=[1, 2, 3, 4, 5]):
    pm = cosmo_dict[sample]['pm']
    delta = pm[0] - pm[1]
    r_c, poles_pm = ds_pm_multipoles(nmocks=nmocks, sample=sample,
        smin=smin, smax=smax, ells=ells, corr_type=corr_type,
        split=split, quantiles=quantiles, return_sep=True,
        combine_quantiles=True)
    poles_pm = np.mean(poles_pm, axis=1)
    if sample == 'Mnu':
        r_c, poles_fid = ds_fiducial_multipoles(sample='fiducial_za',
            nmocks=nmocks, ells=ells, smin=smin, smax=smax, corr_type=corr_type,
            split=split, quantiles=quantiles, return_sep=True, combine_quantiles=True)
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
    if return_dist:
        return r_c, dxi
    else:
        return dxi


def ds_fisher_matrix_multipoles(samples, save=False, ncov=7000, nderiv=1500, smin=0, smax=150,
    ells=(0, 2), los=('z',), corr_type='cross', split='z', quantiles=[1, 2, 3, 4, 5],
    correction_type='percival'):
    derivatives = []

    if corr_type == 'full':
        covmat = ds_covariance_multipoles_full(nmocks=ncov, smin=smin,
            smax=smax, ells=ells, split=split, quantiles=quantiles)

        if correction_type == 'percival':
            n_s = ncov
            n_d = len(covmat)
            n_theta = len(samples)
            correction = (n_s - n_d + n_theta - 1)/(n_s - 1)
        elif correction_type == 'hartlap':
            n_d = len(covmat)
            n_s = ncov
            correction = (n_s - 1)/(n_s - n_d - 2)
        precision_matrix = np.linalg.inv(correction * covmat)

        for sample in samples:
            derivatives_cross = ds_derivatives_multipoles(sample, nmocks=nderiv,
                smin=smin, smax=smax, los=los, ells=ells, corr_type='cross', split=split,
                quantiles=quantiles)

            derivatives_auto = ds_derivatives_multipoles(sample, nmocks=nderiv,
                smin=smin, smax=smax, los=los, ells=(0,), corr_type='auto',
                split=split, quantiles=quantiles)

            derivatives_full = np.concatenate((derivatives_cross, derivatives_auto))
            derivatives.append(derivatives_full)
    else:
        covmat = ds_covariance_multipoles(nmocks=ncov, smin=smin, smax=smax, ells=ells,
            corr_type=corr_type, split=split, quantiles=quantiles, combine_quantiles=True)

        if correction_type == 'percival':
            n_s = ncov
            n_d = len(covmat)
            n_theta = len(samples)
            correction = (n_s - n_d + n_theta - 1)/(n_s - 1)
        elif correction_type == 'hartlap':
            n_d = len(covmat)
            n_s = ncov
            correction = (n_s - 1)/(n_s - n_d - 2)
        precision_matrix = np.linalg.inv(correction * covmat)

        for sample in samples:
            derivatives.append(ds_derivatives_multipoles(sample, nmocks=nderiv,
                smin=smin, smax=smax, los=los, ells=ells, corr_type=corr_type,
                split=split, quantiles=quantiles))

    derivatives = np.asarray(derivatives)
    return derivatives @ precision_matrix @ derivatives.T




def ds_covariance_density(nmocks, sample='fiducial', split='z'):

    density = ds_fiducial_density(nmocks=nmocks, sample=sample, split=split)

    return np.cov(density, rowvar=False)


def ds_fiducial_density(nmocks, sample='fiducial', split='z'):

    density_phases = []
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_density, f'ds_average_density_{split}split_{sample}.npy')
    data = np.load(data_fn, allow_pickle=True)
    for phase in phases:
        density_quantiles = []
        for ids in range(5):
            density = data[phase, ids]
            density_quantiles.append(density)
        density_phases.append(density_quantiles)
    density_phases = np.asarray(density_phases, dtype=float)

    return density_phases


def ds_pm_density(nmocks, sample, split='z'):
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_density, f'ds_average_density_{split}split.npy')
    data = np.load(data_fn, allow_pickle=True)

    density_pm = []
    pm_list = ['p', 'pp', 'ppp'] if sample == 'Mnu' else ['p', 'm']
    for ipm, pm in enumerate(pm_list):
        print(f'ds_pm_multipoles: {sample}, {pm}, {nmocks}')
        density_phases = []
        for phase in phases:
            density_quantiles = []
            for ids in range(5):
                density = data[global_samples.index(sample)][ipm][phase][ids]
                density_quantiles.append(density)
            density_phases.append(density_quantiles)
        density_pm.append(density_phases)
    density_pm = np.asarray(density_pm, dtype=float)
    print('asd', np.shape(density_pm))
    return density_pm


def ds_derivatives_density(sample, nmocks, estimator=3, split='z'):
    pm = cosmo_dict[sample]['pm']
    delta = pm[0] - pm[1]
    poles_pm = ds_pm_density(nmocks=nmocks, sample=sample,
        split=split)
    poles_pm = np.mean(poles_pm, axis=1)
    if sample == 'Mnu':
        poles_fid = ds_fiducial_density(sample='fiducial_za',
            nmocks=nmocks, split=split)
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
    return dxi


def ds_fisher_matrix_density(samples, ncov=7000, nderiv=1500, split='z',
    correction_type='percival'):
    derivatives = []

    covmat = ds_covariance_density(nmocks=ncov, split=split)

    if correction_type == 'percival':
        n_s = ncov
        n_d = len(covmat)
        n_theta = len(samples)
        correction = (n_s - n_d + n_theta - 1)/(n_s - 1)
    elif correction_type == 'hartlap':
        n_d = len(covmat)
        n_s = ncov
        correction = (n_s - 1)/(n_s - n_d - 2)
    precision_matrix = np.linalg.inv(correction * covmat)

    for sample in samples:
        derivatives.append(ds_derivatives_density(sample, nmocks=nderiv,
            split=split))

    derivatives = np.asarray(derivatives)

    return derivatives @ precision_matrix @ derivatives.T



def tpcf_covariance_multipoles(nmocks, sample='fiducial',
    smin=10, smax=150, ells=(0,), norm=False):
    r_c = np.linspace(0, 150, 30)
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_multipoles, f'tpcf_rspace_multipoles_fiducial_30bins.npy')
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
        print(f'covariance_multipoles:')
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
            f'tpcf_rspace_multipoles_neutrinos_30bins.npy')
    else:
        data_fn = Path(dir_multipoles,
            f'tpcf_rspace_multipoles_30bins.npy')
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
                xi_0 = data[global_samples.index(sample)][ipm][phase][0]
                xi_2 = data[global_samples.index(sample)][ipm][phase][1]

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
            f'tpcf_rspace_multipoles_fiducial_za_30bins.npy')
    else:
        data_fn = Path(dir_multipoles,
            f'tpcf_rspace_multipoles_fiducial_30bins.npy')
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
        print(f'fiducial_multipoles:')
        print(f'sample, shape: {sample}, {np.shape(multipoles_phases)}')

    if return_sep:
        return r_c[idx], multipoles_phases
    else:
        return multipoles_phases


def tpcf_derivatives_multipoles(sample, nmocks, smin=0, smax=210,
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
            dxi = (poles[0] - poles_fid) / delta
        elif estimator == 2:
            dxi = (-poles[1] + 4*poles[0]
                - 3*poles_fid) / (2*delta)
        elif estimator == 3:
            dxi = (poles_pm[2] - 12*poles_pm[1] +
                32*poles_pm[0] - 21*poles_fid) / (12*delta)
    else:
        dxi = (poles_pm[0] - poles_pm[1]) / delta

    derivatives = dxi
    return derivatives


def tpcf_fisher_matrix_multipoles(samples, ncov=7000, nderiv=500,
    smin=0, smax=150, ells=(0,), correction_type='percival'):

    covmat = tpcf_covariance_multipoles(nmocks=ncov, smin=smin,
        smax=smax, ells=ells)

    if correction_type == 'percival':
        n_s = ncov
        n_d = len(covmat)
        n_theta = len(samples)
        correction = (n_s - n_d + n_theta - 1)/(n_s - 1)
    elif correction_type == 'hartlap':
        n_d = len(covmat)
        n_s = ncov
        correction = (n_s - 1)/(n_s - n_d - 2)
    precision_matrix = np.linalg.inv(correction * covmat)

    derivatives = []
    for sample in samples:
        derivatives.append(tpcf_derivatives_multipoles(
            sample, nmocks=nderiv, smin=smin, smax=smax, ells=ells))
    derivatives = np.asarray(derivatives)
    fisher = derivatives @ precision_matrix @ derivatives.T

    return fisher





def ds_fiducial_density(nmocks, sample='fiducial', split='z'):

    density_phases = []
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_density, f'ds_average_density_{split}split_{sample}.npy')
    data = np.load(data_fn, allow_pickle=True)
    for phase in phases:
        density_quantiles = []
        for ids in range(2):
            density = data[phase, ids]
            density_quantiles.append(density)
        density_phases.append(density_quantiles)
    density_phases = np.asarray(density_phases, dtype=float)

    return density_phases


def ds_pm_density(nmocks, sample, split='z'):
    phases = [i for i in range(nmocks)]
    data_fn = Path(dir_density, f'ds_average_density_{split}split.npy')
    data = np.load(data_fn, allow_pickle=True)

    density_pm = []
    pm_list = ['p', 'pp', 'ppp'] if sample == 'Mnu' else ['p', 'm']
    for ipm, pm in enumerate(pm_list):
        density_phases = []
        for phase in phases:
            density_quantiles = []
            for ids in range(2):
                density = data[global_samples.index(sample)][ipm][phase][ids]
                density_quantiles.append(density)
            density_phases.append(density_quantiles)
        density_pm.append(density_phases)
    density_pm = np.asarray(density_pm, dtype=float)
    return density_pm


def ds_covariance_density(nmocks, sample='fiducial', split='z'):

    density = ds_fiducial_density(nmocks=nmocks, sample=sample, split=split)

    return np.cov(density, rowvar=False)


def tpcf_covariance_density_multipoles(nmocks, sample='fiducial', split='z',
    ells=(0,), smin=0, smax=150):

    density = ds_fiducial_density(nmocks=nmocks, sample=sample, split=split)

    multipoles =  tpcf_fiducial_multipoles(nmocks, sample=sample, 
        ells=ells, smin=smin, smax=smax)

    full = np.concatenate([multipoles, density], axis=1)
    cov = np.cov(full, rowvar=False)

    return cov


def ds_derivatives_density(sample, nmocks, estimator=3, split='z'):
    pm = cosmo_dict[sample]['pm']
    delta = pm[0] - pm[1]
    poles_pm = ds_pm_density(nmocks=nmocks, sample=sample,
        split=split)
    poles_pm = np.mean(poles_pm, axis=1)
    if sample == 'Mnu':
        poles_fid = ds_fiducial_density(sample='fiducial_za',
            nmocks=nmocks, split=split)
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
    return dxi


def ds_fisher_matrix_density(samples, ncov=7000, nderiv=1500, split='z'):
    derivatives = []

    covmat = ds_covariance_density(nmocks=ncov, split=split)
    correction = (ncov - 2 - len(covmat))/(ncov - 1)
    precision_matrix = correction * np.linalg.inv(covmat)

    for sample in samples:
        derivatives.append(ds_derivatives_density(sample, nmocks=nderiv,
            split=split))

    derivatives = np.asarray(derivatives)

    return derivatives @ precision_matrix @ derivatives.T



def tpcf_fisher_matrix_density_multipoles(samples, ncov=7000, nderiv=500,
    smin=0, smax=150, ells=(0,), correction_type='percival', split='r'):

    covmat = tpcf_covariance_density_multipoles(nmocks=ncov, smin=smin,
        smax=smax, ells=ells, split=split)

    if correction_type == 'percival':
        n_s = ncov
        n_d = len(covmat)
        n_theta = len(samples)
        correction = (n_s - n_d + n_theta - 1)/(n_s - 1)
    elif correction_type == 'hartlap':
        n_d = len(covmat)
        n_s = ncov
        correction = (n_s - 1)/(n_s - n_d - 2)
    precision_matrix = np.linalg.inv(correction * covmat)

    derivatives = []
    for sample in samples:

        derivatives_multipoles = tpcf_derivatives_multipoles(
            sample=sample, nmocks=nderiv, smin=smin, smax=smax,
            ells=ells)

        derivatives_density = ds_derivatives_density(sample=sample,
            nmocks=100, split=split)
        derivatives_full = np.concatenate([derivatives_multipoles,
            derivatives_density])
        derivatives.append(derivatives_full)
    derivatives = np.asarray(derivatives)
    fisher = derivatives @ precision_matrix @ derivatives.T

    return fisher


