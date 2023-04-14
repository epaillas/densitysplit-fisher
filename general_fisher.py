from typing import Optional, List
import numpy as np
from pathlib import Path
from cosmology import quijote_cosmology as cosmo_dict


DATA_DIR = Path("data/multipoles/")
# order in which the parameters are stored in the multipole arrays
parameters = [
    "Om",
    "Ob2",
    "h",
    "s8",
    "ns",
    "Mmin",
    "Mnu",
]


def compute_numerical_derivative(
    summary_non_positive: np.array,
    summary_positive,
    fiducial_summary_positive,
    params_non_positive: List[str] = parameters[:-1],
    params_positive: List[str] = ["Mnu"],
    n_mocks: Optional[int] = None,
) -> np.array:
    """Estimate the numerical derivative of a summary statistic
    measured on a grid of cosmological parameters

    Args:
        summary (np.array): summary statistic with shape
            [n_params, 2, n_mocks, n_data], where n_params is the number
            of cosmological parameters, 2 denotes plus and minus variations,
            n_mocks is the number of mocks used to estimate derivates
            and n_data is the dimensionality of the data vector
        n_mocks (Optional[int]): n_mocks, number of mocks to use

    Returns:
        np.array:
        numerical derivatives, shape [n_data]
    """
    derivative = []
    for i, param in enumerate(params_non_positive):
        pm = cosmo_dict[param]["pm"]
        delta_param = pm[0] - pm[1]
        derivative.append(
            (summary_non_positive[i, 0] - summary_non_positive[i, 1]) / delta_param
        )
    for i, param in enumerate(params_positive):
        derivative.append(
            compute_numerical_derivative_positive_bounded(
                summary=summary_positive[i],
                fiducial_summary=fiducial_summary_positive[i],
                param=param,
            )
        )
    derivative = np.array(derivative)
    if n_mocks is not None:
        derivative = derivative[:, :n_mocks, ...]
    return np.mean(derivative, axis=1)


def compute_numerical_derivative_positive_bounded(
    summary: np.array,
    fiducial_summary: np.array,
    estimator_order: int = 3,
    param="Mnu",
):
    pm = cosmo_dict[param]["pm"]
    delta_param = pm[0] - pm[1]
    if estimator_order == 1:
        derivative = (summary[0] - fiducial_summary) / delta_param
    elif estimator_order == 2:
        derivative = (
            (-summary[1] + 4.0 * summary[0] - 3.0 * fiducial_summary)
            / 2.0
            / delta_param
        )
    elif estimator_order == 3:
        derivative = (
            (
                summary[2]
                - 12.0 * summary[1]
                + 32.0 * summary[0]
                - 21.0 * fiducial_summary
            )
            / 12.0
            / delta_param
        )
    return derivative


def compute_covariance(
    summary: np.array,
    n_mocks: Optional[int] = None,
) -> np.array:
    """Estimates the covariance matrix of a summary statistitc

    Args:
        summary (np.array): summary statistic measured at one cosmology
            on a set of ```n_mocks``` simulations
            with shape [n_mocks, n_data]
        n_mocks (Optional[int]): number of simulations to use

    Returns:
        np.array: covariance matrix of shape [n_data, n_data]
    """
    if n_mocks is not None:
        summary = summary[:n_mocks]
    return np.cov(summary.T)


def compute_precision_matrix(
    summary: np.array,
    n_mocks: Optional[int] = None,
    correction: Optional[str] = "hartlap",
    n_params: Optional[int] = None,
) -> np.array:
    if n_mocks is None:
        n_mocks = len(summary)
    covariance = compute_covariance(
        summary=summary,
        n_mocks=n_mocks,
    )
    n_data = len(covariance)
    if correction == "percival":
        if n_params is not None:
            B = (
                (n_mocks - n_data - 2.0)
                / (n_mocks - n_data - 1)
                / (n_mocks - n_data - 4.0)
            )
            correction_factor = (
                (n_mocks - 1.0)
                * (1 + B * (n_data - n_params))
                / (n_mocks - n_data + n_params - 1.0)
            )
            covariance *= correction_factor
        else:
            raise ValueError(f"Percival correction factor needs to know n_params")
    precision_matrix = np.linalg.solve(
        covariance, np.eye(len(covariance), len(covariance))
    )
    if correction == "hartlap":
        correction_factor = (n_mocks - 2 - n_data) / (n_mocks - 1)
        precision_matrix *= correction_factor
    return precision_matrix


def compute_fisher_matrix(
    derivatives: np.array, precision_matrix: np.array
) -> np.array:
    return derivatives @ precision_matrix @ derivatives.T


def compute_fisher_for_summary(
    summary: np.array,
    fiducial_summary: np.array,
    summary_positive: np.array,
    fiducial_summary_positive: np.array,
    correction: Optional[str] = "percival",
    n_mocks_derivative: Optional[int] = None,
    n_mocks_covariance: Optional[int] = None,
    return_ingredients: bool = False,
) -> np.array:
    n_params = summary.shape[1]
    derivatives = compute_numerical_derivative(
        summary_non_positive=summary,
        summary_positive=summary_positive,
        fiducial_summary_positive=fiducial_summary_positive,
        n_mocks=n_mocks_derivative,
    )
    precision_matrix = compute_precision_matrix(
        fiducial_summary,
        correction=correction,
        n_mocks=n_mocks_covariance,
        n_params=n_params,
    )
    fisher = compute_fisher_matrix(
        derivatives,
        precision_matrix,
    )
    if return_ingredients:
        return fisher, derivatives, precision_matrix
    return fisher


def compute_error_from_summary(
    summary: np.array,
    fiducial_summary: np.array,
    summary_positive: np.array,
    fiducial_summary_positive: np.array,
    correction: Optional[str] = "percival",
    n_mocks_derivative: Optional[int] = None,
    n_mocks_covariance: Optional[int] = None,
):
    fisher = compute_fisher_for_summary(
        summary=summary,
        summary_positive=summary_positive,
        fiducial_summary_positive=fiducial_summary_positive,
        fiducial_summary=fiducial_summary,
        correction=correction,
        n_mocks_derivative=n_mocks_derivative,
        n_mocks_covariance=n_mocks_covariance,
    )
    inverse_fisher = np.linalg.solve(fisher, np.eye(len(fisher), len(fisher)))
    return np.sqrt(np.diag(inverse_fisher))


def ds_fisher_matrix_full(
    ncov=7000,
    nderiv=1500,
    smin=0,
    smax=150,
    ells=(0, 2),
    los=("z",),
    split="z",
    corr_type=("auto", "cross"),
    quantiles=[1, 2, 3, 4, 5],
    correction="hartlap",
    return_ingredients=False,
    **kwargs,
):
    s_bins = np.linspace(0, 150, 31)
    s = 0.5 * (s_bins[1:] + s_bins[:-1])
    s_mask = (s >= smin) & (s <= smax)
    ells_mask = [True if ell in ells else False for ell in [0, 2]]
    quantile_mask = [True if q + 1 in quantiles else False for q in range(5)]
    if "cross" in corr_type:
        ds_cross = np.load(DATA_DIR / f"ds_cross_multipoles_{split}split_30bins.npy")[
            ..., ells_mask, :
        ]
        fiducial_ds_cross = np.load(
            DATA_DIR / f"ds_cross_multipoles_{split}split_fiducial_30bins.npy"
        )[..., ells_mask, :]
        ds_cross_nu = np.load(
            DATA_DIR / f"ds_cross_multipoles_{split}split_neutrinos_30bins.npy"
        )[np.newaxis][..., ells_mask, :]
        fiducial_ds_cross_nu = np.load(
            DATA_DIR / f"ds_cross_multipoles_{split}split_fiducial_za_30bins.npy"
        )[np.newaxis][..., ells_mask, :]
    if "auto" in corr_type:
        if split == "r" and ells == (2,):
            ells_mask = [False, False]
        elif split == "r" and 2 in ells:
            ells_mask = [True, False]
        ds_auto = np.load(DATA_DIR / f"ds_auto_multipoles_{split}split_30bins.npy")[
            ..., ells_mask, :
        ]
        fiducial_ds_auto = np.load(
            DATA_DIR / f"ds_auto_multipoles_{split}split_fiducial_30bins.npy"
        )[..., ells_mask, :]

        ds_auto_nu = np.load(
            DATA_DIR / f"ds_auto_multipoles_{split}split_neutrinos_30bins.npy"
        )[np.newaxis][..., ells_mask, :]
        fiducial_ds_auto_nu = np.load(
            DATA_DIR / f"ds_auto_multipoles_{split}split_fiducial_za_30bins.npy"
        )[np.newaxis][..., ells_mask, :]
    if "cross" in corr_type and "auto" in corr_type:
        ds = np.concatenate((ds_auto, ds_cross), axis=-2)
        fiducial_ds = np.concatenate((fiducial_ds_auto, fiducial_ds_cross), axis=-2)
        ds_nu = np.concatenate((ds_auto_nu, ds_cross_nu), axis=-2)
        fiducial_ds_nu = np.concatenate(
            (fiducial_ds_auto_nu, fiducial_ds_cross_nu), axis=-2
        )
    elif "cross" in corr_type:
        ds = ds_cross
        fiducial_ds = fiducial_ds_cross
        ds_nu = ds_cross_nu
        fiducial_ds_nu = fiducial_ds_cross_nu
    elif "auto" in corr_type:
        ds = ds_auto
        fiducial_ds = fiducial_ds_auto
        ds_nu = ds_auto_nu
        fiducial_ds_nu = fiducial_ds_auto_nu
    ds = ds[..., quantile_mask, :, :][..., s_mask]
    fiducial_ds = fiducial_ds[..., quantile_mask, :, :][..., s_mask]
    ds_nu = ds_nu[..., quantile_mask, :, :][..., s_mask]
    fiducial_ds_nu = fiducial_ds_nu[..., quantile_mask, :, :][..., s_mask]
    return compute_fisher_for_summary(
        summary=ds.reshape(*ds.shape[:3], -1),
        fiducial_summary=fiducial_ds.reshape(fiducial_ds.shape[0], -1),
        summary_positive=ds_nu.reshape(*ds_nu.shape[:3], -1),
        fiducial_summary_positive=fiducial_ds_nu.reshape(*fiducial_ds_nu.shape[:2], -1),
        correction=correction,
        return_ingredients=return_ingredients,
    )
