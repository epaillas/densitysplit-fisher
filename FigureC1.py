import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.style.use(["science.mplstyle", "bright.mplstyle"])


def compute_mean_cov(summaries):
    mean = np.mean(summaries, axis=0)
    covariance = np.cov(summaries.T)
    inverse_covariance = np.linalg.solve(
        covariance, np.eye(len(covariance), len(covariance))
    )
    return mean, covariance, inverse_covariance


def compute_xi2(vector, mean, inverse_covariance):
    return (vector - mean) @ inverse_covariance @ (vector - mean).T


def sample_from_multigaussian(mean, covariance, n_samples):
    return np.random.multivariate_normal(mean, covariance, size=n_samples)


def plot_gaussianity(summaries, ax):
    mean, covariance, inverse_covariance = compute_mean_cov(summaries)

    generated_fiducial = sample_from_multigaussian(
        mean,
        covariance,
        n_samples=len(summaries),
    )

    xi2_data = [compute_xi2(fid, mean, inverse_covariance) for fid in summaries]

    xi2_random = [
        compute_xi2(fid, mean, inverse_covariance) for fid in generated_fiducial
    ]
    dof = summaries.shape[1]
    x = np.linspace(np.min(xi2_data), np.max(xi2_data))
    bins = ax.hist(
        xi2_data, bins=60, density=True, label="Data", alpha=0.3, edgecolor=None
    )
    ax.hist(
        xi2_random,
        bins=bins[1],
        density=True,
        alpha=0.3,
        label="Gaussian",
        edgecolor=None,
    )
    ax.plot(x, stats.chi2.pdf(x, dof), label=r"$\chi^2$", color="gray", linewidth=2)


if __name__ == "__main__":
    DATA_DIR = Path("data/multipoles/")

    r = np.arange(2.5, 150, 5.0)
    r_min = 10.0
    r_max = 150.0
    r_mask = (r >= r_min) & (r <= r_max)
    auto_zsplit = np.load(DATA_DIR / "ds_auto_multipoles_zsplit_fiducial_30bins.npy")[
        ..., r_mask
    ]
    cross_zsplit = np.load(DATA_DIR / "ds_cross_multipoles_zsplit_fiducial_30bins.npy")[
        ..., r_mask
    ]
    zsplit = np.concatenate((auto_zsplit, cross_zsplit), axis=1)

    auto_rsplit = np.load(DATA_DIR / "ds_auto_multipoles_rsplit_fiducial_30bins.npy")[
        ..., r_mask
    ]
    cross_rsplit = np.load(DATA_DIR / "ds_cross_multipoles_rsplit_fiducial_30bins.npy")[
        ..., r_mask
    ]
    rsplit = np.concatenate((auto_rsplit, cross_rsplit), axis=1)

    tpcf = np.load(DATA_DIR / "tpcf_multipoles_fiducial_30bins.npy")[..., r_mask]
    fig, ax = plt.subplots(ncols=3, figsize=(16, 3.5))

    plot_gaussianity(tpcf.reshape(len(tpcf), -1), ax[0])
    plot_gaussianity(rsplit.reshape(len(rsplit), -1), ax[1])
    plot_gaussianity(zsplit.reshape(len(zsplit), -1), ax[2])

    ax[0].legend(
        fontsize=16,
    )
    ax[0].set_xlabel(r"$\chi^2$", fontsize=22)
    ax[1].set_xlabel(r"$\chi^2$", fontsize=22)
    ax[2].set_xlabel(r"$\chi^2$", fontsize=22)

    ax[0].set_ylabel("PDF", fontsize=22)
    ax[0].set_title("2PCF", fontsize=22)

    ax[1].set_title(
        r"$\mathrm{DS}^\mathrm{qq+qh}$ ($\it{r}$-split)",
        fontsize=22,
    )
    ax[2].set_title(
        r"$\mathrm{DS}^\mathrm{qq+qh}$ ($\it{z}$-split)",
        fontsize=22,
    )
    ax[0].tick_params(axis="both", which="major", labelsize=18)
    ax[1].tick_params(axis="both", which="major", labelsize=18)
    ax[2].tick_params(axis="both", which="major", labelsize=18)

    plt.tight_layout()

    plt.savefig("paper_figures/pdf/gaussianity_likelihood.pdf")
    plt.savefig("paper_figures/png/gaussianity_likelihood.png", dpi=300)
