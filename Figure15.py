import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from general_fisher import ds_fisher_matrix_full

plt.style.use(["science.mplstyle", "bright.mplstyle"])
from cosmology import quijote_cosmology as cosmo_dict

DATA_DIR = Path("data/multipoles")


def get_bias_in_summary(smin, smax=150.0, mode=0):
    s_bins = np.linspace(0, 150, 31)
    s = 0.5 * (s_bins[1:] + s_bins[:-1])
    s_mask = (s >= smin) & (s <= smax)

    if mode == 0 or mode == 3:
        fiducial_ds_auto = np.load(
            DATA_DIR / "ds_auto_multipoles_rsplit_fiducial_30bins.npy",
            allow_pickle=True,
        )[..., 0, :][..., np.newaxis, :]
        fiducial_ds_auto = np.delete(fiducial_ds_auto, 2, 1)
        reconstructed_fiducial_ds_auto = np.load(
            DATA_DIR / "ds_auto_multipoles_reconsplit_fiducial.npy", allow_pickle=True
        )[..., 0, :][..., np.newaxis, :]
        reconstructed_fiducial_ds_auto = np.delete(reconstructed_fiducial_ds_auto, 2, 1)
        if mode == 0:
            fiducial_ds = fiducial_ds_auto
            reconstructed_fiducial_ds = reconstructed_fiducial_ds_auto
    if mode in range(1, 4):
        fiducial_ds_cross = np.load(
            DATA_DIR / "ds_cross_multipoles_rsplit_fiducial_30bins.npy",
            allow_pickle=True,
        )
        reconstructed_fiducial_ds_cross = np.load(
            DATA_DIR / "ds_cross_multipoles_reconsplit_fiducial_30bins.npy",
            allow_pickle=True,
        )
        fiducial_ds_cross = np.delete(fiducial_ds_cross, 2, 1)
        reconstructed_fiducial_ds_cross = np.delete(reconstructed_fiducial_ds_cross, 2, 1)
        if mode == 1:
            fiducial_ds = fiducial_ds_cross[..., 0, :][..., np.newaxis, :]
            reconstructed_fiducial_ds = reconstructed_fiducial_ds_cross[..., 0, :][
                ..., np.newaxis, :
            ]
        elif mode == 2:
            fiducial_ds = fiducial_ds_cross[..., 1, :][..., np.newaxis, :]
            reconstructed_fiducial_ds = reconstructed_fiducial_ds_cross[..., 1, :][
                ..., np.newaxis, :
            ]
    if mode == 3:
        fiducial_ds = np.concatenate((fiducial_ds_auto, fiducial_ds_cross), axis=-2)
        reconstructed_fiducial_ds = np.concatenate(
            (reconstructed_fiducial_ds_auto, reconstructed_fiducial_ds_cross), axis=-2
        )
    return np.mean(reconstructed_fiducial_ds - fiducial_ds[:499], axis=0)[..., s_mask]


def get_bias_reconstruction(smin, mode=0):
    if mode == 0:
        ells = (0,)
        corr_type = ("auto",)
    if mode == 1:
        ells = (0,)
        corr_type = ("cross",)
    if mode == 2:
        ells = (2,)
        corr_type = ("cross",)
    if mode == 3:
        ells = (
            0,
            2,
        )
        corr_type = (
            "auto",
            "cross",
        )

    fisher, derivatives, precision_matrix = ds_fisher_matrix_full(
        split="r",
        return_ingredients=True,
        smin=smin,
        ells=ells,
        corr_type=corr_type,
        quantiles=[1, 2, 4, 5],
    )

    inverse_fisher = np.linalg.solve(fisher, np.eye(len(fisher), len(fisher)))
    diff = get_bias_in_summary(smin=smin, mode=mode).reshape(-1)
    bias_term = diff @ precision_matrix @ derivatives.T
    bias = (inverse_fisher * bias_term).sum(axis=-1)
    error = np.sqrt(np.diag(inverse_fisher))
    return bias, error


def errors_smin(
    nderiv=1500,
    ncov=7000,
    smin=0,
):
    params_to_plot = [
        "Om",
        "Ob2",
        "h",
        "s8",
        "ns",
        "Mnu",
    ]
    mean_theta = [cosmo_dict[i]["fiducial"] for i in params_to_plot]
    labels = [cosmo_dict[i]["label_latex"] for i in params_to_plot]
    skip_param = 6  # avoid Mmin
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["ytick.labelsize"] = 14
    s_mins = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0])

    bias_mono_qq = np.zeros((len(s_mins), len(params_to_plot)))
    bias_mono_qh = np.zeros((len(s_mins), len(params_to_plot)))
    bias_quad_qh = np.zeros((len(s_mins), len(params_to_plot)))
    bias = np.zeros((len(s_mins), len(params_to_plot)))
    error_mono_qq = np.zeros((len(s_mins), len(params_to_plot)))
    error_mono_qh = np.zeros((len(s_mins), len(params_to_plot)))
    error_quad_qh = np.zeros((len(s_mins), len(params_to_plot)))
    error = np.zeros((len(s_mins), len(params_to_plot)))

    for i, smin in enumerate(s_mins):
        bqq, eqq = get_bias_reconstruction(smin=smin, mode=0)
        bias_mono_qq[i] = np.delete(bqq, skip_param)
        error_mono_qq[i] = np.delete(eqq, skip_param)

        bqh, eqh = get_bias_reconstruction(smin=smin, mode=1)
        bias_mono_qh[i] = np.delete(bqh, skip_param)
        error_mono_qh[i] = np.delete(eqh, skip_param)

        bqh2, eqh2 = get_bias_reconstruction(smin=smin, mode=2)
        bias_quad_qh[i] = np.delete(bqh2, skip_param)
        error_quad_qh[i] = np.delete(eqh2, skip_param)

        b, q = get_bias_reconstruction(smin=smin, mode=3)
        bias[i] = np.delete(b, skip_param)
        error[i] = np.delete(q, skip_param)

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 4.5))
    for i in range(len(params_to_plot)):
        j, k = i % 3, i % 2
        ax[j, k].errorbar(
            s_mins - 2.0,
            mean_theta[i] + bias_mono_qq[:, i],
            yerr=error_mono_qq[:, i],
            marker="o",
            markersize=3,
            linestyle="",
            label=r"$\xi_0^\mathrm{qq}$",
        )

        ax[j, k].errorbar(
            s_mins - 1.0,
            mean_theta[i] + bias_mono_qh[:, i],
            yerr=error_mono_qh[:, i],
            marker="o",
            markersize=3,
            linestyle="",
            label=r"$\xi_0^\mathrm{qh}$",
        )

        ax[j, k].errorbar(
            s_mins,
            mean_theta[i] + bias[:, i],
            yerr=error[:, i],
            marker="o",
            markersize=3,
            linestyle="",
            label=r"$\xi_0^\mathrm{qq} + \xi_0^\mathrm{qh} + \xi_2^\mathrm{qh}$",
        )

        ax[j, k].axhline(
            mean_theta[i], linestyle="dashed", color="gray", alpha=0.3, linewidth=3
        )
        if j == 2:
            ax[j, k].set_xlabel(
                r"$s_{\rm min}\,\left[h^{-1}{\rm Mpc}\right]$", fontsize=16
            )
        ax[j, k].set_ylabel(labels[i], fontsize=19)

    handles, legend_labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=18,
        bbox_to_anchor=(0.52, 1.18),
        columnspacing=0.7,
        handletextpad=0.1
    )
    plt.tight_layout()

    plt.savefig(f"paper_figures/pdf/errors_recon.pdf")
    plt.savefig(f"paper_figures/png/errors_recon.png", dpi=300)


if __name__ == "__main__":
    errors_smin()
