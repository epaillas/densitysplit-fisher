import numpy as np
from densitysplit import pipeline
from pycorr import TwoPointCorrelationFunction
from astropy.io import fits
import matplotlib.pyplot as plt

plt.style.use(["science.mplstyle", "bright.mplstyle"])


def get_multipoles(
    pos, boxsize, s_bins, mu_bins, n_threads, ells=(0, 2), los="z", right_pos=None
):
    edges = (s_bins, mu_bins)

    result = TwoPointCorrelationFunction(
        "smu",
        edges,
        data_positions1=pos.T,
        data_positions2=right_pos,
        engine="corrfunc",
        nthreads=4,
        boxsize=boxsize,
        los=los,
    )
    return result(ells=ells, return_sep=False)


def get_quantile_by_centre(quantiles_dict, n_centres):
    quantile_by_centre = np.zeros(n_centres)
    for key, value in quantiles_dict.items():
        for v in value:
            quantile_by_centre[v] = key
    return quantile_by_centre


def get_quantiles_by_idx(density, n_quantiles=5):
    idx = np.argsort(density)
    n_centres = len(density)
    quantiles = {}
    for i in range(1, n_quantiles + 1):
        quantiles[i] = idx[
            int((i - 1) * n_centres / n_quantiles) : int(i * n_centres / n_quantiles)
        ]
    return quantiles


def get_contributions(centres_r, centres_z, DS, pos_spheres, s_bins, mu_bins):
    multipoles_intersection = get_multipoles(
        pos_spheres[(centres_r == DS) & (centres_z == DS)],
        boxsize=boxsize,
        s_bins=s_bins,
        mu_bins=mu_bins,
        n_threads=4,
    )
    weights_intersection = (
        len(pos_spheres[(centres_r == DS) & (centres_z == DS)])
        / len(pos_spheres[centres_r == DS])
    ) ** 2
    multipoles_added = get_multipoles(
        pos_spheres[(centres_r != DS) & (centres_z == DS)],
        boxsize=boxsize,
        s_bins=s_bins,
        mu_bins=mu_bins,
        n_threads=4,
    )
    weights_added = (
        len(pos_spheres[(centres_r != DS) & (centres_z == DS)])
        / len(pos_spheres[centres_r == DS])
    ) ** 2
    multipoles_cross = get_multipoles(
        pos_spheres[(centres_r == DS) & (centres_z == DS)],
        right_pos=pos_spheres[(centres_r != DS) & (centres_z == DS)].T,
        boxsize=boxsize,
        s_bins=s_bins,
        mu_bins=mu_bins,
        n_threads=4,
    )
    weights_cross = np.sqrt(weights_intersection) * np.sqrt(weights_added)
    return (
        weights_intersection * multipoles_intersection[1],
        weights_added * multipoles_added[1],
        weights_cross * multipoles_cross[1],
    )


def plot_split(ax, quad_s, quad_intersection, quad_added, quad_cross):
    ax.plot(s_bins_c, s_bins_c ** 2 * quad_s, label=rf"$\mathrm{{qq}}$")

    ax.plot(s_bins_c, s_bins_c ** 2 * quad_intersection, label=rf"$\rm Z \cap \rm R$")

    ax.plot(s_bins_c, s_bins_c ** 2 * quad_added, label=rf"$\rm Z \notin \rm R$")

    ax.plot(
        s_bins_c,
        s_bins_c ** 2 * 2.0 * quad_cross,
        label=rf"$\rm Z  \cap \rm R, \rm Z \notin \rm R$",
    )


def read_pos(phase):
    DATA_DIR = "/cosma/home/analyse/epaillas/data/quijote/halos/"
    fname = DATA_DIR + f"halos_fiducial_ph{phase}_z0.0.fits"
    hdul = fits.open(fname)
    data = hdul[1].data
    x = data["X"]
    y = data["Y"]
    z = data["Z"]
    z_rsd = data["Z_RSD"]
    pos = np.stack((x, y, z)).T
    pos_s = np.stack((x, y, z_rsd)).T
    return pos, pos_s


def get_autocorr_multipoles(
    quantiles,
    pos_spheres,
    boxsize: float,
    s_bins: np.array,
    mu_bins: np.array,
    n_threads: int = 4,
    ells=(0, 2,),
    los="z",
):
    edges = (s_bins, mu_bins)
    n_quantiles = len(quantiles.keys())
    xi_ells = np.zeros((n_quantiles, 2, len(s_bins) - 1,))
    for qt in range(1, n_quantiles + 1):
        pos_ds = quantiles[f"DS{qt}"]
        result = TwoPointCorrelationFunction(
            "smu",
            edges,
            data_positions1=pos_ds.T,
            engine="corrfunc",
            nthreads=4,
            boxsize=boxsize,
            los=los,
        )
        xi_ells[qt - 1] = result(ells=ells, return_sep=False)
    return xi_ells


def get_quantiles(pos_spheres: np.array, deltas: np.array, n_quantiles: int):
    return pipeline.get_quantiles(
        seeds=pos_spheres, density_pdf=deltas, nquantiles=n_quantiles,
    )


if __name__ == "__main__":
    n_spheres = 1_000_000
    boxsize = 1000.0
    s_bins = np.linspace(0.0, 150.0, 30)
    s_bins_c = 0.5 * (s_bins[1:] + s_bins[:-1])
    mu_bins = np.linspace(-1, 1, 101)
    smoothing_radius = 20.0
    n_phases = 10 
    quadrupole = {
        "DS1": {"s": [], "intersection": [], "added": [], "cross": []},
        "DS5": {"s": [], "intersection": [], "added": [], "cross": []},
    }
    for phase in range(n_phases):
        pos_spheres = pipeline.get_seeds(nseeds=n_spheres, box_size=boxsize)
        pos, pos_s = read_pos(phase=phase)
        delta_r = pipeline.get_density_pdf(
            smooth_radius=smoothing_radius,
            data_positions1=pos_spheres,
            data_weights1=np.ones(len(pos_spheres)),
            data_positions2=pos,
            data_weights2=np.ones(len(pos)),
            box_size=boxsize,
        )
        delta_s = pipeline.get_density_pdf(
            smooth_radius=smoothing_radius,
            data_positions1=pos_spheres,
            data_weights1=np.ones(len(pos_spheres)),
            data_positions2=pos_s,
            data_weights2=np.ones(len(pos)),
            box_size=boxsize,
        )

        quantiles_r = get_quantiles_by_idx(delta_r, 5)
        quantiles_s = get_quantiles_by_idx(delta_s, 5)

        pos_quantiles_r = get_quantiles(pos_spheres, delta_r, 5)
        pos_quantiles_s = get_quantiles(pos_spheres, delta_s, 5)

        centres_r = get_quantile_by_centre(quantiles_r, n_spheres)
        centres_z = get_quantile_by_centre(quantiles_s, n_spheres)
        multipoles_s = get_autocorr_multipoles(
            pos_quantiles_s,
            pos_spheres,
            boxsize=boxsize,
            s_bins=s_bins,
            mu_bins=mu_bins,
        )
        quadrupole["DS1"]["s"].append(multipoles_s[0][1])
        quadrupole["DS5"]["s"].append(multipoles_s[4][1])
        for ds in [1, 5]:
            intersection, added, cross = get_contributions(
                centres_r=centres_r,
                centres_z=centres_z,
                DS=ds,
                pos_spheres=pos_spheres,
                s_bins=s_bins,
                mu_bins=mu_bins,
            )
            quadrupole[f"DS{ds}"]["intersection"].append(intersection)
            quadrupole[f"DS{ds}"]["added"].append(added)
            quadrupole[f"DS{ds}"]["cross"].append(cross)

    # get average
    keys = ["s", "intersection", "added", "cross"]
    fig, ax = plt.subplots(figsize=(8, 4.5), ncols=2, sharey=True)
    for i, ds in enumerate([1, 5]):
        for key in keys:
            quadrupole[f"DS{ds}"][key] = np.mean(quadrupole[f"DS{ds}"][key], axis=0)
        plot_split(
            ax[i],
            quadrupole[f"DS{ds}"]["s"],
            quadrupole[f"DS{ds}"]["intersection"],
            quadrupole[f"DS{ds}"]["added"],
            quadrupole[f"DS{ds}"]["cross"],
        )
    ax[0].set_ylabel(r"$s^2 \xi_2(s)$")
    ax[0].set_xlabel("s [$h^{-1} \ \mathrm{Mpc}$]")
    ax[1].set_xlabel("s [$h^{-1} \ \mathrm{Mpc}$]")

    ax[0].legend()
    ax[0].set_title(rf"$\mathrm{{DS}}_1^\mathrm{{qq}}$")
    ax[1].set_title(rf"$\mathrm{{DS}}_5^\mathrm{{qq}}$")
    ax[0].axhline(y=0, linestyle="dashed", color="lightgray")
    ax[1].axhline(y=0, linestyle="dashed", color="lightgray")

    plt.savefig(f"paper_figures/pdf/quadrupole_contributions_ds.pdf")
    plt.savefig(f"paper_figures/png/quadrupole_contributions_ds.png")
