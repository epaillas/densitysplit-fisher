import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def get_quantiles_by_idx(density, n_quantiles=5):
    idx = np.argsort(density)
    n_centres = len(density)
    quantiles = {}
    for i in range(1, n_quantiles+1):
        quantiles[i] = idx[int((i-1)*n_centres/n_quantiles):int(i*n_centres/n_quantiles)]
    return quantiles

def get_quantile_by_centre(quantiles_dict, n_centres):
    quantile_by_centre = np.zeros(n_centres)
    for key, value in quantiles_dict.items():
        for v in value:
            quantile_by_centre[v] = key
    return quantile_by_centre

if __name__ == '__main__':
    delta_r = np.load('data/density/delta_r.npy')
    delta_s = np.load('data/density/delta_s.npy')
    n_spheres = 1_000_000

    quantiles_r = get_quantiles_by_idx(delta_r,5)

    quantiles_s = get_quantiles_by_idx(delta_s,5)

    centres_r = get_quantile_by_centre(quantiles_r,n_spheres)
    centres_z = get_quantile_by_centre(quantiles_s,n_spheres)

    cmap = plt.cm.RdBu_r
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    cm = confusion_matrix(
        centres_r, 
        centres_z, 
        normalize='true'
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[rf'$\mathrm{{DS}}_{ds}$' for ds in np.arange(1,6)],
    )
    disp.plot(cmap='Blues',values_format='.2f')
    disp.ax_.set_xlabel('Real Space Quantile')
    disp.ax_.set_ylabel('Redshift Space Quantile')

    plt.savefig(f'paper_figures/pdf/confusion_r_s.pdf')  
    plt.savefig(f'paper_figures/png/confusion_r_s.png', dpi=300)  