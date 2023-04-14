import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# plt.style.use('enrique-science')

delta_r = np.load('data/density/delta_r.npy')
delta_s = np.load('data/density/delta_s.npy')

# Downsample to plot
N = 50_000
g = sns.jointplot(x=delta_r[:N], y=delta_s[:N],kind="kde",xlim = (-1.2,2.), ylim = (-1.2,2.))# color="#4CB391")
g.ax_joint.plot([-1.2,2.], [-1.2,2], )
g.ax_marg_x.set_axis_off()
g.ax_marg_y.set_axis_off()
g.ax_joint.text(1.15,1.35,r'$\Delta^r = \Delta^z$',rotation=45, color='indianred', fontsize=20)

g.set_axis_labels(r'$\Delta^r$', r'$\Delta^z$', fontsize=24)
g.ax_joint.tick_params(axis='both', which='major', labelsize=22)

g.figure.savefig("output.png")

# plt.savefig(f'paper_figures/pdf/delta_r_s_joint.pdf')  
# plt.savefig(f'paper_figures/png/delta_r_s_joint.png', dpi=300)  
