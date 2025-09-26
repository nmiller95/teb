# Plot of parameter value v. step number to check convergence
#
# Run this script in the same directory as you used to run teb, e.g. 
# $ python teb.py
# $ python scripts/trail_plot.py
 
# Output FITS file from teb
fits_file = 'output/BEBOP-3_final.fits'  

# List of parameters to include in plot, or set to "all", e.g. 
parameter_list = ['teff1', 'teff2', 'E(B-V)',  'log_prob']
#parameter_list = 'all'

# Figure size - each parameter is plotted in a panel of height panel_height
fig_width = 8
panel_height = 0.8

# Opacity: 0 (transparent) to 1 (solid)
alpha = 0.2

# Colour for plot values, or None to assign random colors to each walker, e.g. 
# color = 'blue'
color = None

#--------------------------

import matplotlib.pyplot as plt
from astropy.table import Table
import os

chain = Table.read(fits_file, hdu='EMCEE_CHAIN')
step = chain['step']
walker = chain['walker']

if parameter_list == 'all':
    parameter_list  = chain.colnames
    parameter_list.pop(parameter_list.index('walker'))
    parameter_list.pop(parameter_list.index('step'))

npar = len(parameter_list)
fig,axes = plt.subplots(npar, figsize=(fig_width, panel_height*npar),
                        sharex=True)
axes[0].set_xlim(0, max(step))

for i,p in enumerate(parameter_list):
    ax = axes[i]
    for w in set(walker):
        j = walker == w
        ax.plot(step[j],chain[p][j],alpha=alpha,color=color)
    if p == 'teff1':
        ax.set_ylabel(r'T$_{\rm eff,1}$')
    elif p == 'teff2':
        ax.set_ylabel(r'T$_{\rm eff,2}$')
    elif p == 'theta_1':
        ax.set_ylabel(r'$\theta_1$')
    elif p == 'theta_2':
        ax.set_ylabel(r'$\theta_2$')
    elif p == 'E(B-V)':
        ax.set_ylabel('E(B$-$V)')
    elif p == 'sigma_r':
        ax.set_ylabel(r'$\sigma_c$')
    elif p == 'sigma_c':
        ax.set_ylabel(r'$\sigma_m$')
    elif p == 'sigma_m':
        ax.set_ylabel(r'$\sigma_r$')
    elif p == 'log_prob':
        ax.set_ylabel(r'$\log$(p)')
    else:
        ax.set_ylabel('$c_{'+p[2:]+'}$')
    ax.yaxis.set_label_coords(-0.1, 0.5)
fig.tight_layout()

plt.show()
star_name = chain.meta['STARNAME']
run_id = chain.meta['RUN_ID']
plot_file = f'{star_name}_{run_id}_trail.png'.replace(' ','_')
plot_path = os.path.join('output', plot_file)
fig.savefig(plot_path, dpi=300)
print('Trail plot saved to ',plot_path)
