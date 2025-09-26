# Corner plot of parameter values - https://corner.readthedocs.io/en/latest/
#
# Run this script in the same directory as you used to run teb, e.g. 
# $ python teb.py
# $ python scripts/corner_plot.py

# Output FITS file from teb
fits_file = 'output/BEBOP-3_final.fits'  

# List of parameters to include in plot, or set to "all", e.g. 
# parameter_list = ['teff1', 'teff2', 'E(B-V)', 'c_1,1', 'c_2,1']
# Can include the derived parameters Fbol_1, Fbol_2, logL_1, logL_2
parameter_list = ['teff1', 'teff2', 'E(B-V)']

# Dictionary of keywords to pass to corner.corner
# See https://corner.readthedocs.io/en/latest/api/
# e.g.
#   kwargs = {'show_titles':True}
# or, to use default keywords for everything, 
#   kwargs = {}
kwargs = {'show_titles':True}

# Figure size. Set to None to have size calculated automatically, e.g. 
#  fig_size = (8,8)
fig_size = (8,8)

#--------------------------

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from corner import corner
import os


chain = Table.read(fits_file, hdu='EMCEE_CHAIN')

if parameter_list == 'all':
    parameter_list = chain.colnames
    parameter_list.pop(parameter_list.index('walker'))
    parameter_list.pop(parameter_list.index('step'))
    parameter_list.pop(parameter_list.index('log_prob'))

npar = len(parameter_list)

X = np.zeros([len(chain),npar])
walker = chain['walker']
labels = []
for i,p in enumerate(parameter_list):
    X[:,i] = chain[p]
    if p == 'teff1':
        labels.append(r'T$_{\rm eff,1}$')
    elif p == 'teff2':
        labels.append(r'T$_{\rm eff,2}$')
    elif p == 'theta_1':
        labels.append(r'$\theta_1$')
    elif p == 'theta_2':
        labels.append(r'$\theta_2$')
    elif p == 'E(B-V)':
        labels.append('E(B$-$V)')
    elif p == 'sigma_r':
        labels.append(r'$\sigma_r$')
    elif p == 'sigma_c':
        labels.append(r'$\sigma_c$')
    elif p == 'sigma_m':
        labels.append(r'$\sigma_m$')
    elif p == 'log_prob':
        labels.append(r'$\log$(p)')
    else:
        labels.append('$c_{'+p[2:]+'}$')
kwargs['labels'] = labels
if fig_size is None:
    fig = corner(X, **kwargs)
else:
    fig = plt.figure(figsize = fig_size)
    corner(X, fig=fig, **kwargs)
fig.tight_layout()

plt.show()

star_name = chain.meta['STARNAME']
run_id = chain.meta['RUN_ID']
plot_file = f'{star_name}_{run_id}_corner.png'.replace(' ','_')
plot_path = os.path.join('output', plot_file)
fig.savefig(plot_path, dpi=300)
print('Corner plot saved to ',plot_path)

