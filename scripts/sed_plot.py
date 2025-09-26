# Plot of model spectral energy distribution and filters used. 
# Also prints a caption
#
# Run this script in the same directory as you used to run teb, e.g. 
# $ python teb.py
# $ python scripts/sed_plot.py

# Output FITS file from teb
fits_file = 'output/BEBOP-3_final.fits'  

# The top panel is always a plot of the SED for the total flux and one or
# both fluxes from the two stars. The  observed magnitudes (as fluxes) are
# plotted as points with error # bars, and the synthetic magnitudes (as fluxes)
# are plotted as open circles.

# Dictionary of keyword arguments to use when plotting best-fit total 
# flux, e.g. flux_kwargs = {'color':'darkblue', 'linewidth':3}
flux_kwargs = {'color':'navy','linewidth':1}

# Dictionary of keyword arguments to use when using plt.fill_between
# to plot +- 1-sigma total flux, e.g. 
# fill_between_kwargs = {'color':'darkblue', 'alpha':0.3}
fill_between_kwargs = {'color':'navy', 'alpha':0.3}

# Dictionary of keyword arguments to use for plotting observed fluxes and flux
# ratios, e.g.
#  obs_kwargs = {'lw':3, 'color':'blue'}
obs_kwargs = {}

# Dictionary of keyword arguments to use for plotting synthetic fluxes and
# flux ratios, e.g.
#  syn_kwargs = {'ms':8, 'color':'darkblue'}
syn_kwargs = {'ms':8, 'color':'darkblue'}

# Plot fluxes in top panel from star 1 and/or star 2 (or not)
plot_flux1 = True 
plot_flux2 = True

# Dictionary of keyword arguments to use when  plotting flux of star 1. 
flux1_kwargs = {'color':'darkgreen'}

# Dictionary of keyword arguments to use when  plotting flux of star 2. 
flux2_kwargs = {'color':'darkorange'}

# Add a panel showing the fluxes on a log scale, e.g. plot_logflux = True
plot_logflux = True

# Add a panel showing the model flux ratio and observed flux ratios.
plot_flux_ratios = False

# Dictionary of keyword arguments to send to plt.semilogx() for plotting 
# profiles used to measure magnitudes, e.g. 
#  filter_kwargs = {'color':'grey'}
filter_kwargs = {'color':'grey'}

# Height of filter profiles in plot relative to total plot height
filter_plot_height = 0.2

# Dictionary of keyword arguments to send to plt.plot() for plotting filter
# profiles used to measure magnitudes, e.g. 
#  mag_kwargs = {'ms':4, 'color':'darkblue'}
mag_kwargs = {'ms':4, 'color':'darkblue'}

# Dictionary of keyword arguments to send to plt.plot() for plotting filter
# profiles used to measure flux ratios, e.g. 
# filter_ratios_kwargs = {'color':'darkred'}
filter_ratios_kwargs = {'color':'grey'}

# Figure size - each panel of height panel_height
fig_width = 8
panel_height = 3

# gridspec keyword arguments to pass through plt.subplots, e.g. 
#  gridspec_kw={'height_ratios':[3,2]}
# or (if only plotting one panel)
#gridspec_kw={}
#gridspec_kw={'height_ratios':[3,2,2]}
gridspec_kw={'height_ratios':[3,2]}

#--------------------------

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.ticker import ScalarFormatter
from scipy.integrate import simpson
import os
import xml.etree.ElementTree as ET
from astropy.io import fits
import textwrap


def getprofile(filtername):
    fileroot = filtername.replace('/','.')
    # First look for user-defined transmission function in FITS file
    filterfile = os.path.join('cache','fps',fileroot+'.fits')
    if os.path.exists(filterfile):
        filtertable = Table.read(filterfile)
        photon = filtertable.meta['PHOTON']
    else:
        filterfile = os.path.join('cache','fps',fileroot+'.xml')
        filtertable = Table.read(filterfile)
        root = ET.parse(filterfile).getroot()
        for par in root.find('RESOURCE').find('TABLE').findall('PARAM'):
            if par.attrib['name'] ==  'DetectorType':
                DetectorType = int(par.attrib['value'])
        if DetectorType == 0:
                photon = False
        else:
                photon = True
    return filtertable, photon

fluxes = Table.read(fits_file, hdu='FLUX')
mags = Table.read(fits_file, hdu='MAGNITUDES')

nrows = 1
if plot_logflux:
    nrows += 1
if plot_flux_ratios:
    nrows += 1
fig,axes = plt.subplots(nrows=nrows, sharex=True, gridspec_kw=gridspec_kw,
                        figsize=(fig_width, nrows*panel_height))
if nrows == 1:
    ax = axes
    ax1 = ax
else:
    ax = axes[0]
    ax1 = axes[-1]

# Extract and scale wavelengths and fluxes
wave = fluxes['wave']/10
flux = fluxes['flux_best']
logscale = int(np.floor(-np.log10(max(flux))))
scale = 10**logscale
flux_1 = fluxes['flux_1_best']
flux_2 = fluxes['flux_2_best']
flux_lo = fluxes['flux_mean'] - fluxes['flux_err']
flux_hi = fluxes['flux_mean'] + fluxes['flux_err']

# SED plot in top panel
ax.set_ylim(0.0, 1.1*np.max(flux*scale))
yl = fr'Flux [10$^{{{-logscale}}}$ ergs cm$^{{-2}}$ s$^{{-1}}$ $\AA^{{-1}}$]'
ax.set_ylabel(yl)
ax.semilogx(wave, flux*scale, **flux_kwargs, zorder=4)
ax.fill_between(wave, flux_lo*scale, flux_hi*scale, **fill_between_kwargs,
                zorder=3)
if plot_flux1:
    ax.semilogx(wave, flux_1*scale, **flux1_kwargs, zorder=2)
if plot_flux2:
    ax.semilogx(wave, flux_2*scale, **flux2_kwargs, zorder=1)
fscale = scale*max(flux) # scaling factor for plotting filter profiles
fluxes = Table.read(fits_file, hdu='FLUX')
h = fits.getheader(fits_file,extname='EMCEE_CHAIN')
star_name = h['STARNAME']
run_id = h['RUN_ID']
if nrows > 1:
    caption = 'Upper panel: '
else:
    caption = ' '
caption += f'The SED of {star_name}. The best-fit SED is plotted as a line'
caption += r' and the mean SED $\pm 1-\sigma$ is plotted as a filled region.'
caption += ' The observed fluxes are plotted as points'
caption += ' with error bars and predicted fluxes for the best-fit SED'
caption += ' integrated over the response functions shown are plotted with'
caption += ' open circles. '
if plot_flux1 and plot_flux2:
    caption += ' The SEDs of the two stars are also plotted. '
elif plot_flux1:
    caption += ' The SED of star 1 is also plotted. '
elif plot_flux2:
    caption += ' The SED of star 2 is also plotted. '


# log-log flux plot
if plot_logflux:
    axl = axes[1]
    axl.loglog(wave, flux, **flux_kwargs, zorder=3)
    j = (wave > 120) & (wave < 18_000)
    ymin = np.min(flux[j])
    ymax = np.max(flux[j])
    if plot_flux1:
        axl.loglog(wave, flux_1, **flux1_kwargs, zorder=2)
        ymin = min([ymin, np.min(flux_1[j])])
        ymax = max([ymax, np.max(flux_1[j])])
    if plot_flux2:
        axl.loglog(wave, flux_2, **flux2_kwargs, zorder=1)
        ymin = min([ymin, np.min(flux_2[j])])
        ymax = max([ymax, np.max(flux_2[j])])
    yl = fr'Flux [ergs cm$^{{-2}}$ s$^{{-1}}$ $\AA^{{-1}}$]'
    axl.set_ylabel(yl)
    axl.set_ylim(0.25*ymin,4*ymax)
    fscale = ymax # scaling factor for plotting filter profiles
    if nrows > 2:
        caption += 'Middle panel: '
    else:
        caption += 'Lower panel: '
    caption += 'Same as the upper panel but with fluxes plotted on a'
    caption += ' logarithmic scale. '
    if nrows == 2:
        caption += 'Filters used to measure flux ratios are also plotted here.'

# magnitudes (as fluxes) 
for row in mags:
    ft, ph = getprofile(row['Filter'])
    xp = ft['Wavelength']/10
    yp = ft['Transmission']
    if ph:
        yp *= xp
    r = np.interp(wave, xp, yp, left=0, right=0)
    flx_syn = simpson(flux*r, x=wave)/simpson(r, x=wave)
    flx_obs = flx_syn * 10**(-0.4*(row['obs'] - row['syn']))
    flx_err = row['e_obs'] * flx_obs/1.086
    ax.errorbar(row['Pivot']/10, scale*flx_obs,
        yerr=scale*flx_err,  fmt='o', zorder=4, **mag_kwargs)
    ax.plot(row['Pivot']/10, scale*flx_syn, 'o',  fillstyle='none', zorder=5,
            **syn_kwargs)
    if plot_logflux:
        axl.errorbar(row['Pivot']/10, flx_obs,
            yerr=flx_err,  fmt='o', zorder=4, **mag_kwargs)
        axl.plot(row['Pivot']/10, flx_syn, 'o',  fillstyle='none', zorder=5,
                **syn_kwargs)


# Flux ratio plot
try:
    flux_ratios = Table.read(fits_file, hdu='FLUX_RATIOS')
except KeyError:
    flux_ratios = []
if plot_flux_ratios:
    axr = axes[-1]
    frat = flux_2/flux_1
    axr.semilogx(wave, flux_2/flux_1, **flux_kwargs, zorder=3)
    for row in flux_ratios:
        axr.errorbar(row['Pivot']/10, row['obs'], row['e_obs'], fmt='o',
                     zorder=4, **mag_kwargs)
        axr.plot(row['Pivot']/10, row['syn'], 'o', fillstyle='none', 
                 zorder=5, **syn_kwargs)
    axr.set_ylabel('Flux ratio, $f_2/f_1$')
    j = (wave > 120) & (wave < 18_000)
    axr.set_ylim(0.9*np.min(frat[j]), 1.1*np.max(frat[j]))
    fscale = np.max(frat[j]) # scaling factor for plotting filter profiles
    if nrows > 2:
        caption += 'Middle panel: '
    else:
        caption += 'Lower panel: '
    caption += 'Flux ratio as a function of wavelength for the best-fit SEDs. '
    caption += 'The observed flux ratios are plotted as points with error bars'
    caption += ' and the predicted flux ratios integrated over the filter '
    caption += ' profiles shown are plotted as open circles.'

# Filter profiles for measured magnitudes
for row in mags:
    ft, ph = getprofile(row['Filter'])
    w = ft['Wavelength']/10
    r = ft['Transmission']
    r /= np.max(r) 
    r *= np.max(flux*scale)*filter_plot_height
    ax.semilogx(w, r, **filter_kwargs)

# Filter profiles for flux ratios - always plotted in the bottom panel.
for row in flux_ratios:
        ft, ph = getprofile(row['Filter'])
        w = ft['Wavelength']/10
        r = ft['Transmission']
        r *= fscale*filter_plot_height/np.max(r)
        ax1.semilogx(w, r, **filter_ratios_kwargs)

# Set shared x axis properties
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.set_xlabel('Wavelength [nm]')
if 'WISE/WISE.W3' in mags['Filter']:
    ax1.set_xlim(120,18000)
    ax1.set_xticks([200,400,800,1600, 4000, 8000, 18000 ])
else:
    ax1.set_xlim(120,6400)
    ax1.set_xticks([200,400,800,1600, 3200, 6400])

fig.tight_layout()
print()
print(textwrap.fill(caption,78))
print()

plt.show()
plot_file = f'{star_name}_{run_id}_sed.png'.replace(' ','_')
plot_path = os.path.join('output', plot_file)
fig.savefig(plot_path, dpi=300)
print('SED plot saved to ',plot_path)

