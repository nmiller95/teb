# Print results as tables in LaTeX format
#
# Run this script in the same directory as you used to run teb, e.g. 
# $ python teb.py
# $ python scripts/make_latex_tables.py

# Output FITS file from teb
fits_file = 'output/BEBOP-3_final.fits'  

# Include estimated magnitude of star 2 in table of magnitudes?
include_m2 = True

# Include flux ratio in table of magnitudes?
include_lrat = True
# Print flux ratios as percent, or not?
lrat_percent = True

# Use full SVO FPS filter names for table of magnitudes? If False, the
# tag name for each magnitude will be used. 
#full_names = True
full_names = False

# Include warning about possible systematic error in Teff2?
# This should be included for faint M-dwarf companions where only one
# or two direct flux ratio measurements are available.
include_teff2_warning = True

#--------------------------

import numpy as np
from astropy.table import Table
import textwrap
from astropy.io import fits
from uncertainties import ufloat

chain = Table.read(fits_file, hdu='EMCEE_CHAIN')
star_name = chain.meta['STARNAME']
run_id = chain.meta['RUN_ID']

try:
    colors = Table.read(fits_file, hdu='FLUX_COLORS')
except KeyError:
    colors = False

# Magnitudes
print()
if colors:
    print('%% Observed and computed magnitudes and flux ratios %%')
else:
    print('%% Observed and computed magnitudes, colors and flux ratios %%')

print()
print(r'\begin{table*}')

if colors:
    caption = r'\caption{Observed magnitudes and flux ratios for '
else:
    caption = r'\caption{Observed magnitudes, colours and flux ratios for '
caption += star_name
caption += r'''
and predicted values based on our synthetic photometry.
The predicted magnitudes are shown with error estimates from the uncertainty on
the zero-points for each photometric system. 
The pivot wavelength for each band pass is shown in the column headed 
$\lambda_{\rm pivot}$.
'''
if full_names:
    caption += '''Band names are taken from the Spanish Virtual Observatory
Filter Profile Service.'''
if include_m2:
    caption += r'''
The estimated apparent magnitudes for each star are shown given in the
columns headed $m_1$ and headed $m_2$.
'''
else:
    caption += r'''
The estimated apparent magnitudes of the primary star corrected for the
contribution from the secondary star are shown in the column headed $m_1$.
'''

if include_lrat:
    caption += r'''
The flux ratio in each band is shown in the final column.
'''
caption += '}'
print(textwrap.fill(caption,128))

print(r'\label{tab:mags}')
print(r'\centering')

l = r'\begin{tabular}{@{}lrrrrr'
if include_m2: l += 'r'
if include_lrat: l += 'r'
l += r'}'
print(l)
print('\hline')
h = r'Band &  $\lambda_{\rm pivot}$ [nm]& \multicolumn{1}{c}{Observed} &'
h += r'\multicolumn{1}{c}{Computed} & \multicolumn{1}{c}{$\rm O-\rm C$} &'
h += r'\multicolumn{1}{c}{$m_1$} '
if include_m2: h += r' & \multicolumn{1}{c}{$m_2$} '
if include_lrat:
    if lrat_percent:
        h += r' & \multicolumn{1}{c}{$\ell$} [\%] '
    else:
        h += r' & \multicolumn{1}{c}{$\ell$} '
h += r'\\'
print(textwrap.fill(h, 98))

mags = Table.read(fits_file, hdu='MAGNITUDES')
mags.sort('Pivot')

if full_names:
    cn = 'Filter'
else:
    cn = 'Tag'
w = max([len(s) for s in mags[cn]])


print(r'\hline')
print(r'\noalign{\smallskip}')

for row in mags:
    fn = row[cn].replace('_',r'\_')
    l = f'{fn:{w}} & '
    l += f'{row["Pivot"]/10:7.1f} & '
    l += f'${row["obs"]:6.3f}\pm {row["e_obs"]:5.3f} $& '
    l += f'${row["syn"]:6.3f}\pm {row["e_syn"]:5.3f} $& '
    r = ufloat(row["obs"],row["e_obs"]) - ufloat(row["syn"],row["e_syn"])
    l += f'${r.n:+6.3f} \pm {r.s:5.3f} $& '
    l += f'${row["mag1"]:6.3f}\pm {row["e_mag1"]:5.3f} $'
    if include_m2:
        l += f' & ${row["mag2"]:6.3f}\pm {row["e_mag2"]:5.3f} $'
    if include_lrat:
        lrat = 10**(0.4*(row["mag1"]-row["mag2"]))
        if lrat_percent:
            l += f' & {100*lrat:5.2f} '
        else:
            l += f' & {lrat:6.3f} '
    l += r'\\'

    print(l)

print(r'\noalign{\smallskip}')
if lrat_percent:
    print(r'\multicolumn{5}{@{}l}{Flux ratios [\%]} \\')
    s = 100
else:
    print(r'\multicolumn{5}{@{}l}{Flux ratios} \\')
    s = 1
print(r'\noalign{\smallskip}')

flux_ratios = Table.read(fits_file, hdu='FLUX_RATIOS')
flux_ratios.sort('Pivot')
for row in flux_ratios:
    l = f'{row[cn]:{w}} & '
    l += f'{row["Pivot"]/10:7.1f} & '
    l += f'${s*row["obs"]:6.3f} \pm {s*row["e_obs"]:5.3f} $& '
    l += f'${s*row["syn"]:6.3f} $& '
    r = s*(ufloat(row["obs"],row["e_obs"]) - row["syn"])
    l += f'${r.n:+6.3f} \pm {r.s:5.3f} $ '
    l += r'\\'
    print(l)

if colors:
    print(r'\noalign{\smallskip}')
    print(r'\multicolumn{5}{@{}l}{Colours} \\')
    print(r'\noalign{\smallskip}')
    print(r' TO DO  \\')

print(r'''\noalign{\smallskip}
\hline
\end{tabular}
\end{table*}

''')

# Input and posterior parameter mean +/- stderr
print('')
print('%% emcee chain parameter mean and standard error %%')
print('')
print(r'\begin{table}')

caption = r'\caption{Results from our analysis to measure the effective'
caption += fr' temperatures for both stars in {star_name}.'
caption += r' The output parameter values are calculated using the mean and '
caption += r'standard error of the posterior probability distribution sampled '
caption += r'using {\tt emcee}. '
if include_teff2_warning:
    caption += r'Note that the results for stars 2 are very dependent on the '
    caption += r'flux ratio priors calculated from our assumed colour\,--\,T'
    caption += r'$_{\rm eff}$ relations, and so may be subject to additional '
    caption += r'systematic error.'
caption += '}'
print(textwrap.fill(caption, 96))
print(r'''\label{tab:teb}
\centering
\begin{tabular}{@{}lr} 
\hline
Parameter & \multicolumn{1}{l}{Value} \\ 
\hline
\noalign{\smallskip}
\multicolumn{2}{@{}l}{Priors} \\''')
m = chain.meta
print(fr'$\theta_1$ [mas] & ${m["THETA1"]:0.5f} \pm {m["E_THETA1"]:0.5f}$ \\')
print(fr'$\theta_2$ [mas] & ${m["THETA2"]:0.5f} \pm {m["E_THETA2"]:0.5f}$ \\')
print(fr'E(B$-$V) & ${m["EBV"]:0.3f} \pm {m["E_EBV"]:0.3f} $ \\')
print(fr'$\sigma_{{\rm m}}$& {m["SM_PRIOR"]} \\')
print(fr'$\sigma_{{\rm r}}$& {m["SR_PRIOR"]} \\')
if colors:
    print(fr'$\sigma_{{\rm c}}$& {m["SC_PRIOR"]} \\')
print(r'\noalign{\smallskip}')
print(r'\multicolumn{2}{@{}l}{Model parameters} \\')
t = chain["teff1"]
l = fr'$T_{{\rm eff,1}}$ [K] & ${t.mean():.0f} \pm {t.std():.0f}$ (rnd) '
l += fr'$\pm {m["TEFFSYS1"]:.0f}$ (sys) \\'
print(l)
t = chain["teff2"]
l = fr'$T_{{\rm eff,2}}$ [K] & ${t.mean():.0f} \pm {t.std():.0f}$ (rnd) '
l += fr'$\pm {m["TEFFSYS2"]:.0f}$ (sys) \\'
print(l)
t = chain["theta_1"]
err = t.std()
ndp = int(1 - min(0, np.floor((np.log10(err)))))
l = fr'$\theta_1$ [mas] & ${t.mean():.{ndp}f} \pm {err:.{ndp}f} $ \\'
print(l)
t = chain["theta_2"]
err = t.std()
ndp = int(1 - min(0, np.floor((np.log10(err)))))
l = fr'$\theta_2$ [mas] & ${t.mean():.{ndp}f} \pm {err:.{ndp}f} $ \\'
print(l)
t = chain['E(B-V)']
print(fr'E(B$-$V) &$ {t.mean():0.3f} \pm {t.std():0.3f} $ \\')
t = chain['sigma_m']
print(fr'$\sigma_{{\rm m}} $&$ {t.mean():0.3f} \pm {t.std():0.3f} $ \\')
t = chain['sigma_r']
print(fr'$\sigma_{{\rm r}} $&$ {t.mean():0.3f} \pm {t.std():0.3f} $ \\')
if colors:
    t = chain['sigma_c']
    print(fr'$\sigma_{{\rm c}} $&$ {t.mean():0.3f} \pm {t.std():0.3f} $ \\')
for j in range(1,m['N_COEFFS']+1):
    for i in [1,2]:
        t = chain[f'c_{i},{j}']
        print(fr'$c_{{{i},{j}}} $&$ {t.mean():6.3f} \pm {t.std():0.3f} $ \\')
print(r'\noalign{\smallskip}')
print(r'\multicolumn{2}{@{}l}{Derived parameters} \\')
t = chain["Fbol_1"]
val = t.mean()
err = t.std()
logscale = np.floor(-np.log10(val))
val *= 10**logscale
err *= 10**logscale
ndp = int(1 - min(0, np.floor((np.log10(err)))))
l = r'${\mathcal F}_{\oplus,1}$'
l += fr'[$10^{{{-logscale:0.0f}}}$'
l += r' erg\,cm$^{-2}$\,s$^{-1}$]'
l += fr' & ${val:.{ndp}f} \pm {err:.{ndp}f}$ (rnd) '
l += fr'$\pm {m["FBOLSYS1"]*10**logscale:.{ndp}f}$ (sys)  \\'
print(l)
t = chain["Fbol_2"]
val = t.mean()
err = t.std()
logscale = np.floor(-np.log10(val))
val *= 10**logscale
err *= 10**logscale
ndp = int(1 - min(0, np.floor((np.log10(err)))))
l = r'${\mathcal F}_{\oplus,2}$'
l += fr'[$10^{{{-logscale:0.0f}}}$'
l += r' erg\,cm$^{-2}$\,s$^{-1}$]'
l += fr' & ${val:.{ndp}f} \pm {err:.{ndp}f}$ (rnd) '
l += fr'$\pm {m["FBOLSYS2"]*10**logscale:.{ndp}f}$ (sys)  \\'
print(l)
t = chain["logL_1"]
val = t.mean()
err = t.std()
ndp = int(1 - min(0, np.floor((np.log10(err)))))
l = r'$\log(L_1/L_{\odot})$ '
l += fr' & ${val:.{ndp}f} \pm {err:.{ndp}f}$ (rnd) '
l += fr'$\pm {m["LOGLSYS1"]:.{ndp}f}$ (sys)  \\'
print(l)
t = chain["logL_2"]
val = t.mean()
err = t.std()
ndp = int(1 - min(0, np.floor((np.log10(err)))))
l = r'$\log(L_2/L_{\odot})$ '
l += fr' & ${val:.{ndp}f} \pm {err:.{ndp}f}$ (rnd) '
l += fr'$\pm {m["LOGLSYS2"]:.{ndp}f}$ (sys)  \\'
print(l)
print(r'''\noalign{\smallskip}
\hline
\end{tabular}
\end{table}

''')
