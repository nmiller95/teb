from astropy.table import Table, join
import numpy as np
from astropy import units as u
from astroquery.xmatch import XMatch
from scipy.stats.mstats import theilslopes
from scipy.optimize import minimize

t_hdu1 = Table.read('GCS3_WISE.fits', hdu=1)
t_hdu2 = Table.read('GCS3_WISE.fits', hdu=2)
t = join(t_hdu1, t_hdu2[(t_hdu2['HIP'] > 0)], 'HIP')

ebv_range = (-1, 0.01)
logg1_range = (3.5, 4.5)
logg2_range = (3.8, 4.8)
teff1_range = (5400, 6600)
teff2_range = (5300, 6500)

qual = t['l'] == 0
ebv = (t['E(B-V)'] > ebv_range[0]) & (t['E(B-V)'] < ebv_range[1])
logg1 = (t['logg'] > logg1_range[0]) & (t['logg'] < logg1_range[1])
logg2 = (t['logg'] > logg2_range[0]) & (t['logg'] < logg2_range[1])
teff1 = (t['Teff'] > teff1_range[0]) & (t['Teff'] < teff1_range[1])
teff2 = (t['Teff'] > teff2_range[0]) & (t['Teff'] < teff2_range[1])

t1 = t[qual & ebv & logg1 & teff1]
t2 = t[qual & ebv & logg2 & teff2]

table1 = XMatch.query(cat1=t1, cat2='vizier:II/311/wise',
                      max_distance=6 * u.arcsec, colRA1='RAJ2000', colDec1='DEJ2000',
                      colRA2='RAJ2000', colDec2='DEJ2000')
table2 = XMatch.query(cat1=t2, cat2='vizier:II/311/wise',
                      max_distance=6 * u.arcsec, colRA1='RAJ2000', colDec1='DEJ2000',
                      colRA2='RAJ2000', colDec2='DEJ2000')


def mad(p, x, y):
    """Mean absolute deviation of the residuals"""
    return np.mean(abs(y - p[0] - p[1] * x - p[2] * x ** 2))


def fitcol(band, Tbl, Tref=5777):
    # dictionary of R values from Yuan et al., MNRAS 430, 2188–2199 (2013)
    # extrapolated/guessed for w3 and w4 - negligible for E(B-V)<0.05 anyway
    R = {'Jmag': 0.72, 'Hmag': 0.46, 'Kmag': 0.306,
         'W1mag': 0.18, 'W2mag': 0.16, 'W3mag': 0.12, 'W4mag': 0.06}
    x = (Tbl['Teff'] - Tref) / 1000
    V0 = Tbl['Vmag1'] - 3.1 * Tbl['E(B-V)']
    m0 = Tbl[band] - R[band] * Tbl['E(B-V)']
    y = V0 - m0
    i = np.isfinite(x) & np.isfinite(y)
    x = x[i]
    y = y[i]
    p = theilslopes(y, x)  # starting values for optimiser
    p = np.array([p[1], p[0], 0])
    result = minimize(mad, p, args=(x, y), method='Nelder-Mead')
    p = result.x
    y0 = p[0] + p[1] * x + p[2] * x ** 2
    res = abs(y - y0)
    med = np.median(np.abs(res))  # warning comes from this line
    i = np.abs(res) < 5 * med
    rms = res[i].std()
    return p[0], p[1], p[2], result.fun
