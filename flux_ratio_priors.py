from astropy.table import Table, join
import numpy as np
from astropy import units as u
from astroquery.xmatch import XMatch
from scipy.stats.mstats import theilslopes
from scipy.optimize import minimize
from uncertainties import ufloat

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


def fitcol(band, Tbl, Tref=5777, method='quad'):
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

    if method == 'lin':
        y0 = p[1] + p[0] * x
        res = abs(y - y0)
        med = np.median(np.abs(res))
        i = np.abs(res) < 5 * med
        rms = res[i].std()
        return p[1], p[0], rms
    elif method == 'quad':
        p = np.array([p[1], p[0], 0])
        result = minimize(mad, p, args=(x, y), method='Nelder-Mead')
        p = result.x
        y0 = p[0] + p[1] * x + p[2] * x ** 2
        res = abs(y - y0)
        med = np.median(np.abs(res))  # warning comes from this line
        i = np.abs(res) < 5 * med
        rms = res[i].std()
        return p[0], p[1], p[2], result.fun
    else:
        print('incorrect')


def frp_coeffs(Tref1, Tref2, table1, table2, method='quad'):
    data = {}
    bands = ['Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag', 'W4mag']
    tags = ['J', 'H', 'Ks', 'W1', 'W2', 'W3', 'W4']

    if method == 'lin':
        for band, tag in zip(bands, tags):
            c1, m1, r1 = fitcol(band, table1, Tref1, method='lin')
            c2, m2, r2 = fitcol(band, table2, Tref2, method='lin')
            data[tag] = {'c1': c1, 'm1': m1, 'r1': r1, 'c2': c2, 'm2': m2, 'r2': r2}
    elif method == 'quad':
        for band, tag in zip(bands, tags):
            p01, p11, p21, r1 = fitcol(band, table1, Tref1, method='quad')
            p02, p12, p22, r2 = fitcol(band, table2, Tref2, method='quad')
            data[tag] = {'p01': p01, 'p11': p11, 'p21': p21, 'r1': r2,
                         'p02': p02, 'p12': p12, 'p22': p22, 'r2': r2}
    else:
        print('method must be lin or quad')
    return data


def flux_ratio_priors(Vrat, Teff1, Teff2, Tref1, Tref2, coeffs, method='quad'):
    if method == 'lin':
        # Return a dictionary of ufloat priors on flux ratios
        d = {}
        for b in coeffs.keys():
            col1 = coeffs[b]['c1'] + coeffs[b]['m1'] * (Teff1 - Tref1) / 1000.0
            col2 = coeffs[b]['c2'] + coeffs[b]['m2'] * (Teff2 - Tref2) / 1000.0
            L = Vrat * 10 ** (0.4 * (col2 - col1))
            e_L = np.hypot(coeffs[b]['r1'], coeffs[b]['r2'])
            d[b] = ufloat(L, e_L)
        return d

    elif method == 'quad':
        # Return a dictionary of ufloat priors on flux ratios
        d = {}
        for b in coeffs.keys():
            col1 = coeffs[b]['p01'] + coeffs[b]['p11'] * (Teff1 - Tref1) / 1000.0
            + coeffs[b]['p21'] * ((Teff1 - Tref1) / 1000.0) ** 2
            col2 = coeffs[b]['p02'] + coeffs[b]['p12'] * (Teff2 - Tref2) / 1000.0
            + coeffs[b]['p22'] * ((Teff2 - Tref2) / 1000.0) ** 2
            L = Vrat * 10 ** (0.4 * (col2 - col1))
            e_L = np.hypot(coeffs[b]['r1'], coeffs[b]['r2'])
            d[b] = ufloat(L, e_L)
        return d
    else:
        print('incorrect - method is lin or quad only')
