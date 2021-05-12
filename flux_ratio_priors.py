from astropy.table import Table, join
import numpy as np
from astropy import units as u
from astroquery.xmatch import XMatch
from scipy.stats.mstats import theilslopes
from scipy.optimize import minimize
from uncertainties import ufloat
import yaml
import warnings
warnings.filterwarnings('ignore')


def configure():
    """
    Reads the GCS III and WISE data, applies cuts based on config file

    :return: Reference temperatures for primary and secondary star, cross-matched tables
        corresponding to primary and secondary stars based on the ranges specified in the config file
    """
    t_hdu1 = Table.read('GCS3_WISE.fits', hdu=1)
    t_hdu2 = Table.read('GCS3_WISE.fits', hdu=2)
    t = join(t_hdu1, t_hdu2[(t_hdu2['HIP'] > 0)], 'HIP')

    stream = open('config/flux_ratio_priors.yaml', 'r')
    constraints = yaml.safe_load(stream)

    qual = t['l'] == 0
    ebv = (t['E(B-V)'] > constraints['E(B-V)'][0]) & (t['E(B-V)'] < constraints['E(B-V)'][1])
    logg1 = (t['logg'] > constraints['logg1'][0]) & (t['logg'] < constraints['logg1'][1])
    logg2 = (t['logg'] > constraints['logg2'][0]) & (t['logg'] < constraints['logg2'][1])
    teff1 = (t['Teff'] > constraints['Tref1'][0]) & (t['Teff'] < constraints['Tref1'][1])
    teff2 = (t['Teff'] > constraints['Tref2'][0]) & (t['Teff'] < constraints['Tref2'][1])

    t1 = t[qual & ebv & logg1 & teff1]
    t2 = t[qual & ebv & logg2 & teff2]

    table1 = XMatch.query(cat1=t1, cat2='vizier:II/311/wise',
                          max_distance=6 * u.arcsec, colRA1='RAJ2000', colDec1='DEJ2000',
                          colRA2='RAJ2000', colDec2='DEJ2000')
    table2 = XMatch.query(cat1=t2, cat2='vizier:II/311/wise',
                          max_distance=6 * u.arcsec, colRA1='RAJ2000', colDec1='DEJ2000',
                          colRA2='RAJ2000', colDec2='DEJ2000')

    Tref1 = np.mean(np.array(constraints['Tref1']))
    Tref2 = np.mean(np.array(constraints['Tref2']))
    method = constraints['method'][0]
    flux_ratio = constraints['flux_ratio'][0]
    Teff1 = constraints['Teff1'][0]
    Teff2 = constraints['Teff2'][0]
    return Tref1, Tref2, table1, table2, method, flux_ratio, Teff1, Teff2


def mad(p, x, y):
    """Mean absolute deviation of the residuals"""
    return np.mean(abs(y - p[0] - p[1] * x - p[2] * x ** 2))


def fitcol(band, Tbl, Tref=5777, method='quad'):
    """
    Fits sample of GCS-WISE stars with either polynomial or linear relation

    :param band: Which photometric band to fit over, e.g. 'J'
    :param Tbl: Table containing GCS+WISE data
    :param Tref: Float, reference temperature in K. Default is nominal solar Teff.
    :param method: Type of fit to use. Accepts 'lin' and 'quad' only.
    :type Tbl: astropy.table.Table

    :return: Coefficients and rms of the fit
    """
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
        print("Invalid method specified. Use 'quad' or 'lin'.")


def frp_coeffs(Tref1, Tref2, table1, table2, method='quad'):
    """
    Calculates coefficients for flux ratio prior calculation.

    :param Tref1: Reference temperature in K for the primary star
    :param Tref2: Reference temperature in K for the secondary star
    :param table1: Table containing GCS-WISE data in Teff range of primary star
    :param table2: Table containing GCS-WISE data in Teff range of secondary star
    :param method: Type of fit to use. Accepts 'lin' and 'quad' only.
    :type table1: astropy.table.Table
    :type table2: astropy.table.Table

    :return: Dictionary of coefficients to use in generation of flux ratio priors
    """
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
        print("Invalid method specified. Use 'quad' or 'lin'.")
    return data


def flux_ratio_priors(Vrat, Teff1, Teff2, Tref1, Tref2, coeffs, method='quad'):
    """
    Calculates a predicted flux ratio for your star in each of the 2MASS and WISE bands

    :param Vrat: Flux ratio of your star in the V band
    :param Teff1: Temperature of primary star in K
    :param Teff2: Temperature of secondary star in K
    :param Tref1: Reference temperature used in generation of coefficients for primary
    :param Tref2: Reference temperature used in generation of coefficients for secondary
    :param coeffs: Dictionary of coefficients calculated using frp_coeffs
    :param method: Type of fit to use. Accepts 'lin' and 'quad' only.

    :return: Dictionary of flux ratio priors for 2MASS and WISE bands
    """
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
        print("Invalid method specified. Use 'quad' or 'lin'.")
