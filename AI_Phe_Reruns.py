import numpy as np
from matplotlib import pylab as plt
from astropy.table import Table
from scipy.integrate import simps
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from uncertainties import ufloat, covariance_matrix, correlation_matrix
from uncertainties.umath import log10
from scipy.interpolate import interp1d
import flint
import astropy.units as u
from synphot import units, ReddeningLaw
from scipy.special import legendre
import emcee
import corner
from multiprocessing import Pool
import pickle
from scipy.optimize import minimize
from response import extra_data, colors_data


class Flux2mag:

    # extra_data is a list of dictionaries, each with the following data ...
    #  'tag' - string, photometric band/system name/label
    #  'mag' - ufloat, observed magnitude with standard error
    #  'zp'  - ufloat, zero-point and standard error
    #  'wave' - array, wavelengths for response function, ** in angstrom **
    #  'resp'  - array, response function
    # Example,
    # AKARI/IRC mid-IR all-sky Survey S9W-band flux measurement of AI Phe
    # S09 = 9.796e-02 +/- 2.70e-02 J
    # Error in flux scale at S9W is given as "0.070 rms".
    # Convert flux to AB magnitude
    # >>> from uncertainties.umath import log10
    # >>> S09 = {'tag':'S09',
    # ...   'mag':-2.5*log10(ufloat(9.796e-02, 2.70e-02)/3631),
    # ...   'zp':ufloat(-48.60, 2.5*log10(1.070)),
    # ...   'wave':np.array([5.78, 6.97, 8.63, 9.98, 11.38, 11.80, 12.41])*1e4,
    # ...   'resp':np.array([0.00, 0.39, 0.86, 1.00, 0.71, 0.07, 0.00]) }
    # >>> flux2mag = Flux2mag('AI Phe', extra_data=[S09])
    def __init__(self, name, extra_data=[], colors_data=[]):  # STROMGREN_COLORS

        self.name = name
        # Zero-point data as a dictionary
        # Error in "zero-point" for GALEX FUV & NUV are RMS from Camarota and Holberg, 2014
        # N.B. For WISE, these are offsets from vega to AB magnitudes from Jarret et al.
        # "This Vega basis has an overall systematic uncertainty of ∼1.45%." (0.016 mag)

        self.zp = {
            'FUV': ufloat(-48.60, 0.134),
            'NUV': ufloat(-48.60, 0.154),
            'G': ufloat(25.6914, 0.0011),
            'BP': ufloat(25.3488, 0.0005),
            'RP': ufloat(24.7627, 0.0035),
            'J': ufloat(-0.025, 0.005),
            'H': ufloat(+0.004, 0.005),
            'Ks': ufloat(-0.015, 0.005),
            'W1': ufloat(2.699, 0.0016),
            'W2': ufloat(3.339, 0.0016),
            'W3': ufloat(5.174, 0.0016),
            'W4': ufloat(6.620, 0.0016)
        }

        #
        # Flux integrals for reference Vega spectrum
        self.f_vega = {
            'J': 6.272182574976323e-11,
            'H': 4.705019995520602e-11,
            'Ks': 2.4274737135123822e-11
        }
        # Response functions as a dictionary of interpolating functions
        R = dict()

        # Pivot wavelength using equation (A16) from Bessell & Murphy
        def wp(w, r):
            return np.sqrt(simps(r * w, w) / simps(r / w, w))

        w_pivot = dict()

        # Johnson bands from Bessell, 2012 PASP, 124:140-157
        T = Table.read('Response/J_PASP_124_140_table1.dat.fits')
        for b in ['U', 'B', 'V', 'R', 'I']:
            wtmp = T['lam.{}'.format(b)]
            rtmp = T[b]
            rtmp = rtmp[wtmp > 0]
            wtmp = wtmp[wtmp > 0]
            R[b] = interp1d(wtmp, rtmp, bounds_error=False, fill_value=0)
            w_pivot[b] = wp(wtmp, rtmp)

        # GALEX
        for b in ['FUV', 'NUV']:
            T = Table.read('Response/EA-{}_im.tbl'.format(b.lower()), format='ascii', names=['w', 'A'])
            R[b] = interp1d(T['w'], T['A'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(T['w'], T['A'])

        # Gaia revised band passes
        names = ['wave', 'G', 'e_G', 'BP', 'e_BP', 'RP', 'e_RP']
        T = Table.read('Response/GaiaDR2_RevisedPassbands.dat',
                       format='ascii', names=names)
        w = T['wave'] * 10
        for b in ['G', 'BP', 'RP']:
            i = (T[b] < 99).nonzero()
            R[b] = interp1d(w[i], T[b][i], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(w[i], T[b][i])

        # 2MASS
        for k, b in enumerate(['J', 'H', 'Ks']):
            T = Table.read('Response/sec6_4a.tbl{:1.0f}.dat'.format(k + 1), format='ascii', names=['w', 'T'])
            R[b] = interp1d(T['w'] * 1e4, T['T'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(T['w'] * 1e4, T['T'])

        # ALLWISE QE-based RSRs from
        # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
        for b in ('W1', 'W2', 'W3', 'W4'):
            T = Table.read("Response/RSR-{}.txt".format(b),
                           format='ascii', names=['w', 'T', 'e'])
            R[b] = interp1d(T['w'] * 1e4, T['T'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(T['w'] * 1e4, T['T'])

        # Process response functions in extra_data
        for x in extra_data:
            b = x['tag']
            w = x['wave']
            r = x['resp']
            R[b] = interp1d(w, r, bounds_error=False, fill_value=0)
            w_pivot[b] = wp(w, r)
            self.zp[b] = x['zp']

        self.R = R
        self.w_pivot = w_pivot
        self.extra_data = extra_data
        self.colors_data = colors_data

        # Create catalogue query functions for Gaia DR2, 2MASS, GALEX and WISE
        Vizier_r = Vizier(columns=["*", "+_r"])
        # THIS_IS_NEW -
        # Use WISE All Sky values instead of ALLWISE for consistency with flux ratio
        # calibration and because these are more reliable at the bright end for W1 and W2
        v = Vizier_r.query_object(name, catalog=['I/345/gaia2', 'II/311/wise', 'II/335/galex_ais'])

        sb = Simbad()
        sb.add_votable_fields('flux(J)', 'flux_error(J)', 'flux(H)', 'flux_error(H)', 'flux(K)', 'flux_error(K)')

        obs_mag = dict()
        obs_mag['G'] = 0.0505 + 0.9966 * ufloat(v[0][0]['Gmag'], v[0][0]['e_Gmag'])
        obs_mag['BP'] = ufloat(v[0][0]['BPmag'], v[0][0]['e_BPmag']) - 0.0026
        obs_mag['RP'] = ufloat(v[0][0]['RPmag'], v[0][0]['e_RPmag']) + 0.0008
        for b in ['J', 'H', 'K', 'W1', 'W2', 'W3', 'W4']:  # THIS_IS_NEW include wise W4
            if b == 'J' or b == 'H' or b == 'K':
                sb_tab = sb.query_object(name)
                if b == 'K':
                    obs_mag['Ks'] = ufloat(sb_tab['FLUX_{}'.format(b)][0], sb_tab['FLUX_ERROR_{}'.format(b)][0])
                else:
                    obs_mag[b] = ufloat(sb_tab['FLUX_{}'.format(b)][0], sb_tab['FLUX_ERROR_{}'.format(b)][0])
            else:
                obs_mag[b] = ufloat(v[1][0]['{}mag'.format(b)], v[1][0]['e_{}mag'.format(b)])
        obs_mag['FUV'] = ufloat(v[2][0]['FUVmag'], v[2][0]['e_FUVmag'])
        # obs_mag['NUV'] = ufloat(v[2][0]['NUVmag'], v[2][0]['e_NUVmag'])

        # Add magnitudes from extra_data
        for x in extra_data:
            obs_mag[x['tag']] = x['mag']

        self.obs_mag = obs_mag

        # Add colors from colors_data # STROMGREN_COLORS
        obs_col = dict()
        for x in colors_data:
            obs_col[x['tag']] = x['color']
        self.obs_col = obs_col

    def __call__(self, wave, f_lambda, sig_ext=0, sig_col=0):  # SIGMA_COL
        # Integrate f_lambda over passbands and return chi-square of fit to observed magnitudes
        # f_lambda must be call-able with argument lambda = wavelength in Angstrom
        syn_mag = dict()
        R = self.R['FUV']
        f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                simps(R(wave) * 2.998e10 / (wave * 1e-8), wave))
        syn_mag['FUV'] = -2.5 * np.log10(f_nu) + self.zp['FUV']

        P_A = 0.7278
        hc9 = 1.986445824e-16  # 10^9 hc
        # /1000 is conversion from erg/s/cm^2/A to W/m^2/nm
        for b in ['G', 'BP', 'RP']:
            photon_flux = P_A * simps(self.R[b](wave) * wave * f_lambda / 10000, wave) / hc9
            syn_mag[b] = -2.5 * np.log10(photon_flux) + self.zp[b]

        # "+20" to account for wave in A not um
        for b in ['J', 'H', 'Ks']:
            v = self.f_vega[b]
            zp = self.zp[b]
            R = self.R[b]
            syn_mag[b] = -2.5 * np.log10(simps(R(wave) * f_lambda * wave, wave) / v) + 20 + zp

        # For ALLWISE, calculate AB magnitudes and then convert Vega magnitudes using correctiond from
        # Jarrett_2011_ApJ_735_112,
        # "Conversion to the monochromatic AB system entails an additional 2.699, 3.339, 5.174, and 6.620 added
        #  to the Vega magnitudes for W1, W2, W3, and W4, respectively"
        # "+20" to account for wave in A not um

        for b in ('W1', 'W2', 'W3', 'W4'):  # THIS_IS_NEW Include W4
            zp = self.zp[b]
            R = self.R[b]
            f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                    simps(R(wave) * 2.998e10 / (wave), wave))
            syn_mag[b] = -2.5 * np.log10(f_nu) + 20 - 48.60 - self.zp[b]
        self.syn_mag = syn_mag

        # Process extra_data
        for x in self.extra_data:
            b = x['tag']
            R = self.R[b]
            f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                    simps(R(wave) * 2.998e10 / (wave * 1e-8), wave))
            syn_mag[b] = -2.5 * np.log10(f_nu) + self.zp[b]

            # Process colors_data
        syn_col = {}
        for x in self.colors_data:
            b = x['tag']
            mag = {}
            f_interp = interp1d(wave, f_lambda)
            for m in ('u', 'v', 'b', 'y'):
                mag[m] = -2.5 * log10(simps(x['resp'][m] * f_interp(x['wave'][m]), x['wave'][m]) / x['vega_zp'][m])
            if x['type'] == 'by':
                by_o = x['color']
                syn_col[b] = mag['b'] - mag['y'] + x['zp'] + 0.003
            if x['type'] == 'm1':
                m1_o = x['color']
                by_c = mag['b'] - mag['y']
                vb_c = mag['v'] - mag['b']
                syn_col[b] = vb_c - by_c + x['zp'] + 0.157
            if x['type'] == 'c1':
                c1_o = x['color']
                uv_c = mag['u'] - mag['v']
                vb_c = mag['v'] - mag['b']
                syn_col[b] = uv_c - vb_c + x['zp'] + 1.088
        self.syn_col = syn_col

        lnlike_m = 0
        chisq = 0
        for k, v in zip(syn_mag.keys(), syn_mag.values()):
            z = self.obs_mag[k] - syn_mag[k]
            wt = 1 / (z.s ** 2 + sig_ext ** 2)
            chisq += z.n ** 2 * wt
            lnlike_m += -0.5 * (z.n ** 2 * wt - np.log(wt))
        lnlike_c = 0
        for k, v in zip(syn_col.keys(), syn_col.values()):
            z = self.obs_col[k] - syn_col[k]
            wt = 1 / (z.s ** 2 + sig_col ** 2)  # SIGMA_COL
            chisq += z.n ** 2 * wt
            lnlike_c += -0.5 * (z.n ** 2 * wt - np.log(wt))
        return chisq, lnlike_m, lnlike_c


# # Response functions for IUE fluxes
# transmission = pickle.load(open("Response/transmission.pickle", "rb"))
# wave = pickle.load(open("Response/wave.pickle", "rb"))
#
# u320 = {'tag': 'u320',
#         'mag': -2.5 * log10(ufloat(1.846e-24, 6.034e-26)) - 48.6,
#         'zp': ufloat(-48.60, round(2.5 * log10(1.040), 3)),  # 4% according to Nichols & Linsky (1996)
#         'wave': wave,
#         'resp': transmission[0]}
# u220n = {'tag': 'u220n',
#          'mag': -2.5 * log10(ufloat(9.713e-26, 8.924e-27)) - 48.6,
#          'zp': ufloat(-48.60, round(2.5 * log10(1.040), 3)),  # 4% according to Nichols & Linsky (1996)
#          'wave': wave,
#          'resp': transmission[1]}
# u220w = {'tag': 'u220w',
#          'mag': -2.5 * log10(ufloat(8.583e-26, 9.401e-27)) - 48.6,
#          'zp': ufloat(-48.60, round(2.5 * log10(1.040), 3)),  # 4% according to Nichols & Linsky (1996)
#          'wave': wave,
#          'resp': transmission[2]}
#
# # Zero point and error is average colour o-c offset and standard deviation
#
# stromgren_w, stromgren_r = {}, {}
#
# for x in ('u', 'v', 'b', 'y'):
#     T = Table.read("Response/{}_Bessell2005.csv".format(x))
#     stromgren_w[x] = np.array(T['wave'])
#     stromgren_r[x] = np.array(T['response'])
#
# stromgren_v = {
#     'u': 9.139833801171253e-07,
#     'v': 1.2227871972005228e-06,
#     'b': 1.0300321185051395e-06,
#     'y': 7.412341517648064e-07
# }
#
# # Colours from GCS III. Errors from Olsen 1994
#
# Gby = {'tag': '(b-y)_G',
#        'type': 'by',
#        'color': ufloat(0.431, 0.0037),
#        'zp': ufloat(-0.0014, 0.0045),
#        'wave': stromgren_w,
#        'resp': stromgren_r,
#        'vega_zp': stromgren_v}
# Gm1 = {'tag': 'm1_G',
#        'type': 'm1',
#        'color': ufloat(0.209, 0.0041),
#        'zp': ufloat(0.0243, 0.0062),
#        'wave': stromgren_w,
#        'resp': stromgren_r,
#        'vega_zp': stromgren_v}
# Gc1 = {'tag': 'c1_G',
#        'type': 'c1',
#        'color': ufloat(0.356, 0.0066),
#        'zp': ufloat(-0.0102, 0.0083),
#        'wave': stromgren_w,
#        'resp': stromgren_r,
#        'vega_zp': stromgren_v}
#
# # Colours Reipurth 1978
#
# Rby = {'tag': '(b-y)_R',
#        'type': 'by',
#        'color': ufloat(0.424, 0.0037),
#        'zp': ufloat(-0.0014, 0.0045),
#        'wave': stromgren_w,
#        'resp': stromgren_r,
#        'vega_zp': stromgren_v}
# Rm1 = {'tag': 'm1_R',
#        'type': 'm1',
#        'color': ufloat(0.219, 0.0041),
#        'zp': ufloat(0.0243, 0.0062),
#        'wave': stromgren_w,
#        'resp': stromgren_r,
#        'vega_zp': stromgren_v}
# Rc1 = {'tag': 'c1_R',
#        'type': 'c1',
#        'color': ufloat(0.357, 0.0066),
#        'zp': ufloat(-0.0102, 0.0083),
#        'wave': stromgren_w,
#        'resp': stromgren_r,
#        'vega_zp': stromgren_v}

# flux2mag = Flux2mag('AI Phe', extra_data=[u320, u220w, u220n],
#                     colors_data=[Gby, Gm1, Gc1, Rby, Rm1, Rc1])
flux2mag = Flux2mag('AI Phe', extra_data, colors_data)

# print("Magnitudes")
for k in flux2mag.obs_mag.keys():
    o = flux2mag.obs_mag[k]
    w = flux2mag.w_pivot[k]
    # print("{:4s} {:6.0f} {:6.4f}".format(k,w,o))
# print("Color data")
for col in flux2mag.colors_data:
    s = col['tag']
    t = col['type']
    o = col['color']
    # print("{:8s} {:3s} {:6.4f}".format(s,t,o))


l = pickle.load(open("lratio_priors.pickle", "rb"))
# Fix typo in R1, R2 values
l['R1']['Value'] = ufloat(1.197, 0.024)
l['R2']['Value'] = ufloat(1.198, 0.024)
l['u']['Value'] = ufloat(0.475, 0.017)
l['v']['Value'] = ufloat(0.624, 0.009)
l['b']['Value'] = ufloat(0.870, 0.006)
l['y']['Value'] = ufloat(1.036, 0.007)
l['u320']['Value'] = ufloat(0.342, 0.042)
l['u220n']['Value'] = ufloat(0.030, 0.066)
l['u220w']['Value'] = ufloat(0.059, 0.090)
# Fix broken Wavelength array for TESS entry
# l['TESS']['Wavelength'] = np.array([10*float(s[:-1]) for s in l['TESS']['Wavelength']])
# .. and update the value
l['TESS']['Value'] = ufloat(1.319, 0.001)
# Convert wavelength/response to interpolating functions
lratios = {}
for k in l.keys():
    if l[k]['Response'] is not None:

        d = {}
        d['Value'] = l[k]['Value']
        w = np.array(l[k]['Wavelength'], dtype='f8')
        R = l[k]['Response']
        d['R'] = interp1d(w, R, bounds_error=False, fill_value=0)
        if k == "TESS":
            d['photon'] = True
        else:
            d['photon'] = False
        lratios[k] = d
        print(k, d['Value'], d['photon'])
# H-band flux ratio from Gallenne et al., 2019
# Use a nominal error of 0.01
k = 'H'
d = {
    'Value': ufloat(100 / 49.7, 0.01),
    'R': flux2mag.R[k],
    'photon': True
}
lratios[k] = d
print(k, d['Value'], d['photon'])


def flux_ratio_priors(Vrat, TeffF, TeffK):
    # Vrat in the sense Flux_F/Flux_K
    data = {
        'J': {'cF': 0.919, 'mF': -0.408, 'sigF': 0.015, 'cK': 1.511, 'mK': -0.605, 'sigK': 0.018},
        'H': {'cF': 1.118, 'mF': -0.549, 'sigF': 0.019, 'cK': 1.918, 'mK': -0.821, 'sigK': 0.027},
        'Ks': {'cF': 1.181, 'mF': -0.564, 'sigF': 0.017, 'cK': 2.033, 'mK': -0.872, 'sigK': 0.025},
        'W1': {'cF': 1.230, 'mF': -0.568, 'sigF': 0.027, 'cK': 2.094, 'mK': -0.865, 'sigK': 0.035},
        'W2': {'cF': 1.234, 'mF': -0.547, 'sigF': 0.039, 'cK': 2.101, 'mK': -0.928, 'sigK': 0.062},
        'W3': {'cF': 1.182, 'mF': -0.554, 'sigF': 0.021, 'cK': 2.062, 'mK': -0.907, 'sigK': 0.036},
        'W4': {'cF': 1.225, 'mF': -0.519, 'sigF': 0.050, 'cK': 2.095, 'mK': -0.951, 'sigK': 0.060}
    }
    # Return a dictionary of ufloat priors on flux ratios
    d = {}
    for b in data.keys():
        colF = data[b]['cF'] + data[b]['mF'] * (TeffF - 6400) / 1000.0
        colK = data[b]['cK'] + data[b]['mK'] * (TeffK - 5200) / 1000.0
        L = Vrat * 10 ** (0.4 * (colK - colF))
        e_L = np.hypot(data[b]['sigF'], data[b]['sigK'])
        d[b] = ufloat(L, e_L)
    return d


frp = flux_ratio_priors(1.05, 6440, 5220)


def lnprob(params,  # Model parameters and hyper-parameters
           flux2mag,  # Magnitude data and flux-to-mag log-likelihood calculator
           lratios,  # Flux ratios and responses
           theta1, theta2,  # angular diameters as ufloats **in milli-arcseconds**
           spec1, spec2,  # Model spectra
           ebv_prior,  # ufloat
           redlaw,  # Reddening law
           Nc1,  # Number of distortion coeffs for star 1
           wmin=1000, wmax=300000,
           return_flux=False,
           blobs=False, apply_flux_ratio_priors=True,
           debug=False, verbose=False):
    SIGMA_SB = 5.670367E-5  # erg.cm-2.s-1.K-4

    Teff1, Teff2, Theta1, Theta2, ebv, sigma_ext, sigma_l, sigma_col = params[0:8]  # SIGMA_COL

    if Theta1 < 0: return -np.inf
    if Theta2 < 0: return -np.inf
    if ebv < 0: return -np.inf
    if sigma_ext < 0: return -np.inf
    if sigma_col < 0: return -np.inf  # SIGMA_COL
    if sigma_l < 0: return -np.inf

    wave = spec1.waveset
    i = ((wmin * u.angstrom < wave) & (wave < wmax * u.angstrom)).nonzero()
    wave = wave[i]
    flux1 = spec1(wave, flux_unit=units.FLAM)
    flux2 = spec2(wave, flux_unit=units.FLAM)
    wave = wave.value  # Converts to numpy array
    flux1 = flux1.value
    flux2 = flux2.value

    x = 2 * np.log(wave / np.min(wave)) / np.log(np.max(wave) / np.min(wave)) - 1
    distort1 = np.zeros_like(flux1)
    for n, c in enumerate(params[8:8 + Nc1]):
        if abs(c) > 1: return -np.inf
        distort1 = distort1 + c * legendre(n + 1)(x)
    # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
    i_5556 = np.argmin(abs(wave - 5556))
    distort1 = distort1 - distort1[i_5556]
    if min(distort1) < -1:
        return -np.inf
    flux1 = flux1 * (1 + distort1)
    flux1 = flux1 / simps(flux1, wave)

    distort2 = np.zeros_like(flux2)
    for n, c in enumerate(params[8 + Nc1:]):
        if abs(c) > 1: return -np.inf
        distort2 = distort2 + c * legendre(n + 1)(x)
    # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
    distort2 = distort2 - distort2[i_5556]
    if min(distort2) < -1:
        return -np.inf
    flux2 = flux2 * (1 + distort2)
    flux2 = flux2 / simps(flux2, wave)

    extinc = redlaw.extinction_curve(ebv)(wave).value
    f_1 = 0.25 * SIGMA_SB * (Theta1 / 206264806) ** 2 * Teff1 ** 4 * flux1
    f_2 = 0.25 * SIGMA_SB * (Theta2 / 206264806) ** 2 * Teff2 ** 4 * flux2
    flux = (f_1 + f_2) * extinc
    if return_flux:
        return wave, flux, f_1 * extinc, f_2 * extinc, distort1, distort2

    chisq, lnlike_m, lnlike_c = flux2mag(wave, flux, sigma_ext, sigma_col)  # SIGMA_COL
    if verbose:
        print('Magnitudes')
        for k in flux2mag.syn_mag.keys():
            o = flux2mag.obs_mag[k]
            c = flux2mag.syn_mag[k]
            w = flux2mag.w_pivot[k]
            print("{:6s} {:6.0f} {:6.3f} {:6.3f} {:+6.3f}".format(k, w, o, c, o - c))

    if verbose:  # STROMGREN_COLORS
        print('Colours')
        for k in flux2mag.syn_col.keys():
            o = flux2mag.obs_col[k]
            c = flux2mag.syn_col[k]
            print("{:8s} {:6.3f} {:6.3f} {:+6.3f}".format(k, o, c, o - c))

    lnlike_l = 0
    blob_data = []
    if verbose:
        print('Flux ratios')
    for k in lratios.keys():
        R = lratios[k]['R']
        if lratios[k]['photon']:
            l1 = simps(R(wave) * f_1 * wave, wave)
            l2 = simps(R(wave) * f_2 * wave, wave)
        else:
            l1 = simps(R(wave) * f_1, wave)
            l2 = simps(R(wave) * f_2, wave)
        v = lratios[k]['Value']
        lrat = l2 / l1
        blob_data.append(lrat)
        wt = 1 / (v.s ** 2 + sigma_l ** 2)
        lnlike_l += -0.5 * ((lrat - v.n) ** 2 * wt - np.log(wt))
        if verbose:
            print("{:4s} {:6.3f} {:6.3f} +/- {:5.3f}".format(k, lrat, v.n, v.s))

    # See http://mathworld.wolfram.com/BivariateNormalDistribution.html, equation (1)
    rho = correlation_matrix([theta1, theta2])[0][1]
    z = ((Theta1 - theta1.n) ** 2 / theta1.s ** 2 -
         2 * rho * (Theta1 - theta1.n) * (Theta2 - theta2.n) / theta1.s / theta2.s +
         (Theta2 - theta2.n) ** 2 / theta2.s ** 2)
    lnlike_theta = -0.5 * z / (1 - rho ** 2)

    lnlike = lnlike_m + lnlike_c + lnlike_theta + lnlike_l

    lnprior = 0
    if ebv_prior is not None:
        lnprior += -0.5 * (ebv - ebv_prior.n) ** 2 / ebv_prior.s ** 2
    # Priors on IR flux ratios
    RV = flux2mag.R['V'](wave)
    lV = simps(RV * f_2, wave) / simps(RV * f_1, wave)
    frp = flux_ratio_priors(lV, Teff1, Teff2)
    if verbose: print('Flux ratio priors:')
    chisq_flux_ratio_priors = 0
    for b in frp.keys():
        RX = flux2mag.R[b](wave)
        lX = simps(RX * f_2 * wave, wave) / simps(RX * f_1 * wave, wave)  # Predicted value
        if verbose: print('{:<2s}: {:0.3f}  {:0.3f}  {:+0.3f}'.format(b, frp[b], lX, frp[b] - lX))
        chisq_flux_ratio_priors += (lX - frp[b].n) ** 2 / (frp[b].s ** 2 + sigma_l ** 2)
        if apply_flux_ratio_priors:
            wt = 1 / (frp[b].s ** 2 + sigma_l ** 2)
            lnprior += -0.5 * ((lX - frp[b].n) ** 2 * wt - np.log(wt))
    if verbose: print('Flux ratio priors: chi-squared = {:0.2}'.format(chisq_flux_ratio_priors))

    if debug:
        f = "{:0.1f} {:0.1f} {:0.4f} {:0.4f} {:0.4f} {:0.1f} {:0.4f}"
        f += " {:0.1e}" * Nc1
        Nc2 = len(params) - 8 - Nc1
        f += " {:0.1e}" * Nc2
        f += " {:0.2f}"
        print(f.format(*(tuple(params) + (lnlike,))))
    if np.isfinite(lnlike):
        if blobs:
            return (lnlike + lnprior, *blob_data)
        else:
            return lnlike + lnprior
    else:
        return -np.inf


plx_Gallenne = ufloat(5.905, 0.024)
gaia_zp = ufloat(-0.031, 0.011)
plx_DR2 = ufloat(5.8336, 0.0262) - gaia_zp
plx = (plx_Gallenne + plx_DR2) / 2


Tref1 = 6200
Tref2 = 5100
M_H = -0.14
aFe = 0.06
spec1a = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=50, M_H=0.0, aFe=0.0, reload=True)
spec1b = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=50, M_H=-0.5, aFe=0.2, reload=True)
spec2a = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=50, M_H=0.0, aFe=0.0, reload=True)
spec2b = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=50, M_H=-0.5, aFe=0.2, reload=True)

spec1 = 0.72 * spec1a + 0.28 * spec1b
spec2 = 0.72 * spec2a + 0.28 * spec2b

# No detectable NaI lines so E(B-V) must be very close to 0 - see 2010NewA...15..444K
ebv_prior = ufloat(0.0, 0.005)  # No detectable NaI lines so E(B-V) must be very close to 0
redlaw = ReddeningLaw.from_extinction_model('mwavg')

# Angular diameter = 2*R/d = 2*R*parallax = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
# R_Sun = 6.957e8 m
# parsec = 3.085677581e16 m
# R_1 = ufloat(1.835, 0.014)    # JK-K values
# R_2 = ufloat(2.912, 0.014)    # JK-K values
R_1 = ufloat(1.8050, 0.0046)  # Prelimanary values from TESS analysis
R_2 = ufloat(2.9343, 0.0034)  # Prelimanary values from TESS analysis
theta1 = 2 * plx * R_1 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
theta2 = 2 * plx * R_2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
# print('theta1 = {:0.4f} mas'.format(theta1))
# print('theta2 = {:0.4f} mas'.format(theta2))
theta_cov = covariance_matrix([theta1, theta2])[0][1]
theta_cor = correlation_matrix([theta1, theta2])[0][1]
# print('cov(theta_1,theta2) = {:0.2e}'.format(theta_cov))
# print('cor(theta_1,theta2) = {:0.2f}'.format(theta_cor))

Teff1 = 6223
Teff2 = 5135
# Copy starting values to new variables
theta1_ = theta1.n
theta2_ = theta2.n
ebv_ = ebv_prior.n
sigma_ext = 0.008
sigma_l = 0.01
sigma_c = 0.005
Nc1 = 10
Nc2 = 10
params = [Teff1, Teff2, theta1_, theta2_, ebv_, sigma_ext, sigma_l, sigma_c]
params = params + [0] * Nc1
params = params + [0] * Nc2

parname = ['T_eff,1', 'T_eff,2', 'theta_1', 'theta_2', 'E(B-V)', 'sigma_ext', 'sigma_l', 'sigma_c']
parname = parname + ["c_1,{}".format(j + 1) for j in range(Nc1)]
parname = parname + ["c_2,{}".format(j + 1) for j in range(Nc2)]

for pn, pv in zip(parname, params):
    print('{} = {}'.format(pn, pv))

lnlike = lnprob(params, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1, verbose=True)
# print('Initial log-likelihood = {:0.2f}'.format(lnlike))


nll = lambda *args: -lnprob(*args)
args = (flux2mag, lratios, theta1, theta2,
        spec1, spec2, ebv_prior, redlaw, Nc1)
soln = minimize(nll, params, args=args, method='Nelder-Mead')

# print('theta1 = {:0.4f} mas'.format(theta1))
# print('theta2 = {:0.4f} mas'.format(theta2))
# print()
# for pn,pv in zip(parname, soln.x):
#     print('{} = {}'.format(pn,pv))

lnlike = lnprob(soln.x, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1, verbose=True)
# print('Final log-likelihood = {:0.2f}'.format(lnlike))


steps = [25, 25,  # T_eff,1, T_eff,2
         0.0005, 0.0007,  # theta_1 ,theta_2
         0.001, 0.001, 0.001, 0.001,  # E(B-V), sigma_ext, sigma_l, sigma_c
         *[0.01] * Nc1, *[0.01] * Nc2]  # c_1,1 ..   c_2,1 ..

nwalkers = 256
ndim = len(soln.x)
pos = np.zeros([nwalkers, ndim])
for i, x in enumerate(soln.x):
    pos[:, i] = x + steps[i] * np.random.randn(nwalkers)

nsteps = 1000
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, pool=pool)
    sampler.run_mcmc(pos, nsteps, progress=True)

af = sampler.acceptance_fraction
# print('\nMedian acceptance fraction =',np.median(af))
best_index = np.unravel_index(np.argmax(sampler.lnprobability),
                              (nwalkers, nsteps))
best_lnlike = np.max(sampler.lnprobability)
# print('\n Best log(likelihood) = ',best_lnlike,' in walker ',best_index[0],
#        ' at step ',best_index[1])
best_pars = sampler.chain[best_index[0], best_index[1], :]

fig, axes = plt.subplots(4, figsize=(10, 7), sharex='all')
samples = sampler.get_chain()
i0 = 0
labels = parname[i0:i0 + 4]
for i in range(4):
    ax = axes[i]
    ax.plot(samples[:, :, i0 + i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")
fig.show()

flat_samples = sampler.get_chain(discard=4000, thin=8, flat=True)  # nsteps//2
fig = corner.corner(flat_samples, labels=parname)
fig.show()

for i, pn in enumerate(parname):
    val = flat_samples[:, i].mean()
    err = flat_samples[:, i].std()
    ndp = 1 - min(0, np.floor(log10(err)))
    fmt = '{{:0.{:0.0f}f}}'.format(ndp)
    vstr = fmt.format(val)
    estr = fmt.format(err)
    # print('{} = {} +/- {}'.format(pn,vstr,estr))

lnlike = lnprob(best_pars, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1, verbose=True)
# print('Final log-likelihood = {:0.2f}'.format(lnlike))

wave, flux, f_1, f_2, d1, d2 = lnprob(
    best_pars, flux2mag, lratios,
    theta1, theta2, spec1, spec2,
    ebv_prior, redlaw, Nc1, return_flux=True)
fig1, ax1 = plt.subplots(3, figsize=(10, 7), sharex='all')
ax1[0].semilogx(wave, 1e12 * f_1, c='c')
ax1[0].semilogx(wave, 1e12 * f_2, c='orange')
ax1[0].set_xlim(1000, 300000)
ax1[0].set_ylabel(r'$f_{\lambda}\:\:[10^{-12}\,{\rm ergs}\,{\rm cm}^{-2}\,{\rm s}^{-1}\,{\rm \AA}^{-1}}]$')
ax1[1].semilogx(wave, d1, c='b')
ax1[1].set_ylabel('$\Delta_1$')
ax1[1].set_ylim(-0.25, 0.25)
ax1[2].semilogx(wave, d2, c='b')
ax1[2].set_ylabel('$\Delta_2$')
ax1[2].set_xlabel(r'Wavelength [$\rm \AA$]')
ax1[2].set_ylim(-0.25, 0.25)

for i in range(0, len(flat_samples), len(flat_samples) // 64):
    _, _, _, _, _d1, _d2 = lnprob(
        flat_samples[i, :], flux2mag, lratios,
        theta1, theta2, spec1, spec2,
        ebv_prior, redlaw, Nc1, return_flux=True)
    ax1[1].semilogx(wave, _d1, c='b', alpha=0.1)
    ax1[2].semilogx(wave, _d2, c='b', alpha=0.1)

fig.show()

VegaZeroPointErrorPercent = 0.5
Fig14Data = Table.read('Bohlin2014_Fig14.csv', names=['w', 'err'])
WDScaleErrorWavelengthAngstrom = Fig14Data['w'] * 10000
WDScaleErrorPercent = Fig14Data['err']
TotalSystematicErrorPercent = VegaZeroPointErrorPercent + WDScaleErrorPercent
plt.semilogx(WDScaleErrorWavelengthAngstrom, TotalSystematicErrorPercent, 'bo')
Interpolator = interp1d(WDScaleErrorWavelengthAngstrom, TotalSystematicErrorPercent, bounds_error=False,
                        fill_value=50.0)
WavelengthGrid = np.linspace(min(WDScaleErrorWavelengthAngstrom), max(WDScaleErrorWavelengthAngstrom), 50001)
TotSysErrPercentGrid = Interpolator(WavelengthGrid)
plt.semilogx(WavelengthGrid, TotSysErrPercentGrid)
plt.xlabel(r'Wavelength [$\AA$]')
plt.ylabel('Flux scale error [%]')
plt.show()

TotSysErrPercentGrid = Interpolator(wave)
T_eff_1 = flat_samples[:, 0].mean()
rnderr_1 = flat_samples[:, 0].std()
fint_1 = simps(f_1, wave)
fint_1p = simps(f_1 * (1 + TotSysErrPercentGrid / 100), wave)
syserr_1 = (fint_1p / fint_1 - 1) * T_eff_1 / 4  # /4 because L \propto Teff^4
# print('Systematic error in integrated flux = {:0.2%}%'.format((fint_1p/fint_1-1)))
# print('T_eff,1 = {:0.0f} +/- {:0.0f} (rnd.) +/- {:0.0f} (sys.) K'.
#     format(T_eff_1, rnderr_1, syserr_1))

T_eff_2 = flat_samples[:, 1].mean()
rnderr_2 = flat_samples[:, 1].std()
fint_2 = simps(f_2, wave)
fint_2p = simps(f_2 * (1 + TotSysErrPercentGrid / 100), wave)
syserr_2 = (fint_2p / fint_2 - 1) * T_eff_2 / 4
# print('Systematic error in integrated flux = {:0.2%}%'.format((fint_2p/fint_2-1)))
# print('T_eff,2 = {:0.0f} +/- {:0.0f} (rnd.) +/- {:0.0f} (sys.) K'.
#          format(T_eff_2, rnderr_2, syserr_2))


tag = "C"
pfile = "{}_{:0.0f}_{:0.0f}+{:+0.1f}_{}_coeffs.p".format(tag, Tref1, Tref2, M_H, Nc1)
with open(pfile, 'wb') as f:
    pickle.dump(sampler, f)
