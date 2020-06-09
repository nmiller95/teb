from uncertainties import ufloat
from uncertainties.umath import log10
from scipy.interpolate import interp1d
from scipy.integrate import simps
import numpy as np
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad


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

