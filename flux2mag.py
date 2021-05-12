from uncertainties import ufloat
from uncertainties.umath import log10
from scipy.interpolate import interp1d
from scipy.integrate import simps
import numpy as np
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad


class Flux2mag:
    """
    Flux2mag methods:
        __init__: Accesses and consolidates magnitude and photometric colors of a star.
        __call__: Generates synthetic magnitudes and colors from integrating a reference spectrum of Vega.
    """

    def __init__(self, name, extra_data=None, colors_data=None):
        """
        Reads in bandpass data from package files. Calculates a standardised response function and pivot wavelength of
        each bandpass. Retrieves standard photometric data from SIMBAD and processes any extra data supplied.

        :param name: Name of star, as understood by SIMBAD
        :type name: basestring
        :param extra_data: List of dictionaries containing information about non-standard magnitudes.
            Each dictionary must contain:
                'tag' - string, photometric band/system name/label
                'mag' - ufloat, observed AB magnitude with standard error
                'zp'  - ufloat, zero-point and standard error
                'wave' - array, wavelengths for response function, ** in angstrom **
                'resp'  - array, response function
        :type extra_data: list or none
        :param colors_data: List of dictionaries containing information about photometric colors.
            Each dictionary must contain:
                'tag' - string, photometric color name/label (must be unique), e.g. '(b-y)_1'
                'type' - string, type of color, e.g. 'by', 'm1', 'c1'
                'color' - ufloat, observed color with standard error
                'zp' - ufloa, zero-point and standard error
                'wave' - array, wavelengths for response function, ** in angstrom **
                'resp' - array, response function
                'vega_zp' - dictionary, contains zero-points of the component passbands, e.g. u,b,v,y, from Vega (?)
        :type colors_data: list or none
        """

        self.name = name

        # Zero-point data as a dictionary
        # Error in "zero-point" for GALEX FUV & NUV are RMS from Camarota and Holberg, 2014
        # N.B. For WISE, these are offsets from vega to AB magnitudes from Jarret et al.
        # "This Vega basis has an overall systematic uncertainty of ∼1.45%." (0.016 mag)
        self.zp = {
            'FUV': ufloat(-48.60, 0.134),
            'NUV': ufloat(-48.60, 0.154),
            # 'G': ufloat(25.6914, 0.0011),  # Values for Gaia DR2
            # 'BP': ufloat(25.3488, 0.0005),  # Values for Gaia DR2
            # 'RP': ufloat(24.7627, 0.0035),  # Values for Gaia DR2
            'G': ufloat(25.8010, 0.0028),
            'BP': ufloat(25.3540, 0.0023),
            'RP': ufloat(24.7627, 0.0016),
            'J': ufloat(-0.025, 0.005),
            'H': ufloat(+0.004, 0.005),
            'Ks': ufloat(-0.015, 0.005),
            'W1': ufloat(2.699, 0.0016),
            'W2': ufloat(3.339, 0.0016),
            'W3': ufloat(5.174, 0.0016),
            'W4': ufloat(6.620, 0.0016)
        }

        # Flux integrals for reference Vega spectrum
        self.f_vega = {
            'J': 6.272182574976323e-11,
            'H': 4.705019995520602e-11,
            'Ks': 2.4274737135123822e-11
        }
        # Response functions as a dictionary of interpolating functions
        R = dict()

        # Pivot wavelength using equation (A16) from Bessell & Murphy (2011)
        def wp(wavelength, response):
            return np.sqrt(simps(response * wavelength, wavelength) / simps(response / wavelength, wavelength))

        w_pivot = dict()

        # -------------- READ IN BANDS, CALCULATE RESPONSE FUNCTIONS AND PIVOT WAVELENGTHS -------------- #
        # Johnson - from Bessell, 2012 PASP, 124:140-157
        t = Table.read('Response/J_PASP_124_140_table1.dat.fits')
        for b in ['U', 'B', 'V', 'R', 'I']:
            wtmp = t['lam.{}'.format(b)]
            rtmp = t[b]
            rtmp = rtmp[wtmp > 0]
            wtmp = wtmp[wtmp > 0]
            R[b] = interp1d(wtmp, rtmp, bounds_error=False, fill_value=0)
            w_pivot[b] = wp(wtmp, rtmp)

        # GALEX - source needed
        for b in ['FUV', 'NUV']:
            t = Table.read('Response/EA-{}_im.tbl'.format(b.lower()), format='ascii', names=['w', 'A'])
            R[b] = interp1d(t['w'], t['A'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(t['w'], t['A'])

        # Gaia DR2 - revised band passes
        names = ['wave', 'G', 'e_G', 'BP', 'e_BP', 'RP', 'e_RP']
        # t = Table.read('Response/GaiaDR2_RevisedPassbands.dat',  # Values for Gaia DR2
        #                format='ascii', names=names)
        t = Table.read('Response/GaiaEDR3_passbands.dat',
                       format='ascii', names=names)
        w = t['wave'] * 10
        for b in ['G', 'BP', 'RP']:
            i = (t[b] < 99).nonzero()
            R[b] = interp1d(w[i], t[b][i], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(w[i], t[b][i])

        # 2MASS - source needed
        for k, b in enumerate(['J', 'H', 'Ks']):
            t = Table.read('Response/sec6_4a.tbl{:1.0f}.dat'.format(k + 1), format='ascii', names=['w', 'T'])
            R[b] = interp1d(t['w'] * 1e4, t['T'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(t['w'] * 1e4, t['T'])

        # ALLWISE - QE-based RSRs from
        # http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
        for b in ('W1', 'W2', 'W3', 'W4'):
            t = Table.read("Response/RSR-{}.txt".format(b),
                           format='ascii', names=['w', 'T', 'e'])
            R[b] = interp1d(t['w'] * 1e4, t['T'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(t['w'] * 1e4, t['T'])

        # Process response functions in extra_data
        if extra_data:
            for x in extra_data:
                b = x['tag']
                w = x['wave']
                r = x['resp']
                R[b] = interp1d(w, r, bounds_error=False, fill_value=0)
                w_pivot[b] = wp(w, r)
                self.zp[b] = x['zp']
            self.extra_data = extra_data

        self.R = R
        self.w_pivot = w_pivot

        if colors_data:
            for x in colors_data:
                if x['type'] in ['by', 'm1', 'c1']:
                    stromgren_w, stromgren_r = {}, {}
                    for m in ('u', 'v', 'b', 'y'):
                        # Reads individual filter info from files
                        t = Table.read("Response/{}_Bessell2005.csv".format(m))
                        stromgren_w[m] = np.array(t['wave'])
                        stromgren_r[m] = np.array(t['response'])

                    stromgren_v = {
                        'u': 9.139833801171253e-07,
                        'v': 1.2227871972005228e-06,
                        'b': 1.0300321185051395e-06,
                        'y': 7.412341517648064e-07
                    }
                    # Zero point and error is average colour o-c offset and standard deviation TODO: Check this
                    if x['type'] == 'by':
                        x['zp'] = ufloat(-0.0014, 0.0045)
                    elif x['type'] == 'm1':
                        x['zp'] = ufloat(0.0243, 0.0062)
                    elif x['type'] == 'c1':
                        x['zp'] = ufloat(-0.0102, 0.0083)

                    x['wave'] = stromgren_w
                    x['resp'] = stromgren_r
                    x['vega_zp'] = stromgren_v

                elif x['type'] == 'BPRP':
                    # TODO: Gaia BP-RP colour
                    pass

            self.colors_data = colors_data

        # -------------- RETRIEVE STANDARD PHOTOMETRY WITH SIMBAD + VIZIER QUERIES -------------- #
        # Catalogue query functions for Gaia EDR3, 2MASS, GALEX and WISE
        Vizier_r = Vizier(columns=["*", "+_r"])
        # Use WISE All Sky values instead of ALLWISE for consistency with flux ratio
        # calibration and because these are more reliable at the bright end for W1 and W2
        # v = Vizier_r.query_object(name, catalog=['I/345/gaia2', 'II/311/wise', 'II/335/galex_ais'])  # Gaia DR2
        v = Vizier_r.query_object(name, catalog=['I/350/gaiaedr3', 'II/311/wise', 'II/335/galex_ais'])

        sb = Simbad()
        sb.add_votable_fields('flux(J)', 'flux_error(J)', 'flux(H)', 'flux_error(H)', 'flux(K)', 'flux_error(K)')

        obs_mag = dict()
        # obs_mag['G'] = 0.0505 + 0.9966 * ufloat(v[0][0]['Gmag'], v[0][0]['e_Gmag'])  # Gaia DR2
        # obs_mag['BP'] = ufloat(v[0][0]['BPmag'], v[0][0]['e_BPmag']) - 0.0026  # Gaia DR2
        # obs_mag['RP'] = ufloat(v[0][0]['RPmag'], v[0][0]['e_RPmag']) + 0.0008  # Gaia DR2
        obs_mag['G'] = ufloat(v[0][0]['Gmag'], v[0][0]['e_Gmag'])
        obs_mag['BP'] = ufloat(v[0][0]['BPmag'], v[0][0]['e_BPmag'])
        obs_mag['RP'] = ufloat(v[0][0]['RPmag'], v[0][0]['e_RPmag'])

        for b in ['J', 'H', 'K', 'W1', 'W2', 'W3', 'W4']:  # TODO: Add fail-safe for when no magnitudes found.
            if b == 'J' or b == 'H' or b == 'K':
                sb_tab = sb.query_object(name)
                if b == 'K':
                    obs_mag['Ks'] = ufloat(sb_tab['FLUX_{}'.format(b)][0], sb_tab['FLUX_ERROR_{}'.format(b)][0])
                else:
                    obs_mag[b] = ufloat(sb_tab['FLUX_{}'.format(b)][0], sb_tab['FLUX_ERROR_{}'.format(b)][0])
            else:
                obs_mag[b] = ufloat(v[1][0]['{}mag'.format(b)], v[1][0]['e_{}mag'.format(b)])
        try:
            obs_mag['FUV'] = ufloat(v[2][0]['FUVmag'], v[2][0]['e_FUVmag'])
            obs_mag['NUV'] = ufloat(v[2][0]['NUVmag'], v[2][0]['e_NUVmag'])
        except IndexError as error:
            print(error, ': No GALEX photometry found')

        # Add magnitudes from extra_data
        if extra_data:
            for x in extra_data:
                obs_mag[x['tag']] = x['mag']

        self.obs_mag = obs_mag

        # Add colors from colors_data
        if colors_data:
            obs_col = dict()
            for x in colors_data:
                obs_col[x['tag']] = x['color']
            self.obs_col = obs_col
        else:
            self.obs_col = None

    def __call__(self, wave, f_lambda, sig_ext=0, sig_col=0):
        """
        Integrates flux over the defined passbands and returns chi-square of fit to observed magnitudes.

        :param wave: Wavelength range over which the flux is defined, in Angstrom
        :param f_lambda: Must be call-able with argument lambda = wavelength in Angstrom
        :param sig_ext: Amount of external noise to the magnitudes
        :param sig_col: Amount of external noise to the colors
        :return: chi-square of the fit to observed magnitudes, and log likelihoods
        """

        syn_mag = dict()

        for b in ['FUV', 'NUV']:
            R = self.R[b]
            f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                    simps(R(wave) * 2.998e10 / (wave * 1e-8), wave))
            syn_mag[b] = -2.5 * np.log10(f_nu) + self.zp[b]

        P_A = 0.7278
        hc9 = 1.986445824e-16  # 10^9 hc
        # /10000 is conversion from erg/s/cm^2/A to W/m^2/nm
        for b in ['G', 'BP', 'RP']:
            photon_flux = P_A * simps(self.R[b](wave) * wave * f_lambda / 10000, wave) / hc9
            syn_mag[b] = -2.5 * np.log10(photon_flux) + self.zp[b]

        # "+20" to account for wave in A not um
        for b in ['J', 'H', 'Ks']:
            v = self.f_vega[b]
            zp = self.zp[b]
            R = self.R[b]
            syn_mag[b] = -2.5 * np.log10(simps(R(wave) * f_lambda * wave, wave) / v) + 20 + zp

        # For ALLWISE, calculate AB magnitudes and then convert Vega magnitudes using corrections from
        # Jarrett_2011_ApJ_735_112,
        # "Conversion to the monochromatic AB system entails an additional 2.699, 3.339, 5.174, and 6.620 added
        #  to the Vega magnitudes for W1, W2, W3, and W4, respectively"

        for b in ('W1', 'W2', 'W3', 'W4'):
            R = self.R[b]
            f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                    simps(R(wave) * 2.998e10 / wave, wave))
            syn_mag[b] = -2.5 * np.log10(f_nu) + 20 - 48.60 - self.zp[b]  # "+20" to account for wave in A not um
        self.syn_mag = syn_mag

        # Process extra_data
        if self.extra_data:
            for x in self.extra_data:
                b = x['tag']
                R = self.R[b]
                f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                        simps(R(wave) * 2.998e10 / (wave * 1e-8), wave))
                syn_mag[b] = -2.5 * np.log10(f_nu) + self.zp[b]

        # Process colors_data
        if self.colors_data:
            syn_col = {}
            for x in self.colors_data:
                b = x['tag']
                mag = {}
                f_interp = interp1d(wave, f_lambda)

                if b in ['by', 'm1', 'c1']:
                    # Calculates the Stromgren synthetic colours
                    for m in ('u', 'v', 'b', 'y'):
                        mag[m] = -2.5 * log10(simps(x['resp'][m] * f_interp(x['wave'][m]), x['wave'][m]) / x['vega_zp'][m])
                    if x['type'] == 'by':
                        syn_col[b] = mag['b'] - mag['y'] + x['zp'] + 0.003
                    if x['type'] == 'm1':
                        by_c = mag['b'] - mag['y']
                        vb_c = mag['v'] - mag['b']
                        syn_col[b] = vb_c - by_c + x['zp'] + 0.157
                    if x['type'] == 'c1':
                        uv_c = mag['u'] - mag['v']
                        vb_c = mag['v'] - mag['b']
                        syn_col[b] = uv_c - vb_c + x['zp'] + 1.088
            self.syn_col = syn_col
        else:
            self.syn_col = None

        lnlike_m, lnlike_c = 0, 0
        chisq = 0
        for k, v in zip(self.syn_mag.keys(), self.syn_mag.values()):
            z = self.obs_mag[k] - self.syn_mag[k]
            wt = 1 / (z.s ** 2 + sig_ext ** 2)
            chisq += z.n ** 2 * wt
            lnlike_m += -0.5 * (z.n ** 2 * wt - np.log(wt))
        if self.colors_data:
            for k, v in zip(self.syn_col.keys(), self.syn_col.values()):
                z = self.obs_col[k] - self.syn_col[k]
                wt = 1 / (z.s ** 2 + sig_col ** 2)
                chisq += z.n ** 2 * wt
                lnlike_c += -0.5 * (z.n ** 2 * wt - np.log(wt))
            return chisq, lnlike_m, lnlike_c
        else:
            return chisq, lnlike_m
