from uncertainties import ufloat
# noinspection PyUnresolvedReferences
from uncertainties.umath import log10
from scipy.interpolate import interp1d
from scipy.integrate import simps
import numpy as np
from astropy.table import Table
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad


def galex_zp_correction(band, mag):
    """
    Corrects zero-point for GALEX bands using data from Table 4 of Camarota & Holberg (2014), using the distribution of
    points about the observed magnitude of the binary.

    Parameters
    ----------
    band: string
        Photometric band of observation: NUV, FUV
    mag: uncertainties.ufloat
        Observed magnitude and error in band

    Returns
    -------
    ufloat containing correction and error to be applied to the default zeropoint for the specified band
    """
    # Read data from Table 4 of Camarota & Holmberg (2014)
    t = Table.read('Table4complete.txt', format='ascii', data_start=1)
    if band == 'NUV':
        galex_mag, galex_mag_err, synth_mag = t['col7'], t['col8'], t['col10']
        # Check magnitude is within reasonable range for this correction
        if 12.5 <= mag.n <= 15.5:
            sample_width = 0.5
        elif 11.5 <= mag.n < 12.5 or 15.5 < mag.n <= 17:
            sample_width = 1.0
        else:
            return ufloat(0, 0.5)  # TODO: Replace with average scatter of graph and/or alternative method
    elif band == 'FUV':
        galex_mag, galex_mag_err, synth_mag = t['col5'], t['col6'], t['col9']
        # Check magnitude is within reasonable range for this correction
        if 12 <= mag.n <= 17:
            sample_width = 0.5
        elif 11 <= mag.n < 12 or 17 < mag.n <= 20:
            sample_width = 1.0
        else:
            return ufloat(0, 0.5)
    else:
        print('Tried to make correction to GALEX magnitude zeropoint but band not read correctly.')
        return ufloat(0, 0)

    # Select sample of stars and find mean value + standard deviation of sample
    sample_filter = []
    for i, _ in enumerate(galex_mag):
        if abs(mag.n - galex_mag[i]) < sample_width:
            if synth_mag[i] > 0.0:
                sample_filter.append(True)
            else:
                sample_filter.append(False)
        else:
            sample_filter.append(False)
    zp_offset = np.mean((galex_mag - synth_mag)[sample_filter])
    # zp_mad = mad((galex_mag-synth_mag)[sample_filter])
    zp_stdev = np.std((galex_mag - synth_mag)[sample_filter])

    return ufloat(zp_offset, zp_stdev)


class Flux2mag:
    """
    Stores observed magnitudes and colours and calculates synthetic magnitudes and colours

    __init__: Accesses and consolidates magnitude and photometric colors of a star.
    __call__: Generates synthetic magnitudes and colors from integrating a reference spectrum of Vega.
    """

    def __init__(self, name, extra_data=None, colors_data=None):
        """
        Reads in bandpass data from package files. Calculates a standardised response function and pivot wavelength of
        each bandpass. Retrieves standard photometric data from SIMBAD and processes any extra data supplied.

        Parameters
        ----------
        name: str
            Name of star, as understood by SIMBAD
        extra_data: list, optional
            List of dictionaries containing information about non-standard magnitudes.
            Each dictionary must contain:
                * tag: str,
                    Photometric band name/label (must be unique)
                * mag: `uncertainties.ufloat`
                    Observed AB magnitude with standard error
                * zp: `uncertainties.ufloat`
                    Photometric band zero-point and its standard error
                * wave: array_like
                    Wavelength array for response function, in angstrom
                * resp: array_like
                    Response function
        colors_data: list, optional
            List of dictionaries containing information about photometric colors.
            Each dictionary must contain:
                * tag: string
                    Photometric color name/label (must be unique), e.g. '(b-y)_1'
                * type: string
                    Type of color. Currently supported are:
                        * Stromgren: 'by', 'm1', 'c1'
                * color: `uncertainties.ufloat`
                    Observed color with standard error.
                * zp: `uncertainties.ufloat`
                    Colour zero-point and standard error.
                * wave: array_like
                    Wavelength array for response function, in angstrom
                * resp: array_like
                    Response function
                * vega_zp: dict
                    Contains zero-points for magnitudes used to construct colors, e.g. u,b,v,y, in Vega system
        """

        self.name = name

        # Zero-point (Vega magnitude system) information as a dictionary
        # Zero-point for GALEX bands is from Camarota & Holberg with correction made using the scatter of points
        # in Figure 4 about the observed magnitude of the binary
        # N.B. For WISE, these are offsets from Vega to AB magnitudes from Jarret et al.
        #   "This Vega basis has an overall systematic uncertainty of ∼1.45%." (0.016 mag)
        self.zp = {
            'FUV': ufloat(-48.60, 0.134),  # ufloat(-48.43, 0.374),
            'NUV': ufloat(-48.60, 0.154),  # ufloat(-49.04, 0.883),
            'G': ufloat(25.6874, 0.0028),
            'BP': ufloat(25.3385, 0.0028),
            'RP': ufloat(24.7479, 0.0038),
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

        # GALEX
        for b in ['FUV', 'NUV']:
            t = Table.read('Response/EA-{}_im.tbl'.format(b.lower()), format='ascii', names=['w', 'A'])
            R[b] = interp1d(t['w'], t['A'], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(t['w'], t['A'])

        # Gaia (E)DR3
        names = ['wave', 'G', 'e_G', 'BP', 'e_BP', 'RP', 'e_RP']
        t = Table.read('Response/GaiaEDR3_passbands.dat',
                       format='ascii', names=names)
        w = t['wave'] * 10
        for b in ['G', 'BP', 'RP']:
            i = (t[b] < 99).nonzero()
            R[b] = interp1d(w[i], t[b][i], bounds_error=False, fill_value=0)
            w_pivot[b] = wp(w[i], t[b][i])

        # 2MASS
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
        else:
            self.extra_data = None

        self.R = R
        self.w_pivot = w_pivot

        # Prepare colours data (if given)
        if colors_data:
            for x in colors_data:
                # Strömgren colours
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
                    # Zero point and error is average colour o-c offset and standard deviation
                    if x['type'] == 'by':
                        x['zp'] = ufloat(-0.0014, 0.0045)
                    elif x['type'] == 'm1':
                        x['zp'] = ufloat(0.0243, 0.0062)
                    elif x['type'] == 'c1':
                        x['zp'] = ufloat(-0.0102, 0.0083)

                    x['wave'] = stromgren_w
                    x['resp'] = stromgren_r
                    x['vega_zp'] = stromgren_v

            self.colors_data = colors_data

        # -------------- RETRIEVE STANDARD PHOTOMETRY WITH VIZIER + SIMBAD QUERIES -------------- #
        # Catalogue query functions for Gaia EDR3, 2MASS, GALEX and WISE.
        # This uses WISE All Sky values instead of ALLWISE for consistency with flux ratio calibration and
        # because these are more reliable at the bright end for W1 and W2.

        # Search Vizier for Gaia, WISE and GALEX magnitudes.
        vizier_r = Vizier(columns=["*", "+_r"])
        v = vizier_r.query_object(name, catalog=['I/350/gaiaedr3', 'II/311/wise', 'II/335/galex_ais'])

        obs_mag = dict()
        try:
            obs_mag['G'] = ufloat(v[0][0]['Gmag'], v[0][0]['e_Gmag'])
            obs_mag['BP'] = ufloat(v[0][0]['BPmag'], v[0][0]['e_BPmag'])
            obs_mag['RP'] = ufloat(v[0][0]['RPmag'], v[0][0]['e_RPmag'])
        except IndexError:
            raise AttributeError("Something went wrong reading Gaia (E)DR3 magnitudes. "
                                 "Try checking star name is correct and resolved by SIMBAD")

        # Retrieve WISE photometry direct from catalogue
        for b in ['W1', 'W2', 'W3', 'W4']:
            try:
                if type(v[1][0][f'{b}mag']) == np.float32 and type(v[1][0][f'e_{b}mag']) == np.float32:
                    obs_mag[b] = ufloat(v[1][0][f'{b}mag'], v[1][0][f'e_{b}mag'])
                else:
                    print(f"Unable to find magnitude for {b} band in WISE catalog (II/311/wise).")
            except IndexError:
                print(f"Unable to find magnitude for {b} band in WISE catalog (II/311/wise).")

        # Retrieve GALEX photometry direct from catalogue
        for b in ['FUV', 'NUV']:
            try:
                if type(v[2][0][f'{b}mag']) == np.float64:
                    obs_mag[b] = ufloat(v[2][0][f'{b}mag'], v[2][0][f'e_{b}mag'])
                    self.zp[b] -= galex_zp_correction(b, obs_mag[b])  # TODO testing
                    print(f"Correcting GALEX photometric ZP: Observed {b} mag: {obs_mag[b]}, New ZP: {self.zp[b]}")
            except IndexError:
                print(f"Unable to find magnitude for {b} band in GALEX catalog (II/335/galex_ais).")

        # Search SIMBAD for 2MASS J, H, Ks magnitudes.
        sb = Simbad()
        sb.add_votable_fields('flux(J)', 'flux_error(J)', 'flux(H)', 'flux_error(H)', 'flux(K)', 'flux_error(K)')
        sb_tab = sb.query_object(name)
        for b in ['J', 'H', 'K']:
            try:
                if b == 'K':
                    obs_mag['Ks'] = ufloat(sb_tab['FLUX_{}'.format(b)][0], sb_tab['FLUX_ERROR_{}'.format(b)][0])
                else:
                    obs_mag[b] = ufloat(sb_tab['FLUX_{}'.format(b)][0], sb_tab['FLUX_ERROR_{}'.format(b)][0])
            except KeyError:
                print(f"Unable to find magnitude for 2MASS {b} band via SIMBAD search.")

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

    def __call__(self, wave, f_lambda, sig_ext=0, sig_col=0, apply_colors=False):
        """
        Integrates flux over the defined passbands and returns chi-square of fit to observed magnitudes.

        Parameters
        ----------
        wave: `synphot.SourceSpectrum.waveset`
            Wavelength range over which the flux is defined, in Angstrom
        f_lambda: array_like
            Must be call-able with argument lambda = wavelength in Angstrom
        sig_ext: float, optional
            Amount of external noise to the magnitudes
        sig_col: float, optional
            Amount of external noise to the colors
        apply_colors: bool. optional
            Whether to include colours in the chi-square fit to observed magnitudes

        Returns
        -------
        Chi-square of the fit to observed magnitudes, and log likelihoods
        """

        syn_mag = dict()

        # Calculate synthetic magnitude for GALEX bands
        for b in ['FUV', 'NUV']:
            try:
                if self.obs_mag[b]:
                    R = self.R[b]
                    f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                            simps(R(wave) * 2.998e10 / (wave * 1e-8), wave))
                    syn_mag[b] = -2.5 * np.log10(f_nu) + self.zp[b]
            except KeyError:
                pass

        # Calculate synthetic magnitude for Gaia bands
        p_a = 0.7278  # collecting (pupil) area of Gaia
        hc9 = 1.986445824e-16  # 10^9 hc
        # /10000 is conversion from erg/s/cm^2/A to W/m^2/nm
        for b in ['G', 'BP', 'RP']:
            photon_flux = p_a * simps(self.R[b](wave) * wave * f_lambda / 10000, wave) / hc9
            syn_mag[b] = -2.5 * np.log10(photon_flux) + self.zp[b]

        # Calculate synthetic magnitude for 2MASS bands
        # "+20" to account for wave in A not um
        for b in ['J', 'H', 'Ks']:
            try:
                if self.obs_mag[b]:
                    v = self.f_vega[b]
                    zp = self.zp[b]
                    R = self.R[b]
                    syn_mag[b] = -2.5 * np.log10(simps(R(wave) * f_lambda * wave, wave) / v) + 20 + zp
            except KeyError:
                pass

        # Calculate synthetic magnitude for WISE bands
        # For ALLWISE, calculate AB magnitudes and then convert Vega magnitudes using corrections from
        # Jarrett_2011_ApJ_735_112: "Conversion to the monochromatic AB system entails an additional 2.699, 3.339,
        # 5.174, and 6.620 added to the Vega magnitudes for W1, W2, W3, and W4, respectively"
        for b in ('W1', 'W2', 'W3', 'W4'):
            try:
                if self.obs_mag[b]:
                    R = self.R[b]
                    f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                            simps(R(wave) * 2.998e10 / wave, wave))
                    syn_mag[b] = -2.5 * np.log10(f_nu) + 20 - 48.60 - self.zp[b]  # +20 to account for wave in A not um
                self.syn_mag = syn_mag
            except KeyError:
                pass

        # Process extra_data
        if self.extra_data:
            for x in self.extra_data:
                b = x['tag']
                R = self.R[b]
                f_nu = (simps(f_lambda * R(wave) * wave, wave) /
                        simps(R(wave) * 2.998e10 / (wave * 1e-8), wave))
                syn_mag[b] = -2.5 * np.log10(f_nu) + self.zp[b]

        # Process colors_data
        if apply_colors and self.colors_data:
            syn_col = {}
            for x in self.colors_data:
                b = x['tag']
                mag = {}
                f_interp = interp1d(wave, f_lambda)

                # Calculates the Stromgren synthetic colours
                if x['type'] in ['by', 'm1', 'c1']:
                    for m in ('u', 'v', 'b', 'y'):
                        mag[m] = -2.5 * log10(simps(x['resp'][m] * f_interp(x['wave'][m]),
                                                    x['wave'][m]) / x['vega_zp'][m])
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

        # Compare observed and synthetic magnitudes and colours and calculate lnlike and fit chi-squared
        lnlike_m, chisq = 0, 0
        for k in self.syn_mag.keys():
            try:
                z = self.obs_mag[k] - self.syn_mag[k]
                wt = 1 / (z.s ** 2 + sig_ext ** 2)
                chisq += z.n ** 2 * wt
                lnlike_m += -0.5 * (z.n ** 2 * wt - np.log(wt))
            except KeyError:
                pass

        if apply_colors and self.syn_col:
            lnlike_c = 0
            for k in self.syn_col.keys():
                z = self.obs_col[k] - self.syn_col[k]
                wt = 1 / (z.s ** 2 + sig_col ** 2)
                chisq += z.n ** 2 * wt
                lnlike_c += -0.5 * (z.n ** 2 * wt - np.log(wt))
            return chisq, lnlike_m, lnlike_c
        else:
            return chisq, lnlike_m
