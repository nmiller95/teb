import os
import numpy as np
from uncertainties import ufloat
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from astropy.table import Table
from astroquery.vizier import Vizier
import logging
import astropy.units as u
from functions import get_parallax
from calspec import getprofile

logger = logging.getLogger(__name__)


class Flux2mag:
    """
    Stores and computed magnitudes, colours and flux ratios

    __init__: Initialise instance of class star.
    __call__: Generate synthetic magnitudes, colors and flux ratios.
    """

    def __init__(self, star_name, star_data=None):
        """
        Class to store star data and compare it to synthetic photometry
        Star data (photometry, parallax, radii) can be loaded from the
        dictionary star_data or obtained from on-line catalogues.

        Parameters
        ----------
        star_name: str
            Name of star - resolvable by SIMBAD if star_data=None
        star_data: dict, optional
            Dictionary containing photometry, parallax and radii.
            Use make_files to create a YAML file that can be loaded as a
            dictionary with the correct format.
        """

        def loadfiltertable(f_name: str, phot, newfilter=False):
            if newfilter:
                filtertable, phot = getprofile(f_name, photon=None)
            else:
                filtertable = getprofile(f_name, -1)
            wave = filtertable['Wavelength']
            resp = filtertable['Transmission']
            # Normalise spectral response function here
            if phot:
                resp /= simpson(wave*resp, x=wave)
            else:
                resp /= simpson(resp, x=wave)
            t = interp1d(wave, resp, bounds_error=False, fill_value=0)
            if newfilter:
                piv = np.sqrt(simpson(resp*wave, x=wave) / simpson(resp/wave, x=wave))
                return t, phot, piv
            else:
                return t

        # Load photometry filter database
        dbpath = os.path.join('config','database.csv')
        # Put bool ahead of str in converters so True,False are boolean
        converters = {'*': [int, float, bool, str]}
        database = Table.read(dbpath, converters=converters)

        self.filters = {}  # Transmission curve interpolation functions
        for db in database:
            filtername = db['filtername']
            if filtername in ['by','m1','c1']:  # Stromgren uvby indices
                filterdata = {'type':'col'}
                for k in db.colnames:
                    if not k in ['filtername','pivot']:
                        filterdata[k] = db[k]
            else:
                filterdata = {'type':'mag'}
                for k in db.colnames:
                    if not k in ['filtername']:
                        filterdata[k] = db[k]
                filterdata['T'] = loadfiltertable(filtername, db['photon'])
            self.filters[filtername] = filterdata

        # Stromgren filters for computing by, m1, c1
        # zero-point calibration in calspec.py assumes photon=False
        self.uvby = {}
        for b in ['u','v','b','y']:
            filtername = f'Generic/Stromgren.{b}'
            self.uvby[b] = loadfiltertable(filtername, False)

        if star_data is None:  # Populate star_data from catalogues
            self.obs_mag = {}

            # Gaia DR3 parallax and applied zeropoint correction
            self.parallax = get_parallax(star_name)

            # Define the bands and catalogues for the observed magnitudes to be fetched.
            # 'f2c' dict converts filtername to column names with the
            # help of the string formats 'cfmt' and 'efmt' (for the error).
            # Column names are also used as the keys to store magnitudes
            # 'f2k' is the filter name to dict key - can the same as f2c
            unpack = {
                'Gaia DR3': {
                    'cat': 'I/355/gaiadr3',
                    'f2c': {'GAIA/GAIA3.G': 'G', 'GAIA/GAIA3.Gbp': 'BP', 'GAIA/GAIA3.Grp': 'RP'},
                    'f2k': {'GAIA/GAIA3.G': 'G', 'GAIA/GAIA3.Gbp': 'Gbp', 'GAIA/GAIA3.Grp': 'Grp'},
                    'cfmt': '{}mag', 'efmt': 'e_{}mag',
                },
                '2MASS': {
                    'cat': 'II/246/out',
                    'f2c': {'2MASS/2MASS.J': 'J', '2MASS/2MASS.H': 'H', '2MASS/2MASS.Ks': 'K'},
                    'f2k': {'2MASS/2MASS.J': 'J', '2MASS/2MASS.H': 'H', '2MASS/2MASS.Ks': 'Ks'},
                    'cfmt': '{}mag', 'efmt': 'e_{}mag',
                },
                'GALEX': {
                    'cat': 'II/335/galex_ais',
                    'f2c': {'GALEX/GALEX.FUV': 'FUV', 'GALEX/GALEX.NUV': 'NUV'},
                    'cfmt': '{}mag', 'efmt': 'e_{}mag',
                },
                'Tycho-2': {
                    'cat': 'I/259/tyc2',
                    'f2c': {'TYCHO/TYCHO.B_MvB': 'BT', 'TYCHO/TYCHO.V_MvB': 'VT'},
                    'cfmt': '{}mag', 'efmt': 'e_{}mag',
                },
                'SkyMapper DR4': {
                    'cat': 'II/379/smssdr4',
                    'f2c': {'SkyMapper/SkyMapper.u': 'u', 'SkyMapper/SkyMapper.v': 'v'},
                    'cfmt': '{}PSF', 'efmt': 'e_{}PSF',
                },
                'PAN-STARSS': {
                    'cat': 'II/349/ps1',
                    'f2c': {'PAN-STARRS/PS1.g': 'g', 'PAN-STARRS/PS1.r': 'r', 'PAN-STARRS/PS1.i': 'i',
                            'PAN-STARRS/PS1.z': 'z', 'PAN-STARRS/PS1.y': 'y'},
                    'cfmt': '{}mag', 'efmt': 'e_{}mag',
                },
                'ALLWISE': {
                    'cat': 'II/328/allwise',
                    'f2c': {'WISE/WISE.W1': 'W1', 'WISE/WISE.W2': 'W2', 'WISE/WISE.W3': 'W3'},
                    'cfmt': '{}mag', 'efmt': 'e_{}mag',
                },
            }
            # Where f2k wasn't given explicitly, it's identical to f2c
            for entry in unpack.values():
                entry.setdefault('f2k', entry['f2c'])

            # Vizier photometry catalogue query. Note use of 'all' here
            # instead of '*' to _really_ get all the columns avaiable.
            vizier_r = Vizier(columns=["all", "+_r"])
            cats = [entry['cat'] for entry in unpack.values()]
            star_name_ = star_name.replace(' ','_')
            v = vizier_r.query_object(star_name_, catalog=cats, radius=2 * u.arcsec)
            if len(v) == 0:
                msg = f'No data returned from Vizier for target {star_name_}'
                raise AttributeError(msg)

            # Copy over Vizier results where applicable
            for name in unpack:
                cat = unpack[name]['cat']
                if cat in v.keys():
                    r = v[v.keys().index(cat)]
                    ns = len(r)
                    if ns > 1:
                        m = f'Found {ns} {name} sources within 2" of {star_name}'
                        logger.info(m)
                    r = r[0]
                    f2c = unpack[name]['f2c']
                    f2k = unpack[name]['f2k']
                    for f in f2c:  # Loop over filter names
                        cfmt = unpack[name]['cfmt']
                        efmt = unpack[name]['efmt']
                        tag = f2c[f]
                        cv = cfmt.format(tag)  # Column name for value
                        ce = efmt.format(tag)  # Column name for error
                        try:
                            if np.ma.is_masked(r[cv]) or np.ma.is_masked(r[ce]):
                                m = f'No {f2c[f]} magnitude available for {star_name} in {name} - skipping'
                                logger.info(m)
                                continue
                            umag = ufloat(r[cv], r[ce], tag=f)
                        except KeyError:
                            continue
                        if not (np.isfinite(umag.n) and np.isfinite(umag.s)):
                            continue
                        key = f2k[f]
                        self.obs_mag[key] = umag
                        if f in self.filters:
                            magmin = self.filters[f]['magmin']
                            magmax = self.filters[f]['magmax']
                            if (r[cv] < magmin) or (r[cv] > magmax):
                                m = f'{f2c[f]} magnitude {r[cv]:0.2f} outside zero-point calibration range '
                                m += f'{magmin:0.2f} - {magmax:0.2f}'
                                logger.warning(m)
                        else:
                            m = f'Filter {f} missing from zero-point database'
                            logger.warning(m)
                else:
                    m = f'No {name} source found within 2" of {star_name} - skipping'
                    logger.info(m)

        else:
            # Load photometry, parallax, colours and flux ratios from star_data
            # Angular diameter = 2*R/d
            #                  = 2*R*parallax
            #                  = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
            # R_Sun = 6.957e8 m. parsec = 3.085677581e16 m
            plx = ufloat(*star_data['parallax'])
            r1 = ufloat(*star_data['primary_radius'])
            if 'radius_ratio' in star_data:
                if 'secondary_radius' in star_data:
                    m = 'Both radius_ratio and secondary_radius in star data. Choose one!'
                    raise ValueError(m)
                r2 = ufloat(*star_data['radius_ratio']) * r1
            else:
                r2 = ufloat(*star_data['secondary_radius'])
            _const_ = 2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
            t1 = _const_ * plx * r1
            t2 = _const_ * plx * r2
            star_data['theta1'] = t1
            star_data['theta2'] = t2
            logger.info(f' Radius_1 = {r1:0.4f} R_SunN')
            logger.info(f' Radius_2 = {r2:0.4f} R_SunN')
            logger.info(f' Parallax = {plx:0.4f} mas')
            logger.info(f' theta_1 = {t1:0.4f} mas')
            logger.info(f' theta_2 = {t2:0.4f} mas')

            try:
                self.ebv = ufloat(*star_data['ebv'])
            except KeyError:
                self.ebv = None
                logger.warning('No prior on E(B-V)')

            self.obs_mag = {}
            if 'magnitudes' in star_data:
                for mag in star_data['magnitudes']:
                    key = mag['tag']
                    filtername = mag['band']
                    umag = ufloat(*mag['mag'], tag=filtername)
                    self.obs_mag[key] = umag
            logger.info(f'Loaded {len(self.obs_mag)} magnitudes.')

            self.obs_col = {}
            if 'colors' in star_data:
                for color in star_data['colors']:
                    key = color['tag']
                    colorname = color['type']
                    ucol = ufloat(*color['color'], tag=colorname)
                    self.obs_col[key] = ucol
            logger.info(f"{len(self.obs_col)} color indices loaded.")

            self.obs_rat = {}
            if 'flux_ratios' in star_data:
                for flux_ratio in star_data['flux_ratios']:
                    key = flux_ratio['tag']
                    fn = flux_ratio['band']  # filtername
                    flux_ratio = ufloat(*flux_ratio['value'], tag=fn)
                    self.obs_rat[key] = flux_ratio
                    if not fn in self.filters:
                        T, photon, pivot = loadfiltertable(fn, None, newfilter=True)
                        self.filters[fn] = {'photon': photon,
                                            'pivot': pivot,
                                            'T': T}
            logger.info(f"{len(self.obs_rat)} color indices loaded.")

    # ------------------------------------------------------

    def __call__(self, wave, flux, flux_ratio, sigma_m, sigma_r, sigma_c):
        """
        Calculate synthetic photometry magnitudes, colours, and flux ratios.
        All bands and colors must be in config/database.csv

        Parameters
        ----------
        wave: `synphot.SourceSpectrum.waveset`
            Wavelength range over which the flux is defined, in Angstrom
        flux: array_like
           Flux of source (f_lambda in ergs.s-1.cm-2.A-1)
        flux_ratio: array_like
            Flux ratio on the same wavelength scale are f_lambda:
        sigma_m: array_like
            External error on the magnitudes
        sigma_r: array_like
            External error on the flux ratios
        sigma_c: array_like
            External error on the colors

        Returns
        -------
        loglikelihoods and chisqs of mags, cols, fratios - lists of ufloats.
        """
        # Process magnitudes
        syn_mag = {}
        syn_mag1 = {}
        syn_mag2 = {}
        lnlike_m = 0
        chisq_m = 0
        flux1 = flux / (1 + flux_ratio)
        flux2 = flux1 * flux_ratio
        for tag in self.obs_mag:
            umag = self.obs_mag[tag]
            fn = umag.tag  # filter name
            photon = self.filters[fn]['photon']
            vega = self.filters[fn]['vega']
            T = self.filters[fn]['T']
            zp = ufloat(self.filters[fn]['zp'], self.filters[fn]['zp_err'])
            sigma_x = self.filters[fn]['sigma_x']  # Scatter around zp calib
            s_ = ufloat(0, sigma_x)
            if vega:
                if photon:
                    f_lambda = simpson(wave * flux * T(wave), x=wave)
                    f_lambda1 = simpson(wave * flux1 * T(wave), x=wave)
                    f_lambda2 = simpson(wave * flux2 * T(wave), x=wave)
                else:
                    f_lambda = simpson(flux * T(wave), x=wave)
                    f_lambda1 = simpson(flux1 * T(wave), x=wave)
                    f_lambda2 = simpson(flux2 * T(wave), x=wave)
                syn_mag[tag] = -2.5 * np.log10(f_lambda) + zp + s_
                syn_mag1[tag] = -2.5 * np.log10(f_lambda1) + zp + s_
                syn_mag2[tag] = -2.5 * np.log10(f_lambda2) + zp + s_
            else:
                # Using Bessel & Murphy, 2012 PASP 124 140, equation (A15)
                pivot = self.filters[fn]['pivot']
                c_ = pivot ** 2 * 1e-10 / 2.99792e8
                if photon:
                    f_nu = simpson(wave * flux * T(wave), x=wave) * c_
                    f_nu1 = simpson(wave * flux1 * T(wave), x=wave) * c_
                    f_nu2 = simpson(wave * flux2 * T(wave), x=wave) * c_
                else:
                    f_nu = simpson(flux * T(wave), x=wave) * c_
                    f_nu1 = simpson(flux1 * T(wave), x=wave) * c_
                    f_nu2 = simpson(flux2 * T(wave), x=wave) * c_
                syn_mag[tag] = -2.5 * np.log10(f_nu) + zp + s_
                syn_mag1[tag] = -2.5 * np.log10(f_nu1) + zp + s_
                syn_mag2[tag] = -2.5 * np.log10(f_nu2) + zp + s_
            z = self.obs_mag[tag] - syn_mag[tag]
            wt = 1 / (z.s ** 2 + sigma_m ** 2)
            chisq_m += z.n ** 2 * wt
            lnlike_m += -0.5 * (z.n ** 2 * wt - np.log(wt))
        self.syn_mag = syn_mag
        self.syn_mag1 = syn_mag1
        self.syn_mag2 = syn_mag2

        # Process colors_data
        syn_col = {}
        lnlike_c = 0
        chisq_c = 0
        color_types = [s.tag for s in self.obs_col.values()]
        synth_col = {}
        if len(color_types) > 0:
            b_ = -2.5 * np.log10(simpson(flux * self.uvby['b'](wave), x=wave))
            y_ = -2.5 * np.log10(simpson(flux * self.uvby['y'](wave), x=wave))
            d = self.filters['by']
            zp_by = ufloat(d['zp'], d['zp_err'])
            s_ = ufloat(0, d['sigma_x'])  # Scatter around zp calibration
            synth_col['by'] = b_ - y_ + zp_by + s_
            if 'm1' in color_types or 'c1' in color_types:
                v_ = -2.5 * np.log10(simpson(flux * self.uvby['v'](wave), x=wave))
                d = self.filters['m1']
                zp_m1 = ufloat(d['zp'], d['zp_err'])
                s_ = ufloat(0, d['sigma_x'])  # Scatter around zp calibration
                synth_col['m1'] = (v_ - b_) - (b_ - y_) + zp_m1 + s_
                if 'c1' in color_types:
                    u_ = -2.5 * np.log10(simpson(flux * self.uvby['u'](wave), x=wave))
                    d = self.filters['c1']
                    zp_c1 = ufloat(d['zp'], d['zp_err'])
                    s_ = ufloat(0, d['sigma_x'])  # Scatter around zp calibration
                    synth_col['c1'] = (u_ - v_) - (v_ - b_) + zp_c1 + s_

        for tag in self.obs_col:
            ucol = self.obs_col[tag]
            if ucol.tag not in synth_col:
                raise NotImplementedError(f'Color {ucol.tag} not implemented')
            syn_col[tag] = synth_col[ucol.tag]
            z = self.obs_col[tag] - syn_col[tag]
            wt = 1 / (z.s ** 2 + sigma_c ** 2)
            chisq_c += z.n ** 2 * wt
            lnlike_c += -0.5 * (z.n ** 2 * wt - np.log(wt))
        self.syn_col = syn_col

        # Flux ratios
        lnlike_r = 0
        chisq_r = 0
        syn_rat = {}
        for tag in self.obs_rat:
            urat = self.obs_rat[tag]
            fn = urat.tag  # filter name
            photon = self.filters[fn]['photon']
            T = self.filters[fn]['T']
            if photon:
                f_ratio = simpson(wave * flux_ratio * T(wave), x=wave)
            else:
                # pivot = self.filters[fn]['pivot']
                f_ratio = simpson(flux_ratio * T(wave), x=wave)
            syn_rat[tag] = f_ratio
            z = self.obs_rat[tag] - f_ratio
            wt = 1 / (z.s ** 2 + sigma_r ** 2)
            chisq_r += z.n ** 2 * wt
            lnlike_r += -0.5 * (z.n ** 2 * wt - np.log(wt))
        self.syn_rat = syn_rat

        return chisq_m, lnlike_m, chisq_c, lnlike_c, lnlike_r, chisq_r
