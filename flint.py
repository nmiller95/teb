import numpy as np
from os.path import join, abspath, dirname
from astropy.io.votable import parse
from astropy.io.votable.exceptions import W10, W42
from synphot import SpectralElement, Empirical1D, SourceSpectrum
from synphot import BaseUnitlessSpectrum
from astropy.table import Table
from astropy.units import UnitsWarning
from scipy.special import legendre
import warnings
import os
import pyvo as vo
from math import ceil, floor


__all__ = ['SpectralEnergyDistribution', 'ModelSpectrum', 'Bandpass',
           'DistortionPolynomial']


def load_spectrum_as_table(s, params, source):
    teff, logg, m_h, afe = params
    cond_teff = s['teff'] == teff
    cond_logg = s['logg'] == logg
    cond_meta = s['meta'] == m_h
    if source == 'bt-settl':
        cond_alpha = s['alpha'] == afe
    elif source == 'coelho-sed':
        cond_alpha = s['afe'] == afe
    else:
        raise NameError('model not supported')
    s = s[cond_teff & cond_logg & cond_meta & cond_alpha]
    try:
        url = (s[0]['Spectrum']).decode("utf-8")
    except AttributeError:
        raise FileNotFoundError("Spectrum with specified parameters not found")
    url += '&format=ascii'
    return Table.read(url, format='ascii.fast_no_header')


def model_interpolate_teff(s, teff, logg, m_h, afe, source):
    upper_teff = ceil(teff / 100) * 100
    lower_teff = floor(teff / 100) * 100
    upper_params = (upper_teff, logg, m_h, afe)
    lower_params = (lower_teff, logg, m_h, afe)
    upper_model = load_spectrum_as_table(s, upper_params, source)
    lower_model = load_spectrum_as_table(s, lower_params, source)
    wave = upper_model['col1']
    flux = ((teff-lower_teff)/100)*upper_model['col2'] + ((upper_teff-teff)/100)*lower_model['col2']
    return wave, flux


class ModelSpectrum(SourceSpectrum):
    cache_path = join(dirname(abspath(__file__)), 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_path = join(cache_path, 'BT-Settl')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @classmethod
    def from_parameters(cls, teff, logg, m_h=0, afe=0, binning=10, reload=False,
                        source='bt-settl'):
                        # version='CIFIST2011_2015'):

        ############################################################
        # very very rough example
        if source == 'bt-settl':
            service = vo.dal.SSAService("http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl&")
        elif source == 'coelho':
            service = vo.dal.SSAService("http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=coelho_sed&")
        else:
            raise ValueError(source, "Invalid source of models specified")
        # Read list of available models
        s = service.search()
        s = s.to_table()
        if teff % 100:  # Temperature not in list: load 2 nearest and interpolate
            wave, flux = model_interpolate_teff(s, teff, logg, m_h, afe, source)
        else:
            params = (teff, logg, m_h, afe)
            model = load_spectrum_as_table(s, params, source)
            wave = model['col1']
            flux = model['col2']

        ############################################################

        subdir = {'CIFIST2011_2015': 'CIFIST2011'}
        tag = cls.make_tag(teff, logg, m_h, afe, source)
        # _f = "{}.BT-Settl.{}.fits".format(tag, version)
        # fits_file_0 = join(cls.cache_path, _f)
        if binning is None:
            fits_file = fits_file_0
        else:  # Makes new pathname for a new file that's been binned up
            ffmt = "{}.BT-Settl.{}_{:04d}.fits"
            fits_file = join(cls.cache_path, ffmt.format(tag, version,
                                                         int(binning)))

        # If file exists (i.e. already downloaded and binned) and you don't want to re-download it, simples!
        if os.path.isfile(fits_file) and not reload:
            return SourceSpectrum.from_file(fits_file)

        if binning is not None and os.path.isfile(fits_file_0) and not reload:
            t = Table.read(fits_file_0)

        f_url = ("http://phoenix.ens-lyon.fr/Grids/BT-Settl/" +
                 "{}/SPECTRA/{}.BT-Settl.spec.7.bz2")
        url = f_url.format(subdir[version], tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            try:
                t = Table.read(url, hdu=1, format='fits')
            except OSError:
                raise ValueError(url)
        # t.remove_column('BBFLUX')
        t.sort('WAVELENGTH')
        t_g = t.group_by('WAVELENGTH')
        t = t_g.groups.aggregate(np.mean)
        t['FLUX'].unit = 'FLAM'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            if not os.path.isfile(fits_file_0):
                t.write(fits_file_0)
            if os.path.isfile(fits_file_0) and reload:
                t.write(fits_file_0, overwrite=True)
            if binning is not None:
                t.add_column(t['WAVELENGTH'] // int(binning), name='BIN')
                t_b = t.group_by('BIN')
                t = t_b.groups.aggregate(np.mean)
            t.write(fits_file, overwrite=reload)
        return SourceSpectrum.from_file(fits_file)


class Bandpass(SpectralElement):
    cache_path = join(dirname(abspath(__file__)), 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_path = join(cache_path, 'fps')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @classmethod
    def from_svo(cls, filter_id, keep_neg=False, reload=False):

        facility, filter_name = filter_id.split('/')
        cls.facility = facility
        cls.filter_name = filter_name

        cache_path = join(cls.cache_path, facility)
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        xml_file = join(cache_path, filter_name + '.xml')

        if os.path.isfile(xml_file) and not reload:
            votable = parse(xml_file)
        else:
            _url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID={}'
            url = _url.format(filter_id)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", W42)
                warnings.simplefilter("error", W10)
                try:
                    votable = parse(url)
                except W10:
                    raise ValueError('FilterID {} not found'.format(filter_id))
            votable.to_xml(xml_file)

        meta = dict()
        for field in votable.iter_fields_and_params():
            try:
                meta[field.name] = float(field.value)
            except:
                pass
            try:
                meta[field.name] = field.value.decode()
            except:
                pass

        thruput_table = votable.get_first_table().to_table()
        wavelengths = thruput_table['Wavelength']
        fluxes = thruput_table['Transmission']
        return cls(Empirical1D, points=wavelengths, lookup_table=fluxes,
                   keep_neg=keep_neg, meta=meta)


class DistortionPolynomial(BaseUnitlessSpectrum):

    @classmethod
    def from_coeffs(cls, coeffs, wmin=500, wmax=500000, nwave=499501):
        x = np.linspace(-1, 1, nwave)
        w = wmin + 0.5 * (x + 1) * (wmax - wmin)
        d = np.ones(nwave)
        for n, c in enumerate(coeffs):
            d += c * legendre(n + 1)(x)
        return cls(Empirical1D, points=w, lookup_table=d, keep_neg=True)


class SpectralEnergyDistribution(SourceSpectrum):

    @classmethod
    def from_model(cls, model_spectrum, reddening_law, distortion_polynomial,
                   total_flux):
        return cls(Empirical1D, np.ones(1001))
