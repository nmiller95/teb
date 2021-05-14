import numpy as np
from os.path import join, abspath, dirname
from astropy.io.votable import parse
from astropy.io.votable.exceptions import W10, W42
from synphot import SpectralElement, Empirical1D, SourceSpectrum
from synphot import BaseUnitlessSpectrum, ReddeningLaw
from astropy.table import Table
from astropy.units import UnitsWarning
from scipy.special import legendre
import warnings
import os


__all__ = ['SpectralEnergyDistribution', 'ModelSpectrum', 'Bandpass',
           'DistortionPolynomial']


class ModelSpectrum(SourceSpectrum):
    cache_path = join(dirname(abspath(__file__)), 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_path = join(cache_path, 'BT-Settl')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    def make_tag(Teff, logg, M_H, aFe):
        if M_H > 0:
            tfmt = "lte{:03.0f}-{:3.1f}+{:3.1f}a{:+3.1f}"
            return tfmt.format(Teff / 100, logg, M_H, aFe)
        else:
            tfmt = "lte{:03.0f}-{:3.1f}-{:3.1f}a{:+3.1f}"
            return tfmt.format(Teff / 100, logg, abs(M_H), aFe)

    @classmethod
    def from_parameters(cls, Teff, logg, M_H=0, aFe=0, binning=10, reload=False,
                        version='CIFIST2011_2015'):

        subdir = {'CIFIST2011_2015': 'CIFIST2011'}
        tag = cls.make_tag(Teff, logg, M_H, aFe)
        _f = "{}.BT-Settl.{}.fits".format(tag, version)
        fits_file_0 = join(cls.cache_path, _f)
        if binning is None:
            fits_file = fits_file_0
        else:
            ffmt = "{}.BT-Settl.{}_{:04d}.fits"
            fits_file = join(cls.cache_path, ffmt.format(tag, version,
                                                         int(binning)))

        if os.path.isfile(fits_file) and not reload:
            return SourceSpectrum.from_file(fits_file)

        if binning is not None and os.path.isfile(fits_file_0) and not reload:
            T = Table.read(fits_file_0)

        f_url = ("http://phoenix.ens-lyon.fr/Grids/BT-Settl/" +
                 "{}/SPECTRA/{}.BT-Settl.spec.7.bz2")
        url = f_url.format(subdir[version], tag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            try:
                T = Table.read(url, hdu=1, format='fits')
            except OSError:
                raise ValueError(url)
        T.remove_column('BBFLUX')
        T.sort('WAVELENGTH')
        T_g = T.group_by('WAVELENGTH')
        T = T_g.groups.aggregate(np.mean)
        T['FLUX'].unit = 'FLAM'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            if not os.path.isfile(fits_file_0):
                T.write(fits_file_0)
            if os.path.isfile(fits_file_0) and reload:
                T.write(fits_file_0, overwrite=True)
            if binning is not None:
                T.add_column(T['WAVELENGTH'] // int(binning), name='BIN')
                T_b = T.group_by('BIN')
                T = T_b.groups.aggregate(np.mean)
            T.write(fits_file, overwrite=reload)
        return SourceSpectrum.from_file(fits_file)


class Bandpass(SpectralElement):
    cache_path = join(dirname(abspath(__file__)), 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_path = join(cache_path, 'fps')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @classmethod
    def from_svo(cls, FilterID, keep_neg=False, reload=False):

        facility, filtername = FilterID.split('/')
        cls.facility = facility
        cls.filtername = filtername

        cache_path = join(cls.cache_path, facility)
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        xml_file = join(cache_path, filtername + '.xml')

        if os.path.isfile(xml_file) and not reload:
            votable = parse(xml_file)
        else:
            _url = 'http://svo2.cab.inta-csic.es/svo/theory/fps/fps.php?ID={}'
            url = _url.format(FilterID)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", W42)
                warnings.simplefilter("error", W10)
                try:
                    votable = parse(url)
                except W10:
                    raise ValueError('FilterID {} not found'.format(FilterID))
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
