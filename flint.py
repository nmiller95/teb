import numpy as np
from os.path import join, abspath, dirname
from synphot import SourceSpectrum
from astropy.table import Table
from astropy.units import UnitsWarning
import warnings
import os
import pyvo as vo


__all__ = ['ModelSpectrum']


def make_tag(params):
    """
    Makes unique tag to use in saving the model as a local file
    :param params: :param params: Tuple containing teff, logg, m/h and a/Fe
    :return: String unique to the model
    """
    teff, logg, m_h, afe = params
    if m_h > 0:
        tfmt = "lte{:03.0f}-{:3.1f}+{:3.1f}a{:+3.1f}"
        return tfmt.format(teff / 100, logg, m_h, afe)
    else:
        tfmt = "lte{:03.0f}-{:3.1f}-{:3.1f}a{:+3.1f}"
        return tfmt.format(teff / 100, logg, abs(m_h), afe)


def make_pathname(cache_path, params, source, binning):
    """
    Constructs pathname to binned and un-binned versions of models
    :param cache_path: Path to cache folder
    :param params: Tuple containing teff, logg, m/h and a/Fe
    :param source: Name of model database being used. 'bt-settl' and 'coelho-sed' supported.
    :param binning: Size of bins in Angstrom
    :return: Tuple of pathname to binned and un-binned versions of models
    """
    tag = make_tag(params)
    _f = "{}.{}.dat".format(tag, source)
    model_file_0 = join(cache_path, _f)
    if binning is None:
        model_file = model_file_0
    else:  # Gets pathname for file that's been binned up
        ffmt = "{}.{}_{:04d}.dat"
        model_file = join(cache_path, ffmt.format(tag, source, int(binning)))
    return model_file, model_file_0


def load_spectrum_as_table(s, params, source):
    """
    Attempts to read a specific model from a source catalog of models
    :param s: Service object from VO
    :param params: Tuple containing teff, logg, m/h and a/Fe
    :param source: Name of model database being used. 'bt-settl' and 'coelho-sed' supported.
    :return: astropy.Table containing model wavelength and flux
    """
    teff, logg, m_h, afe = params
    cond_teff = s['teff'] == teff
    cond_logg = s['logg'] == logg
    cond_meta = s['meta'] == m_h
    # Column names depend on source
    if source == 'bt-settl':
        cond_alpha = s['alpha'] == afe
    elif source == 'coelho-sed':
        cond_alpha = s['afe'] == afe
    else:
        raise NameError('model not supported')
    # Restricts table to just the requested parameters
    s = s[cond_teff & cond_logg & cond_meta & cond_alpha]
    try:
        url = (s[0]['Spectrum']).decode("utf-8")
    except IndexError:
        raise FileNotFoundError("Spectrum with specified parameters not found. Check [M/H] and [a/Fe]")
    url += '&format=ascii'
    return Table.read(url, format='ascii.fast_no_header', names=('wave', 'flux'))


def nearest_teff_models(s, params):
    """
    Finds nearest temperature above and below the temperature specified
    :param s: Service object from VO
    :param params: Tuple containing teff, logg, m/h and a/Fe
    :return: Upper and lower temperature that can be read into model
    """
    teff, _, _, _ = params
    # Check which teffs are supported by the model
    s_list = []
    for val in s['teff']:
        if val not in s_list:
            s_list.append(val)
    i = 0
    while teff > s_list[i]:
        i += 1
    return s_list[i],  s_list[i-1]


def valid_teff(s, params, source):
    """
    Checks if specified teff is supported by model choice
    :param s: Service object from VO
    :param params: Tuple containing teff, logg, m/h and a/Fe
    :param source: Model catalog 'bt-settl' or 'coelho-sed'
    :return: True if teff supported, otherwise false
    """
    teff, _, _, _ = params
    if source == 'bt-settl':
        if teff < 400 or teff > 70000:
            raise ValueError("Temperature outside range supported by BT-Settl models. Must be 400K <= Teff <= 70000K")
        elif teff % 100:
            return False
        else:
            return True
    elif source == 'coelho-sed':
        if teff < 3000 or teff > 26000:
            raise ValueError("Temperature outside range supported by Coelho SEDs. Must be 3000K <= Teff <= 26000K")
        # Get list of temperatures supported
        upper, lower = nearest_teff_models(s, params)
        if teff == upper or teff == lower:
            return True
        else:
            return False


def process_spectrum(model, model_file, model_file_0, reload, binning):
    """
    Bins spectrum if binning=True and saves output to cache
    :param model: Unprocessed SED as astropy.Table
    :param model_file: Pathname for binned version
    :param model_file_0: Pathname for un-binned version
    :param reload: Whether to use existing file or load new version
    :param binning: Size of bins in Angstrom
    :return: Binned model as astropy.Table
    """
    model.sort('wave')
    model_g = model.group_by('wave')
    model = model_g.groups.aggregate(np.mean)
    model['flux'].unit = 'FLAM'  # TODO: check this is right for both sets of models
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        if not os.path.isfile(model_file_0):
            model.write(model_file_0, format='ascii')
        if os.path.isfile(model_file_0) and reload:
            model.write(model_file_0, format='ascii', overwrite=True)
        if binning is not None:
            model.add_column(model['wave'] // int(binning), name='bin')
            model_b = model.group_by('bin')
            model = model_b.groups.aggregate(np.mean)
            model.write(model_file, format='ascii', overwrite=True)


class ModelSpectrum(SourceSpectrum):
    cache_path = join(dirname(abspath(__file__)), 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_path = join(cache_path, 'Models')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @classmethod
    def from_parameters(cls, teff, logg, m_h=0, afe=0, binning=10, reload=False, source='bt-settl'):
        params = (teff, logg, m_h, afe)
        model_file, model_file_0 = make_pathname(cls.cache_path, params, source, binning)

        # If file exists (i.e. already downloaded and binned) and you don't want to re-download it, simples!
        if os.path.isfile(model_file) and not reload:
            return SourceSpectrum.from_file(model_file)
        # Get un-binned file if already downloaded
        if binning is not None and os.path.isfile(model_file_0) and not reload:
            model = Table.read(model_file_0, format='ascii')
            process_spectrum(model, model_file, model_file_0, reload, binning)
        else:
            if source == 'bt-settl':
                service = vo.dal.SSAService("http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl&")
            elif source == 'coelho-sed':
                service = vo.dal.SSAService("http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=coelho_sed&")
            else:
                raise ValueError(source, "Invalid source of models specified")
            # Read list of available models
            s = service.search()
            s = s.to_table()
            # Find and load model; interpolate if necessary
            if valid_teff(s, params, source):
                model = load_spectrum_as_table(s, params, source)
                process_spectrum(model, model_file, model_file_0, reload, binning)
            else:
                upper, lower = nearest_teff_models(s, params)
                t_diff = upper - lower
                spectra = []
                for t_step in (upper, lower):
                    t_params = (t_step, logg, m_h, afe)
                    t_model = load_spectrum_as_table(s, t_params, source)
                    model_file, model_file_0 = make_pathname(cls.cache_path, params, source, binning)
                    process_spectrum(t_model, model_file, model_file_0, reload, binning)
                    if source == 'coelho-sed':
                        spectra.append(SourceSpectrum.from_file(model_file)*np.pi)
                    else:
                        spectra.append(SourceSpectrum.from_file(model_file))
                return ((teff - lower) / t_diff) * spectra[0] + ((upper - teff) / t_diff) * spectra[1]

        return SourceSpectrum.from_file(model_file)
