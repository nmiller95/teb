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

    Parameters
    ----------
    params: tuple
        Must contain teff, logg, m/h and a/Fe

    Returns
    -------
    String unique to the model
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

    Parameters
    ----------
    cache_path: str or None
        Path to cache folder
    params: tuple
        Must contain teff, logg, m/h and a/Fe
    source: str
        Name of model database being used. Models supported are:
        * bt-settl
        * bt-settl-cifist
        * coelho-sed
    binning: int or None
        Size of bins in Angstrom

    Returns
    -------
    Tuple of pathname to binned and un-binned versions of models
    """
    tag = make_tag(params)
    _f = f"{source}/{tag}.dat"
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

    Parameters
    ----------
    s: `pyvo.service`
        Service object from pyVO
    params: tuple
        Must contain teff, logg, m/h and a/Fe
    source: str
        Name of model database being used. Models supported are:
        * bt-settl
        * bt-settl-cifist
        * coelho-sed

    Returns
    -------
    `astropy.table.Table` containing model wavelength and flux
    """
    teff, logg, m_h, afe = params
    cond_teff = s['teff'] == teff
    cond_logg = s['logg'] == logg
    cond_meta = s['meta'] == m_h
    # Column names depend on source
    if source == 'bt-settl' or source == 'bt-settl-cifist':
        cond_alpha = s['alpha'] == afe
    elif source == 'coelho-sed':
        cond_alpha = s['afe'] == afe
    else:
        raise NameError("Specified model source is not supported.")
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

    Parameters
    ----------
    s: `pyvo.service`
        Service object from pyVO
    params: tuple
        Must contain teff, logg, m/h and a/Fe

    Returns
    -------
    Upper and lower temperature that can be read into model
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

    Parameters
    ----------
    s: `pyvo.service`
        Service object from pyVO
    params: tuple
        Must contain teff, logg, m/h and a/Fe
    source: str
        Name of model database being used. Models supported are:
        * bt-settl
        * bt-settl-cifist
        * coelho-sed

    Returns
    -------
    True if teff supported, otherwise false
    """
    teff, _, _, _ = params
    if source == 'bt-settl':
        if teff < 400 or teff > 70000:
            raise ValueError("Temperature outside range supported by BT-Settl models. Must be 400K <= Teff <= 70000K")
        elif teff % 100:
            return False
        else:
            return True
    elif source == 'bt-settl-cifist':
        if teff < 1200 or teff > 7000:
            raise ValueError("Temperature outside range supported by BT-Settl-CIFIST. Must be 400K <= Teff <= 7000K")
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
    Bins spectrum (if binning is specified) and saves output to cache

    Parameters
    ----------
    model: `astropy.table.Table`
        Unprocessed SED
    model_file: str
        Pathname for binned version
    model_file_0: str
        Pathname for un-binned version
    reload: bool
        Whether to use existing file or load new version
    binning: int or None
        Size of bins in Angstrom
    """
    model.sort('wave')
    model_g = model.group_by('wave')
    model = model_g.groups.aggregate(np.mean)
    model['flux'].unit = 'FLAM'
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


def interpolate_teff(s, params, source, cache_path, reload, binning):
    """
    Finds closest two models in temperature space and linearly interpolates between them

    Parameters
    ----------
    s: `pyvo.service`
        Service object from pyVO
    params: tuple
        Must contain teff, logg, m/h and a/Fe
    source: str
        Name of model database being used. Models supported are:
        * bt-settl
        * bt-settl-cifist
        * coelho-sed
    cache_path: str or None
        Path to cache folder
    reload: bool
        Whether to use existing file or load new version
    binning: int or None
        Size of bins in Angstrom

    Returns
    -------
    Spectrum that has been linearly interpolated between two nearest temperatures
    """
    teff, logg, m_h, afe = params
    upper, lower = nearest_teff_models(s, params)
    t_diff = upper - lower
    spectra = []  # TODO: at a later date.... make this not repeat
    for t_step in (upper, lower):
        t_params = (t_step, logg, m_h, afe)
        t_model = load_spectrum_as_table(s, t_params, source)
        model_file, model_file_0 = make_pathname(cache_path, params, source, binning)
        process_spectrum(t_model, model_file, model_file_0, reload, binning)
        spectra.append(SourceSpectrum.from_file(model_file))
    return ((teff - lower) / t_diff) * spectra[0] + ((upper - teff) / t_diff) * spectra[1]


def interpolate_logg(s, params, source, cache_path, reload, binning):
    """
    Finds closest two models in logg space and linearly interpolates between them.

    Parameters
    ----------
    s: `pyvo.service`
        Service object from pyVO
    params: tuple
        Must contain teff, logg, m/h and a/Fe
    source: str
        Name of model database being used. Models supported are:
        * bt-settl
        * bt-settl-cifist
        * coelho-sed
    cache_path: str or None
        Path to cache folder
    reload: bool
        Whether to use existing file or load new version
    binning: int or None
        Size of bins in Angstrom

    Returns
    -------
    Spectrum that has been linearly interpolated between two nearest logg grid points
    """
    teff, logg, m_h, afe = params
    lower = np.floor(logg*2)/2  # Round down to the nearest interval of 0.5
    upper = lower + 0.5
    logg_diff = 1.0
    spectra = []
    for logg_step in (upper, lower):
        logg_params = (teff, logg_step, m_h, afe)
        logg_model = load_spectrum_as_table(s, logg_params, source)
        model_file, model_file_0 = make_pathname(cache_path, params, source, binning)
        process_spectrum(logg_model, model_file, model_file_0, reload, binning)
        spectra.append(SourceSpectrum.from_file(model_file))
    return ((logg - lower) / logg_diff) * spectra[0] + ((upper - logg) / logg_diff) * spectra[1]


class ModelSpectrum(SourceSpectrum):
    """
    Model spectrum class.

    For details about the SourceSpectrum output, see
        https://synphot.readthedocs.io/en/latest/synphot/spectrum.html
    """
    # Get pathname for where to make a cache of model files you will download
    cache_path = join(dirname(abspath(__file__)), 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    cache_path = join(cache_path, 'Models')  # TODO: different folders for different models!!
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    @classmethod
    def from_parameters(cls, teff, logg, m_h=0, afe=0, binning=10, reload=False, source='bt-settl'):
        """
        Loads the spectrum/spectra closest to your specified parameters from your specified source.

        Parameters
        ----------
        teff: int
            Effective temperature of model to load, in Kelvin
        logg: float
            Logarithm of surface gravity of model to load, in cgs
        m_h: float, optional
            Metallicity of model to load.
        afe: float, optional
            Alpha fraction of model to load.
        binning: int or None, optional
            Size of wavelength bins to use when binning the model.
        reload: bool
            Whether to re-download the model from SVO.
        source: str
            Name of model database being used. Models supported are:
            * bt-settl
            * bt-settl-cifist
            * coelho-sed

        Returns
        -------
        `synphot.SourceSpectrum`
        """
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
                print("Loading BT-Settl model(s) (Allard et al 2012, RSPTA 370. 2765A)\n "
                      "For more information on these models, see "
                      "http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl")
                service = vo.dal.SSAService(
                    "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl&"
                )
            elif source == 'bt-settl-cifist':
                print("Loading BT-Settl-CIFIST model(s) (Baraffe et al. 2015, A&A 577A, 42B)\n"
                      "For more information on these models, see "
                      "http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-cifist")
                service = vo.dal.SSAService(
                    "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=bt-settl-cifist&"
                )
            elif source == 'coelho-sed':
                print("Loading Coelho Synthetic stellar library (SEDs) model(s) (Coelho 2014, MNRAS 440, 1027C)\n"
                      "CAUTION: Does not support wavelengths > 10um.\n"
                      "For more information on these models, see "
                      "http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=coelho_sed")
                service = vo.dal.SSAService(
                    "http://svo2.cab.inta-csic.es/theory/newov2/ssap.php?model=coelho_sed&"
                )
            else:
                raise ValueError(source, "Specified model source is not supported.")

            # Read list of available models
            s = service.search()
            s = s.to_table()

            # If exact teff & logg model exists, load and process model
            if valid_teff(s, params, source):
                if logg % 0.5 and 5.5 > logg > 2.5:
                    model = load_spectrum_as_table(s, params, source)
                    process_spectrum(model, model_file, model_file_0, reload, binning)
                    return SourceSpectrum.from_file(model_file)
                # If temperature matches a model but logg needs interpolating...
                elif 5.5 > logg > 2.5:
                    return interpolate_logg(s, params, source, cls.cache_path, reload, binning)
                else:
                    raise ValueError("teb does not currently support logg less than 2.5 or greater than 5.5")
            # Otherwise, load and interpolate the two closest models in temperature, and then process
            else:
                # If logg matches a model but teff needs interpolating...
                if logg % 0.5:
                    return interpolate_teff(s, params, source, cls.cache_path, reload, binning)
                # Interpolating along both teff and logg axes
                elif 5.5 > logg > 2.5:
                    lower_logg = np.floor(logg * 2) / 2  # Round down to the nearest interval of 0.5
                    upper_logg = lower_logg + 0.5
                    lower_logg_params = (teff, lower_logg, m_h, afe)
                    upper_logg_params = (teff, upper_logg, m_h, afe)
                    lower_logg_model = interpolate_teff(s, lower_logg_params, source, cls.cache_path, reload, binning)
                    upper_logg_model = interpolate_teff(s, upper_logg_params, source, cls.cache_path, reload, binning)
                    return ((logg - lower_logg) / 1.0) * upper_logg_model + \
                           ((upper_logg - logg) / 1.0) * lower_logg_model
                else:
                    raise ValueError("teb does not currently support logg less than 2.5 or greater than 5.5")

        return SourceSpectrum.from_file(model_file)
