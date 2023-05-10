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
    if m_h >= 0:
        # if extending Teff range beyond 1000-9999 in the future, change this format
        tfmt = "lte{:03.0f}-{:3.1f}+{:3.1f}a{:+3.1f}"
        return tfmt.format(teff, logg, m_h, afe)
    else:
        tfmt = "lte{:03.0f}-{:3.1f}-{:3.1f}a{:+3.1f}"
        return tfmt.format(teff, logg, abs(m_h), afe)


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
    binning: int or None
        Size of bins in Angstrom

    Returns
    -------
    Tuple of pathname to binned and un-binned versions of models
    """
    tag = make_tag(params)
    _f = f"{source}-{tag}.dat"
    model_file_0 = join(cache_path, _f)
    if binning is None:
        model_file = model_file_0
    else:  # Gets pathname for file that's been binned up
        ffmt = "{}-{}_binning={:04d}.dat"
        model_file = join(cache_path, ffmt.format(source, tag, int(binning)))
    return model_file, model_file_0


def load_spectrum_as_table(s, params, source):  # TODO still loads things it doesn't need to
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
    else:
        raise NameError("Specified model source is not supported.")
    # Restricts table to just the requested parameters
    s = s[cond_teff & cond_logg & cond_meta & cond_alpha]
    # print(s[0])
    try:
        url = str(s[0]['Spectrum'], 'utf-8')
    except TypeError:
        url = s[0]['Spectrum']
    except KeyError:
        url = s[0]['Access.Reference']
    except IndexError:
        raise FileNotFoundError(f"Spectrum with teff = {teff}, logg = {logg}, [M/H] = {m_h}, [a/Fe] = {afe} not found "
                                f"in the {source} model catalog. \nCheck that your [M/H] and [a/Fe] are supported.")
    url += '&format=ascii'
    try:
        # Works for python 3.7.0 and astropy 4.2.1 on Scientific Linux
        return Table.read(url, format='ascii.fast_no_header', names=('wave', 'flux'))
    except FileNotFoundError:
        try:
            # Works for python 3.7.3 and astropy 4.0.1 on MacOS
            return Table.read(url, format='ascii', names=('wave', 'flux'))
        except FileNotFoundError:
            raise FileNotFoundError("Problem occurred in astropy.table.Table.read().")


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


def nearest_m_h_models(s, params):
    """
    Finds nearest M/H above and below the M/H specified

    Parameters
    ----------
    s: `pyvo.service`
        Service object from pyVO
    params: tuple
        Must contain teff, logg, m/h and a/Fe

    Returns
    -------
    Upper and lower M/H that can be read into model
    """
    _, _, m_h, _ = params
    # Check which M/H are supported by the model
    s_list = []
    for val in s['meta']:
        if val not in s_list:
            s_list.append(val)
    s_list.sort()
    i = 0
    while m_h > s_list[i]:
        i += 1
    return s_list[i],  s_list[i-1]


def valid_m_h(params, source):
    """
    Checks if specified M/H is supported by model choice

    Parameters
    ----------
    params: tuple
        Must contain teff, logg, m/h and a/Fe
    source: str
        Name of model database being used. Models supported are:
        * bt-settl
        * bt-settl-cifist

    Returns
    -------
    True if M/H supported, otherwise false
    """
    _, _, m_h, _ = params
    if source == 'bt-settl':
        if m_h < -4.0 or m_h > 0.5:
            raise ValueError("M/H outside range supported by BT-Settl models. Must be -4.0 <= M/H <= 0.5")
        elif m_h in [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5]:
            return True
        else:
            return False
    elif source == 'bt-settl-cifist':
        if m_h == 0.0:
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


def interpolate_two_models(params, upper_model, lower_model, upper_val, lower_val, which):
    """
    Performs linear interpolation on two models already loaded

    Parameters
    ----------
    params
    upper_model
    lower_model
    upper_val
    lower_val
    which: str
        Which axis to interpolate along. Options: 'teff', 'logg' or 'm_h'

    Returns
    -------
    Interpolated model
    """
    diff = upper_val - lower_val
    if which == 'teff':
        nominal_val, _, _, _ = params
    elif which == 'logg':
        _, nominal_val, _, _ = params
    elif which == 'm_h':
        _, _, nominal_val, _ = params
    else:
        raise ValueError(f"teb currently does not support interpolation along {which} axis.")

    return ((nominal_val - lower_val) / diff) * upper_model + ((upper_val - nominal_val) / diff) * lower_model


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
    # BT Settl models are limited in the metallicity-alpha fraction space.
    # Must hard-code alpha fraction for successful model retrieval
    if m_h == -0.5:
        afe = 0.2
    elif m_h == 0.0:
        afe = 0.0
    elif m_h < -0.5:
        afe = 0.4
    upper, lower = nearest_teff_models(s, params)
    spectra = []
    for t_step in (upper, lower):
        t_params = (t_step, logg, m_h, afe)
        t_model = load_spectrum_as_table(s, t_params, source)
        model_file, model_file_0 = make_pathname(cache_path, t_params, source, binning)
        process_spectrum(t_model, model_file, model_file_0, reload, binning)
        spectra.append(SourceSpectrum.from_file(model_file))
    return interpolate_two_models(params, spectra[0], spectra[1], upper, lower, which='teff')


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
    spectra = []
    for logg_step in (upper, lower):
        logg_params = (teff, logg_step, m_h, afe)
        logg_model = load_spectrum_as_table(s, logg_params, source)
        model_file, model_file_0 = make_pathname(cache_path, logg_params, source, binning)
        process_spectrum(logg_model, model_file, model_file_0, reload, binning)
        spectra.append(SourceSpectrum.from_file(model_file))
    return interpolate_two_models(params, spectra[0], spectra[1], upper, lower, which='logg')


def interpolate_m_h(s, params, source, cache_path, reload, binning):
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
    upper, lower = nearest_m_h_models(s, params)
    spectra = []
    print(f'metallicity interpolated between: {upper} and {lower}')
    for m_h_step in (upper, lower):
        if m_h_step == -0.5:
            m_h_params = (teff, logg, -0.5, 0.2)
        elif m_h_step == 0.0:
            m_h_params = (teff, logg, 0.0, 0.0)
        else:
            m_h_params = (teff, logg, m_h_step, afe)
        m_h_model = load_spectrum_as_table(s, m_h_params, source)
        model_file, model_file_0 = make_pathname(cache_path, m_h_params, source, binning)
        process_spectrum(m_h_model, model_file, model_file_0, reload, binning)
        spectra.append(SourceSpectrum.from_file(model_file))
    return interpolate_two_models(params, spectra[0], spectra[1], upper, lower, which='m_h')


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
    cache_path = join(cache_path, 'Models')
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

        Returns
        -------
        `synphot.SourceSpectrum`
        """
        params = (teff, logg, m_h, afe)

        # Parameter validity checks
        if not 5.5 > logg > 2.5:
            raise ValueError("teb does not currently support logg less than 2.5 or greater than 5.5")

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
            else:
                raise ValueError(source, "Specified model source is not supported.")

            # Read list of available models
            s = service.search()
            s = s.to_table()

            # Temperature matches available models
            if valid_teff(s, params, source):
                # logg matches available models
                if not logg % 0.5:
                    # M/H matches available models --> no interpolation needed
                    if valid_m_h(params, source):
                        model = load_spectrum_as_table(s, params, source)
                        process_spectrum(model, model_file, model_file_0, reload, binning)
                        return SourceSpectrum.from_file(model_file)
                    # --> M/H interpolation only
                    else:
                        return interpolate_m_h(s, params, source, cls.cache_path, reload, binning)
                # logg doesn't match available models
                else:
                    # --> logg interpolation only
                    if valid_m_h(params, source):
                        return interpolate_logg(s, params, source, cls.cache_path, reload, binning)
                    # --> logg + M/H interpolation
                    else:
                        lower_logg = np.floor(logg * 2) / 2  # Round down to the nearest interval of 0.5
                        upper_logg = lower_logg + 0.5
                        lower_logg_params = (teff, lower_logg, m_h, afe)
                        upper_logg_params = (teff, upper_logg, m_h, afe)
                        lower_logg_model = interpolate_m_h(s, lower_logg_params, source, cls.cache_path, reload,
                                                           binning)
                        upper_logg_model = interpolate_m_h(s, upper_logg_params, source, cls.cache_path, reload,
                                                           binning)
                        return interpolate_two_models(params, upper_logg_model, lower_logg_model,
                                                      upper_logg, lower_logg, which='logg')
            # Temperature doesn't match available models
            else:
                # logg matches available models
                if not logg % 0.5:
                    # M/H matches available models --> Temperature interpolation only
                    if valid_m_h(params, source):
                        return interpolate_teff(s, params, source, cls.cache_path, reload, binning)
                    # --> M/H and temperature interpolation
                    else:
                        upper_teff, lower_teff = nearest_teff_models(s, params)
                        lower_teff_params = (lower_teff, logg, m_h, afe)
                        upper_teff_params = (upper_teff, logg, m_h, afe)
                        lower_teff_model = interpolate_m_h(s, lower_teff_params, source, cls.cache_path, reload,
                                                           binning)
                        upper_teff_model = interpolate_m_h(s, upper_teff_params, source, cls.cache_path, reload,
                                                           binning)
                        return interpolate_two_models(params, upper_teff_model, lower_teff_model,
                                                      upper_teff, lower_teff, which='teff')
                # logg doesn't match available models
                else:
                    # M/H matches available models --> Temperature and logg interpolation
                    if valid_m_h(params, source):
                        lower_logg = np.floor(logg * 2) / 2  # Round down to the nearest interval of 0.5
                        upper_logg = lower_logg + 0.5
                        lower_logg_params = (teff, lower_logg, m_h, afe)
                        upper_logg_params = (teff, upper_logg, m_h, afe)
                        lower_logg_model = interpolate_teff(s, lower_logg_params, source, cls.cache_path, reload, binning)
                        upper_logg_model = interpolate_teff(s, upper_logg_params, source, cls.cache_path, reload, binning)
                        return interpolate_two_models(params, upper_logg_model, lower_logg_model,
                                                      upper_logg, lower_logg, which='logg')
                    # Nothing matches available models --> logg, M/H and temperature interpolation
                    else:
                        upper_m_h, lower_m_h = nearest_m_h_models(s, params)
                        lower_logg = np.floor(logg * 2) / 2  # Round down to the nearest interval of 0.5
                        upper_logg = lower_logg + 0.5
                        spectra = []
                        for m_h_step in (upper_m_h, lower_m_h):
                            lower_logg_params = (teff, lower_logg, m_h_step, afe)
                            upper_logg_params = (teff, upper_logg, m_h_step, afe)
                            lower_logg_model = interpolate_teff(s, lower_logg_params, source, cls.cache_path, reload,
                                                                binning)
                            upper_logg_model = interpolate_teff(s, upper_logg_params, source, cls.cache_path, reload,
                                                                binning)
                            spectra.append(interpolate_two_models(params, upper_logg_model, lower_logg_model,
                                           upper_logg, lower_logg, which='logg'))
                        return interpolate_two_models(params, spectra[0], spectra[1],
                                                      upper_m_h, lower_m_h, which='m_h')

        return SourceSpectrum.from_file(model_file)
