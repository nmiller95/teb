"""
Main file for running all this nonsense
Module will eventually be named teb (temperatures of eclipsing binaries)
TODO Look into how best to get this working: command line based or something to read into scripts?
TODO: adapt flint to be part of this module
TODO: fix flint's model load breakages
"""
import numpy as np
from matplotlib import pylab as plt
from astropy.table import Table
from scipy.integrate import simps
from uncertainties import ufloat, covariance_matrix, correlation_matrix
from uncertainties.umath import log10
from scipy.interpolate import interp1d
import flint
from synphot import ReddeningLaw
import emcee
import corner
from multiprocessing import Pool
import pickle
from scipy.optimize import minimize
from response import extra_data, colors_data
from flux2mag import Flux2mag
import flux_ratio_priors as frp
from flux_ratios import FluxRatio
import yaml
from functions import lnprob


def list_to_ufloat(two_item_list):
    """Turns a two item list from yaml input into a ufloat"""
    return ufloat(two_item_list[0], two_item_list[1])


def goat(teff, logg, mh, afe, bin_size):
    # this is where the intricate interpolation goodness will happen - perhaps export to functions.py if it gets long
    try:
        if teff % 50:
            # select temperature step below and above; work out interpolation weights
            pass
        if logg % 0.5:
            # select logg above and below; etc
            pass
        if mh % 0.1:
            # as before.
            pass
        else:  # everything is simple
            pass
        spec_a = flint.ModelSpectrum.from_parameters(teff, 4.0, binning=bin_size, M_H=mh, aFe=afe, reload=True)
        spec_b = flint.ModelSpectrum.from_parameters(teff, 4.0, binning=bin_size, M_H=mh, aFe=afe, reload=True)
        return 0.8*spec_a + 0.2*spec_b
    except ValueError:
        print("something went wrong loading your model. try better values or something")
    # it might be easier just to heckin' round the temperature and logg to nearest acceptable value,
    # and accept that m/h and a/Fe is going to be 0 for these models...


if __name__ == "__main__":
    # Load and initialise photometric data from photometry_data.yaml
    stream = open('config/photometry_data.yaml', 'r')
    photometry = yaml.safe_load(stream)
    # Flux ratios - initialised with FluxRatio from flux_ratios.py
    try:
        flux_ratios = dict()
        for f in photometry['flux_ratios']:
            fr = FluxRatio(f['tag'], f['type'], f['value'][0], f['value'][1])
            tag, d = fr()
            flux_ratios[tag] = d
    except KeyError:
        print("No flux ratios provided in photometry_data.yaml")
        flux_ratios = None
    # Extra magnitudes - read wavelength and response read from specified file
    try:
        for e in photometry['extra_data']:
            e['mag'] = list_to_ufloat(e['mag'])
            e['zp'] = list_to_ufloat(e['zp'])
            try:
                t = Table.read(e['file'], format='ascii')
                e['wave'] = np.array(t['col1'])
                e['resp'] = np.array(t['col2'])
            except OSError:
                raise SystemExit(f"Unable to read {e['file']}.")
        extra_data = photometry['extra_data']
    except KeyError:
        print("No additional magnitudes provided in photometry_data.yaml")
        extra_data = None
    # Colors - simple conversion from list to ufloat
    try:
        for c in photometry['colors_data']:
            c['color'] = list_to_ufloat(c['color'])
        colors_data = photometry['colors_data']
    except KeyError:
        print("No colours provided in photometry_data.yaml")
        colors_data = None

    # Load basic, custom and model parameters from config.yaml
    stream = open('config/config.yaml', 'r')
    parameters = yaml.safe_load(stream)
    # Create Flux2mag object from name and photometry data
    try:
        name = parameters['name']
        f2m = Flux2mag(name, extra_data, colors_data)
    except IndexError:
        raise SystemExit("Star name not resolved by SIMBAD")

    # Flux ratio prior calculation with methods from flux_ratio_priors.py
    if parameters['apply_fratio_prior']:  # TODO: change this stuff to logging rather than print to screen?
        print('Configuring flux ratio prior settings...')
        tref1, tref2, tab1, tab2, method, fratio, teff1, teff2 = frp.configure()
        print('Fitting V-K vs. Teff for specified subset of stars...')
        coeffs = frp.frp_coeffs(tref1, tref2, tab1, tab2, method=method)
        print('Calculating flux ratio priors...')
        frp_dictionary = frp.flux_ratio_priors(fratio, teff1, teff2, tref1, tref2, coeffs, method=method)
        print('Flux ratio priors setup complete.')
    else:
        frp_dictionary = None

    # Parallax load and apply zeropoint
    gaia_zp = ufloat(-0.017, 0.011)  # Gaia EDR3. ZP error fixed at DR2 value for now
    plx = list_to_ufloat(parameters['plx']) - gaia_zp

    # Loading models (interpolating if required)
    binning = parameters['binning']
    tref1 = parameters['tref1']
    tref2 = parameters['tref2']
    m_h = parameters['m_h']
    aFe = parameters['aFe']

    # spec1 = flint.ModelSpectrum.from_parameters(tref1, 4.0, binning=binning, M_H=0.0, aFe=0.0, reload=True)
    # url = "http://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011/SPECTRA/lte064-4.0-0.0a+0.0.BT-Settl.spec.7.bz2"
    # Table.read(url, hdu=1, format='fits')

    # No detectable NaI lines so E(B-V) must be very close to 0 - see 2010NewA...15..444K
    ebv_prior = list_to_ufloat(parameters['ebv'])
    redlaw = ReddeningLaw.from_extinction_model('mwavg')

    # Angular diameter = 2*R/d = 2*R*parallax = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
    # R_Sun = 6.957e8 m. parsec = 3.085677581e16 m
    r1 = list_to_ufloat(parameters['r1'])
    r2 = list_to_ufloat(parameters['r2'])
    theta1 = 2 * plx * r1 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
    theta2 = 2 * plx * r2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
    # print('theta1 = {:0.4f} mas'.format(theta1))
    # print('theta2 = {:0.4f} mas'.format(theta2))
    theta_cov = covariance_matrix([theta1, theta2])[0][1]
    theta_cor = correlation_matrix([theta1, theta2])[0][1]
    # print('cov(theta_1,theta2) = {:0.2e}'.format(theta_cov))
    # print('cor(theta_1,theta2) = {:0.2f}'.format(theta_cor))

    # Getting the lnlike set up and print initial result
    Teff1 = 6400
    Teff2 = 6270
    # Copy starting values to new variables
    theta1_ = theta1.n
    theta2_ = theta2.n
    ebv_ = ebv_prior.n
    sigma_ext = 0.008
    sigma_l = 0.01
    sigma_c = 0.005
    nc = parameters['n_coeffs']
    params = [Teff1, Teff2, theta1_, theta2_, ebv_, sigma_ext, sigma_l]  # , sigma_c]
    params = params + [0] * nc
    # params = params + [0] * nc  # This from single vs double distortion

    parname = ['T_eff,1', 'T_eff,2', 'theta_1', 'theta_2', 'E(B-V)', 'sigma_ext', 'sigma_l']  # ,'sigma_c']
    parname = parname + ["c_1,{}".format(j + 1) for j in range(nc)]
    # parname = parname + ["c_2,{}".format(j+1) for j in range(nc)]

    for pn, pv in zip(parname, params):
        print('{} = {}'.format(pn, pv))

    # lnlike = lnprob(params, f2m, flux_ratios,
    #                 theta1, theta2, spec1, spec2,
    #                 ebv_prior, redlaw, nc, verbose=True, debug=False)
    # print('Initial log-likelihood = {:0.2f}'.format(lnlike))
