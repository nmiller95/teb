"""
Main file for running all this nonsense
Module will eventually be named teb (temperatures of eclipsing binaries)
TODO Look into how best to get this working: command line based or something to read into scripts?
TODO: fix flint's model load breakages
"""
import numpy as np
from matplotlib import pylab as plt
from astropy.table import Table
from scipy.integrate import simps
from uncertainties import ufloat, covariance_matrix, correlation_matrix
from uncertainties.umath import log10 as ulog10
from scipy.interpolate import interp1d
import flint
from synphot import ReddeningLaw
import emcee
import corner
from multiprocessing import Pool
import pickle
from scipy.optimize import minimize
from flux2mag import Flux2mag
import flux_ratio_priors as frp
from flux_ratios import FluxRatio
import yaml
from functions import lnprob, list_to_ufloat, angular_diameters, initial_parameters


if __name__ == "__main__":
    # TODO: stick some of this photometry stuff in flux_ratios.py or functions.py
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

    ############################################################
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
        coeffs = None
        frp_dictionary = None

    # Angular diameters
    theta1_in, theta2_in = angular_diameters(parameters)
    theta_cov = covariance_matrix([theta1_in, theta2_in])[0][1]
    theta_cor = correlation_matrix([theta1_in, theta2_in])[0][1]

    # Reddening - prior from config.yaml and reddening law from flint
    ebv_prior = list_to_ufloat(parameters['ebv'])
    redlaw = ReddeningLaw.from_extinction_model('mwavg')

    ############################################################
    # Loading models (interpolating if required)
    binning = parameters['binning']
    tref1 = parameters['tref1']
    tref2 = parameters['tref2']
    m_h = parameters['m_h']
    aFe = parameters['aFe']

    # Total bodge. TODO: this, properly
    spec1 = flint.ModelSpectrum.from_parameters(6300, 4.0, binning=binning, reload=False)
    spec2 = flint.ModelSpectrum.from_parameters(6200, 4.0, binning=binning, reload=False)

    ############################################################
    # Getting the lnlike set up and print initial result
    nc = parameters['n_coeffs']
    params, parname = initial_parameters(parameters, theta1_in, theta2_in, ebv_prior)

    for pn, pv in zip(parname, params):
        print('{} = {}'.format(pn, pv))

    lnlike = lnprob(params, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,
                    ebv_prior, redlaw, nc,
                    config_dict=parameters, frp_coeffs=coeffs,
                    verbose=True, debug=False)
    print('Initial log-likelihood = {:0.2f}'.format(lnlike))

    ############################################################
    # Nelder-Mead optimisation
    nll = lambda *args: -lnprob(*args)
    args = (f2m, flux_ratios, theta1_in, theta2_in,
            spec1, spec2, ebv_prior, redlaw, nc)
    soln = minimize(nll, params, args=args, method='Nelder-Mead')

    # Re-initialise log likelihood function with optimised solution
    lnlike = lnprob(soln.x, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,
                    ebv_prior, redlaw, nc,
                    config_dict=parameters, frp_coeffs=coeffs,
                    verbose=True)
