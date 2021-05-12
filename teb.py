"""
Main file for running all this nonsense
Module will eventually be named teb (temperatures of eclipsing binaries)
TODO Look into how best to get this working: command line based or something to read into scripts?
TODO Fix this horrid mess one step at a time (split into functions at least)
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
import yaml
import sys


def list_to_ufloat(two_item_list):
    """Turns a two item list from yaml input into a ufloat"""
    return ufloat(two_item_list[0], two_item_list[1])


if __name__ == "__main__":
    # Load and initialise photometric data from photometry_data.yaml
    stream = open('photometry_data.yaml', 'r')
    photometry = yaml.safe_load(stream)
    try:
        for f in photometry['flux_ratios']:
            f['value'] = list_to_ufloat(f['value'])
        flux_ratios = photometry['flux_ratios']
    except KeyError:
        print("No flux ratios provided in photometry_data.yaml")
        flux_ratios = None
    try:
        for e in photometry['extra_data']:
            e['mag'] = list_to_ufloat(e['mag'])
            e['zp'] = list_to_ufloat(e['zp'])
        flux_ratios = photometry['extra_data']
    except KeyError:
        print("No additional magnitudes provided in photometry_data.yaml")
        extra_data = None
    try:
        for c in photometry['colors_data']:
            c['color'] = list_to_ufloat(c['color'])
        colors_data = photometry['colors_data']
    except KeyError:
        print("No colours provided in photometry_data.yaml")
        colors_data = None

    # Load basic, custom and model parameters from config.yaml
    stream = open('config.yaml', 'r')
    parameters = yaml.safe_load(stream)

    try:
        name = parameters['name']
        f2m = Flux2mag(name, extra_data, colors_data)
    except IndexError:
        raise SystemExit("Star name not resolved by SIMBAD")

    plx = list_to_ufloat(parameters['plx'])
    r1 = list_to_ufloat(parameters['r1'])
    r2 = list_to_ufloat(parameters['r2'])
    ebv_prior = list_to_ufloat(parameters['ebv'])
    print(parameters)

    if parameters['apply_fratio_prior']:
        print('Configuring flux ratio prior settings...')
        tref1, tref2, tab1, tab2, method, fratio, teff1, teff2 = frp.configure()
        print('Fitting V-K vs. Teff for specified subset of stars...')
        coeffs = frp.frp_coeffs(tref1, tref2, tab1, tab2, method=method)
        print('Calculating flux ratio priors...')
        frp_dictionary = frp.flux_ratio_priors(fratio, teff1, teff2, tref1, tref2, coeffs, method=method)
        print('Flux ratio priors setup complete.')
