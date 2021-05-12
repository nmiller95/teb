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
import flux_ratios as fr
import yaml


def list_to_ufloat(two_item_list):
    """Turns a two item list from yaml input into a ufloat"""
    return ufloat(two_item_list[0], two_item_list[1])


if __name__ == "__main__":
    # Load and initialise photometric data
    stream = open('photometry_data.yaml', 'r')
    photometry = yaml.safe_load(stream)
    print(photometry)

    for f in photometry['flux_ratios']:
        f['value'] = list_to_ufloat(f['value'])
    for e in photometry['extra_data']:
        e['mag'] = list_to_ufloat(e['mag'])
        e['zp'] = list_to_ufloat(e['zp'])
    for c in photometry['colors_data']:
        c['color'] = list_to_ufloat(c['color'])
    print(photometry)

    # Set up models / run details
    stream = open('config.yaml', 'r')
    parameters = yaml.safe_load(stream)

    print(parameters)



    # f2m = Flux2mag('ASAS_J051753-5406.0', extra_data=extra_data, colors_data=[Gby, Gm1, Gc1])