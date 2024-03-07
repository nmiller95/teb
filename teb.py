"""
teb - a python tool for calculating fundamental effective [t]emperatures of [e]clipsing [b]inaries

Authors: Nikki Miller, Pierre Maxted (2021)
"""
import _pickle as pickle  # cPickle is faster than pickle
import getopt
import os.path
from os import mkdir
import sys

import corner
import numpy as np
import yaml
from matplotlib import pylab as plt
from scipy.optimize import minimize
from synphot import ReddeningLaw

import flux_ratio_priors as frp
from flint import ModelSpectrum
from flux2mag import Flux2mag
from functions import lnprob, list_to_ufloat, angular_diameters, initial_parameters, \
    run_mcmc_simulations, load_photometry, convergence_plot, print_mcmc_solution, synthetic_optical_lratios
from make_config_files import make_config, make_photometry_data, make_flux_ratio_priors


def inputs(argv):
    def usage():
        print('\nCorrect usage:\n--------------')
        print('    teb.py -c <configfile> -p <photometryfile> -f <frpfile>')
        print('\nteb assumes input files are in subdirectory config/')
        print('\nIf no input file names are specified, the defaults are: \n    --config = \"config.yaml\"'
              '\n    --photometry = \"photometry_data.yaml\" \n    --frp = \"flux_ratio_priors.yaml\"')
        print('To try a fit with synthetic UVBRI flux ratios (not recommended for science), '
              '\nadd the argument \"--synth\"')
        print('\nYou can create empty configuration files using')
        print('    teb.py -m <filename> or teb.py --makefile <filename>')
        print('Filename extension does not need to be specified.\n')

    c_file, p_file, f_file = None, None, None
    synth_lratios = False
    try:
        opts, args = getopt.getopt(argv, "hc:p:f:s:m:",
                                   ["help", "config=", "photometry=", "frp=", "synth", "makefile="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-c", "--config"):
            c_file = arg
        elif opt in ("-p", "--photometry"):
            p_file = arg
        elif opt in ("-f", "--frp"):
            f_file = arg
        elif opt in ("-s", "--synth"):
            synth_lratios = True
        elif opt in ("-m", "--makefile"):
            print(f'Generating blank configuration files with tag: {arg}\n')
            make_config(arg)
            make_photometry_data(arg)
            make_flux_ratio_priors(arg)
            sys.exit()
        else:
            assert False, 'unhandled option'
    return c_file, p_file, f_file, synth_lratios


if __name__ == "__main__":

    print("""
    teb -- a python tool for calculating fundamental effective temperatures for 
    stars in eclipsing binaries
    
    Written by N. J. Miller and P. F. L. Maxted (2020-21)
    Please cite: Miller, Maxted & Smalley (2020) and Miller, Maxted et al (2022)
    
    Most recent version of teb is stored at https://github.com/nmiller95/teb
    Contact nikkimillerastro@gmail.com with questions or suggestions
    
    """)

    # Load file names from command line inputs
    config_file, photometry_file, frp_file, synth_lratios = inputs(sys.argv[1:])
    if config_file is None:
        config_file = "config.yaml"
    if photometry_file is None:
        photometry_file = "photometry_data.yaml"
    if frp_file is None:
        frp_file = "flux_ratio_priors.yaml"
    print(f"teb will perform the calculations using these input files: "
          f"\n--config = {config_file} \n--photometry = {photometry_file} \n--frp = {frp_file}\n")

    # Load photometry data from photometry.yaml
    flux_ratios, extra_data, colors_data = load_photometry(photometry_file)

    ############################################################
    # Load basic, custom and model parameters from config.yaml
    # config_name = input("Configuration file name: config/")
    try:
        stream = open('config/' + config_file, 'r')
    except FileNotFoundError as err:
        print(err)
        sys.exit()
    config_dict = yaml.safe_load(stream)
    # Create Flux2mag object from name and photometry data
    try:
        name = config_dict['name']
        f2m = Flux2mag(name, extra_data, colors_data)
    except IndexError:
        raise SystemExit("Star name not resolved by SIMBAD")

    # Flux ratio prior calculation with methods from flux_ratio_priors.py
    if config_dict['apply_fratio_prior']:
        print('Configuring flux ratio prior settings...')
        tref1, tref2, tab1, tab2, method, fratio, teff1, teff2 = frp.configure()
        print('Fitting V-K vs. Teff for specified subset of stars...')
        coeffs = frp.frp_coeffs(tref1, tref2, tab1, tab2, method=method)
        print('Calculating flux ratio priors...')
        frp_dictionary = frp.flux_ratio_priors(fratio, teff1, teff2, tref1, tref2, coeffs, method=method)
        print('Flux ratio priors setup complete.\n')
    else:
        coeffs = None
        frp_dictionary = None

    # Angular diameters
    theta1_in, theta2_in = angular_diameters(config_dict)

    # Reddening - prior from config.yaml and reddening law from flint
    ebv_prior = list_to_ufloat(config_dict['ebv'])
    redlaw = ReddeningLaw.from_extinction_model('mwavg')

    ############################################################
    # Loading models (interpolating if required)
    model_library = config_dict['model_sed']
    binning = config_dict['binning']
    teff1, teff2 = config_dict['teff1'], config_dict['teff2']
    if round(teff1) > 9999 or round(teff2) > 9999:
        raise SystemExit("teb only supports Teff < 10000 K")
    elif round(teff1) < 1000 or round(teff2) < 1000:
        raise SystemExit("teb only supports Teff >= 1000 K")
    logg1, logg2 = config_dict['logg1'], config_dict['logg2']
    if model_library == 'bt-settl-cifist':
        m_h, aFe = (0.0, 0.0)
    elif model_library == 'bt-settl':
        m_h = config_dict['m_h']
        aFe = config_dict['aFe']
    else:
        raise ValueError(f"Invalid model SED library specified: {model_library}")

    print("\n------------------------------------\n"
          "Loading and interpolating model SEDs"
          "\n------------------------------------")
    print("\nPrimary component\n-----------------")
    spec1 = ModelSpectrum.from_parameters(teff1, logg1, m_h, aFe, binning=binning, reload=False, source=model_library)
    print("\nSecondary component\n-------------------")
    spec2 = ModelSpectrum.from_parameters(teff2, logg2, m_h, aFe, binning=binning, reload=False, source=model_library)
    print('\n')

    ############################################################
    # Synthetic optical flux ratios (don't use for real science)
    if synth_lratios:
        v_ratio = yaml.safe_load(open(f'config/{frp_file}', 'r'))['flux_ratio']
        synthetic_optical_lratios(config_dict, spec1, spec2, theta1_in, theta2_in, redlaw, v_ratio, flux_ratios)

    ############################################################
    # Getting the lnlike set up and print initial result
    nc = config_dict['n_coeffs']
    params, parname = initial_parameters(config_dict, theta1_in, theta2_in, ebv_prior)

    for pn, pv in zip(parname, params):
        print('{} = {}'.format(pn, pv))

    lnlike = lnprob(params, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,
                    ebv_prior, redlaw, nc,
                    config_dict=config_dict, frp_coeffs=coeffs,
                    verbose=True, debug=False)
    print('Initial log-likelihood = {:0.2f}'.format(lnlike))

    ############################################################
    # Nelder-Mead optimisation
    nll = lambda *args: -lnprob(*args)
    args = (f2m, flux_ratios, theta1_in, theta2_in,
            spec1, spec2, ebv_prior, redlaw, nc, config_dict, coeffs)
    print("Finding initial solution with Nelder-Mead optimisation...")
    soln = minimize(nll, params, args=args, method='Nelder-Mead')

    # Print solutions
    for pn, pv in zip(parname, soln.x):
        print('{} = {}'.format(pn, pv))

    # Re-initialise log likelihood function with optimised solution
    lnlike = lnprob(soln.x, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,
                    ebv_prior, redlaw, nc,
                    config_dict=config_dict, frp_coeffs=coeffs,
                    verbose=True)
    # Print solutions
    print('Final log-likelihood = {:0.2f}'.format(lnlike))

    ############################################################
    # Run MCMC simulations
    print("Running MCMC simulations...")
    n_steps, n_walkers = (config_dict['mcmc_n_steps'], config_dict['mcmc_n_walkers'])
    sampler = run_mcmc_simulations(args, config_dict, soln, n_steps=n_steps, n_walkers=n_walkers)

    # Retrieve output from sampler and print key attributes
    af = sampler.acceptance_fraction
    print(f'Median acceptance fraction = {np.median(af)}')
    best_index = np.unravel_index(np.argmax(sampler.lnprobability), (n_walkers, n_steps))
    best_lnlike = np.max(sampler.lnprobability)
    print(f'Best log(likelihood) = {best_lnlike} in walker {best_index[0]} at step {best_index[1]}')
    best_pars = sampler.chain[best_index[0], best_index[1], :]

    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=n_steps // 2, thin=8, flat=True)

    # Prints the best solution from MCMC
    print_mcmc_solution(flat_samples, parname)

    lnlike = lnprob(best_pars, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,  # TODO: this should print *output* theta?
                    ebv_prior, redlaw, nc,
                    config_dict=config_dict, frp_coeffs=coeffs,
                    verbose=True)
    print('Final log-likelihood = {:0.2f}'.format(lnlike))

    # AIC and BIC calculation
    # Counts the number of photometry data used in order to calculate the AIC and BIC
    n_photometry_data = len(f2m.obs_mag)
    if flux_ratios:
        n_photometry_data += len(flux_ratios)
    if frp_dictionary:
        n_photometry_data += len(frp_dictionary)
    # Counts the number of other parameters used in the fit
    if config_dict['apply_colors']:
        n_parameters = 8
    else:
        n_parameters = 7
    if config_dict['distortion'] == 1:
        n_coeffs_total = nc
    elif config_dict['distortion'] == 2:
        n_coeffs_total = nc*2
    else:
        n_coeffs_total = 0
    aic = 2 * (n_coeffs_total + n_parameters) - 2 * np.log(lnlike)
    bic = (n_coeffs_total + n_parameters) * np.log(n_photometry_data) - 2 * np.log(lnlike)
    print(f'AIC: {round(aic, 3)} \nBIC: {round(bic, 3)}')

    # Prepare output directory to save plots and chain
    if not os.path.isdir('output/'):
        mkdir('output/')

    show_plots = config_dict['show_plots']
    if show_plots:
        # Convergence of chains plot for params excluding distortion coefficients
        convergence_plot(samples, parname, config_dict)

        # Corner plot for all free parameters excluding distortion coefficients
        fig = corner.corner(flat_samples, labels=parname)
        fig.suptitle(f"Corner plot for {name} ({config_dict['run_id']}) \n"
                     f"Model SED source: {config_dict['model_sed']}\n"
                     f"Teff1 = {teff1}, Teff2 = {teff2}, M/H = {m_h}, a/Fe = {aFe}", fontsize=14)
        if config_dict['save_plots']:
            f_name = f"output/{config_dict['run_id']}_{config_dict['name']}_{teff1}_{teff2}_{m_h}_{aFe}" \
                     f"_{binning}A_bins_corner.png"
            plt.savefig(f_name)
        plt.show()

        # Distortion plot with final SED for both stars  # TODO: fix this - hint below
        # File "<ipython-input-8-52a4d834a98f>", line 49, in distortion_plot
        #  redlaw, nc, config_dict, frp_coeffs, return_flux=True)
        # ValueError: too many values to unpack (expected 5)
        # distortion_plot(best_pars, f2m, flux_ratios, theta1_in, theta2_in, spec1, spec2, ebv_prior,
        #                 redlaw, nc, frp_dictionary, config_dict, flat_samples)

    ############################################################
    f_name = f"output/{config_dict['run_id']}_{config_dict['name']}_{teff1}_{teff2}_{m_h}_{aFe}_{binning}A_bins.pkl"
    with open(f_name, 'wb') as output:
        pickle.dump(sampler, output)
