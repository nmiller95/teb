"""
Main file for running all this nonsense
Module will eventually be named teb (temperatures of eclipsing binaries)
TODO Look into how best to get this working: command line based or something to read into scripts?
"""
import numpy as np
from matplotlib import pylab as plt
# from uncertainties import ufloat, covariance_matrix, correlation_matrix
import flint
from synphot import ReddeningLaw
import corner
from scipy.optimize import minimize
from flux2mag import Flux2mag
import flux_ratio_priors as frp
import yaml
import _pickle as pickle  # cPickle is faster than pickle
from functions import lnprob, list_to_ufloat, angular_diameters, initial_parameters, \
    run_mcmc_simulations, load_photometry, convergence_plot


if __name__ == "__main__":
    # Load photometry data from photometry.yaml
    flux_ratios, extra_data, colors_data = load_photometry()

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
    if parameters['apply_fratio_prior']:
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
    # theta_cov = covariance_matrix([theta1_in, theta2_in])[0][1]
    # theta_cor = correlation_matrix([theta1_in, theta2_in])[0][1]

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

    # Load models
    spec1 = flint.ModelSpectrum.from_parameters(6350, 4.0, binning=binning, reload=False)
    spec2 = flint.ModelSpectrum.from_parameters(6200, 4.0, binning=binning, reload=False)
    print('success!')
    breakpoint()

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
            spec1, spec2, ebv_prior, redlaw, nc, parameters)
    print("Finding initial solution with Nelder-Mead optimisation...")
    soln = minimize(nll, params, args=args, method='Nelder-Mead')

    # Print solutions
    for pn, pv in zip(parname, soln.x):
        print('{} = {}'.format(pn, pv))

    # Re-initialise log likelihood function with optimised solution
    lnlike = lnprob(soln.x, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,
                    ebv_prior, redlaw, nc,
                    config_dict=parameters, frp_coeffs=coeffs,
                    verbose=True)
    # Print solutions
    print('Final log-likelihood = {:0.2f}'.format(lnlike))

    ############################################################
    # Run MCMC simulations
    print("Running MCMC simulations...")
    n_steps, n_walkers = (parameters['mcmc_n_steps'], parameters['mcmc_n_walkers'])
    sampler = run_mcmc_simulations(args, parameters, soln, n_steps=n_steps, n_walkers=n_walkers)

    # Retrieve output from sampler and print key attributes
    af = sampler.acceptance_fraction
    print(f'Median acceptance fraction = {np.median(af)}')
    best_index = np.unravel_index(np.argmax(sampler.lnprobability), (n_walkers, n_steps))
    best_lnlike = np.max(sampler.lnprobability)
    print(f'Best log(likelihood) = {best_lnlike} in walker {best_index[0]} at step {best_index[1]}')
    best_pars = sampler.chain[best_index[0], best_index[1], :]

    samples = sampler.get_chain()
    flat_samples = sampler.get_chain(discard=n_steps // 2, thin=8, flat=True)

    show_plots = parameters['show_plots']
    if show_plots:
        # Convergence of chains...
        convergence_plot(samples, parname)
        # Corner plot for all free parameters
        fig = corner.corner(flat_samples, labels=parname)
        plt.show()

    # TODO: also stick this in a function
    for i, pn in enumerate(parname):
        val = flat_samples[:, i].mean()
        err = flat_samples[:, i].std()
        ndp = 1 - min(0, np.floor((np.log10(err))))
        fmt = '{{:0.{:0.0f}f}}'.format(ndp)
        vstr = fmt.format(val)
        estr = fmt.format(err)
        print('{} = {} +/- {}'.format(pn, vstr, estr))

    lnlike = lnprob(best_pars, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,  # TODO: this should print *output* theta
                    ebv_prior, redlaw, nc,
                    config_dict=parameters, frp_coeffs=coeffs,
                    verbose=True)
    print('Final log-likelihood = {:0.2f}'.format(lnlike))

    ############################################################
    # TODO: check this is stable i.e. what went wrong with reading in the AI Phe ones.
    f_name = f"{parameters['run_id']}_{parameters['name']}_{tref1}_{tref2}_{m_h}_{aFe}_{binning}_bins.pkl"
    with f_name as output:
        pickle.dump(sampler, output, pickle.HIGHEST_PROTOCOL)
