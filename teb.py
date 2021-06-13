"""
teb - a python tool for calculating fundamental effective [t]emperatures of [e]clipsing [b]inaries

Authors: Nikki Miller, Pierre Maxted (2021)
"""
import numpy as np
from matplotlib import pylab as plt
import yaml
import _pickle as pickle  # cPickle is faster than pickle
from synphot import ReddeningLaw
from scipy.optimize import minimize
import corner
from flint import ModelSpectrum
from flux2mag import Flux2mag
import flux_ratio_priors as frp
from functions import lnprob, list_to_ufloat, angular_diameters, initial_parameters, \
    run_mcmc_simulations, load_photometry, convergence_plot, print_mcmc_solution, distortion_plot


if __name__ == "__main__":
    # Load photometry data from photometry.yaml
    flux_ratios, extra_data, colors_data = load_photometry()

    ############################################################
    # Load basic, custom and model parameters from config.yaml
    stream = open('config/config.yaml', 'r')
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
        print('Flux ratio priors setup complete.')
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
    logg1, logg2 = config_dict['logg1'], config_dict['logg2']
    if logg1 % 0.5 or logg2 % 0.5:  # TODO: round instead of error
        raise ValueError("Invalid surface gravity - check allowed values in config.yaml")
    if model_library == 'bt-settl-cifist':
        m_h, aFe = (0.0, 0.0)
    elif model_library == 'bt-settl' or model_library == 'coelho-sed':
        m_h = config_dict['m_h']
        aFe = config_dict['aFe']
    else:
        raise ValueError(f"Invalid model SED library specified: {model_library}")

    spec1 = ModelSpectrum.from_parameters(teff1, logg1, m_h, aFe, binning=binning, reload=False, source=model_library)
    spec2 = ModelSpectrum.from_parameters(teff2, logg2, m_h, aFe, binning=binning, reload=False, source=model_library)

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

    if config_dict['override_initial_optimisation']:
        if config_dict['apply_colors']:
            soln['x'] = np.array([config_dict['teff1'], config_dict['teff2'], theta2_in.n, theta2_in.n,
                                  ebv_prior.n, config_dict['sigma_ext'], config_dict['sigma_l'],
                                  config_dict['sigma_c']] + list(soln['x'][8:]))
        else:
            soln['x'] = np.array([config_dict['teff1'], config_dict['teff2'], theta2_in.n, theta2_in.n,
                                  ebv_prior.n, config_dict['sigma_ext'], config_dict['sigma_l']]
                                 + list(soln['x'][7:]))

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

    show_plots = config_dict['show_plots']
    if show_plots:
        # Convergence of chains plot for params excluding distortion coefficients
        convergence_plot(samples, parname, config_dict)

        # Corner plot for all free parameters excluding distortion coefficients
        fig = corner.corner(flat_samples, labels=parname)
        fig.suptitle(f"Corner plot for {name} ({config_dict['run_id']}) \n"
                     f"Model SED source: {config_dict['model']}\n"
                     f"Teff1 = {teff1}, Teff2 = {teff2}, M/H = {m_h}, a/Fe = {aFe}", fontsize=14)
        if config_dict['save_plots']:
            f_name = f"output/{config_dict['run_id']}_{config_dict['name']}_{teff1}_{teff2}_{m_h}_{aFe}" \
                     f"_{binning}A_bins_corner.png"
            plt.savefig(f_name)
        plt.show()

        # Distortion plot with final SED for both stars
        distortion_plot(best_pars, f2m, flux_ratios, theta1_in, theta2_in, spec1, spec2, ebv_prior,
                        redlaw, nc, frp_dictionary, config_dict, flat_samples)

    # Prints best solution from MCMC
    print_mcmc_solution(flat_samples, parname)

    lnlike = lnprob(best_pars, f2m, flux_ratios,
                    theta1_in, theta2_in, spec1, spec2,  # TODO: this should print *output* theta
                    ebv_prior, redlaw, nc,
                    config_dict=config_dict, frp_coeffs=coeffs,
                    verbose=True)
    print('Final log-likelihood = {:0.2f}'.format(lnlike))

    ############################################################
    f_name = f"output/{config_dict['run_id']}_{config_dict['name']}_{teff1}_{teff2}_{m_h}_{aFe}_{binning}A_bins.pkl"
    with open(f_name, 'wb') as output:
        pickle.dump(sampler, output)
