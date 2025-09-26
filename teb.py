"""
teb - a python tool for calculating fundamental effective [t]emperatures of [e]clipsing [b]inaries

Authors: Nikki Miller, Pierre Maxted (2025)
"""

version = 20250826

import os
import sys
import numpy as np
##import _pickle as pickle  # cPickle is faster than pickle
import getopt
import yaml
from scipy.optimize import minimize
from synphot import ReddeningLaw
from astropy.table import Table, Column
from astropy.io import fits
from flux_ratio_priors import Flux_ratio_priors
from flint import ModelSpectrum
from flux2mag import Flux2mag
from functions import lnprob, initial_parameters, run_mcmc_simulations
from make_file import make_file
from uncertainties import ufloat
from scipy.integrate import simpson
import astropy.units as u
import warnings
from datetime import datetime

def inputs(argv):
    def usage():
        print('Usage: teb.py [-c config_file] [-m star_name]')
        print('\nOptions:')
        print('  -c, --config')
        print('          Configuration file name')
        print('  -m, --make-file')
        print('          Create new input star data file')
        print('  -o, --over-write')
        print('          Over-write existing files')
        print('\nteb assumes input files are in subdirectory config/')
        print('\nstar_name must be resolvable by SIMBAD or in the form Jhhhmmmss.s+ddmmss.s')
        print(' Replace spaces in the star name with \"_\", e.g. \"AI_Phe\"')
        print(' Star name is used to find data for the target from online catalogues.')

    config_file = 'config.yaml'
    overwrite = False
    make_file_ = False
    try:
        opts, args = getopt.getopt(argv, "hoc:m:",
                           ["help", "over-write", "config=", "make-file="])
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-c", "--config"):
            config_file = arg
        elif opt in ("-o", "--over-write"):
            overwrite = True
        elif opt in ("-m", "--make-file"):
            make_file_ = True
            star_name = arg
        elif opt in ("-o", "--over-write"):
            overwrite = True
        else:
            assert False, 'unhandled option'

    if make_file_:
        print(f'Making input file {star_name}.yaml\n')
        make_file(arg, overwrite)
        sys.exit()

    return config_file, overwrite


if __name__ == "__main__":

    print("""
    teb -- calculate Teff for stars in eclipsing binaries
    
    Written by N. J. Miller and P. F. L. Maxted (2025)   
    Please cite: Miller, Maxted & Smalley (2020) and Maxted et al (2025)
    
    Most recent version of teb is stored at https://github.com/nmiller95/teb

    Contact nikkimillerastro@gmail.com with questions or suggestions
    
    """)

    print(f' This version: {version}')

    # Load file names and options from command line inputs
    config_file, overwrite = inputs(sys.argv[1:])
    ts = datetime.isoformat(datetime.now())[:19].replace('T',' ')
    print(f' Run started',ts)
    if not config_file.endswith('.yaml'):
        config_file = f'{config_file}.yaml'

    with open(os.path.join('config', config_file), 'r') as stream:
        config_dict = yaml.safe_load(stream)
    star_name = config_dict['star_name']
    print(f' Loaded configuration for target {star_name} from {config_file}\n')
    run_id = config_dict['run_id']
    print(f' Run identifier run_id is {run_id}')

    star_name_ = star_name.replace(' ','_')
    data_file = f'{star_name_}.yaml'
    with open(os.path.join('config', data_file), 'r') as stream:
        star_data = yaml.safe_load(stream)
    print(f' Loaded parallax and photometry data from {star_name_}.yaml')
    print('\n', flush=True)
    
    teff1,teff2 = star_data['teff1'], star_data['teff2']
    logg1, logg2 = star_data['logg1'], star_data['logg2']
    m_h, aFe = star_data['m_h'], star_data['aFe']
    binning = config_dict['binning']

    chain_file = f"output/{star_name}_{run_id}.fits"
    if not overwrite and os.path.exists(chain_file):
        raise FileExistsError('Use -o option to over-write output files.')

    # Create Flux2mag object from name and photometry data
    flux2mag = Flux2mag(star_name, star_data=star_data)

    # Flux ratio priors
    if 'flux_ratio_priors' in config_dict:
        frp_band_list  = config_dict['flux_ratio_priors']
        ctable1 = config_dict['color_table1']
        ctable2 = config_dict['color_table2']
        flux_ratio_priors = Flux_ratio_priors(frp_band_list, teff1, teff2,
                                              ctable1, ctable2)
        ctable1 = flux_ratio_priors.color_table1
        ctable2 = flux_ratio_priors.color_table2
        # Check if any flux ratio priors are not defined for nominal Teffs
        # Also raise a warning if Teff1 or Teff2 near limit of color tables.
        frp = flux_ratio_priors(1, teff1, teff2)
        for b in config_dict['flux_ratio_priors']:
            if not np.isfinite(frp[b].n):
                m = f'Flux ratio prior undefined for band {b}'
                raise ValueError(m)
        frp = flux_ratio_priors(1, teff1-100, teff2)
        for b in config_dict['flux_ratio_priors']:
            if not np.isfinite(frp[b].n):
                m = f'Teff1 near lower limit for column {b} of {ctable1}'
                warnings.warn(m)
        frp = flux_ratio_priors(1, teff1+100, teff2)
        for b in config_dict['flux_ratio_priors']:
            if not np.isfinite(frp[b].n):
                m = f'Teff1 near upper limit for column {b} of {ctable1}'
                warnings.warn(m)
        frp = flux_ratio_priors(1, teff1, teff2-100)
        for b in config_dict['flux_ratio_priors']:
            if not np.isfinite(frp[b].n):
                m = f'Teff2 near lower limit for column {b} of {ctable2}'
                warnings.warn(m)
        frp = flux_ratio_priors(1, teff1, teff2+100)
        for b in config_dict['flux_ratio_priors']:
            if not np.isfinite(frp[b].n):
                m = f'Teff2 near upper limit for column {b} of {ctable1}'
                warnings.warn(m)
    else:
        flux_ratio_priors = Flux_ratio_priors([],6000,6000)

    # Load extinction model into config_dict
    ext_mod = config_dict['extinction_model']
    config_dict['reddening_law'] = ReddeningLaw.from_extinction_model(ext_mod)

    # Loading models (interpolating if required)
    if round(teff1) > 9999 or round(teff2) > 9999:
        raise SystemExit("teb only supports Teff < 10000 K")
    elif round(teff1) < 1000 or round(teff2) < 1000:
        raise SystemExit("teb only supports Teff >= 1000 K")

    model_library = config_dict['model_sed']
    if model_library not in ['coelho-sed', 'bt-settl-cifist', 'bt-settl']:
        raise ValueError(f"Invalid model SED library: {model_library}")
    if model_library == 'bt-settl-cifist':
        print(' Setting [Fe/H]=0 and [a/Fe]=0 for bt-settl-cifist models.')
        m_h, aFe = 0, 0

    print("\n------------------------------------\n"
          "Loading and interpolating model SEDs"
          "\n------------------------------------")
    print("\nPrimary component\n-----------------")
    spec1 = ModelSpectrum.from_parameters(teff1, logg1, m_h, aFe,
                                          binning=binning, reload=False,
                                          source=model_library)
    print("\nSecondary component\n-------------------")
    spec2 = ModelSpectrum.from_parameters(teff2, logg2, m_h, aFe,
                                          binning=binning, reload=False,
                                          source=model_library)
    print('\n')
    # spec1 and spec2 go into config_dict because these are reference models.
    # The distorted versions that are the best fit to the star data go in
    # star_data
    config_dict['spec1'],config_dict['spec2'] = spec1, spec2

    ############################################################
    # Getting the lnlike set up and print initial result
    param_dict = initial_parameters(config_dict, star_data)

    for p in  param_dict:
        print(f'{p} = {param_dict[p]}')
    print(' Exponential priors on external noise hyper-parameters')
    print(f'  Width of prior on sigma_m = {config_dict["sigma_m_prior"]}')
    print(f'  Width of prior on sigma_r = {config_dict["sigma_r_prior"]}')
    if ('sigma_c_prior' in config_dict) and ('colors' in star_data):
        print(f'  Width of prior on sigma_c = {config_dict["sigma_c_prior"]}')
    print('', flush=True)

    params = list(param_dict.values())
    args = (param_dict, config_dict, flux2mag, flux_ratio_priors, star_data)
    lnlike = lnprob(params, *args,  verbose=True)[0]
    print(' Initial log-likelihood = {:0.2f}'.format(lnlike))
    print('',flush=True)

    ############################################################
    # Nelder-Mead optimisation
    nll = lambda *args: -lnprob(*args)[0]
    print("Finding initial solution with Nelder-Mead optimisation...")
    soln = minimize(nll, params, args=args, method='Nelder-Mead')

    # Print solutions
    for pn, pv in zip(param_dict, soln.x):
        if pv  > 1000:
            print(f'{pn} = {pv:6.1f}')
        else:
            print(f'{pn} = {pv:0.6f}')
    print('',flush=True)

    # Re-initialise log likelihood function with optimised solution
    lnlike = lnprob(soln.x, *args, verbose=True)[0]
    # Print solutions
    print('Optimised log-likelihood = {:0.2f}'.format(lnlike))
    print('',flush=True)

    ############################################################
    # Run MCMC simulations
    sampler = run_mcmc_simulations(soln, args)

    # Retrieve output from sampler and print key attributes
    af = sampler.acceptance_fraction
    print(f'\n Median acceptance fraction = {np.median(af)}')
    n_thin = config_dict['mcmc_n_thin']
    flat_samples = sampler.get_chain(thin=n_thin, flat=True)
    flat_lnprob = sampler.get_log_prob(thin=n_thin, flat=True)
    best_index = np.argmax(flat_lnprob)
    best_lnlike = np.max(flat_lnprob)
    print(f'\n Best log(likelihood) {best_lnlike:0.2f}')
    best_pars = flat_samples[best_index,:]

    ### Systematic errors on Teff, logL and Fbol
    # See Bohlin et al. 2014PASP..126..711B
    t = Table.read(os.path.join('Tables','WDcovar_002.fits'))
    w = t['WAVE'][0]
    covar = t['COVAR'][0]
    flux_data = lnprob(best_pars, *args, return_flux=True)
    wave, flux, flux_1, flux_2, flux0_1, flux0_2 = flux_data
    # Total fractional error in CALSPEC flux scale including 0.5% error
    # in the zero point from flux of Vega at 5556A
    sys_err = np.sqrt(0.005**2+np.diag(covar))
    # Arbitary error assumed for UV < 1000A = 5%, for IR > 30um = 1%
    teff1 = best_pars[0]
    syserr_1 = flux0_1*np.interp(wave, w, sys_err, left=0.005, right=0.001)
    Fbol_1 = simpson(flux0_1, x=wave)
    syserr_Fbol_1 = simpson(flux0_1+syserr_1, x=wave) - Fbol_1
    syserr_Teff_1 = 0.25*teff1*syserr_Fbol_1/Fbol_1
    syserr_logL_1 = syserr_Fbol_1/Fbol_1/np.log(10)
    teff2 = best_pars[1]
    syserr_2 = flux0_2*np.interp(wave, w, sys_err, left=0.005, right=0.001)
    Fbol_2 = simpson(flux0_2, x=wave)
    syserr_Fbol_2 = simpson(flux0_2+syserr_2, x=wave) - Fbol_2
    syserr_Teff_2 = 0.25*teff2*syserr_Fbol_2/Fbol_2
    syserr_logL_2 = syserr_Fbol_2/Fbol_2/np.log(10)
                         
    ###  print_mcmc_solution
    for i, pn in enumerate(param_dict):
        val = flat_samples[:, i].mean()
        err = flat_samples[:, i].std()
        ndp = 1 - min(0, np.floor((np.log10(err))))
        fmt = '{{:0.{:0.0f}f}}'.format(ndp)
        v_str = fmt.format(val)
        e_str = fmt.format(err)
        if pn == 'teff1':
            v_str = f'{val:.0f}'
            e_str = f'{err:.0f} (rnd) +/- {syserr_Teff_1:.0f} (sys)'
        if pn == 'teff2':
            v_str = f'{val:.0f}'
            e_str = f'{err:.0f} (rnd) +/- {syserr_Teff_2:.0f} (sys)'
        print(f' {pn} = {v_str} +/- {e_str}')
    lnlike_best = lnprob(best_pars, *args,  verbose=True)[0]

    # AIC and BIC calculation
    # Counts the number of photometry data used in order to calculate the AIC
    # and BIC
    n_photometry_data = len(flux2mag.obs_mag)
    n_photometry_data += len(flux2mag.obs_rat)
    n_photometry_data += len(flux2mag.obs_col)
    # Counts the number of other parameters used in the fit
    if len(flux2mag.obs_col) > 0:
        n_par = 8
    else:
        n_par = 7
    n_par += 2*config_dict['n_coeffs'] 
    aic = 2*n_par - 2*lnlike_best
    bic = n_par*np.log(n_photometry_data) - 2*lnlike_best
    print(f' n_obs: {n_photometry_data}')
    print(f' n_par: {n_par}')
    print(f' AIC: {aic:0.3f}')
    print(f' BIC: {bic:0.3f}')
    print('',flush=True)

    # Prepare output directory to save output data and chains
    os.makedirs('output', exist_ok=True)

    # Construct output FITS file
    hdul = fits.HDUList(fits.PrimaryHDU())

    # Output results to a FITS file
    paramtable = Table()
    # emcee chain
    for i,k in enumerate(param_dict):
        paramtable[k] = flat_samples[:, i]
    n_walkers = config_dict['mcmc_n_walkers']
    n_sample = config_dict['mcmc_n_sample']
    indices_ = np.mgrid[0:n_walkers,0:n_sample//n_thin]
    step_ = indices_[1].flatten()
    walker_ = indices_[0].flatten()
    paramtable['step'] = step_
    paramtable['walker'] = walker_
    paramtable['log_prob'] = flat_lnprob
    blobs = sampler.get_blobs(thin=n_thin, flat=True)
    for p in blobs.dtype.names:
        paramtable[p] = blobs[p]
        val = np.mean(blobs[p])
        err = np.std(blobs[p])
        if 'Fbol' in p:
            logscale = np.floor(-np.log10(val))
            val *= 10**logscale
            err *= 10**logscale
            units = f'e-{logscale:0.0f} erg cm−2 s−1'
        else:
            units = 'solar units'
        ndp = 1 - min(0, np.floor((np.log10(err))))
        fmt = '{{:0.{:0.0f}f}}'.format(ndp)
        v_str = fmt.format(val)
        e_str = fmt.format(err)
        if p == 'Fbol_1':
            s_str = fmt.format(syserr_Fbol_1*10**logscale)
            e_str += f' (rnd) +/- {s_str} (sys)'
        if p == 'logL_1':
            s_str = fmt.format(syserr_logL_1)
            e_str += f' (rnd) +/- {s_str} (sys)'
        if p == 'Fbol_2':
            s_str = fmt.format(syserr_Fbol_2*10**logscale)
            e_str += f' (rnd) +/- {s_str} (sys)'
        if p == 'logL_2':
            s_str = fmt.format(syserr_logL_2)
            e_str += f' (rnd) +/- {s_str} (sys)'
        print(f' {p} = {v_str} +/- {e_str} {units}')
    print('', flush=True)
    
    param_hdu = fits.table_to_hdu(paramtable)
    param_hdu.header['RUN_ID'] = run_id
    param_hdu.header['STARNAME'] = config_dict['star_name']
    param_hdu.header['VERSION'] = version
    param_hdu.header['MODELSED'] = config_dict['model_sed']
    param_hdu.header.comments['MODELSED'] = 'model_sed'
    param_hdu.header['BINNING'] = config_dict['binning']
    param_hdu.header['N_COEFFS'] = config_dict['n_coeffs']
    param_hdu.header['EXTMODEL'] = config_dict['extinction_model']
    param_hdu.header.comments['EXTMODEL'] = 'extinction_model'
    param_hdu.header['SM_PRIOR'] = config_dict['sigma_m_prior']
    param_hdu.header.comments['SM_PRIOR'] = 'sigma_m_prior'
    param_hdu.header['SR_PRIOR'] = config_dict['sigma_r_prior']
    param_hdu.header.comments['SR_PRIOR'] = 'sigma_r_prior'
    if 'sigma_c_prior' in config_dict:
        param_hdu.header['SC_PRIOR'] = config_dict['sigma_c_prior']
        param_hdu.header.comments['SC_PRIOR'] = 'sigma_c_prior'
    param_hdu.header['CTABLE1'] = flux_ratio_priors.color_table1
    param_hdu.header.comments['CTABLE1'] = 'color_table1'
    param_hdu.header['CTABLE2'] = flux_ratio_priors.color_table2
    param_hdu.header.comments['CTABLE2'] = 'color_table2'
    param_hdu.header['TEFF1'] = teff1
    param_hdu.header.comments['TEFF1'] = 'Reference SED Teff, star 1'
    param_hdu.header['TEFF2'] = teff2
    param_hdu.header.comments['TEFF2'] = 'Reference SED Teff, star 2'
    param_hdu.header['LOGG1'] = logg1
    param_hdu.header.comments['LOGG1'] = 'Reference SED log g, star 1'
    param_hdu.header['LOGG2'] = logg2
    param_hdu.header.comments['LOGG2'] = 'Reference SED log g, star 2'
    param_hdu.header['M_H'] = m_h
    param_hdu.header.comments['M_H'] = 'Reference SED [M/H]'
    param_hdu.header['AFE'] = aFe
    param_hdu.header.comments['AFE'] = 'Reference SED [alpha/Fe]'
    param_hdu.header['THETA1'] = star_data['theta1'].n
    param_hdu.header['E_THETA1'] = star_data['theta1'].s
    param_hdu.header.comments['THETA1'] = 'Angular diameter, star 1'
    param_hdu.header['THETA2'] = star_data['theta2'].n
    param_hdu.header['E_THETA2'] = star_data['theta2'].s
    param_hdu.header.comments['THETA2'] = 'Angular diameter, star 2'
    param_hdu.header['EBV'] = star_data['ebv'][0]
    param_hdu.header['E_EBV'] = star_data['ebv'][1]
    param_hdu.header.comments['EBV'] = 'Prior on E(B-V)'
    param_hdu.header['N_OBS'] = n_photometry_data
    param_hdu.header['N_PAR'] = n_par
    param_hdu.header['AIC'] = aic
    param_hdu.header['BIC'] = bic
    param_hdu.header['TEFFSYS1'] = syserr_Teff_1
    param_hdu.header['TEFFSYS2'] = syserr_Teff_2
    param_hdu.header['FBOLSYS1'] = syserr_Fbol_1
    param_hdu.header['FBOLSYS2'] = syserr_Fbol_2
    param_hdu.header['LOGLSYS1'] = syserr_logL_1
    param_hdu.header['LOGLSYS2'] = syserr_logL_2
    param_hdu.header.comments['TEFFSYS1'] = 'Systematic error in T_eff,1'
    param_hdu.header.comments['TEFFSYS2'] = 'Systematic error in T_eff,2'
    param_hdu.header.comments['FBOLSYS1'] = 'Systematic error in F_bol,1'
    param_hdu.header.comments['FBOLSYS2'] = 'Systematic error in F_bol,1'
    param_hdu.header.comments['LOGLSYS1'] = 'Systematic error in log L_1'
    param_hdu.header.comments['LOGLSYS2'] = 'Systematic error in log L_2'

    param_hdu.name = 'EMCEE_CHAIN'
    hdul.append(param_hdu)
    
    # Magnitudes
    magtable = Table([[tag for tag in flux2mag.obs_mag]], names=['Tag'])
    obs_mag = [flux2mag.obs_mag[tag] for tag in flux2mag.obs_mag]
    filters = [o.tag for o in obs_mag]
    magtable['Filter'] = filters
    magtable['Pivot'] = [flux2mag.filters[f]['pivot'] for f in filters]
    magtable['obs'] = [o.n for o in obs_mag]
    magtable['e_obs'] = [o.s for o in obs_mag]
    # Call lnprob with best_pars so flux2mag.syn_mag corresponds to best fit
    _ = lnprob(best_pars, *args)
    syn_mag = [flux2mag.syn_mag[tag] for tag in flux2mag.syn_mag]
    magtable['syn'] = [s.n for s in syn_mag]
    magtable['e_syn'] = [s.s for s in syn_mag]
    syn_mag1 = [flux2mag.syn_mag1[tag] for tag in flux2mag.syn_mag1]
    magtable['mag1'] = [s.n for s in syn_mag1]
    magtable['e_mag1'] = [s.s for s in syn_mag1]
    syn_mag2 = [flux2mag.syn_mag2[tag] for tag in flux2mag.syn_mag2]
    magtable['mag2'] = [s.n for s in syn_mag2]
    magtable['e_mag2'] = [s.s for s in syn_mag2]
    mag_hdu = fits.table_to_hdu(magtable)
    mag_hdu.name = 'MAGNITUDES'
    hdul.append(mag_hdu)

    # Flux ratios
    if len(flux2mag.obs_rat) > 0:
        rattable = Table([[tag for tag in flux2mag.obs_rat]], names=['Tag'])
        obs_rat = [flux2mag.obs_rat[tag] for tag in flux2mag.obs_rat]
        filters = [o.tag for o in obs_rat]
        rattable['Filter'] = filters
        rattable['Pivot'] = [flux2mag.filters[f]['pivot'] for f in filters]
        rattable['obs'] = [o.n for o in obs_rat]
        rattable['e_obs'] = [o.s for o in obs_rat]
        rattable['syn'] = [flux2mag.syn_rat[tag] for tag in flux2mag.syn_rat] 
        rat_hdu = fits.table_to_hdu(rattable)
        rat_hdu.name = 'FLUX_RATIOS'
        hdul.append(rat_hdu)

    # Colors
    if len(flux2mag.obs_col) > 0:
        coltable = Table([[tag for tag in flux2mag.obs_col]], names=['Tag'])
        obs_col = [flux2mag.obs_col[tag] for tag in flux2mag.obs_col]
        colors = [o.tag for o in obs_col]
        coltable['Color'] = colors
        coltable['obs'] = [o.n for o in obs_col]
        coltable['e_obs'] = [o.s for o in obs_col]
        syn_col = [flux2mag.syn_col[tag] for tag in flux2mag.syn_col]
        coltable['syn'] = [s.n for s in syn_col]
        coltable['e_syn'] = [s.s for s in syn_col]
        col_hdu = fits.table_to_hdu(coltable)
        col_hdu.name = 'FLUX_COLORS'
        hdul.append(col_hdu)

    # Spectral energy distributions
    ns = flat_samples.shape[0]
    nw = len(wave)
    flux_ = np.zeros([nw, 100])
    frat_ = np.zeros([nw, 100])
    # Use 100 random samples from the emcee chain to compute std dev
    for i,j in enumerate(np.random.choice(ns, 100)):
        f_ = lnprob(flat_samples[j,:], *args, return_flux=True)
        flux_[:,i] = f_[1]
        frat_[:,i] = f_[3]/f_[2]
    flux_mean = flux_.mean(axis=1)
    flux_err  = flux_.std(axis=1)
    frat_mean = frat_.mean(axis=1)
    frat_err  = frat_.std(axis=1) 
    del flux_, frat_
    flux_data = lnprob(best_pars, *args, return_flux=True)
    names = ['wave', 'flux_best', 'flux_1_best', 'flux_2_best',
             'flux0_1_best', 'flux0_2_best']
    fluxtable = Table(flux_data, names=names)
    fluxtable['flux_mean'] = flux_mean
    fluxtable['flux_err'] = flux_err
    fluxtable['flux_ratio_mean'] = frat_mean
    fluxtable['flux_ratio_err'] = frat_err
    for c in fluxtable.colnames:
        if c == 'wave':
            fluxtable[c].unit = u.angstrom
        else:
            fluxtable[c].unit = u.erg/u.cm**2/u.s/u.angstrom

    flux_hdu = fits.table_to_hdu(fluxtable)
    flux_hdu.name = 'FLUX'
    hdul.append(flux_hdu)

    # Flux ratio priors
    if len(flux_ratio_priors.bands) > 0:
        frptable = Table([[b for b in flux_ratio_priors.bands]],
                         names=['Band'])
        # Response function of RP band over wavelength range
        RRP = flux_ratio_priors.T['RP'](wave) 
        if flux_ratio_priors.photon['RP']:
            RRP *= wave
        # Synthetic flux ratio in RP band
        lRP = simpson(RRP*flux_2, x=wave) / simpson(RRP*flux_1, x=wave) 
        priors = flux_ratio_priors(lRP, teff1, teff2)
        frptable['PRIOR'] = [priors[p].n for p in priors]
        frptable['E_PRIOR'] = [priors[p].s for p in priors]
        calc = []
        for b in flux_ratio_priors.bands:
            # Response function 
            RX = flux_ratio_priors.T[b](wave) 
            if flux_ratio_priors.photon[b]:
                RX *= wave
            # Synthetic flux ratio in any X band
            lX = (simpson(RX*flux_2*wave, x=wave) / 
                  simpson(RX*flux_1*wave, x=wave) )
            calc.append(lX)
        frptable['CALC'] = calc
        frp_hdu = fits.table_to_hdu(frptable)
        frp_hdu.name = 'FLUX RATIO PRIORS'
        hdul.append(frp_hdu)

    hdul.writeto(chain_file, overwrite=overwrite)
    print('Output written to chain_file')
    ts = datetime.isoformat(datetime.now())[:19].replace('T',' ')
    print(f' Run finished',ts)
