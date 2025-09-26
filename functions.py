import os
import numpy as np
from uncertainties import ufloat, correlation_matrix
from astropy.table import Table
from synphot import units
from scipy.special import legendre
from scipy.integrate import simpson
import emcee
from multiprocessing import Pool
from scipy.interpolate import interp1d
from astroquery.vizier import Vizier
from zero_point import zpt
from collections import OrderedDict

def get_parallax(star_name, zp_correction=True):
    """
    Gets Gaia DR3 parallax and applied zeropoint correction for named star

    Parameters
    ----------
    star_name: str
        Source identifier recognised by Vizier.

    zp_correction: bool
        Whether or not to apply a zero-point correction

    Returns
    -------
    Parallax + error with zeropoint correction (if applied) as ufloat 

    """
    # Read data from Gaia DR3
    vizier_r = Vizier(columns=["**", "+_r"])
    v = vizier_r.query_object(star_name, catalog='I/355/gaiadr3')
    plx = ufloat(v[0][0]['Plx'], v[0][0]['e_Plx'])
    if not zp_correction:
        return plx

    phot_g_mean_mag = v[0][0]['Gmag']
    ecl_lat = v[0][0]['ELAT']
    nu_eff_used_in_astrometry = v[0][0]['nueff']
    astrometric_params_solved = v[0][0]['Solved']
    # for 5-parameter solutions, pseudocolour is arbitrary.
    if astrometric_params_solved == 31:
        pseudocolour = 0
    else:
        pseudocolour = v[0][0]['pscol']

    # Check whether target meets validity range described in docstring
    if phot_g_mean_mag < 6 or phot_g_mean_mag > 21:
        print(f"G magnitude ({phot_g_mean_mag}) outside of supported range"
              "(6-21).")
        print("Setting parallax zero-point to mean offset based on quasars"
              "(-0.021mas)")
        return plx - ufloat(-0.021, 0.013)
    elif nu_eff_used_in_astrometry < 1.1 or nu_eff_used_in_astrometry > 1.9:
        print(f"nu_eff_used_in_astronometry of {nu_eff_used_in_astrometry}"
              "outside of supported range (1.1-1.9).")
        print("Setting parallax zero-point to mean offset based on quasars"
              "(-0.021mas)")
        return plx - ufloat(-0.021, 0.013)

    try:
        # Calculate zeropoint for target
        zpt.load_tables()
        zp = zpt.get_zpt(
            phot_g_mean_mag,
            nu_eff_used_in_astrometry,
            pseudocolour,
            ecl_lat,
            astrometric_params_solved)

        if phot_g_mean_mag <= 11:
            # Flynn+2022 correction based on open and globular clusters
            # Data from Table 1
            bprp_arr = np.array([0.02, 0.19, 0.40, 0.65, 1.56, 2.72])
            offset = np.array([-10.8, -8.9, -4.4, 2.7, 9.8, 7.3])
            offset_err = np.array([3.3, 2.7, 2.7, 5.2, 1.9, 8.4])

            # Linear interpolation
            f = interp1d(bprp_arr, offset)
            f_err = interp1d(bprp_arr, offset_err)
            x = np.linspace(min(bprp_arr), max(bprp_arr), num=100,
                            endpoint=True)

            # Apply color-based correction - a bit hacky
            bprp_target = v[0][0]['BP-RP']
            correction = float(f(bprp_target))  # in uas
            corr_err = float(f_err(bprp_target))  # error interpolated

            combined_zp = zp + correction / 1000
            combined_err = np.sqrt(0.013 ** 2 + (corr_err / 1000) ** 2)
            # Return value of Lindegren et al 2021 adjusted by Flynn et al 2022
            print('Correction to Gaia parallax from Flynn+2022 applied '
                  f'{combined_zp:0.3f}')
            return plx - ufloat(combined_zp, combined_err)

        else:
            # Return value of Lindegren et al 2021
            print(f"Correction to Gaia parallax from Lindegren+2021 applied"
                  f"{zp:0.3f}")
            return plx - ufloat(zp, 0.013)

    except ValueError:
        print('Problem with zero-point offset calculation: check value of'
              'astrometric_params_solved')
        print('Setting parallax zero-point to mean offset based on quasars'
              '(-0.021mas)')
        return plx - ufloat(-0.021, 0.013)

def initial_parameters(config_dict, star_data):
    """
    Loads and generates parameters for log likelihood calculations

    Parameters
    ----------
    config_dict: dict
        Dictionary containing parameters, loaded from config.yaml
    star_data: dict
        Dictionary containing stellar data

    Returns
    -------
     Model parameters as an OrderedDict
    """
    param_dict = OrderedDict()
    param_dict['teff1'] = star_data['teff1']
    param_dict['teff2'] = star_data['teff2']
    # Copy starting values to new variables
    param_dict['theta_1'] = np.round(star_data['theta1'].n, 6)
    param_dict['theta_2'] = np.round(star_data['theta2'].n, 6)
    param_dict['E(B-V)']  = star_data['ebv'][0]
    param_dict['sigma_m'] = min([config_dict['sigma_m_prior'], 0.001])
    param_dict['sigma_r'] = min([config_dict['sigma_r_prior'], 0.001])

    if 'colors' in star_data:
        param_dict['sigma_c'] = min([config_dict['sigma_c_prior'], 0.001])

    nc = config_dict['n_coeffs']
    for j in range(nc):
        param_dict[f'c_1,{j+1}'] = 0
        param_dict[f'c_2,{j+1}'] = 0
    return param_dict

#-----------------------

def lnprob(param_list, param_dict, config_dict, flux2mag, flux_ratio_priors,
           star_data, wmin=1000, wmax=300000, return_flux=False, verbose=False):
    """
    Log probability function for the fundamental effective temperature of
    eclipsing binary stars method.

    Parameters
    ----------
    param_list: list
        Model parameters and hyper-parameters as dict.
    param_dict: OrderedDict
        OrderedDict of parameters in the same order as param_list
    config_dict: dict
        Dictionary containing configuration parameters, from config.yaml file
    star_data: dict
        Dictionary containing star data
    flux2mag: `flux2mag.Flux2Mag`
        Magnitude data and log-likelihood calculator (Flux2Mag object)
    flux_ratio_priors: object
        Instance of Flux_ratio_priors class
    wmin: int, optional
        Lower wavelength cut for model spectrum, in Angstroms
    wmax: int, optional
        Upper wavelength cut for model spectrum, in Angstroms
    return_flux: bool, optional
        Whether to return the wavelength, flux and distortion arrays
    verbose: bool, optional
        Whether to print out all the parameters

    Returns
    -------
    Either  [lnprob, Fbol1, Fbol2, logL1, logL2] or
    or wavelength, flux and extinction corrected fluxes (return_flux=True)
    """
    sigma_sb = 5.670367E-5  # erg.cm-2.s-1.K-4

    # Update parameter values in param_dict from param_list
    for p,v in zip(param_dict, param_list):
        param_dict[p] = v

    for p in ['theta_1', 'theta_2', 'E(B-V)', 'teff1', 'teff2', 'sigma_m',
              'sigma_r']:
        if param_dict[p] < 0:
            return -np.inf, *[None]*4

    if 'sigma_c' in param_dict:
        sigma_c = param_dict['sigma_c']
        if sigma_c <0:
            return -np.inf, *[None]*4
    else:
        sigma_c = 0

    # Get wave and flux information from spec1 and spec2 objects
    spec1 = config_dict['spec1']
    spec2 = config_dict['spec2']
    wave = spec1.waveset
    wave = wave[(wmin < wave.value) & (wave.value < wmax)]
    flux1 = spec1(wave, flux_unit=units.FLAM)
    flux2 = spec2(wave, flux_unit=units.FLAM)
    wave = wave.value  # Converts to numpy array
    flux1 = flux1.value
    flux2 = flux2.value

    # Converts wavelength space to x coordinates for Legendre polynomials
    x = 2*np.log(wave/np.min(wave)) / np.log(np.max(wave)/np.min(wave)) - 1
    # Make empty distortion polynomial object
    nc = config_dict['n_coeffs']
    distort1 = np.zeros_like(flux1)
    distort2 = np.zeros_like(flux2)
    coeffs1 = [param_dict[f'c_1,{j+1}'] for j in range(nc)]
    for n,c in enumerate(coeffs1):
        if abs(c) > 1: # Check distortion coefficients are between -1 and +1
            return -np.inf, *[None]*4
        distort1 +=  c * legendre(n + 1)(x)
    coeffs2 = [param_dict[f'c_2,{j+1}'] for j in range(nc)]
    for n,c in enumerate(coeffs2):
        if abs(c) > 1: # Check distortion coefficients are between -1 and +1
            return -np.inf, *[None]*4
        distort2 += c * legendre(n + 1)(x)
    # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
    i_5556 = np.argmin(abs(wave - 5556))
    distort1 -= distort1[i_5556]
    distort2 -= distort2[i_5556]
    if min(distort1) < -1:
        return -np.inf, *[None]*4
    flux1 *= (1 + distort1)
    flux1 /= simpson(flux1,x=wave)
    if min(distort2) < -1:
        return -np.inf, *[None]*4
    flux2 *= (1 + distort2)
    flux2 /= simpson(flux2,x=wave)

    # Convert these bolometric fluxes to fluxes observed at the top of Earth's
    # atmosphere
    redlaw = config_dict['reddening_law']
    extinction = redlaw.extinction_curve(param_dict['E(B-V)'])(wave).value
    theta1 = param_dict['theta_1']
    teff1 = param_dict['teff1']
    flux0_1 = 0.25 * sigma_sb * (theta1 / 206264806) ** 2 * teff1 ** 4 * flux1
    theta2 = param_dict['theta_2']
    teff2 = param_dict['teff2']
    flux0_2 = 0.25 * sigma_sb * (theta2 / 206264806) ** 2 * teff2 ** 4 * flux2
    flux = (flux0_1 + flux0_2) * extinction # Total "observed" flux
    flux_1 = flux0_1*extinction
    flux_2 = flux0_2*extinction
    if return_flux:
        return wave, flux, flux_1, flux_2, flux0_1, flux0_2

    flux_ratio = flux_2/flux_1
    sigma_m = param_dict['sigma_m']
    sigma_r = param_dict['sigma_r']
    r = flux2mag(wave, flux, flux_ratio, sigma_m, sigma_r, sigma_c)
    chisq_m, lnlike_m, chisq_c, lnlike_c, lnlike_r, chisq_r = r


    if verbose:
        print('')
        print(' Magnitudes')
        print(' Tag     Pivot Observed         Calculated                  O-C')
        for tag in flux2mag.obs_mag:
            o = flux2mag.obs_mag[tag]
            c = flux2mag.syn_mag[tag]
            fn = o.tag  # filter name
            w = flux2mag.filters[fn]['pivot']
            print(f" {tag:6s} {w:6.0f} {o:6.3f} {c:8.4f} {o-c:+9.4f}")
        print(f' N = {len(flux2mag.obs_mag)}')
        print(f' sigma_m = {sigma_m:0.4f}')
        print(f' chi-squared = {chisq_m:0.2f}')
        print('',flush=True)

        if len(flux2mag.obs_col) > 0:
            print(' Colors')
            print(' Tag     Color  Observed        Calculated       O-C')
        for tag in flux2mag.obs_col:
            o = flux2mag.obs_col[tag]
            c = flux2mag.syn_col[tag]
            print(f" {tag:8s} {o.tag:5} {o:6.3f} {c:6.3f} {o-c:+6.3f}")
        if len(flux2mag.obs_col) > 0:
            print(f' N = {len(flux2mag.obs_col)}')
            print(f' sigma_c = {sigma_c:0.4f}')
            print(f' chi-squared = {chisq_c:0.2f}')
        print('',flush=True)

        if len(flux2mag.obs_rat) > 0:
            print(' Flux ratios')
            print(' Tag    Pivot   Observed           Calculated     O-C')
        for tag in flux2mag.obs_rat:
            o = flux2mag.obs_rat[tag]
            c = flux2mag.syn_rat[tag]
            fn = o.tag  # Filter name stored as a tag to observed mag
            w = flux2mag.filters[fn]['pivot']
            if o.s > 0.2:
                print(f" {tag:6s} {w:8.1f} {o:7.1f} {c:7.1f}   {o-c:+6.2f}")
            elif o.s > 0.02:
                print(f" {tag:6s} {w:8.1f} {o:7.2f} {c:7.2f}   {o-c:+6.2f}")
            elif o.s > 0.002:
                print(f" {tag:6s} {w:8.1f} {o:7.3f} {c:7.3f}   {o-c:+6.3f}")
            elif o.s > 0.0002:
                print(f" {tag:6s} {w:8.1f} {o:8.4f} {c:8.4f}   {o-c:+7.4f}")
            else:
                print(f" {tag:6s} {w:8.1f} {o:7.4f} {c:7.4f}   {o-c:+6.4f}")
        if len(flux2mag.obs_rat) > 0:
            print(f' N = {len(flux2mag.obs_rat)}')
            print(f' sigma_r = {sigma_r:0.4f}')
            print(f' chi-squared = {chisq_r:0.2f}')
        print('',flush=True)

    # Angular diameter log likelihood. See equation (1) from 
    # See http://mathworld.wolfram.com/BivariateNormalDistribution.html
    theta1_in = star_data['theta1']
    theta2_in = star_data['theta2']
    rho = correlation_matrix([theta1_in, theta2_in])[0][1]
    z = ((theta1 - theta1_in.n) ** 2 / theta1_in.s ** 2 +
         (theta2 - theta2_in.n) ** 2 / theta2_in.s ** 2 -
         2 * rho * (theta1 - theta1_in.n) * (theta2 - theta2_in.n) /
         theta1_in.s / theta2_in.s )
    lnlike_theta = -0.5 * z / (1 - rho ** 2)

    # Combine log likelihoods calculated so far
    lnlike = lnlike_m + lnlike_theta + lnlike_r + lnlike_c

    # Applying prior on interstellar reddening (if relevant)
    lnprior = 0
    if star_data['ebv']:
        ebv_prior = ufloat(star_data['ebv'][0], star_data['ebv'][1])
        lnprior += -0.5*ebv_prior.std_score(param_dict['E(B-V)'])**2
    # Exponential priors on external noise hyper-parameters
    sigma_m_prior = float(config_dict["sigma_m_prior"])
    lnprior += -sigma_m / sigma_m_prior - np.log(sigma_m_prior)
    sigma_r_prior = float(config_dict["sigma_r_prior"])
    lnprior += -sigma_r / sigma_r_prior - np.log(sigma_r_prior)
    if 'sigma_c' in param_dict:
        sigma_c_prior = float(config_dict["sigma_c_prior"])
        lnprior += -sigma_c / sigma_c_prior - np.log(sigma_c_prior)

    # Applying priors on UV/NIR flux ratios 
    if 'flux_ratio_priors' in config_dict:
        # Response function of RP band over wavelength range
        RRP = flux_ratio_priors.T['RP'](wave) 
        if flux_ratio_priors.photon['RP']:
            RRP *= wave
        # Synthetic flux ratio in RP band
        lRP = simpson(RRP*flux_2, x=wave) / simpson(RRP*flux_1, x=wave) 
        if verbose:
            print(' Flux ratio priors')
            print(' Band   Prior      Calculated    O-C')
        chisq_frp = 0
        priors = flux_ratio_priors(lRP, teff1, teff2)
        for b in flux_ratio_priors.bands:
            # Response function 
            RX = flux_ratio_priors.T[b](wave) 
            if flux_ratio_priors.photon[b]:
                RX *= wave
            # Synthetic flux ratio in any X band
            lX = (simpson(RX*flux_2*wave, x=wave) / 
                  simpson(RX*flux_1*wave, x=wave) )
            prior = priors[b]
            if verbose:
                print(f' {b:<4s} {prior:0.3f}  {lX:0.3f}  {prior-lX:+0.3f}')
            chisq_frp += prior.std_score(lX)**2
            # Apply the prior to overall log prior
            lnprior += -0.5*chisq_frp
        if verbose:
            print(f' Flux ratio priors: chi-squared = {chisq_frp:0.2f}')

    if np.isfinite(lnlike) and np.isfinite(lnprior):
        # Bolometric fluxes
        Fbol_1 = simpson(flux0_1, x=wave)  
        Fbol_2 = simpson(flux0_2, x=wave)  
        # We are randomly sampling theta_1 and theta_2, which accounts for
        # both the errors in the parallax and the stellar radii. So, sample
        # a random value of the parallax for calculation of radii from
        # theta_1, theta_2
        plx = np.random.normal(*star_data['parallax'])
        logL_1 = np.log10(Fbol_1/plx**2) + 10.494939
        logL_2 = np.log10(Fbol_2/plx**2) + 10.494939
        return lnlike + lnprior, Fbol_1, Fbol_2, logL_1, logL_2
    else:
        return -np.inf, *[None]*4


def run_mcmc_simulations(least_squares_solution, args):
    """
    Runs MCMC via the emcee module, using the least squares solution as a starting point

    Parameters
    ----------
    least_squares_solution: `scipy.optimize.OptimizeResult`
      Output of minimization
    args: list
      Parameters to pass through to lnprob
      args = (param_dict, config_dict, flux2mag, flux_ratio_priors, star_data)

    Returns
    -------
    `emcee.sampler` object
    """
    param_dict, config_dict, flux2mag, flux_ratio_priors, star_data = args
    nc = config_dict['n_coeffs']
    th1 = star_data['theta1']
    th2 = star_data['theta2']
    if ('colors' in star_data) and len(star_data['colors']) > 0:
        npositive = 8
        steps = np.array([5, 5,  # T_eff,1, T_eff,2
                          th1.s/10, th2.s/10, 0.0001,  # theta_1 ,theta_2, E(B-V)
                          0.0001, 0.0001, 0.0001,  # sigma_m, sigma_r, sigma_c
                          *[0.001] * nc, *[0.001] * nc])  # c_1,1 ..   c_2,1 ..
    else:
        npositive = 7
        steps = np.array([5, 5,  # T_eff,1, T_eff,2
                          th1.s/10, th2.s/10,  # theta_1 ,theta_2
                          0.0001, 0.0001, 0.0001,  # E(B-V), sigma_m, sigma_r
                          *[0.001] * nc, *[0.001] * nc])  # c_1,1 ..   c_2,1 ..

    n_burnin = config_dict['mcmc_n_burnin']
    n_walkers = config_dict['mcmc_n_walkers']
    n_sample = config_dict['mcmc_n_sample']
    ndim = len(least_squares_solution.x)
    pos = np.zeros([n_walkers, ndim])
    for i, x in enumerate(least_squares_solution.x):
        pos[:, i] = x + steps[i] * np.random.randn(n_walkers)
        if i < npositive:
            pos[:, i] = abs(pos[:, i])

    with Pool() as pool:
        print("Running emcee burn-in ...")
        dtype = [("Fbol_1", float), ("Fbol_2", float),
                 ("logL_1", float), ("logL_2", float)]
        sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, args=args,
                                        blobs_dtype=dtype, pool=pool)
        state = sampler.run_mcmc(pos, n_burnin, 
                                 progress=config_dict['mcmc_show_progress'])
        sampler.reset()
        print("Running emcee sampler ...")
        sampler.run_mcmc(pos, n_sample, 
                         progress=config_dict['mcmc_show_progress'])

    return sampler

