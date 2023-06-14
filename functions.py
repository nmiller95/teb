from uncertainties import ufloat, correlation_matrix
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from astropy.table import Table
import astropy.units as u
from synphot import units
from scipy.special import legendre
from scipy.integrate import simps
import yaml
import emcee
from multiprocessing import Pool
import flux_ratio_priors as frp
from flux_ratios import FluxRatio
import sys
from scipy.interpolate import interp1d


def list_to_ufloat(two_item_list):
    """Turns a two item list from yaml input into a ufloat"""
    return ufloat(two_item_list[0], two_item_list[1])


def load_photometry(photometry_file):
    """
    Reads photometry inputs from photometry.yaml and prepares them for the method

    Returns
    -------
    Flux ratios, extra data, colors data as dictionaries
    """
    # Load and initialise photometric data from photometry_data.yaml
    try:
        stream = open(f'config/{photometry_file}', 'r')
    except FileNotFoundError as err:
        print(err)
        sys.exit()
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
    except TypeError:
        print("No additional magnitudes provided in photometry_data.yaml")
        extra_data = None
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
    return flux_ratios, extra_data, colors_data


def angular_diameters(config_dict):
    """
    Converts input radii and Gaia parallax to angular diameters in mas

    Parameters
    ----------
    config_dict: dict
        Dictionary containing parameters, loaded from config.yaml

    Returns
    -------
    Angular diameters for primary and secondary stars as `uncertainties.ufloat` objects
    """
    # Parallax load and apply zeropoint
    gaia_zp = ufloat(-0.0373, 0.013)  # Custom ZP for TYC 1243-402-1
    # gaia_zp = ufloat(-0.0055, 0.014) Gaia EDR3 custom ZP for CPD-54 810 from Lindegren+Flynn correction TODO automate
    print(f'Applied Gaia parallax zero point correction {gaia_zp:0.3f}')
    plx = list_to_ufloat(config_dict['plx']) - gaia_zp

    # Angular diameter = 2*R/d = 2*R*parallax = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
    # R_Sun = 6.957e8 m. parsec = 3.085677581e16 m
    r1 = list_to_ufloat(config_dict['r1'])
    r2 = list_to_ufloat(config_dict['r2'])
    theta1 = 2 * plx * r1 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
    theta2 = 2 * plx * r2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
    return theta1, theta2


def initial_parameters(config_dict, theta1, theta2, ebv_prior):
    """
    Loads and generates parameters for log likelihood calculations

    Parameters
    ----------
    config_dict: dict
        Dictionary containing parameters, loaded from config.yaml
    theta1: `uncertainties.ufloat`
        Angular diameter of primary star in mas
    theta2: `uncertainties.ufloat`
        Angular diameter of secondary star in mas
    ebv_prior: `uncertainties.ufloat`
        Prior on interstellar reddening as ufloat

    Returns
    -------
    Parameters and parameter names as two lists
    """
    teff1 = config_dict['teff1']
    teff2 = config_dict['teff2']
    # Copy starting values to new variables
    theta1_ = theta1.n
    theta2_ = theta2.n
    ebv_ = ebv_prior.n
    sigma_ext = config_dict['sigma_ext']
    sigma_l = config_dict['sigma_l']
    nc = config_dict['n_coeffs']

    params = [teff1, teff2, theta1_, theta2_, ebv_, sigma_ext, sigma_l]
    parname = ['T_eff,1', 'T_eff,2', 'theta_1', 'theta_2', 'E(B-V)', 'sigma_ext', 'sigma_l']

    if config_dict['apply_colors']:
        sigma_c = config_dict['sigma_c']
        params = params + [sigma_c]
        parname = parname + ['sigma_c']

    if config_dict['distortion'] == 0:
        pass
    elif config_dict['distortion'] == 1:
        params = params + [0] * nc
        parname = parname + ["c_1,{}".format(j + 1) for j in range(nc)]
    elif config_dict['distortion'] == 2:
        params = params + [0] * 2 * nc
        parname = parname + ["c_1,{}".format(j + 1) for j in range(nc)]
        parname = parname + ["c_2,{}".format(j + 1) for j in range(nc)]
    return params, parname


def synthetic_optical_lratios(config_dict, spec1, spec2, theta1, theta2, redlaw, v_ratio, flux_ratios):
    """
    Estimates the flux ratio in the Johnson/Cousins UBVRI bands using the TESS-band flux ratio and model SEDs.

    WARNING: Don't use this for your proper scientific results! This is for estimating the impact of multi-band light
    curves on the TEB output (e.g. for observing proposals).

    """
    print("CAUTION: Calculating synthetic flux ratios using model SEDs. \n"
          "These are to be used only for information purposes and not for final results!")
    sigma_sb = 5.670367E-5  # erg.cm-2.s-1.K-4
    wmin, wmax = (1000, 300000)

    wave = spec1.waveset
    i = ((wmin * u.angstrom < wave) & (wave < wmax * u.angstrom)).nonzero()
    wave = wave[i].value
    flux1 = spec1(wave, flux_unit=units.FLAM).value
    flux2 = spec2(wave, flux_unit=units.FLAM).value
    flux1 = flux1 / simps(flux1, wave)
    flux2 = flux2 / simps(flux2, wave)

    extinction = redlaw.extinction_curve(config_dict['ebv'][0])(wave).value
    f_1 = 0.25 * sigma_sb * (theta1 / 206264806) ** 2 * config_dict['teff1'] ** 4 * flux1 * extinction
    f_2 = 0.25 * sigma_sb * (theta2 / 206264806) ** 2 * config_dict['teff2'] ** 4 * flux2 * extinction

    # RESPONSE FUNCTIONS
    R = dict()
    syn_lratio = dict()
    t = Table.read('Response/J_PASP_124_140_table1.dat.fits')  # Johnson - from Bessell, 2012 PASP, 124:140-157
    for b in ['U', 'B', 'V', 'R', 'I']:
        wtmp = t['lam.{}'.format(b)]
        rtmp = t[b]
        rtmp = rtmp[wtmp > 0]
        wtmp = wtmp[wtmp > 0]
        R[b] = interp1d(wtmp, rtmp, bounds_error=False, fill_value=0)

        # SYNTHETIC PHOTOMETRY
        f_nu1 = (simps(f_1 * R[b](wave) * wave, wave) /
                 simps(R[b](wave) * 2.998e10 / (wave * 1e-8), wave))
        f_nu2 = (simps(f_2 * R[b](wave) * wave, wave) /
                 simps(R[b](wave) * 2.998e10 / (wave * 1e-8), wave))
        syn_lratio[b] = f_nu2 / f_nu1
        print(f'Synthetic {b} flux ratio: {syn_lratio[b]}')

        fr = FluxRatio(f'Synth_{b}', b, float(syn_lratio[b].n), float(syn_lratio[b].s))
        tag, d = fr()
        flux_ratios[tag] = d

    return flux_ratios


def lnprob(params, flux2mag, lratios, theta1_in, theta2_in, spec1, spec2, ebv_prior, redlaw, nc, config_dict,
           frp_coeffs=None, wmin=1000, wmax=300000, return_flux=False, blobs=False,
           debug=False, verbose=False):
    """
    Log probability function for the fundamental effective temperature of eclipsing binary stars method.

    Parameters
    ----------
    params: list
        Model parameters and hyper-parameters as list. Should contain:
            teff1, teff2, theta1, theta2, ebv, sigma_ext, sigma_l(, sigma_c, [0]*(nc * 1 or 2))
    flux2mag: `flux2mag.Flux2Mag`
        Magnitude data and flux-to-mag log-likelihood calculator (Flux2Mag object)
    lratios: dict
        Flux ratios and response functions
    theta1_in: `uncertainties.ufloat`
        Angular diamater of primary star in mas (initial value)
    theta2_in: `uncertainties.ufloat`
        Angular diameter of secondary star in mas (initial value)
    spec1: `synphot.SourceSpectrum`
        Model spectrum of primary star
    spec2: `synphot.SourceSpectrum`
        Model spectrum of secondary star
    ebv_prior: `uncertainties.ufloat`
        Prior on E(B-V)
    redlaw: `synphot.ReddeningLaw`
        Reddening law
    nc: int
        Number of distortion coefficients for primary star (star 1)
    config_dict: dict
        Dictionary containing configuration parameters, from config.yaml file
    frp_coeffs: dict, optional
        Dictionary of flux ratio prior coefficients over suitable temperature range
    wmin: int, optional
        Lower wavelength cut for model spectrum, in Angstroms
    wmax: int, optional
        Upper wavelength cut for model spectrum, in Angstroms
    return_flux: bool, optional
        Whether to return the wavelength, flux and distortion arrays
    blobs: bool, optional
        For the MCMC
    debug: bool, optional
        Whether to print extra stuff
    verbose: bool, optional
        Whether to print out all the parameters

    Returns
    -------
    Either log likelihood and log prior (return_flux=False),
    or wavelength, flux and distortion arrays (return_flux=True)
    """
    sigma_sb = 5.670367E-5  # erg.cm-2.s-1.K-4
    distortion_type = config_dict['distortion']

    # Read parameters from input and check they're sensible
    if not config_dict['apply_colors']:
        len_params = 7
        teff1, teff2, theta1, theta2, ebv, sigma_ext, sigma_l = params[0:len_params]
        sigma_col = None
    else:
        len_params = 8
        teff1, teff2, theta1, theta2, ebv, sigma_ext, sigma_l, sigma_col = params[0:len_params]
        if sigma_col < 0:
            return -np.inf

    if theta1 < 0:
        return -np.inf
    if theta2 < 0:
        return -np.inf
    if ebv < 0:
        return -np.inf
    if sigma_ext < 0:
        return -np.inf
    if sigma_l < 0:
        return -np.inf

    # Get wave and flux information from spec1 and spec2 objects
    wave = spec1.waveset
    i = ((wmin * u.angstrom < wave) & (wave < wmax * u.angstrom)).nonzero()
    wave = wave[i]
    flux1 = spec1(wave, flux_unit=units.FLAM)
    flux2 = spec2(wave, flux_unit=units.FLAM)
    wave = wave.value  # Converts to numpy array
    flux1 = flux1.value
    flux2 = flux2.value

    # Converts wavelength space to x coordinates for Legendre polynomials
    x = 2 * np.log(wave / np.min(wave)) / np.log(np.max(wave) / np.min(wave)) - 1
    # Make empty distortion polynomial object
    distort1 = np.zeros_like(flux1)
    for n, c in enumerate(params[len_params:len_params+nc]):
        if abs(c) > 1:
            return -np.inf  # Check distortion coefficients are between -1 and +1
        distort1 = distort1 + c * legendre(n + 1)(x)
    # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
    i_5556 = np.argmin(abs(wave - 5556))
    distort1 = distort1 - distort1[i_5556]
    if min(distort1) < -1:
        return -np.inf
    flux1 = flux1 * (1 + distort1)
    flux1 = flux1 / simps(flux1, wave)

    if distortion_type == 1 or distortion_type == 0:
        # Applies same distortion polynomial to flux of second star
        distort2 = distort1
        flux2 = flux2 * (1 + distort2)
        flux2 = flux2 / simps(flux2, wave)
    else:
        # Makes new distortion polynomial for second star
        distort2 = np.zeros_like(flux2)
        for n, c in enumerate(params[len_params+nc:len_params+2*nc:]):
            if abs(c) > 1:
                return -np.inf  # Check distortion coefficients are between -1 and +1
            distort2 = distort2 + c * legendre(n + 1)(x)
        # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
        distort2 = distort2 - distort2[i_5556]
        if min(distort2) < -1:
            return -np.inf
        flux2 = flux2 * (1 + distort2)
        flux2 = flux2 / simps(flux2, wave)

    # Convert these bolometric fluxes to fluxes observed at the top of Earth's atmosphere
    extinction = redlaw.extinction_curve(ebv)(wave).value
    f_1 = 0.25 * sigma_sb * (theta1 / 206264806) ** 2 * teff1 ** 4 * flux1
    f_2 = 0.25 * sigma_sb * (theta2 / 206264806) ** 2 * teff2 ** 4 * flux2
    flux = (f_1 + f_2) * extinction  # Total "observed" flux
    if return_flux:
        # Bit of a bodge - not much point returning distort2 if = distort1, but saves it breaking somewhere else
        return wave, flux, f_1 * extinction, f_2 * extinction, distort1, distort2

    # Synthetic vs observed colours and magnitudes
    if config_dict['apply_colors']:
        chisq, lnlike_m, lnlike_c = flux2mag(wave, flux, sigma_ext, sigma_col, apply_colors=True)
    else:
        chisq, lnlike_m = flux2mag(wave, flux, sigma_ext)

    if verbose:
        print('Magnitudes')
        print('Band    Pivot     Observed     Calculated     O-C')
        for k in flux2mag.syn_mag.keys():
            o = flux2mag.obs_mag[k]
            c = flux2mag.syn_mag[k]
            w = flux2mag.w_pivot[k]
            print("{:6s} {:6.0f} {:6.3f} {:6.3f} {:+6.3f}".format(k, w, o, c, o-c))
        if config_dict['apply_colors']:
            print('Colours')
            print('Band    Observed     Calculated     O-C')
            for k in flux2mag.syn_col.keys():
                o = flux2mag.obs_col[k]
                c = flux2mag.syn_col[k]
                print("{:8s} {:6.3f} {:6.3f} {:+6.3f}".format(k, o, c, o-c))

    # Synthetic vs observed flux ratios
    lnlike_l = 0
    blob_data = []
    if verbose:
        print('Flux ratios')
        print('Band  Synthetic  Observed +/- Error')
    for k in lratios.keys():
        R = lratios[k]['R']
        if lratios[k]['photon']:
            l1 = simps(R(wave) * f_1 * wave, wave)
            l2 = simps(R(wave) * f_2 * wave, wave)
        else:
            l1 = simps(R(wave) * f_1, wave)
            l2 = simps(R(wave) * f_2, wave)
        v = lratios[k]['Value']
        lrat = l2 / l1
        blob_data.append(lrat)
        wt = 1 / (v.s ** 2 + sigma_l ** 2)
        lnlike_l += -0.5 * ((lrat - v.n) ** 2 * wt - np.log(wt))
        if verbose:
            print("{:4s} {:6.3f} {:6.3f} +/- {:5.3f}".format(k, lrat, v.n, v.s))

    # Angular diameter log likelihood
    # See http://mathworld.wolfram.com/BivariateNormalDistribution.html, equation (1)
    rho = correlation_matrix([theta1_in, theta2_in])[0][1]
    z = ((theta1 - theta1_in.n) ** 2 / theta1_in.s ** 2 -
         2 * rho * (theta1 - theta1_in.n) * (theta2 - theta2_in.n) / theta1_in.s / theta2_in.s +
         (theta2 - theta2_in.n) ** 2 / theta2_in.s ** 2)
    lnlike_theta = -0.5 * z / (1 - rho ** 2)

    # Combine log likelihoods calculated so far
    lnlike = lnlike_m + lnlike_theta + lnlike_l
    if config_dict['apply_colors']:
        lnlike += lnlike_c

    # Applying prior on interstellar reddening (if relevant)
    lnprior = 0
    if config_dict['apply_ebv_prior']:
        lnprior += -0.5 * (ebv - ebv_prior.n) ** 2 / ebv_prior.s ** 2

    # Applying prior on radius ratio (if needed)
    if config_dict['apply_k_prior']:
        k_prior = list_to_ufloat(config_dict['k'])
        lnprior += -0.5*((theta2/theta1 - k_prior.n)/k_prior.s) ** 2

    # Applying priors on NIR flux ratios (if relevant)
    if config_dict['apply_fratio_prior']:
        if not frp_coeffs:
            pass
        RV = flux2mag.R['V'](wave)  # Response function of V band over wavelength range
        lV = simps(RV * f_2, wave) / simps(RV * f_1, wave)  # Synthetic flux ratio in V band
        # Loads reference temperatures and method from frp configuration file
        stream = open('config/flux_ratio_priors.yaml', 'r')
        constraints = yaml.safe_load(stream)
        tref1_frp = np.mean(np.array(constraints['tref1']))
        tref2_frp = np.mean(np.array(constraints['tref2']))
        method = constraints['method']
        # Calculates flux ratio priors using synthetic flux ratio and temperatures just calculated
        frp_dict = frp.flux_ratio_priors(lV, teff1, teff2, tref1_frp, tref2_frp, frp_coeffs, method=method)
        if verbose:
            print('Flux ratio priors:')
        chisq_flux_ratio_priors = 0
        for b in frp_dict.keys():
            RX = flux2mag.R[b](wave)  # Response function in any X band over wavelength range
            lX = simps(RX * f_2 * wave, wave) / simps(RX * f_1 * wave, wave)  # Synthetic flux ratio in any X band
            if verbose:
                print('{:<2s}: {:0.3f}  {:0.3f}  {:+0.3f}'.format(b, frp_dict[b], lX, frp_dict[b] - lX))
            chisq_flux_ratio_priors += (lX - frp_dict[b].n) ** 2 / (frp_dict[b].s ** 2 + sigma_l ** 2)
            # Apply the prior to overall log prior
            wt = 1 / (frp_dict[b].s ** 2 + sigma_l ** 2)
            lnprior += -0.5 * ((lX - frp_dict[b].n) ** 2 * wt - np.log(wt))
        if verbose:
            print('Flux ratio priors: chi-squared = {:0.2}'.format(chisq_flux_ratio_priors))

    if debug:
        f = "{:0.1f} {:0.1f} {:0.4f} {:0.4f} {:0.4f} {:0.1f} {:0.4f}"
        f += " {:0.1e}" * nc
        if distortion_type == 2:
            nc2 = len(params) - 8 - nc  # TODO: potential issue here, length of params hard coded
            f += " {:0.1e}" * nc2
            f += " {:0.2f}"
        print(f.format(*(tuple(params) + (lnlike,))))
    if np.isfinite(lnlike):
        if blobs:
            return (lnlike + lnprior, *blob_data)
        else:
            return lnlike + lnprior
    else:
        return -np.inf


def run_mcmc_simulations(arguments, config_dict, least_squares_solution, n_steps=1000, n_walkers=256):
    """
    Runs MCMC via the emcee module, using the least squares solution as a starting point

    Parameters
    ----------
    arguments: list
        Starting values as list
    config_dict: dict
        Dictionary of configuration parameters from config.yaml
    least_squares_solution: `scipy.optimize.OptimizeResult`
        Output of minimization
    n_steps: int, optional
        Number of MCMC simulations to perform. Default = 1000.
    n_walkers: int, optional
        Number of walkers to use. Default = 256.

    Returns
    -------
    `emcee.sampler` object
    """
    nc = config_dict['n_coeffs']
    th1, th2 = angular_diameters(config_dict)
    # Fix problem with number of elements in steps if sigma_c, sigma_l are not parameters
    if config_dict['apply_colors']:
        steps = np.array([25, 25,  # T_eff,1, T_eff,2
                          th1.s, th2.s,  # theta_1 ,theta_2
                          0.001, 0.001, 0.001, 0.001,  # E(B-V), sigma_ext, sigma_l, sigma_c
                          *[0.01] * nc, *[0.01] * nc])  # c_1,1 ..   c_2,1 ..
    else:
        steps = np.array([25, 25,  # T_eff,1, T_eff,2
                          th1.s, th2.s,  # theta_1 ,theta_2
                          0.001, 0.001, 0.001,  # E(B-V), sigma_ext, sigma_l
                          *[0.01] * nc, *[0.01] * nc])  # c_1,1 ..   c_2,1 ..

    ndim = len(least_squares_solution.x)
    pos = np.zeros([n_walkers, ndim])
    for i, x in enumerate(least_squares_solution.x):
        pos[:, i] = x + steps[i] * np.random.randn(n_walkers)

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, args=arguments, pool=pool)
        sampler.run_mcmc(pos, n_steps, progress=True)

    return sampler


def convergence_plot(samples, parameter_names, config_dict):
    """
    Generates plots to show convergence of the temperatures and angular diameters

    Parameters
    ----------
    samples:
        samples object from emcee.sampler.get_chain()
    parameter_names: list
        List of the parameter names
    config_dict: dict
        Dictionary containing parameters, loaded from config.yaml

    Returns
    -------
    Convergence plot
    """
    # Read info from config file
    if config_dict['apply_colors']:
        n_panels = 8
    else:
        n_panels = 7
    teff1, teff2 = config_dict['teff1'], config_dict['teff2']
    m_h, aFe = config_dict['m_h'], config_dict['aFe']
    model, binning = config_dict['model_sed'], config_dict['binning']
    run_id, name = config_dict['run_id'], config_dict['name']

    # Create plot
    fig, axes = plt.subplots(n_panels, figsize=(10, 10), sharex='col')
    i0 = 0
    labels = parameter_names[i0:i0 + n_panels]
    for i in range(n_panels):
        ax = axes[i]
        ax.plot(samples[:, :, i0 + i], "k", alpha=0.3)
        ax.set(xlim=(0, len(samples)), ylabel=labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set(xlabel="Step number")
    fig.suptitle(f"Convergence plot for {name} ({run_id}) \n"
                 f"Model SED source: {model}\n"
                 f"Teff1 = {teff1}, Teff2 = {teff2}, [M/H] = {m_h}, [a/Fe] = {aFe}", fontsize=14)
    fig.align_labels()

    # Save and display
    if config_dict['save_plots']:
        fname = f"output/{run_id}_{name}_{teff1}_{teff2}_{m_h}_{aFe}_{binning}A_bins_convergence.png"
        plt.savefig(fname)
    plt.show()


def print_mcmc_solution(flat_samples, parameter_names):
    """
    Prints the best values from the MCMC

    Parameters
    ----------
    flat_samples: array_like
        Flattened chain from the emcee sampler
    parameter_names: list
        List of the parameter names

    """
    for i, pn in enumerate(parameter_names):
        val = flat_samples[:, i].mean()
        err = flat_samples[:, i].std()
        ndp = 1 - min(0, np.floor((np.log10(err))))
        fmt = '{{:0.{:0.0f}f}}'.format(ndp)
        v_str = fmt.format(val)
        e_str = fmt.format(err)
        print('{} = {} +/- {}'.format(pn, v_str, e_str))


def distortion_plot(best_pars, flux2mag, lratios, theta1, theta2, spec1, spec2, ebv_prior, redlaw, nc,
                    frp_coeffs, config_dict, flat_samples):
    """
    Generates plot showing final integrating functions and distortion for both stars
    Parameters
    ----------
    best_pars: list
        Model parameters and hyper-parameters as list. Should contain:
            teff1, teff2, theta1, theta2, ebv, sigma_ext, sigma_l(, sigma_c, [0]*(nc * 1 or 2))
    flux2mag: `flux2mag.Flux2Mag`
        Magnitude data and flux-to-mag log-likelihood calculator (Flux2Mag object)
    lratios: dict
        Flux ratios and response functions
    theta1: `uncertainties.ufloat`
        Angular diameter of primary star in mas (initial value)
    theta2: `uncertainties.ufloat`
        Angular diameter of secondary star in mas (initial value)
    spec1: `synphot.SourceSpectrum`
        Model spectrum of primary star
    spec2: `synphot.SourceSpectrum`
        Model spectrum of secondary star
    ebv_prior: `uncertainties.ufloat`
        Prior on E(B-V)
    redlaw: `synphot.ReddeningLaw`
        Reddening law
    nc: int
        Number of distortion coefficients for primary star (star 1)
    config_dict: dict
        Dictionary containing configuration parameters, from config.yaml file
    frp_coeffs: dict, optional
        Dictionary of flux ratio prior coefficients over suitable temperature range
    flat_samples: array_like
        Flattened chain from the emcee sampler

    Returns
    -------
    Distortion plot
    """
    if config_dict['distortion'] == 2:
        n_panels = 3
        gridspec = {'height_ratios': [5, 2, 2]}
        wave, flux, f1, f2, d1, d2 = lnprob(best_pars, flux2mag, lratios, theta1, theta2, spec1, spec2, ebv_prior,
                                            redlaw, nc, config_dict, frp_coeffs, return_flux=True)
    elif config_dict['distortion'] == 1:
        n_panels = 2
        gridspec = {'height_ratios': [4, 2]}
        wave, flux, f1, f2, d1, _ = lnprob(best_pars, flux2mag, lratios, theta1, theta2, spec1, spec2, ebv_prior,
                                        redlaw, nc, config_dict, frp_coeffs, return_flux=True)
    else:
        return None

    fig, ax = plt.subplots(n_panels, figsize=(8, 3*n_panels), sharex='col', gridspec_kw=gridspec)
    fig.subplots_adjust(hspace=0.05)
    fig.suptitle(f"Convergence plot for {config_dict['name']} ({config_dict['run_id']}) \n"
                 f"Model SED source: {config_dict['model_sed']}\n"
                 f"Teff1 = {config_dict['teff1']}, Teff2 = {config_dict['teff2']}, "
                 f"[M/H] = {config_dict['m_h']}, [a/Fe] = {config_dict['aFe']}", fontsize=14)

    # Integrating functions panel
    ax[0].semilogx(wave, 1e12 * f1, c='#003f5c', label='Primary')  # c='#252A6C'
    ax[0].semilogx(wave, 1e12 * f2, c='#ffa600', label='Secondary')  # c='#CEC814'
    ax[0].set(xlim=(1002, 299998), ylim=(-0.0, 0.25), yticks=(np.arange(0, 0.25, 0.05)),
              ylabel=r'$f_{\lambda}$  [10$^{-12}$ erg cm$^{-2}$ s$^{-1}$]')
    ax[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    ax[0].legend()

    # Primary distortion functions panel
    ax[1].semilogx(wave, d1, c='black')
    ax[1].set(xlabel=r'Wavelength [$\AA$]', ylabel=r'$\Delta_1$', ylim=(-0.35, 0.35))
    ax[1].yaxis.set_minor_locator(MultipleLocator(0.05))

    if config_dict['distortion'] == 2:
        # Secondary distortion functions panel
        ax[2].semilogx(wave, d2, c='black')
        ax[2].set(xlabel=r'Wavelength [$\AA$]', ylabel=r'$\Delta_2$', ylim=(-0.35, 0.35))
        ax[2].yaxis.set_minor_locator(MultipleLocator(0.05))

        # Plot a subset of distortion polynomials for both distortion panels
        for i in range(0, len(flat_samples), len(flat_samples) // 64):
            _, _, _, _, _d1, _d2 = lnprob(
                flat_samples[i, :], flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, nc, config_dict, frp_coeffs, return_flux=True)
            ax[1].semilogx(wave, _d1, c='black', alpha=0.1)
            ax[2].semilogx(wave, _d2, c='black', alpha=0.1)
    elif config_dict['distortion'] == 1:
        # Plot a subset of distortion polynomials in only one panel
        for i in range(0, len(flat_samples), len(flat_samples) // 64):
            _, _, _, _, _d1, _ = lnprob(
                flat_samples[i, :], flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, nc, config_dict, frp_coeffs, return_flux=True)
            ax[1].semilogx(wave, _d1, c='black', alpha=0.1)

    fig.align_ylabels()

    # Display plot and save plot to output directory
    if config_dict['save_plots']:
        f_name = f"output/{config_dict['run_id']}_{config_dict['name']}_{config_dict['teff1']}_" \
                 f"{config_dict['teff2']}_{config_dict['m_h']}_{config_dict['aFe']}" \
                 f"_{config_dict['binning']}A_bins_corner.png"
        plt.savefig(f_name)
    plt.show()


def input_photometry_plot(config_dict, flux2mag):
    """
    Makes a simple plot of input observed magnitudes along with filter response functions
    Parameters
    ----------
    config_dict: dict
        Dictionary containing configuration parameters, from config.yaml file
    flux2mag: `flux2mag.Flux2Mag`
        Magnitude data and flux-to-mag log-likelihood calculator (Flux2Mag object)

    Returns
    -------
    Plot of magnitude and response against wavelength
    """
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.suptitle(f"Input photometry for {config_dict['name']} ({config_dict['run_id']})")
    wave_plot_range = np.linspace(1000, 300000, 15000)
    ax.semilogx(wave_plot_range, np.linspace(-1, -0.1, 15000))
    ax2 = ax.twinx()

    for key in flux2mag.obs_mag:
        ax.errorbar(flux2mag.w_pivot[key], flux2mag.obs_mag[key].n, yerr=flux2mag.obs_mag[key].s,
                    fmt='o', ms=4, c='#003f5c')
        ax2.plot(wave_plot_range, flux2mag.R[key](wave_plot_range) / max(flux2mag.R[key](wave_plot_range)), c='#003f5c',
                 alpha=0.2)

    ax.set(xlim=(1000, 299998), ylim=(max(flux2mag.obs_mag.values()).n + 0.5, min(flux2mag.obs_mag.values()).n - 0.5),
           xlabel=r'Wavelength [$\AA$]', ylabel='AB Magnitude')
    ax2.set(ylim=(0, 1.1), ylabel='Normalised Response')

    # Display plot and save plot to output directory
    if config_dict['save_plots']:
        f_name = f"output/{config_dict['run_id']}_{config_dict['name']}_input_photometry.png"
        plt.savefig(f_name)
    plt.show()
