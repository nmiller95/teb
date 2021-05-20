from uncertainties import ufloat, correlation_matrix
import numpy as np
import astropy.units as u
from synphot import units
from scipy.special import legendre
from scipy.integrate import simps
import flux_ratio_priors as frp
import yaml


def list_to_ufloat(two_item_list):
    """Turns a two item list from yaml input into a ufloat"""
    return ufloat(two_item_list[0], two_item_list[1])


def angular_diameters(config_dict):
    """
    Converts input radii and Gaia parallax to angular diameters in mas
    :param config_dict: Dictionary containing parameters, loaded from config.yaml
    :return: Angular diameters for primary and secondary stars as ufloats.
    """
    # Parallax load and apply zeropoint
    gaia_zp = ufloat(-0.017, 0.011)  # Gaia EDR3. ZP error fixed at DR2 value for now
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
    :param config_dict: Dictionary containing parameters, loaded from config.yaml
    :param theta1: Angular diameter of primary star in mas as ufloat
    :param theta2: Angular diameter of secondary star in mas as ufloat
    :param ebv_prior: Prior on interstellar reddening as ufloat
    :return: Parameters and parameter names as two lists
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


def lnprob(params, flux2mag, lratios, theta1_in, theta2_in, spec1, spec2, ebv_prior, redlaw, nc, config_dict,
           frp_coeffs=None, wmin=1000, wmax=300000, return_flux=False, blobs=False,
           debug=False, verbose=False):  # TODO: not really sure what thetas are doing here but hey
    """
    Log probability function for the fundamental effective temperature of eclipsing binary stars method.

    :param params: Model parameters and hyper-parameters as list. Should contain:
        teff1, teff2, theta1, theta2, ebv, sigma_ext, sigma_l(, sigma_c, [0]*(nc * 1 or 2))
    :param flux2mag: Magnitude data and flux-to-mag log-likelihood calculator (Flux2Mag object)
    :param lratios: Flux ratios and response functions (dictionary)
    :param theta1_in: Angular diamater of primary star in mas
    :param theta2_in: Angular diameter of secondary star in mas
    :param spec1: Model spectrum of primary star (flint.ModelSpectrum object)
    :param spec2: Model spectrum of secondary star (flint.ModelSpectrum object)
    :param ebv_prior: Prior on E(B-V) as ufloat
    :param redlaw: Reddening law (synphot.ReddeningLaw object)
    :param nc: Number of distortion coefficients for primary star (star 1)
    :param frp_coeffs: Dictionary of flux ratio prior coefficients over suitable temperature range
    :param config_dict: Dictionary containing configuration parameters, from config.yaml file
    :param wmin: Lower wavelength cut for model spectrum, in Angstroms
    :param wmax: Upper wavelength cut for model spectrum, in Angstroms
    :param return_flux: Whether to return the wavelength, flux and distortion arrays
    :param blobs: For the MCMC
    :param debug: Whether to print extra stuff
    :param verbose: Whether to print out all the parameters
    :return: Either log likelihood and log prior, or wavelength, flux and distortion arrays
    """
    sigma_sb = 5.670367E-5  # erg.cm-2.s-1.K-4
    distortion_type = config_dict['distortion']

    # Read parameters from input and check they're sensible
    len_params = 7
    teff1, teff2, theta1, theta2, ebv, sigma_ext, sigma_l = params[0:len_params]
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

    if config_dict['apply_colors']:
        sigma_col = params[7]
        len_params = 8
        if sigma_col < 0:
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
            # TODO: check this should indeed be using secondary coeffs or whether the primary ones were deliberate
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
        chisq, lnlike_m, lnlike_c = flux2mag(wave, flux, sigma_ext, sigma_col)  # Ignore warning, pycharm is being fussy
    else:
        chisq, lnlike_m, _ = flux2mag(wave, flux, sigma_ext)  # TODO: fix this bodge

    if verbose:
        print('Magnitudes')
        for k in flux2mag.syn_mag.keys():
            o = flux2mag.obs_mag[k]
            c = flux2mag.syn_mag[k]
            w = flux2mag.w_pivot[k]
            print("{:6s} {:6.0f} {:6.3f} {:6.3f} {:+6.3f}".format(k, w, o, c, o-c))
        if config_dict['apply_colors']:
            print('Colours')
            for k in flux2mag.syn_col.keys():
                o = flux2mag.obs_col[k]
                c = flux2mag.syn_col[k]
                print("{:8s} {:6.3f} {:6.3f} {:+6.3f}".format(k, o, c, o-c))

    # Synthetic vs observed flux ratios
    lnlike_l = 0
    blob_data = []
    if verbose:
        print('Flux ratios')
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
    print(theta1, theta2)
    rho = correlation_matrix([theta1_in, theta2_in])[0][1]
    z = ((theta1 - theta1_in.n) ** 2 / theta1_in.s ** 2 -
         2 * rho * (theta1 - theta1_in.n) * (theta2 - theta2_in.n) / theta1_in.s / theta2_in.s +
         (theta2 - theta2_in.n) ** 2 / theta2_in.s ** 2)
    lnlike_theta = -0.5 * z / (1 - rho ** 2)

    # Combine log likelihoods calculated so far
    lnlike = lnlike_m + lnlike_theta + lnlike_l
    if config_dict['apply_colors']:
        lnlike += lnlike_c  # Ignore warning, pycharm is being fussy

    # Applying prior on interstellar reddening (if relevant)
    lnprior = 0
    if config_dict['apply_ebv_prior']:
        lnprior += -0.5 * (ebv - ebv_prior.n) ** 2 / ebv_prior.s ** 2

    # Applying priors on NIR flux ratios (if relevant)
    if config_dict['apply_flux_ratios']:
        if not frp_coeffs:
            pass
        RV = flux2mag.R['V'](wave)  # Response function of V band over wavelength range
        lV = simps(RV * f_2, wave) / simps(RV * f_1, wave)  # Synthetic flux ratio in V band
        # Loads reference temperatures and method from frp configuration file
        stream = open('config/flux_ratio_priors.yaml', 'r')
        constraints = yaml.safe_load(stream)
        tref1_frp = np.mean(np.array(constraints['Tref1']))
        tref2_frp = np.mean(np.array(constraints['Tref2']))
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
            nc2 = len(params) - 8 - nc
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
