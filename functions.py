from uncertainties import ufloat, correlation_matrix
import numpy as np
import astropy.units as u
from synphot import units
from scipy.special import legendre
from scipy.integrate import simps


def list_to_ufloat(two_item_list):
    """Turns a two item list from yaml input into a ufloat"""
    return ufloat(two_item_list[0], two_item_list[1])


def angular_diameters(param_dict):
    """
    Converts input radii and Gaia parallax to angular diameters in mas
    :param param_dict: Dictionary containing parameters, loaded from config.yaml
    :return: Angular diameters for primary and secondary stars as ufloats.
    """
    # Parallax load and apply zeropoint
    gaia_zp = ufloat(-0.017, 0.011)  # Gaia EDR3. ZP error fixed at DR2 value for now
    plx = list_to_ufloat(param_dict['plx']) - gaia_zp

    # Angular diameter = 2*R/d = 2*R*parallax = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
    # R_Sun = 6.957e8 m. parsec = 3.085677581e16 m
    r1 = list_to_ufloat(param_dict['r1'])
    r2 = list_to_ufloat(param_dict['r2'])
    theta1 = 2 * plx * r1 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
    theta2 = 2 * plx * r2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
    return theta1, theta2


def initial_parameters(param_dict, theta1, theta2, ebv_prior):
    """
    Loads and generates parameters for log likelihood calculations
    :param param_dict: Dictionary containing parameters, loaded from config.yaml
    :param theta1: Angular diameter of primary star in mas as ufloat
    :param theta2: Angular diameter of secondary star in mas as ufloat
    :param ebv_prior: Prior on interstellar reddening as ufloat
    :return: Parameters and parameter names as two lists
    """
    teff1 = param_dict['teff1']
    teff2 = param_dict['teff2']
    # Copy starting values to new variables
    theta1_ = theta1.n
    theta2_ = theta2.n
    ebv_ = ebv_prior.n
    sigma_ext = param_dict['sigma_ext']
    sigma_l = param_dict['sigma_l']
    nc = param_dict['n_coeffs']

    params = [teff1, teff2, theta1_, theta2_, ebv_, sigma_ext, sigma_l]
    parname = ['T_eff,1', 'T_eff,2', 'theta_1', 'theta_2', 'E(B-V)', 'sigma_ext', 'sigma_l']

    if param_dict['apply_colors']:
        sigma_c = param_dict['sigma_c']
        params = params + [sigma_c]
        parname = parname + ['sigma_c']

    if param_dict['distortion'] == 0:
        pass
    elif param_dict['distortion'] == 1:
        params = params + [0] * nc
        parname = parname + ["c_1,{}".format(j + 1) for j in range(nc)]
    elif param_dict['distortion'] == 2:
        params = params + [0] * 2 * nc
        parname = parname + ["c_1,{}".format(j + 1) for j in range(nc)]
        parname = parname + ["c_2,{}".format(j + 1) for j in range(nc)]
    return params, parname


def lnprob(params, flux2mag, lratios, theta1, theta2, spec1, spec2, ebv_prior, redlaw, Nc1,
           wmin=1000, wmax=300000, return_flux=False, blobs=False, apply_flux_ratio_priors=True,
           debug=False, verbose=False, distortion_type=2):
    """
    Log probability function for the fundamental effective temperature of eclipsing binary stars method.

    :param params: Model parameters and hyper-parameters as list. Should contain:
        Teff1, Teff2, theta1, theta2, ebv, sigma_ext, sigma_l, sigma_c, [0]*(Nc1+Nc2)
    :param flux2mag: Magnitude data and flux-to-mag log-likelihood calculator (Flux2Mag object)
    :param lratios: Flux ratios and response functions (dictionary)
    :param theta1: Angular diameter of primary star as ufloat in milli-arcseconds
    :param theta2: Angular diamter of secondary star as ufloat in milli-arcseconds
    :param spec1: Model spectrum of primary star (flint.ModelSpectrum object)
    :param spec2: Model spectrum of secondary star (flint.ModelSpectrum object)
    :param ebv_prior: Prior on E(B-V) as ufloat
    :param redlaw: Reddening law (synphot.ReddeningLaw object)
    :param Nc1: Number of distortion coefficients for primary star (star 1)
    :param wmin: Lower wavelength cut for model spectrum, in Angstroms
    :param wmax: Upper wavelength cut for model spectrum, in Angstroms
    :param return_flux: Whether to return the wavelength, flux and distortion arrays
    :param blobs: For the MCMC
    :param apply_flux_ratio_priors: Whether to apply the flux ratio priors
    :param debug: Whether to print extra stuff
    :param verbose: Whether to print out all the parameters
    :param distortion_type: 0, 1 or 2. If the 2 stars are very similar, can use 1 set of
        distortion coefficients for both stars.
    :return: Either log likelihood and log prior, or wavelength, flux and distortion arrays
    """
    sigma_sb = 5.670367E-5  # erg.cm-2.s-1.K-4

    Teff1, Teff2, Theta1, Theta2, ebv, sigma_ext, sigma_l, sigma_col = params[0:8]  # SIGMA_COL

    if Theta1 < 0:
        return -np.inf
    if Theta2 < 0:
        return -np.inf
    if ebv < 0:
        return -np.inf
    if sigma_ext < 0:
        return -np.inf
    if sigma_col < 0:
        return -np.inf  # SIGMA_COL
    if sigma_l < 0:
        return -np.inf

    wave = spec1.waveset
    i = ((wmin * u.angstrom < wave) & (wave < wmax * u.angstrom)).nonzero()
    wave = wave[i]
    flux1 = spec1(wave, flux_unit=units.FLAM)
    flux2 = spec2(wave, flux_unit=units.FLAM)
    wave = wave.value  # Converts to numpy array
    flux1 = flux1.value
    flux2 = flux2.value

    x = 2 * np.log(wave / np.min(wave)) / np.log(np.max(wave) / np.min(wave)) - 1
    distort1 = np.zeros_like(flux1)
    for n, c in enumerate(params[8:8 + Nc1]):
        if abs(c) > 1: return -np.inf
        distort1 = distort1 + c * legendre(n + 1)(x)
    # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
    i_5556 = np.argmin(abs(wave - 5556))
    distort1 = distort1 - distort1[i_5556]
    if min(distort1) < -1:
        return -np.inf
    flux1 = flux1 * (1 + distort1)
    flux1 = flux1 / simps(flux1, wave)

    if distortion_type == 1:
        flux2 = flux2 * (1 + distort1)
        flux2 = flux2 / simps(flux2, wave)
    elif distortion_type == 2:
        distort2 = np.zeros_like(flux2)
        for n, c in enumerate(params[8 + Nc1:]):
            if abs(c) > 1: return -np.inf
            distort2 = distort2 + c * legendre(n + 1)(x)
        # Make distortion = 0 at 5556A (where Vega z.p. flux is defined)
        distort2 = distort2 - distort2[i_5556]
        if min(distort2) < -1:
            return -np.inf
        flux2 = flux2 * (1 + distort2)
        flux2 = flux2 / simps(flux2, wave)
    else:
        pass  # TODO: zero distortion option

    extinc = redlaw.extinction_curve(ebv)(wave).value
    f_1 = 0.25 * sigma_sb * (Theta1 / 206264806) ** 2 * Teff1 ** 4 * flux1
    f_2 = 0.25 * sigma_sb * (Theta2 / 206264806) ** 2 * Teff2 ** 4 * flux2
    flux = (f_1 + f_2) * extinc
    if return_flux:
        if distortion_type == 1:
            return wave, flux, f_1 * extinc, f_2 * extinc, distort1
        elif distortion_type ==2:
            return wave, flux, f_1 * extinc, f_2 * extinc, distort1, distort2
        else:
            pass  # TODO: you know what.

    chisq, lnlike_m, lnlike_c = flux2mag(wave, flux, sigma_ext, sigma_col)  # SIGMA_COL
    if verbose:
        print('Magnitudes')
        for k in flux2mag.syn_mag.keys():
            o = flux2mag.obs_mag[k]
            c = flux2mag.syn_mag[k]
            w = flux2mag.w_pivot[k]
            print("{:6s} {:6.0f} {:6.3f} {:6.3f} {:+6.3f}".format(k, w, o, c, o - c))

    if verbose:  # STROMGREN_COLORS
        print('Colours')
        for k in flux2mag.syn_col.keys():
            o = flux2mag.obs_col[k]
            c = flux2mag.syn_col[k]
            print("{:8s} {:6.3f} {:6.3f} {:+6.3f}".format(k, o, c, o - c))

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

    # See http://mathworld.wolfram.com/BivariateNormalDistribution.html, equation (1)
    rho = correlation_matrix([theta1, theta2])[0][1]
    z = ((Theta1 - theta1.n) ** 2 / theta1.s ** 2 -
         2 * rho * (Theta1 - theta1.n) * (Theta2 - theta2.n) / theta1.s / theta2.s +
         (Theta2 - theta2.n) ** 2 / theta2.s ** 2)
    lnlike_theta = -0.5 * z / (1 - rho ** 2)

    lnlike = lnlike_m + lnlike_c + lnlike_theta + lnlike_l

    lnprior = 0
    if ebv_prior is not None:
        lnprior += -0.5 * (ebv - ebv_prior.n) ** 2 / ebv_prior.s ** 2
    # Priors on IR flux ratios
    RV = flux2mag.R['V'](wave)
    lV = simps(RV * f_2, wave) / simps(RV * f_1, wave)
    # IT WILL BREAK HERE TODO: stop this from breaking
    frp = flux_ratio_priors(lV,Teff1,Teff2, Tref1_frp, Tref2_frp, frp_coeffs)
    if verbose:
        print('Flux ratio priors:')
    chisq_flux_ratio_priors = 0
    for b in frp.keys():
        RX = flux2mag.R[b](wave)
        lX = simps(RX * f_2 * wave, wave) / simps(RX * f_1 * wave, wave)  # Predicted value
        if verbose: print('{:<2s}: {:0.3f}  {:0.3f}  {:+0.3f}'.format(b, frp[b], lX, frp[b] - lX))
        chisq_flux_ratio_priors += (lX - frp[b].n) ** 2 / (frp[b].s ** 2 + sigma_l ** 2)
        if apply_flux_ratio_priors:
            wt = 1 / (frp[b].s ** 2 + sigma_l ** 2)
            lnprior += -0.5 * ((lX - frp[b].n) ** 2 * wt - np.log(wt))
    if verbose:
        print('Flux ratio priors: chi-squared = {:0.2}'.format(chisq_flux_ratio_priors))

    if debug:
        f = "{:0.1f} {:0.1f} {:0.4f} {:0.4f} {:0.4f} {:0.1f} {:0.4f}"
        f += " {:0.1e}" * Nc1
        if not single_dist_function:
            Nc2 = len(params) - 8 - Nc1
            f += " {:0.1e}" * Nc2
            f += " {:0.2f}"
        print(f.format(*(tuple(params) + (lnlike,))))
    if np.isfinite(lnlike):
        if blobs:
            return (lnlike + lnprior, *blob_data)
        else:
            return lnlike + lnprior
    else:
        return -np.inf
