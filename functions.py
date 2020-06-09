from uncertainties import ufloat, correlation_matrix
import numpy as np
import astropy.units as u
from scipy.special import legendre
from scipy.integrate import simps


def flux_ratio_priors(Vrat, TeffF, TeffK):
    # Vrat in the sense Flux_F/Flux_K
    data = {
        'J': {'cF': 0.919, 'mF': -0.408, 'sigF': 0.015, 'cK': 1.511, 'mK': -0.605, 'sigK': 0.018},
        'H': {'cF': 1.118, 'mF': -0.549, 'sigF': 0.019, 'cK': 1.918, 'mK': -0.821, 'sigK': 0.027},
        'Ks': {'cF': 1.181, 'mF': -0.564, 'sigF': 0.017, 'cK': 2.033, 'mK': -0.872, 'sigK': 0.025},
        'W1': {'cF': 1.230, 'mF': -0.568, 'sigF': 0.027, 'cK': 2.094, 'mK': -0.865, 'sigK': 0.035},
        'W2': {'cF': 1.234, 'mF': -0.547, 'sigF': 0.039, 'cK': 2.101, 'mK': -0.928, 'sigK': 0.062},
        'W3': {'cF': 1.182, 'mF': -0.554, 'sigF': 0.021, 'cK': 2.062, 'mK': -0.907, 'sigK': 0.036},
        'W4': {'cF': 1.225, 'mF': -0.519, 'sigF': 0.050, 'cK': 2.095, 'mK': -0.951, 'sigK': 0.060}
    }
    # Return a dictionary of ufloat priors on flux ratios
    d = {}
    for b in data.keys():
        colF = data[b]['cF'] + data[b]['mF'] * (TeffF - 6400) / 1000.0
        colK = data[b]['cK'] + data[b]['mK'] * (TeffK - 5200) / 1000.0
        L = Vrat * 10 ** (0.4 * (colK - colF))
        e_L = np.hypot(data[b]['sigF'], data[b]['sigK'])
        d[b] = ufloat(L, e_L)
    return d


def lnprob(params,  # Model parameters and hyper-parameters
           flux2mag,  # Magnitude data and flux-to-mag log-likelihood calculator
           lratios,  # Flux ratios and responses
           theta1, theta2,  # angular diameters as ufloats **in milli-arcseconds**
           spec1, spec2,  # Model spectra
           ebv_prior,  # ufloat
           redlaw,  # Reddening law
           Nc1,  # Number of distortion coeffs for star 1
           wmin=1000, wmax=300000,
           return_flux=False,
           blobs=False, apply_flux_ratio_priors=True,
           debug=False, verbose=False):
    SIGMA_SB = 5.670367E-5  # erg.cm-2.s-1.K-4

    Teff1, Teff2, Theta1, Theta2, ebv, sigma_ext, sigma_l, sigma_col = params[0:8]  # SIGMA_COL

    if Theta1 < 0: return -np.inf
    if Theta2 < 0: return -np.inf
    if ebv < 0: return -np.inf
    if sigma_ext < 0: return -np.inf
    if sigma_col < 0: return -np.inf  # SIGMA_COL
    if sigma_l < 0: return -np.inf

    wave = spec1.waveset
    i = ((wmin * u.angstrom < wave) & (wave < wmax * u.angstrom)).nonzero()
    wave = wave[i]
    flux1 = spec1(wave, flux_unit=u.FLAM)
    flux2 = spec2(wave, flux_unit=u.FLAM)
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

    extinc = redlaw.extinction_curve(ebv)(wave).value
    f_1 = 0.25 * SIGMA_SB * (Theta1 / 206264806) ** 2 * Teff1 ** 4 * flux1
    f_2 = 0.25 * SIGMA_SB * (Theta2 / 206264806) ** 2 * Teff2 ** 4 * flux2
    flux = (f_1 + f_2) * extinc
    if return_flux:
        return wave, flux, f_1 * extinc, f_2 * extinc, distort1, distort2

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
    frp = flux_ratio_priors(lV, Teff1, Teff2)
    if verbose: print('Flux ratio priors:')
    chisq_flux_ratio_priors = 0
    for b in frp.keys():
        RX = flux2mag.R[b](wave)
        lX = simps(RX * f_2 * wave, wave) / simps(RX * f_1 * wave, wave)  # Predicted value
        if verbose: print('{:<2s}: {:0.3f}  {:0.3f}  {:+0.3f}'.format(b, frp[b], lX, frp[b] - lX))
        chisq_flux_ratio_priors += (lX - frp[b].n) ** 2 / (frp[b].s ** 2 + sigma_l ** 2)
        if apply_flux_ratio_priors:
            wt = 1 / (frp[b].s ** 2 + sigma_l ** 2)
            lnprior += -0.5 * ((lX - frp[b].n) ** 2 * wt - np.log(wt))
    if verbose: print('Flux ratio priors: chi-squared = {:0.2}'.format(chisq_flux_ratio_priors))

    if debug:
        f = "{:0.1f} {:0.1f} {:0.4f} {:0.4f} {:0.4f} {:0.1f} {:0.4f}"
        f += " {:0.1e}" * Nc1
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
