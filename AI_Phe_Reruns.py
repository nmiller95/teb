import numpy as np
from matplotlib import pylab as plt
from astropy.table import Table
from scipy.integrate import simps
from uncertainties import ufloat, covariance_matrix, correlation_matrix
from uncertainties.umath import log10
from scipy.interpolate import interp1d
import flint
import astropy.units as u
from synphot import units, ReddeningLaw
from scipy.special import legendre
import emcee
import corner
from multiprocessing import Pool
import pickle
from scipy.optimize import minimize
from response import extra_data, colors_data
from flux2mag import Flux2mag


flux2mag = Flux2mag('AI Phe', extra_data, colors_data)

for k in flux2mag.obs_mag.keys():
    o = flux2mag.obs_mag[k]
    w = flux2mag.w_pivot[k]
    # print("{:4s} {:6.0f} {:6.4f}".format(k,w,o))
# print("Color data")
for col in flux2mag.colors_data:
    s = col['tag']
    t = col['type']
    o = col['color']
    # print("{:8s} {:3s} {:6.4f}".format(s,t,o))


l = pickle.load(open("lratio_priors.pickle", "rb"))
# Fix typo in R1, R2 values
l['R1']['Value'] = ufloat(1.197, 0.024)
l['R2']['Value'] = ufloat(1.198, 0.024)
l['u']['Value'] = ufloat(0.475, 0.017)
l['v']['Value'] = ufloat(0.624, 0.009)
l['b']['Value'] = ufloat(0.870, 0.006)
l['y']['Value'] = ufloat(1.036, 0.007)
l['u320']['Value'] = ufloat(0.342, 0.042)
l['u220n']['Value'] = ufloat(0.030, 0.066)
l['u220w']['Value'] = ufloat(0.059, 0.090)
# Fix broken Wavelength array for TESS entry
# l['TESS']['Wavelength'] = np.array([10*float(s[:-1]) for s in l['TESS']['Wavelength']])
# .. and update the value
l['TESS']['Value'] = ufloat(1.319, 0.001)
# Convert wavelength/response to interpolating functions
lratios = {}
for k in l.keys():
    if l[k]['Response'] is not None:

        d = {}
        d['Value'] = l[k]['Value']
        w = np.array(l[k]['Wavelength'], dtype='f8')
        R = l[k]['Response']
        d['R'] = interp1d(w, R, bounds_error=False, fill_value=0)
        if k == "TESS":
            d['photon'] = True
        else:
            d['photon'] = False
        lratios[k] = d
        print(k, d['Value'], d['photon'])
# H-band flux ratio from Gallenne et al., 2019
# Use a nominal error of 0.01
k = 'H'
d = {
    'Value': ufloat(100 / 49.7, 0.01),
    'R': flux2mag.R[k],
    'photon': True
}
lratios[k] = d
print(k, d['Value'], d['photon'])


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


frp = flux_ratio_priors(1.05, 6440, 5220)


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


plx_Gallenne = ufloat(5.905, 0.024)
gaia_zp = ufloat(-0.031, 0.011)
plx_DR2 = ufloat(5.8336, 0.0262) - gaia_zp
plx = (plx_Gallenne + plx_DR2) / 2


Tref1 = 6200
Tref2 = 5100
M_H = -0.14
aFe = 0.06
spec1a = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=50, M_H=0.0, aFe=0.0, reload=True)
spec1b = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=50, M_H=-0.5, aFe=0.2, reload=True)
spec2a = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=50, M_H=0.0, aFe=0.0, reload=True)
spec2b = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=50, M_H=-0.5, aFe=0.2, reload=True)

spec1 = 0.72 * spec1a + 0.28 * spec1b
spec2 = 0.72 * spec2a + 0.28 * spec2b

# No detectable NaI lines so E(B-V) must be very close to 0 - see 2010NewA...15..444K
ebv_prior = ufloat(0.0, 0.005)  # No detectable NaI lines so E(B-V) must be very close to 0
redlaw = ReddeningLaw.from_extinction_model('mwavg')

# Angular diameter = 2*R/d = 2*R*parallax = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
# R_Sun = 6.957e8 m
# parsec = 3.085677581e16 m
# R_1 = ufloat(1.835, 0.014)    # JK-K values
# R_2 = ufloat(2.912, 0.014)    # JK-K values
R_1 = ufloat(1.8050, 0.0046)  # Prelimanary values from TESS analysis
R_2 = ufloat(2.9343, 0.0034)  # Prelimanary values from TESS analysis
theta1 = 2 * plx * R_1 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
theta2 = 2 * plx * R_2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
# print('theta1 = {:0.4f} mas'.format(theta1))
# print('theta2 = {:0.4f} mas'.format(theta2))
theta_cov = covariance_matrix([theta1, theta2])[0][1]
theta_cor = correlation_matrix([theta1, theta2])[0][1]
# print('cov(theta_1,theta2) = {:0.2e}'.format(theta_cov))
# print('cor(theta_1,theta2) = {:0.2f}'.format(theta_cor))

Teff1 = 6223
Teff2 = 5135
# Copy starting values to new variables
theta1_ = theta1.n
theta2_ = theta2.n
ebv_ = ebv_prior.n
sigma_ext = 0.008
sigma_l = 0.01
sigma_c = 0.005
Nc1 = 10
Nc2 = 10
params = [Teff1, Teff2, theta1_, theta2_, ebv_, sigma_ext, sigma_l, sigma_c]
params = params + [0] * Nc1
params = params + [0] * Nc2

parname = ['T_eff,1', 'T_eff,2', 'theta_1', 'theta_2', 'E(B-V)', 'sigma_ext', 'sigma_l', 'sigma_c']
parname = parname + ["c_1,{}".format(j + 1) for j in range(Nc1)]
parname = parname + ["c_2,{}".format(j + 1) for j in range(Nc2)]

for pn, pv in zip(parname, params):
    print('{} = {}'.format(pn, pv))

lnlike = lnprob(params, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1, verbose=True)
# print('Initial log-likelihood = {:0.2f}'.format(lnlike))


nll = lambda *args: -lnprob(*args)
args = (flux2mag, lratios, theta1, theta2,
        spec1, spec2, ebv_prior, redlaw, Nc1)
soln = minimize(nll, params, args=args, method='Nelder-Mead')

# print('theta1 = {:0.4f} mas'.format(theta1))
# print('theta2 = {:0.4f} mas'.format(theta2))
# print()
# for pn,pv in zip(parname, soln.x):
#     print('{} = {}'.format(pn,pv))

lnlike = lnprob(soln.x, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1, verbose=True)
# print('Final log-likelihood = {:0.2f}'.format(lnlike))


steps = [25, 25,  # T_eff,1, T_eff,2
         0.0005, 0.0007,  # theta_1 ,theta_2
         0.001, 0.001, 0.001, 0.001,  # E(B-V), sigma_ext, sigma_l, sigma_c
         *[0.01] * Nc1, *[0.01] * Nc2]  # c_1,1 ..   c_2,1 ..

nwalkers = 256
ndim = len(soln.x)
pos = np.zeros([nwalkers, ndim])
for i, x in enumerate(soln.x):
    pos[:, i] = x + steps[i] * np.random.randn(nwalkers)

nsteps = 10
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, pool=pool)
    sampler.run_mcmc(pos, nsteps, progress=True)

af = sampler.acceptance_fraction
# print('\nMedian acceptance fraction =',np.median(af))
best_index = np.unravel_index(np.argmax(sampler.lnprobability),
                              (nwalkers, nsteps))
best_lnlike = np.max(sampler.lnprobability)
# print('\n Best log(likelihood) = ',best_lnlike,' in walker ',best_index[0],
#        ' at step ',best_index[1])
best_pars = sampler.chain[best_index[0], best_index[1], :]

fig, axes = plt.subplots(4, figsize=(10, 7), sharex='all')
samples = sampler.get_chain()
i0 = 0
labels = parname[i0:i0 + 4]
for i in range(4):
    ax = axes[i]
    ax.plot(samples[:, :, i0 + i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("Step number")
fig.show()

flat_samples = sampler.get_chain(discard=4000, thin=8, flat=True)  # nsteps//2
fig = corner.corner(flat_samples, labels=parname)
fig.show()

for i, pn in enumerate(parname):
    val = flat_samples[:, i].mean()
    err = flat_samples[:, i].std()
    ndp = 1 - min(0, np.floor(log10(err)))
    fmt = '{{:0.{:0.0f}f}}'.format(ndp)
    vstr = fmt.format(val)
    estr = fmt.format(err)
    # print('{} = {} +/- {}'.format(pn,vstr,estr))

lnlike = lnprob(best_pars, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1, verbose=True)
# print('Final log-likelihood = {:0.2f}'.format(lnlike))

wave, flux, f_1, f_2, d1, d2 = lnprob(
    best_pars, flux2mag, lratios,
    theta1, theta2, spec1, spec2,
    ebv_prior, redlaw, Nc1, return_flux=True)
fig1, ax1 = plt.subplots(3, figsize=(10, 7), sharex='all')
ax1[0].semilogx(wave, 1e12 * f_1, c='c')
ax1[0].semilogx(wave, 1e12 * f_2, c='orange')
ax1[0].set_xlim(1000, 300000)
ax1[0].set_ylabel(r'$f_{\lambda}\:\:[10^{-12}\,{\rm ergs}\,{\rm cm}^{-2}\,{\rm s}^{-1}\,{\rm \AA}^{-1}}]$')
ax1[1].semilogx(wave, d1, c='b')
ax1[1].set_ylabel('$\Delta_1$')
ax1[1].set_ylim(-0.25, 0.25)
ax1[2].semilogx(wave, d2, c='b')
ax1[2].set_ylabel('$\Delta_2$')
ax1[2].set_xlabel(r'Wavelength [$\rm \AA$]')
ax1[2].set_ylim(-0.25, 0.25)

for i in range(0, len(flat_samples), len(flat_samples) // 64):
    _, _, _, _, _d1, _d2 = lnprob(
        flat_samples[i, :], flux2mag, lratios,
        theta1, theta2, spec1, spec2,
        ebv_prior, redlaw, Nc1, return_flux=True)
    ax1[1].semilogx(wave, _d1, c='b', alpha=0.1)
    ax1[2].semilogx(wave, _d2, c='b', alpha=0.1)

fig.show()

VegaZeroPointErrorPercent = 0.5
Fig14Data = Table.read('Bohlin2014_Fig14.csv', names=['w', 'err'])
WDScaleErrorWavelengthAngstrom = Fig14Data['w'] * 10000
WDScaleErrorPercent = Fig14Data['err']
TotalSystematicErrorPercent = VegaZeroPointErrorPercent + WDScaleErrorPercent
plt.semilogx(WDScaleErrorWavelengthAngstrom, TotalSystematicErrorPercent, 'bo')
Interpolator = interp1d(WDScaleErrorWavelengthAngstrom, TotalSystematicErrorPercent, bounds_error=False,
                        fill_value=50.0)
WavelengthGrid = np.linspace(min(WDScaleErrorWavelengthAngstrom), max(WDScaleErrorWavelengthAngstrom), 50001)
TotSysErrPercentGrid = Interpolator(WavelengthGrid)
plt.semilogx(WavelengthGrid, TotSysErrPercentGrid)
plt.xlabel(r'Wavelength [$\AA$]')
plt.ylabel('Flux scale error [%]')
plt.show()

TotSysErrPercentGrid = Interpolator(wave)
T_eff_1 = flat_samples[:, 0].mean()
rnderr_1 = flat_samples[:, 0].std()
fint_1 = simps(f_1, wave)
fint_1p = simps(f_1 * (1 + TotSysErrPercentGrid / 100), wave)
syserr_1 = (fint_1p / fint_1 - 1) * T_eff_1 / 4  # /4 because L \propto Teff^4
# print('Systematic error in integrated flux = {:0.2%}%'.format((fint_1p/fint_1-1)))
# print('T_eff,1 = {:0.0f} +/- {:0.0f} (rnd.) +/- {:0.0f} (sys.) K'.
#     format(T_eff_1, rnderr_1, syserr_1))

T_eff_2 = flat_samples[:, 1].mean()
rnderr_2 = flat_samples[:, 1].std()
fint_2 = simps(f_2, wave)
fint_2p = simps(f_2 * (1 + TotSysErrPercentGrid / 100), wave)
syserr_2 = (fint_2p / fint_2 - 1) * T_eff_2 / 4
# print('Systematic error in integrated flux = {:0.2%}%'.format((fint_2p/fint_2-1)))
# print('T_eff,2 = {:0.0f} +/- {:0.0f} (rnd.) +/- {:0.0f} (sys.) K'.
#          format(T_eff_2, rnderr_2, syserr_2))


tag = "C"
pfile = "{}_{:0.0f}_{:0.0f}+{:+0.1f}_{}_coeffs.p".format(tag, Tref1, Tref2, M_H, Nc1)
with open(pfile, 'wb') as f:
    pickle.dump(sampler, f)
