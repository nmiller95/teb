import numpy as np
from matplotlib import pylab as plt
from astropy.table import Table
from scipy.integrate import simps
from uncertainties import ufloat, covariance_matrix, correlation_matrix
from uncertainties.umath import log10
from scipy.interpolate import interp1d
import flint
from synphot import ReddeningLaw
import emcee
import corner
from multiprocessing import Pool
import pickle
from scipy.optimize import minimize
from response import extra_data, colors_data
from flux2mag import Flux2mag
from functions import flux_ratio_priors, lnprob

# CHANGE RUN INFO HERE
# RUNS L, M, N REQUIRE MORE THOROUGH CHANGES TO THE SOURCE CODE
run = "B"
# Parameters
Nc1 = 10
Nc2 = 10
binning = 50
Tref1 = 6200
Tref2 = 5100
M_H = -0.14
aFe = 0.06
include_ebv_prior = True
apply_nir_priors = True

# INITIALISE FLUX2MAG CLASS WITH TARGET STAR
flux2mag = Flux2mag('AI Phe', extra_data, colors_data)

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

# H-band flux ratio from Gallenne et al., 2019
# Use a nominal error of 0.01
k = 'H'
d = {
    'Value': ufloat(100 / 49.7, 0.01),
    'R': flux2mag.R[k],
    'photon': True
}
lratios[k] = d

# CREATE FLUX RATIO PRIORS
# frp = flux_ratio_priors(1.05, 6440, 5220)

# DATA INPUT - Parallax
plx_Gallenne = ufloat(5.905, 0.024)
gaia_zp = ufloat(-0.031, 0.011)
plx_DR2 = ufloat(5.8336, 0.0262) - gaia_zp
plx = (plx_Gallenne + plx_DR2) / 2

# READ / GENERATE SUITABLE STELLAR MODELS
if M_H == -0.14:
    spec1a = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=binning, M_H=0.0, aFe=0.0, reload=True)
    spec1b = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=binning, M_H=-0.5, aFe=0.2, reload=True)
    spec2a = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=binning, M_H=0.0, aFe=0.0, reload=True)
    spec2b = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=binning, M_H=-0.5, aFe=0.2, reload=True)
    spec1 = 0.72 * spec1a + 0.28 * spec1b
    spec2 = 0.72 * spec2a + 0.28 * spec2b
elif M_H == 0.0 or M_H == -0.5:
    spec1 = flint.ModelSpectrum.from_parameters(Tref1, 4.0, binning=binning, M_H=M_H, aFe=aFe, reload=True)
    spec2 = flint.ModelSpectrum.from_parameters(Tref2, 3.5, binning=binning, M_H=M_H, aFe=aFe, reload=True)
else:
    raise ValueError

# No detectable NaI lines so E(B-V) must be very close to 0 - see 2010NewA...15..444K
if include_ebv_prior:
    ebv_prior = ufloat(0.0, 0.005)
redlaw = ReddeningLaw.from_extinction_model('mwavg')

# DATA INPUT - stellar radius
R_1 = ufloat(1.8050, 0.0022)  # Final values from TESS light curve - Maxted et al. (2020)
R_2 = ufloat(2.9332, 0.0023)  # Final values from TESS light curve - Maxted et al. (2020)
# Angular diameter = 2*R/d = 2*R*parallax = 2*(R/Rsun)*(pi/mas) * R_Sun/kpc
# R_Sun = 6.957e8 m,  parsec = 3.085677581e16 m
theta1 = 2 * plx * R_1 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
theta2 = 2 * plx * R_2 * 6.957e8 / 3.085677581e19 * 180 * 3600 * 1000 / np.pi
theta_cov = covariance_matrix([theta1, theta2])[0][1]
theta_cor = correlation_matrix([theta1, theta2])[0][1]

Teff1 = 6223
Teff2 = 5135
# Copy starting values to new variables
theta1_ = theta1.n
theta2_ = theta2.n
ebv_ = ebv_prior.n
sigma_ext = 0.008
sigma_l = 0.01
sigma_c = 0.005
params = [Teff1, Teff2, theta1_, theta2_, ebv_, sigma_ext, sigma_l, sigma_c]
params = params + [0] * Nc1
params = params + [0] * Nc2

parname = ['T_eff,1', 'T_eff,2', 'theta_1', 'theta_2', 'E(B-V)', 'sigma_ext', 'sigma_l', 'sigma_c']
parname = parname + ["c_1,{}".format(j + 1) for j in range(Nc1)]
parname = parname + ["c_2,{}".format(j + 1) for j in range(Nc2)]

# INITIAL LOG LIKELIHOOD FUNCTION
lnlike = lnprob(params, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1,
                apply_flux_ratio_priors=apply_nir_priors,
                verbose=True)
print('Initial log-likelihood = {:0.2f}'.format(lnlike))

# NELDER-MEAD OPTIMISE THE SOLUTION
nll = lambda *args: -lnprob(*args)
args = (flux2mag, lratios, theta1, theta2,
        spec1, spec2, ebv_prior, redlaw, Nc1)
soln = minimize(nll, params, args=args, method='Nelder-Mead')

# RE-INITIALISE LOG LIKELIHOOD FUNCTION WITH OPTIMISED SOLUTION
lnlike = lnprob(soln.x, flux2mag, lratios,
                theta1, theta2, spec1, spec2,
                ebv_prior, redlaw, Nc1,
                apply_flux_ratio_priors=apply_nir_priors,
                verbose=True)


# INITIALISE PARAMETER SPACE FOR MCMC
steps = [25, 25,  # T_eff,1, T_eff,2
         0.0005, 0.0007,  # theta_1 ,theta_2
         0.001, 0.001, 0.001, 0.001,  # E(B-V), sigma_ext, sigma_l, sigma_c
         *[0.01] * Nc1, *[0.01] * Nc2]  # c_1,1 ..   c_2,1 ..
nwalkers = 256
ndim = len(soln.x)
pos = np.zeros([nwalkers, ndim])
for i, x in enumerate(soln.x):
    pos[:, i] = x + steps[i] * np.random.randn(nwalkers)
nsteps = 5000


# RUN MCMC
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args, pool=pool)
    sampler.run_mcmc(pos, nsteps, progress=True)

# af = sampler.acceptance_fraction
# print('\nMedian acceptance fraction =',np.median(af))
# best_index = np.unravel_index(np.argmax(sampler.lnprobability),
#                               (nwalkers, nsteps))
# best_lnlike = np.max(sampler.lnprobability)
# print('\n Best log(likelihood) = ',best_lnlike,' in walker ',best_index[0],
#        ' at step ',best_index[1])
# best_pars = sampler.chain[best_index[0], best_index[1], :]
#
# fig, axes = plt.subplots(4, figsize=(10, 7), sharex='all')
# samples = sampler.get_chain()
# i0 = 0
# labels = parname[i0:i0 + 4]
# for i in range(4):
#     ax = axes[i]
#     ax.plot(samples[:, :, i0 + i], "k", alpha=0.3)
#     ax.set_xlim(0, len(samples))
#     ax.set_ylabel(labels[i])
#     ax.yaxis.set_label_coords(-0.1, 0.5)
#
# axes[-1].set_xlabel("Step number")
# fig.show()
#
# flat_samples = sampler.get_chain(discard=4000, thin=8, flat=True)  # nsteps//2
# fig = corner.corner(flat_samples, labels=parname)
# fig.show()
#
# for i, pn in enumerate(parname):
#     val = flat_samples[:, i].mean()
#     err = flat_samples[:, i].std()
#     ndp = 1 - min(0, np.floor(log10(err)))
#     fmt = '{{:0.{:0.0f}f}}'.format(ndp)
#     vstr = fmt.format(val)
#     estr = fmt.format(err)
#     # print('{} = {} +/- {}'.format(pn,vstr,estr))
#
# lnlike = lnprob(best_pars, flux2mag, lratios,
#                 theta1, theta2, spec1, spec2,
#                 ebv_prior, redlaw, Nc1, verbose=True)
# print('Final log-likelihood = {:0.2f}'.format(lnlike))

# wave, flux, f_1, f_2, d1, d2 = lnprob(
#     best_pars, flux2mag, lratios,
#     theta1, theta2, spec1, spec2,
#     ebv_prior, redlaw, Nc1, return_flux=True)
# fig1, ax1 = plt.subplots(3, figsize=(10, 7), sharex='all')
# ax1[0].semilogx(wave, 1e12 * f_1, c='c')
# ax1[0].semilogx(wave, 1e12 * f_2, c='orange')
# ax1[0].set_xlim(1000, 300000)
# ax1[0].set_ylabel(r'$f_{\lambda}\:\:[10^{-12}\,{\rm ergs}\,{\rm cm}^{-2}\,{\rm s}^{-1}\,{\rm \AA}^{-1}}]$')
# ax1[1].semilogx(wave, d1, c='b')
# ax1[1].set_ylabel('$\Delta_1$')
# ax1[1].set_ylim(-0.25, 0.25)
# ax1[2].semilogx(wave, d2, c='b')
# ax1[2].set_ylabel('$\Delta_2$')
# ax1[2].set_xlabel(r'Wavelength [$\rm \AA$]')
# ax1[2].set_ylim(-0.25, 0.25)
#
# for i in range(0, len(flat_samples), len(flat_samples) // 64):
#     _, _, _, _, _d1, _d2 = lnprob(
#         flat_samples[i, :], flux2mag, lratios,
#         theta1, theta2, spec1, spec2,
#         ebv_prior, redlaw, Nc1, return_flux=True)
#     ax1[1].semilogx(wave, _d1, c='b', alpha=0.1)
#     ax1[2].semilogx(wave, _d2, c='b', alpha=0.1)
# fig.show()
#
#
# VegaZeroPointErrorPercent = 0.5
# Fig14Data = Table.read('Bohlin2014_Fig14.csv', names=['w', 'err'])
# WDScaleErrorWavelengthAngstrom = Fig14Data['w'] * 10000
# WDScaleErrorPercent = Fig14Data['err']
# TotalSystematicErrorPercent = VegaZeroPointErrorPercent + WDScaleErrorPercent
# plt.semilogx(WDScaleErrorWavelengthAngstrom, TotalSystematicErrorPercent, 'bo')
# Interpolator = interp1d(WDScaleErrorWavelengthAngstrom, TotalSystematicErrorPercent, bounds_error=False,
#                         fill_value=50.0)
# WavelengthGrid = np.linspace(min(WDScaleErrorWavelengthAngstrom), max(WDScaleErrorWavelengthAngstrom), 50001)
# TotSysErrPercentGrid = Interpolator(WavelengthGrid)
# plt.semilogx(WavelengthGrid, TotSysErrPercentGrid)
# plt.xlabel(r'Wavelength [$\AA$]')
# plt.ylabel('Flux scale error [%]')
# plt.show()
#
# TotSysErrPercentGrid = Interpolator(wave)
# T_eff_1 = flat_samples[:, 0].mean()
# rnderr_1 = flat_samples[:, 0].std()
# fint_1 = simps(f_1, wave)
# fint_1p = simps(f_1 * (1 + TotSysErrPercentGrid / 100), wave)
# syserr_1 = (fint_1p / fint_1 - 1) * T_eff_1 / 4  # /4 because L \propto Teff^4
# print('Systematic error in integrated flux = {:0.2%}%'.format((fint_1p/fint_1-1)))
# print('T_eff,1 = {:0.0f} +/- {:0.0f} (rnd.) +/- {:0.0f} (sys.) K'.
#     format(T_eff_1, rnderr_1, syserr_1))
#
# T_eff_2 = flat_samples[:, 1].mean()
# rnderr_2 = flat_samples[:, 1].std()
# fint_2 = simps(f_2, wave)
# fint_2p = simps(f_2 * (1 + TotSysErrPercentGrid / 100), wave)
# syserr_2 = (fint_2p / fint_2 - 1) * T_eff_2 / 4
# print('Systematic error in integrated flux = {:0.2%}%'.format((fint_2p/fint_2-1)))
# print('T_eff,2 = {:0.0f} +/- {:0.0f} (rnd.) +/- {:0.0f} (sys.) K'.
#          format(T_eff_2, rnderr_2, syserr_2))


# OUTPUT DATA
pfile = "{}_{:0.0f}_{:0.0f}+{:+0.1f}_{}_coeffs.p".format(run, Tref1, Tref2, M_H, Nc1)
with open(pfile, 'wb') as f:
    pickle.dump(sampler, f)
