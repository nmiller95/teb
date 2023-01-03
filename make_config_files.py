# Makes empty versions of the configuration yaml files

def make_config(filename):
    """
    Makes a configuration file for main methods of teb

    Parameters
    ----------
    filename: str
        Name with which to create new file

    Returns
    -------
    New empty (ish) yaml configuration file

    """
    with open(f'config/{filename}.yaml', 'w') as c:
        c.write('### Configure the main teb code here.\n')
        c.write('## Read the guidance at the bottom of this file before use.\n')
        c.write('\n')
        c.write('## Target and file names\n')
        c.write('name:                         # Target name; must be resolvable by SIMBAD\n')
        c.write(f'run_id: {filename}            # Prefix used in output save file\n')
        c.write('\n')
        c.write('## Physical properties of the binary stars\n')
        c.write('teff1:                        # Best estimate of T_eff,1, in Kelvin\n')
        c.write('teff2:                        # Best estimate of T_eff,2, in Kelvin\n')
        c.write('logg1:                        # Surface gravity for primary star, in cgs.\n')
        c.write('logg2:                        # Surface gravity for secondary star, in cgs.\n')
        c.write('m_h:                          # Metallicity for both stars.\n')
        c.write('aFe:                          # Alpha fraction for both stars.\n')
        c.write('r1:                           # Primary star radius and error, in solar units\n')
        c.write('  -\n')
        c.write('  -\n')
        c.write('r2:                           # Secondary star radius and error, in solar units\n')
        c.write('  -\n')
        c.write('  -\n')
        c.write('k:                            # Ratio of the stellar radii and error, from light curve\n')
        c.write('  -\n')
        c.write('  -\n')
        c.write('plx:                          # Gaia EDR3 parallax and error, in mas\n')
        c.write('  -\n')
        c.write('  -\n')
        c.write('ebv:                          # Prior on interstellar E(B-V) and error\n')
        c.write('  -\n')
        c.write('  -\n')
        c.write('\n')
        c.write('## Configuration parameters for the teb method\n')
        c.write('# Model SED source and distortion\n')
        c.write('model_sed: bt-settl           # Model SEDs to use [bt-settl, bt-settl-cifist]\n')
        c.write('binning: 50                   # Wavelength resolution of model SED, in Angstrom\n')
        c.write('distortion: 2                 # Type of distortion to use:\n')
        c.write('                              # 0 = no distortion applied (simple SED fit)\n')
        c.write('                              # 1 = single distortion function for both (not recommended)\n')
        c.write('                              # 2 = each star\'s SED distorted separately (recommended)\n')
        c.write('n_coeffs: 3                   # Number of distortion coefficients to use.\n')
        c.write('# Priors and data to include\n')
        c.write('apply_ebv_prior: True         # Put a prior on E(B-V)? Uses provided value and error.\n')
        c.write('apply_fratio_prior: True      # Calculate and apply priors on flux ratios in the near-IR?\n')
        c.write('apply_k_prior: False           # Put a prior on the radius ratio (total eclipses only)\n')
        c.write('apply_colors: False           # Include any colors data in photometry_data.yaml?\n')
        c.write('sigma_ext: 0.008              # Prior on external noise on magnitude data\n')
        c.write('sigma_l: 0.01                 # Prior on external noise on flux ratios\n')
        c.write('sigma_c: 0.005                # Prior on external noise on colors\n')
        c.write('# MCMC and display options\n')
        c.write('mcmc_n_steps: 500             # Number of steps to use in MCMC simulations\n')
        c.write('mcmc_n_walkers: 256           # Number of walkers to use in MCMC simulations\n')
        c.write('show_plots: True              # Generate convergence, corner and final SED plots?\n')
        c.write('save_plots: True              # Whether to save these plots in output directory\n')
        c.write('corner_exclude_coeffs: False  # Whether to exclude distortion coeffs from being displayed\n')
        c.write('                              # in corner plot (recommended for large \'n_coeffs\')\n')
        c.write('\n')
        c.write('\n')
        c.write('### HOW TO USE THIS CONFIGURATION FILE\n')
        c.write('\n')
        c.write('## A note on the stellar radii\n')
        c.write('# If you have very precise measurements for radii (0.2% or better), these should technically be the '
                'Rosseland radii\n')
        c.write('\n')
        c.write('## A note on the distortion type\n')
        c.write('# Using a single set of distortion coefficients will reduce the Bayesian Information Criterion '
                'on the fit BUT\n')
        c.write('# will fix the flux ratio at a constant value for all wavelengths. For a better approach use 2.\n')
        c.write('\n')
        c.write('## A note on the model SEDs\n')
        c.write('# BT-Settl and BT-Settl-CIFIST models are supported by teb\n')
        c.write('# teb can integrate between T_eff, log(g) and [M/H] if given value != existing model SED\n')
        c.write('# N.B. not all T_eff-log(g) combinations are supported by all model catalogs.\n')
        c.write('# * BT-Settl: Asplund et al. (2009) solar abundances. M/H = [0.5, 0.3, 0.0, -0.5, -1.0, ..., -4.0]\n')
        c.write('# http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl\n')
        c.write('# * BT-Settl-CIFIST: Caffau et al. (2011) solar abundances, only [M/H] = 0. teb will overwrite [M/H] '
                'if != 0.\n')
        c.write('# http://svo2.cab.inta-csic.es/theory/newov2/index.php?models=bt-settl-cifist\n')


def make_photometry_data(filename):
    """
    Makes a configuration file for photometry data input by user

    Parameters
    ----------
    filename: str
        Name with which to create new file

    Returns
    -------
    New empty (ish) yaml photometry data configuration file

    """
    with open(f'config/{filename}_photometry_data.yaml', 'w') as c:
        c.write('### Include your custom photometric data here.\n')
        c.write('## Binary flux ratios, extra magnitudes not from catalogs, and photometric colors supported.\n')
        c.write('## Read the guidance at the bottom of this file before use.\n')
        c.write('\n')
        c.write('# Flux ratios from light curve fits\n')
        c.write('flux_ratios:\n')
        c.write('  - tag: TESS\n')
        c.write('    type: TESS\n')
        c.write('    value:\n')
        c.write('      - \n')
        c.write('      - \n')
        c.write('\n')
        c.write('# Additional magnitudes (see guidance below).\n')
        c.write('# extra_data:\n')
        c.write('#   - tag: u\n')
        c.write('#     type: skymapper_u\n')
        c.write('#     mag: \n')
        c.write('#       - \n')
        c.write('#       - \n')
        c.write('#     zp:\n')
        c.write('#       - \n')
        c.write('#       - \n')
        c.write('#     file: Response/SkyMapper_SkyMapper.u.dat\n')
        c.write('\n')
        c.write('# Photometric colors\n')
        c.write('# colors_data:\n')
        c.write('#   - tag: BP-RP\n')
        c.write('#     type: BPRP\n')
        c.write('#     color:\n')
        c.write('#       - \n')
        c.write('#       - \n')
        c.write('\n')
        c.write('\n')
        c.write('### HOW TO USE THIS CONFIGURATION FILE\n')
        c.write('\n')
        c.write('## Flux ratios\n')
        c.write('# tag (str): Unique name for measurement, can be same as band\n')
        c.write('# type (str): Bandpass name. Supported bands are:\n')
        c.write('# * GALEX bands: FUV, NUV\n')
        c.write('# * Stromgren bands: u_stromgren, v_stromgren, b_stromgren, y_stromgren\n')
        c.write('# * Gaia EDR3: G, BP, RP\n')
        c.write('# * 2MASS: J, H, Ks\n')
        c.write('# * Skymapper: u_skymapper, v_skymapper g_skymapper r_skymapper i_skymapper z_skymapper\n')
        c.write('# * TESS: TESS\n')
        c.write('# * Johnson/Cousins: U, B, V, R, I\n')
        c.write('# value (float): Flux ratio value. Must be greater than 0.\n')
        c.write('# error (float): Error in flux ratio.\n')
        c.write('\n')
        c.write('## Extra data\n')
        c.write('# Magnitudes not read automatically from common catalogs; measured in the AB magnitude scale.\n')
        c.write('# There is no need to input Galex, Gaia, 2MASS and WISE magnitudes.\n')
        c.write('# tag (str): Unique name for measurement, can be same as band\n')
        c.write('# type (str): Bandpass name.\n')
        c.write('# mag (float): Value and error for the magnitude\n')
        c.write('# zp (float): Zero point of magnitude measurement and its error\n')
        c.write('# file (str): Must be ascii. 2 columns: wave (in angstrom) & response (normalised).\n')
        c.write('\n')
        c.write('## Colors\n')
        c.write('# tag (str): Unique name for measurement, can be same as color ID\n')
        c.write('# type (str): Color name. Only Strömgren b-y (by), m1 (m1), c1 (c1), and Gaia BP-RP (BPRP)\n')
        c.write('# color (float): color value and error.\n')


def make_flux_ratio_priors(filename):
    """
    Makes a configuration file for NIR flux ratio prior calculations

    Parameters
    ----------
    filename: str
        Name with which to create new file

    Returns
    -------
    New empty (ish) yaml NIR flux ratio prior configuration file

    """
    with open(f'config/{filename}_flux_ratio_priors.yaml', 'w') as c:
        c.write('### Adjust your parameters for the flux ratio prior calculations here.\n')
        c.write('## Read the guidance at the bottom of this file before use.\n')
        c.write('\n')
        c.write('## Initial settings\n')
        c.write('method: quad         # Type of fit to the V-K vs. Teff diagram\n')
        c.write('flux_ratio:          # V-band flux ratio of binary, i.e. flux2/flux1\n')
        c.write('teff1:               # Nominal effective temperature of primary (K)\n')
        c.write('teff2:               # Nominal effective temperature of secondary (K)\n')
        c.write('\n')
        c.write('## Ranges to restrict subset of reference stars by\n')
        c.write('E(B-V):              # Interstellar reddening range\n')
        c.write('  - \n')
        c.write('  - \n')
        c.write('logg1:               # Primary star surface gravity range (cgs)\n')
        c.write('  - \n')
        c.write('  - \n')
        c.write('logg2:               # Secondary star surface gravity range (cgs)\n')
        c.write('  - \n')
        c.write('  - \n')
        c.write('tref1:               # Primary star effective temperature range (K)\n')
        c.write('  - \n')
        c.write('  - \n')
        c.write('tref2:               # Secondary star effective temperature range (K)\n')
        c.write('  - \n')
        c.write('  - \n')
        c.write('\n')
        c.write('### HOW TO USE THIS CONFIGURATION FILE\n')
        c.write('# teff1, teff2 are the nominal temperatures used to calculate the initial flux ratio priors.\n')
        c.write('# flux_ratio is a starting value that will quickly become irrelevant as teb runs.\n')
        c.write('# It\'s only really important in [deprecated method] not recommended for normal use of teb.\n')
        c.write('\n')
        c.write('# If unsure about whether to use a linear (lin) or quadratic (quad) fit, check by plotting\n')
        c.write('# V-K colour against catalog temperature for your subset of stars in the GCS III and WISE catalog.\n')
        c.write('\n')
        c.write('# Specify the range of temperatures, surface gravities and interstellar reddening to restrict '
                'the sample by.\n')
        c.write('# The size of search range should reflect the uncertainty on the parameter plus some extra buffer,\n')
        c.write('# e.g. at least twice the error bar.\n')
