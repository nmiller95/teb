# Makes a .yaml file containing star data 
import os
from flux2mag import Flux2mag

def make_file(star_name, overwrite):
    """
    Makes a data file for target star_name

    Parameters
    ----------
    star_name: str
        Name with which to look up data for target. Also used for
        yaml file where these data are stored.

    overwrite: bool
        Over-write existing file if True

    star_name must be resolvable by SIMBAD or in the form Jhhhmmmss.s+ddmmss.s
    Spaces in the star name will be replaced with "_", e.g. "AI_Phe"

    """
    # First collect all the photometry and parallax values needed.
    flux2mag = Flux2mag(star_name)
    star_name_ = star_name.replace(' ','_')
    print(f'Collecting photometric data and parallax for {star_name_}\n')
    data_file_path = f'config/{star_name_}.yaml'
    if not overwrite and os.path.exists(data_file_path):
        raise FileExistsError(f'{config_file_path} - use -o to over-write file')
    c = open(data_file_path, 'w') 
    c.write('## Read the guidance at the bottom of this file before use.\n')
    c.write('#\n')
    c.write('\n')
    c.write('## Model parameters for reference SED\n')
    c.write('teff1:                        # T_eff,1 [K]\n')
    c.write('teff2:                        # T_eff,2 [K]\n')
    c.write('logg1:                        # log(g) for star 1 [cgs]\n')
    c.write('logg2:                        # log(g) for star 2 [cgs]\n')
    c.write('m_h:                          # [M/H] for both stars.\n')
    c.write('aFe:                          # [alpha/Fe] for both stars.\n')
    c.write('\n')
    c.write('# Parallax \n')
    c.write('parallax:     # parallax and error, in mas\n')
    try:
        c.write(f'  - {flux2mag.parallax.n:0.4f}\n')
        c.write(f'  - {flux2mag.parallax.s:0.4f}\n')
    except:
        c.write('  - \n')
        c.write('  - \n')
    c.write('\n')
    c.write('## Stellar radii - all in nominal solar units.\n')
    c.write('# Use EITHER secondary_radius: OR radius_ratio \n')
    c.write('primary_radius:    # Primary star radius and error\n')
    c.write('  - \n')
    c.write('  - \n')
    c.write('radius_ratio:     # Secondary radius / primary radius and error\n')
    c.write('  - \n')
    c.write('  - \n')
    c.write('#secondary_radius: # Secondary star radius and error\n')
    c.write('#  - \n')
    c.write('#  - \n')
    c.write('\n')
    c.write('## Reddening\n')
    c.write('# Prior on interstellar E(B-V) - value and error\n')
    c.write('# Comment out or delete this section to not use a prior\n')
    c.write('ebv:\n')
    c.write('  - \n')
    c.write('  - \n')
    c.write('\n')
    c.write('# Measured flux ratios\n')
    c.write('flux_ratios:\n')
    c.write('  - tag: TESS\n')
    c.write('    band: TESS/TESS.Red\n')
    c.write('    value:         # Value and standard error\n')
    c.write('      - \n')
    c.write('      - \n')
    c.write('\n')
    c.write('# Photometry\n')
    c.write('magnitudes:\n')

    # Gaia 
    for b in ['G', 'Gbp', 'Grp']:
        c.write(f'  - tag: {b}\n')
        c.write(f'    band: GAIA/GAIA3.{b}\n')
        c.write( '    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}']
            c.write(f'      - {m.n:0.4f}\n')
            c.write(f'      - {m.s:0.4f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')
        c.write('\n')

    # 2MASS
    for i,b in enumerate(['J','H','Ks']):
        c.write(f'  - tag: {b}\n')
        c.write(f'    band: 2MASS/2MASS.{b}\n')
        c.write('    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}']
            c.write(f'      - {m.n:+0.3f}\n')
            c.write(f'      - {m.s:0.3f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')
    # GALEX
    for b in ['NUV','FUV']:
        c.write('\n')
        c.write(f'  - tag: {b}\n')
        c.write(f'    band: GALEX/GALEX.{b}\n')
        c.write( '    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}']
            c.write(f'      - {m.n:+0.3f}\n')
            c.write(f'      - {m.s:0.3f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')

    # Tycho-2
    for b in ['B','V']:
        c.write('\n')
        c.write(f'  - tag: {b}T\n')
        c.write(f'    band: TYCHO/TYCHO.{b}_MvB\n')
        c.write( '    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}T']
            c.write(f'      - {m.n:+0.3f}\n')
            c.write(f'      - {m.s:0.3f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')

    # SkyMapper
    for b in ['u','v']:
        c.write('\n')
        c.write(f'  - tag: {b}\n')
        c.write(f'    band: SkyMapper/SkyMapper.{b}\n')
        c.write( '    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}']
            c.write(f'      - {m.n:+0.3f}\n')
            c.write(f'      - {m.s:0.3f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')

    # PAN-STARRS
    for b in ['g','r', 'i', 'z', 'y']:
        c.write('\n')
        c.write(f'  - tag: {b}\n')
        c.write(f'    band: PAN-STARRS/PS1.{b}\n')
        c.write( '    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}']
            c.write(f'      - {m.n:+0.3f}\n')
            c.write(f'      - {m.s:0.3f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')

    # ALLWISE
    for b in ['W1','W2','W3']:
        c.write('\n')
        c.write(f'  - tag: {b}\n')
        c.write(f'    band: WISE/WISE.{b}\n')
        c.write( '    mag:           # Value and standard error \n')
        try:
            m = flux2mag.obs_mag[f'{b}']
            c.write(f'      - {m.n:+0.3f}\n')
            c.write(f'      - {m.s:0.3f}\n')
        except:
            c.write('      - \n')
            c.write('      - \n')
    c.write('\n')
    c.write('\n')
    c.write('# Photometric colors\n')
    c.write('#colors:\n')
    c.write('#  - tag: b-y\n')
    c.write('#    type: by\n')
    c.write('#    color:         # Value and standard error\n')
    c.write('#      - \n')
    c.write('#      - \n')
    c.write('\n')
    c.write('### HOW TO USE THIS CONFIGURATION FILE\n')
    c.write('#\n')
    c.write('## Parallax\n')
    c.write('#  With zero-point correction for Gaia, if needed.\n')
    c.write('#\n')
    c.write('## A note on the stellar radii\n')
    c.write('# If you have very precise measurements for radii (0.2% or better),\n')
    c.write('# these should be the Rosseland radii. See Miller & Maxted, \n')
    c.write('# 2020MNRAS.497.2899M \n')
    c.write('#\n')
    c.write('## Flux ratios\n')
    c.write('# tag (str): Unique name for measurement, can be same as band\n')
    c.write('# band (str): Bandpass name from SVO filter profile service.\n')
    c.write('# value (float): Flux ratio value. Must be greater than 0.\n')
    c.write('# error (float): Error in flux ratio.\n')
    c.write('\n')
    c.write('## Magnitudes\n')
    c.write('# tag (str): Unique name for measurement, can be same as band\n')
    c.write('# band (str): Bandpass name from SVO filter profile service.\n')
    c.write('# mag (float): Value and error for the magnitude\n')
    c.write('#\n')
    c.write('# Data on the zero-point, etc. for each band must be listed in\n')
    c.write('#  config/database.csv. See calspec.py for instructions on generating\n')
    c.write('#  these data.\n')
    c.write('#\n')
    c.write('## colors\n')
    c.write('# tag (str): Unique name for measurement, can be same as color ID\n')
    c.write('# type (str): Color name. Only Strömgren b-y (by), m1 (m1), c1 (c1)\n')
    c.write('# color (float): color value and error.\n')
    c.close()
    print('\n')
    
