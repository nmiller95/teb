import os
import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from uncertainties import ufloat
from calspec import getprofile
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from collections import OrderedDict

class Flux_ratio_priors():
    """
    Calculate flux ratio prior using tabulated color-Teff relations.

    __init__: Load data from selected tables and create interpolators
    __call__: Return flux ratio prior for selected band

    """

    def __init__(self, bands, teff1=None, teff2=None, 
                 color_table1='auto', color_table2='auto'):
        """
        Reads data from tabulated color-Teff and creates an interpolator

        Parameters
        ----------

        bands: Selection from :
            ['FUV', 'NUV', 'BP', 'J', 'H', 'Ks', 'W1', 'W2', 'W3']
        teff1 - estimate of Teff for star 1 - only used to select table
        teff2 - estimate of Teff for star 1 - only used to select table
        color_table1 - 'mdwarf', 'galah' or 'auto'
        color_table2 - 'mdwarf', 'galah' or 'auto'

        """
        if color_table1 == 'auto':
            if teff1 > 4000:
                self.color_table1 = 'galah'
            else:
                self.color_table1 = 'mdwarf'
        else:
            self.color_table1 = color_table1

        if color_table2 == 'auto':
            if teff2 > 4000:
                self.color_table2 = 'galah'
            else:
                self.color_table2 = 'mdwarf'
        else:
            self.color_table2 = color_table2

        table_path1 = os.path.join('Tables',f'{self.color_table1}_colors.fits')
        table1 = Table.read(table_path1)
        table_path2 = os.path.join('Tables',f'{self.color_table2}_colors.fits')
        table2 = Table.read(table_path2)

        colors1 = [s.replace('s_','') for s in  table1.colnames if 's_' in s]
        bands1 = [c.strip('RP').strip('-') for c in colors1]
        colors2 = [s.replace('s_','') for s in  table2.colnames if 's_' in s]
        bands2 = [c.strip('RP').strip('-') for c in colors2]

        # Check for duplicate entries
        if len(set(bands)) < len(bands):
            raise ValueError('Duplicate band in flux_ratio_priors.')

        self.bands = bands

        nbands = len(bands)
        values = np.empty([len(table1), len(table2), nbands])
        sigmas = np.empty([len(table1), len(table2), nbands])

        b2f  = {}  # Translational from band name to SVO FPS filter name
        b2f['FUV'] = 'GALEX/GALEX.FUV'
        b2f['NUV'] = 'GALEX/GALEX.NUV'
        b2f['BP'] = 'GAIA/GAIA3/Gbp'
        b2f['J'] = '2MASS/2MASS.J'
        b2f['H'] = '2MASS/2MASS.H'
        b2f['Ks'] = '2MASS/2MASS.Ks'
        b2f['W1'] = 'WISE/WISE.W1'
        b2f['W2'] = 'WISE/WISE.W3'
        b2f['W3'] = 'WISE/WISE.W3'
        
        # Reference band is RP = GAIA/GAIA3.Grp
        filtername = 'GAIA/GAIA3.Grp'
        filtertable,photon_ = getprofile(filtername, None)
        wave = filtertable['Wavelength']
        resp = filtertable['Transmission']
        if photon_: 
            resp /= simpson(wave*resp, x=wave)
        else:      
            resp /= simpson(resp, x=wave)
        T = {'RP': interp1d(wave, resp, bounds_error=False, fill_value=0)}
        photon = {'RP': photon_}
        for i,band in enumerate(bands):
            filtername = b2f[band]
            filtertable,photon_ = getprofile(filtername, None)
            wave = filtertable['Wavelength']
            resp = filtertable['Transmission']
            if photon_: 
                resp /= simpson(wave*resp, x=wave)
            else:      
                resp /= simpson(resp, x=wave)
            T[band] = interp1d(wave, resp, bounds_error=False, fill_value=0)
            photon[band] = photon_

            if band in ['FUV', 'NUV', 'BP']:
                col = f'{band}-RP'
                c1 = table1[col].filled(np.nan)
                c2 = table2[col].filled(np.nan)
                C2,C1 = np.meshgrid(c2,c1)
                dm = (C2-C1)
            else:
                col = f'RP-{band}'
                c1 = table1[col].filled(np.nan)
                c2 = table2[col].filled(np.nan)
                C2,C1 = np.meshgrid(c2,c1)
                dm = (C1-C2)
            values[:,:,i] = 10**(-0.4*dm)
            s1 = table1[f's_{col}'].filled(np.nan)
            s2 = table2[f's_{col}'].filled(np.nan)
            S2,S1 = np.meshgrid(s2,s1)
            sm = np.hypot(S1,S2) 
            sigmas[:,:,i] = (10**(-0.4*(dm-sm)) - 10**(-0.4*(dm+sm)))/2
       
        points = (table1['T_eff'], table2['T_eff'])
        self.v_interpolator = RegularGridInterpolator(points, values,
                                                 bounds_error=False)
        self.s_interpolator = RegularGridInterpolator(points, sigmas,
                                                 bounds_error=False)
        self.T = T
        self.photon= photon
                                

    def __call__(self, RP_flux_ratio, teff1, teff2, flux_ratio_band='all'):
        """
        Return flux ratio in band flux_ratio_band and error as ufloat

        May return np.nan if teff1 or teff2 are out of range.

        Set flux_ratio_band='all' to return a dict with all the flux ratios 
        that were initially loaded.

        """
        v = self.v_interpolator([teff1,teff2])
        s = self.s_interpolator([teff1,teff2])
        d = {}
        if flux_ratio_band == 'all':
            for i,band in enumerate(self.bands):
                d[band] = RP_flux_ratio*ufloat(v[0][i],s[0][i])
            return d

        i = self.bands.index(flux_ratio_band)
        return RP_flux_ratio*ufloat(v[0][i], s[0][i])

