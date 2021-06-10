from astropy.table import Table
from uncertainties import ufloat
from scipy.interpolate import interp1d
from numpy import array


# Dictionary containing path names to response files of all bands supported (Gaia EDR3 treated separately)
response_files = dict(
    # GALEX bands
    FUV='Response/EA-fuv_im.tbl', NUV='Response/EA-nuv_im.tbl',
    # Stromgren bands
    u_stromgren='Response/u_Bessell2005.csv', v_stromgren='Response/v_Bessell2005.csv',
    b_stromgren='Response/b_Bessell2005.csv', y_stromgren='Response/y_Bessell2005.csv',
    # 2MASS J, H, Ks bands
    J='Response/sec6_4a.tbl1.dat', H='Response/sec6_4a.tbl2.dat', Ks='Response/sec6_4a.tbl3.dat',
    # WISE IR bands
    W1='Response/RSR-W1.txt', W2='Response/RSR-W2.txt', W3='Response/RSR-W3.txt', W4='Response/RSR-W4.txt',
    # Skymapper bands
    u_skymapper='Response/SkyMapper_SkyMapper.u.dat', v_skymapper='Response/SkyMapper_SkyMapper.v.dat',
    g_skymapper='Response/SkyMapper_SkyMapper.g.dat', r_skymapper='Response/SkyMapper_SkyMapper.r.dat',
    i_skymapper='Response/SkyMapper_SkyMapper.i.dat', z_skymapper='Response/SkyMapper_SkyMapper.z.dat',
    # TESS band
    TESS='Response/tess-response-function-v1.0.csv',
    # Johnson/Cousins bands
    U='Response/Generic_Johnson.U.dat', B='Response/Generic_Johnson.B.dat', V='Response/Generic_Johnson.V.dat',
    R='Response/Generic_Johnson.R.dat', I='Response/Generic_Johnson.I.dat'
)


class FluxRatio:
    """
    Flux ratio class.

    Prepares user input for use in calculations.
    """
    def __init__(self, unique_id, band, value, error):
        """
        Initialises input data, reads and processes response functions from file.
        
        Parameters
        ----------
        unique_id: str
            Unique name for bandpass, can be same as band
        band: str
            Band name. Supported bands are:
            * GALEX bands:
                * FUV, NUV
            * Stromgren bands:
                * u_stromgren, v_stromgren, b_stromgren, y_stromgren
            * Gaia EDR3:
                * G, BP, RP
            * 2MASS:
                * J, H, Ks
            * Skymapper:
                * u_skymapper, v_skymapper g_skymapper r_skymapper i_skymapper z_skymapper
            * TESS:
                * TESS
            * Johnson/Cousins:
                * U, B, V, R, I
        value: float
            Flux ratio value. Must be greater than 0.
        error: float
            Error in flux ratio.
        """
        self.id = unique_id
        self.band = band

        # Checks value and error inputs are valid
        if type(value) is float and type(error) is float:
            if value >= 0:
                self.value = ufloat(value, error)
            else:
                raise ValueError('Flux ratio value must be greater than 0')
        else:
            raise TypeError('Flux ratio value and error must have type = float')

        # Process response function if band is in response_files dictionary
        if self.band in response_files.keys():
            response_table = Table.read(response_files[self.band], format='ascii',
                                        names=['wave', 'resp'])
            if self.band in ['J', 'H', 'Ks', 'W1', 'W2', 'W3', 'W4']:
                self.wave = array(response_table['wave'] * 1e4, dtype='f8')  # um to Angstrom
            elif self.band == 'TESS':
                response_table = Table.read(response_files[self.band], format='ascii.csv',
                                            names=['wave', 'resp'], data_start=7)
                self.wave = array(response_table['wave'] * 10, dtype='f8')  # nm to Angstrom
            else:
                self.wave = array(response_table['wave'], dtype='f8')
            self.response = interp1d(self.wave, response_table['resp'],
                                     bounds_error=False, fill_value=0)

        # Process response function if band is Gaia EDR3
        elif self.band in ['G', 'RP', 'BP']:
            column_names = ['wave', 'G', 'e_G', 'BP', 'e_BP', 'RP', 'e_RP']
            response_table = Table.read('Response/GaiaDR2_RevisedPassbands.dat', format='ascii',
                                        names=column_names)
            self.wave = array(response_table['wave'] * 10, dtype='f8')  # nm to Angstrom
            i = (response_table[self.band] < 99).nonzero()
            self.response = interp1d(self.wave[i], response_table[self.band][i],
                                     bounds_error=False, fill_value=0)

        else:
            raise ValueError("Specified band not currently supported.")

        if self.band == 'TESS':
            self.photon = True
        else:
            self.photon = False

    def __call__(self):
        """
        Saves flux ratio value, response interpolating function to dictionary

        Returns
        -------
        Unique band ID (str), flux ratio value, response function and detector type (dict)
        """
        flux_ratio_dict = {
            'Value': self.value,
            'R': self.response,
            'photon': self.photon
        }
        return self.id, flux_ratio_dict
