from astropy.table import Table
from uncertainties import ufloat
from scipy.interpolate import interp1d
from numpy import array

response_files = dict(FUV='Response/EA-fuv_im.tbl', NUV='Response/EA-nuv_im.tbl',
                      u_stromgren='Response/u_Bessell2005.csv', v_stromgren='Response/v_Bessell2005.csv',
                      b_stromgren='Response/b_Bessell2005.csv', y_stromgren='Response/y_Bessell2005.csv',
                      J='Response/sec6_4a.tbl1.dat', H='Response/sec6_4a.tbl2.dat', Ks='Response/sec6_4a.tbl3.dat',
                      W1='Response/RSR-W1.txt', W2='Response/RSR-W2.txt', W3='Response/RSR-W3.txt',
                      W4='Response/RSR-W4.txt', u_skymapper='Response/SkyMapper_SkyMapper.u.dat',
                      v_skymapper='Response/SkyMapper_SkyMapper.v.dat',
                      g_skymapper='Response/SkyMapper_SkyMapper.g.dat',
                      r_skymapper='Response/SkyMapper_SkyMapper.r.dat',
                      i_skymapper='Response/SkyMapper_SkyMapper.i.dat',
                      z_skymapper='Response/SkyMapper_SkyMapper.z.dat', TESS='Response/tess-response-function-v1.0.csv',
                      U='Response/Generic_Johnson.U.dat', B='Response/Generic_Johnson.B.dat',
                      V='Response/Generic_Johnson.V.dat', R='Response/Generic_Johnson.R.dat',
                      I='Response/Generic_Johnson.I.dat')


class FluxRatio:
    def __init__(self, unique_id, band, value, error):
        """
        Initialises input data, reads and processes response functions from file.

        :param unique_id: Unique name for bandpass, can be same as band
        :param band: Band, must match a key in response_files dictionary
        :param value: Flux ratio value, float.
        :param error: Error in flux ratio, float.
        """
        self.id = unique_id
        self.band = band
        if type(value) is float and type(error) is float:
            if value >= 0:
                self.value = ufloat(value, error)
            else:
                raise ValueError('Flux ratio value must be greater than 0')
        else:
            raise TypeError('Flux ratio value and error must have type = float')

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

        elif self.band in ['G', 'RP', 'BP']:
            column_names = ['wave', 'G', 'e_G', 'BP', 'e_BP', 'RP', 'e_RP']
            response_table = Table.read('Response/GaiaDR2_RevisedPassbands.dat', format='ascii',
                                        names=column_names)
            self.wave = array(response_table['wave'] * 10, dtype='f8')  # nm to Angstrom
            i = (response_table[self.band] < 99).nonzero()
            self.response = interp1d(self.wave[i], response_table[self.band][i],
                                     bounds_error=False, fill_value=0)

        else:
            raise ValueError('Specified band not in response files dictionary.')

        if self.band == 'TESS':
            self.photon = True
        else:
            self.photon = False

    def __call__(self):
        """
        Saves flux ratio value, response interpolating function to dictionary

        :return: Unique band ID (str), flux ratio value, response function and detector type (dict)
        """
        flux_ratio_dict = {
            'Value': self.value,
            'R': self.response,
            'photon': self.photon
        }
        return self.id, flux_ratio_dict
