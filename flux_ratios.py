from astropy.table import Table
from uncertainties import ufloat

response_files = {
    # Gaia and Johnson are in different format. Need to include Johnson as minimum
    'FUV': 'Response/EA-fuv_im.tbl',
    'NUV': 'Response/EA-nuv_im.tbl',
    'u_stromgren': 'Response/u_Bessell2005.csv',
    'v_stromgren': 'Response/v_Bessell2005.csv',
    'b_stromgren': 'Response/b_Bessell2005.csv',
    'y_stromgren': 'Response/y_Bessell2005.csv',
    'J': 'Response/sec6_4a.tbl1.dat',
    'H': 'Response/sec6_4a.tbl2.dat',
    'Ks': 'Response/sec6_4a.tbl3.dat',
    'W1': 'Response/RSR-W1.txt',
    'W2': 'Response/RSR-W2.txt',
    'W3': 'Response/RSR-W3.txt',
    'W4': 'Response/RSR-W4.txt',
    'u_skymapper': 'SkyMapper_SkyMapper.u.dat',
    'v_skymapper': 'SkyMapper_SkyMapper.v.dat',
    'g_skymapper': 'SkyMapper_SkyMapper.g.dat',
    'r_skymapper': 'SkyMapper_SkyMapper.r.dat',
    'i_skymapper': 'SkyMapper_SkyMapper.i.dat',
    'z_skymapper': 'SkyMapper_SkyMapper.z.dat'
}


class FluxRatio:
    def __init__(self, id, band, value, error):
        # This should take the ID, band, value and error of a flux ratio and
        # read in the corresponding wavelength and response arrays. Returns a
        # dictionary of flux ratios as before, with extra tag (band)
        self.id = id
        self.band = band
        self.value = ufloat(value, error)
        if id in response_files.keys:
            response_table = Table.read(response_files[id], format='ascii')
            self.response = response_table['col2']
            if id in ['J', 'H', 'Ks', 'W1', 'W2', 'W3', 'W4']:
                self.wave = response_table['col1'] * 1e4  # Convert microns to Angstrom
            else:
                self.wave = response_table['col1']
        elif id in ['G', 'RP', 'BP']:
            pass
