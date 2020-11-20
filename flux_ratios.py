from astropy.table import Table
from uncertainties import ufloat

response_files = {
    # Gaia and Johnson are in different format
    'FUV': 'Response/EA-fuv_im.tbl',
    'NUV': 'Response/EA-nuv_im.tbl',
    'stromgren_u': 'Response/u_Bessell2005.csv',
    'stromgren_v': 'Response/v_Bessell2005.csv',
    'stromgren_b': 'Response/b_Bessell2005.csv',
    'stromgren_y': 'Response/y_Bessell2005.csv',
    'J': 'Response/sec6_4a.tbl1.dat',
    'H': 'Response/sec6_4a.tbl2.dat',
    'Ks': 'Response/sec6_4a.tbl3.dat',
    'W1': 'Response/RSR-W1.txt',
    'W2': 'Response/RSR-W2.txt',
    'W3': 'Response/RSR-W3.txt',
    'W4': 'Response/RSR-W4.txt',
    'skymapper_u': 'SkyMapper_SkyMapper.u.dat',
    'skymapper_v': 'SkyMapper_SkyMapper.v.dat',
    'skymapper_g': 'SkyMapper_SkyMapper.g.dat',
    'skymapper_r': 'SkyMapper_SkyMapper.r.dat',
    'skymapper_i': 'SkyMapper_SkyMapper.i.dat',
    'skymapper_z': 'SkyMapper_SkyMapper.z.dat'
}


class FluxRatio:
    def __init__(self, id, band, value, error):
        # This should take the ID, band, value and error of a flux ratio and
        # read in the corresponding wavelength and response arrays. Returns a
        # dictionary of flux ratios as before, with extra tag (band)
        self.id = id
        self.band = band
        self.value = ufloat(value, error)
