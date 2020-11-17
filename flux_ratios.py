from astropy.table import Table
from uncertainties import ufloat

response_files = {}


class FluxRatio:
    def __init__(self, id, band, value, error):
        # This should take the ID, band, value and error of a flux ratio and
        # read in the corresponding wavelength and response arrays. Returns a
        # dictionary of flux ratios as before, with extra tag (band)
        self.id = id
        self.band = band
        self.value = ufloat(value, error)
