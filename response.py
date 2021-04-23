import pickle
from uncertainties import ufloat
from uncertainties.umath import log10
import numpy as np
from astropy.table import Table
# TODO sort this nonsense out

# Response functions for IUE fluxes
transmission = pickle.load(open("Response/transmission.pickle", "rb"))
wave = pickle.load(open("Response/wave.pickle", "rb"))

u320 = {'tag': 'u320',
        'mag': -2.5 * log10(ufloat(1.846e-24, 6.034e-26)) - 48.6,
        'zp': ufloat(-48.60, round(2.5 * log10(1.040), 3)),  # 4% according to Nichols & Linsky (1996)
        'wave': wave,
        'resp': transmission[0]}
u220n = {'tag': 'u220n',
         'mag': -2.5 * log10(ufloat(9.713e-26, 8.924e-27)) - 48.6,
         'zp': ufloat(-48.60, round(2.5 * log10(1.040), 3)),  # 4% according to Nichols & Linsky (1996)
         'wave': wave,
         'resp': transmission[1]}
u220w = {'tag': 'u220w',
         'mag': -2.5 * log10(ufloat(8.583e-26, 9.401e-27)) - 48.6,
         'zp': ufloat(-48.60, round(2.5 * log10(1.040), 3)),  # 4% according to Nichols & Linsky (1996)
         'wave': wave,
         'resp': transmission[2]}

# Zero point and error is average colour o-c offset and standard deviation

stromgren_w, stromgren_r = {}, {}

for x in ('u', 'v', 'b', 'y'):
    T = Table.read("Response/{}_Bessell2005.csv".format(x))
    stromgren_w[x] = np.array(T['wave'])
    stromgren_r[x] = np.array(T['response'])

stromgren_v = {
    'u': 9.139833801171253e-07,
    'v': 1.2227871972005228e-06,
    'b': 1.0300321185051395e-06,
    'y': 7.412341517648064e-07
}

# Colours from GCS III. Errors from Olsen 1994

Gby = {'tag': '(b-y)_G',
       'type': 'by',
       'color': ufloat(0.431, 0.0037),
       'zp': ufloat(-0.0014, 0.0045),
       'wave': stromgren_w,
       'resp': stromgren_r,
       'vega_zp': stromgren_v}
Gm1 = {'tag': 'm1_G',
       'type': 'm1',
       'color': ufloat(0.209, 0.0041),
       'zp': ufloat(0.0243, 0.0062),
       'wave': stromgren_w,
       'resp': stromgren_r,
       'vega_zp': stromgren_v}
Gc1 = {'tag': 'c1_G',
       'type': 'c1',
       'color': ufloat(0.356, 0.0066),
       'zp': ufloat(-0.0102, 0.0083),
       'wave': stromgren_w,
       'resp': stromgren_r,
       'vega_zp': stromgren_v}

# Colours Reipurth 1978

Rby = {'tag': '(b-y)_R',
       'type': 'by',
       'color': ufloat(0.424, 0.0037),
       'zp': ufloat(-0.0014, 0.0045),
       'wave': stromgren_w,
       'resp': stromgren_r,
       'vega_zp': stromgren_v}
Rm1 = {'tag': 'm1_R',
       'type': 'm1',
       'color': ufloat(0.219, 0.0041),
       'zp': ufloat(0.0243, 0.0062),
       'wave': stromgren_w,
       'resp': stromgren_r,
       'vega_zp': stromgren_v}
Rc1 = {'tag': 'c1_R',
       'type': 'c1',
       'color': ufloat(0.357, 0.0066),
       'zp': ufloat(-0.0102, 0.0083),
       'wave': stromgren_w,
       'resp': stromgren_r,
       'vega_zp': stromgren_v}


extra_data = [u320, u220w, u220n]
colors_data = [Gby, Gm1, Gc1, Rby, Rm1, Rc1]