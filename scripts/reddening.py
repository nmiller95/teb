# Estimate E(B-V) using equivalent width of interstellar NaI lines
# See Maxted 2025RNAAS...9..146M
#
# Equivalent width of Na I D1 (589.592 nm) in Angstrom, or None to skip
EW_NaI_D1 = 0.0074
EW_NaI_D1_err = 0.0006

# Equivalent width of Na I D2 (588.995 nm) in Angstrom, or None to skip
EW_NaI_D2 = None
EW_NaI_D2_err = None

#-----------------------------------------------------------------------

import numpy as np
from math import factorial
def eq2(ebv, alpha=0.354, beta=11.0):
    w = np.zeros_like(ebv)
    for n in range(1,20):
        w += (-1)**(n-1)*(beta*ebv)**n/factorial(n)/np.sqrt(n)
    return alpha*w

alpha1 = 1.1413
beta1 = 1.5454
c1 = 0.013
d1 = 2.35

xp = np.linspace(0, 0.2) 

if EW_NaI_D1 is not None:
    yp1 = eq2(xp, alpha=alpha1, beta=beta1)
    ebv_1 = np.interp(EW_NaI_D1, yp1, xp)
    s_1 = c1 + EW_NaI_D1**d1   # Scatter around best-fit relation
    e_ebv_1 = np.hypot(np.interp(EW_NaI_D1+EW_NaI_D1_err,yp1,xp) - ebv_1, s_1)
    print(f'NaI D1: E(B-V) = {ebv_1:0.3f} +/- {e_ebv_1:0.3f}')

c2 = 0.014
d2 = 2.45
alpha2 = 0.8306
beta2 = 2.7397
if EW_NaI_D2 is not None:
    yp2 = eq2(xp, alpha=alpha2, beta=beta2)
    ebv_2 = np.interp(EW_NaI_D2, yp2, xp)
    s_2 = c2 + EW_NaI_D2**d2   # Scatter around best-fit relation
    e_ebv_2 = np.hypot(np.interp(EW_NaI_D2+EW_NaI_D2_err,yp2,xp) - ebv_2, s_2)
    print(f'NaI D2: E(B-V) = {ebv_2:0.3f} +/- {e_ebv_2:0.3f}')

if (EW_NaI_D1 is not None) and (EW_NaI_D2 is not None):
    ebv = (ebv_1+ebv_2)/2
    e_ebv = (e_ebv_1+e_ebv_2)/2  # because these estimates are not independent
    print(f'Average: E(B-V) = {ebv:0.3f} +/- {e_ebv:0.3f}')

