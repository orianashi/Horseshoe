import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from functools import reduce
import operator

plt.ion()

# =======
# un-pikl
# =======
with open('./output/OIII/OIII_Hbeta_ratios.pkl', 'rb') as f_OIII:
    OIIIbetas = dill.load(f_OIII)
with open('./output/NII/NII_Halpha_ratios.pkl', 'rb') as f_NII:
    NIIalphas = dill.load(f_NII)


# define function to calculate log values and uncertainty
def log_uncert(ratio, uncert):
    log_ratio = np.log10(ratio)
    log_ratio_uncert = 0.434 * uncert / ratio
    return log_ratio, log_ratio_uncert


log_OIIIbeta_A, log_OIIIbeta_A_uncert = log_uncert(
    OIIIbetas['A']['OIIIbeta'], OIIIbetas['A']['OIIIbeta_uncert'])
log_OIIIbeta_B, log_OIIIbeta_B_uncert = log_uncert(
    OIIIbetas['B']['OIIIbeta'], OIIIbetas['B']['OIIIbeta_uncert'])
log_NIIalpha_A, log_NIIalpha_A_uncert = log_uncert(
    NIIalphas['A']['NIIalpha'], NIIalphas['A']['NIIalpha_uncert'])
log_NIIalpha_B, log_NIIalpha_B_uncert = log_uncert(
    NIIalphas['B']['NIIalpha'], NIIalphas['B']['NIIalpha_uncert'])

print(
    f"log([OIII/Hbeta]) for source A: {log_OIIIbeta_A} +- {log_OIIIbeta_A_uncert}"
)
print(
    f"log([NII/Halpha]) for source A: {log_NIIalpha_A} +- {log_NIIalpha_A_uncert}"
)
print('=' * 20)
print(
    f"log([OIII/Hbeta]) for source B: {log_OIIIbeta_B} +- {log_OIIIbeta_B_uncert}"
)
print(
    f"log([NII/Halpha]) for source B: {log_NIIalpha_B} +- {log_NIIalpha_B_uncert}"
)
