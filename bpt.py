import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from functools import reduce
import operator
from matplotlib.ticker import MultipleLocator

plt.ion()

z_A = 1.679 # redshift for source A
z_B = 1.677  # redshift for source B

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

#calculate!
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

# plot 
fig, axes = plt.subplots(2,1,figsize=(4,8), gridspec_kw={"hspace": 0})

def bpt_line(log_NIIalpha, z):
    denom = (log_NIIalpha - 0.02 - 0.1833*z) 
    return 0.61/denom + 1.2 + 0.03*z

asymptote = 0.02 + 0.1833 * z_B

x_left = min(log_NIIalpha_A - log_NIIalpha_A_uncert,
             log_NIIalpha_B - log_NIIalpha_B_uncert) - 0.15
x = np.linspace(x_left, 0.2, 1000)
y_B = bpt_line(x, z_B)
y_A = bpt_line(x, z_A)

axes[0].plot(x, y_A, color='blue', lw=0.8)
axes[0].errorbar(log_NIIalpha_A, log_OIIIbeta_A, xerr=log_NIIalpha_A_uncert, yerr=log_OIIIbeta_A_uncert, fmt='o')
axes[0].set_xlim(x_left, 0.2)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_ylabel("log([OIII]/H$\\beta$)")
axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[0].text(0.05, 0.92, f"Source A,  z = {z_A}", transform=axes[0].transAxes, fontsize=9, va='top')
axes[0].tick_params(axis='x', which='both', labelbottom=False, bottom=False)

axes[1].set_xlim(x_left, 0.2)
axes[1].set_ylim(-1.5, 1.5)
axes[1].plot(x, y_B, color='blue', lw=0.8)
axes[1].errorbar(log_NIIalpha_B, log_OIIIbeta_B, xerr=log_NIIalpha_B_uncert, yerr=log_OIIIbeta_B_uncert, fmt='o')
axes[1].xaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].tick_params(axis='x', which='both', top=True, labeltop=False)
axes[1].set_xlabel("log([NII]/H$\\alpha$)")
axes[1].set_ylabel("log([OIII]/H$\\beta$)")
axes[1].text(0.05, 0.92, f"Source B,  z = {z_B}", transform=axes[1].transAxes, fontsize=9, va='top')
fig.tight_layout()
plt.show()


