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

z_A = 1.679  # redshift for source A
z_B = 1.677  # redshift for source B

# =======
# un-pikl
# =======
with open('./output/OIII/OIII_Hbeta_ratios.pkl', 'rb') as f_OIII:
    OIIIbetas = dill.load(f_OIII)
with open('./output/NII/NII_Halpha_ratios.pkl', 'rb') as f_NII:
    NIIalphas = dill.load(f_NII)
with open('./output/OII/OII_fluxes.pkl', "rb") as f_OII:
    OIIfluxes = dill.load(f_OII)


# define function to calculate log values and uncertainty
def log_uncert(ratio, uncert):
    log_ratio = np.log10(ratio)
    log_ratio_uncert = 0.434 * uncert / ratio
    return log_ratio, log_ratio_uncert


def ratios(num, denom, num_uncert, denom_uncert):
    ratio = num / denom
    ratio_uncert = np.sqrt((num_uncert / num)**2 +
                           (denom_uncert / denom)**2) * ratio
    return ratio, ratio_uncert


# ====================================
# USING [NII]/HALPHA FROM P04 FOR METALLICITY
# ====================================
print("=" * 20)
print("From pp04: crude metallicity estimate from [Nii]/halpha")
print("=" * 20)


def pp04_N2_metallicity_line(N2):
    return 8.9 + 0.57 * N2


def pp04_N2_metallicity_cubic(N2):
    return 9.37 + 2.03 * N2 + 1.26 * (N2**2) + 0.32 * (N2**3)


# calculate N2
log_NIIalpha_A, log_NIIalpha_A_uncert = log_uncert(
    NIIalphas['A']['NIIalpha'], NIIalphas['A']['NIIalpha_uncert'])
log_NIIalpha_B, log_NIIalpha_B_uncert = log_uncert(
    NIIalphas['B']['NIIalpha'], NIIalphas['B']['NIIalpha_uncert'])

# make the graph
logN2s = np.linspace(-2.5, 0, 500)
pettini_N2_metals = pp04_N2_metallicity_line(logN2s)
pettini_N2_metals_s = pp04_N2_metallicity_cubic(logN2s)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(logN2s, pettini_N2_metals, color='black', ls='--', lw=0.8)
ax.plot(logN2s, pettini_N2_metals_s, color='red', ls='-', lw=0.5)
ax.axvline(log_NIIalpha_A, label='Source A', ls='--', color='pink')
ax.axvline(log_NIIalpha_B, label='Source B', ls='--', color='purple')
ax.axhline(8.66, label="solar", ls='--', color='black', lw=0.5)
ax.legend()
ax.set_xlabel("Log([NII]/Halpha)")
ax.set_ylabel("12 + log(O/H)")
ax.set_title("From Pettini and Pagel 2004")
plt.show()

# fake metallicity estimate
N2_metal_A = pp04_N2_metallicity_line(log_NIIalpha_A)
N2_metal_B = pp04_N2_metallicity_line(log_NIIalpha_B)
print(
    f"src A metallicity from intersection w/ pettini 2004 N2 diagnostic line: {N2_metal_A}"
)
print(
    f"src B metallicity from intersection w/ pettini 2004 N2 diagnostic line: {N2_metal_B}"
)

# ====================================
# COMPUTING O3N2 FROM PETTINI 2004
# ====================================
print("=" * 20)
print("From Pettini 2004: crude metallicity estimate from O3N2")
print("=" * 20)


def pettini_metallicity(o3n2):
    return 8.73 - 0.32 * o3n2


# calcualte O3N2
o3n2A, o3n2A_err = ratios(OIIIbetas['A']['OIIIbeta'],
                          NIIalphas['A']['NIIalpha'],
                          OIIIbetas['A']['OIIIbeta_uncert'],
                          NIIalphas['A']['NIIalpha_uncert'])
o3n2B, o3n2B_err = ratios(OIIIbetas['B']['OIIIbeta'],
                          NIIalphas['B']['NIIalpha'],
                          OIIIbetas['B']['OIIIbeta_uncert'],
                          NIIalphas['B']['NIIalpha_uncert'])
log_o3n2A, log_o3n2A_err = log_uncert(o3n2A, o3n2A_err)
log_o3n2B, log_o3n2B_err = log_uncert(o3n2B, o3n2B_err)
print(
    f"log([OIII]/Hbeta / [NII/Halpha]) for source A: {log_o3n2A} +- {log_o3n2A_err}"
)
print(
    f"log([OIII]/Hbeta / [NII/Halpha]) for source B: {log_o3n2B} +- {log_o3n2B_err}"
)

# make the pettini graph
logo3n2s = np.linspace(-1, 1.9, 500)
logo3n2s_invalid = np.linspace(1.9, 3.5, 200)
pettini_metals = pettini_metallicity(logo3n2s)
pettini_metals_invalid = pettini_metallicity(logo3n2s_invalid)
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(logo3n2s, pettini_metals)
ax.plot(logo3n2s_invalid, pettini_metals_invalid, ls='--', color='red')
ax.axvline(log_o3n2A, label='Source A', ls='--', color='pink')
ax.axvline(log_o3n2B, label='Source B', ls='--', color='purple')
ax.axhline(8.66, label="solar", ls='--', color='black', lw=0.5)
ax.legend()
ax.set_xlabel("Log([O3N2])")
ax.set_ylabel("12 + log(O/H)")
ax.set_title("From Pettini and Pagel 2004")
plt.show()

# fake metallicity estimate
pettini_metal_A = pettini_metallicity(log_o3n2A)
pettini_metal_B = pettini_metallicity(log_o3n2B)
print(
    f"src A metallicity from intersection w/ pettini 2004: {pettini_metal_A}")
print(
    f"src B metallicity from intersection w/ pettini 2004: {pettini_metal_B}")

# ====================================
# COMPUTING R23
# ====================================
print("=" * 20)
print("R23")
print("=" * 20)


def R23(oii3726, oii3729, oiii4959, oiii5007, hbeta, oii3726unc, oii3729unc,
        oiii4959unc, oiii5007unc, hbetaunc):
    num = oii3726 + oii3729 + oiii4959 + oiii5007
    R23 = num / hbeta
    numerr = np.sqrt(oii3726unc**2 + oii3729unc**2 + oiii4959unc**2 +
                     oiii5007unc**2)
    R23err = np.sqrt((numerr / num)**2 + (hbetaunc / hbeta)**2) * R23
    return R23, R23err


def kewley_R23(oii3726, oiii4959, oiii5007, hbeta, oii3726unc, oiii4959unc,
               oiii5007unc, hbetaunc):
    num = oii3726 + oiii4959 + oiii5007
    R23 = num / hbeta
    numerr = np.sqrt(oii3726unc**2 + oiii4959unc**2 + oiii5007unc**2)
    R23err = np.sqrt((numerr / num)**2 + (hbetaunc / hbeta)**2) * R23
    return R23, R23err


# double oii R23
R23_A, R23_A_err = R23(
    OIIfluxes['fluxes']['A'][0], OIIfluxes['fluxes']['A'][1],
    OIIIbetas['fluxes']['A'][1], OIIIbetas['fluxes']['A'][2],
    OIIIbetas['fluxes']['A'][0], OIIfluxes['flux_uncerts']['A'][0],
    OIIfluxes['flux_uncerts']['A'][1], OIIIbetas['flux_uncerts']['A'][1],
    OIIIbetas['flux_uncerts']['A'][2], OIIIbetas['flux_uncerts']['A'][0])
R23_B, R23_B_err = R23(
    OIIfluxes['fluxes']['B'][0], OIIfluxes['fluxes']['B'][1],
    OIIIbetas['fluxes']['B'][1], OIIIbetas['fluxes']['B'][2],
    OIIIbetas['fluxes']['B'][0], OIIfluxes['flux_uncerts']['B'][0],
    OIIfluxes['flux_uncerts']['B'][1], OIIIbetas['flux_uncerts']['B'][1],
    OIIIbetas['flux_uncerts']['B'][2], OIIIbetas['flux_uncerts']['B'][0])
logR23_A, logR23A_err = log_uncert(R23_A, R23_A_err)
logR23_B, logR23B_err = log_uncert(R23_B, R23_B_err)
#print(f"R23 for source A: {R23_A:.5f} +- {R23_A_err:.5f}")
#print(f"R23 for source B: {R23_B:.5f} +- {R23_B_err:.5f}")
print(f"log(R23_A): {logR23_A} +- {logR23A_err}")
print(f"log(R23_B): {logR23_B} +- {logR23B_err}")

#kewley R23
kR23_A, kR23_A_err = kewley_R23(
    OIIfluxes['fluxes']['A'][0], OIIIbetas['fluxes']['A'][1],
    OIIIbetas['fluxes']['A'][2], OIIIbetas['fluxes']['A'][0],
    OIIfluxes['flux_uncerts']['A'][0], OIIIbetas['flux_uncerts']['A'][1],
    OIIIbetas['flux_uncerts']['A'][2], OIIIbetas['flux_uncerts']['A'][0])
kR23_B, kR23_B_err = kewley_R23(
    OIIfluxes['fluxes']['B'][0], OIIIbetas['fluxes']['B'][1],
    OIIIbetas['fluxes']['B'][2], OIIIbetas['fluxes']['B'][0],
    OIIfluxes['flux_uncerts']['B'][0], OIIIbetas['flux_uncerts']['B'][1],
    OIIIbetas['flux_uncerts']['B'][2], OIIIbetas['flux_uncerts']['B'][0])
logkR23_A, logkR23A_err = log_uncert(kR23_A, kR23_A_err)
logkR23_B, logkR23B_err = log_uncert(kR23_B, kR23_B_err)
#print(f"KK04 R23 for source A: {kR23_A:.5f} +- {kR23_A_err:.5f}")
#print(f"KK04 R23 for source B: {kR23_B:.5f} +- {kR23_B_err:.5f}")
print(f"KK04 log(R23_A): {logkR23_A} +- {logkR23A_err}")
print(f"KK04 log(R23_B): {logkR23_B} +- {logkR23B_err}")

# ====================================
# MAKING THE BPT
# ====================================
print("=" * 20)
print("BPT")
print("=" * 20)
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
print(f"log(N2) for source A: {log_NIIalpha_A} +- {log_NIIalpha_A_uncert}")
print(
    f"log([OIII/Hbeta]) for source B: {log_OIIIbeta_B} +- {log_OIIIbeta_B_uncert}"
)
print(f"log(N2) for source B: {log_NIIalpha_B} +- {log_NIIalpha_B_uncert}")

# plot
fig, axes = plt.subplots(2, 1, figsize=(4, 8), gridspec_kw={"hspace": 0})


def bpt_line(log_NIIalpha, z):
    denom = (log_NIIalpha - 0.02 - 0.1833 * z)
    return 0.61 / denom + 1.2 + 0.03 * z


asymptote = 0.02 + 0.1833 * z_B

x_left = min(log_NIIalpha_A - log_NIIalpha_A_uncert,
             log_NIIalpha_B - log_NIIalpha_B_uncert) - 0.15
x = np.linspace(x_left, 0.2, 1000)
y_B = bpt_line(x, z_B)
y_A = bpt_line(x, z_A)

axes[0].plot(x, y_A, color='blue', lw=0.8)
axes[0].errorbar(log_NIIalpha_A,
                 log_OIIIbeta_A,
                 xerr=log_NIIalpha_A_uncert,
                 yerr=log_OIIIbeta_A_uncert,
                 fmt='o')
axes[0].set_xlim(x_left, 0.2)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_ylabel("log([OIII]/H$\\beta$)")
axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[0].text(0.05,
             0.92,
             f"Source A,  z = {z_A}",
             transform=axes[0].transAxes,
             fontsize=9,
             va='top')
axes[0].tick_params(axis='x', which='both', labelbottom=False, bottom=False)

axes[1].set_xlim(x_left, 0.2)
axes[1].set_ylim(-1.5, 1.5)
axes[1].plot(x, y_B, color='blue', lw=0.8)
axes[1].errorbar(log_NIIalpha_B,
                 log_OIIIbeta_B,
                 xerr=log_NIIalpha_B_uncert,
                 yerr=log_OIIIbeta_B_uncert,
                 fmt='o')
axes[1].xaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].tick_params(axis='x', which='both', top=True, labeltop=False)
axes[1].set_xlabel("log([NII]/H$\\alpha$)")
axes[1].set_ylabel("log([OIII]/H$\\beta$)")
axes[1].text(0.05,
             0.92,
             f"Source B,  z = {z_B}",
             transform=axes[1].transAxes,
             fontsize=9,
             va='top')
fig.tight_layout()

fig.savefig('./output/diagnostics/bpt.png')
plt.show()
