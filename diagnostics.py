import os

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.ion()

z_A = 1.679  # redshift for source A
z_B = 1.677  # redshift for source B

FLUXDIR = './output/joint_fit/jointfit_all/fluxes'
DIAGDIR = './output/joint_fit/jointfit_all/diagnostics'
os.makedirs(DIAGDIR, exist_ok=True)

# source A components are [central, red_wing]; source B are [red_wing,
# central, blue_wing] -- except [OII]3726/3729 where A has only 1 (roleless)
# component and B has only 2 (no blue_wing). [NII]6584 has a single
# (roleless) component for BOTH sources -- no wing decomposition at all, see
# the component-wise section below for why it's excluded from that loop.
WING_NAMES = {
    'A': ['central', 'red_wing'],
    'B': ['red_wing', 'central', 'blue_wing'],
}


# =======
# un-pikl: fluxes from the joint fit (see jointfit_all.py +
# multiple_gaussian_integration.py), full covariance-propagated uncertainty
# =======
def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


Halpha = load_pkl(f'{FLUXDIR}/Halpha_fluxes.pkl')
Hbeta = load_pkl(f'{FLUXDIR}/Hbeta_fluxes.pkl')
Hgamma = load_pkl(f'{FLUXDIR}/Hgamma_fluxes.pkl')
OIII4959 = load_pkl(f'{FLUXDIR}/OIII4959_fluxes.pkl')
OIII5007 = load_pkl(f'{FLUXDIR}/OIII5007_fluxes.pkl')
OII3726 = load_pkl(f'{FLUXDIR}/OII3726_fluxes.pkl')
OII3729 = load_pkl(f'{FLUXDIR}/OII3729_fluxes.pkl')
NII6584 = load_pkl(f'{FLUXDIR}/NII6584_fluxes.pkl')


# ====================
# define functions
# ====================
def log_uncert(ratio, uncert):
    log_ratio = np.log10(ratio)
    log_ratio_uncert = 0.434 * uncert / ratio
    return log_ratio, log_ratio_uncert


def ratios(num, denom, num_uncert, denom_uncert):
    ratio = num / denom
    ratio_uncert = np.sqrt((num_uncert / num)**2 +
                           (denom_uncert / denom)**2) * ratio
    return ratio, ratio_uncert


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


def KD02_O32(oii3726, oii3729, oiii5007, oii3726unc, oii3729unc, oiii5007unc):
    denom = oii3726 + oii3729
    denom_unc = np.sqrt(oii3726unc**2 + oii3729unc**2)
    return ratios(oiii5007, denom, oiii5007unc, denom_unc)


def KK04_O32(oii3726, oiii4959, oiii5007, oii3726unc, oiii4959unc, oiii5007unc):
    num = oiii4959 + oiii5007
    num_unc = np.sqrt(oiii4959unc**2 + oiii5007unc**2)
    return ratios(num, oii3726, num_unc, oii3726unc)


def pp04_N2_metallicity_line(N2):
    return 8.9 + 0.57 * N2


def pp04_N2_metallicity_cubic(N2):
    return 9.37 + 2.03 * N2 + 1.26 * (N2**2) + 0.32 * (N2**3)


def pettini_metallicity(o3n2):
    return 8.73 - 0.32 * o3n2


def bpt_line(log_NIIalpha, z):
    denom = (log_NIIalpha - 0.02 - 0.1833 * z)
    return 0.61 / denom + 1.2 + 0.03 * z


# ====================================
# cumulative, per source
# ====================================
cumulative_out = {}
logs = {}

for src in ('A', 'B'):
    N2, N2_err = ratios(NII6584['fluxes'][src], Halpha['fluxes'][src],
                        NII6584['flux_uncerts'][src], Halpha['flux_uncerts'][src])
    OIIIbeta, OIIIbeta_err = ratios(OIII5007['fluxes'][src], Hbeta['fluxes'][src],
                                    OIII5007['flux_uncerts'][src], Hbeta['flux_uncerts'][src])
    O3N2, O3N2_err = ratios(OIIIbeta, N2, OIIIbeta_err, N2_err)
    NII_OII, NII_OII_err = ratios(NII6584['fluxes'][src], OII3726['fluxes'][src],
                                  NII6584['flux_uncerts'][src], OII3726['flux_uncerts'][src])
    Halpha_Hbeta, Halpha_Hbeta_err = ratios(Halpha['fluxes'][src], Hbeta['fluxes'][src],
                                            Halpha['flux_uncerts'][src], Hbeta['flux_uncerts'][src])
    Hgamma_Hbeta, Hgamma_Hbeta_err = ratios(Hgamma['fluxes'][src], Hbeta['fluxes'][src],
                                            Hgamma['flux_uncerts'][src], Hbeta['flux_uncerts'][src])

    R23_val, R23_err = R23(
        OII3726['fluxes'][src], OII3729['fluxes'][src],
        OIII4959['fluxes'][src], OIII5007['fluxes'][src], Hbeta['fluxes'][src],
        OII3726['flux_uncerts'][src], OII3729['flux_uncerts'][src],
        OIII4959['flux_uncerts'][src], OIII5007['flux_uncerts'][src], Hbeta['flux_uncerts'][src])
    kR23_val, kR23_err = kewley_R23(
        OII3726['fluxes'][src], OIII4959['fluxes'][src], OIII5007['fluxes'][src],
        Hbeta['fluxes'][src], OII3726['flux_uncerts'][src], OIII4959['flux_uncerts'][src],
        OIII5007['flux_uncerts'][src], Hbeta['flux_uncerts'][src])
    KD02_O32_val, KD02_O32_err = KD02_O32(
        OII3726['fluxes'][src], OII3729['fluxes'][src], OIII5007['fluxes'][src],
        OII3726['flux_uncerts'][src], OII3729['flux_uncerts'][src], OIII5007['flux_uncerts'][src])
    KK04_O32_val, KK04_O32_err = KK04_O32(
        OII3726['fluxes'][src], OIII4959['fluxes'][src], OIII5007['fluxes'][src],
        OII3726['flux_uncerts'][src], OIII4959['flux_uncerts'][src], OIII5007['flux_uncerts'][src])

    log_N2, log_N2_err = log_uncert(N2, N2_err)
    log_OIIIbeta, log_OIIIbeta_err = log_uncert(OIIIbeta, OIIIbeta_err)
    log_O3N2, log_O3N2_err = log_uncert(O3N2, O3N2_err)
    log_NII_OII, log_NII_OII_err = log_uncert(NII_OII, NII_OII_err)
    log_R23, log_R23_err = log_uncert(R23_val, R23_err)
    log_kR23, log_kR23_err = log_uncert(kR23_val, kR23_err)
    log_KD02_O32_val, log_KD02_O32_err = log_uncert(KD02_O32_val, KD02_O32_err)
    log_KK04_O32_val, log_KK04_O32_err = log_uncert(KK04_O32_val, KK04_O32_err)
    log_Halpha_Hbeta, log_Halpha_Hbeta_err = log_uncert(Halpha_Hbeta, Halpha_Hbeta_err)
    log_Hgamma_Hbeta, log_Hgamma_Hbeta_err = log_uncert(Hgamma_Hbeta, Hgamma_Hbeta_err)

    logs[src] = {
        'log_N2': log_N2,
        'log_N2_err': log_N2_err,
        'log_OIIIbeta': log_OIIIbeta,
        'log_OIIIbeta_err': log_OIIIbeta_err,
        'log_O3N2': log_O3N2,
        'log_O3N2_err': log_O3N2_err,
    }

    cumulative_out[src] = {
        'N2': N2,
        'N2_err': N2_err,
        'log_N2': log_N2,
        'log_N2_err': log_N2_err,
        '[OIII]/Hbeta': OIIIbeta,
        '[OIII]/Hbeta_err': OIIIbeta_err,
        'log_OIIIbeta': log_OIIIbeta,
        'log_OIIIbeta_err': log_OIIIbeta_err,
        'O3N2': O3N2,
        'O3N2_err': O3N2_err,
        'log_O3N2': log_O3N2,
        'log_O3N2_err': log_O3N2_err,
        '[NII]/[OII]': NII_OII,
        '[NII]/[OII]_err': NII_OII_err,
        'log_NII_OII': log_NII_OII,
        'log_NII_OII_err': log_NII_OII_err,
        'R23': R23_val,
        'R23_err': R23_err,
        'log_R23': log_R23,
        'log_R23_err': log_R23_err,
        'kk04_R23': kR23_val,
        'kk04_R23_err': kR23_err,
        'log_kk04_R23': log_kR23,
        'log_kk04_R23_err': log_kR23_err,
        'KD02_O32': KD02_O32_val,
        'KD02_O32_err': KD02_O32_err,
        'log_KD02_O32': log_KD02_O32_val,
        'log_KD02_O32_err': log_KD02_O32_err,
        'KK04_O32': KK04_O32_val,
        'KK04_O32_err': KK04_O32_err,
        'log_KK04_O32': log_KK04_O32_val,
        'log_KK04_O32_err': log_KK04_O32_err,
        'Halpha/Hbeta': Halpha_Hbeta,
        'Halpha/Hbeta_err': Halpha_Hbeta_err,
        'log_Halpha_Hbeta': log_Halpha_Hbeta,
        'log_Halpha_Hbeta_err': log_Halpha_Hbeta_err,
        'Hgamma/Hbeta': Hgamma_Hbeta,
        'Hgamma/Hbeta_err': Hgamma_Hbeta_err,
        'log_Hgamma_Hbeta': log_Hgamma_Hbeta,
        'log_Hgamma_Hbeta_err': log_Hgamma_Hbeta_err,
    }

cumulative_A = cumulative_out['A']
cumulative_B = cumulative_out['B']

# ====================================
# component-wise, per source per wing -- [OIII]/Hbeta, R23, kk04_R23,
# KD02_O32, KK04_O32, Halpha/Hbeta, Hgamma/Hbeta only. [NII]6584 is fit as a
# single, non-decomposed Gaussian per source (see jointfit_all.py) -- it has
# no central/red/blue wing components the way the other lines do. Its
# component_fluxes arrays technically have a valid index-0 entry, but pairing
# that single total flux against another line's *specific* wing sub-component
# would be physically meaningless (it would double/mis-count NII's whole
# flux as if it were just "the red wing" or "the central" piece of the
# line). So N2, O3N2, and [NII]/[OII] -- anything needing NII -- stay
# cumulative-only.
# ====================================
component_out = {'A': {}, 'B': {}}
for src, wings in WING_NAMES.items():
    for i, wing in enumerate(wings):
        # not every line has a component at this wing index for every source
        # (see [OII] truncation) -- skip a wing once any required line runs out
        try:
            oii3726_f = OII3726['component_fluxes'][src][i]
            oii3729_f = OII3729['component_fluxes'][src][i]
            oiii4959_f = OIII4959['component_fluxes'][src][i]
            oiii5007_f = OIII5007['component_fluxes'][src][i]
            hbeta_f = Hbeta['component_fluxes'][src][i]
            halpha_f = Halpha['component_fluxes'][src][i]
            hgamma_f = Hgamma['component_fluxes'][src][i]
            oii3726_u = OII3726['component_flux_uncerts'][src][i]
            oii3729_u = OII3729['component_flux_uncerts'][src][i]
            oiii4959_u = OIII4959['component_flux_uncerts'][src][i]
            oiii5007_u = OIII5007['component_flux_uncerts'][src][i]
            hbeta_u = Hbeta['component_flux_uncerts'][src][i]
            halpha_u = Halpha['component_flux_uncerts'][src][i]
            hgamma_u = Hgamma['component_flux_uncerts'][src][i]
        except IndexError:
            continue

        OIIIbeta_f, OIIIbeta_u = ratios(oiii5007_f, hbeta_f, oiii5007_u, hbeta_u)
        Halpha_Hbeta_f, Halpha_Hbeta_u = ratios(halpha_f, hbeta_f, halpha_u, hbeta_u)
        Hgamma_Hbeta_f, Hgamma_Hbeta_u = ratios(hgamma_f, hbeta_f, hgamma_u, hbeta_u)
        R23_val, R23_err = R23(oii3726_f, oii3729_f, oiii4959_f, oiii5007_f, hbeta_f,
                               oii3726_u, oii3729_u, oiii4959_u, oiii5007_u, hbeta_u)
        kR23_val, kR23_err = kewley_R23(oii3726_f, oiii4959_f, oiii5007_f, hbeta_f,
                                        oii3726_u, oiii4959_u, oiii5007_u, hbeta_u)
        KD02_O32_val, KD02_O32_err = KD02_O32(oii3726_f, oii3729_f, oiii5007_f,
                                              oii3726_u, oii3729_u, oiii5007_u)
        KK04_O32_val, KK04_O32_err = KK04_O32(oii3726_f, oiii4959_f, oiii5007_f,
                                              oii3726_u, oiii4959_u, oiii5007_u)

        log_OIIIbeta, log_OIIIbeta_err = log_uncert(OIIIbeta_f, OIIIbeta_u)
        log_Halpha_Hbeta, log_Halpha_Hbeta_err = log_uncert(Halpha_Hbeta_f, Halpha_Hbeta_u)
        log_Hgamma_Hbeta, log_Hgamma_Hbeta_err = log_uncert(Hgamma_Hbeta_f, Hgamma_Hbeta_u)
        log_R23, log_R23_err = log_uncert(R23_val, R23_err)
        log_kR23, log_kR23_err = log_uncert(kR23_val, kR23_err)
        log_KD02_O32_val, log_KD02_O32_err = log_uncert(KD02_O32_val, KD02_O32_err)
        log_KK04_O32_val, log_KK04_O32_err = log_uncert(KK04_O32_val, KK04_O32_err)

        component_out[src][wing] = {
            '[OIII]/Hbeta': OIIIbeta_f,
            '[OIII]/Hbeta_err': OIIIbeta_u,
            'log_OIIIbeta': log_OIIIbeta,
            'log_OIIIbeta_err': log_OIIIbeta_err,
            'R23': R23_val,
            'R23_err': R23_err,
            'log_R23': log_R23,
            'log_R23_err': log_R23_err,
            'kk04_R23': kR23_val,
            'kk04_R23_err': kR23_err,
            'log_kk04_R23': log_kR23,
            'log_kk04_R23_err': log_kR23_err,
            'KD02_O32': KD02_O32_val,
            'KD02_O32_err': KD02_O32_err,
            'log_KD02_O32': log_KD02_O32_val,
            'log_KD02_O32_err': log_KD02_O32_err,
            'KK04_O32': KK04_O32_val,
            'KK04_O32_err': KK04_O32_err,
            'log_KK04_O32': log_KK04_O32_val,
            'log_KK04_O32_err': log_KK04_O32_err,
            'Halpha/Hbeta': Halpha_Hbeta_f,
            'Halpha/Hbeta_err': Halpha_Hbeta_u,
            'log_Halpha_Hbeta': log_Halpha_Hbeta,
            'log_Halpha_Hbeta_err': log_Halpha_Hbeta_err,
            'Hgamma/Hbeta': Hgamma_Hbeta_f,
            'Hgamma/Hbeta_err': Hgamma_Hbeta_u,
            'log_Hgamma_Hbeta': log_Hgamma_Hbeta,
            'log_Hgamma_Hbeta_err': log_Hgamma_Hbeta_err,
        }

# ====================================
# print all
# ====================================
print("=" * 60)
print("Cumulative diagnostics")
print("=" * 60)
for src in ('A', 'B'):
    print(f"Source {src}:")
    for key, val in cumulative_out[src].items():
        print(f"  {key}: {val:.4f}")

print()
print("=" * 60)
print("Component-wise diagnostics -- [OIII]/Hbeta / R23 / kk04_R23 / KD02_O32 / KK04_O32 / Halpha/Hbeta / Hgamma/Hbeta only")
print("(N2, O3N2, [NII]/[OII] need [NII]6584, which has no per-wing decomposition)")
print("=" * 60)
for src, wings in component_out.items():
    for wing, vals in wings.items():
        print(f"Source {src} {wing}:")
        for key, val in vals.items():
            print(f"  {key}: {val:.4f}")

# ====================================
# N2 metallicity plot (Pettini & Pagel 2004)
# ====================================
logN2s = np.linspace(-2.5, 0, 500)
pettini_N2_metals = pp04_N2_metallicity_line(logN2s)
pettini_N2_metals_s = pp04_N2_metallicity_cubic(logN2s)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(logN2s, pettini_N2_metals, color='black', ls='--', lw=0.8)
ax.plot(logN2s, pettini_N2_metals_s, color='red', ls='-', lw=0.5)
ax.axvline(logs['A']['log_N2'], label='Source A', ls='--', color='pink')
ax.axvline(logs['B']['log_N2'], label='Source B', ls='--', color='purple')
ax.axhline(8.66, label="solar", ls='--', color='black', lw=0.5)
ax.legend()
ax.set_xlabel("Log([NII]/Halpha)")
ax.set_ylabel("12 + log(O/H)")
ax.set_title("From Pettini and Pagel 2004 (joint fit fluxes)")
plt.show()

# ====================================
# O3N2 / Pettini 2004 plot
# ====================================
logo3n2s = np.linspace(-1, 1.9, 500)
logo3n2s_invalid = np.linspace(1.9, 3.5, 200)
pettini_metals = pettini_metallicity(logo3n2s)
pettini_metals_invalid = pettini_metallicity(logo3n2s_invalid)
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(logo3n2s, pettini_metals)
ax.plot(logo3n2s_invalid, pettini_metals_invalid, ls='--', color='red')
ax.axvline(logs['A']['log_O3N2'], label='Source A', ls='--', color='pink')
ax.axvline(logs['B']['log_O3N2'], label='Source B', ls='--', color='purple')
ax.axhline(8.66, label="solar", ls='--', color='black', lw=0.5)
ax.legend()
ax.set_xlabel("Log([O3N2])")
ax.set_ylabel("12 + log(O/H)")
ax.set_title("From Pettini and Pagel 2004 (joint fit fluxes)")
plt.show()

# ====================================
# BPT diagram (cumulative only -- N2 has no component-wise version)
# ====================================
log_N2_A = logs['A']['log_N2']
log_N2_A_err = logs['A']['log_N2_err']
log_N2_B = logs['B']['log_N2']
log_N2_B_err = logs['B']['log_N2_err']
log_OIIIbeta_A = logs['A']['log_OIIIbeta']
log_OIIIbeta_A_err = logs['A']['log_OIIIbeta_err']
log_OIIIbeta_B = logs['B']['log_OIIIbeta']
log_OIIIbeta_B_err = logs['B']['log_OIIIbeta_err']

fig, axes = plt.subplots(2, 1, figsize=(4, 8), gridspec_kw={"hspace": 0})

x_left = min(log_N2_A - log_N2_A_err, log_N2_B - log_N2_B_err) - 0.15
x = np.linspace(x_left, 0.2, 1000)
y_B = bpt_line(x, z_B)
y_A = bpt_line(x, z_A)

axes[0].plot(x, y_A, color='blue', lw=0.8)
axes[0].errorbar(log_N2_A, log_OIIIbeta_A, xerr=log_N2_A_err, yerr=log_OIIIbeta_A_err, fmt='o')
axes[0].set_xlim(x_left, 0.2)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_ylabel("log([OIII]/H$\\beta$)")
axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[0].text(0.05, 0.92, f"Source A,  z = {z_A}", transform=axes[0].transAxes, fontsize=9, va='top')
axes[0].tick_params(axis='x', which='both', labelbottom=False, bottom=False)

axes[1].set_xlim(x_left, 0.2)
axes[1].set_ylim(-1.5, 1.5)
axes[1].plot(x, y_B, color='blue', lw=0.8)
axes[1].errorbar(log_N2_B, log_OIIIbeta_B, xerr=log_N2_B_err, yerr=log_OIIIbeta_B_err, fmt='o')
axes[1].xaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].tick_params(axis='x', which='both', top=True, labeltop=False)
axes[1].set_xlabel("log([NII]/H$\\alpha$)")
axes[1].set_ylabel("log([OIII]/H$\\beta$)")
axes[1].text(0.05, 0.92, f"Source B,  z = {z_B}", transform=axes[1].transAxes, fontsize=9, va='top')
fig.tight_layout()

fig.savefig(f'{DIAGDIR}/bpt.png')
plt.show()

# ====================================
# csv tables
# ====================================
CUMULATIVE_KEYS = [
    'N2', 'log_N2', '[OIII]/Hbeta', 'log_OIIIbeta', 'O3N2', 'log_O3N2',
    '[NII]/[OII]', 'log_NII_OII', 'R23', 'log_R23', 'kk04_R23',
    'log_kk04_R23', 'KD02_O32', 'log_KD02_O32', 'KK04_O32', 'log_KK04_O32',
    'Halpha/Hbeta', 'log_Halpha_Hbeta', 'Hgamma/Hbeta', 'log_Hgamma_Hbeta'
]

COMPONENT_KEYS = [
    '[OIII]/Hbeta', 'log_OIIIbeta', 'R23', 'log_R23', 'kk04_R23',
    'log_kk04_R23', 'KD02_O32', 'log_KD02_O32', 'KK04_O32', 'log_KK04_O32',
    'Halpha/Hbeta', 'log_Halpha_Hbeta', 'Hgamma/Hbeta', 'log_Hgamma_Hbeta'
]


def build_cumulative_table(src):
    rows = []
    for key in CUMULATIVE_KEYS:
        rows.append(dict(ratio=key,
                         value=cumulative_out[src][key],
                         uncert=cumulative_out[src][f'{key}_err']))
    return pd.DataFrame(rows).round(3)


def build_component_table(src):
    rows = []
    for wing in WING_NAMES[src]:
        if wing not in component_out[src]:
            continue
        vals = component_out[src][wing]
        row = dict(wing=wing)
        for key in COMPONENT_KEYS:
            row[key] = vals[key]
            row[f'{key}_err'] = vals[f'{key}_err']
        rows.append(row)
    return pd.DataFrame(rows).round(3)


df_A_cumulative = build_cumulative_table('A')
df_B_cumulative = build_cumulative_table('B')
df_A_component = build_component_table('A')
df_B_component = build_component_table('B')

df_A_cumulative.to_csv(f'{DIAGDIR}/emission_lines_A_ratios.csv', index=False)
df_B_cumulative.to_csv(f'{DIAGDIR}/emission_lines_B_ratios.csv', index=False)
df_A_component.to_csv(f'{DIAGDIR}/wing_ratios_A.csv', index=False)
df_B_component.to_csv(f'{DIAGDIR}/wing_ratios_B.csv', index=False)

print()
print("Source A cumulative ratios:")
print(df_A_cumulative.to_string(index=False))
print()
print("Source B cumulative ratios:")
print(df_B_cumulative.to_string(index=False))
print()
print("Source A component-wise ratios:")
print(df_A_component.to_string(index=False))
print()
print("Source B component-wise ratios:")
print(df_B_component.to_string(index=False))

# ====================================
# save all the ratios
# ====================================
with open(f'{DIAGDIR}/A_ratios.pkl', 'wb') as fA:
    dill.dump({'cumulative': cumulative_A, 'component': component_out['A']}, fA)
with open(f'{DIAGDIR}/B_ratios.pkl', 'wb') as fB:
    dill.dump({'cumulative': cumulative_B, 'component': component_out['B']}, fB)
