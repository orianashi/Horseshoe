import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from multiple_gaussian_integration import (LINES, load_model, load_tie_map,
                                           flux_and_uncert)

plt.ion()

z_A = 1.679  # redshift for source A
z_B = 1.677  # redshift for source B


# =======
# un-pikl: improved (multi-gaussian, full-covariance) fluxes where available
# =======
def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def load_balmer(line_name):
    """Halpha/Hbeta/Hgamma flux + uncertainty computed directly from the
    joint tied fit (balmer_joint_gaussian_fitting.py), via the same
    full-covariance flux_and_uncert() used for every other line -- rather
    than depending on a separately-run multiple_gaussian_integration.py
    output, so this always reflects the current joint-fit model. tie_map
    carries the baked tie ratios so tied components' uncertainty still
    propagates correctly (see load_tie_map). No diagonal-only variant is
    needed here."""
    cfg = LINES[line_name]
    model = load_model(cfg['pkl'])
    tie_map = load_tie_map(cfg['pkl'])
    flux = {}
    uncert = {}
    for src, indices in (('A', cfg['A_indices']), ('B', cfg['B_indices'])):
        flux[src], uncert[src] = flux_and_uncert(model, indices, tie_map)
    return {'fluxes': flux, 'flux_uncerts': uncert}


Halpha_new = load_balmer('Halpha')
Hbeta_new = load_balmer('Hbeta')
Hgamma_new = load_balmer('Hgamma')
OIII4959_new = load_pkl(
    './output/improved_gaussians/OIII4959/OIII4959_fluxes.pkl')
OIII5007_new = load_pkl(
    './output/improved_gaussians/OIII5007/OIII5007_fluxes.pkl')
OII_new = load_pkl('./output/improved_gaussians/OII/OII_fluxes.pkl')

# lines with no improved_gaussians refit yet: fall back to legacy single-gaussian fluxes
NIIalphas_legacy = load_pkl('./output/NII/NII_Halpha_ratios.pkl')

# NII_Halpha_ratios.pkl['fluxes'][src] = [flux_Halpha, flux_NII6583, flux_NII6548]
NII6583 = {
    'flux': {
        src: NIIalphas_legacy['fluxes'][src][1]
        for src in ('A', 'B')
    },
    'uncert': {
        src: NIIalphas_legacy['flux_uncerts'][src][1]
        for src in ('A', 'B')
    },
}
# OII_fluxes.pkl['component_fluxes'][src] = [flux_3726, flux_3729], both directly
# fit (and full-covariance propagated) by the improved 4-gaussian OII model.
OII3726 = {
    'flux': {
        src: OII_new['component_fluxes'][src][0]
        for src in ('A', 'B')
    },
    'uncert': {
        src: OII_new['component_flux_uncerts'][src][0]
        for src in ('A', 'B')
    },
}
OII3729 = {
    'flux': {
        src: OII_new['component_fluxes'][src][1]
        for src in ('A', 'B')
    },
    'uncert': {
        src: OII_new['component_flux_uncerts'][src][1]
        for src in ('A', 'B')
    },
}


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
# per-line balmer decrement

print(Halpha_new['fluxes']['A'])

# ====================================
# recompute ratios per source, swapping in improved fluxes where available
# ====================================
ratios_out = {}
logs = {}

for src in ('A', 'B'):
    NIIalpha, NIIalpha_err = ratios(NII6583['flux'][src],
                                    Halpha_new['fluxes'][src],
                                    NII6583['uncert'][src],
                                    Halpha_new['flux_uncerts'][src])
    OIIIbeta, OIIIbeta_err = ratios(OIII5007_new['fluxes'][src],
                                    Hbeta_new['fluxes'][src],
                                    OIII5007_new['flux_uncerts'][src],
                                    Hbeta_new['flux_uncerts'][src])
    Halpha_Hbeta, Halpha_Hbeta_err = ratios(Halpha_new['fluxes'][src],
                                            Hbeta_new['fluxes'][src],
                                            Halpha_new['flux_uncerts'][src],
                                            Hbeta_new['flux_uncerts'][src])
    Hgamma_Hbeta, Hgamma_Hbeta_err = ratios(Hgamma_new['fluxes'][src],
                                            Hbeta_new['fluxes'][src],
                                            Hgamma_new['flux_uncerts'][src],
                                            Hbeta_new['flux_uncerts'][src])
    o3n2, o3n2_err = ratios(OIIIbeta, NIIalpha, OIIIbeta_err, NIIalpha_err)

    R23_val, R23_err = R23(
        OII3726['flux'][src], OII3729['flux'][src],
        OIII4959_new['fluxes'][src], OIII5007_new['fluxes'][src],
        Hbeta_new['fluxes'][src], OII3726['uncert'][src],
        OII3729['uncert'][src], OIII4959_new['flux_uncerts'][src],
        OIII5007_new['flux_uncerts'][src], Hbeta_new['flux_uncerts'][src])
    kR23_val, kR23_err = kewley_R23(
        OII3726['flux'][src], OIII4959_new['fluxes'][src],
        OIII5007_new['fluxes'][src], Hbeta_new['fluxes'][src],
        OII3726['uncert'][src], OIII4959_new['flux_uncerts'][src],
        OIII5007_new['flux_uncerts'][src], Hbeta_new['flux_uncerts'][src])

    log_arg = Halpha_Hbeta / 2.86
    log_arg_err = Halpha_Hbeta_err / 2.86
    log10E, log10E_err = log_uncert(log_arg, log_arg_err)
    E_BV = 1.97 * log10E
    E_BV_err = 1.97 * log10E_err

    log_NIIalpha, log_NIIalpha_err = log_uncert(NIIalpha, NIIalpha_err)
    log_OIIIbeta, log_OIIIbeta_err = log_uncert(OIIIbeta, OIIIbeta_err)
    log_o3n2, log_o3n2_err = log_uncert(o3n2, o3n2_err)
    logR23, logR23_err = log_uncert(R23_val, R23_err)
    logkR23, logkR23_err = log_uncert(kR23_val, kR23_err)

    logs[src] = {
        'log_NIIalpha': log_NIIalpha,
        'log_NIIalpha_err': log_NIIalpha_err,
        'log_OIIIbeta': log_OIIIbeta,
        'log_OIIIbeta_err': log_OIIIbeta_err,
        'log_o3n2': log_o3n2,
        'log_o3n2_err': log_o3n2_err,
        'logR23': logR23,
        'logR23_err': logR23_err,
        'logkR23': logkR23,
        'logkR23_err': logkR23_err,
    }

    ratios_out[src] = {
        'N2': NIIalpha,
        'N2_err': NIIalpha_err,
        'log(N2)': log_NIIalpha,
        'log(N2)_err': log_NIIalpha_err,
        'O3N2': o3n2,
        'O3N2_err': o3n2_err,
        'log(O3N2)': log_o3n2,
        'log(O3N2)_err': log_o3n2_err,
        'Halpha/Hbeta': Halpha_Hbeta,
        'Halpha/Hbeta_err': Halpha_Hbeta_err,
        'Hgamma/Hbeta': Hgamma_Hbeta,
        'Hgamma/Hbeta_err': Hgamma_Hbeta_err,
        'E(B-V)': E_BV,
        'E(B-V)_err': E_BV_err,
        'R23': R23_val,
        'R23_err': R23_err,
        'log(R23)': logR23,
        'log(R23)_err': logR23_err,
        'kk04_R23': kR23_val,
        'kk04_R23_err': kR23_err,
        '[OIII]/Hbeta': OIIIbeta,
        '[OIII]/Hbeta_err': OIIIbeta_err,
    }
"""
    print("=" * 20)
    print(f"Source {src}")
    print("=" * 20)
    print(f"log([NII]/Halpha): {log_NIIalpha} +- {log_NIIalpha_err}")
    print(
        f"src {src} metallicity from intersection w/ pettini 2004 N2 diagnostic line: "
        f"{pp04_N2_metallicity_line(log_NIIalpha)}")
    print(f"log(O3N2): {log_o3n2} +- {log_o3n2_err}")
    print(f"src {src} metallicity from intersection w/ pettini 2004 O3N2: "
          f"{pettini_metallicity(log_o3n2)}")
    print(f"log(R23_{src}): {logR23} +- {logR23_err}")
    print(f"KK04 log(R23_{src}): {logkR23} +- {logkR23_err}")
    print(
        f"log([OIII]/Hbeta) for source {src}: {log_OIIIbeta} +- {log_OIIIbeta_err}"
    )
    print(f"E(B-V) for source {src}: {E_BV} +- {E_BV_err}")
"""
ratios_A = ratios_out['A']
ratios_B = ratios_out['B']

# ====================================
# N2 metallicity plot
# ====================================
logN2s = np.linspace(-2.5, 0, 500)
pettini_N2_metals = pp04_N2_metallicity_line(logN2s)
pettini_N2_metals_s = pp04_N2_metallicity_cubic(logN2s)

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(logN2s, pettini_N2_metals, color='black', ls='--', lw=0.8)
ax.plot(logN2s, pettini_N2_metals_s, color='red', ls='-', lw=0.5)
ax.axvline(logs['A']['log_NIIalpha'], label='Source A', ls='--', color='pink')
ax.axvline(logs['B']['log_NIIalpha'],
           label='Source B',
           ls='--',
           color='purple')
ax.axhline(8.66, label="solar", ls='--', color='black', lw=0.5)
ax.legend()
ax.set_xlabel("Log([NII]/Halpha)")
ax.set_ylabel("12 + log(O/H)")
ax.set_title("From Pettini and Pagel 2004 (improved fluxes)")
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
ax.axvline(logs['A']['log_o3n2'], label='Source A', ls='--', color='pink')
ax.axvline(logs['B']['log_o3n2'], label='Source B', ls='--', color='purple')
ax.axhline(8.66, label="solar", ls='--', color='black', lw=0.5)
ax.legend()
ax.set_xlabel("Log([O3N2])")
ax.set_ylabel("12 + log(O/H)")
ax.set_title("From Pettini and Pagel 2004 (improved fluxes)")
plt.show()

# ====================================
# BPT diagram
# ====================================
log_NIIalpha_A = logs['A']['log_NIIalpha']
log_NIIalpha_A_uncert = logs['A']['log_NIIalpha_err']
log_NIIalpha_B = logs['B']['log_NIIalpha']
log_NIIalpha_B_uncert = logs['B']['log_NIIalpha_err']
log_OIIIbeta_A = logs['A']['log_OIIIbeta']
log_OIIIbeta_A_uncert = logs['A']['log_OIIIbeta_err']
log_OIIIbeta_B = logs['B']['log_OIIIbeta']
log_OIIIbeta_B_uncert = logs['B']['log_OIIIbeta_err']

fig, axes = plt.subplots(2, 1, figsize=(4, 8), gridspec_kw={"hspace": 0})

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

fig.savefig('./output/diagnostics/bpt_improved.png')
plt.show()

# ====================================
# save all the ratios
# ====================================
with open('./output/tabling/A_ratios_improved.pkl', 'wb') as fA:
    dill.dump(ratios_A, fA)
with open('./output/tabling/B_ratios_improved.pkl', 'wb') as fB:
    dill.dump(ratios_B, fB)
print("*= 20")
print(ratios_A)
print("*= 20")
print(ratios_B)