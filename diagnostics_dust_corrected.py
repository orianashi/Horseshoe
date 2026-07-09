import os

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.ion()

z_A = 1.679  # redshift for source A
z_B = 1.677  # redshift for source B

DUSTDIR = './output/dust_corrected'
DIAGDIR = f'{DUSTDIR}/dust_corrected_diagnostics'
os.makedirs(DIAGDIR, exist_ok=True)

# source A components are [central, red_wing]; source B are [red_wing,
# central, blue_wing] -- except [OII]3726/3729 where A has only 1 (roleless)
# component and B has only 2 (no blue_wing).
WING_NAMES = {
    'A': ['central', 'red_wing'],
    'B': ['red_wing', 'central', 'blue_wing'],
}


# =======
# un-pikl: dust-corrected fluxes (see dust_extinction.py)
# =======
def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


Halpha = load_pkl(f'{DUSTDIR}/Halpha_dust_corrected.pkl')
Hbeta = load_pkl(f'{DUSTDIR}/Hbeta_dust_corrected.pkl')
Hgamma = load_pkl(f'{DUSTDIR}/Hgamma_dust_corrected.pkl')
OIII4959 = load_pkl(f'{DUSTDIR}/OIII4959_dust_corrected.pkl')
OIII5007 = load_pkl(f'{DUSTDIR}/OIII5007_dust_corrected.pkl')
OII3726 = load_pkl(f'{DUSTDIR}/OII3726_dust_corrected.pkl')
OII3729 = load_pkl(f'{DUSTDIR}/OII3729_dust_corrected.pkl')
NII6583 = load_pkl(f'{DUSTDIR}/NII6583_dust_corrected.pkl')

# Balmer decrement and E(B-V) are duplicated identically into every
# dust_corrected pickle (see dust_extinction.py) -- pull them from Halpha's.
balmer_decrement_cumulative = {
    'Halpha/Hbeta': Halpha['Halpha/Hbeta_cumulative'],
    'Halpha/Hbeta_err': Halpha['Halpha/Hbeta_cumulative_err'],
    'Hgamma/Hbeta': Halpha['Hgamma/Hbeta_cumulative'],
    'Hgamma/Hbeta_err': Halpha['Hgamma/Hbeta_cumulative_err'],
}
balmer_decrement_component = {
    'Halpha/Hbeta': Halpha['Halpha/Hbeta_component'],
    'Halpha/Hbeta_err': Halpha['Halpha/Hbeta_component_err'],
    'Hgamma/Hbeta': Halpha['Hgamma/Hbeta_component'],
    'Hgamma/Hbeta_err': Halpha['Hgamma/Hbeta_component_err'],
}
EBV_cumulative = {
    'value': Halpha['E(B-V)_cumulative'],
    'err': Halpha['E(B-V)_cumulative_err'],
}
EBV_component = {
    'value': Halpha['E(B-V)_component'],
    'err': Halpha['E(B-V)_component_err'],
}


# ====================
# define functions (duplicated from diagnostics.py)
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


def bpt_line(log_NIIalpha, z):
    denom = (log_NIIalpha - 0.02 - 0.1833 * z)
    return 0.61 / denom + 1.2 + 0.03 * z


# ====================================
# cumulative, per source
# ====================================
cumulative_out = {}
logs = {}

for src in ('A', 'B'):
    N2, N2_err = ratios(NII6583['fluxes'][src], Halpha['fluxes'][src],
                        NII6583['flux_uncerts'][src], Halpha['flux_uncerts'][src])
    OIIIbeta, OIIIbeta_err = ratios(OIII5007['fluxes'][src], Hbeta['fluxes'][src],
                                    OIII5007['flux_uncerts'][src], Hbeta['flux_uncerts'][src])
    R23_val, R23_err = R23(
        OII3726['fluxes'][src], OII3729['fluxes'][src],
        OIII4959['fluxes'][src], OIII5007['fluxes'][src], Hbeta['fluxes'][src],
        OII3726['flux_uncerts'][src], OII3729['flux_uncerts'][src],
        OIII4959['flux_uncerts'][src], OIII5007['flux_uncerts'][src], Hbeta['flux_uncerts'][src])
    kR23_val, kR23_err = kewley_R23(
        OII3726['fluxes'][src], OIII4959['fluxes'][src], OIII5007['fluxes'][src],
        Hbeta['fluxes'][src], OII3726['flux_uncerts'][src], OIII4959['flux_uncerts'][src],
        OIII5007['flux_uncerts'][src], Hbeta['flux_uncerts'][src])

    log_N2, log_N2_err = log_uncert(N2, N2_err)
    log_OIIIbeta, log_OIIIbeta_err = log_uncert(OIIIbeta, OIIIbeta_err)

    logs[src] = {
        'log_N2': log_N2,
        'log_N2_err': log_N2_err,
        'log_OIIIbeta': log_OIIIbeta,
        'log_OIIIbeta_err': log_OIIIbeta_err,
    }

    cumulative_out[src] = {
        'N2': N2,
        'N2_err': N2_err,
        '[OIII]/Hbeta': OIIIbeta,
        '[OIII]/Hbeta_err': OIIIbeta_err,
        'R23': R23_val,
        'R23_err': R23_err,
        'kk04_R23': kR23_val,
        'kk04_R23_err': kR23_err,
        'Halpha/Hbeta': balmer_decrement_cumulative['Halpha/Hbeta'][src],
        'Halpha/Hbeta_err': balmer_decrement_cumulative['Halpha/Hbeta_err'][src],
        'Hgamma/Hbeta': balmer_decrement_cumulative['Hgamma/Hbeta'][src],
        'Hgamma/Hbeta_err': balmer_decrement_cumulative['Hgamma/Hbeta_err'][src],
        'E(B-V)': EBV_cumulative['value'][src],
        'E(B-V)_err': EBV_cumulative['err'][src],
    }

cumulative_A = cumulative_out['A']
cumulative_B = cumulative_out['B']

# ====================================
# component-wise, per source per wing -- R23 and kk04_R23 only
# (N2/BPT need NII, which has no per-wing decomposition)
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
            oii3726_u = OII3726['component_flux_uncerts'][src][i]
            oii3729_u = OII3729['component_flux_uncerts'][src][i]
            oiii4959_u = OIII4959['component_flux_uncerts'][src][i]
            oiii5007_u = OIII5007['component_flux_uncerts'][src][i]
            hbeta_u = Hbeta['component_flux_uncerts'][src][i]
        except IndexError:
            continue

        R23_val, R23_err = R23(oii3726_f, oii3729_f, oiii4959_f, oiii5007_f, hbeta_f,
                               oii3726_u, oii3729_u, oiii4959_u, oiii5007_u, hbeta_u)
        kR23_val, kR23_err = kewley_R23(oii3726_f, oiii4959_f, oiii5007_f, hbeta_f,
                                        oii3726_u, oiii4959_u, oiii5007_u, hbeta_u)

        component_out[src][wing] = {
            'R23': R23_val,
            'R23_err': R23_err,
            'kk04_R23': kR23_val,
            'kk04_R23_err': kR23_err,
            'Halpha/Hbeta': balmer_decrement_component['Halpha/Hbeta'][src][wing],
            'Halpha/Hbeta_err': balmer_decrement_component['Halpha/Hbeta_err'][src][wing],
            'Hgamma/Hbeta': balmer_decrement_component['Hgamma/Hbeta'][src][wing],
            'Hgamma/Hbeta_err': balmer_decrement_component['Hgamma/Hbeta_err'][src][wing],
            'E(B-V)': EBV_component['value'][src][wing],
            'E(B-V)_err': EBV_component['err'][src][wing],
        }

# ====================================
# print all before and afters
# ====================================
print("=" * 60)
print("Cumulative diagnostics (dust-corrected)")
print("NOTE: Halpha/Hbeta, Hgamma/Hbeta, and E(B-V) below are computed from")
print("observed, pre-dust-correction fluxes (see dust_extinction.py) -- only")
print("N2, [OIII]/Hbeta, R23, and kk04_R23 use dust-corrected fluxes.")
print("=" * 60)
for src in ('A', 'B'):
    print(f"Source {src}:")
    for key, val in cumulative_out[src].items():
        print(f"  {key}: {val:.4f}")

print()
print("=" * 60)
print("Component-wise diagnostics (dust-corrected) -- R23 / kk04_R23 only")
print("NOTE: Halpha/Hbeta, Hgamma/Hbeta, and E(B-V) below are computed from")
print("observed, pre-dust-correction fluxes (see dust_extinction.py) -- only")
print("R23 and kk04_R23 use dust-corrected fluxes.")
print("=" * 60)
for src, wings in component_out.items():
    for wing, vals in wings.items():
        print(f"Source {src} {wing}:")
        for key, val in vals.items():
            print(f"  {key}: {val:.4f}")

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

fig.savefig(f'{DIAGDIR}/bpt_dust_corrected.png')
plt.show()

# ====================================
# csv tables
# ====================================
CUMULATIVE_KEYS = [
    'N2', '[OIII]/Hbeta', 'R23', 'kk04_R23', 'Halpha/Hbeta', 'Hgamma/Hbeta', 'E(B-V)'
]

# Halpha/Hbeta, Hgamma/Hbeta, and E(B-V) are computed from observed,
# pre-dust-correction fluxes (E(B-V) is derived from that same decrement, so
# it can't be dust-corrected either) -- everything else uses dust-corrected
# fluxes. Flagged per-row via the 'notes' column (see emission_line_table_improved.py's
# flux_row() for the same convention).
OBSERVED_FLUX_NOTE = 'from observed, pre-dust-correction fluxes'
CUMULATIVE_NOTES = {
    'Halpha/Hbeta': OBSERVED_FLUX_NOTE,
    'Hgamma/Hbeta': OBSERVED_FLUX_NOTE,
    'E(B-V)': 'derived from observed Halpha/Hbeta decrement',
}


def build_cumulative_table(src):
    rows = []
    for key in CUMULATIVE_KEYS:
        rows.append(dict(ratio=key,
                         value=cumulative_out[src][key],
                         uncert=cumulative_out[src][f'{key}_err'],
                         notes=CUMULATIVE_NOTES.get(key, '')))
    return pd.DataFrame(rows).round(3)


def build_component_table(src):
    rows = []
    for wing in WING_NAMES[src]:
        if wing not in component_out[src]:
            continue
        vals = component_out[src][wing]
        rows.append(dict(wing=wing,
                         R23=vals['R23'], R23_err=vals['R23_err'],
                         kk04_R23=vals['kk04_R23'], kk04_R23_err=vals['kk04_R23_err'],
                         **{
                             # suffixed _observed: computed from observed,
                             # pre-dust-correction fluxes, unlike R23/kk04_R23
                             'Halpha/Hbeta_observed': vals['Halpha/Hbeta'],
                             'Halpha/Hbeta_observed_err': vals['Halpha/Hbeta_err'],
                             'Hgamma/Hbeta_observed': vals['Hgamma/Hbeta'],
                             'Hgamma/Hbeta_observed_err': vals['Hgamma/Hbeta_err'],
                         },
                         **{'E(B-V)': vals['E(B-V)'], 'E(B-V)_err': vals['E(B-V)_err']}))
    return pd.DataFrame(rows).round(3)


df_A_cumulative = build_cumulative_table('A')
df_B_cumulative = build_cumulative_table('B')
df_A_component = build_component_table('A')
df_B_component = build_component_table('B')

df_A_cumulative.to_csv(f'{DIAGDIR}/emission_lines_A_ratios_dust_corrected.csv', index=False)
df_B_cumulative.to_csv(f'{DIAGDIR}/emission_lines_B_ratios_dust_corrected.csv', index=False)
df_A_component.to_csv(f'{DIAGDIR}/wing_ratios_dust_corrected_A.csv', index=False)
df_B_component.to_csv(f'{DIAGDIR}/wing_ratios_dust_corrected_B.csv', index=False)

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
with open(f'{DIAGDIR}/A_ratios_dust_corrected.pkl', 'wb') as fA:
    dill.dump({'cumulative': cumulative_A, 'component': component_out['A']}, fA)
with open(f'{DIAGDIR}/B_ratios_dust_corrected.pkl', 'wb') as fB:
    dill.dump({'cumulative': cumulative_B, 'component': component_out['B']}, fB)
