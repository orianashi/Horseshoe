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

# physical_values_A.csv / physical_values_B.csv are shared with btfr.py and
# metallicity.py, each owning a different subset of quantities --
# upsert-by-quantity-name so re-running any one script updates only its own
# rows and leaves the others' rows intact, regardless of order.
def write_physical_values(source, rows):
    path = f'./output/physical_values_{source}.csv'
    new_df = pd.DataFrame(rows)
    try:
        # skipinitialspace + strip: some editor/linter re-aligns this file
        # with padding spaces around each field (incl. before quoted fields,
        # which otherwise breaks the default C parser's quote detection)
        existing = pd.read_csv(path, skipinitialspace=True)
        existing.columns = existing.columns.str.strip()
        existing['quantity'] = existing['quantity'].astype(str).str.strip()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame(columns=['quantity', 'value', 'uncertainty', 'notes'])
    existing = existing[~existing['quantity'].isin(new_df['quantity'])]
    pd.concat([existing, new_df], ignore_index=True).to_csv(path, index=False)


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


def KD02_O32(oii3726, oii3729, oiii5007, oii3726unc, oii3729unc, oiii5007unc):
    denom = oii3726 + oii3729
    denom_unc = np.sqrt(oii3726unc**2 + oii3729unc**2)
    return ratios(oiii5007, denom, oiii5007unc, denom_unc)


def KK04_O32(oii3726, oiii4959, oiii5007, oii3726unc, oiii4959unc, oiii5007unc):
    num = oiii4959 + oiii5007
    num_unc = np.sqrt(oiii4959unc**2 + oiii5007unc**2)
    return ratios(num, oii3726, num_unc, oii3726unc)


# Pettini & Pagel (2004) eq. 1: 12 + log(O/H) = 8.90 + 0.57 * N2, where
# N2 = log([NII]6583/Halpha) -- linear in log_N2, so the uncertainty
# propagates as 0.57 * log_N2_err
def pp04_N2_metallicity(log_N2, log_N2_err):
    metallicity = 8.90 + 0.57 * log_N2
    metallicity_err = 0.57 * log_N2_err
    return metallicity, metallicity_err


def bpt_line(log_NIIalpha, z):
    denom = (log_NIIalpha - 0.02 - 0.1833 * z)
    return 0.61 / denom + 1.2 + 0.03 * z


# [NII]6583 was fit as a single un-decomposed Gaussian per source (see
# dust_extinction.py's NII6583_results -- no component_fluxes breakdown
# exists for it, unlike every other line here), so there's no way to know
# from the fit itself how much of its flux belongs to which velocity
# component. Per-wing N2/[NII]/[OII] are therefore only computed for the
# 'central' wing, under the assumption that ALL of [NII]'s flux belongs to
# the central component -- the same convention this file already applies
# implicitly to every other single-Gaussian ("roleless") component, e.g.
# [OII]3726_A, which occupies index 0 (WING_NAMES[src][0] == 'central') in
# the per-wing loops below with no special-casing needed.
def add_central_N2_NII_OII(component_dict, halpha_f, halpha_u, oii3726_f, oii3726_u, nii_f, nii_u):
    N2_val, N2_err = ratios(nii_f, halpha_f, nii_u, halpha_u)
    NII_OII_val, NII_OII_err = ratios(nii_f, oii3726_f, nii_u, oii3726_u)
    log_N2_val, log_N2_err = log_uncert(N2_val, N2_err)
    log_NII_OII_val, log_NII_OII_err = log_uncert(NII_OII_val, NII_OII_err)
    metallicity_N2_val, metallicity_N2_err = pp04_N2_metallicity(log_N2_val, log_N2_err)
    component_dict['central'].update({
        'N2': N2_val, 'N2_err': N2_err,
        'log_N2': log_N2_val, 'log_N2_err': log_N2_err,
        '[NII]/[OII]': NII_OII_val, '[NII]/[OII]_err': NII_OII_err,
        'log_NII_OII': log_NII_OII_val, 'log_NII_OII_err': log_NII_OII_err,
        '12+log(O/H)_N2': metallicity_N2_val, '12+log(O/H)_N2_err': metallicity_N2_err,
    })


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
    NII_OII, NII_OII_err = ratios(NII6583['fluxes'][src], OII3726['fluxes'][src],
                                  NII6583['flux_uncerts'][src], OII3726['flux_uncerts'][src])
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
    log_NII_OII, log_NII_OII_err = log_uncert(NII_OII, NII_OII_err)
    log_kR23, log_kR23_err = log_uncert(kR23_val, kR23_err)
    log_KD02_O32_val, log_KD02_O32_err = log_uncert(KD02_O32_val, KD02_O32_err)
    log_KK04_O32_val, log_KK04_O32_err = log_uncert(KK04_O32_val, KK04_O32_err)
    metallicity_N2, metallicity_N2_err = pp04_N2_metallicity(log_N2, log_N2_err)

    logs[src] = {
        'log_N2': log_N2,
        'log_N2_err': log_N2_err,
        'log_OIIIbeta': log_OIIIbeta,
        'log_OIIIbeta_err': log_OIIIbeta_err,
    }

    cumulative_out[src] = {
        'N2': N2,
        'N2_err': N2_err,
        'log_N2': log_N2,
        'log_N2_err': log_N2_err,
        '12+log(O/H)_N2': metallicity_N2,
        '12+log(O/H)_N2_err': metallicity_N2_err,
        '[OIII]/Hbeta': OIIIbeta,
        '[OIII]/Hbeta_err': OIIIbeta_err,
        '[NII]/[OII]': NII_OII,
        '[NII]/[OII]_err': NII_OII_err,
        'log_NII_OII': log_NII_OII,
        'log_NII_OII_err': log_NII_OII_err,
        'R23': R23_val,
        'R23_err': R23_err,
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
# component-wise, per source per wing -- R23, kk04_R23, KD02_O32, and KK04_O32
# for every wing; N2/[NII]/[OII] added afterward for 'central' only (see
# add_central_N2_NII_OII above -- BPT still stays cumulative-only, since it
# plots the source's overall position, not a per-component one)
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
        log_kR23_val, log_kR23_err = log_uncert(kR23_val, kR23_err)
        KD02_O32_val, KD02_O32_err = KD02_O32(oii3726_f, oii3729_f, oiii5007_f,
                                              oii3726_u, oii3729_u, oiii5007_u)
        log_KD02_O32_val, log_KD02_O32_err = log_uncert(KD02_O32_val, KD02_O32_err)
        KK04_O32_val, KK04_O32_err = KK04_O32(oii3726_f, oiii4959_f, oiii5007_f,
                                              oii3726_u, oiii4959_u, oiii5007_u)
        log_KK04_O32_val, log_KK04_O32_err = log_uncert(KK04_O32_val, KK04_O32_err)

        component_out[src][wing] = {
            'R23': R23_val,
            'R23_err': R23_err,
            'kk04_R23': kR23_val,
            'kk04_R23_err': kR23_err,
            'log_kk04_R23': log_kR23_val,
            'log_kk04_R23_err': log_kR23_err,
            'KD02_O32': KD02_O32_val,
            'KD02_O32_err': KD02_O32_err,
            'log_KD02_O32': log_KD02_O32_val,
            'log_KD02_O32_err': log_KD02_O32_err,
            'KK04_O32': KK04_O32_val,
            'KK04_O32_err': KK04_O32_err,
            'log_KK04_O32': log_KK04_O32_val,
            'log_KK04_O32_err': log_KK04_O32_err,
            'Halpha/Hbeta': balmer_decrement_component['Halpha/Hbeta'][src][wing],
            'Halpha/Hbeta_err': balmer_decrement_component['Halpha/Hbeta_err'][src][wing],
            'Hgamma/Hbeta': balmer_decrement_component['Hgamma/Hbeta'][src][wing],
            'Hgamma/Hbeta_err': balmer_decrement_component['Hgamma/Hbeta_err'][src][wing],
            'E(B-V)': EBV_component['value'][src][wing],
            'E(B-V)_err': EBV_component['err'][src][wing],
        }

for src in ('A', 'B'):
    idx_central = WING_NAMES[src].index('central')
    add_central_N2_NII_OII(
        component_out[src],
        Halpha['component_fluxes'][src][idx_central], Halpha['component_flux_uncerts'][src][idx_central],
        OII3726['component_fluxes'][src][idx_central], OII3726['component_flux_uncerts'][src][idx_central],
        NII6583['fluxes'][src], NII6583['flux_uncerts'][src])

# Source A's [OII] doublet is a single un-split Gaussian (no red_wing entry
# in OII3726['component_fluxes']['A']), so the wing loop above skips 'A'
# 'red_wing' entirely via its IndexError guard -- but Halpha/Hbeta and
# Hgamma/Hbeta themselves DO have a red_wing component (see
# dust_extinction.py's balmer_decrement_component/EBV_component), so add
# that wing back in with just the Balmer-decrement fields; R23/KD02_O32/
# KK04_O32 stay genuinely absent (build_component_table fills them with NaN).
def add_balmer_only_wing(component_dict, src, wing, ebv_value, ebv_err):
    component_dict[src][wing] = {
        'Halpha/Hbeta': balmer_decrement_component['Halpha/Hbeta'][src][wing],
        'Halpha/Hbeta_err': balmer_decrement_component['Halpha/Hbeta_err'][src][wing],
        'Hgamma/Hbeta': balmer_decrement_component['Hgamma/Hbeta'][src][wing],
        'Hgamma/Hbeta_err': balmer_decrement_component['Hgamma/Hbeta_err'][src][wing],
        'E(B-V)': ebv_value,
        'E(B-V)_err': ebv_err,
    }

add_balmer_only_wing(component_out, 'A', 'red_wing',
                     EBV_component['value']['A']['red_wing'], EBV_component['err']['A']['red_wing'])

# same fix, for source B's blue_wing: B's [OII]3726/3729 and [OIII]4959/5007
# only have 2 components (red_wing, central -- no blue_wing), so the wing
# loop above skips 'B' 'blue_wing' entirely via its IndexError guard, even
# though Halpha/Hbeta/Hgamma DO have a blue_wing component for B -- add that
# wing back in with just the Balmer-decrement fields, same as A's red_wing.
add_balmer_only_wing(component_out, 'B', 'blue_wing',
                     EBV_component['value']['B']['blue_wing'], EBV_component['err']['B']['blue_wing'])

# ====================================
# component-wise, per source per wing, using the CUMULATIVE E(B-V) applied to
# each component's flux (see dust_extinction.py's component_results_cumulativeEBV)
# instead of a separate E(B-V) derived per wing -- same R23/kk04_R23/KD02_O32/
# KK04_O32 diagnostics as component_out above (N2/[NII]/[OII] added afterward
# for 'central' only, BPT stays cumulative-only), just built from
# component_fluxes_cumulativeEBV/
# component_flux_uncerts_cumulativeEBV, with a single per-source E(B-V) (not
# per-wing) reported alongside every wing since that's the one value actually
# applied to all of that source's components.
# ====================================
component_out_cumulativeEBV = {'A': {}, 'B': {}}
for src, wings in WING_NAMES.items():
    for i, wing in enumerate(wings):
        try:
            oii3726_f = OII3726['component_fluxes_cumulativeEBV'][src][i]
            oii3729_f = OII3729['component_fluxes_cumulativeEBV'][src][i]
            oiii4959_f = OIII4959['component_fluxes_cumulativeEBV'][src][i]
            oiii5007_f = OIII5007['component_fluxes_cumulativeEBV'][src][i]
            hbeta_f = Hbeta['component_fluxes_cumulativeEBV'][src][i]
            oii3726_u = OII3726['component_flux_uncerts_cumulativeEBV'][src][i]
            oii3729_u = OII3729['component_flux_uncerts_cumulativeEBV'][src][i]
            oiii4959_u = OIII4959['component_flux_uncerts_cumulativeEBV'][src][i]
            oiii5007_u = OIII5007['component_flux_uncerts_cumulativeEBV'][src][i]
            hbeta_u = Hbeta['component_flux_uncerts_cumulativeEBV'][src][i]
        except IndexError:
            continue

        R23_val, R23_err = R23(oii3726_f, oii3729_f, oiii4959_f, oiii5007_f, hbeta_f,
                               oii3726_u, oii3729_u, oiii4959_u, oiii5007_u, hbeta_u)
        kR23_val, kR23_err = kewley_R23(oii3726_f, oiii4959_f, oiii5007_f, hbeta_f,
                                        oii3726_u, oiii4959_u, oiii5007_u, hbeta_u)
        log_kR23_val, log_kR23_err = log_uncert(kR23_val, kR23_err)
        KD02_O32_val, KD02_O32_err = KD02_O32(oii3726_f, oii3729_f, oiii5007_f,
                                              oii3726_u, oii3729_u, oiii5007_u)
        log_KD02_O32_val, log_KD02_O32_err = log_uncert(KD02_O32_val, KD02_O32_err)
        KK04_O32_val, KK04_O32_err = KK04_O32(oii3726_f, oiii4959_f, oiii5007_f,
                                              oii3726_u, oiii4959_u, oiii5007_u)
        log_KK04_O32_val, log_KK04_O32_err = log_uncert(KK04_O32_val, KK04_O32_err)

        component_out_cumulativeEBV[src][wing] = {
            'R23': R23_val,
            'R23_err': R23_err,
            'kk04_R23': kR23_val,
            'kk04_R23_err': kR23_err,
            'log_kk04_R23': log_kR23_val,
            'log_kk04_R23_err': log_kR23_err,
            'KD02_O32': KD02_O32_val,
            'KD02_O32_err': KD02_O32_err,
            'log_KD02_O32': log_KD02_O32_val,
            'log_KD02_O32_err': log_KD02_O32_err,
            'KK04_O32': KK04_O32_val,
            'KK04_O32_err': KK04_O32_err,
            'log_KK04_O32': log_KK04_O32_val,
            'log_KK04_O32_err': log_KK04_O32_err,
            'Halpha/Hbeta': balmer_decrement_component['Halpha/Hbeta'][src][wing],
            'Halpha/Hbeta_err': balmer_decrement_component['Halpha/Hbeta_err'][src][wing],
            'Hgamma/Hbeta': balmer_decrement_component['Hgamma/Hbeta'][src][wing],
            'Hgamma/Hbeta_err': balmer_decrement_component['Hgamma/Hbeta_err'][src][wing],
            'E(B-V)': EBV_cumulative['value'][src],
            'E(B-V)_err': EBV_cumulative['err'][src],
        }

for src in ('A', 'B'):
    idx_central = WING_NAMES[src].index('central')
    add_central_N2_NII_OII(
        component_out_cumulativeEBV[src],
        Halpha['component_fluxes_cumulativeEBV'][src][idx_central],
        Halpha['component_flux_uncerts_cumulativeEBV'][src][idx_central],
        OII3726['component_fluxes_cumulativeEBV'][src][idx_central],
        OII3726['component_flux_uncerts_cumulativeEBV'][src][idx_central],
        NII6583['fluxes'][src], NII6583['flux_uncerts'][src])

# same fix as component_out above, for the cumulative-E(B-V) variant --
# every wing here shares the source's single cumulative E(B-V), so A's
# red_wing gets that same value (matching how every other wing/source is
# handled in this variant), not its own per-wing E(B-V).
add_balmer_only_wing(component_out_cumulativeEBV, 'A', 'red_wing',
                     EBV_cumulative['value']['A'], EBV_cumulative['err']['A'])

# same fix as component_out_cumulativeEBV above, for source B's blue_wing
add_balmer_only_wing(component_out_cumulativeEBV, 'B', 'blue_wing',
                     EBV_cumulative['value']['B'], EBV_cumulative['err']['B'])

# ====================================
# print all before and afters
# ====================================
print("=" * 60)
print("Cumulative diagnostics (dust-corrected)")
print("NOTE: Halpha/Hbeta, Hgamma/Hbeta, and E(B-V) below are computed from")
print("observed, pre-dust-correction fluxes (see dust_extinction.py) -- only")
print("N2, 12+log(O/H)_N2, [OIII]/Hbeta, R23, kk04_R23, KD02_O32, and KK04_O32 use dust-corrected fluxes.")
print("=" * 60)
for src in ('A', 'B'):
    print(f"Source {src}:")
    for key, val in cumulative_out[src].items():
        print(f"  {key}: {val:.4f}")

print()
print("=" * 60)
print("Component-wise diagnostics using COMPONENT-WISE E(B-V) (dust-corrected)")
print("-- R23 / kk04_R23 / KD02_O32 / KK04_O32 only")
print("NOTE: Halpha/Hbeta, Hgamma/Hbeta below are computed from observed,")
print("pre-dust-correction fluxes (see dust_extinction.py) -- only")
print("R23, kk04_R23, KD02_O32, and KK04_O32 use dust-corrected fluxes, each")
print("wing corrected using its OWN per-wing E(B-V).")
print("=" * 60)
for src, wings in component_out.items():
    for wing, vals in wings.items():
        print(f"Source {src} {wing}:")
        for key, val in vals.items():
            print(f"  {key}: {val:.4f}")

print()
print("=" * 60)
print("Component-wise diagnostics using CUMULATIVE E(B-V) (dust-corrected)")
print("-- R23 / kk04_R23 / KD02_O32 / KK04_O32 only")
print("NOTE: Halpha/Hbeta, Hgamma/Hbeta below are computed from observed,")
print("pre-dust-correction fluxes (see dust_extinction.py) -- only")
print("R23, kk04_R23, KD02_O32, and KK04_O32 use dust-corrected fluxes, every")
print("wing of a source corrected using that SAME source's cumulative E(B-V).")
print("=" * 60)
for src, wings in component_out_cumulativeEBV.items():
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

plt.rcParams['font.family'] = 'serif'
fig, axes = plt.subplots(2, 1, figsize=(4, 8), gridspec_kw={"hspace": 0})

# source B's N2 is a one-sided upper limit (NII6583 flux is a 3-sigma
# ceiling, see multiple_gaussian_integration.py/dust_extinction.py), so
# log_N2_B_err is NaN -- use a fixed nominal arrow length instead, purely to
# size the left-pointing upper-limit arrow drawn below, not a real error.
ARROW_LEN = 0.15
x_left = min(log_N2_A - log_N2_A_err, log_N2_B - ARROW_LEN) - 0.15
x = np.linspace(x_left, 0.2, 1000)
y_B = bpt_line(x, z_B)
y_A = bpt_line(x, z_A)

# SF/AGN region labels -- same fixed data-coordinate position on both subplots
sf_x, sf_y = -1.5, -0.5
agn_x, agn_y = -0.4, 1.2

axes[0].plot(x, y_A, color='blue', lw=0.8)
axes[0].errorbar(log_N2_A, log_OIIIbeta_A, xerr=log_N2_A_err, yerr=log_OIIIbeta_A_err,
                 fmt='o', color='orange', markersize=6, lw=0.45)
axes[0].set_xlim(x_left, 0.2)
axes[0].set_ylim(-1.5, 1.5)
axes[0].set_ylabel("log([OIII]/H$\\beta$)")
axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[0].text(log_N2_A, log_OIIIbeta_A - 0.15, r"Source A ($z$ = 1.679)",
             color='orange', fontsize=9, ha='center', va='top')
axes[0].text(sf_x, sf_y, "SF", color='blue', fontsize=9, ha='center', va='top')
axes[0].text(agn_x, agn_y, "AGN", color='blue', fontsize=9, ha='center', va='bottom')
axes[0].tick_params(axis='x', which='both', labelbottom=False, bottom=False)

axes[1].set_xlim(x_left, 0.2)
axes[1].set_ylim(-1.5, 1.5)
axes[1].plot(x, y_B, color='blue', lw=0.8)
# N2 is a one-sided upper limit for B -- draw a leftward arrow at the limit
# (xuplims=True) instead of a symmetric x error bar; ARROW_LEN only sizes
# the arrow glyph, it isn't a real uncertainty.
axes[1].errorbar(log_N2_B, log_OIIIbeta_B, xerr=ARROW_LEN, xuplims=True, yerr=log_OIIIbeta_B_err,
                 fmt='o', color='purple', markersize=6, lw=0.45)
axes[1].xaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
axes[1].tick_params(axis='x', which='both', top=True, labeltop=False)
axes[1].set_xlabel("log([NII]/H$\\alpha$)")
axes[1].set_ylabel("log([OIII]/H$\\beta$)")
axes[1].text(log_N2_B, log_OIIIbeta_B - 0.25, r"Source B ($z$ = 1.677)",
             color='purple', fontsize=9, ha='center', va='top')
axes[1].text(sf_x, sf_y, "SF", color='blue',fontsize=9, ha='center', va='top')
axes[1].text(agn_x, agn_y, "AGN", color='blue',fontsize=9, ha='center', va='bottom')
fig.tight_layout()

fig.savefig(f'{DIAGDIR}/bpt_dust_corrected.png')
plt.show()

# ====================================
# csv tables
# ====================================
CUMULATIVE_KEYS = [
    'N2', 'log_N2', '12+log(O/H)_N2', '[OIII]/Hbeta', '[NII]/[OII]',
    'log_NII_OII', 'R23', 'kk04_R23', 'log_kk04_R23', 'KD02_O32',
    'log_KD02_O32', 'KK04_O32', 'log_KK04_O32', 'Halpha/Hbeta',
    'Hgamma/Hbeta', 'E(B-V)'
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

# NII6583 is a 3-sigma upper limit for source B only (see
# multiple_gaussian_integration.py/dust_extinction.py) -- every quantity
# derived from it inherits a NaN uncertainty and a one-sided-limit caveat.
NII_UPPER_LIMIT_NOTE = '[NII] is a 3-sigma upper limit (Source B); value is a ceiling, uncertainty is NaN'
NII_DERIVED_KEYS = {'N2', 'log_N2', '12+log(O/H)_N2', '[NII]/[OII]', 'log_NII_OII'}


def build_cumulative_table(src):
    rows = []
    for key in CUMULATIVE_KEYS:
        if src == 'B' and key in NII_DERIVED_KEYS:
            note = NII_UPPER_LIMIT_NOTE
        else:
            note = CUMULATIVE_NOTES.get(key, '')
        rows.append(dict(ratio=key,
                         value=cumulative_out[src][key],
                         uncert=cumulative_out[src][f'{key}_err'],
                         notes=note))
    return pd.DataFrame(rows).round(3)


def build_component_table(src, out):
    rows = []
    for wing in WING_NAMES[src]:
        if wing not in out[src]:
            continue
        vals = out[src][wing]
        rows.append(dict(wing=wing,
                         # R23/KD02_O32/KK04_O32 need [OII], which for source
                         # A is a single un-split Gaussian (no red_wing
                         # entry) -- absent (NaN) for wings without it, e.g.
                         # A's red_wing (see add_balmer_only_wing below)
                         **{
                             'R23': vals.get('R23', np.nan), 'R23_err': vals.get('R23_err', np.nan),
                             'kk04_R23': vals.get('kk04_R23', np.nan), 'kk04_R23_err': vals.get('kk04_R23_err', np.nan),
                             'log_kk04_R23': vals.get('log_kk04_R23', np.nan), 'log_kk04_R23_err': vals.get('log_kk04_R23_err', np.nan),
                             'KD02_O32': vals.get('KD02_O32', np.nan), 'KD02_O32_err': vals.get('KD02_O32_err', np.nan),
                             'log_KD02_O32': vals.get('log_KD02_O32', np.nan), 'log_KD02_O32_err': vals.get('log_KD02_O32_err', np.nan),
                             'KK04_O32': vals.get('KK04_O32', np.nan), 'KK04_O32_err': vals.get('KK04_O32_err', np.nan),
                             'log_KK04_O32': vals.get('log_KK04_O32', np.nan), 'log_KK04_O32_err': vals.get('log_KK04_O32_err', np.nan),
                         },
                         # only populated for the 'central' wing (see
                         # add_central_N2_NII_OII) -- NaN elsewhere
                         **{
                             'N2': vals.get('N2', np.nan), 'N2_err': vals.get('N2_err', np.nan),
                             'log_N2': vals.get('log_N2', np.nan), 'log_N2_err': vals.get('log_N2_err', np.nan),
                             '12+log(O/H)_N2': vals.get('12+log(O/H)_N2', np.nan),
                             '12+log(O/H)_N2_err': vals.get('12+log(O/H)_N2_err', np.nan),
                             '[NII]/[OII]': vals.get('[NII]/[OII]', np.nan),
                             '[NII]/[OII]_err': vals.get('[NII]/[OII]_err', np.nan),
                             'log_NII_OII': vals.get('log_NII_OII', np.nan),
                             'log_NII_OII_err': vals.get('log_NII_OII_err', np.nan),
                         },
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

# component-wise tables come in two flavors, distinguished by which E(B-V)
# was applied to the component fluxes -- each wing's OWN per-wing E(B-V)
# (component_out, "componentEBV") vs. that source's single well-constrained
# cumulative E(B-V) applied uniformly to every wing (component_out_cumulativeEBV,
# "cumulativeEBV"). File names below are suffixed accordingly so it's never
# ambiguous which correction a given table used.
df_A_component_componentEBV = build_component_table('A', component_out)
df_B_component_componentEBV = build_component_table('B', component_out)
df_A_component_cumulativeEBV = build_component_table('A', component_out_cumulativeEBV)
df_B_component_cumulativeEBV = build_component_table('B', component_out_cumulativeEBV)

df_A_cumulative.to_csv(f'{DIAGDIR}/emission_lines_A_ratios_dust_corrected.csv', index=False)
df_B_cumulative.to_csv(f'{DIAGDIR}/emission_lines_B_ratios_dust_corrected.csv', index=False)
df_A_component_componentEBV.to_csv(f'{DIAGDIR}/wing_ratios_dust_corrected_componentEBV_A.csv', index=False)
df_B_component_componentEBV.to_csv(f'{DIAGDIR}/wing_ratios_dust_corrected_componentEBV_B.csv', index=False)
df_A_component_cumulativeEBV.to_csv(f'{DIAGDIR}/wing_ratios_dust_corrected_cumulativeEBV_A.csv', index=False)
df_B_component_cumulativeEBV.to_csv(f'{DIAGDIR}/wing_ratios_dust_corrected_cumulativeEBV_B.csv', index=False)

print()
print("Source A cumulative ratios:")
print(df_A_cumulative.to_string(index=False))
print()
print("Source B cumulative ratios:")
print(df_B_cumulative.to_string(index=False))
print()
print("Source A component-wise ratios (component-wise E(B-V)):")
print(df_A_component_componentEBV.to_string(index=False))
print()
print("Source B component-wise ratios (component-wise E(B-V)):")
print(df_B_component_componentEBV.to_string(index=False))
print()
print("Source A component-wise ratios (cumulative E(B-V)):")
print(df_A_component_cumulativeEBV.to_string(index=False))
print()
print("Source B component-wise ratios (cumulative E(B-V)):")
print(df_B_component_cumulativeEBV.to_string(index=False))

# ====================================
# save all the ratios
# ====================================
with open(f'{DIAGDIR}/A_ratios_dust_corrected_componentEBV.pkl', 'wb') as fA:
    dill.dump({'cumulative': cumulative_A, 'component': component_out['A']}, fA)
with open(f'{DIAGDIR}/B_ratios_dust_corrected_componentEBV.pkl', 'wb') as fB:
    dill.dump({'cumulative': cumulative_B, 'component': component_out['B']}, fB)
with open(f'{DIAGDIR}/A_ratios_dust_corrected_cumulativeEBV.pkl', 'wb') as fA:
    dill.dump({'cumulative': cumulative_A, 'component': component_out_cumulativeEBV['A']}, fA)
with open(f'{DIAGDIR}/B_ratios_dust_corrected_cumulativeEBV.pkl', 'wb') as fB:
    dill.dump({'cumulative': cumulative_B, 'component': component_out_cumulativeEBV['B']}, fB)

# ====================================
# Balmer decrements (componentwise + cumulative) and cumulative E(B-V), both
# sources -- Halpha/Hbeta and Hgamma/Hbeta are always computed from observed,
# pre-dust-correction fluxes (identical between the componentEBV/
# cumulativeEBV variants), so component_out_cumulativeEBV is as good a source
# for them as component_out; cumulativeEBV is used here for consistency with
# metallicity.py, which loads that same variant.
# ====================================
for src in ('A', 'B'):
    cumulative = cumulative_out[src]
    rows = [
        dict(quantity='Halpha/Hbeta_cumulative', value=cumulative['Halpha/Hbeta'],
             uncertainty=cumulative['Halpha/Hbeta_err'], notes='observed, pre-dust-correction fluxes'),
        dict(quantity='Hgamma/Hbeta_cumulative', value=cumulative['Hgamma/Hbeta'],
             uncertainty=cumulative['Hgamma/Hbeta_err'], notes='observed, pre-dust-correction fluxes'),
    ]
    for wing, vals in component_out_cumulativeEBV[src].items():
        rows.append(dict(quantity=f'Halpha/Hbeta_{wing}', value=vals['Halpha/Hbeta'],
                         uncertainty=vals['Halpha/Hbeta_err'], notes=f'{wing} component, observed fluxes'))
        rows.append(dict(quantity=f'Hgamma/Hbeta_{wing}', value=vals['Hgamma/Hbeta'],
                         uncertainty=vals['Hgamma/Hbeta_err'], notes=f'{wing} component, observed fluxes'))
    rows.append(dict(quantity='E(B-V)_cumulative', value=cumulative['E(B-V)'],
                     uncertainty=cumulative['E(B-V)_err'], notes='derived from cumulative Halpha/Hbeta decrement'))
    write_physical_values(src, rows)
