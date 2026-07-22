import os

import dill
import numpy as np
import astropy.units as u

from multiple_gaussian_integration import LINES

# ===================
# constants
# ===================
R_v = 3.1  # Cardelli
ha_hb = 2.86  # Case B, T = 10^4 K and Ne = 10^2

LINE_NAMES = ['Halpha', 'Hbeta', 'Hgamma', 'OIII5007', 'OIII4959', 'OII3726', 'OII3729']

wavelengths_AA = {
    'Halpha': 6562.819,
    'OIII5007': 5006.843,
    'OIII4959': 4958.911,
    'Hbeta': 4861.333,
    'Hgamma': 4340.47,
    'OII3726': 3726.03,
    'OII3729': 3728.815,
    'NII6583': 6583.45,
}

# source A components are [central, red_wing]; source B are [red_wing,
# central, blue_wing] -- except [OII]3726/3729 where both sources have only 1
# (roleless) component (source B's red wing was removed as noise-dominated,
# see jointfit_all.py). Slicing to len(component_fluxes) handles ordinary
# truncation (a source missing its LAST wing(s)) with no special-casing --
# but [OII] for source B is missing its FIRST wing (red_wing) instead, so its
# lone surviving component would be mislabeled 'red_wing' by that generic
# slicing. OII_WING_NAMES gives [OII] lines their own (source-specific)
# wing-name list so their sole component is correctly labeled 'central'.
WING_NAMES = {
    'A': ['central', 'red_wing'],
    'B': ['red_wing', 'central', 'blue_wing'],
}
OII_LINES = {'OII3726', 'OII3729'}
OII_WING_NAMES = {
    'A': ['central'],
    'B': ['central'],
}

OUTDIR = './output/dust_corrected'

# ====================
# load in data
# ====================
def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


# fluxes + errors, cumulative and per-component, from multiple_gaussian_integration.py
fluxes = {line: load_pkl(LINES[line]['save']) for line in LINE_NAMES}

# NII6583 (jointfit_all's fit calls it [NII]6584 -- same 6583.45AA line, see
# multiple_gaussian_integration.py's 'NII6584' entry) now has a proper joint
# fit tied into the same compound model/covariance as every other line here
# -- still a single, un-decomposed component per source (A_indices=[31],
# B_indices=[32]), just no longer the old standalone single-Gaussian fit.
NII6584_fit = load_pkl('./output/joint_fit/jointfit_all/fluxes/NII6584_fluxes.pkl')
NII6583_flux = NII6584_fit['fluxes']
NII6583_flux_err = NII6584_fit['flux_uncerts']
# source B's is a 3-sigma upper limit (see multiple_gaussian_integration.py) --
# NaN flux_err, propagated as such below
NII6583_is_upper_limit = NII6584_fit['is_upper_limit']


# ====================
# define functions
# ====================
def wave_num(wl_AA):
    wl_AA = wl_AA * u.AA
    wl_um = wl_AA.to(u.um)
    return wl_um.value


def k(wl_AA):
    wl_AA = wl_AA * u.AA
    wl_um = wl_AA.to(u.um)
    x = 1 / wl_um.value
    y = x - 1.82
    a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    k = R_v * a + b
    # wavelength and R_v are fixed constants -- nothing to propagate
    k_err = 0.0
    return k, k_err


k_Ha, _ = k(wavelengths_AA['Halpha'])
k_Hb, _ = k(wavelengths_AA['Hbeta'])


def EB_V(bd, bd_err):
    denom = k_Hb - k_Ha
    EB_V = 2.5 / denom * np.log10(bd / ha_hb)
    # denom and ha_hb are constants; only bd carries uncertainty.
    # d(log10(bd/ha_hb))/dbd = 1 / (bd * ln(10))
    EB_V_err = (2.5 / denom) * (bd_err / (bd * np.log(10)))
    return EB_V, EB_V_err


def flux_correct(flux, flux_err, line, EB_V, EB_V_err):
    k_line, _ = k(wavelengths_AA[line])
    exp = 0.4 * k_line * EB_V
    flux_correct = flux * 10**exp
    # flux and EB_V treated as independent (no covariance matrix): standard
    # propagation through f = flux * 10^(c*EB_V), c = 0.4*k_line
    rel_flux_err = flux_err / flux # not the final flux error, this is just math to do the absolute error propogation 
    rel_EB_V_err = 0.4 * k_line * np.log(10) * EB_V_err
    flux_correct_err = flux_correct * np.sqrt(rel_flux_err**2 + rel_EB_V_err**2)
    return flux_correct, flux_correct_err


# duplicated from diagnostics.py, so this file computes its own Balmer
# decrement directly from the raw per-line fluxes rather than depending on
# diagnostics.py's or wing_balmer_flux_ratios.py's precomputed output.
def ratios(num, denom, num_uncert, denom_uncert):
    ratio = num / denom
    ratio_uncert = np.sqrt((num_uncert / num)**2 +
                           (denom_uncert / denom)**2) * ratio
    return ratio, ratio_uncert


# ====================
# balmer decrement -- cumulative
# ====================
Halpha_Hbeta_cumulative = {}
Hgamma_Hbeta_cumulative = {}
for src in ('A', 'B'):
    Halpha_Hbeta_cumulative[src] = ratios(
        fluxes['Halpha']['fluxes'][src], fluxes['Hbeta']['fluxes'][src],
        fluxes['Halpha']['flux_uncerts'][src], fluxes['Hbeta']['flux_uncerts'][src])
    Hgamma_Hbeta_cumulative[src] = ratios(
        fluxes['Hgamma']['fluxes'][src], fluxes['Hbeta']['fluxes'][src],
        fluxes['Hgamma']['flux_uncerts'][src], fluxes['Hbeta']['flux_uncerts'][src])

# ====================
# balmer decrement -- component-wise
# ====================
Halpha_Hbeta_component = {'A': {}, 'B': {}}
Hgamma_Hbeta_component = {'A': {}, 'B': {}}
for src, wings in WING_NAMES.items():
    for i, wing in enumerate(wings):
        Halpha_Hbeta_component[src][wing] = ratios(
            fluxes['Halpha']['component_fluxes'][src][i], fluxes['Hbeta']['component_fluxes'][src][i],
            fluxes['Halpha']['component_flux_uncerts'][src][i], fluxes['Hbeta']['component_flux_uncerts'][src][i])
        Hgamma_Hbeta_component[src][wing] = ratios(
            fluxes['Hgamma']['component_fluxes'][src][i], fluxes['Hbeta']['component_fluxes'][src][i],
            fluxes['Hgamma']['component_flux_uncerts'][src][i], fluxes['Hbeta']['component_flux_uncerts'][src][i])

# ====================
# run while not distinguishing different BD for outflows:
# ====================
EBV_cumulative = {}
for src in ('A', 'B'):
    bd, bd_err = Halpha_Hbeta_cumulative[src]
    EBV_cumulative[src] = EB_V(bd, bd_err)

cumulative_results = {line: {'A': {}, 'B': {}} for line in LINE_NAMES}
for line in LINE_NAMES:
    for src in ('A', 'B'):
        flux = fluxes[line]['fluxes'][src]
        flux_err = fluxes[line]['flux_uncerts'][src]
        EBV_val, EBV_err = EBV_cumulative[src]
        corrected, corrected_err = flux_correct(flux, flux_err, line, EBV_val, EBV_err)
        cumulative_results[line][src] = {
            'flux_before': flux,
            'flux_before_err': flux_err,
            'flux_after': corrected,
            'flux_after_err': corrected_err,
        }

# NII6583 has no component breakdown, so it only gets a cumulative correction
NII6583_results = {}
for src in ('A', 'B'):
    EBV_val, EBV_err = EBV_cumulative[src]
    corrected, corrected_err = flux_correct(NII6583_flux[src], NII6583_flux_err[src],
                                            'NII6583', EBV_val, EBV_err)
    NII6583_results[src] = {
        'flux_before': NII6583_flux[src],
        'flux_before_err': NII6583_flux_err[src],
        'flux_after': corrected,
        'flux_after_err': corrected_err,
    }

# ====================
# run while distinguishing outflows
# ====================
EBV_component = {'A': {}, 'B': {}}
for src, wings in WING_NAMES.items():
    for wing in wings:
        bd, bd_err = Halpha_Hbeta_component[src][wing]
        EBV_component[src][wing] = EB_V(bd, bd_err)

component_results = {line: {'A': {}, 'B': {}} for line in LINE_NAMES}
for line in LINE_NAMES:
    for src in ('A', 'B'):
        comp_flux = fluxes[line]['component_fluxes'][src]
        comp_flux_err = fluxes[line]['component_flux_uncerts'][src]
        wing_names = OII_WING_NAMES if line in OII_LINES else WING_NAMES
        wings = wing_names[src][:len(comp_flux)]
        for i, wing in enumerate(wings):
            EBV_val, EBV_err = EBV_component[src][wing]
            corrected, corrected_err = flux_correct(comp_flux[i], comp_flux_err[i], line, EBV_val, EBV_err)
            component_results[line][src][wing] = {
                'flux_before': comp_flux[i],
                'flux_before_err': comp_flux_err[i],
                'flux_after': corrected,
                'flux_after_err': corrected_err,
            }

# ====================
# apply the CUMULATIVE E(B-V) (one well-constrained value per source) to each
# component's individual flux -- as opposed to component_results above, which
# derives a separate (often poorly-constrained) E(B-V) per wing and applies it
# only to that wing's own flux. Uses the same 3-component-B fluxes as
# component_results (jointfit_all, not the two-component-B refit).
# ====================
component_results_cumulativeEBV = {line: {'A': {}, 'B': {}} for line in LINE_NAMES}
for line in LINE_NAMES:
    for src in ('A', 'B'):
        comp_flux = fluxes[line]['component_fluxes'][src]
        comp_flux_err = fluxes[line]['component_flux_uncerts'][src]
        wing_names = OII_WING_NAMES if line in OII_LINES else WING_NAMES
        wings = wing_names[src][:len(comp_flux)]
        EBV_val, EBV_err = EBV_cumulative[src]
        for i, wing in enumerate(wings):
            corrected, corrected_err = flux_correct(comp_flux[i], comp_flux_err[i], line, EBV_val, EBV_err)
            component_results_cumulativeEBV[line][src][wing] = {
                'flux_before': comp_flux[i],
                'flux_before_err': comp_flux_err[i],
                'flux_after': corrected,
                'flux_after_err': corrected_err,
            }

# ====================
# print all before and afters
# ====================
print("=" * 60)
print("Balmer decrement -- cumulative (per source)")
print("NOTE: computed from observed, pre-dust-correction fluxes -- this is")
print("what E(B-V) is derived from, so it cannot itself be dust-corrected.")
print("=" * 60)
for src in ('A', 'B'):
    ab, ab_err = Halpha_Hbeta_cumulative[src]
    gb, gb_err = Hgamma_Hbeta_cumulative[src]
    print(f"Source {src}: Halpha/Hbeta = {ab:.4f} +/- {ab_err:.4f}, "
          f"Hgamma/Hbeta = {gb:.4f} +/- {gb_err:.4f}")

print()
print("=" * 60)
print("Balmer decrement -- component-wise (per source, per wing)")
print("NOTE: computed from observed, pre-dust-correction fluxes -- this is")
print("what E(B-V) is derived from, so it cannot itself be dust-corrected.")
print("=" * 60)
for src, wings in WING_NAMES.items():
    for wing in wings:
        ab, ab_err = Halpha_Hbeta_component[src][wing]
        gb, gb_err = Hgamma_Hbeta_component[src][wing]
        print(f"Source {src} {wing}: Halpha/Hbeta = {ab:.4f} +/- {ab_err:.4f}, "
              f"Hgamma/Hbeta = {gb:.4f} +/- {gb_err:.4f}")

print()
print("=" * 60)
print("E(B-V) -- cumulative (per source)")
print("=" * 60)
for src in ('A', 'B'):
    EBV_val, EBV_err = EBV_cumulative[src]
    print(f"Source {src}: E(B-V) = {EBV_val:.4f} +/- {EBV_err:.4f}")

print()
print("=" * 60)
print("E(B-V) -- component-wise (per source, per wing)")
print("=" * 60)
for src, wings in WING_NAMES.items():
    for wing in wings:
        EBV_val, EBV_err = EBV_component[src][wing]
        print(f"Source {src} {wing}: E(B-V) = {EBV_val:.4f} +/- {EBV_err:.4f}")

print()
print("=" * 60)
print("Cumulative flux correction (before -> after), ergs/s/cm2")
print("=" * 60)
for line in LINE_NAMES:
    for src in ('A', 'B'):
        r = cumulative_results[line][src]
        print(f"{line:10s} source {src}: "
              f"{r['flux_before']:.3f} +/- {r['flux_before_err']:.3f}  ->  "
              f"{r['flux_after']:.3f} +/- {r['flux_after_err']:.3f}")
for src in ('A', 'B'):
    r = NII6583_results[src]
    print(f"{'NII6583':10s} source {src}: "
          f"{r['flux_before']:.3f} +/- {r['flux_before_err']:.3f}  ->  "
          f"{r['flux_after']:.3f} +/- {r['flux_after_err']:.3f}")

print()
print("=" * 60)
print("Component-wise flux correction (before -> after), ergs/s/cm2")
print("=" * 60)
for line in LINE_NAMES:
    for src in ('A', 'B'):
        for wing, r in component_results[line][src].items():
            print(f"{line:10s} source {src} {wing:10s}: "
                  f"{r['flux_before']:.3f} +/- {r['flux_before_err']:.3f}  ->  "
                  f"{r['flux_after']:.3f} +/- {r['flux_after_err']:.3f}")

print()
print("=" * 60)
print("Component-wise flux correction using CUMULATIVE E(B-V) (before -> after), ergs/s/cm2")
print("=" * 60)
for line in LINE_NAMES:
    for src in ('A', 'B'):
        for wing, r in component_results_cumulativeEBV[line][src].items():
            print(f"{line:10s} source {src} {wing:10s}: "
                  f"{r['flux_before']:.3f} +/- {r['flux_before_err']:.3f}  ->  "
                  f"{r['flux_after']:.3f} +/- {r['flux_after_err']:.3f}")

# ====================
# save dust-corrected fluxes
# ====================
os.makedirs(OUTDIR, exist_ok=True)

for line in LINE_NAMES:
    result = {
        'fluxes': {
            src: cumulative_results[line][src]['flux_after']
            for src in ('A', 'B')
        },
        'flux_uncerts': {
            src: cumulative_results[line][src]['flux_after_err']
            for src in ('A', 'B')
        },
        'fluxes_before': {
            src: cumulative_results[line][src]['flux_before']
            for src in ('A', 'B')
        },
        'flux_uncerts_before': {
            src: cumulative_results[line][src]['flux_before_err']
            for src in ('A', 'B')
        },
        'component_fluxes': {
            src: np.array([component_results[line][src][wing]['flux_after']
                           for wing in WING_NAMES[src]
                           if wing in component_results[line][src]])
            for src in ('A', 'B')
        },
        'component_flux_uncerts': {
            src: np.array([component_results[line][src][wing]['flux_after_err']
                           for wing in WING_NAMES[src]
                           if wing in component_results[line][src]])
            for src in ('A', 'B')
        },
        'component_fluxes_before': {
            src: np.array([component_results[line][src][wing]['flux_before']
                           for wing in WING_NAMES[src]
                           if wing in component_results[line][src]])
            for src in ('A', 'B')
        },
        'component_flux_uncerts_before': {
            src: np.array([component_results[line][src][wing]['flux_before_err']
                           for wing in WING_NAMES[src]
                           if wing in component_results[line][src]])
            for src in ('A', 'B')
        },
        'component_fluxes_cumulativeEBV': {
            src: np.array([component_results_cumulativeEBV[line][src][wing]['flux_after']
                           for wing in WING_NAMES[src]
                           if wing in component_results_cumulativeEBV[line][src]])
            for src in ('A', 'B')
        },
        'component_flux_uncerts_cumulativeEBV': {
            src: np.array([component_results_cumulativeEBV[line][src][wing]['flux_after_err']
                           for wing in WING_NAMES[src]
                           if wing in component_results_cumulativeEBV[line][src]])
            for src in ('A', 'B')
        },
        'E(B-V)_cumulative': {src: EBV_cumulative[src][0] for src in ('A', 'B')},
        'E(B-V)_cumulative_err': {src: EBV_cumulative[src][1] for src in ('A', 'B')},
        'E(B-V)_component': {
            src: {wing: EBV_component[src][wing][0] for wing in WING_NAMES[src]}
            for src in ('A', 'B')
        },
        'E(B-V)_component_err': {
            src: {wing: EBV_component[src][wing][1] for wing in WING_NAMES[src]}
            for src in ('A', 'B')
        },
        'Halpha/Hbeta_cumulative': {src: Halpha_Hbeta_cumulative[src][0] for src in ('A', 'B')},
        'Halpha/Hbeta_cumulative_err': {src: Halpha_Hbeta_cumulative[src][1] for src in ('A', 'B')},
        'Hgamma/Hbeta_cumulative': {src: Hgamma_Hbeta_cumulative[src][0] for src in ('A', 'B')},
        'Hgamma/Hbeta_cumulative_err': {src: Hgamma_Hbeta_cumulative[src][1] for src in ('A', 'B')},
        'Halpha/Hbeta_component': {
            src: {wing: Halpha_Hbeta_component[src][wing][0] for wing in WING_NAMES[src]}
            for src in ('A', 'B')
        },
        'Halpha/Hbeta_component_err': {
            src: {wing: Halpha_Hbeta_component[src][wing][1] for wing in WING_NAMES[src]}
            for src in ('A', 'B')
        },
        'Hgamma/Hbeta_component': {
            src: {wing: Hgamma_Hbeta_component[src][wing][0] for wing in WING_NAMES[src]}
            for src in ('A', 'B')
        },
        'Hgamma/Hbeta_component_err': {
            src: {wing: Hgamma_Hbeta_component[src][wing][1] for wing in WING_NAMES[src]}
            for src in ('A', 'B')
        },
        'units': 'ergs / s / cm2',
    }
    with open(f'{OUTDIR}/{line}_dust_corrected.pkl', 'wb') as f:
        dill.dump(result, f)

NII_result = {
    'fluxes': {src: NII6583_results[src]['flux_after'] for src in ('A', 'B')},
    'flux_uncerts': {src: NII6583_results[src]['flux_after_err'] for src in ('A', 'B')},
    'fluxes_before': {src: NII6583_results[src]['flux_before'] for src in ('A', 'B')},
    'flux_uncerts_before': {src: NII6583_results[src]['flux_before_err'] for src in ('A', 'B')},
    'E(B-V)_cumulative': {src: EBV_cumulative[src][0] for src in ('A', 'B')},
    'E(B-V)_cumulative_err': {src: EBV_cumulative[src][1] for src in ('A', 'B')},
    'is_upper_limit': NII6583_is_upper_limit,
    'units': 'ergs / s / cm2',
}
with open(f'{OUTDIR}/NII6583_dust_corrected.pkl', 'wb') as f:
    dill.dump(NII_result, f)

print()
print(f"Saved dust-corrected fluxes for {len(LINE_NAMES)} lines + NII6583 to {OUTDIR}/")
