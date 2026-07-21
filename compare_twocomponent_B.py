"""Compare jointfit_all.py (Source B: red+central+blue, 3 components) against
jointfit_twocomponent_B.py (Source B: core [merged red+central]+blue, 2
components).

Two questions this answers:
  1. Degeneracy: did merging red+central into one free 'core' component
     actually remove the -0.937 amplitude degeneracy jointfit_all.py had
     between them, or did a new core-vs-blue degeneracy appear in its place?
  2. Fit quality: does the simpler (fewer free parameters) two-component-B
     model fit the data comparably well, judged via per-window chi2/AIC/BIC?

Standalone rather than folded into either fitting script: it needs to load
BOTH fits' saved pkls and must not require re-running either slow
TRFLSQFitter(calc_uncertainties=True, maxiter=5000) call. Imports both
jointfit_all and jointfit_twocomponent_B purely so dill.load() can resolve
MeanTie/StdTie/WindowedConst1D (safe -- both scripts guard their actual
fit/save logic behind `if __name__ == "__main__":`).
"""
import csv

import dill
import numpy as np
import astropy.units as u
from astropy.io import fits

import jointfit_all
import legacy_files.python_files.jointfit_twocomponent_B as jointfit_twocomponent_B

OLD_DIR = './output/joint_fit/jointfit_all'
NEW_DIR = './output/joint_fit/twocomponent_B'
COMPARISON_DIR = f'{NEW_DIR}/comparison'

# lines with a real, independent lam_fit/flux_fit/noise_fit array (i.e.
# excludes [NII]6584, which shares Halpha's window/continuum and has no
# separate fit array of its own -- see both scripts' windows['[NII]6584'])
FIT_LINE_ORDER = ['Halpha', '[OIII]5007', '[OIII]4959', 'Hbeta', 'Hgamma', '[OII]']

z_A = jointfit_all.z_A
z_B = jointfit_all.z_B
rest = jointfit_all.rest
mask_ranges = jointfit_all.mask_ranges


# ======================================
# data loading/windowing -- deliberately duplicated from jointfit_all.py /
# jointfit_twocomponent_B.py (identical in both, since neither changes the
# data or window boundaries, only the model). Neither script exposes this as
# an importable function, and no pkl stores the raw per-window arrays, so a
# small bounded copy here is simpler than refactoring either fitting script
# (out of scope for this comparison).
# ======================================
def build_windows():
    spec_lib_nir = "./Data/X-Shooter/1D/stacked_NIR.fits"
    with fits.open(spec_lib_nir) as hdu:
        h = hdu[1].header
        flux_data = hdu[1].data
        noise_data = hdu[4].data
    lam = ((h["CRVAL1"] +
            (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"]) *
           u.Unit(h["CUNIT1"])).to("AA").value
    flux_norm = flux_data / np.nanmedian(flux_data)
    noise_norm = noise_data / np.nanmedian(flux_data)

    spec_lib_vis = "./Data/X-Shooter/1D/stacked_VIS.fits"
    with fits.open(spec_lib_vis) as hdu:
        h_vis = hdu[1].header
        flux_data_vis = hdu[1].data
        noise_data_vis = hdu[4].data
    lam_vis = ((h_vis["CRVAL1"] +
                (np.arange(h_vis["NAXIS1"]) + 1.0 - h_vis["CRPIX1"]) *
                h_vis["CDELT1"]) * u.Unit(h_vis["CUNIT1"])).to("AA").value
    flux_norm_vis = flux_data_vis / np.nanmedian(flux_data_vis)
    noise_norm_vis = noise_data_vis / np.nanmedian(flux_data_vis)

    windows = {}
    for name in ['Halpha', '[OIII]5007', '[OIII]4959', 'Hbeta', 'Hgamma']:
        rest_wl = rest[name]
        if name == 'Halpha':
            lo_obs = rest_wl * (1 + z_B) - 50
            hi_obs = rest['[NII]6584'] * (1 + z_A) + 50
        else:
            center = rest_wl * (1 + z_B)
            lo_obs, hi_obs = center - 50, center + 50
        zoom = np.where((lam >= lo_obs) & (lam <= hi_obs))[0]
        lam_trim, flux_trim, noise_trim = lam[zoom], flux_norm[zoom], noise_norm[zoom]
        bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) | (noise_trim < 0))
        lam_full, flux_full, noise_full = lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]

        fit_bad = np.zeros(len(lam_full), dtype=bool)
        if name in mask_ranges:
            gap_lo, gap_hi, _ = mask_ranges[name]
            fit_bad |= (lam_full >= gap_lo) & (lam_full <= gap_hi)

        windows[name] = {
            'lam_full': lam_full, 'flux_full': flux_full, 'noise_full': noise_full,
            'lam_fit': lam_full[~fit_bad], 'flux_fit': flux_full[~fit_bad],
            'noise_fit': noise_full[~fit_bad],
        }

    oii_lo = rest['[OII]3726'] * (1 + z_B) - 50
    oii_hi = rest['[OII]3729'] * (1 + z_A) + 50
    zoom_oii = np.where((lam_vis >= oii_lo) & (lam_vis <= oii_hi))[0]
    lam_trim, flux_trim, noise_trim = lam_vis[zoom_oii], flux_norm_vis[zoom_oii], noise_norm_vis[zoom_oii]
    bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) | (noise_trim < 0))
    lam_full, flux_full, noise_full = lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]
    windows['[OII]'] = {
        'lam_full': lam_full, 'flux_full': flux_full, 'noise_full': noise_full,
        'lam_fit': lam_full, 'flux_fit': flux_full, 'noise_fit': noise_full,
    }
    return windows


def load_line_pkl(path):
    with open(path, 'rb') as f:
        obj = dill.load(f)
    return obj['model'], obj.get('tie_map', {})


def correlation_matrix(bestfit_model):
    cov = bestfit_model.cov_matrix.cov_matrix
    names = bestfit_model.cov_matrix.param_names
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    idx = {n: k for k, n in enumerate(names)}
    return corr, idx


def chi2_for_window(model, windows, name, comp_indices):
    lam_fit = windows[name]['lam_fit']
    flux_fit = windows[name]['flux_fit']
    noise_fit = windows[name]['noise_fit']
    residual_sigma = (flux_fit - model(lam_fit)) / noise_fit
    chi2 = np.sum(residual_sigma**2)

    n_sigma_window = 3.5
    near_line_mask = np.zeros(len(lam_fit), dtype=bool)
    for i in comp_indices:
        near_line_mask |= (np.abs(lam_fit - model[i].mean.value)
                            <= n_sigma_window * model[i].stddev.value)
    residual_near = residual_sigma[near_line_mask]
    frac_1s = np.mean(np.abs(residual_near) <= 1) if len(residual_near) else np.nan
    frac_2s = np.mean(np.abs(residual_near) <= 2) if len(residual_near) else np.nan
    frac_3s = np.mean(np.abs(residual_near) <= 3) if len(residual_near) else np.nan
    return chi2, len(lam_fit), frac_1s, frac_2s, frac_3s


def k_window(model_module, name, is_halpha_window):
    comp_indices = model_module.A_INDICES[name] + model_module.B_INDICES[name]
    k = len(comp_indices)  # every component's amplitude is always free
    k += 1  # this window's own continuum amplitude
    if is_halpha_window:
        # Halpha's own master components additionally have free mean+stddev
        # (every other line's matching components have those tied, so only
        # amplitude counts there)
        k += 2 * len(model_module.B_INDICES['Halpha']) + 2 * len(model_module.A_INDICES['Halpha'])
    return k


def bic(chi2, k, n):
    return chi2 + k * np.log(n)


def aic(chi2, k):
    return chi2 + 2 * k


def main():
    windows = build_windows()

    print("=" * 72)
    print("DEGENERACY REPORT")
    print("=" * 72)
    print("Old fit (jointfit_all): Source B had 3 free Halpha components")
    print("(red, central, blue). Amplitude correlation red-vs-central was the")
    print("headline degeneracy (-0.937, essentially unresolvable). blue was")
    print("weakly correlated with either (~0.29-0.31). Source A's own")
    print("central-vs-red correlation (0.132) is the 'healthy' benchmark.")
    print()
    print("That exact -0.937 degeneracy is now structurally impossible: the")
    print("two amplitude parameters that were correlated (old B_red, B_central)")
    print("no longer both exist -- they were replaced by one merged parameter.")
    print("What matters now is whether a NEW core-vs-blue degeneracy appeared.")
    print()

    old_model_ha, _ = load_line_pkl(f'{OLD_DIR}/Halpha_joint_tied_fit.pkl')
    new_model_ha, _ = load_line_pkl(f'{NEW_DIR}/Halpha_joint_tied_fit.pkl')

    old_corr, old_idx = correlation_matrix(old_model_ha)
    new_corr, new_idx = correlation_matrix(new_model_ha)

    old_red, old_central, old_blue = jointfit_all.B_INDICES['Halpha']
    new_core, new_blue = jointfit_twocomponent_B.B_INDICES['Halpha']

    print(f"[Halpha] old B_red-vs-B_central amplitude corr:  "
          f"{old_corr[old_idx[f'amplitude_{old_red}'], old_idx[f'amplitude_{old_central}']]:+.3f}")
    print(f"[Halpha] old B_red-vs-B_blue amplitude corr:     "
          f"{old_corr[old_idx[f'amplitude_{old_red}'], old_idx[f'amplitude_{old_blue}']]:+.3f}")
    print(f"[Halpha] old B_central-vs-B_blue amplitude corr: "
          f"{old_corr[old_idx[f'amplitude_{old_central}'], old_idx[f'amplitude_{old_blue}']]:+.3f}")
    headline = new_corr[new_idx[f'amplitude_{new_core}'], new_idx[f'amplitude_{new_blue}']]
    print(f"[Halpha] NEW B_core-vs-B_blue amplitude corr:    {headline:+.3f}   "
          f"<-- headline number (benchmark: A central/red = 0.132)")
    print()

    print("[Halpha] full core-vs-blue correlation block (mean/stddev/amplitude):")
    for p_core in ['mean', 'stddev', 'amplitude']:
        row = []
        for p_blue in ['mean', 'stddev', 'amplitude']:
            v = new_corr[new_idx[f'{p_core}_{new_core}'], new_idx[f'{p_blue}_{new_blue}']]
            row.append(f"{p_blue}_blue={v:+.3f}")
        print(f"  {p_core}_core: " + ", ".join(row))
    print()

    degeneracy_rows = []
    for name in ['[OIII]5007', '[OIII]4959', 'Hbeta', 'Hgamma']:
        old_model, _ = load_line_pkl(f'{OLD_DIR}/{name}_joint_tied_fit.pkl')
        new_model, _ = load_line_pkl(f'{NEW_DIR}/{name}_joint_tied_fit.pkl')
        o_corr, o_idx = correlation_matrix(old_model)
        n_corr, n_idx = correlation_matrix(new_model)
        o_red, o_central, o_blue = jointfit_all.B_INDICES[name]
        n_core, n_blue = jointfit_twocomponent_B.B_INDICES[name]
        old_rc = o_corr[o_idx[f'amplitude_{o_red}'], o_idx[f'amplitude_{o_central}']]
        new_cb = n_corr[n_idx[f'amplitude_{n_core}'], n_idx[f'amplitude_{n_blue}']]
        print(f"[{name}] old red-vs-central amp corr: {old_rc:+.3f}   "
              f"NEW core-vs-blue amp corr: {new_cb:+.3f}")
        degeneracy_rows.append((name, old_rc, new_cb))

    # bonus: [OII]'s B-side had the same red/central split in the old fit --
    # check whether it had the same degeneracy, now resolved the same way
    old_oii, _ = load_line_pkl(f'{OLD_DIR}/[OII]_joint_tied_fit.pkl')
    o_corr, o_idx = correlation_matrix(old_oii)
    print()
    for label, (red_i, central_i) in [
        ('[OII]3726_B', (jointfit_all.B_INDICES['[OII]'][0], jointfit_all.B_INDICES['[OII]'][1])),
        ('[OII]3729_B', (jointfit_all.B_INDICES['[OII]'][2], jointfit_all.B_INDICES['[OII]'][3])),
    ]:
        c = o_corr[o_idx[f'amplitude_{red_i}'], o_idx[f'amplitude_{central_i}']]
        print(f"[{label}] old red-vs-central amp corr (bonus, now collapsed to 1 component): {c:+.3f}")
    print()

    print("=" * 72)
    print("FIT-QUALITY COMPARISON (per window, using lam_fit/flux_fit/noise_fit --")
    print("the arrays the fitter actually saw, NOT lam_full/flux_full which the")
    print("per-line prints inside jointfit_all.py/jointfit_twocomponent_B.py use")
    print("for display and include masked-out points)")
    print("=" * 72)

    summary_rows = []
    for name in FIT_LINE_ORDER:
        old_model, _ = load_line_pkl(f'{OLD_DIR}/{name}_joint_tied_fit.pkl')
        new_model, _ = load_line_pkl(f'{NEW_DIR}/{name}_joint_tied_fit.pkl')

        old_comp = jointfit_all.A_INDICES[name] + jointfit_all.B_INDICES[name]
        new_comp = jointfit_twocomponent_B.A_INDICES[name] + jointfit_twocomponent_B.B_INDICES[name]

        chi2_old, n_old, f1_old, f2_old, f3_old = chi2_for_window(old_model, windows, name, old_comp)
        chi2_new, n_new, f1_new, f2_new, f3_new = chi2_for_window(new_model, windows, name, new_comp)
        assert n_old == n_new, "fit and comparison windows must match in point count"
        n = n_old

        is_ha = (name == 'Halpha')
        k_old = k_window(jointfit_all, name, is_ha)
        k_new = k_window(jointfit_twocomponent_B, name, is_ha)

        bic_old, bic_new = bic(chi2_old, k_old, n), bic(chi2_new, k_new, n)
        aic_old, aic_new = aic(chi2_old, k_old), aic(chi2_new, k_new)

        d_chi2 = chi2_new - chi2_old
        d_bic = bic_new - bic_old
        d_aic = aic_new - aic_old

        print(f"\n[{name}]  n={n}  k_old={k_old} -> k_new={k_new}  (Δk={k_new - k_old})")
        print(f"  chi2:         old={chi2_old:8.1f} (red={chi2_old/(n-k_old):.2f})   "
              f"new={chi2_new:8.1f} (red={chi2_new/(n-k_new):.2f})   Δchi2={d_chi2:+.1f}")
        print(f"  AIC:          old={aic_old:8.1f}   new={aic_new:8.1f}   ΔAIC={d_aic:+.1f}")
        print(f"  BIC:          old={bic_old:8.1f}   new={bic_new:8.1f}   ΔBIC={d_bic:+.1f}")
        print(f"  within 1/2/3σ (old): {f1_old:.1%}/{f2_old:.1%}/{f3_old:.1%}   "
              f"(new): {f1_new:.1%}/{f2_new:.1%}/{f3_new:.1%}")

        summary_rows.append({
            'line': name, 'n': n, 'k_old': k_old, 'k_new': k_new,
            'chi2_old': chi2_old, 'chi2_new': chi2_new, 'd_chi2': d_chi2,
            'aic_old': aic_old, 'aic_new': aic_new, 'd_aic': d_aic,
            'bic_old': bic_old, 'bic_new': bic_new, 'd_bic': d_bic,
            'frac1s_old': f1_old, 'frac1s_new': f1_new,
            'frac2s_old': f2_old, 'frac2s_new': f2_new,
            'frac3s_old': f3_old, 'frac3s_new': f3_new,
        })

    print()
    print("ΔBIC interpretation guide: |ΔBIC|<2 inconclusive, 2-6 positive, "
          "6-10 strong, >10 very strong evidence for the lower-BIC model.")
    print("(negative ΔBIC/ΔAIC favors the new two-component-B fit)")

    import os
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    with open(f'{COMPARISON_DIR}/summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    degeneracy_summary = {
        'halpha_old_red_central': old_corr[old_idx[f'amplitude_{old_red}'], old_idx[f'amplitude_{old_central}']],
        'halpha_new_core_blue': headline,
        'per_line_old_red_central': {r[0]: r[1] for r in degeneracy_rows},
        'per_line_new_core_blue': {r[0]: r[2] for r in degeneracy_rows},
    }
    with open(f'{COMPARISON_DIR}/degeneracy_summary.pkl', 'wb') as f:
        dill.dump(degeneracy_summary, f)

    print(f"\nSaved comparison summary to {COMPARISON_DIR}/summary.csv "
          f"and {COMPARISON_DIR}/degeneracy_summary.pkl")


if __name__ == "__main__":
    main()
