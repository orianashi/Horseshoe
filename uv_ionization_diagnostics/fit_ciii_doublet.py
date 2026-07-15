import os

import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.modeling import models, fitting
from functools import reduce
import operator

from fit_uv_lines import (rest, MeanTie, WindowedConst1D, STDDEV_BOUNDS,
                           STDDEV_BOUND_ATOL, AMPLITUDE_COLLAPSE_ATOL,
                           WINDOW_HALFWIDTH, load_arm, make_window, plot_group,
                           z_A, SQRT2PI)

plt.ion()

# source B, reinstated here only (removed from the main SOURCES={'A':...}
# pipeline in fit_uv_lines.py) purely to fit its own CIII]1907/1909 pair --
# same repo-wide constant used everywhere else
z_B = 1.677

# amp(1907)/amp(1909) -- user-specified physical bound for this doublet
AMP_RATIO_BOUNDS = (1.2, 1.6)

UV_DIAGDIR = os.path.dirname(os.path.abspath(__file__))
FITDIR = f'{UV_DIAGDIR}/output/fits'
FLUXDIR = f'{UV_DIAGDIR}/output/fluxes'
CIIIDIR = f'{FITDIR}/CIII_doublet'
os.makedirs(CIIIDIR, exist_ok=True)


def _stddev_at_bound(std):
    lo, hi = STDDEV_BOUNDS
    return (std - lo) < STDDEV_BOUND_ATOL or (hi - std) < STDDEV_BOUND_ATOL


def _at_edge(value, lo, hi, atol=1e-3):
    return (value - lo) < atol or (hi - value) < atol


class StdTie:
    """Ties one component's stddev, live, to another component's stddev
    within the SAME final joint-fit model -- the width-tying half of the
    OII-doublet technique (oii_gaussian_fitting.py's create_std_tie),
    distinct from freezing it at whatever the separate wing fit measured."""

    def __init__(self, ref_idx):
        self.ref_idx = ref_idx

    def __call__(self, model):
        return getattr(model, f'stddev_{self.ref_idx}')


def flux_and_uncert(bestfit_model, index, stddev_tie_ref=None):
    """Same covariance-propagated flux/uncertainty as fit_uv_lines.py's
    version, extended to handle a component whose stddev is LIVE-tied
    (StdTie above, ratio exactly 1.0) to another component's stddev within
    this same model: a tied parameter isn't itself a free parameter (so it's
    absent from cov_matrix.param_names), but its exact derivative w.r.t. the
    referenced free stddev is 1.0, so that contribution is added to the
    reference's gradient slot instead of being silently dropped."""
    names = bestfit_model.cov_matrix.param_names
    cov = bestfit_model.cov_matrix.cov_matrix
    idx = {n: k for k, n in enumerate(names)}

    amp = getattr(bestfit_model, f'amplitude_{index}').value
    std = getattr(bestfit_model, f'stddev_{index}').value
    total = amp * std * SQRT2PI

    grad = np.zeros(len(names))
    if f'amplitude_{index}' in idx:
        grad[idx[f'amplitude_{index}']] += std * SQRT2PI
    if stddev_tie_ref is not None:
        grad[idx[f'stddev_{stddev_tie_ref}']] += amp * SQRT2PI
    elif f'stddev_{index}' in idx:
        grad[idx[f'stddev_{index}']] += amp * SQRT2PI

    uncert = np.sqrt(grad @ cov @ grad)
    return total, uncert


REST_1907 = rest['[CIII]1907']
REST_1909 = rest['[CIII]1909']
REST_HEII = rest['HeII1640']

OBS = {
    ('1907', 'A'): REST_1907 * (1 + z_A),
    ('1907', 'B'): REST_1907 * (1 + z_B),
    ('1909', 'A'): REST_1909 * (1 + z_A),
    ('1909', 'B'): REST_1909 * (1 + z_B),
}
# blue-to-red order: 1907_B < 1907_A < 1909_B < 1909_A -- same "outer
# anchors / inner blended pair" structure as the OII doublet, just
# compressed into a much smaller wavelength span (see plan/context).
ANCHOR_1909_A = OBS[('1909', 'A')]   # reddest overall -- unblended anchor, source A
ANCHOR_1907_B = OBS[('1907', 'B')]   # bluest overall -- unblended anchor, source B
BLENDED_1907_A = OBS[('1907', 'A')]  # bluer of the two middle/blended components
BLENDED_1909_B = OBS[('1909', 'B')]  # redder of the two middle/blended components


def amp_guess(lam, flux, center, halfwidth=15, default=0.1):
    nearby = np.abs(lam - center) < halfwidth
    if not nearby.any():
        return default
    return max(np.nanmax(flux[nearby]) - np.nanmedian(flux), 0.05)


if __name__ == "__main__":
    lam_uvb, flux_uvb, noise_uvb, norm_uvb = load_arm('UVB')

    # ==================
    # stage 1: wing fit -- fit ONLY the two unblended anchors (1909_A,
    # 1907_B), on data with the central blend region (a 3 AA buffer around
    # the two middle/blended components' expected peaks) masked out.
    # Mirrors oii_gaussian_fitting.py:82-127 exactly.
    # ==================
    ciii_lo = min(OBS.values()) - WINDOW_HALFWIDTH
    ciii_hi = max(OBS.values()) + WINDOW_HALFWIDTH
    lam_w, flux_w, noise_w = make_window(lam_uvb, flux_uvb, noise_uvb, ciii_lo, ciii_hi)

    blend_lo = BLENDED_1907_A - 3
    blend_hi = BLENDED_1909_B + 3
    unblended = (lam_w < blend_lo) | (lam_w > blend_hi)
    lam_wing, flux_wing, noise_wing = lam_w[unblended], flux_w[unblended], noise_w[unblended]

    g_1909A_wing = models.Gaussian1D(name='[CIII]1909_A_wing', mean=ANCHOR_1909_A,
                                      amplitude=amp_guess(lam_wing, flux_wing, ANCHOR_1909_A),
                                      stddev=0.5)
    g_1909A_wing.mean.bounds = (ANCHOR_1909_A - 10, ANCHOR_1909_A + 10)

    g_1907B_wing = models.Gaussian1D(name='[CIII]1907_B_wing', mean=ANCHOR_1907_B,
                                      amplitude=amp_guess(lam_wing, flux_wing, ANCHOR_1907_B),
                                      stddev=0.5)
    g_1907B_wing.mean.bounds = (ANCHOR_1907_B - 10, ANCHOR_1907_B + 10)

    cont_wing = models.Const1D(amplitude=np.nanmedian(flux_wing), name='continuum_wing')

    wing_model = g_1909A_wing + g_1907B_wing + cont_wing
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    wing_bestfit = fitter(wing_model, lam_wing, flux_wing, weights=1.0 / noise_wing, maxiter=5000)
    wing_converged = fitter.fit_info.get('success', True)

    wing_amp_1909A = wing_bestfit.amplitude_0.value
    wing_amp_1907B = wing_bestfit.amplitude_1.value
    print(f"Wing fit converged={wing_converged}  amp(1909_A)={wing_amp_1909A:.4f}  "
          f"amp(1907_B)={wing_amp_1907B:.4f}")

    lam_model = np.linspace(lam_w[0], lam_w[-1], 5000)
    fig, ax = plt.subplots(nrows=2, height_ratios=[3, 1], sharex=True, figsize=(11, 8))
    ax[0].plot(lam_w, flux_w, c='black', label='data', ds='steps')
    ax[0].fill_between(lam_w, flux_w - noise_w, flux_w + noise_w, color='dimgray', alpha=0.6, label='noise')
    ax[0].axvspan(blend_lo, blend_hi, color='yellow', alpha=0.2, label='blend region (excluded)')
    ax[0].plot(lam_model, wing_bestfit(lam_model), color='orange', label='wing fit', lw=2)
    ax[0].plot(lam_model, wing_bestfit[0](lam_model), color='blue', ls='--', label='[CIII]1909_A (anchor)')
    ax[0].plot(lam_model, wing_bestfit[1](lam_model), color='red', ls='--', label='[CIII]1907_B (anchor)')
    ax[0].set_xlabel('Observed Wavelength [Angstroms]', fontsize=13)
    ax[0].set_ylabel('Normalised Flux', fontsize=13)
    ax[0].set_title(f'CIII doublet wing fit  |  z_A={z_A}  z_B={z_B}', fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)
    residual_wing = (flux_wing - wing_bestfit(lam_wing)) / noise_wing
    ax[1].scatter(lam_wing, residual_wing, s=10, c='orange', alpha=0.5, label='(flux-model)/noise, wing only')
    ax[1].axhline(0, ls='--', alpha=0.4)
    ax[1].legend(frameon=True, fontsize=8)
    fig.savefig(f'{CIIIDIR}/CIII_wingfit.png')
    plt.close(fig)
    with open(f'{CIIIDIR}/CIII_wingfit.pkl', 'wb') as f:
        dill.dump(wing_bestfit, f)

    # ==================
    # stage 2: full joint fit -- HeII1640_A/B (free masters, needed as this
    # pipeline's mean-tying anchor since source B was removed from
    # fit_uv_lines.py) + all 4 CIII components over the UNMASKED window.
    # ==================
    heii_lo = min(REST_HEII * (1 + z_A), REST_HEII * (1 + z_B)) - WINDOW_HALFWIDTH
    heii_hi = max(REST_HEII * (1 + z_A), REST_HEII * (1 + z_B)) + WINDOW_HALFWIDTH
    lam_heii, flux_heii, noise_heii = make_window(lam_uvb, flux_uvb, noise_uvb, heii_lo, heii_hi)

    gaussians = []

    for z, source in [(z_A, 'A'), (z_B, 'B')]:
        mean_guess = REST_HEII * (1 + z)
        g = models.Gaussian1D(name=f'HeII1640_{source}', mean=mean_guess,
                               amplitude=amp_guess(lam_heii, flux_heii, mean_guess, default=0.1),
                               stddev=0.5)
        g.mean.bounds = (mean_guess - 5, mean_guess + 5)
        g.stddev.bounds = STDDEV_BOUNDS
        g.amplitude.bounds = (0, None)
        gaussians.append(g)
    HEII_A_IDX, HEII_B_IDX = 0, 1

    # anchor: 1909_A, mean tied to HeII_A, free amplitude/stddev
    g_1909A = models.Gaussian1D(name='[CIII]1909_A', mean=1.0,
                                 amplitude=max(wing_amp_1909A, 0.05), stddev=0.5)
    g_1909A.mean.tied = MeanTie(HEII_A_IDX, REST_1909 / REST_HEII)
    g_1909A.stddev.bounds = STDDEV_BOUNDS
    g_1909A.amplitude.bounds = (0, None)
    gaussians.append(g_1909A)
    ANCHOR_1909A_IDX = len(gaussians) - 1

    # anchor: 1907_B, mean tied to HeII_B, free amplitude/stddev
    g_1907B = models.Gaussian1D(name='[CIII]1907_B', mean=1.0,
                                 amplitude=max(wing_amp_1907B, 0.05), stddev=0.5)
    g_1907B.mean.tied = MeanTie(HEII_B_IDX, REST_1907 / REST_HEII)
    g_1907B.stddev.bounds = STDDEV_BOUNDS
    g_1907B.amplitude.bounds = (0, None)
    gaussians.append(g_1907B)
    ANCHOR_1907B_IDX = len(gaussians) - 1

    # blended sibling: 1907_A -- amplitude hard-bounded from the wing fit's
    # 1909_A anchor via the user-specified ratio, stddev tied live to 1909_A
    amp_lo_1907A = AMP_RATIO_BOUNDS[0] * wing_amp_1909A
    amp_hi_1907A = AMP_RATIO_BOUNDS[1] * wing_amp_1909A
    g_1907A = models.Gaussian1D(name='[CIII]1907_A', mean=1.0,
                                 amplitude=0.5 * (amp_lo_1907A + amp_hi_1907A), stddev=0.5)
    g_1907A.mean.tied = MeanTie(HEII_A_IDX, REST_1907 / REST_HEII)
    g_1907A.amplitude.bounds = (amp_lo_1907A, amp_hi_1907A)
    g_1907A.stddev.tied = StdTie(ANCHOR_1909A_IDX)
    gaussians.append(g_1907A)
    SIBLING_1907A_IDX = len(gaussians) - 1

    # blended sibling: 1909_B -- amplitude hard-bounded from the wing fit's
    # 1907_B anchor via the (inverted) ratio, stddev tied live to 1907_B
    amp_lo_1909B = wing_amp_1907B / AMP_RATIO_BOUNDS[1]
    amp_hi_1909B = wing_amp_1907B / AMP_RATIO_BOUNDS[0]
    g_1909B = models.Gaussian1D(name='[CIII]1909_B', mean=1.0,
                                 amplitude=0.5 * (amp_lo_1909B + amp_hi_1909B), stddev=0.5)
    g_1909B.mean.tied = MeanTie(HEII_B_IDX, REST_1909 / REST_HEII)
    g_1909B.amplitude.bounds = (amp_lo_1909B, amp_hi_1909B)
    g_1909B.stddev.tied = StdTie(ANCHOR_1907B_IDX)
    gaussians.append(g_1909B)
    SIBLING_1909B_IDX = len(gaussians) - 1

    cont_heii = WindowedConst1D(amplitude=np.nanmedian(flux_heii), lo=lam_heii.min(),
                                 hi=lam_heii.max(), name='continuum_HeII1640')
    cont_heii.lo.fixed = True
    cont_heii.hi.fixed = True
    cont_ciii = WindowedConst1D(amplitude=np.nanmedian(flux_w), lo=lam_w.min(),
                                 hi=lam_w.max(), name='continuum_CIII')
    cont_ciii.lo.fixed = True
    cont_ciii.hi.fixed = True
    continua = [cont_heii, cont_ciii]
    CONT_HEII_IDX = len(gaussians)
    CONT_CIII_IDX = len(gaussians) + 1

    lam_all = np.concatenate([lam_heii, lam_w])
    flux_all = np.concatenate([flux_heii, flux_w])
    noise_all = np.concatenate([noise_heii, noise_w])

    compound_model = reduce(operator.add, gaussians + continua)
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    bestfit_model = fitter(compound_model, lam_all, flux_all, weights=1.0 / noise_all, maxiter=5000)
    converged = fitter.fit_info.get('success', True)

    # ==================
    # detection logic
    # ==================
    def anchor_is_nondetection(idx):
        amp = getattr(bestfit_model, f'amplitude_{idx}').value
        std = getattr(bestfit_model, f'stddev_{idx}').value
        return (not converged) or amp < AMPLITUDE_COLLAPSE_ATOL or _stddev_at_bound(std)

    def sibling_is_nondetection(idx, bound_lo, bound_hi, anchor_nondetection):
        amp = getattr(bestfit_model, f'amplitude_{idx}').value
        return (not converged) or anchor_nondetection or _at_edge(amp, bound_lo, bound_hi)

    nondet_1909A = anchor_is_nondetection(ANCHOR_1909A_IDX)
    nondet_1907B = anchor_is_nondetection(ANCHOR_1907B_IDX)
    nondet_1907A = sibling_is_nondetection(SIBLING_1907A_IDX, amp_lo_1907A, amp_hi_1907A, nondet_1909A)
    nondet_1909B = sibling_is_nondetection(SIBLING_1909B_IDX, amp_lo_1909B, amp_hi_1909B, nondet_1907B)

    print("Amplitude ratios (sanity check against "
          f"{AMP_RATIO_BOUNDS}):")
    ratio_A = (getattr(bestfit_model, f'amplitude_{SIBLING_1907A_IDX}').value /
               getattr(bestfit_model, f'amplitude_{ANCHOR_1909A_IDX}').value)
    ratio_B = (getattr(bestfit_model, f'amplitude_{ANCHOR_1907B_IDX}').value /
               getattr(bestfit_model, f'amplitude_{SIBLING_1909B_IDX}').value)
    print(f"  source A: amp(1907_A)/amp(1909_A) = {ratio_A:.3f}")
    print(f"  source B: amp(1907_B)/amp(1909_B) = {ratio_B:.3f}")

    # ==================
    # flux -- SOURCE A ONLY, per plan: source B's components are fit (for
    # the width/ratio constraints above to work) but never integrated or
    # added to any table.
    # ==================
    def resolve_flux(idx, rest_wl, is_nondet, stddev_tie_ref=None):
        if is_nondet:
            local_noise = np.nanmedian(noise_w)
            heii_std_A = getattr(bestfit_model, f'stddev_{HEII_A_IDX}').value
            assumed_std = heii_std_A * (rest_wl / REST_HEII)
            amp_3sigma = 3 * local_noise
            flux_norm = amp_3sigma * assumed_std * SQRT2PI
            flux_uncert_norm = np.nan
            stddev_out = assumed_std
        else:
            flux_norm, flux_uncert_norm = flux_and_uncert(bestfit_model, idx, stddev_tie_ref)
            stddev_out = getattr(bestfit_model, f'stddev_{idx}').value
        mean_out = getattr(bestfit_model, f'mean_{idx}').value
        return flux_norm * norm_uvb, flux_uncert_norm * norm_uvb, mean_out, stddev_out

    flux_1907A, flux_uncert_1907A, mean_1907A, stddev_1907A = resolve_flux(
        SIBLING_1907A_IDX, REST_1907, nondet_1907A, stddev_tie_ref=ANCHOR_1909A_IDX)
    flux_1909A, flux_uncert_1909A, mean_1909A, stddev_1909A = resolve_flux(
        ANCHOR_1909A_IDX, REST_1909, nondet_1909A)

    print(f"[CIII]1907  A  flux = {flux_1907A:.4g} +/- {flux_uncert_1907A:.4g} erg/s/cm2"
          f"{'  [3-sigma UPPER LIMIT]' if nondet_1907A else ''}")
    print(f"[CIII]1909  A  flux = {flux_1909A:.4g} +/- {flux_uncert_1909A:.4g} erg/s/cm2"
          f"{'  [3-sigma UPPER LIMIT]' if nondet_1909A else ''}")
    print(f"[CIII]1907  B  (fit only, not integrated/tabulated): "
          f"amplitude={getattr(bestfit_model, f'amplitude_{ANCHOR_1907B_IDX}').value:.4f}"
          f"{'  [non-detection]' if nondet_1907B else ''}")
    print(f"[CIII]1909  B  (fit only, not integrated/tabulated): "
          f"amplitude={getattr(bestfit_model, f'amplitude_{SIBLING_1909B_IDX}').value:.4f}"
          f"{'  [non-detection]' if nondet_1909B else ''}")

    # ==================
    # full-fit plot -- all 4 components (A and B) shown for visual
    # confirmation, even though only A's are tabulated
    # ==================
    comp_indices = [ANCHOR_1909A_IDX, ANCHOR_1907B_IDX, SIBLING_1907A_IDX, SIBLING_1909B_IDX]
    comp_names = ['[CIII]1909_A', '[CIII]1907_B', '[CIII]1907_A', '[CIII]1909_B']
    plot_group(lam_w, flux_w, noise_w, bestfit_model, comp_indices, comp_names,
               f'CIII doublet full fit (both sources)  |  z_A={z_A}  z_B={z_B}',
               f'{CIIIDIR}/CIII_doublet_fullfit.png')
    with open(f'{CIIIDIR}/CIII_doublet_fullfit.pkl', 'wb') as f:
        dill.dump({'model': bestfit_model, 'is_upper_limit': {
            '1907_A': nondet_1907A, '1909_A': nondet_1909A,
            '1907_B': nondet_1907B, '1909_B': nondet_1909B,
        }}, f)

    # ==================
    # update the main flux table in place -- source A's [CIII]1907/1909
    # rows only
    # ==================
    with open(f'{FLUXDIR}/uv_line_fluxes.pkl', 'rb') as f:
        rows = dill.load(f)

    new_rows = []
    for r in rows:
        if r['line'] == '[CIII]1907' and r['source'] == 'A':
            new_rows.append({'line': '[CIII]1907', 'source': 'A', 'mean_obs_AA': mean_1907A,
                              'stddev_obs_AA': stddev_1907A, 'flux': flux_1907A,
                              'flux_uncert': flux_uncert_1907A, 'is_upper_limit': nondet_1907A,
                              'units': r['units']})
        elif r['line'] == '[CIII]1909' and r['source'] == 'A':
            new_rows.append({'line': '[CIII]1909', 'source': 'A', 'mean_obs_AA': mean_1909A,
                              'stddev_obs_AA': stddev_1909A, 'flux': flux_1909A,
                              'flux_uncert': flux_uncert_1909A, 'is_upper_limit': nondet_1909A,
                              'units': r['units']})
        else:
            new_rows.append(r)

    df = pd.DataFrame(new_rows)
    df.to_csv(f'{FLUXDIR}/uv_line_fluxes.csv', index=False)
    with open(f'{FLUXDIR}/uv_line_fluxes.pkl', 'wb') as f:
        dill.dump(new_rows, f)

    print(f"\nUpdated {FLUXDIR}/uv_line_fluxes.csv with the improved [CIII]1907_A/1909_A fit "
          "(source B not added).")
