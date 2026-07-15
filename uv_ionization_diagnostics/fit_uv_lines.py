import os

import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting, custom_model
from specutils.utils.wcs_utils import air_to_vac
from functools import reduce
import operator

plt.ion()

# ==================
# redshifts (same constants used throughout the repo). Source B is excluded
# from this pipeline -- only source A is fit.
# ==================
z_A = 1.679
SOURCES = {'A': z_A}

# ==================
# rest-frame vacuum wavelengths (Angstroms) -- see
# absorption_wls/horseshoe_atoms.dat for the same values. [CII]2325 is
# anchored at 2324.69 AA, the strongest of the blend's 5 components (K19
# Table 1 footnote b), and fit here as a single Gaussian representing the
# whole blend.
# ==================
rest = {
    'HeII1640': 1640.42,
    '[SiII]1808': 1808.0126,
    '[SiIII]1883': 1883.00,
    '[SiIII]1892': 1892.03,
    '[CIII]1907': 1906.68,
    '[CIII]1909': 1908.734,
    '[CII]2325': 2324.69,
}

# ==================
# UVB line groups: lines whose +-50 AA observed-frame windows (for EITHER
# source) would overlap are fit within one SHARED window/continuum instead
# of separate overlapping ones -- same rationale as jointfit_all.py's
# combined '[OII]' window. HeII1640 (~4392 AA obs) and [SiII]1808 (~4842 AA)
# are isolated from each other and from everything else by >100 AA, but
# [SiIII]1883/1892 and [CIII]1907/1909 all land within ~5040-5115 AA
# (a ~75 AA span across both sources), so their individual +-50 AA windows
# heavily overlap and must share one window.
# ==================
UVB_GROUPS = [
    ('HeII1640', ['HeII1640']),
    ('[SiII]1808', ['[SiII]1808']),
    ('SiIII_CIII', ['[SiIII]1883', '[SiIII]1892', '[CIII]1907', '[CIII]1909']),
]
UVB_LINES = [line for _, lines in UVB_GROUPS for line in lines]

WINDOW_HALFWIDTH = 50  # AA, observed frame -- matches the fit-window convention used everywhere else in this codebase
AMPLITUDE_COLLAPSE_ATOL = 1e-3  # amplitude this close to its 0 lower bound counts as "collapsed" (non-detection)
# velocity-motivated stddev bound (~150 km/s at these observed wavelengths --
# comparable to the ~50-85 km/s bounds jointfit_all.py uses for Halpha's
# central/wing components). A free amplitude+stddev fit to a weak/absent line
# can trade a vanishing amplitude for a ballooning stddev to fit noise
# wiggles instead of collapsing to zero -- bounding stddev keeps that
# degenerate solution from masquerading as a real, if broad, detection.
STDDEV_BOUNDS = (0.15, 2.5)
STDDEV_BOUND_ATOL = 1e-3  # stddev this close to either bound signals the fit didn't converge to an interior optimum

OUTDIR = os.path.dirname(os.path.abspath(__file__)) + '/output'
FITDIR = f'{OUTDIR}/fits'
FLUXDIR = f'{OUTDIR}/fluxes'
os.makedirs(FLUXDIR, exist_ok=True)

SQRT2PI = np.sqrt(2 * np.pi)


# ==================
# picklable tying class (not a closure) -- dill's by-value pickling of
# nested closures can silently corrupt their reconstructed bytecode, see
# jointfit_all.py's MeanTie for the same pattern/rationale.
# ==================
class MeanTie:

    def __init__(self, ref_idx, line_ratio):
        self.ref_idx = ref_idx
        self.line_ratio = line_ratio

    def __call__(self, model):
        return getattr(model, f'mean_{self.ref_idx}') * self.line_ratio


# ==================
# windowed continuum: a plain Const1D would leak into every window once
# evaluated over the concatenated x array, so restrict each window's
# continuum to its own range (same as jointfit_all.py)
# ==================
def _windowed_const(x, amplitude=1.0, lo=0.0, hi=1.0):
    return np.where((x >= lo) & (x <= hi), amplitude, 0.0)


WindowedConst1D = custom_model(_windowed_const)


# ==================
# data loading
# ==================
def load_arm(arm):
    """Load a stacked X-Shooter 1D arm, convert its wavelength axis to
    vacuum Angstroms (specutils air_to_vac -- X-Shooter arms are calibrated
    in air), and normalize flux/noise by the arm's own nanmedian(flux).
    Returns lam (vacuum AA), flux_norm, noise_norm, and norm_const (the
    nanmedian flux used to normalize -- needed to convert normalized fluxes
    from different arms back to a common physical scale, since [CIII]/[CII]
    is a cross-arm ratio and the two arms' normalization constants don't
    otherwise cancel)."""
    path = os.path.dirname(os.path.abspath(__file__)) + f'/../Data/X-Shooter/1D/stacked_{arm}.fits'
    with fits.open(path) as hdu:
        h = hdu[1].header
        flux_data = hdu[1].data
        noise_data = hdu[4].data
    lam_air = ((h["CRVAL1"] +
                (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"]) *
               u.Unit(h["CUNIT1"])).to("AA")
    lam_vac = air_to_vac(lam_air).value

    norm_const = np.nanmedian(flux_data)
    flux_norm = flux_data / norm_const
    noise_norm = noise_data / norm_const
    return lam_vac, flux_norm, noise_norm, norm_const


def make_window(lam, flux_norm, noise_norm, lo_obs, hi_obs):
    zoom = np.where((lam >= lo_obs) & (lam <= hi_obs))[0]
    lam_trim = lam[zoom]
    flux_trim = flux_norm[zoom]
    noise_trim = noise_norm[zoom]
    bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) |
           (noise_trim < 0))
    return lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]


def group_window_bounds(lines_in_group, halfwidth=WINDOW_HALFWIDTH):
    """One window spanning every line in the group at BOTH sources'
    redshifts, so overlapping lines/sources are fit exactly once instead of
    each pulling its own overlapping copy of the same pixels."""
    centers = [rest[line] * (1 + z) for line in lines_in_group for z in SOURCES.values()]
    return min(centers) - halfwidth, max(centers) + halfwidth


# ==================
# flux + covariance-propagated uncertainty for one Gaussian component
# (adapted from multiple_gaussian_integration.py's flux_and_uncert; no
# tie_map handling needed here since the only tied parameters -- means --
# are baked to fixed/derived constants before this is ever called, so
# they carry zero variance and correctly contribute nothing)
# ==================
def _stddev_at_bound(std):
    lo, hi = STDDEV_BOUNDS
    return (std - lo) < STDDEV_BOUND_ATOL or (hi - std) < STDDEV_BOUND_ATOL


def flux_and_uncert(bestfit_model, index):
    names = bestfit_model.cov_matrix.param_names
    cov = bestfit_model.cov_matrix.cov_matrix
    idx = {n: k for k, n in enumerate(names)}

    amp = getattr(bestfit_model, f'amplitude_{index}').value
    std = getattr(bestfit_model, f'stddev_{index}').value
    total = amp * std * SQRT2PI

    grad = np.zeros(len(names))
    if f'amplitude_{index}' in idx:
        grad[idx[f'amplitude_{index}']] += std * SQRT2PI
    if f'stddev_{index}' in idx:
        grad[idx[f'stddev_{index}']] += amp * SQRT2PI

    uncert = np.sqrt(grad @ cov @ grad)
    return total, uncert


# ==================
# per-line result container
# ==================
def make_result(line, source, mean, stddev, flux_norm_units, flux_uncert_norm_units,
                 norm_const, is_upper_limit):
    flux_phys = flux_norm_units * norm_const
    flux_uncert_phys = flux_uncert_norm_units * norm_const
    return {
        'line': line,
        'source': source,
        'mean': mean,
        'stddev': stddev,
        'flux': flux_phys,
        'flux_uncert': flux_uncert_phys,
        'is_upper_limit': is_upper_limit,
        'units': 'erg / s / cm2',
    }


def plot_group(lam_full, flux_full, noise_full, model_total, highlight_indices,
               highlight_names, title, savepath_png):
    lam_model = np.linspace(lam_full[0], lam_full[-1], 5000)
    fig, ax = plt.subplots(nrows=2, height_ratios=[3, 1], sharex=True, figsize=(11, 8))

    ax[0].plot(lam_full, flux_full, c="black", label="data", ds="steps")
    ax[0].fill_between(lam_full, flux_full - noise_full, flux_full + noise_full,
                        color="dimgray", alpha=0.6, label="noise")
    ax[0].plot(lam_model, model_total(lam_model), color="orange", label="bestfit total", lw=2)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(highlight_indices), 1)))
    for i, name, color in zip(highlight_indices, highlight_names, colors):
        comp = model_total[i]
        ax[0].plot(lam_model, comp(lam_model), color=color, ls="--", lw=1.5,
                   label=f"{name}  mean={comp.mean.value:.2f}  sigma={comp.stddev.value:.2f} AA")

    ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=13)
    ax[0].set_ylabel("Normalised Flux", fontsize=13)
    ax[0].set_title(title, fontsize=11)
    ax[0].legend(frameon=False, fontsize=8)

    residual_sigma = (flux_full - model_total(lam_full)) / noise_full
    ax[1].scatter(lam_full, residual_sigma, s=10, c="orange", alpha=0.5,
                  label="(flux-model)/noise")
    ax[1].axhline(0, ls='--', alpha=0.4)
    for level, style in [(1, ':'), (2, '--')]:
        ax[1].axhline(level, ls=style, color='red', alpha=0.3, lw=0.8)
        ax[1].axhline(-level, ls=style, color='red', alpha=0.3, lw=0.8)
    ax[1].legend(frameon=True, fontsize=8)

    os.makedirs(os.path.dirname(savepath_png), exist_ok=True)
    fig.savefig(savepath_png)
    plt.close(fig)


def resolve_detection(bestfit_model, index, window, rest_line, heii_stddev_source, converged):
    """Decide detection vs. 3-sigma upper limit for one Gaussian component,
    and return (flux_norm_units, flux_uncert_norm_units, mean, stddev,
    is_upper_limit). 'fit-convergence only': a component counts as a
    non-detection if the overall fit didn't converge, its amplitude
    collapsed to ~0, or its stddev is pinned at either bound (a boundary
    solution means the fit did NOT converge to an interior optimum for that
    parameter -- see STDDEV_BOUNDS comment above)."""
    lam_w, flux_w, noise_w = window
    amp = getattr(bestfit_model, f'amplitude_{index}').value
    std = getattr(bestfit_model, f'stddev_{index}').value
    mean = getattr(bestfit_model, f'mean_{index}').value

    is_non_detection = ((not converged) or amp < AMPLITUDE_COLLAPSE_ATOL or
                         len(lam_w) == 0 or _stddev_at_bound(std))

    if is_non_detection:
        local_noise = np.nanmedian(noise_w) if len(noise_w) else np.nan
        assumed_std = heii_stddev_source * (rest_line / rest['HeII1640'])
        amp_3sigma = 3 * local_noise
        flux_norm_units = amp_3sigma * assumed_std * SQRT2PI
        flux_uncert_norm_units = np.nan
        stddev_out = assumed_std
    else:
        flux_norm_units, flux_uncert_norm_units = flux_and_uncert(bestfit_model, index)
        stddev_out = std

    return flux_norm_units, flux_uncert_norm_units, mean, stddev_out, is_non_detection


# ==================
# UVB joint fit: HeII1640 (free master, per source) + the 5 other UVB
# lines (mean tied to that source's HeII mean; amplitude & stddev free --
# only the mean is tied, per the task's instruction). Fit one shared window
# per group (see UVB_GROUPS) rather than per line/source, so overlapping
# pixels are never double-counted.
# ==================
def fit_uvb(lam_uvb, flux_uvb, noise_uvb, norm_const_uvb):
    group_windows = {}
    for group_name, lines_in_group in UVB_GROUPS:
        lo_obs, hi_obs = group_window_bounds(lines_in_group)
        group_windows[group_name] = make_window(lam_uvb, flux_uvb, noise_uvb, lo_obs, hi_obs)

    gaussians = []
    index_of = {}  # (line, source) -> compound-model index
    master_idx = {}
    group_of_index = {}  # compound-model index of a Gaussian -> its group_name

    # HeII1640 master (free mean/stddev/amplitude per source)
    for source, z in SOURCES.items():
        lam_w, flux_w, noise_w = group_windows['HeII1640']
        mean_guess = rest['HeII1640'] * (1 + z)
        g = models.Gaussian1D(name=f'HeII1640_{source}', mean=mean_guess,
                               amplitude=max(np.nanmax(flux_w) - np.nanmedian(flux_w), 0.1) if len(flux_w) else 0.1,
                               stddev=0.5)
        g.mean.bounds = (mean_guess - 5, mean_guess + 5)
        g.stddev.bounds = STDDEV_BOUNDS
        g.amplitude.bounds = (0, None)
        master_idx[source] = len(gaussians)
        index_of[('HeII1640', source)] = len(gaussians)
        group_of_index[len(gaussians)] = 'HeII1640'
        gaussians.append(g)

    # every other line: mean tied to its source's HeII master
    for group_name, lines_in_group in UVB_GROUPS:
        if group_name == 'HeII1640':
            continue
        lam_w, flux_w, noise_w = group_windows[group_name]
        for line in lines_in_group:
            for source in SOURCES:
                ref_idx = master_idx[source]
                line_ratio = rest[line] / rest['HeII1640']
                mean_guess = rest[line] * (1 + SOURCES[source])
                nearby = np.abs(lam_w - mean_guess) < 15
                amp_guess = max(np.nanmax(flux_w[nearby]) - np.nanmedian(flux_w), 0.05) \
                    if len(flux_w) and nearby.any() else 0.1
                g = models.Gaussian1D(name=f'{line}_{source}', mean=1.0,
                                       amplitude=amp_guess, stddev=0.5)
                g.mean.tied = MeanTie(ref_idx, line_ratio)
                g.stddev.bounds = STDDEV_BOUNDS
                g.amplitude.bounds = (0, None)
                index_of[(line, source)] = len(gaussians)
                group_of_index[len(gaussians)] = group_name
                gaussians.append(g)

    continua = []
    cont_idx = {}
    for group_name, _ in UVB_GROUPS:
        lam_w, flux_w, noise_w = group_windows[group_name]
        cont = WindowedConst1D(
            amplitude=np.nanmedian(flux_w) if len(flux_w) else 0.0,
            lo=lam_w.min() if len(lam_w) else 0.0,
            hi=lam_w.max() if len(lam_w) else 1.0,
            name=f'continuum_{group_name}')
        cont.lo.fixed = True
        cont.hi.fixed = True
        cont_idx[group_name] = len(gaussians) + len(continua)
        continua.append(cont)

    group_order = [g for g, _ in UVB_GROUPS]
    lam_all = np.concatenate([group_windows[g][0] for g in group_order])
    flux_all = np.concatenate([group_windows[g][1] for g in group_order])
    noise_all = np.concatenate([group_windows[g][2] for g in group_order])

    compound_model = reduce(operator.add, gaussians + continua)
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    bestfit_model = fitter(compound_model, lam_all, flux_all,
                            weights=1.0 / noise_all, maxiter=5000)

    converged = getattr(fitter, 'fit_info', {}).get('success', True)

    heii_stddev = {}
    heii_mean_fit = {}
    for source in SOURCES:
        i = master_idx[source]
        heii_stddev[source] = getattr(bestfit_model, f'stddev_{i}').value
        heii_mean_fit[source] = getattr(bestfit_model, f'mean_{i}').value

    results = []
    for group_name, lines_in_group in UVB_GROUPS:
        lam_w, flux_w, noise_w = group_windows[group_name]
        comp_indices = [index_of[(line, source)] for line in lines_in_group for source in SOURCES]
        comp_names = [f'{line}_{source}' for line in lines_in_group for source in SOURCES]

        for line in lines_in_group:
            for source in SOURCES:
                i = index_of[(line, source)]
                (flux_norm_units, flux_uncert_norm_units, mean, stddev_out,
                 is_non_detection) = resolve_detection(
                    bestfit_model, i, group_windows[group_name], rest[line],
                    heii_stddev[source], converged)

                results.append(make_result(line, source, mean, stddev_out,
                                            flux_norm_units, flux_uncert_norm_units,
                                            norm_const_uvb, is_non_detection))

                tag = "3-sigma UPPER LIMIT" if is_non_detection else "detection"
                safe_line = line.replace("[","").replace("]","")
                if len(lam_w):
                    plot_group(lam_w, flux_w, noise_w, bestfit_model, comp_indices, comp_names,
                               f"{group_name}  |  highlighting {line} source {source} ({tag})",
                               f'{FITDIR}/{safe_line}/{safe_line}_{source}_fit.png')
                    os.makedirs(f'{FITDIR}/{safe_line}', exist_ok=True)
                    with open(f'{FITDIR}/{safe_line}/{safe_line}_{source}_fit.pkl', 'wb') as f:
                        dill.dump({
                            'model': bestfit_model[i],
                            'group_model': bestfit_model,
                            'is_upper_limit': is_non_detection,
                        }, f)

    return results, heii_stddev, heii_mean_fit


# ==================
# VIS fit: [CII]2325, mean fixed from the UVB HeII fit's derived redshift
# per source (a live cross-model .tied isn't possible since this is a
# separate compound model/fit on a different file's data). Both sources
# share ONE window/continuum (same double-counting rationale as the UVB
# groups above), since they're only ~5 AA apart, well inside a +-50 AA
# window.
# ==================
def fit_vis(lam_vis, flux_vis, noise_vis, norm_const_vis, heii_stddev, heii_mean_fit):
    z_fit = {source: heii_mean_fit[source] / rest['HeII1640'] - 1 for source in SOURCES}
    mean_fixed = {source: rest['[CII]2325'] * (1 + z_fit[source]) for source in SOURCES}

    lo_obs = min(mean_fixed.values()) - WINDOW_HALFWIDTH
    hi_obs = max(mean_fixed.values()) + WINDOW_HALFWIDTH
    lam_w, flux_w, noise_w = make_window(lam_vis, flux_vis, noise_vis, lo_obs, hi_obs)

    gaussians = []
    for source in SOURCES:
        nearby = np.abs(lam_w - mean_fixed[source]) < 15
        amp_guess = max(np.nanmax(flux_w[nearby]) - np.nanmedian(flux_w), 0.05) \
            if len(flux_w) and nearby.any() else 0.1
        g = models.Gaussian1D(name=f'[CII]2325_{source}', mean=mean_fixed[source],
                               amplitude=amp_guess, stddev=0.5)
        g.mean.fixed = True
        g.stddev.bounds = STDDEV_BOUNDS
        g.amplitude.bounds = (0, None)
        gaussians.append(g)

    cont = WindowedConst1D(
        amplitude=np.nanmedian(flux_w) if len(flux_w) else 0.0,
        lo=lam_w.min() if len(lam_w) else 0.0,
        hi=lam_w.max() if len(lam_w) else 1.0,
        name='continuum_[CII]2325')
    cont.lo.fixed = True
    cont.hi.fixed = True
    cont_idx = len(gaussians)

    compound_model = reduce(operator.add, gaussians + [cont])
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    if len(lam_w):
        bestfit_model = fitter(compound_model, lam_w, flux_w,
                                weights=1.0 / noise_w, maxiter=5000)
        converged = getattr(fitter, 'fit_info', {}).get('success', True)
    else:
        bestfit_model = compound_model
        converged = False

    results = []
    comp_indices = list(range(len(gaussians)))
    comp_names = [f'[CII]2325_{source}' for source in SOURCES]
    for idx, source in enumerate(SOURCES):
        (flux_norm_units, flux_uncert_norm_units, mean, stddev_out,
         is_non_detection) = resolve_detection(
            bestfit_model, idx, (lam_w, flux_w, noise_w), rest['[CII]2325'],
            heii_stddev[source], converged)

        results.append(make_result('[CII]2325', source, mean, stddev_out,
                                    flux_norm_units, flux_uncert_norm_units,
                                    norm_const_vis, is_non_detection))

        tag = "3-sigma UPPER LIMIT" if is_non_detection else "detection"
        if len(lam_w):
            plot_group(lam_w, flux_w, noise_w, bestfit_model, comp_indices, comp_names,
                       f"[CII]2325  |  highlighting source {source} ({tag})",
                       f'{FITDIR}/CII2325/CII2325_{source}_fit.png')
            os.makedirs(f'{FITDIR}/CII2325', exist_ok=True)
            with open(f'{FITDIR}/CII2325/CII2325_{source}_fit.pkl', 'wb') as f:
                dill.dump({
                    'model': bestfit_model[idx],
                    'group_model': bestfit_model,
                    'is_upper_limit': is_non_detection,
                }, f)

    return results


if __name__ == "__main__":
    lam_uvb, flux_uvb, noise_uvb, norm_uvb = load_arm('UVB')
    lam_vis, flux_vis, noise_vis, norm_vis = load_arm('VIS')

    uvb_results, heii_stddev, heii_mean_fit = fit_uvb(lam_uvb, flux_uvb, noise_uvb, norm_uvb)
    vis_results = fit_vis(lam_vis, flux_vis, noise_vis, norm_vis, heii_stddev, heii_mean_fit)

    all_results = uvb_results + vis_results

    rows = []
    for r in all_results:
        rows.append({
            'line': r['line'],
            'source': r['source'],
            'mean_obs_AA': r['mean'],
            'stddev_obs_AA': r['stddev'],
            'flux': r['flux'],
            'flux_uncert': r['flux_uncert'],
            'is_upper_limit': r['is_upper_limit'],
            'units': r['units'],
        })
        print(f"{r['line']:14s} {r['source']}  "
              f"flux = {r['flux']:.4g} +/- {r['flux_uncert']:.4g} {r['units']}"
              f"{'  [3-sigma UPPER LIMIT]' if r['is_upper_limit'] else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(f'{FLUXDIR}/uv_line_fluxes.csv', index=False)
    with open(f'{FLUXDIR}/uv_line_fluxes.pkl', 'wb') as f:
        dill.dump(rows, f)

    print(f"\nSaved flux table to {FLUXDIR}/uv_line_fluxes.csv")
