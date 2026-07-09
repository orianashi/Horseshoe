import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting, custom_model
from functools import reduce
import operator
from matplotlib.ticker import MultipleLocator

plt.ion()

SQRT2PI = np.sqrt(2 * np.pi)

# ==================
# redshifts and rest wavelengths
# ==================
z_A = 1.679
z_B = 1.677

rest = {
    'Halpha': 6562.819,
    '[OIII]5007': 5006.843,
    '[OIII]4959': 4958.911,
    '[OII]3726': 3726.03,
    '[OII]3729': 3728.815,
}
# window/plot iteration order -- '[OII]' is a single combined window covering both
# [OII]3726 and [OII]3729 (see windows construction below)
line_order = ['[OIII]5007', '[OIII]4959', '[OII]']


# ==================
# windowed continuum: a plain Const1D would leak into all windows once
# evaluated over the concatenated x array, so restrict each window's
# continuum to its own range
# ==================
def _windowed_const(x, amplitude=1.0, lo=0.0, hi=1.0):
    return np.where((x >= lo) & (x <= hi), amplitude, 0.0)


WindowedConst1D = custom_model(_windowed_const)


def build_fixed(label, ref_idx, ratio, halpha_mean, halpha_std, amplitude_guess,
                amplitude_bound):
    """Unlike jointfit_all.py's build_tied, there's no live master component in
    this script's own compound model to reference -- Halpha's shape comes from
    a separate (already-completed) fit, so mean/stddev are baked in as plain
    fixed floats up front rather than a live `.tied` callable."""
    mean_val = halpha_mean[ref_idx] * ratio
    std_val = halpha_std[ref_idx] * ratio
    g = models.Gaussian1D(name=label, mean=mean_val, amplitude=amplitude_guess,
                          stddev=std_val)
    g.mean.fixed = True
    g.stddev.fixed = True
    g.amplitude.bounds = amplitude_bound
    return g


A_INDICES = {
    '[OIII]5007': [0, 1],
    '[OIII]4959': [5, 6],
    '[OII]': [10, 11, 15, 16],
}
B_INDICES = {
    '[OIII]5007': [2, 3, 4],
    '[OIII]4959': [7, 8, 9],
    '[OII]': [12, 13, 14, 17, 18, 19],
}

# per-line (not per-plot-window) index groups, used for the flux/uncertainty demo
FLUX_LINES = {
    '[OIII]5007': {'A': [0, 1], 'B': [2, 3, 4]},
    '[OIII]4959': {'A': [5, 6], 'B': [7, 8, 9]},
    '[OII]3726': {'A': [10, 11], 'B': [12, 13, 14]},
    '[OII]3729': {'A': [15, 16], 'B': [17, 18, 19]},
}

colors = {
    ('A', 'central'): 'purple',
    ('A', 'red'): 'orangered',
    ('B', 'central'): 'darkorange',
    ('B', 'red'): 'crimson',
    ('B', 'blue'): 'royalblue',
}


def component_color(label):
    _, source, role = label.split('_')
    return colors[(source, role)]


# ==================
# every tied-to-Halpha component here shares the same 5-way ref_idx layout as
# jointfit_all.py: 0=A_central 1=A_red 2=B_red 3=B_central 4=B_blue. Structural
# (ref_idx, ratio) only -- independent of the actual fitted Halpha values, so
# it's built once and reused both for the main fit and every perturbation
# refit used by the rigorous uncertainty correction.
# ==================
def build_external_tie_map(r_oiii5_ha, r_oiii4_ha, r_oii3726_ha, r_oii3729_ha):
    ratios_by_gaussian = (
        [r_oiii5_ha] * 5 + [r_oiii4_ha] * 5 + [r_oii3726_ha] * 5 +
        [r_oii3729_ha] * 5)
    ref_idx_by_gaussian = [0, 1, 2, 3, 4] * 4
    return {
        i: {
            'ref_idx': ref_idx_by_gaussian[i],
            'ratio': ratios_by_gaussian[i]
        }
        for i in range(20)
    }


# ==================
# uncertainty propagation: see plan / conversation for the derivation. Both
# take Stage 2's bestfit model + component indices + the external_tie_map
# above, plus Stage 1's bestfit model (which carries its OWN cov_matrix over
# Halpha's mean/stddev, since Stage 1 was fit completely separately).
# ==================
def flux_and_uncert_twostage_simple(stage2_model, indices, external_tie_map,
                                    stage1_model):
    """Assumes Stage 2's best-fit amplitudes wouldn't themselves shift if
    Halpha's true (Stage 1) width were slightly different -- i.e. only the
    EXPLICIT stddev_i = ratio * stddev_ref dependence of flux is propagated,
    routed into Stage 1's own covariance matrix at the ref stddev's slot,
    exactly like the existing tie_map branch of flux_and_uncert() in
    multiple_gaussian_integration.py, just pointed at a second model."""
    names2 = stage2_model.cov_matrix.param_names
    cov2 = stage2_model.cov_matrix.cov_matrix
    idx2 = {n: k for k, n in enumerate(names2)}

    names1 = stage1_model.cov_matrix.param_names
    cov1 = stage1_model.cov_matrix.cov_matrix
    idx1 = {n: k for k, n in enumerate(names1)}

    total = 0.0
    grad2 = np.zeros(len(names2))
    grad1 = np.zeros(len(names1))

    for i in indices:
        amp = getattr(stage2_model, f'amplitude_{i}').value
        std = getattr(stage2_model, f'stddev_{i}').value
        total += amp * std * SQRT2PI

        # amplitude is always free (never tied/fixed) in this script's model
        grad2[idx2[f'amplitude_{i}']] += std * SQRT2PI

        ref = external_tie_map[i]
        ref_name = f"stddev_{ref['ref_idx']}"
        grad1[idx1[ref_name]] += amp * SQRT2PI * ref['ratio']

    # Stage 1 and Stage 2 fit disjoint spectral windows -> independent noise
    # -> no cross-covariance term between the two variance sources
    var2 = grad2 @ cov2 @ grad2
    var1 = grad1 @ cov1 @ grad1
    return total, np.sqrt(var2 + var1)


def compute_amp_sensitivities(build_stage2_fn, halpha_mean, halpha_std, lam_all,
                              flux_all, noise_all, baseline_model, rel_step=1e-3):
    """For each of Halpha's 5 master stddevs, numerically estimate
    d(amplitude_j)/d(stddev_ref) for every one of Stage 2's 20 line
    amplitudes by re-solving Stage 2 (fast: it's linear in its free
    parameters once mean/stddev are fixed) at stddev_ref +/- a small step.
    Returns {ref_idx: length-20 array}, computed once and shared across every
    flux_and_uncert_twostage_rigorous() call so the 10 perturbation refits
    aren't repeated per component."""
    sensitivities = {}
    for ref_idx in range(5):
        step = rel_step * halpha_std[ref_idx]
        amps_plus = _refit_amplitudes(build_stage2_fn, halpha_mean, halpha_std,
                                      ref_idx, step, lam_all, flux_all,
                                      noise_all, baseline_model)
        amps_minus = _refit_amplitudes(build_stage2_fn, halpha_mean, halpha_std,
                                       ref_idx, -step, lam_all, flux_all,
                                       noise_all, baseline_model)
        sensitivities[ref_idx] = (amps_plus - amps_minus) / (2 * step)
    return sensitivities


def _refit_amplitudes(build_stage2_fn, halpha_mean, halpha_std, ref_idx, delta,
                      lam_all, flux_all, noise_all, baseline_model):
    perturbed_std = list(halpha_std)
    perturbed_std[ref_idx] += delta
    model = build_stage2_fn(halpha_mean, perturbed_std)
    # seed every free parameter (all 20 line amplitudes + 3 continuum levels)
    # from the baseline fit for fast, stable convergence on this near-linear
    # problem
    for i in range(23):
        pname = f'amplitude_{i}'
        getattr(model, pname).value = getattr(baseline_model, pname).value
    fitter = fitting.TRFLSQFitter(calc_uncertainties=False)
    bf = fitter(model, lam_all, flux_all, weights=1.0 / noise_all, maxiter=200)
    return np.array([getattr(bf, f'amplitude_{i}').value for i in range(20)])


def flux_and_uncert_twostage_rigorous(stage2_model, indices, external_tie_map,
                                      stage1_model, amp_sensitivities):
    """Adds the term the simple version ignores: since a component's fixed
    stddev is a plug-in estimate from Stage 1, Stage 2's own best-fit
    amplitude would itself shift if that plugged-in value were slightly off
    (the classic two-step/plug-in-estimator "sandwich" correction). The total
    sensitivity of flux_i to stddev_ref is now
        amp_i * ratio * sqrt(2pi)              (explicit, as in the simple version)
      + std_i * sqrt(2pi) * d(amp_i)/d(stddev_ref)   (implicit, from amp_sensitivities)
    """
    names2 = stage2_model.cov_matrix.param_names
    cov2 = stage2_model.cov_matrix.cov_matrix
    idx2 = {n: k for k, n in enumerate(names2)}

    names1 = stage1_model.cov_matrix.param_names
    cov1 = stage1_model.cov_matrix.cov_matrix
    idx1 = {n: k for k, n in enumerate(names1)}

    total = 0.0
    grad2 = np.zeros(len(names2))
    grad1 = np.zeros(len(names1))

    for i in indices:
        amp = getattr(stage2_model, f'amplitude_{i}').value
        std = getattr(stage2_model, f'stddev_{i}').value
        total += amp * std * SQRT2PI

        grad2[idx2[f'amplitude_{i}']] += std * SQRT2PI

        ref = external_tie_map[i]
        ref_name = f"stddev_{ref['ref_idx']}"
        d_amp_d_stddevref = amp_sensitivities[ref['ref_idx']][i]
        grad1[idx1[ref_name]] += (amp * ref['ratio'] +
                                  std * d_amp_d_stddevref) * SQRT2PI

    var2 = grad2 @ cov2 @ grad2
    var1 = grad1 @ cov1 @ grad1
    return total, np.sqrt(var2 + var1)


# ==================
# everything below actually loads data, fits, and writes output files
# ==================
if __name__ == "__main__":
    # ==================
    # Stage 1: load (don't refit) Halpha's already-completed Balmer-only
    # joint fit -- balmer_joint_gaussian_fitting.py has already been run and
    # its Halpha mean_0..4/stddev_0..4 are genuinely free (never tied)
    # parameters with a working cov_matrix.
    # ==================
    stage1_pkl_path = './output/joint_fit/Balmer/Halpha_joint_tied_fit.pkl'
    with open(stage1_pkl_path, 'rb') as f:
        stage1 = dill.load(f)
    stage1_model = stage1['model']
    halpha_mean = [getattr(stage1_model, f'mean_{i}').value for i in range(5)]
    halpha_std = [getattr(stage1_model, f'stddev_{i}').value for i in range(5)]

    # ==================
    # wing-fit amplitudes for the blended OII lines ([OII]3726_A and
    # [OII]3729_B) -- pulled from oii_gaussian_fitting.py's already-fit
    # unblended wing model, same as jointfit_all.py.
    # ==================
    with open('./output/OII/OII_wingfit_gaussians.pkl', 'rb') as f:
        wing_bestfit = dill.load(f)
    amp0_wing = wing_bestfit.amplitude_0.value
    amp1_wing = wing_bestfit.amplitude_1.value

    # ==================
    # load the shared NIR spectrum ([OIII]5007, [OIII]4959)
    # ==================
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

    # ==================
    # load the VIS spectrum ([OII]3726/3729 redshift into the VIS arm)
    # ==================
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

    # ==================
    # per-line windows: +-50 AA around each line's expected observed
    # wavelength ([OIII]5007/[OIII]4959 from NIR); [OII]3726+3729 share one
    # combined window from VIS (only ~7 AA apart in observed wavelength, vs
    # the 50 AA window half-width, so two separate windows would heavily
    # overlap).
    # ==================
    windows = {}
    for name in ['[OIII]5007', '[OIII]4959']:
        center = rest[name] * (1 + z_B)
        zoom = np.where((lam >= center - 50) & (lam <= center + 50))[0]
        lam_trim = lam[zoom]
        flux_trim = flux_norm[zoom]
        noise_trim = noise_norm[zoom]
        bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) |
               (noise_trim < 0))
        lam_full, flux_full, noise_full = lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]
        windows[name] = {
            'lam_full': lam_full,
            'flux_full': flux_full,
            'noise_full': noise_full,
            'lam_fit': lam_full,
            'flux_fit': flux_full,
            'noise_fit': noise_full,
        }

    oii_lo = rest['[OII]3726'] * (1 + z_B) - 50
    oii_hi = rest['[OII]3729'] * (1 + z_A) + 50
    zoom_oii = np.where((lam_vis >= oii_lo) & (lam_vis <= oii_hi))[0]
    lam_trim = lam_vis[zoom_oii]
    flux_trim = flux_norm_vis[zoom_oii]
    noise_trim = noise_norm_vis[zoom_oii]
    bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) |
           (noise_trim < 0))
    lam_full, flux_full, noise_full = lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]
    windows['[OII]'] = {
        'lam_full': lam_full,
        'flux_full': flux_full,
        'noise_full': noise_full,
        'lam_fit': lam_full,
        'flux_fit': flux_full,
        'noise_fit': noise_full,
    }

    lam_all = np.concatenate([windows[n]['lam_fit'] for n in line_order])
    flux_all = np.concatenate([windows[n]['flux_fit'] for n in line_order])
    noise_all = np.concatenate([windows[n]['noise_fit'] for n in line_order])

    r_oiii5_ha = rest['[OIII]5007'] / rest['Halpha']
    r_oiii4_ha = rest['[OIII]4959'] / rest['Halpha']
    r_oii3726_ha = rest['[OII]3726'] / rest['Halpha']
    r_oii3729_ha = rest['[OII]3729'] / rest['Halpha']

    EXTERNAL_TIE_MAP = build_external_tie_map(r_oiii5_ha, r_oiii4_ha,
                                              r_oii3726_ha, r_oii3729_ha)

    # ==================
    # index layout (fixed order):
    #  0: [OIII]5007_A_central   1: [OIII]5007_A_red
    #  2: [OIII]5007_B_red       3: [OIII]5007_B_central   4: [OIII]5007_B_blue
    #  5: [OIII]4959_A_central   6: [OIII]4959_A_red
    #  7: [OIII]4959_B_red       8: [OIII]4959_B_central   9: [OIII]4959_B_blue
    # 10: [OII]3726_A_central (wing amplitude bounds)  11: [OII]3726_A_red
    # 12: [OII]3726_B_red   13: [OII]3726_B_central   14: [OII]3726_B_blue
    # 15: [OII]3729_A_central   16: [OII]3729_A_red
    # 17: [OII]3729_B_red   18: [OII]3729_B_central (wing amplitude bounds)  19: [OII]3729_B_blue
    # 20: continuum_[OIII]5007  21: continuum_[OIII]4959  22: continuum_[OII]
    # ==================
    def build_stage2_model(halpha_mean, halpha_std):
        gaussians = []

        gaussians.append(
            build_fixed('[OIII]5007_A_central', 0, r_oiii5_ha, halpha_mean,
                       halpha_std, 50, (0, None)))  # 0
        gaussians.append(
            build_fixed('[OIII]5007_A_red', 1, r_oiii5_ha, halpha_mean,
                       halpha_std, 10, (0, None)))  # 1
        gaussians.append(
            build_fixed('[OIII]5007_B_red', 2, r_oiii5_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 2
        gaussians.append(
            build_fixed('[OIII]5007_B_central', 3, r_oiii5_ha, halpha_mean,
                       halpha_std, 2.5, (0, None)))  # 3
        gaussians.append(
            build_fixed('[OIII]5007_B_blue', 4, r_oiii5_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 4

        gaussians.append(
            build_fixed('[OIII]4959_A_central', 0, r_oiii4_ha, halpha_mean,
                       halpha_std, 50, (0, None)))  # 5
        gaussians.append(
            build_fixed('[OIII]4959_A_red', 1, r_oiii4_ha, halpha_mean,
                       halpha_std, 10, (0, None)))  # 6
        gaussians.append(
            build_fixed('[OIII]4959_B_red', 2, r_oiii4_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 7
        gaussians.append(
            build_fixed('[OIII]4959_B_central', 3, r_oiii4_ha, halpha_mean,
                       halpha_std, 2.5, (0, None)))  # 8
        gaussians.append(
            build_fixed('[OIII]4959_B_blue', 4, r_oiii4_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 9

        gaussians.append(
            build_fixed('[OII]3726_A_central', 0, r_oii3726_ha, halpha_mean,
                       halpha_std, amp0_wing / 1.4,
                       (amp0_wing / 1.5, amp0_wing / 0.35)))  # 10
        gaussians.append(
            build_fixed('[OII]3726_A_red', 1, r_oii3726_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 11
        gaussians.append(
            build_fixed('[OII]3726_B_red', 2, r_oii3726_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 12
        gaussians.append(
            build_fixed('[OII]3726_B_central', 3, r_oii3726_ha, halpha_mean,
                       halpha_std, amp1_wing, (0, None)))  # 13
        gaussians.append(
            build_fixed('[OII]3726_B_blue', 4, r_oii3726_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 14

        gaussians.append(
            build_fixed('[OII]3729_A_central', 0, r_oii3729_ha, halpha_mean,
                       halpha_std, amp0_wing, (0, None)))  # 15
        gaussians.append(
            build_fixed('[OII]3729_A_red', 1, r_oii3729_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 16
        gaussians.append(
            build_fixed('[OII]3729_B_red', 2, r_oii3729_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 17
        gaussians.append(
            build_fixed('[OII]3729_B_central', 3, r_oii3729_ha, halpha_mean,
                       halpha_std, amp1_wing * 1.4,
                       (amp0_wing * 0.35, amp1_wing * 1.5)))  # 18
        gaussians.append(
            build_fixed('[OII]3729_B_blue', 4, r_oii3729_ha, halpha_mean,
                       halpha_std, 1, (0, None)))  # 19

        continua = []
        for name in line_order:
            lo, hi = windows[name]['lam_full'].min(), windows[name]['lam_full'].max()
            cont = WindowedConst1D(amplitude=np.nanmedian(windows[name]['flux_full']),
                                   lo=lo,
                                   hi=hi,
                                   name=f'continuum_{name}')
            cont.lo.fixed = True
            cont.hi.fixed = True
            continua.append(cont)

        return reduce(operator.add, gaussians + continua)

    CONT_INDICES = {name: 20 + i for i, name in enumerate(line_order)}

    compound_model = build_stage2_model(halpha_mean, halpha_std)

    # ==================
    # fit Stage 2 jointly across [OIII]5007 + [OIII]4959 + [OII]
    # ==================
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    bestfit_model = fitter(compound_model,
                           lam_all,
                           flux_all,
                           weights=1.0 / noise_all,
                           maxiter=5000)

    # ==================
    # per-line plotting
    # ==================
    for name in line_order:
        lam_clean = windows[name]['lam_full']
        flux_clean = windows[name]['flux_full']
        noise_clean = windows[name]['noise_full']

        lam_model = np.linspace(lam_clean[0], lam_clean[-1], 30000)
        cont_m = bestfit_model[CONT_INDICES[name]]

        fig, ax = plt.subplots(nrows=2,
                               height_ratios=[3, 1],
                               sharex=True,
                               figsize=(12, 10))

        ax[0].plot(lam_clean, flux_clean, c="black", label="data", ds="steps")
        ax[0].fill_between(lam_clean,
                           flux_clean - noise_clean,
                           flux_clean + noise_clean,
                           color="dimgray",
                           alpha=0.8,
                           label="noise")
        ax[0].plot(lam_model,
                  bestfit_model(lam_model),
                  color="orange",
                  label="Bestfit total",
                  lw=2)

        comp_indices = A_INDICES[name] + B_INDICES[name]
        for i in comp_indices:
            comp = bestfit_model[i]
            ax[0].plot(lam_model,
                      comp(lam_model) + cont_m(lam_model),
                      color=component_color(comp.name),
                      ls="--",
                      lw=1.5,
                      label=f"{comp.name}  σ={comp.stddev.value:.2f} AA")

        ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
        ax[0].set_ylabel("Normalised Flux [erg/s/cm2/AA]", fontsize=15)
        ax[0].xaxis.set_minor_locator(MultipleLocator(1))
        ax[0].set_title(
            f"{name}  |  two-stage fit (Halpha shape fixed from Balmer-only fit)  |  z_A={z_A}  z_B={z_B}",
            fontsize=12)
        ax[0].legend(frameon=False, fontsize=8)

        residual_sigma = (flux_clean - bestfit_model(lam_clean)) / noise_clean
        ax[1].scatter(lam_clean,
                     residual_sigma,
                     s=10,
                     c="orange",
                     label="(flux - model)/noise",
                     alpha=0.5)
        ax[1].axhline(0, ls='--', alpha=0.4)
        for level, style in [(1, ':'), (2, '--')]:
            ax[1].axhline(level, ls=style, color='red', alpha=0.3, lw=0.8)
            ax[1].axhline(-level, ls=style, color='red', alpha=0.3, lw=0.8)
        ax[1].legend(frameon=True)
        plt.show()

        n_free = sum(1 for pname in bestfit_model.param_names
                    if not getattr(bestfit_model, pname).fixed
                    and not getattr(bestfit_model, pname).tied)
        dof = len(lam_all) - n_free
        chi2_line = np.sum(residual_sigma**2)
        print(f"{name}: chi2 (this window) = {chi2_line:.1f}, "
             f"n_points = {len(lam_clean)}, reduced chi2 (joint dof) = "
             f"{chi2_line / dof:.2f}")

        n_sigma_window = 3.5
        near_line_mask = reduce(operator.or_, [
            np.abs(lam_clean - bestfit_model[i].mean.value)
            <= n_sigma_window * bestfit_model[i].stddev.value
            for i in comp_indices
        ])
        residual_sigma_near = residual_sigma[near_line_mask]
        print(
            f"  Points near emission lines (within {n_sigma_window}sigma): "
            f"{len(residual_sigma_near)} / {len(residual_sigma)}")
        print(
            f"  Within 1sig: {np.sum(np.abs(residual_sigma_near) <= 1)} "
            f"({100 * np.sum(np.abs(residual_sigma_near) <= 1) / len(residual_sigma_near):.2f}%), "
            f"Within 2sig: {np.sum(np.abs(residual_sigma_near) <= 2)} "
            f"({100 * np.sum(np.abs(residual_sigma_near) <= 2) / len(residual_sigma_near):.2f}%), "
            f"Within 3sig: {np.sum(np.abs(residual_sigma_near) <= 3)} "
            f"({100 * np.sum(np.abs(residual_sigma_near) <= 3) / len(residual_sigma_near):.2f}%)"
        )

        fig.savefig(f'./output/joint_fit/twostage/{name}_twostage_fit.png')
        with open(f'./output/joint_fit/twostage/{name}_twostage_fit.pkl', 'wb') as f:
            dill.dump(
                {
                    'model': bestfit_model,
                    'external_tie_map': EXTERNAL_TIE_MAP,
                    'stage1_pkl': stage1_pkl_path,
                }, f)

    # ==================
    # uncertainty demo: simple vs rigorous propagation for each line's
    # Source A / Source B total flux
    # ==================
    print("\nPrecomputing amplitude sensitivities to Halpha's Stage-1 stddevs "
         "(for the rigorous correction)...")
    amp_sensitivities = compute_amp_sensitivities(build_stage2_model, halpha_mean,
                                                  halpha_std, lam_all, flux_all,
                                                  noise_all, bestfit_model)

    print("\nFlux uncertainty: simple (explicit-term) vs rigorous (sandwich-corrected)")
    for line, sources in FLUX_LINES.items():
        for source, indices in sources.items():
            flux_simple, uncert_simple = flux_and_uncert_twostage_simple(
                bestfit_model, indices, EXTERNAL_TIE_MAP, stage1_model)
            flux_rigorous, uncert_rigorous = flux_and_uncert_twostage_rigorous(
                bestfit_model, indices, EXTERNAL_TIE_MAP, stage1_model,
                amp_sensitivities)
            print(f"{line} source {source}: flux = {flux_simple:.3f}  "
                 f"simple = {uncert_simple:.4f}  "
                 f"rigorous = {uncert_rigorous:.4f}")
