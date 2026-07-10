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

# ==================
# redshifts and rest wavelengths
# ==================
z_A = 1.679
z_B = 1.677

rest = {
    'Halpha': 6562.819,
    'Hbeta': 4861.333,
    'Hgamma': 4340.47,
}
line_order = ['Halpha', 'Hbeta', 'Hgamma']

# Hbeta's window contains a genuine gap of NaN flux/noise pixels between the
# A and B peaks (~13015.8-13020.6 AA observed, same gap masked in
# infinite_gaussians.py) -- excluded explicitly here (on top of the NaN
# filter below) so it's documented and also so it can be shaded on the plot.
# Hgamma's excluded region, unlike Hbeta's, is NOT missing/NaN data -- it's
# real (finite) pixels with a noisy-but-real spike (noise jumps ~0.9->2.2
# there), so it gets its own label rather than "NaN gap".
mask_ranges = {
    'Hbeta': (13015.8, 13020.6, 'missing data (NaN gap)'),
    #'Hgamma': (11622.4, 11625, 'excluded (noisy spike)'),
}


# ==================
# tying helpers (ratio of rest wavelengths -> same redshift & same
# velocity dispersion between the tied component and its master). These are
# picklable classes rather than closures: dill's by-value pickling of nested
# closures can silently corrupt their reconstructed bytecode (confirmed via
# dis.dis() raising on the round-tripped function, and calling it segfaulting
# the interpreter) -- a plain callable class survives dill fine since it's
# pickled as a class reference + a plain-data __dict__.
# ==================
class MeanTie:

    def __init__(self, ref_idx, line_ratio):
        self.ref_idx = ref_idx
        self.line_ratio = line_ratio

    def __call__(self, model):
        return getattr(model, f'mean_{self.ref_idx}') * self.line_ratio


class StdTie:

    def __init__(self, ref_idx, line_ratio):
        self.ref_idx = ref_idx
        self.line_ratio = line_ratio

    def __call__(self, model):
        return getattr(model, f'stddev_{self.ref_idx}') * self.line_ratio


# ==================
# windowed continuum: a plain Const1D would leak into all three windows
# once evaluated over the concatenated x array, so restrict each line's
# continuum to its own window
# ==================
def _windowed_const(x, amplitude=1.0, lo=0.0, hi=1.0):
    return np.where((x >= lo) & (x <= hi), amplitude, 0.0)


WindowedConst1D = custom_model(_windowed_const)

# ==================
# master components (free): Halpha's 5 components (A: central, red wing;
# B: red, central, blue). Source A has no blue wing for any line -- Halpha
# never had one, and it's been removed from Hbeta/Hgamma too -- so every
# tied component now anchors directly to one of these 5.
# ==================
master_guesses = {
    'Halpha_A_central': {
        'z_guess': z_A,
        'amplitude': 12,
        'stddev': 1.5,
        'stddev_bounds': (0, 3),
        'amplitude_bound': (0, None),
        'mean_range': 3,
    },
    'Halpha_A_red': {
        'z_guess': z_A + 0.0006,
        'amplitude': 3,
        'stddev': 2,
        'stddev_bounds': (0, 5),
        'amplitude_bound': (0, None),
        'mean_range': 5,
    },
    'Halpha_B_red': {
        'z_guess': z_B + 0.0006,
        'amplitude': 3,
        'stddev': 2,
        'stddev_bounds': (0, 5),
        'amplitude_bound': (0, None),
        'mean_range': 5,
    },
    'Halpha_B_central': {
        'z_guess': z_B,
        'amplitude': 8,
        'stddev': 1.5,
        'stddev_bounds': (0, 3),
        'amplitude_bound': (0, None),
        'mean_range': 3,
    },
    'Halpha_B_blue': {
        'z_guess': z_B - 0.0006,
        'amplitude': 3,
        'stddev': 2,
        'stddev_bounds': (0, 5),
        'amplitude_bound': (0, None),
        'mean_range': 5,
    },
}


def build_master(label, rest_wl):
    g = master_guesses[label]
    mean_guess = rest_wl * (1 + g['z_guess'])
    gauss = models.Gaussian1D(name=label,
                               mean=mean_guess,
                               amplitude=g['amplitude'],
                               stddev=g['stddev'])
    gauss.mean.bounds = (mean_guess - g['mean_range'],
                         mean_guess + g['mean_range'])
    gauss.stddev.bounds = g['stddev_bounds']
    gauss.amplitude.bounds = g['amplitude_bound']
    return gauss


def build_tied(label, ref_idx, line_ratio, amplitude_guess, amplitude_bound):
    g = models.Gaussian1D(name=label, mean=1.0, amplitude=amplitude_guess,
                          stddev=1.0)
    g.mean.tied = MeanTie(ref_idx, line_ratio)
    g.stddev.tied = StdTie(ref_idx, line_ratio)
    g.amplitude.bounds = amplitude_bound
    return g


A_INDICES = {'Halpha': [0, 1], 'Hbeta': [5, 6], 'Hgamma': [10, 11]}
B_INDICES = {
    'Halpha': [2, 3, 4],
    'Hbeta': [7, 8, 9],
    'Hgamma': [12, 13, 14]
}

colors = {
    ('A', 'central'): 'purple',
    ('A', 'red'): 'orangered',
    ('A', 'blue'): 'deepskyblue',
    ('B', 'central'): 'darkorange',
    ('B', 'red'): 'crimson',
    ('B', 'blue'): 'royalblue',
}


def component_color(label):
    _, source, role = label.split('_')
    return colors[(source, role)]


# ==================
# everything below actually loads data, fits, and writes output files --
# guarded so that importing this module (e.g. dill needing to resolve
# MeanTie/StdTie/WindowedConst1D while unpickling a saved fit elsewhere)
# only defines the classes/functions above and doesn't re-run the fit.
# ==================
if __name__ == "__main__":
    # ==================
    # load the shared NIR spectrum
    # ==================
    spec_lib = "./Data/X-Shooter/1D/stacked_NIR.fits"
    with fits.open(spec_lib) as hdu:
        h = hdu[1].header
        flux_data = hdu[1].data  # flux
        noise_data = hdu[4].data  # noise
    lam = ((h["CRVAL1"] +
            (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"]) *
           u.Unit(h["CUNIT1"])).to("AA").value

    flux_norm = flux_data / np.nanmedian(flux_data)
    noise_norm = noise_data / np.nanmedian(flux_data)

    # ==================
    # per-line windows: +-50 AA around each line's expected observed wavelength
    # ==================
    # Two variants of each window are kept: 'full' (only truly-missing/invalid
    # pixels dropped -- this is what gets plotted, so a deliberately-masked
    # region like Hgamma's noisy spike still shows its real data) and 'fit'
    # ('full' with mask_ranges additionally excluded -- this is what actually
    # gets fit, so the fitter never sees the masked-out region).
    windows = {}
    for name, rest_wl in rest.items():
        center = rest_wl * (1 + z_B)
        zoom = np.where((lam >= center - 50) & (lam <= center + 50))[0]
        lam_trim = lam[zoom]
        flux_trim = flux_norm[zoom]
        noise_trim = noise_norm[zoom]
        bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) |
               (noise_trim < 0))
        lam_full, flux_full, noise_full = lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]

        fit_bad = np.zeros(len(lam_full), dtype=bool)
        if name in mask_ranges:
            gap_lo, gap_hi, _ = mask_ranges[name]
            fit_bad |= (lam_full >= gap_lo) & (lam_full <= gap_hi)

        windows[name] = {
            'lam_full': lam_full,
            'flux_full': flux_full,
            'noise_full': noise_full,
            'lam_fit': lam_full[~fit_bad],
            'flux_fit': flux_full[~fit_bad],
            'noise_fit': noise_full[~fit_bad],
        }

    # concatenate the three windows' FIT arrays for the joint fit (excludes any
    # masked regions); plotting below uses each window's FULL arrays instead
    lam_all = np.concatenate([windows[n]['lam_fit'] for n in line_order])
    flux_all = np.concatenate([windows[n]['flux_fit'] for n in line_order])
    noise_all = np.concatenate([windows[n]['noise_fit'] for n in line_order])

    # ==================
    # index layout (fixed order, referenced by the .tied callables above)
    #  0: Halpha_A_central   1: Halpha_A_red
    #  2: Halpha_B_red       3: Halpha_B_central   4: Halpha_B_blue
    #  5: Hbeta_A_central (tied 0)   6: Hbeta_A_red (tied 1)
    #  7: Hbeta_B_red (tied 2)   8: Hbeta_B_central (tied 3)   9: Hbeta_B_blue (tied 4)
    # 10: Hgamma_A_central (tied 0)  11: Hgamma_A_red (tied 1)
    # 12: Hgamma_B_red (tied 2)  13: Hgamma_B_central (tied 3)  14: Hgamma_B_blue (tied 4)
    # 15: continuum_Halpha  16: continuum_Hbeta  17: continuum_Hgamma
    # ==================
    gaussians = []

    gaussians.append(build_master('Halpha_A_central', rest['Halpha']))  # 0
    gaussians.append(build_master('Halpha_A_red', rest['Halpha']))  # 1
    gaussians.append(build_master('Halpha_B_red', rest['Halpha']))  # 2
    gaussians.append(build_master('Halpha_B_central', rest['Halpha']))  # 3
    gaussians.append(build_master('Halpha_B_blue', rest['Halpha']))  # 4

    r_hb_ha = rest['Hbeta'] / rest['Halpha']
    gaussians.append(
        build_tied('Hbeta_A_central', 0, r_hb_ha, 6, (0, None)))  # 5
    gaussians.append(build_tied('Hbeta_A_red', 1, r_hb_ha, 1.5, (0, None)))  # 6
    gaussians.append(build_tied('Hbeta_B_red', 2, r_hb_ha, 1.5, (0, None)))  # 7
    gaussians.append(
        build_tied('Hbeta_B_central', 3, r_hb_ha, 4, (0, None)))  # 8
    gaussians.append(build_tied('Hbeta_B_blue', 4, r_hb_ha, 1.5, (0, None)))  # 9

    r_hg_ha = rest['Hgamma'] / rest['Halpha']
    gaussians.append(
        build_tied('Hgamma_A_central', 0, r_hg_ha, 3, (0, None)))  # 10
    gaussians.append(build_tied('Hgamma_A_red', 1, r_hg_ha, 1, (0, None)))  # 11
    gaussians.append(build_tied('Hgamma_B_red', 2, r_hg_ha, 1, (0, None)))  # 12
    gaussians.append(
        build_tied('Hgamma_B_central', 3, r_hg_ha, 2.5, (0, None)))  # 13
    gaussians.append(build_tied('Hgamma_B_blue', 4, r_hg_ha, 1, (0, None)))  # 14

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

    CONT_INDICES = {
        name: len(gaussians) + i
        for i, name in enumerate(line_order)
    }

    compound_model = reduce(operator.add, gaussians + continua)

    # ==================
    # fit jointly
    # ==================
    fitter = fitting.TRFLSQFitter(calc_uncertainties=True)
    bestfit_model = fitter(compound_model,
                           lam_all,
                           flux_all,
                           weights=1.0 / noise_all,
                           maxiter=5000)

    # sanity check on the tying: confirm each tied mean/stddev matches its
    # master scaled by the rest-wavelength ratio
    print("Tying sanity check (tied value / master value vs expected ratio):")
    tie_checks = [
        (5, 0, r_hb_ha), (6, 1, r_hb_ha), (7, 2, r_hb_ha), (8, 3, r_hb_ha),
        (9, 4, r_hb_ha), (10, 0, r_hg_ha), (11, 1, r_hg_ha), (12, 2, r_hg_ha),
        (13, 3, r_hg_ha), (14, 4, r_hg_ha)
    ]
    for tied_idx, ref_idx, ratio in tie_checks:
        mean_actual = getattr(bestfit_model, f'mean_{tied_idx}').value / getattr(
            bestfit_model, f'mean_{ref_idx}').value
        std_actual = getattr(bestfit_model,
                             f'stddev_{tied_idx}').value / getattr(
                                 bestfit_model, f'stddev_{ref_idx}').value
        print(f"  {tied_idx}<-{ref_idx}: mean ratio {mean_actual:.6f}, "
             f"stddev ratio {std_actual:.6f}, expected {ratio:.6f}")

    # ==================
    # bake tied params to fixed constants before saving, recording the tie
    # relationships as a plain-data dict instead. dill's pickling of live
    # callables (even class instances like MeanTie/StdTie, whose __call__
    # method dill can still serialize by value) has proven unreliable across
    # Python builds/environments -- confirmed corrupted bytecode causing a
    # segfault in one environment and `SystemError: unknown opcode` in
    # another. Once baked, the saved model carries zero callables; tie_map
    # (saved alongside it) lets flux_and_uncert add back each baked
    # component's exact analytic gradient contribution (d(stddev_tied)/
    # d(stddev_ref) = ratio, exactly, since the tie is a plain linear scaling).
    # ==================
    tie_map = {}
    for i in range(len(gaussians)):
        std_param = getattr(bestfit_model, f'stddev_{i}')
        if isinstance(std_param.tied, StdTie):
            tie_map[i] = {
                'stddev_ref': std_param.tied.ref_idx,
                'stddev_ratio': std_param.tied.line_ratio,
            }
        mean_param = getattr(bestfit_model, f'mean_{i}')
        if isinstance(mean_param.tied, MeanTie):
            tie_map.setdefault(i, {})
            tie_map[i]['mean_ref'] = mean_param.tied.ref_idx
            tie_map[i]['mean_ratio'] = mean_param.tied.line_ratio

    for i, entry in tie_map.items():
        if 'stddev_ref' in entry:
            std_param = getattr(bestfit_model, f'stddev_{i}')
            std_param.tied = False
            std_param.fixed = True
        if 'mean_ref' in entry:
            mean_param = getattr(bestfit_model, f'mean_{i}')
            mean_param.tied = False
            mean_param.fixed = True

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
        if name in mask_ranges:
            gap_lo, gap_hi, gap_label = mask_ranges[name]
            ax[0].axvspan(gap_lo, gap_hi, color='pink', alpha=0.2,
                          label=gap_label)
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
        ax[0].set_title(f"{name}  | Balmer-only joint tied fit |  z_A={z_A}  z_B={z_B}",
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

        fig.savefig(
            f'./output/joint_fit/Balmer/{name}_joint_tied_fit.png')
        with open(
                f'./output/joint_fit/Balmer/{name}_joint_tied_fit.pkl',
                'wb') as f:
            dill.dump({'model': bestfit_model, 'tie_map': tie_map}, f)
