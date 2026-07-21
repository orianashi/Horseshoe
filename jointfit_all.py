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
    '[OIII]5007': 5006.843,
    '[OIII]4959': 4958.911,
    'Hbeta': 4861.333,
    'Hgamma': 4340.47,
    '[OII]3726': 3726.03,
    '[OII]3729': 3728.815,
    '[NII]6584': 6583.460,
}
# window/plot iteration order -- '[OII]' is a single combined window covering both
# [OII]3726 and [OII]3729 (see windows construction below), distinct from the two
# individual rest-wavelength keys above (used only for tie ratios)
line_order = ['Halpha', '[OIII]5007', '[OIII]4959', 'Hbeta', 'Hgamma', '[OII]']

# Hbeta's window contains a genuine gap of NaN flux/noise pixels between the
# A and B peaks (~13015.8-13020.6 AA observed, same gap masked in
# infinite_gaussians.py) -- excluded explicitly here (on top of the NaN
# filter below) so it's documented and also so it can be shaded on the plot.
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
# windowed continuum: a plain Const1D would leak into all windows once
# evaluated over the concatenated x array, so restrict each window's
# continuum to its own range
# ==================
def _windowed_const(x, amplitude=1.0, lo=0.0, hi=1.0):
    return np.where((x >= lo) & (x <= hi), amplitude, 0.0)


WindowedConst1D = custom_model(_windowed_const)

# ==================
# master components (free): Halpha's 5 components (A: central, red wing;
# B: red, central, blue). Everything else ties its mean/stddev to these.
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


A_INDICES = {
    'Halpha': [0, 1],
    '[OIII]5007': [5, 6],
    '[OIII]4959': [10, 11],
    'Hbeta': [15, 16],
    'Hgamma': [20, 21],
    '[OII]': [25, 28],
    '[NII]6584': [31],
}
B_INDICES = {
    'Halpha': [2, 3, 4],
    '[OIII]5007': [7, 8, 9],
    '[OIII]4959': [12, 13, 14],
    'Hbeta': [17, 18, 19],
    'Hgamma': [22, 23, 24],
    '[OII]': [26, 27, 29, 30],
    '[NII]6584': [],  # no fitted component for B -- see build_tied comment above
}

colors = {
    ('A', 'central'): 'yellow',
    ('B', 'central'): 'orange',
    ('A', 'red'): 'red',
    ('B', 'red'): 'red',
    ('B', 'blue'): 'blue',
}


def component_color(label):
    # [OII] components are single Gaussians per source (no central/red/blue
    # role, e.g. "[OII]3726_A"), so they fall back to the "central" color
    parts = label.split('_')
    if len(parts) == 2:
        _, source = parts
        return colors[(source, 'central')]
    _, source, role = parts
    return colors[(source, role)]


# ==================
# everything below actually loads data, fits, and writes output files --
# guarded so that importing this module (e.g. dill needing to resolve
# MeanTie/StdTie/WindowedConst1D while unpickling a saved fit elsewhere)
# only defines the classes/functions above and doesn't re-run the fit.
# ==================
if __name__ == "__main__":
    # ==================
    # wing-fit amplitudes for the blended OII lines ([OII]3726_A and
    # [OII]3729_B) -- pulled from oii_gaussian_fitting.py's already-fit
    # unblended wing model rather than re-fit here. amplitude_0 there is
    # [OII]3729_A (red wing, unblended), amplitude_1 is [OII]3726_B (blue
    # wing, unblended).
    # ==================
    with open('./output/OII/OII_wingfit_gaussians.pkl', 'rb') as f:
        wing_bestfit = dill.load(f)
    amp0_wing = wing_bestfit.amplitude_0.value
    amp1_wing = wing_bestfit.amplitude_1.value

    # ==================
    # load the shared NIR spectrum (Halpha, [OIII], Hbeta, Hgamma)
    # ==================
    spec_lib_nir = "./Data/X-Shooter/1D/stacked_NIR.fits"
    with fits.open(spec_lib_nir) as hdu:
        h = hdu[1].header
        flux_data = hdu[1].data  # flux
        noise_data = hdu[4].data  # noise
    lam = ((h["CRVAL1"] +
            (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"]) *
           u.Unit(h["CUNIT1"])).to("AA").value

    flux_norm = flux_data / np.nanmedian(flux_data)
    noise_norm = noise_data / np.nanmedian(flux_data)

    # ==================
    # load the VIS spectrum ([OII]3726/3729 redshift into the VIS arm,
    # unlike the other lines above)
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
    # per-line windows: +-50 AA around each line's expected observed wavelength
    # ==================
    # Two variants of each window are kept: 'full' (only truly-missing/invalid
    # pixels dropped -- this is what gets plotted, so a deliberately-masked
    # region like Hgamma's noisy spike still shows its real data) and 'fit'
    # ('full' with mask_ranges additionally excluded -- this is what actually
    # gets fit, so the fitter never sees the masked-out region).
    windows = {}
    for name in ['Halpha', '[OIII]5007', '[OIII]4959', 'Hbeta', 'Hgamma']:
        rest_wl = rest[name]
        if name == 'Halpha':
            # widened on the red side to also capture [NII]6584 (fit as extra
            # components tied to Halpha's central component, see below) --
            # avoids giving [NII] its own overlapping window/double-counted pixels
            lo_obs = rest_wl * (1 + z_B) - 50
            hi_obs = rest['[NII]6584'] * (1 + z_A) + 50
        else:
            center = rest_wl * (1 + z_B)
            lo_obs, hi_obs = center - 50, center + 50
        zoom = np.where((lam >= lo_obs) & (lam <= hi_obs))[0]
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

    # dedicated narrow window purely for a standalone [NII]6584 plot/save --
    # NOT part of `line_order` (the list concatenated into lam_all for the
    # joint fit): its pixels are already covered by Halpha's widened window
    # above, and its Gaussian components (indices 31, 32) are tied to and
    # fit jointly with Halpha there. Adding it to `line_order` would
    # double-count those pixels and add a second, unconstrained continuum
    # term overlapping Halpha's.
    nii_center = rest['[NII]6584'] * (1 + z_B)
    zoom_nii = np.where((lam >= nii_center - 30) & (lam <= nii_center + 30))[0]
    lam_trim = lam[zoom_nii]
    flux_trim = flux_norm[zoom_nii]
    noise_trim = noise_norm[zoom_nii]
    bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) |
           (noise_trim < 0))
    lam_full, flux_full, noise_full = lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]
    windows['[NII]6584'] = {
        'lam_full': lam_full,
        'flux_full': flux_full,
        'noise_full': noise_full,
    }

    # [OII]3726 and [OII]3729 are only ~7 AA apart in observed wavelength (vs
    # the 50 AA window half-width used above), so they share one combined
    # window/continuum instead of two heavily-overlapping ones. Bracketed by
    # 3726 at z_B (bluest edge) and 3729 at z_A (reddest edge).
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

    # concatenate all windows' FIT arrays for the joint fit (excludes any
    # masked regions); plotting below uses each window's FULL arrays instead
    lam_all = np.concatenate([windows[n]['lam_fit'] for n in line_order])
    flux_all = np.concatenate([windows[n]['flux_fit'] for n in line_order])
    noise_all = np.concatenate([windows[n]['noise_fit'] for n in line_order])

    # ==================
    # index layout (fixed order, referenced by the .tied callables above)
    #  0: Halpha_A_central   1: Halpha_A_red
    #  2: Halpha_B_red       3: Halpha_B_central   4: Halpha_B_blue
    #  5: [OIII]5007_A_central (tied 0)   6: [OIII]5007_A_red (tied 1)
    #  7: [OIII]5007_B_red (tied 2)   8: [OIII]5007_B_central (tied 3)   9: [OIII]5007_B_blue (tied 4)
    # 10: [OIII]4959_A_central (tied 0)  11: [OIII]4959_A_red (tied 1)
    # 12: [OIII]4959_B_red (tied 2)  13: [OIII]4959_B_central (tied 3)  14: [OIII]4959_B_blue (tied 4)
    # 15: Hbeta_A_central (tied 0)  16: Hbeta_A_red (tied 1) 17: Hbeta_B_red (tied 2)
    # 18: Hbeta_B_central (tied 3) 19: Hbeta_B_blue (tied 4)
    # 20: Hgamma_A_central (tied 0)  21: Hgamma_A_red (tied 1) 22: Hgamma_B_red (tied 2)
    # 23: Hgamma_B_central (tied 3) 24: Hgamma_B_blue (tied 4)
    # 25: [OII]3726_A (tied 0, wing amplitude bounds)
    # 26: [OII]3726_B_red (tied 2)  27: [OII]3726_B_central (tied 3)
    # 28: [OII]3729_A (tied 0)
    # 29: [OII]3729_B_red (tied 2)  30: [OII]3729_B_central (tied 3, wing amplitude bounds)
    # 31: [NII]6584_A (tied 0, single component -- no wing decomposition)
    # 32: [NII]6584_B (tied 3, single component -- no wing decomposition)
    # 33: continuum_Halpha  34: continuum_[OIII]5007  35: continuum_[OIII]4959
    # 36: continuum_Hbeta  37: continuum_Hgamma  38: continuum_[OII]
    # ==================
    gaussians = []

    gaussians.append(build_master('Halpha_A_central', rest['Halpha']))  # 0
    gaussians.append(build_master('Halpha_A_red', rest['Halpha']))  # 1
    gaussians.append(build_master('Halpha_B_red', rest['Halpha']))  # 2
    gaussians.append(build_master('Halpha_B_central', rest['Halpha']))  # 3
    gaussians.append(build_master('Halpha_B_blue', rest['Halpha']))  # 4

    r_OIII5_ha = rest['[OIII]5007'] / rest['Halpha']
    gaussians.append(
        build_tied('[OIII]5007_A_central', 0, r_OIII5_ha, 50, (0, None)))  # 5
    gaussians.append(build_tied('[OIII]5007_A_red', 1, r_OIII5_ha, 10, (0, None)))  # 6
    gaussians.append(build_tied('[OIII]5007_B_red', 2, r_OIII5_ha, 1, (0, None)))  # 7
    gaussians.append(
        build_tied('[OIII]5007_B_central', 3, r_OIII5_ha, 2.5, (0, None)))  # 8
    gaussians.append(build_tied('[OIII]5007_B_blue', 4, r_OIII5_ha, 1, (0, None)))  # 9

    r_OIII4_ha = rest['[OIII]4959'] / rest['Halpha']
    gaussians.append(
        build_tied('[OIII]4959_A_central', 0, r_OIII4_ha, 50, (0, None)))  # 10
    gaussians.append(build_tied('[OIII]4959_A_red', 1, r_OIII4_ha, 10, (0, None)))  # 11
    gaussians.append(build_tied('[OIII]4959_B_red', 2, r_OIII4_ha, 1, (0, None)))  # 12
    gaussians.append(
        build_tied('[OIII]4959_B_central', 3, r_OIII4_ha, 2.5, (0, None)))  # 13
    gaussians.append(build_tied('[OIII]4959_B_blue', 4, r_OIII4_ha, 1, (0, None)))  # 14

    r_hb_ha = rest['Hbeta'] / rest['Halpha']
    gaussians.append(
        build_tied('Hbeta_A_central', 0, r_hb_ha, 50, (0, None)))  # 15
    gaussians.append(build_tied('Hbeta_A_red', 1, r_hb_ha, 10, (0, None)))  # 16
    gaussians.append(build_tied('Hbeta_B_red', 2, r_hb_ha, 1, (0, None)))  # 17
    gaussians.append(
        build_tied('Hbeta_B_central', 3, r_hb_ha, 2.5, (0, None)))  # 18
    gaussians.append(build_tied('Hbeta_B_blue', 4, r_hb_ha, 1, (0, None)))  # 19

    r_hg_ha = rest['Hgamma'] / rest['Halpha']
    gaussians.append(
        build_tied('Hgamma_A_central', 0, r_hg_ha, 3, (0, None)))  # 20
    gaussians.append(build_tied('Hgamma_A_red', 1, r_hg_ha, 1, (0, None)))  # 21
    gaussians.append(build_tied('Hgamma_B_red', 2, r_hg_ha, 1, (0, None)))  # 22
    gaussians.append(
        build_tied('Hgamma_B_central', 3, r_hg_ha, 2.5, (0, None)))  # 23
    gaussians.append(build_tied('Hgamma_B_blue', 4, r_hg_ha, 1, (0, None)))  # 24

    # [OII]: source A stays a single Gaussian per line (no central/red/blue
    # decomposition), tied to Halpha's *_central component (ref_idx 0).
    # Source B now also gets a red-wing component (ref_idx 2), same as every
    # other line's B_red. [OII]3726_A and [OII]3729_B_central are blended
    # with each other, so their amplitude bounds replicate
    # oii_gaussian_fitting.py lines 220-233, derived from the unblended wing
    # fit's amplitudes.
    r_oii3726_ha = rest['[OII]3726'] / rest['Halpha']
    gaussians.append(
        build_tied('[OII]3726_A', 0, r_oii3726_ha, amp0_wing / 1.4,
                   (amp0_wing / 1.5, amp0_wing / 0.35)))  # 25
    gaussians.append(build_tied('[OII]3726_B_red', 2, r_oii3726_ha, 1, (0, None)))  # 26
    gaussians.append(
        build_tied('[OII]3726_B_central', 3, r_oii3726_ha, amp1_wing, (0, None)))  # 27

    r_oii3729_ha = rest['[OII]3729'] / rest['Halpha']
    gaussians.append(
        build_tied('[OII]3729_A', 0, r_oii3729_ha, amp0_wing, (0, None)))  # 28
    gaussians.append(build_tied('[OII]3729_B_red', 2, r_oii3729_ha, 1, (0, None)))  # 29
    gaussians.append(
        build_tied('[OII]3729_B_central', 3, r_oii3729_ha, amp1_wing * 1.4,
                   (amp0_wing * 0.35, amp1_wing * 1.5)))  # 30

    # [NII]6584: single Gaussian per source (no red/blue wing decomposition,
    # unlike Halpha/OIII/Hbeta/Hgamma), tied to Halpha's central component.
    # Amplitude guess scaled down from Halpha's own central-component guess
    # (12) to roughly match the ~1-2% NII/Halpha flux ratio seen in the
    # legacy fit (output/NII/NII_Halpha_ratios.pkl).
    # Source B gets NO fitted component here -- its NII is noise-dominated
    # (amplitude 0.089 vs. source A's 0.695 when it WAS fit), so rather than
    # let a noise-driven free amplitude sit in the joint fit, it's excluded
    # entirely and replaced below by a 3-sigma upper limit computed straight
    # from the local noise and Halpha_B_central's own fitted stddev (index
    # 3) scaled by this same tie ratio -- i.e. still "tied" to Halpha_B's
    # width, just not as a live fit parameter.
    r_nii_ha = rest['[NII]6584'] / rest['Halpha']
    gaussians.append(build_tied('[NII]6584_A', 0, r_nii_ha, 0.3, (0, None)))  # 31

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
    # [NII]6584 has no continuum of its own -- it shares Halpha's window/continuum
    CONT_INDICES['[NII]6584'] = CONT_INDICES['Halpha']

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

    # ==================
    # flag components whose amplitude collapsed to near-zero -- i.e. the
    # fitter found the data doesn't support that component (usually a wing)
    # ==================
    print("Components with amplitude < 0.5 (collapsed):")
    collapsed_any = False
    for i in range(len(gaussians)):
        amp = getattr(bestfit_model, f'amplitude_{i}').value
        if amp < 0.5:
            collapsed_any = True
            print(f"  {bestfit_model[i].name}: amplitude = {amp:.4f}")
    if not collapsed_any:
        print("  none")

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
    # Source B's [NII]6584 has no fitted component in this joint fit (see the
    # build_tied comment above) -- its noise-dominated flux is instead a
    # 3-sigma upper limit, using the same recipe as
    # uv_ionization_diagnostics/fit_uv_lines.py's resolve_detection(): 3x the
    # local continuum noise, integrated over an assumed line width via the
    # Gaussian flux formula amp*stddev*sqrt(2pi). The assumed width is
    # Halpha_B_central's own fitted stddev (index 3) scaled by the same
    # r_nii_ha tie ratio NII6584_A uses -- "tied" to Halpha_B's width without
    # NII_B ever being a live fit parameter. windows['[NII]6584']['noise_full']
    # is in the same flux_norm-normalized units as every amplitude in this
    # fit, so no unit conversion is needed.
    # ==================
    SQRT2PI_NII = np.sqrt(2 * np.pi)
    nii_b_local_noise = np.nanmedian(windows['[NII]6584']['noise_full'])
    nii_b_assumed_std = bestfit_model.stddev_3.value * r_nii_ha
    nii_b_flux_3sigma = 3 * nii_b_local_noise * nii_b_assumed_std * SQRT2PI_NII
    print(f"[NII]6584 source B treated as 3-sigma upper limit: "
          f"local_noise={nii_b_local_noise:.4g}, assumed_std={nii_b_assumed_std:.4g}, "
          f"flux_3sigma={nii_b_flux_3sigma:.4g}")

    # ==================
    # per-line plotting -- [NII]6584 is appended here (own dedicated
    # plot/save, matching every other line) even though it's excluded from
    # `line_order` itself (that list drives the joint lam_all fit array, and
    # NII's pixels/continuum are already covered via Halpha's window there)
    # ==================
    for name in line_order + ['[NII]6584']:
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
        ax[0].set_title(f"{name}  |  joint tied fit (all lines tied to Halpha)  |  z_A={z_A}  z_B={z_B}",
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
            f'./output/joint_fit/jointfit_all/{name}_joint_tied_fit.png')
        save_dict = {'model': bestfit_model, 'tie_map': tie_map}
        if name == '[NII]6584':
            save_dict['upper_limit_B'] = {'flux': nii_b_flux_3sigma, 'is_upper_limit': True}
        with open(
                f'./output/joint_fit/jointfit_all/{name}_joint_tied_fit.pkl',
                'wb') as f:
            dill.dump(save_dict, f)
