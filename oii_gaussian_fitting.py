import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from functools import reduce
import operator

plt.ion()

# ==================
# file download and setup
# ==================

# redshifts
z_A = 1.679
z_B = 1.677

lines = [3726.03, 3728.815]
line_names = ["[OII]3726", "[OII]3729"]

#spectrum
spec_lib = "./Data/X-Shooter/1D/stacked_VIS.fits"
with fits.open(spec_lib.format(arm="NIR")) as hdu:
    h = hdu[1].header
    flux_data = hdu[1].data  #flux
    noise_data = hdu[4].data  #noise
# the header wavelength axis is in CUNIT1 (microns for X-Shooter NIR), so convert to Angstroms
lam = ((h["CRVAL1"] +
        (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"]) *
       u.Unit(h["CUNIT1"])).to("AA").value

# normalize
flux_norm = flux_data / np.nanmedian(flux_data)
noise_norm = noise_data / np.nanmedian(flux_data)

# trim after normalization, since we want to trim to only area surrounding the emissions
zoom = np.where((lam <= (max(lines) * (1 + z_A) + 50))
                & (lam >= (min(lines) * (1 + z_A) - 50)))[0]
lam_trim = lam[zoom]
flux_trim = flux_norm[zoom]
noise_trim = noise_norm[zoom]

# delete nans, inf, and negative noises
bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) | (noise_trim < 0)
       )  # flags all inf, nan, and noise<0 as True
lam_clean = lam_trim[~bad]
flux_clean = flux_trim[~bad]
noise_clean = noise_trim[~bad]

# plot
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(lam_clean, flux_clean, color="blue", label="data", ds="steps")
ax.fill_between(lam_clean,
                flux_clean - noise_clean,
                flux_clean + noise_clean,
                color="grey",
                alpha=0.3,
                label="noise")
ax.set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax.set_ylabel("Normalised Flux [ergs/s/cm2/AA]", fontsize=15)
ax.legend()
#plt.show()

# ==================
# initial guesses and tying
# ==================
gauss_guesses = {
    'A': {
        'z_guess': z_A,
        'amplitudes': np.array([10, 8]),
        'stddev': np.array([3, 3])
    },
    'B': {
        'z_guess': z_B,
        'amplitudes': np.array([5, 8]),
        'stddev': np.array([2, 3])
    }
}

ref_emission_index = 0


def create_mean_tie(full_model_ref_emission_idx, line_ratio):

    def tie_mean(model):
        ref_mean = getattr(model, f'mean_{full_model_ref_emission_idx}')
        return ref_mean * line_ratio

    return tie_mean


def create_std_tie(full_model_ref_emission_idx):

    def tie_std(model):
        return getattr(model, f'stddev_{full_model_ref_emission_idx}')

    return tie_std


# ==================
# define the function that creates the tied gaussians we need for one source
# ==================
def create_gaussians(source_label, idx_start_full_model):
    ref_idx_full_model = idx_start_full_model + ref_emission_index
    gauss_params = gauss_guesses[source_label]

    gaussians = []
    for i in range(len(lines)):
        gaussians.append(
            models.Gaussian1D(name=f'{line_names[i]}_{source_label}',
                              mean=lines[i] * (gauss_params['z_guess'] + 1),
                              amplitude=gauss_params['amplitudes'][i],
                              stddev=gauss_params['stddev'][i]))
    for i in range(len(gaussians)):
        if i == ref_emission_index:
            continue
        line_ratio = lines[i] / lines[ref_emission_index]
        gaussians[i].mean.tied = create_mean_tie(ref_idx_full_model, line_ratio)
        # tie stddev within each source so the blended components inherit a
        # well-constrained line width from the clean unblended line of that source
        gaussians[i].stddev.tied = create_std_tie(ref_idx_full_model)

    ref_mean_guess = lines[ref_emission_index] * (1 + gauss_params["z_guess"])
    gaussians[ref_emission_index].mean.bounds = (ref_mean_guess - 10,
                                                 ref_mean_guess + 10)

    return gaussians


# ==================
# initialize!
# ==================
gs_A = create_gaussians("A", 0)
gs_B = create_gaussians("B", len(lines))

continuum = models.Const1D(amplitude=np.nanmedian(flux_clean), name="continuum")

concat_gaussians = gs_A + gs_B + [continuum]
compound_model = reduce(operator.add, concat_gaussians)

# ==================
# two-stage fit to handle the 3726A / 3729B blend
#
# 3726A (source A) and 3729B (source B) land at ~the same observed wavelength
# (~9982 A), making them degenerate in a single-pass fit.
#
# Stage 1 — fit only the unblended wings:
#   red wing  (~9990 A): 3729A  → pins source A redshift (via mean tie) and stddev
#   blue wing (~9975 A): 3726B  → pins source B redshift and stddev
# Stage 2 — fix means and stddevs from stage 1, refit amplitudes over the full region
# ==================
fitter = fitting.TRFLSQFitter(calc_uncertainties=True)

# blend region boundaries (tune by eye if needed)
blend_lo = lines[0] * (1 + z_A) - 3   # just below 3726A peak
blend_hi = lines[1] * (1 + z_B) + 3   # just above 3729B peak

unblended = (lam_clean < blend_lo) | (lam_clean > blend_hi)
lam_wing   = lam_clean[unblended]
flux_wing  = flux_clean[unblended]
noise_wing = noise_clean[unblended]

# Stage 1: anchor means and stddevs from the clean wings
stage1_model = fitter(compound_model, lam_wing, flux_wing,
                      weights=1.0 / noise_wing, maxiter=5000)

# Stage 2: fix means and stddevs, refit amplitudes over the full region
for i in range(len(gs_A) + len(gs_B)):
    getattr(stage1_model, f'mean_{i}').fixed   = True
    getattr(stage1_model, f'stddev_{i}').fixed = True

bestfit_model = fitter(stage1_model, lam_clean, flux_clean,
                       weights=1.0 / noise_clean, maxiter=5000)

# unfix for inspection
for i in range(len(gs_A) + len(gs_B)):
    getattr(bestfit_model, f'mean_{i}').fixed   = False
    getattr(bestfit_model, f'stddev_{i}').fixed = False

# ==================
# plot
# ==================
lam_model = np.linspace(lam_clean[0], lam_clean[-1], 30000)
fig, ax = plt.subplots(nrows=2,
                       height_ratios=[3, 1],
                       sharex=True,
                       figsize=(12, 10))

ax[0].plot(lam_clean, flux_clean, c="black", label="data", ds="steps")
ax[0].fill_between(lam_clean,
                   flux_clean - noise_clean,
                   flux_clean + noise_clean,
                   color="lightgrey",
                   alpha=0.3,
                   label="noise")
ax[0].plot(lam_model, bestfit_model(lam_model), color="orange", label="Bestfit", ds='steps')
ax[0].plot(lam_model, compound_model(lam_model), c="green", label="Initial guess", alpha=0.5, ls=":")
ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax[0].set_ylabel("Normalised Flux [erg/s/cm2/AA]", fontsize=15)
ax[0].legend(frameon=False)

ax[1].scatter(lam_clean,
              flux_clean - bestfit_model(lam_clean),
              s=10, c="orange", label="flux - model residuals", alpha=0.5)
ax[1].legend(frameon=True)
plt.show()

# save
fig.savefig('./output/OII/OII_bestfit_gaussians.png')
with open('./output/OII/OII_bestfit_gaussians.pkl', 'wb') as f:
    dill.dump(bestfit_model, f)
