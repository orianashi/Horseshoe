import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from functools import reduce
import operator
from matplotlib.ticker import MultipleLocator

plt.ion()

# ==================
# file downlaod and setup
# ==================

# redshift
z_B = 1.677
z_A = 1.679

# emission lines - for the sake of fitting (since I don't know which is the actual galaxy vs outflow, should i not tie?)
line = 6562.819
line_name = 'H_alpha'

#spectrum
spec_lib = "./Data/X-Shooter/1D/stacked_NIR.fits"
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
zoom = np.where(((lam <= (line) * (1 + z_B) + 50))
                & (lam >= (line * (1 + z_B) - 50)))[0]
lam_trim = lam[zoom]
flux_trim = flux_norm[zoom]
noise_trim = noise_norm[zoom]

# delete nans, inf, and negative noises
bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) | (noise_trim < 0)
       )  # flags all inf, nan, and noise<0 as True
lam_clean = lam_trim[
    ~bad]  # flips boolean so that we keep all the finite and positive values
flux_clean = flux_trim[~bad]
noise_clean = noise_trim[~bad]

# =================
# Initial guesses for the galaxy and outflow
# =================
# X-Shooter NIR resolution at ~17569 AA: R~5600 -> sigma_instr = lam/(R*2.355) ~1.33 AA
sigma_instr = 17569 / (5600 * 2.355)

gauss_guesses = {
    'gal': {
        'z_guess': z_B,
        'amplitude': 12,
        'stddev': 4,
        'stddev_bounds': (sigma_instr, 40),
        'mean_range': 5,
    },
    'red_outflow': {
        'z_guess': z_B + 0.00037,
        'amplitude': 4,
        'stddev': 4.3,
        'stddev_bounds': (sigma_instr, 40.0),
        'mean_range': 5,
    },
    'blue_outflow': {
        'z_guess': z_B - 0.0009,
        'amplitude': 1,
        'stddev': 1.5,
        'stddev_bounds': (sigma_instr, 40),
        'mean_range': 4,
    },
    'A': {
        'z_guess': z_A,
        'amplitude': 40,
        'stddev': 3,
        'stddev_bounds': (sigma_instr, 40),
        'mean_range': 5,
    }
}

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
ax.axvline(line * (1 + gauss_guesses['gal']['z_guess']),
           color='red',
           ls='--',
           alpha=0.5,
           lw=0.8)
ax.axvline(line * (1 + gauss_guesses['blue_outflow']['z_guess']),
           color='red',
           ls='--',
           alpha=0.5,
           lw=0.8)
ax.axvline(line * (1 + gauss_guesses['red_outflow']['z_guess']),
           color='red',
           ls='--',
           alpha=0.5,
           lw=0.8)
ax.legend()
plt.show()


# =================
# Create gaussian function
# =================
def create_gaussians(source_label):
    gauss_params = gauss_guesses[source_label]

    # create gaussians for the emission lines of a source.
    # use a python LIST (not a numpy array) so that gs_A + gs_B + [continuum] concatenates
    # the components; numpy arrays would add element-wise and scramble the compound model.

    gaussians = []
    gaussians.append(
        models.Gaussian1D(name=f'{line_name}_{source_label}',
                          mean=line * (gauss_params['z_guess'] + 1),
                          amplitude=gauss_params['amplitude'],
                          stddev=gauss_params['stddev']))

    # set bounds for where the mean of the reference line can be
    ref_mean_guess = line * (1 + gauss_params["z_guess"])
    gaussians[0].amplitude.bounds = (0, None)
    gaussians[0].stddev.bounds = gauss_params['stddev_bounds']
    gaussians[0].mean.bounds = (ref_mean_guess - gauss_params['mean_range'],
                                ref_mean_guess + gauss_params['mean_range'])
    return gaussians


# ==================
# mask out unhelpful areas
# ==================
"""
# mask out source A's contaminating emission line
source_A_mask = (lam_clean < 17575) | (lam_clean > 17600)
lam_fit = lam_clean[source_A_mask]
flux_fit = flux_clean[source_A_mask]
noise_fit = noise_clean[source_A_mask]
"""

# ==================
# initialize!
# ==================
gs_B_gal = create_gaussians("gal")
gs_B_blue_outflow = create_gaussians("blue_outflow")
gs_B_red_outflow = create_gaussians("red_outflow")
gs_A = create_gaussians('A')

# also make a continuum
continuum = models.Const1D(amplitude=np.nanmedian(flux_clean),
                           name="continuum")  #makes a flat baseline

# combine into a compound model
concat_gaussians = gs_B_gal + gs_B_blue_outflow + gs_B_red_outflow + gs_A + [
    continuum 
]
compound_model = reduce(operator.add, concat_gaussians)

# ==================
# optimize and save
# ==================
# choose optimization method
fitter = fitting.TRFLSQFitter(calc_uncertainties=True)

# run optimization and error
bestfit_model = fitter(compound_model,
                       lam_clean,
                       flux_clean,
                       weights=1.0 / noise_clean,
                       maxiter=5000)

# plot
# sample finely enough to resolve the narrowest gaussian.
lam_model = np.linspace(lam_clean[0], lam_clean[-1], 30000)
fig, ax = plt.subplots(nrows=2,
                       height_ratios=[3, 1],
                       sharex=True,
                       figsize=(12, 10))

# plot the data
ax[0].plot(lam_clean, flux_clean, c="black", label="data", ds="steps")
ax[0].fill_between(lam_clean,
                   flux_clean - noise_clean,
                   flux_clean + noise_clean,
                   color="lightgrey",
                   alpha=0.3,
                   label="noise")

# plot the bestfit model
ax[0].plot(lam_model,
           bestfit_model(lam_model),
           color="orange",
           label="Bestfit total",
           lw=2)

# individual components
cont_m = bestfit_model[4] 
colors = {
    "gal": "purple",
    "blue_outflow": "royalblue",
    "red_outflow": "crimson",
    "A": "teal"
}
for i, label in enumerate(["gal", "blue_outflow","red_outflow", "A" ]): 
    comp = bestfit_model[i]
    ax[0].plot(lam_model,
               comp(lam_model) + cont_m(lam_model),
               color=colors[label],
               ls="--",
               lw=1.5,
               label=f"{label}  σ={comp.stddev.value:.2f} AA")

# Initial guess
ax[0].plot(lam_model,
           compound_model(lam_model),
           c="green",
           label="Initial guess",
           alpha=0.5,
           ls=":")

# Set the x and y labels and add a legend
ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax[0].set_ylabel("Normalised Flux [erg/s/cm2/AA]", fontsize=15)
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].legend(frameon=False)

# plot the residuals (data - model) in the second subplot
ax[1].scatter(
    lam_clean,
    flux_clean - bestfit_model(lam_clean),
    s=10,
    c="orange",
    label="flux - model residuals",
    alpha=0.5,
)
ax[1].axhline(0, ls='--', alpha=0.4)
ax[1].legend(frameon=True)
plt.show()

# save
fig.savefig('./output/double_gaussians/Halpha_triple_gaussians_sourceA_bounded.png')
with open('./output/double_gaussians/Halpha_triple_gaussians_sourceA_bounded.pkl',
          'wb') as f:
    dill.dump(bestfit_model, f)
