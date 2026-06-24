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
# file downlaod and setup
# ==================

# redshifts
z_A = 1.679
z_B = 1.677

# balmer emission lines in air 
# NOTE: if you add more lines, make sure to edit the amplitude guesses!
lines = [6562.819, 4861.333, 4340.471]              # H_delta = 4101.742
line_names = ["H_alpha", "H_beta", "H_gamma"]

#spectrum
spec_lib = "./Data/X-Shooter/1D/stacked_NIR.fits"
with fits.open(spec_lib.format(arm="NIR")) as hdu:
    h = hdu[1].header 
    flux_data = hdu[1].data #flux
    noise_data = hdu[4].data  #noise 
# the header wavelength axis is in CUNIT1 (microns for X-Shooter NIR), so convert to Angstroms
lam = (
    (h["CRVAL1"] + (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"])
    * u.Unit(h["CUNIT1"])
).to("AA").value

# normalize 
flux_norm = flux_data/np.nanmedian(flux_data)
noise_norm = noise_data/np.nanmedian(flux_data)

# trim after normalization, since we want to trim to only area surrounding the emissions 
zoom = np.where((lam <= (max(lines)*(1+z_A) + 50)) & (lam >= (min(lines)*(1+z_A) - 50)))[0]
lam_trim = lam[zoom]
flux_trim = flux_norm[zoom]
noise_trim = noise_norm[zoom]

# delete nans, inf, and negative noises 
bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) | (noise_trim < 0))    # flags all inf, nan, and noise<0 as True 
lam_clean = lam_trim[~bad]                                                       # flips boolean so that we keep all the finite and positive values
flux_clean = flux_trim[~bad]
noise_clean = noise_trim[~bad]

# plot 
fig,ax = plt.subplots(figsize=(12,10))
ax.plot(lam_clean, flux_clean, color="blue", label="data", ds="steps")
ax.fill_between(lam_clean, flux_clean- noise_clean, flux_clean + noise_clean, color="grey", alpha=0.3, label="noise")
ax.set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax.set_ylabel("Normalised Flux [arbitary units]", fontsize=15)
ax.legend()
plt.show()


# ==================
# initial guesses and tying 
# ==================
# initial guesses for gaussian parameters 
gauss_guesses = {
    'A': { 'z_guess': z_A,
          'amplitudes': np.array([45,18,10]),
          'stddev': 1
    },
    'B': { 'z_guess' : z_B,
          'amplitudes': np.array([25,9,5]),
          'stddev':0.8 
    }
}


# tying functions -- we need to keep in mind that tying requires indexing from the full model 

ref_emission_index = 0                                                          # flags H_alpha as the strongest line that we will tie the other balmers to 
def create_mean_tie(full_model_ref_emission_idx, line_ratio):
    def tie_mean(model):
        ref_mean = getattr(model, f'mean_{full_model_ref_emission_idx}')
        return ref_mean * line_ratio
    return tie_mean                                                             # returns a function 

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
    
    # create gaussians for the emission lines of a source.
    # use a python LIST (not a numpy array) so that gs_A + gs_B + [continuum] concatenates
    # the components; numpy arrays would add element-wise and scramble the compound model.
    gaussians = []
    for i in range(len(lines)):                                  # = 3 right now
        gaussians.append(models.Gaussian1D(
            name = f'{line_names[i]}_{source_label}',
            mean = lines[i] * (gauss_params['z_guess']+1),
            amplitude=gauss_params['amplitudes'][i],
            stddev = gauss_params['stddev']))
    # store the proper tying method 
    for i in range(len(gaussians)):
        if i == ref_emission_index:
            continue
        line_ratio = lines[i] / lines[ref_emission_index]
        gaussians[i].mean.tied = create_mean_tie(ref_idx_full_model, line_ratio)    # sets the function that creates the mean tie 
        gaussians[i].stddev.tied = create_std_tie(ref_idx_full_model)                  
            
    # set bounds for where the mean of the reference line can be (we don't want to accidentally fit another line)
    ref_mean_guess = lines[ref_emission_index] * (1+gauss_params["z_guess"])
    gaussians[ref_emission_index].mean.bounds = (ref_mean_guess - 10, ref_mean_guess + 10)

    return gaussians
# ==================
# initialize!
# ==================
#initalize the gaussians for sources A and B
gs_A = create_gaussians("A",0)
gs_B = create_gaussians("B", len(lines))            # this way we can handle having a different amount of lines (e.g. adding H_delta or OIII)

# also make a continuum!
continuum = models.Const1D(                        # makes a flat baseline 
    amplitude=np.nanmedian(flux_clean),            # should we get the nanmed from the broader flux array?
    name="continuum"
)

# combine into a compound model which now has 5-6 versions of each parameter (e.g. compound_model.mean_5)
concat_gaussians = gs_A + gs_B + [continuum]
compound_model = reduce(operator.add, concat_gaussians) # creates compound models where we have mean_0, mean_1, ... mean_5  

# ==================
# optimize and save 
# ==================
# choose optimization method 
fitter = fitting.LevMarLSQFitter()

# run optimization 
bestfit_model = fitter(compound_model, lam_clean, flux_clean, maxiter=5000)

# plot 
lam_model = np.linspace(lam_clean[0], lam_clean[-1], 1000)
fig, ax = plt.subplots(nrows=2, height_ratios=[3, 1], sharex=True, figsize=(12, 10))

# plot the data 
ax[0].plot(lam_clean, flux_clean, c="black", label="data", ds="steps")
ax[0].fill_between(lam_clean, flux_clean - noise_clean, flux_clean + noise_clean, color="lightgrey", alpha=0.3, label="noise")

# plot the bestfit model
ax[0].plot(
    lam_model, bestfit_model(lam_model), color="orange", label="Bestfit"
)  

# Initial guess 
ax[0].plot(lam_model, compound_model(lam_model), c="green", label="Initial guess", alpha=0.5, ls=":")


# Set the x and y labels and add a legend
ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax[0].set_ylabel("Normalised Flux [arbitary units]", fontsize=15)
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
ax[1].legend(frameon=True)
plt.show()
# save 