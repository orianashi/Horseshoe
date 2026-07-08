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
line = 4101.742
line_name = 'H_delta'

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
# no mask needed here: the region between the A and B peaks in this window
# is not a noisy-but-real spike (like the Hgamma case) -- it's a genuine gap
# of NaN flux/noise pixels (~13015.8-13020.6 AA observed), which is already
# dropped by the `bad` filter above. There's nothing extra to exclude.
# =================
lam_fit = lam_clean
flux_fit = flux_clean
noise_fit = noise_clean
"""
# =================
# tie source B's 3 velocity components to source B's Halpha decomposition
# =================
# Source B's Hbeta fit is degenerate on its own: one of its 3 components sits
# right at the edge of the NaN gap above, so its mean/width are unconstrained
# by data and it becomes near-perfectly correlated with its neighbor
# (amplitude correlation r ~ -0.998 in the untied fit). Halpha's B decomposition
# (output/improved_gaussians/Halpha/5_gaussian_constrained.pkl) doesn't suffer
# from this gap and is well constrained (~1.9% flux uncertainty), so we reuse
# its fitted redshifts/velocity-widths per component here, fixing mean+stddev
# and leaving only the amplitude (i.e. the Hbeta flux of that component) free.
with open('./output/improved_gaussians/Halpha/5_gaussian_constrained.pkl', 'rb') as f:
    halpha_model = dill.load(f)

rest_Halpha = 6562.819
c_kms = 299792.458
B_tied = []
for i in (2, 3, 4):  # Halpha's 3 source-B components
    mean_Ha = getattr(halpha_model, f'mean_{i}').value
    std_Ha = getattr(halpha_model, f'stddev_{i}').value
    z_i = mean_Ha / rest_Halpha - 1
    sigma_v = std_Ha / mean_Ha * c_kms  # velocity dispersion, same for every line
    mean_tied = line * (1 + z_i)
    std_tied = sigma_v / c_kms * mean_tied
    B_tied.append({'mean': mean_tied, 'stddev': std_tied})

# =================
# Initial guesses for the galaxy and outflow
# =================
# X-Shooter NIR resolution at ~17569 AA: R~5600 -> sigma_instr = lam/(R*2.355) ~1.33 AA
sigma_instr = 17569 / (5600 * 2.355)

"""
gauss_guesses = {
    '1': {
        # main core of source A
        'z_guess': z_A + 0.0001,
        'amplitude': 12,
        'stddev': 1.5,
        'stddev_bounds': (0, 3),
        'amplitude_bound': (0, None),
        'mean_range': 3,
    },
    '2': {
        'z_guess': z_A - 0.0002,
        'amplitude': 2,
        'stddev': 1.5,
        'stddev_bounds': (0, 3),
        'amplitude_bound': (0, 6),
        'mean_range': 3,
    },
    '3': {
        'z_guess': z_B,
        'amplitude': 5,
        'stddev': 1.5,
        'stddev_bounds': (0, 5),
        'amplitude_bound': (1, None),
        'mean_range': 3
    },
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
ax.axvline(line * (1 + gauss_guesses['1']['z_guess']),
           color='red',
           ls='--',
           alpha=0.5,
           lw=0.8)
ax.axvline(line * (1 + gauss_guesses['2']['z_guess']),
           color='red',
           ls='--',
           alpha=0.5,
           lw=0.8)
"""
for tied in B_tied:
    ax.axvline(tied['mean'], color='teal', ls='--', alpha=0.5, lw=0.8)
    
ax.axvspan(13015.8, 13020.6, color='pink', alpha=0.2, label='missing data (NaN gap)')
"""

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

    # set bounds for where the mean of the reference line can be, let everything else vary freely
    ref_mean_guess = line * (1 + gauss_params["z_guess"])
    gaussians[0].amplitude.bounds = gauss_params['amplitude_bound']
    gaussians[0].stddev.bounds = gauss_params['stddev_bounds']
    gaussians[0].mean.bounds = (ref_mean_guess - gauss_params['mean_range'],
                                ref_mean_guess + gauss_params['mean_range'])
    return gaussians


def create_tied_gaussian(source_label, mean_tied, stddev_tied, amplitude_guess,
                         amplitude_bound):
    """Source-B component whose mean/width are fixed to the value implied by
    Halpha's well-constrained decomposition (see B_tied above) -- only the
    amplitude (this line's flux for that velocity component) is fit."""
    g = models.Gaussian1D(name=f'{line_name}_{source_label}',
                          mean=mean_tied,
                          amplitude=amplitude_guess,
                          stddev=stddev_tied)
    g.mean.fixed = True
    g.stddev.fixed = True
    g.amplitude.bounds = amplitude_bound
    return [g]


# ==================
# initialize!
# ==================
gs_1 = create_gaussians('1')
gs_2 = create_gaussians("2")
gs_3 = create_gaussians("3")
"""
gs_4 = create_tied_gaussian('4', B_tied[0]['mean'], B_tied[0]['stddev'], 2.6, (0, 15))
gs_5 = create_tied_gaussian('5', B_tied[1]['mean'], B_tied[1]['stddev'], 2.9, (0, 15))
gs_6 = create_tied_gaussian('6', B_tied[2]['mean'], B_tied[2]['stddev'], 2.1, (0, 15))
"""

# also make a continuum
continuum = models.Const1D(amplitude=np.nanmedian(flux_fit),
                           name="continuum")  #makes a flat baseline

# combine into a compound model
concat_gaussians = gs_1 + gs_2 + gs_3 + [continuum]
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
                   color="dimgray",
                   alpha=0.8,
                   label="noise")

#ax[0].axvspan(13015.8, 13020.6, color='pink', alpha=0.2, label='missing data (NaN gap)')

# plot the bestfit model
ax[0].plot(lam_model,
           bestfit_model(lam_model),
           color="orange",
           label="Bestfit total",
           lw=2)

# individual components
cont_m = bestfit_model[3]
colors = {
    "1": "purple",
    "2": "royalblue",
    "3": "crimson",
    "4": "teal",
    "5": 'orange',
    "6": 'green',
}
for i, label in enumerate(["1", "2", '3']):
    comp = bestfit_model[i]
    ax[0].plot(lam_model,
               comp(lam_model) + cont_m(lam_model),
               color=colors[label],
               ls="--",
               lw=1.5,
               label=f"{label}  σ={comp.stddev.value:.2f} AA")
"""
# Initial guess
ax[0].plot(lam_model,
           compound_model(lam_model),
           c="green",
           label="Initial guess",
           alpha=0.5,
           ls=":")
"""
# Set the x and y labels and add a legend
ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax[0].set_ylabel("Normalised Flux [erg/s/cm2/AA]", fontsize=15)
ax[0].xaxis.set_minor_locator(MultipleLocator(1))
ax[0].legend(frameon=False)

# plot the residuals (data - model) in the second subplot, in units of
# sigma (i.e. the "pull") so we can read significance straight off the plot:
# if the model + noise are correct, these should scatter ~N(0,1)
# computed only on the masked-in points, since those are what the fitter saw
residual_sigma = (flux_clean - bestfit_model(lam_clean)) / noise_clean
ax[1].scatter(
    lam_clean,
    residual_sigma,
    s=10,
    c="orange",
    label="(flux - model)/noise",
    alpha=0.5,
)
ax[1].axhline(0, ls='--', alpha=0.4)
# +-1/2/3 sigma reference lines to eyeball how significant the residuals are
for level, style in [(1, ':'), (2, '--')]:
    ax[1].axhline(level, ls=style, color='red', alpha=0.3, lw=0.8)
    ax[1].axhline(-level, ls=style, color='red', alpha=0.3, lw=0.8)
ax[1].legend(frameon=True)
plt.show()

# quantify overall fit significance: reduced chi^2 should be ~1 if the model
# is a good fit and noise_fit is a correctly-calibrated 1-sigma uncertainty;
# >>1 means either the model is missing real structure or noise is underestimated
n_free = sum(1 for name in bestfit_model.param_names
             if not getattr(bestfit_model, name).fixed)
dof = len(lam_clean) - n_free
chi2 = np.sum(residual_sigma**2)
print(f"chi2 = {chi2:.1f}, dof = {dof}, reduced chi2 = {chi2 / dof:.2f}")

# restrict the within-Nsig stats to windows immediately around each fitted
# emission line (mean +- n_sigma_window*stddev), instead of the whole lam_clean
# array, since most of that array is line-free continuum that would dilute
# how well the lines themselves are fit
n_sigma_window = 3.5
near_line_mask = reduce(operator.or_, [
    np.abs(lam_clean - bestfit_model[i].mean.value)
    <= n_sigma_window * bestfit_model[i].stddev.value for i in range(3)
])
residual_sigma_near = residual_sigma[near_line_mask]

print(
    f"Points near emission lines (within {n_sigma_window}sigma of a line center): "
    f"{len(residual_sigma_near)} / {len(residual_sigma)}")
print(
    f"Within 1sig: {np.sum(np.abs(residual_sigma_near)<=1)} ({(100*(np.sum(np.abs(residual_sigma_near)<=1))/len(residual_sigma_near)):.2f}%), "
    f"Within 2sig: {np.sum(np.abs(residual_sigma_near) <= 2)} ({(100*(np.sum(np.abs(residual_sigma_near)<=2))/len(residual_sigma_near)):.2f}%), "
    f"Within 3sig: {np.sum(np.abs(residual_sigma_near) <= 3)} ({ (100*(np.sum(np.abs(residual_sigma_near)<=3))/len(residual_sigma_near)):.2f}%)"
)

# save
fig.savefig('./output/improved_gaussians/Hdelta/3_gaussian_constrained.png')
with open('./output/improved_gaussians/Hdelta/3_gaussian_constrained.pkl',
          'wb') as f:
    dill.dump(bestfit_model, f)
