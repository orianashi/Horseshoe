import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import airvacuumvald as avv
from astropy.io import fits
from astropy.modeling import models, fitting
from functools import reduce
import operator

# you'll probably need to pip install the airvacuumvald package: pip install airvacuumvald
# pip is a package manager a bit like conda. If conda install airvacuumvald, doesn't work, try
# pip. If you don't have pip, you can install it with conda: conda install pip

plt.ion()  # set interactive plotting

"""
Simultaneously fit the three emission lines [OIII]_5006, [OIII]_4959 and Hbeta, for both galaxies
source_A and source_B with guassians (so 3 x 2 = 6 guassians in total). A single gaussian has 3
free parameters, the amplitude (A), mean (μ) and width (σ).
    --> the amplitude is the height of the curve's peak 
    --> the mean is the x-coordinate of the center of the peak 
    --> the width sigma is the standard deviation, which is where ~68% of the area under the curve falls within 

gaussian function f(x) = A * exp(-(x - μ)^2 / (2 * σ^2))

Therefore, our 6 gaussians have a total of 3 x 6 = 18 free parameters. That's quite a lot and in
the case of noisey data, we might not converge to a good solution. We can use our physics knowledge
to make some assumptions to reduce the number of free parameters:

1. Tieing the means (μ)
    We know how far apart the lines in each object are from each other from experimental physics, so we "tie" them together. that way all 3
    lines for each target have the same redshift. Therefore rather than having:
        μ_OIII5007A, μ_OIII4959A, μ_HbetaA, μ_OIII5007B, μ_OIII4959B, μ_HbetaB

    We instead just have:
        μ_OIII5007A, μ_OIII5007B, and the other 4 means are tied to these two.
        (e.g. μ_OIII4959A = μ_OIII5007A * 4959.0 / 5006.84)
    --> CHECK MY PAPERS FOR THE DERIVATION 

2. Tieing the velocity dispersions.
    We can make the assumption that the emission lines are all emitted from the same gas cloud(s),
    which is a reasonable assumption for these lines. Therefore, the velocity dispersion
    (gaussian width) of all three lines for each target should be the same. Therefore rather than
    having:
        σ_OIII5007A, σ_OIII4959A, σ_HbetaA, σ_OIII5007B, σ_OIII4959B, σ_HbetaB
    We instead just have:
        σ_OIII5007A, σ_OIII5007B, and the other 4 widths are the same as these two.
        (e.g. σ_OIII4959A = σ_HbetaA = σ_OIII5007A)

We've now reduced the number of free parameters from 18 to 6, which should make it easier to
converge to a good solution, even in the case of noisey data.

Note: It's the amplitudes we care most about! So we leave these as free parameters for each
individual line.

"""
# -------------------------------------------------------------------------------------------------
#                                Set up and File imports
# -------------------------------------------------------------------------------------------------
z_guessA, z_guessB = (1.679, 1.677)  # redshifts for both targets from Barone+2026

# Key emission lines in Angstroms
# ---------Hbeta--- [OIII]--[OIII]
lines = [4861.33, 4958.92, 5006.84] 
line_names = ["H_beta", "OIII_4959", "OIII_5007"] # can we just get these in air? the NMSU website has air for λ > 2000Å

# take a staked 1D spectrum
spec_lib = "./Data/X-Shooter/1D/stacked_NIR.fits"

with fits.open(spec_lib.format(arm="NIR")) as hdu:
    h = hdu[1].header  # extract the fits header information to get the wavelength array.
    flux_data = hdu[1].data  # extract the flux data
    noise_data = hdu[4].data  # extract the noise data

# normalise the flux and noise by the median flux value
flux_normed = flux_data / np.nanmedian(flux_data)
noise_normed = noise_data / np.nanmedian(flux_data)

# Xshooter *seem* to be in air wavelengths. we'll need to convert this to vaccum.

# the full wavelength array in air wavelengths
# --> INSPECT THIS FURTHER 
wave_air = (
    (
        ((np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"] + h["CRVAL1"])
        * u.Unit(h["CUNIT1"])
    )
    .to("AA")
    .value
)

# convert to air wavelenths using the airvacuumvald package
wave_vac = avv.air_to_vacuum(wave_air)



# no smoothing?


# take data chunk around wavelength range of interest to make fitting easier. The spectra are
# quite noisey so we only want to fit to the relevant chunks of the spectrum.
# roi = range of interest indicies.
# Use big enough chunks that it doesn't matter if the redshift A is used vs. B.
range_interest = np.where(
    (wave_vac >= (lines[0] - 50) * (z_guessA + 1))                  #brings it to the observed wavelength 
    & (wave_vac <= (lines[-1] + 50) * (z_guessA + 1)) 
)[0]                                                                #for 1d array, np.where returns a tuple with one array of the indices where the condition is true, so [0] extracts the array 

# extract the wavelength and flux arrays for the range of interest. Normalise the flux and noise.
wave = wave_vac[range_interest]                                     #trims the array to just the indices in range_interest
flux = flux_normed[range_interest]
noise = noise_normed[range_interest]

# Remove nans                                                       #so then these parts will just be blank when plotted, but it's better to remove so they don't mess up calculations 
wave = np.delete(wave, np.where(np.isnan(flux))) 
flux = np.delete(flux, np.where(np.isnan(flux)))
noise = np.delete(noise, np.where(np.isnan(noise)))                 #should we delete negative noise? 

# Have a look at the spectrum we're working with
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(wave, flux, c="xkcd:black", label="data", ds="steps")
ax.fill_between(wave, flux - noise, flux + noise, color="xkcd:grey", alpha=0.5, label="noise")
ax.set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax.set_ylabel("Normalised Flux [arbitary units]", fontsize=15)
ax.legend()


# -------------------------------------------------------------------------------------------------
#                                Set up the multi-Guassian model to fit
# -------------------------------------------------------------------------------------------------
# Here we set up the 6 Guassian components (3 for each source) that we will fit to the data.
# note that we need to provide initial guesses for the amplitude, mean and width of each gaussian.

# write a python dictionary of dictionaries to store the initial guess parameters for each source, for each line.
specs_info = {
    "A": {
        "z_guess": z_guessA,
        "amplitudes": [14, 31, 89],  # amplitude guess for each line, based on by-eye look at data - in order of Hbeta, OIII4958, OIII5007 
        "stdev": 1,  # width of the lines, based on by-eye look at data.
    },
    "B": {
        "z_guess": z_guessB,
        "amplitudes": [7, 10, 27],  # amplitude guess for each line, based on by-eye look at data
        "stdev": 1,  # width of the lines, based on by-eye look at data.
    },
}

# from a by-eye look at the data, OIII_5007 is the strongest emission line for both galaxies,
# so that will be our "reference anchor" line that we use to tie the means and widths of the other
# lines to.
ref_line_index = 2  # OIII_5007 is the anchor line


# Here are functions that will help us tie the means and widths of the lines together. 
# --------------------
def make_mean_tie(ref_line_index, line_ratio):
    """Tie the mean of one line to a reference line"""
    return lambda model: getattr(model, f"mean_{ref_line_index}") * line_ratio #where the line_ratio has the reference line's rest wavelength in the denominator 
    # Look inside model and grab the attribute whose name is "mean_2".
    # for example, if ref_line_index = 2 then the getattr line gets model.mean_2 

def make_stddev_tie(ref_line_index):
    """Tie the width of one line to a reference line"""
    return lambda model: getattr(model, f"stddev_{ref_line_index}")


# -------------------
# This is the function that creates the 3 Gaussian components for a given source,
# with the appropriate ties between the lines and widths
def make_source_gaussians(source_label, position_in_full_model):
    """Make the 3 Gaussian components for a given source
    Args:
        source_label (str): The label for the source. (Either A or B)
        position_in_full_model (int): The position of the source in the full model. (A starts at index 0, B starts at index 3)
            This is important because when tieing the means and widths of the lines, we need
            to know which gaussian component to tie with respect to the full model.

    Returns:
        list: A list of Gaussian components for the source.
    """
    # index of the reference line in the full model
    ref_index_full_model = position_in_full_model + ref_line_index 
    # for example, Hbeta of source B would be index 3 + 0 = 3 

    # Step 1. Initialise the 3 Gaussian components for the source, with the initial guesses for the amplitude, mean and width.
    # based on the label, take the appropriate dictionary
    spec = specs_info[source_label]  #this pulls out the dictionary of starting guess values for this source. 
    # for examples if source_label =="A" then spec = {  "z_guess": z_guessA, etc }

    #creates a list of 3 gaussians for source A
    #models is a module that was imported from Astropy, and it already has Gaussian construction in it
    #models.Gaussian1D returns an astropy object 
    gaussians = [ 
        models.Gaussian1D(                                              
            amplitude=spec["amplitudes"][i],
            mean=lines[i] * (1 + spec["z_guess"]),  # mean of the lines, based on redshift guess
            stddev=spec["stdev"],  # width of the lines, based on by-eye look at data.
            name=f"{line_names[i]}_{source_label}",
        )
        for i in range(3)
    ]

    # Step 2. 
    # Tie the means and widths of the lines to the reference line (OIII_5007)
    for i, g in enumerate(gaussians): # i is the line index, and g is the actual Gaussian parameter 
        if i == ref_line_index:
            continue  # skip the reference line, we want to fit this one freely

        line_ratio = lines[i] / lines[ref_line_index]
        # now tie the mean and width of this line to the reference line using it's
        # position in the full model
        g.mean.tied = make_mean_tie(ref_index_full_model, line_ratio)                  
        g.stddev.tied = make_stddev_tie(ref_index_full_model)
     # we need mean.tied not mean_tied so that the code doesn't think we have an attribute called mean_tied 
     # also, astropy already knows settings like g.mean.tied and g.mean.bounds --> so here we are just setting a property that astropy already has a slot for 
     # astropy stores a function on g.mean.tied so that during fitting, when it needs the value of g.mean it calls the function. this way the parameter is not independently fitted. 

    # Step 3. 
    # Set bounds on the parameters to help the fitter converge to a good solution.
    # For the mean, because we've tied the OIII_4959 and Hbeta lines to the OIII_5007 line,
    # we only need to set bounds on the OIII_5007 mean.
    ref_mean_guess = lines[ref_line_index] * (1 + spec["z_guess"])

    # We know the redshift pretty well, so set really tight bounds (+/- 10 Angstroms).
    gaussians[ref_line_index].mean.bounds = (ref_mean_guess - 10, ref_mean_guess + 10)

    return gaussians


# -------------------

# Now let's initialise the 6 Gaussian components for both sources, initialising it with the
# intitial guesses specified in the specs_info dictionary.
gs_A = make_source_gaussians(source_label="A", position_in_full_model=0)
gs_B = make_source_gaussians(source_label="B", position_in_full_model=3)

# We'll also add a constant continuum component to the model, to account for any continuum
# emission in the spectrum.
continuum = models.Const1D(
    amplitude=np.nanmedian(flux),
    name="continuum",
)
# OK now let's combine all our model components into a single model that we will fit to the data.
# NOTE that gs_A and gs_B are lists of gaussian components, so to add everything together we need to put the continuum model into a python list as well.
all_model_components = gs_A + gs_B + [continuum]                                    # basically this looks like [H_beta_A, OIII_4959_A, ...,continuum]
full_model = reduce(operator.add, all_model_components)                             # creates a compound astropy model, where now there are 6 means etc. so there would be full_model.mean_1 etc. 
# full_model is the reason that we need to have the indexing go to 5 for ref_index_full_model. make_source_gaussians() only creates 3 Gaussians at a time
# for tying to OIII 5006, we need the other means to be tied to full_model.mean_2 and full_model.mean_5 

# We need to choose a method of optimising our model. Let's just use a Levenberg-Marquardt
# least squares fitter. This is the way we find the best-fit parameters for our model by
# minimising the sum of the squares of the residuals between the data and the model.
fitter = fitting.LevMarLSQFitter()

# Use the chosen fitter algorithm and the data to find the best-fit parameters
# for our full model. maxiter means the maximum number of iterations the fitter will do before it gives up. See what happens if you change it to a smaller number, e.g. maxiter=50 (it should suck)
bestfit_model = fitter(full_model, wave, flux, maxiter=5000) # returns just the new flux array. 

# basically this recalculates the full_model by re-evaluating the 3 gaussians each inside gs_A and gs_B given slightly tweaked parameters. 
# a Gaussian1D object is like a reusable function with adjustable knobs, so fitter doesn't repeatedly re-initialize gaussian1d objects with make_gaussian 
# instead, it tweaks full_model by changing the Gaussian parameters and re-evaluating the gaussian1d objects flux values at wave. gs_A and gs_B each have 3 gaussian1d objects.
# then to calculate how good the fit is, fitter finds the difference of flux and full_model, and stops when the sum of the squares of residuals is lowest

""" How does the model indexing work?
1. Create source A Gaussians
   position_in_full_model = 0
   reference index will be 2

2. Create source B Gaussians
   position_in_full_model = 3
   reference index will be 5

3. Combine:
   all_model_components = gs_A + gs_B + [continuum]

4. Build compound model:
   full_model = reduce(operator.add, all_model_components)

5. Now full_model actually has:
   mean_0, mean_1, mean_2, mean_3, mean_4, mean_5

6. During fitting, Astropy evaluates tied functions using this full_model
 NOTE: 
    g.mean.tied = make_mean_tie(ref_index_full_model, line_ratio) stores a function that says something like: lambda model: getattr(model, "mean_5") * line_ratio
    At the point where we have gs_B = make_source_gaussians but didn't define full_model yet, Astropy is just storing the rule. It is not yet asking for model.mean_5.
 """

# -------------------------------------------------------------------------------------------------
#                               Plot the bestfit model
# -------------------------------------------------------------------------------------------------
# make a smooth wavelength array for plotting the model
wave_model = np.linspace(wave[0], wave[-1], 1000)

# Check the bestfit model.
fig, ax = plt.subplots(nrows=2, height_ratios=[3, 1], sharex=True, figsize=(12, 10))
# plot the data in black and noise as a shaded region
ax[0].plot(wave, flux, c="xkcd:black", label="data", ds="steps")
ax[0].fill_between(wave, flux - noise, flux + noise, color="xkcd:grey", alpha=0.5, label="noise")

# plot the bestfit model
ax[0].plot(
    wave_model, bestfit_model(wave_model), c="xkcd:burnt orange", label="Bestfit"
)  # plot the overall bestfit model

# It's also worth seeing what the initial guess model parameters looked like. If your fit isn't convering to a good solution, it might be because your initial guesses were too far off.
ax[0].plot(wave_model, full_model(wave_model), c="green", label="Initial guess", alpha=0.5, ls=":")


# Set the x and y labels and add a legend
ax[0].set_xlabel("Observed Wavelength [Angstroms]", fontsize=15)
ax[0].set_ylabel("Normalised Flux [arbitary units]", fontsize=15)
ax[0].legend(frameon=False)

# plot the residuals (data - model) in the second subplot
ax[1].scatter(
    wave,
    flux - bestfit_model(wave),
    s=10,
    c="xkcd:burnt orange",
    label="flux - model residuals",
    alpha=0.5,
)
ax[1].legend(frameon=True)

fig.savefig("./output/Tania_OIII5007_OIII4959_Hbeta_bestfit_model.png", bbox_inches="tight")

# -------------------------------------------------------------------------------------------------
#                               Save the bestfit model parameters
# -------------------------------------------------------------------------------------------------

# there are a few different ways to save the bestfit model parameters. Personally, I like
# to "pickle" them, using the `dill` package. This means you can save the entire python object to a
# file, and then load it back into python later. Think of it like pickling a vegetable - you
# preserve it for later use. This is useful if you want to use the bestfit model in another script,
# or if you want to make plots of the bestfit model later on.

# this says open a binary file for writing (wb) called OIII5007_OIII4959_Hbeta_bestfit_model.pkl
with open("./output/Tania_OIII5007_OIII4959_Hbeta_bestfit_model.pkl", "wb") as f:
    dill.dump(bestfit_model, f)

# Then, in another script, you can load the bestfit model back into python like this:
# with open("../outputs/OIII5007_OIII4959_Hbeta_bestfit_model.pkl", "rb") as f:
#     bestfit_model = dill.load(f)

# rb = "read binary" mode (i.e. read a binary file)
# wb = "write binary" mode (i.e. write a binary file)

