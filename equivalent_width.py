import dill
import numpy as np
import astropy.units as u
from astropy.io import fits

# ======================================
# Halpha rest-frame EW lower limits for sources A and B
#
# Both lensed images share ONE 1D extracted spectrum, so Halpha's continuum
# (the flat Const1D fit alongside the 5 Gaussians, see infinite_gaussians.py /
# multiple_gaussian_integration.py) is a single blended level for A+B -- it
# can't be split per source the way the line flux can (via velocity offset).
# The spectrum is also close to flat/noise-level around Halpha, so the
# formal fit-covariance error on that continuum (which looks tight only
# because it's a least-squares average over many pixels) isn't trustworthy
# as a real detection.
#
# Instead: take an N-sigma UPPER bound on the continuum from the actual
# line-free pixels' scatter, and assign that entire (upper-bounded, still
# blended) continuum to a single source. Since the true per-source continuum
# can only be <= that combined bound, F_line / continuum_upper is guaranteed
# to sit at or below the true EW for either source -- a valid lower limit.
# ======================================

Z = {'A': 1.679, 'B': 1.677}
REST_HALPHA = 6562.819
MODEL_PKL = './output/improved_gaussians/Halpha/5_gaussian_constrained.pkl'
FLUXES_PKL = './output/improved_gaussians/Halpha/Halpha_fluxes.pkl'
WINDOW_HALFWIDTH = 50  # AA, observed frame -- matches the fit window convention used everywhere else in this codebase
N_SIGMA_WINDOW = 3  # the convention
SIGMA_LEVELS = [1, 2, 3]


def load_window(lam_center, halfwidth):
    """Same load/trim/clean of the stacked NIR spectrum used at fit time.
    Units stay normalized (flux/nanmedian(flux)) throughout -- that factor
    cancels in the F_line/continuum ratio below, so there's no need to
    undo it here."""
    with fits.open("./Data/X-Shooter/1D/stacked_NIR.fits") as hdu:
        h = hdu[1].header
        flux_data = hdu[1].data
        noise_data = hdu[4].data
    lam = ((h["CRVAL1"] +
            (np.arange(h["NAXIS1"]) + 1.0 - h["CRPIX1"]) * h["CDELT1"]) *
           u.Unit(h["CUNIT1"])).to("AA").value
    flux_norm = flux_data / np.nanmedian(flux_data)
    noise_norm = noise_data / np.nanmedian(flux_data)
    zoom = np.where((lam <= lam_center + halfwidth)
                    & (lam >= lam_center - halfwidth))[0]
    lam_trim, flux_trim, noise_trim = lam[zoom], flux_norm[zoom], noise_norm[
        zoom]
    bad = (~np.isfinite(flux_trim) | ~np.isfinite(noise_trim) |
           (noise_trim < 0))
    return lam_trim[~bad], flux_trim[~bad], noise_trim[~bad]


def continuum_upper_limits(model, lam_fit, flux_fit, noise_fit, n_lines):
    """Weighted mean + standard error of the line-free pixels in the fit
    window, using the ACTUAL data scatter rather than the fit's formal
    covariance -- see module docstring for why."""
    line_free = np.ones(len(lam_fit), dtype=bool)
    for i in range(n_lines):
        mean_i = model[i].mean.value
        std_i = model[i].stddev.value
        line_free &= np.abs(lam_fit - mean_i) > N_SIGMA_WINDOW * std_i

    weights = 1.0 / noise_fit[line_free]**2
    cont_mean = np.sum(weights * flux_fit[line_free]) / np.sum(weights)
    cont_sigma = 1.0 / np.sqrt(np.sum(weights))

    n_free = np.sum(line_free)
    print(f"line-free pixels used for continuum: {n_free}/{len(lam_fit)}")
    print(
        f"measured continuum (line-free scatter): {cont_mean:.4f} +/- {cont_sigma:.4f} (normalized units)"
    )

    cont_idx = model.n_submodels - 1
    fit_cont = model[cont_idx].amplitude.value
    names = model.cov_matrix.param_names
    cov = model.cov_matrix.cov_matrix
    idx = {n: k for k, n in enumerate(names)}
    fit_cont_sigma = np.sqrt(cov[idx[f'amplitude_{cont_idx}'],
                                 idx[f'amplitude_{cont_idx}']])
    print(
        f"fitted continuum (joint least-squares, for comparison): {fit_cont:.4f} +/- {fit_cont_sigma:.4f} (normalized units)"
    )

    return {n: cont_mean + n * cont_sigma for n in SIGMA_LEVELS}


# ======================================
# run
# ======================================
with open(MODEL_PKL, 'rb') as f:
    bestfit_model = dill.load(f)
with open(FLUXES_PKL, 'rb') as f:
    flux_result = dill.load(f)

lam_center = np.mean([bestfit_model[i].mean.value for i in range(5)])
lam_fit, flux_fit, noise_fit = load_window(lam_center, WINDOW_HALFWIDTH)
cont_upper = continuum_upper_limits(bestfit_model,
                                    lam_fit,
                                    flux_fit,
                                    noise_fit,
                                    n_lines=5)

results = {
    'continuum_upper_limits': cont_upper,
    'EW_obs_lower': {},
    'EW_rest_lower': {}
}
for source, z in Z.items():
    F_line = flux_result['fluxes'][source]
    results['EW_obs_lower'][source] = {}
    results['EW_rest_lower'][source] = {}
    print(
        f"\nSource {source} (Halpha flux = {F_line:.3f}, normalized-flux units):"
    )
    for n in SIGMA_LEVELS:
        ew_obs = F_line / cont_upper[n]
        ew_rest = ew_obs / (1 + z)
        results['EW_obs_lower'][source][n] = ew_obs
        results['EW_rest_lower'][source][n] = ew_rest
        print(
            f"  {n}sigma continuum upper limit -> EW_rest > {ew_rest:.2f} AA (EW_obs > {ew_obs:.2f} AA)"
        )

with open('./output/improved_gaussians/Halpha/Halpha_EW_lower_limits.pkl',
          'wb') as f:
    dill.dump(results, f)
