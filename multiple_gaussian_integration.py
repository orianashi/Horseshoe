import dill
import numpy as np

# ======================================
# initialize information
# ======================================
# NOTE : flux units are in ergs/s/cm^2 /Angstrom

# each line here is fit as a SUM of multiple gaussians per source (unlike
# gauss_integration.py / OII_integration.py, which each only have one gaussian
# per source). A_indices/B_indices give which gaussian components (indexed as
# in the compound model, i.e. amplitude_<i>/mean_<i>/stddev_<i>) belong to each
# source for that line's bestfit model.
LINES = {
    'Halpha': {
        'pkl': './output/improved_gaussians/Halpha/5_gaussian_constrained.pkl',
        'A_indices': [0, 1],
        'B_indices': [2, 3, 4],
        'save': './output/improved_gaussians/Halpha/Halpha_fluxes.pkl',
    },
    'Hbeta': {
        'pkl': './output/improved_gaussians/Hbeta/6_gaussian_constrained_halpha.pkl',
        'A_indices': [0, 1, 2],
        'B_indices': [3, 4, 5],
        'save': './output/improved_gaussians/Hbeta/Hbeta_fluxes.pkl',
    },
    'Hgamma': {
        'pkl': './output/improved_gaussians/Hgamma/6_gaussian_masked_constrained.pkl',
        'A_indices': [0, 1, 2],
        'B_indices': [3, 4, 5],
        'save': './output/improved_gaussians/Hgamma/Hgamma_fluxes.pkl',
    },
    'CIII': {
        'pkl': './output/improved_gaussians/CIII/CIII_bestfit_gaussians_1.pkl',
        'A_indices': [0, 1],
        'B_indices': [2, 3],
        'save': './output/improved_gaussians/CIII/CIII_fluxes.pkl',
    },
    'OIII5007': {
        'pkl': './output/improved_gaussians/OIII5007/6_gaussians_3.pkl',
        'A_indices': [0, 1, 2],
        'B_indices': [3, 4, 5],
        'save': './output/improved_gaussians/OIII5007/OIII5007_fluxes.pkl',
    },
    'OIII4959': {
        'pkl': './output/improved_gaussians/OIII4959/6_gaussian_constrained_5007_scaled.pkl',
        'A_indices': [0, 1, 2],
        'B_indices': [3, 4, 5],
        'save': './output/improved_gaussians/OIII4959/OIII4959_fluxes.pkl',
    },
}

SQRT2PI = np.sqrt(2 * np.pi)


def flux_and_uncert(bestfit_model, indices):
    """Total integrated flux (sum of gaussian areas: amp*stddev*sqrt(2pi)) for
    one or more gaussian components, with uncertainty propagated through the
    fitter's full parameter covariance matrix rather than combined-in-quadrature
    independently. This matters when multiple gaussians belong to the same
    source and overlap in wavelength (e.g. a narrow + broad decomposition of one
    line): the fitter can trade amplitude between them, making their parameters
    strongly (anti-)correlated, so summing their variances independently can
    badly over- or under-estimate the combined source's flux uncertainty.
    """
    names = bestfit_model.cov_matrix.param_names
    cov = bestfit_model.cov_matrix.cov_matrix
    idx = {n: k for k, n in enumerate(names)}

    total = 0.0
    grad = np.zeros(len(names))
    for i in indices:
        amp = getattr(bestfit_model, f'amplitude_{i}').value
        std = getattr(bestfit_model, f'stddev_{i}').value
        total += amp * std * SQRT2PI
        # a fixed mean/stddev (e.g. a component tied to another line's fit)
        # has no variance and isn't in the covariance matrix at all -- it
        # contributes zero to the propagated uncertainty, so just skip it.
        if f'amplitude_{i}' in idx:
            grad[idx[f'amplitude_{i}']] += std * SQRT2PI
        if f'stddev_{i}' in idx:
            grad[idx[f'stddev_{i}']] += amp * SQRT2PI

    uncert = np.sqrt(grad @ cov @ grad)
    return total, uncert


# ======================================
# run for each line
# ======================================
for line_name, cfg in LINES.items():
    with open(cfg['pkl'], 'rb') as f:
        bestfit_model = dill.load(f)

    flux_A, uncert_A = flux_and_uncert(bestfit_model, cfg['A_indices'])
    flux_B, uncert_B = flux_and_uncert(bestfit_model, cfg['B_indices'])

    component_fluxes = {'A': [], 'B': []}
    component_flux_uncerts = {'A': [], 'B': []}
    for source, indices in [('A', cfg['A_indices']), ('B', cfg['B_indices'])]:
        for i in indices:
            f_i, u_i = flux_and_uncert(bestfit_model, [i])
            component_fluxes[source].append(f_i)
            component_flux_uncerts[source].append(u_i)
    component_fluxes = {k: np.array(v) for k, v in component_fluxes.items()}
    component_flux_uncerts = {
        k: np.array(v)
        for k, v in component_flux_uncerts.items()
    }

    print(
        f"Flux for {line_name} source A: {flux_A:.3f} +/- {uncert_A:.3f} ergs/s/cm2"
    )
    print(
        f"Flux for {line_name} source B: {flux_B:.3f} +/- {uncert_B:.3f} ergs/s/cm2"
    )

    result = {
        'fluxes': {
            'A': flux_A,
            'B': flux_B
        },
        'flux_uncerts': {
            'A': uncert_A,
            'B': uncert_B
        },
        'component_fluxes': component_fluxes,
        'component_flux_uncerts': component_flux_uncerts,
        'units': 'ergs / s / cm2',
    }
    with open(cfg['save'], 'wb') as f:
        dill.dump(result, f)
