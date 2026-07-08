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
        # joint fit of Halpha+Hbeta+Hgamma (see balmer_joint_gaussian_fitting.py):
        # source A has only 2 components (central, red wing); source B has 3
        # (red, central, blue wing). All three lines' matching components share
        # one compound model and one covariance matrix via real astropy `.tied`
        # mean/stddev (ratio of rest wavelengths), so flux_and_uncert's existing
        # `.tied` handling already propagates cross-line uncertainty correctly --
        # no separate `tie` Monte Carlo config needed.
        'pkl': './output/improved_gaussians/joint_fit/Halpha_joint_tied_fit.pkl',
        'A_indices': [0, 1],
        'B_indices': [2, 3, 4],
        'save': './output/improved_gaussians/Halpha/Halpha_fluxes.pkl',
    },
    'Hbeta': {
        'pkl': './output/improved_gaussians/joint_fit/Hbeta_joint_tied_fit.pkl',
        'A_indices': [5, 6, 7],
        'B_indices': [8, 9, 10],
        'save': './output/improved_gaussians/Hbeta/Hbeta_fluxes.pkl',
    },
    'Hgamma': {
        'pkl': './output/improved_gaussians/joint_fit/Hgamma_joint_tied_fit.pkl',
        'A_indices': [11, 12, 13],
        'B_indices': [14, 15, 16],
        'save': './output/improved_gaussians/Hgamma/Hgamma_fluxes.pkl',
    },
    'Hdelta': {
        'pkl': './output/improved_gaussians/Hdelta/2_gaussian.pkl',
        'A_indices': [0],
        'B_indices': [1],
        'save': './output/improved_gaussians/Hdelta/Hdelta_fluxes.pkl',
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
        'pkl':
        './output/improved_gaussians/OIII4959/6_gaussian_constrained_5007_scaled.pkl',
        'A_indices': [0, 1, 2],
        'B_indices': [3, 4, 5],
        'save': './output/improved_gaussians/OIII4959/OIII4959_fluxes.pkl',
    },
    'OII': {
        # components are [3726_A, 3729_A, 3726_B, 3729_B, continuum]; within
        # each source the two lines' stddevs are tied to each other (forced
        # to the same width), unlike every other line's model above.
        'pkl': './output/improved_gaussians/OII/OII_fullfit_gaussians.pkl',
        'A_indices': [0, 1],
        'B_indices': [2, 3],
        'save': './output/improved_gaussians/OII/OII_fluxes.pkl',
    },
}

SQRT2PI = np.sqrt(2 * np.pi)


def load_model(path):
    """OII_fullfit_gaussians.pkl stores {'model': ..., 'ratio_...': ...}
    (it also carries the fitted doublet ratios); every other line's pkl is
    the bare astropy model. Unwrap either into just the model."""
    with open(path, 'rb') as f:
        obj = dill.load(f)
    return obj['model'] if isinstance(obj, dict) else obj


def _eval_tied(tie_func, model):
    result = tie_func(model)
    return float(getattr(result, 'value', result))


def _tied_param_grad(model, tied_param, idx, rel_step=1e-6):
    """Finite-difference gradient of a tied parameter's resolved value with
    respect to each of the model's free parameters. Needed when a component
    is tied to another component *within the same model* (e.g. OII's doublet
    stddevs, or the cross-line Balmer ties in balmer_joint_gaussian_fitting.py)
    so it still inherits its share of that free parameter's uncertainty --
    the referenced parameter IS in this model's own covariance matrix.
    """
    tie_func = tied_param.tied
    grad = np.zeros(len(idx))
    for name, k in idx.items():
        free_param = getattr(model, name)
        orig = free_param.value
        step = rel_step * (abs(orig) if orig != 0 else 1.0)
        free_param.value = orig + step
        plus = _eval_tied(tie_func, model)
        free_param.value = orig - step
        minus = _eval_tied(tie_func, model)
        free_param.value = orig
        grad[k] = (plus - minus) / (2 * step)
    return grad


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
        amp_param = getattr(bestfit_model, f'amplitude_{i}')
        std_param = getattr(bestfit_model, f'stddev_{i}')
        amp, std = amp_param.value, std_param.value
        total += amp * std * SQRT2PI

        if amp_param.tied:
            grad += std * SQRT2PI * _tied_param_grad(bestfit_model, amp_param,
                                                     idx)
        elif f'amplitude_{i}' in idx:
            grad[idx[f'amplitude_{i}']] += std * SQRT2PI

        # a fixed mean/stddev (e.g. a component tied to another line's fit)
        # has no variance and isn't in the covariance matrix at all -- it
        # contributes zero to the propagated uncertainty, so just skip it.
        if std_param.tied:
            grad += amp * SQRT2PI * _tied_param_grad(bestfit_model, std_param,
                                                     idx)
        elif f'stddev_{i}' in idx:
            grad[idx[f'stddev_{i}']] += amp * SQRT2PI

    uncert = np.sqrt(grad @ cov @ grad)
    return total, uncert


def flux_and_uncert_diagonal(bestfit_model, indices):
    """Alternate, simplistic version of flux_and_uncert(): same total flux,
    but uncertainty from only the diagonal of the covariance matrix (each
    parameter's own variance), combined in quadrature -- the same recipe as
    legacy gauss_integration.py's integrate(), rather than propagating
    through the full covariance matrix. This ignores both the amp/stddev
    correlation within a component and the correlation between components,
    so for overlapping multi-gaussian components (see flux_and_uncert's
    docstring) it will generally over- or under-estimate the true
    uncertainty. Kept alongside flux_and_uncert() for comparison; a fixed or
    tied parameter has no diagonal entry and so contributes zero, same as in
    flux_and_uncert().
    """
    names = bestfit_model.cov_matrix.param_names
    diag = np.diag(bestfit_model.cov_matrix.cov_matrix)
    idx = {n: k for k, n in enumerate(names)}

    total = 0.0
    var_total = 0.0
    for i in indices:
        amp = getattr(bestfit_model, f'amplitude_{i}').value
        std = getattr(bestfit_model, f'stddev_{i}').value
        area = amp * std * SQRT2PI
        total += area

        amp_uncert = np.sqrt(
            diag[idx[f'amplitude_{i}']]) if f'amplitude_{i}' in idx else 0.0
        std_uncert = np.sqrt(
            diag[idx[f'stddev_{i}']]) if f'stddev_{i}' in idx else 0.0
        area_uncert = np.sqrt((amp_uncert / amp)**2 +
                              (std_uncert / std)**2) * area
        var_total += area_uncert**2

    uncert = np.sqrt(var_total)
    return total, uncert


# ======================================
# run for each line
# ======================================
for line_name, cfg in LINES.items():
    bestfit_model = load_model(cfg['pkl'])

    flux_A, uncert_A = flux_and_uncert(bestfit_model, cfg['A_indices'])
    flux_B, uncert_B = flux_and_uncert(bestfit_model, cfg['B_indices'])
    _, uncert_A_diag = flux_and_uncert_diagonal(bestfit_model,
                                                cfg['A_indices'])
    _, uncert_B_diag = flux_and_uncert_diagonal(bestfit_model,
                                                cfg['B_indices'])

    component_fluxes = {'A': [], 'B': []}
    component_flux_uncerts = {'A': [], 'B': []}
    component_flux_uncerts_diagonal = {'A': [], 'B': []}
    for source, indices in [('A', cfg['A_indices']), ('B', cfg['B_indices'])]:
        for i in indices:
            f_i, u_i = flux_and_uncert(bestfit_model, [i])
            _, u_i_diag = flux_and_uncert_diagonal(bestfit_model, [i])
            component_fluxes[source].append(f_i)
            component_flux_uncerts[source].append(u_i)
            component_flux_uncerts_diagonal[source].append(u_i_diag)
    component_fluxes = {k: np.array(v) for k, v in component_fluxes.items()}
    component_flux_uncerts = {
        k: np.array(v)
        for k, v in component_flux_uncerts.items()
    }
    component_flux_uncerts_diagonal = {
        k: np.array(v)
        for k, v in component_flux_uncerts_diagonal.items()
    }

    print(
        f"Flux for {line_name} source A: {flux_A:.3f} +/- {uncert_A:.3f} ergs/s/cm2"
    )
    print(
        f"Flux for {line_name} source B: {flux_B:.3f} +/- {uncert_B:.3f} ergs/s/cm2"
    )
    print(
        f"  [diagonal-only] source A: {flux_A:.3f} +/- {uncert_A_diag:.3f} ergs/s/cm2"
    )
    print(
        f"  [diagonal-only] source B: {flux_B:.3f} +/- {uncert_B_diag:.3f} ergs/s/cm2"
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
        'flux_uncerts_diagonal': {
            'A': uncert_A_diag,
            'B': uncert_B_diag
        },
        'component_fluxes': component_fluxes,
        'component_flux_uncerts': component_flux_uncerts,
        'component_flux_uncerts_diagonal': component_flux_uncerts_diagonal,
        'units': 'ergs / s / cm2',
    }
    with open(cfg['save'], 'wb') as f:
        dill.dump(result, f)
