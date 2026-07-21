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
    # Halpha/[OIII]5007/[OIII]4959/Hbeta/Hgamma/[OII]3726/[OII]3729/[NII]6584
    # are all one joint fit (see jointfit_all.py): every line's mean/stddev is
    # tied to Halpha as master. Source A has 2 components (central, red wing)
    # for every line except [OII] (single role-less component for A) and
    # [NII]6584 (also a single role-less component for A, no wing
    # decomposition); source B has 3 (red, central, blue wing) for every line
    # except [OII] (2: red, central; no blue wing) and [NII]6584 (also a
    # single component for B). All eight lines' matching components share one
    # compound model and one covariance matrix via baked ties (see
    # tie_map/load_tie_map), so flux_and_uncert's tie_map handling already
    # propagates cross-line uncertainty correctly -- no separate `tie` Monte
    # Carlo config needed.
    'Halpha': {
        'pkl': './output/joint_fit/jointfit_all/Halpha_joint_tied_fit.pkl',
        'A_indices': [0, 1],
        'B_indices': [2, 3, 4],
        'save': './output/joint_fit/jointfit_all/fluxes/Halpha_fluxes.pkl',
    },
    'OIII5007': {
        'pkl':
        './output/joint_fit/jointfit_all/[OIII]5007_joint_tied_fit.pkl',
        'A_indices': [5, 6],
        'B_indices': [7, 8, 9],
        'save': './output/joint_fit/jointfit_all/fluxes/OIII5007_fluxes.pkl',
    },
    'OIII4959': {
        'pkl':
        './output/joint_fit/jointfit_all/[OIII]4959_joint_tied_fit.pkl',
        'A_indices': [10, 11],
        'B_indices': [12, 13, 14],
        'save': './output/joint_fit/jointfit_all/fluxes/OIII4959_fluxes.pkl',
    },
    'Hbeta': {
        'pkl': './output/joint_fit/jointfit_all/Hbeta_joint_tied_fit.pkl',
        'A_indices': [15, 16],
        'B_indices': [17, 18, 19],
        'save': './output/joint_fit/jointfit_all/fluxes/Hbeta_fluxes.pkl',
    },
    'Hgamma': {
        'pkl': './output/joint_fit/jointfit_all/Hgamma_joint_tied_fit.pkl',
        'A_indices': [20, 21],
        'B_indices': [22, 23, 24],
        'save': './output/joint_fit/jointfit_all/fluxes/Hgamma_fluxes.pkl',
    },
    # [OII]3726 and [OII]3729 share one saved model/window (see
    # jointfit_all.py's combined '[OII]' window) -- both entries point at the
    # same pkl, just with each line's own component indices.
    'OII3726': {
        'pkl': './output/joint_fit/jointfit_all/[OII]_joint_tied_fit.pkl',
        'A_indices': [25],
        'B_indices': [26, 27],
        'save': './output/joint_fit/jointfit_all/fluxes/OII3726_fluxes.pkl',
    },
    'OII3729': {
        'pkl': './output/joint_fit/jointfit_all/[OII]_joint_tied_fit.pkl',
        'A_indices': [28],
        'B_indices': [29, 30],
        'save': './output/joint_fit/jointfit_all/fluxes/OII3729_fluxes.pkl',
    },
    # single component per source (no wing decomposition) -- see
    # jointfit_all.py's [NII]6584_A/[NII]6584_B, tied to Halpha's central
    # component and fit within Halpha's own (widened) window.
    'NII6584': {
        'pkl': './output/joint_fit/jointfit_all/[NII]6584_joint_tied_fit.pkl',
        'A_indices': [31],
        # source B has no fitted component (see jointfit_all.py) -- flux_and_uncert
        # with an empty index list is a safe no-op (returns (0.0, 0.0)); B's
        # real value is substituted below from the pkl's 'upper_limit_B'.
        'B_indices': [],
        'save': './output/joint_fit/jointfit_all/fluxes/NII6584_fluxes.pkl',
    },
    # Hdelta/CIII have no joint fit yet -- unrelated single-line refits.
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
}

SQRT2PI = np.sqrt(2 * np.pi)


def load_model(path):
    """The joint_fit/all_detections pkls (jointfit_all.py) store
    {'model': ..., ...extra metadata...}; every other line's pkl is the bare
    astropy model. Unwrap either into just the model."""
    with open(path, 'rb') as f:
        obj = dill.load(f)
    return obj['model'] if isinstance(obj, dict) else obj


def load_tie_map(path):
    """The joint_fit/all_detections pkls (jointfit_all.py) store a
    plain-data 'tie_map' alongside the model: {index: {'stddev_ref': ref_idx,
    'stddev_ratio': ratio, ...}} for every component whose stddev was tied
    to another component during the fit. Those ties are baked to `.fixed`
    constants before saving rather than kept as live `.tied` callables --
    dill's pickling of callables (closures, and even class-instance __call__
    methods) has proven unreliable across Python builds/environments
    (confirmed: corrupted bytecode causing a segfault in one environment,
    `SystemError: unknown opcode` in another). flux_and_uncert uses this
    dict to add back each baked component's exact analytic gradient
    contribution. Lines without a tie_map (i.e. every pkl that isn't a dict,
    or a dict without this key) return {}.
    """
    with open(path, 'rb') as f:
        obj = dill.load(f)
    return obj.get('tie_map', {}) if isinstance(obj, dict) else {}


def _eval_tied(tie_func, model):
    result = tie_func(model)
    return float(getattr(result, 'value', result))


def _tied_param_grad(model, tied_param, idx, rel_step=1e-6):
    """Finite-difference gradient of a tied parameter's resolved value with
    respect to each of the model's free parameters. Needed when a component
    is tied to another component *within the same model* via a live `.tied`
    callable (e.g. OII's doublet stddevs) so it still inherits its share of
    that free parameter's uncertainty -- the referenced parameter IS in this
    model's own covariance matrix. The Balmer joint fit no longer uses this
    path (its ties are baked and propagated exactly via tie_map instead, see
    load_tie_map) since live callables don't survive pickling reliably.
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


def flux_and_uncert(bestfit_model, indices, tie_map=None):
    """Total integrated flux (sum of gaussian areas: amp*stddev*sqrt(2pi)) for
    one or more gaussian components, with uncertainty propagated through the
    fitter's full parameter covariance matrix rather than combined-in-quadrature
    independently. This matters when multiple gaussians belong to the same
    source and overlap in wavelength (e.g. a narrow + broad decomposition of one
    line): the fitter can trade amplitude between them, making their parameters
    strongly (anti-)correlated, so summing their variances independently can
    badly over- or under-estimate the combined source's flux uncertainty.

    tie_map (see load_tie_map) covers components whose stddev was baked to a
    `.fixed` constant tied to another component by a plain ratio -- since the
    tie is exactly linear (stddev_i = ratio * stddev_ref), its gradient
    contribution is just `ratio`, added directly to the ref component's slot
    rather than via a live `.tied` callable. Defaults to {} (a no-op) for
    every line that doesn't use this (everything except the Balmer joint fit).
    """
    names = bestfit_model.cov_matrix.param_names
    cov = bestfit_model.cov_matrix.cov_matrix
    idx = {n: k for k, n in enumerate(names)}
    tie_map = tie_map or {}

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
        # contributes zero to the propagated uncertainty unless tie_map says
        # otherwise, so just skip it in that case.
        stddev_tie = tie_map.get(i, {})
        if 'stddev_ref' in stddev_tie:
            ref_name = f"stddev_{stddev_tie['stddev_ref']}"
            grad[idx[ref_name]] += amp * SQRT2PI * stddev_tie['stddev_ratio']
        elif std_param.tied:
            grad += amp * SQRT2PI * _tied_param_grad(bestfit_model, std_param,
                                                     idx)
        elif f'stddev_{i}' in idx:
            grad[idx[f'stddev_{i}']] += amp * SQRT2PI

    uncert = np.sqrt(grad @ cov @ grad)
    return total, uncert


# ======================================
# run for each line
# ======================================
if __name__ == "__main__":
    for line_name, cfg in LINES.items():
        bestfit_model = load_model(cfg['pkl'])
        tie_map = load_tie_map(cfg['pkl'])

        flux_A, uncert_A = flux_and_uncert(bestfit_model, cfg['A_indices'],
                                           tie_map)
        flux_B, uncert_B = flux_and_uncert(bestfit_model, cfg['B_indices'],
                                           tie_map)

        component_fluxes = {'A': [], 'B': []}
        component_flux_uncerts = {'A': [], 'B': []}
        for source, indices in [('A', cfg['A_indices']),
                                ('B', cfg['B_indices'])]:
            for i in indices:
                f_i, u_i = flux_and_uncert(bestfit_model, [i], tie_map)
                component_fluxes[source].append(f_i)
                component_flux_uncerts[source].append(u_i)
        component_fluxes = {
            k: np.array(v)
            for k, v in component_fluxes.items()
        }
        component_flux_uncerts = {
            k: np.array(v)
            for k, v in component_flux_uncerts.items()
        }

        # source B's [NII]6584 is noise-dominated -- jointfit_all.py computes
        # a 3-sigma upper limit and stores it alongside the model/tie_map
        # (see its 'upper_limit_B' comment); swap it in here in place of the
        # normal fitted flux, with NaN uncertainty since a limit isn't a
        # symmetric-error measurement.
        is_upper_limit = {'A': False, 'B': False}
        if line_name == 'NII6584':
            with open(cfg['pkl'], 'rb') as f:
                pkl_obj = dill.load(f)
            if isinstance(pkl_obj, dict) and 'upper_limit_B' in pkl_obj:
                flux_B = pkl_obj['upper_limit_B']['flux']
                uncert_B = np.nan
                component_fluxes['B'] = np.array([flux_B])
                component_flux_uncerts['B'] = np.array([np.nan])
                is_upper_limit['B'] = pkl_obj['upper_limit_B']['is_upper_limit']

        print(
            f"Flux for {line_name} source A: {flux_A:.3f} +/- {uncert_A:.3f} ergs/s/cm2"
        )
        print(
            f"Flux for {line_name} source B: {flux_B:.3f} +/- {uncert_B:.3f} ergs/s/cm2"
            f"{'  [3-sigma UPPER LIMIT]' if is_upper_limit['B'] else ''}"
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
        # scoped to NII6584 only -- no schema change for the other 8 lines
        if line_name == 'NII6584':
            result['is_upper_limit'] = is_upper_limit
        with open(cfg['save'], 'wb') as f:
            dill.dump(result, f)
