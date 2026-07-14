import os

import dill
import numpy as np

from multiple_gaussian_integration import load_model, load_tie_map, flux_and_uncert

# ======================================
# extra_constrained variant of multiple_gaussian_integration.py's LINES --
# same 8 joint-fit lines and component indices (extra_constrained_jointfit.py
# kept the identical 0-32 index layout), just pointed at
# output/joint_fit/extra_constrained/ instead of .../jointfit_all/. Hdelta/CIII
# are dropped -- those are unrelated single-line refits with no
# extra_constrained equivalent.
# ======================================
LINES = {
    'Halpha': {
        'pkl': './output/joint_fit/extra_constrained/Halpha_joint_tied_fit.pkl',
        'A_indices': [0, 1],
        'B_indices': [2, 3, 4],
        'save': './output/joint_fit/extra_constrained/fluxes/Halpha_fluxes.pkl',
    },
    'OIII5007': {
        'pkl':
        './output/joint_fit/extra_constrained/[OIII]5007_joint_tied_fit.pkl',
        'A_indices': [5, 6],
        'B_indices': [7, 8, 9],
        'save': './output/joint_fit/extra_constrained/fluxes/OIII5007_fluxes.pkl',
    },
    'OIII4959': {
        'pkl':
        './output/joint_fit/extra_constrained/[OIII]4959_joint_tied_fit.pkl',
        'A_indices': [10, 11],
        'B_indices': [12, 13, 14],
        'save': './output/joint_fit/extra_constrained/fluxes/OIII4959_fluxes.pkl',
    },
    'Hbeta': {
        'pkl': './output/joint_fit/extra_constrained/Hbeta_joint_tied_fit.pkl',
        'A_indices': [15, 16],
        'B_indices': [17, 18, 19],
        'save': './output/joint_fit/extra_constrained/fluxes/Hbeta_fluxes.pkl',
    },
    'Hgamma': {
        'pkl': './output/joint_fit/extra_constrained/Hgamma_joint_tied_fit.pkl',
        'A_indices': [20, 21],
        'B_indices': [22, 23, 24],
        'save': './output/joint_fit/extra_constrained/fluxes/Hgamma_fluxes.pkl',
    },
    'OII3726': {
        'pkl': './output/joint_fit/extra_constrained/[OII]_joint_tied_fit.pkl',
        'A_indices': [25],
        'B_indices': [26, 27],
        'save': './output/joint_fit/extra_constrained/fluxes/OII3726_fluxes.pkl',
    },
    'OII3729': {
        'pkl': './output/joint_fit/extra_constrained/[OII]_joint_tied_fit.pkl',
        'A_indices': [28],
        'B_indices': [29, 30],
        'save': './output/joint_fit/extra_constrained/fluxes/OII3729_fluxes.pkl',
    },
    'NII6584': {
        'pkl': './output/joint_fit/extra_constrained/[NII]6584_joint_tied_fit.pkl',
        'A_indices': [31],
        'B_indices': [32],
        'save': './output/joint_fit/extra_constrained/fluxes/NII6584_fluxes.pkl',
    },
}


# ======================================
# run for each line
# ======================================
if __name__ == "__main__":
    os.makedirs('./output/joint_fit/extra_constrained/fluxes', exist_ok=True)

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
