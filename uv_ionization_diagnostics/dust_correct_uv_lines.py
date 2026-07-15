import os

import dill
import numpy as np
import pandas as pd

from fit_uv_lines import rest

# ==================
# CCM89 (Cardelli, Clayton & Mathis 1989, ApJ 345, 245) UV-branch extinction
# law. dust_extinction.py's k(wl) only implements CCM89's optical/NIR branch
# (valid 1.1 <= x <= 3.3 um^-1, i.e. rest lambda ~3030-9091 AA) -- the lines
# dust-corrected here (1808-2325 AA, x ~ 4.30-5.53 um^-1) fall in CCM89's
# separate UV branch (3.3 <= x <= 8.0 um^-1) and need its own a(x)/b(x)
# polynomials. All wavelengths here have x < 5.9, so the FUV "bump" terms
# F_a/F_b (only nonzero for x >= 5.9) are omitted.
# ==================
R_v = 3.1  # same value used throughout dust_extinction.py


def k_uv(wl_AA, R_v=R_v):
    x = 1e4 / wl_AA  # 1/lambda[um], lambda in AA
    if np.any((x < 3.3) | (x > 8.0)):
        raise ValueError(f"x={x} um^-1 outside CCM89 UV branch validity (3.3-8.0 um^-1)")
    a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341)
    b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263)
    return R_v * a + b


# ==================
# dust correction: F_corrected = F_observed * 10^(0.4*k(lambda)*E(B-V)),
# with error propagation, identical formula/pattern to
# dust_extinction.py's flux_correct() (k there has zero uncertainty of its
# own, same here since wavelength and R_v are fixed constants).
# ==================
def flux_correct(flux, flux_err, k_line, EB_V, EB_V_err):
    exp = 0.4 * k_line * EB_V
    flux_corrected = flux * 10**exp
    rel_flux_err = flux_err / flux
    rel_EB_V_err = 0.4 * k_line * np.log(10) * EB_V_err
    flux_corrected_err = flux_corrected * np.sqrt(rel_flux_err**2 + rel_EB_V_err**2)
    return flux_corrected, flux_corrected_err


UV_DIAGDIR = os.path.dirname(os.path.abspath(__file__))
FLUXDIR = f'{UV_DIAGDIR}/output/fluxes'
DUST_FLUXDIR = f'{UV_DIAGDIR}/output/fluxes_dust_corrected'
os.makedirs(DUST_FLUXDIR, exist_ok=True)

# every line either ionization-parameter diagnostic needs, source A only
SIIII_LINES = ['[SiII]1808', '[SiIII]1883', '[SiIII]1892']
CIII_LINES = ['[CIII]1907', '[CIII]1909', '[CII]2325']
LINES_TO_CORRECT = SIIII_LINES + CIII_LINES

if __name__ == "__main__":
    with open(f'{FLUXDIR}/uv_line_fluxes.pkl', 'rb') as f:
        rows = dill.load(f)
    flux = {(r['line'], r['source']): r for r in rows}

    # source A's cumulative E(B-V) (Balmer-decrement-derived, one value per
    # source, see dust_extinction.py) -- every dust_corrected pkl carries an
    # identical copy, so any line's pkl works (matching
    # diagnostics_dust_corrected.py's precedent of pulling it from Halpha's).
    with open(f'{UV_DIAGDIR}/../output/dust_corrected/Halpha_dust_corrected.pkl', 'rb') as f:
        halpha_dust = dill.load(f)
    EBV = halpha_dust['E(B-V)_cumulative']['A']
    EBV_err = halpha_dust['E(B-V)_cumulative_err']['A']
    print(f"Source A cumulative E(B-V) = {EBV:.4f} +/- {EBV_err:.4f}")

    out_rows = []
    for line in LINES_TO_CORRECT:
        r = flux[(line, 'A')]
        k_line = k_uv(rest[line])

        # "dust correct the fluxes for the detections" -- an upper limit
        # passes through unchanged rather than being scaled by an uncertain
        # correction on top of an already-uncertain 3-sigma bound.
        if r['is_upper_limit']:
            flux_after, flux_after_err = r['flux'], r['flux_uncert']
            applied = False
        else:
            flux_after, flux_after_err = flux_correct(r['flux'], r['flux_uncert'],
                                                        k_line, EBV, EBV_err)
            applied = True

        out_rows.append({
            'line': line,
            'source': 'A',
            'flux_before': r['flux'],
            'flux_before_uncert': r['flux_uncert'],
            'flux_after': flux_after,
            'flux_after_uncert': flux_after_err,
            'is_upper_limit': r['is_upper_limit'],
            'dust_correction_applied': applied,
            'E(B-V)': EBV,
            'E(B-V)_uncert': EBV_err,
            'k_uv': k_line,
            'units': r['units'],
        })
        tag = '' if applied else '  [upper limit -- passed through unchanged]'
        print(f"{line:14s} A  flux: {r['flux']:.4g} +/- {r['flux_uncert']:.4g}  ->  "
              f"{flux_after:.4g} +/- {flux_after_err:.4g}  (k_uv={k_line:.3f}){tag}")

    df = pd.DataFrame(out_rows)
    df.to_csv(f'{DUST_FLUXDIR}/uv_line_fluxes_dust_corrected.csv', index=False)
    with open(f'{DUST_FLUXDIR}/uv_line_fluxes_dust_corrected.pkl', 'wb') as f:
        dill.dump(out_rows, f)

    print(f"\nSaved dust-corrected UV flux table to "
          f"{DUST_FLUXDIR}/uv_line_fluxes_dust_corrected.csv")
