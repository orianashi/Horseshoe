import os

import dill
import numpy as np
import pandas as pd
from astropy.constants import c as C_LIGHT

# ==================
# K19 (Kewley, Nicholls & Sutherland 2019, ARA&A) Table 1 bicubic surface
# fits for the UV ionization-parameter diagnostics:
#   z = A + Bx + Cy + Dxy + Ex^2 + Fy^2 + Gxy^2 + Hyx^2 + Ix^3 + Jy^3
#   x = log10(R), y = log(O/H)+12, z = log(U)
# R_CIII  = ([CIII]1907 + [CIII]1909) / [CII]2325
# R_SiIII = ([SiIII]1883 + [SiIII]1892) / [SiII]1808
# Coefficients transcribed via pdftotext directly off the published table
# (not OCR'd off the rendered page image) -- see the plan file for the
# extracted table text this was checked against.
# ==================
TABLE1 = {
    5.0: {
        'CIII': dict(A=-354.86, B=62.164, C=137.35, D=-15.604, E=-1.5532,
                     F=-17.901, G=0.9936, H=0.2048, I=0.0000, J=0.7778,
                     Zmin=7.63, Zmax=8.93, logUmin=-3.98, logUmax=-2.98, rms=1.69),
        'SiIII': dict(A=-271.55, B=68.254, C=102.23, D=-16.983, E=-2.7208,
                      F=-13.023, G=1.0647, H=0.3931, I=0.0000, J=0.5531,
                      Zmin=7.63, Zmax=8.93, logUmin=-3.98, logUmax=-2.73, rms=2.71),
    },
    7.0: {
        'CIII': dict(A=-415.93, B=49.915, C=157.61, D=-12.538, E=-0.7865,
                     F=-20.097, G=0.8021, H=0.1180, I=-0.0560, J=0.8551,
                     Zmin=7.63, Zmax=8.93, logUmin=-3.98, logUmax=-2.48, rms=2.43),
        'SiIII': dict(A=190.14, B=30.405, C=-67.224, D=-7.1308, E=-4.1040,
                      F=7.6694, G=0.4308, H=0.5029, I=0.1883, J=-0.2878,
                      Zmin=7.63, Zmax=8.93, logUmin=-3.98, logUmax=-2.48, rms=2.59),
    },
}

# 8.53/8.93 are the two abundance tracks requested up front. The other two
# are source A's own KD02 metallicity estimates from metallicity.py:
# line 115, n2ha_q3_e8(-1.657, ...)[0] -> 8.5400703 (low-metallicity branch,
# [NII]/Halpha), and line 121, n2o2_combined(-0.638, ...)[0] -> 8.83355295
# (high-metallicity branch, [NII]/[OII]).
METALLICITY_TRACKS = [8.53, 8.93, 8.5400703, 8.833552953198032]
PRESSURES = [5.0, 7.0]

LOG_C_CGS = np.log10(C_LIGHT.cgs.value)  # log(q) = log(U) + log10(c), c in cm/s (K19 eq. 8-9 / KD02 eq. 1-2)

OUTDIR = os.path.dirname(os.path.abspath(__file__)) + '/output'
FLUXDIR = f'{OUTDIR}/fluxes'
IPDIR = f'{OUTDIR}/ionization_parameter'
os.makedirs(IPDIR, exist_ok=True)


def bicubic(x, y, coef):
    return (coef['A'] + coef['B'] * x + coef['C'] * y + coef['D'] * x * y +
            coef['E'] * x**2 + coef['F'] * y**2 + coef['G'] * x * y**2 +
            coef['H'] * y * x**2 + coef['I'] * x**3 + coef['J'] * y**3)


def bicubic_dzdx(x, y, coef):
    """Partial derivative d(logU)/d(logR) at fixed metallicity y -- used to
    propagate the measured line-ratio uncertainty through the polynomial.
    Metallicity y is an assumed input here (not measured), so no
    propagation term for it is needed."""
    return (coef['B'] + coef['D'] * y + 2 * coef['E'] * x +
            coef['G'] * y**2 + 2 * coef['H'] * y * x + 3 * coef['I'] * x**2)


def ratio_with_uncert(flux_num_list, flux_denom_list):
    """Sum-then-divide flux ratio with uncertainties propagated in
    quadrature (matching diagnostics.py's ratios() helper). Returns
    (ratio, ratio_uncert, limit_type) where limit_type is 'measurement',
    'upper' (numerator has an upper limit -> ratio is an upper limit), or
    'lower' (denominator has an upper limit -> ratio is a lower limit). If
    both are limits, the ratio's sense is ambiguous and it's marked
    'undetermined'."""
    num = sum(f['flux'] for f in flux_num_list)
    denom = sum(f['flux'] for f in flux_denom_list)
    num_is_limit = any(f['is_upper_limit'] for f in flux_num_list)
    denom_is_limit = any(f['is_upper_limit'] for f in flux_denom_list)

    ratio = num / denom
    if num_is_limit and denom_is_limit:
        limit_type = 'undetermined'
    elif num_is_limit:
        limit_type = 'upper'
    elif denom_is_limit:
        limit_type = 'lower'
    else:
        limit_type = 'measurement'

    if limit_type == 'measurement':
        num_uncert = np.sqrt(sum(f['flux_uncert']**2 for f in flux_num_list))
        denom_uncert = np.sqrt(sum(f['flux_uncert']**2 for f in flux_denom_list))
        ratio_uncert = ratio * np.sqrt((num_uncert / num)**2 + (denom_uncert / denom)**2)
    else:
        ratio_uncert = np.nan

    return ratio, ratio_uncert, limit_type


def apply_calibration(ratio, ratio_uncert, limit_type, pressure, diagnostic, y):
    coef = TABLE1[pressure][diagnostic]
    x = np.log10(ratio)
    logU = bicubic(x, y, coef)

    # add the calibration's own quoted RMS scatter (%) in quadrature as a
    # systematic term, converted from a percent error on U to a dex error
    # on log(U): d(logU) = (1/ln10) * (dU/U) = (1/ln10) * (rms/100)
    logU_sys = (coef['rms'] / 100.0) / np.log(10)

    if limit_type == 'measurement' and np.isfinite(ratio_uncert):
        dzdx = bicubic_dzdx(x, y, coef)
        # d(logR)/dR = 1/(R ln10)
        dlogR = ratio_uncert / (ratio * np.log(10))
        logU_stat = abs(dzdx) * dlogR
        logU_uncert = np.sqrt(logU_stat**2 + logU_sys**2)
    else:
        # logU here is itself a bound (an upper/lower limit propagated from
        # the flux ratio limit), not a measurement, so it has no
        # statistical uncertainty of its own -- only the calibration's
        # intrinsic systematic scatter applies.
        logU_uncert = logU_sys

    in_range_y = coef['Zmin'] <= y <= coef['Zmax']
    in_range_U = coef['logUmin'] <= logU <= coef['logUmax']

    logq = logU + LOG_C_CGS
    logq_uncert = logU_uncert

    return {
        'logU': logU,
        'logU_uncert': logU_uncert,
        'logU_in_range': in_range_U and in_range_y,
        'logq': logq,
        'logq_uncert': logq_uncert,
    }


if __name__ == "__main__":
    with open(f'{FLUXDIR}/uv_line_fluxes.pkl', 'rb') as f:
        rows = dill.load(f)
    flux = {(r['line'], r['source']): r for r in rows}

    DIAGNOSTICS = {
        'CIII': {
            'num': ['[CIII]1907', '[CIII]1909'],
            'denom': ['[CII]2325'],
        },
        'SiIII': {
            'num': ['[SiIII]1883', '[SiIII]1892'],
            'denom': ['[SiII]1808'],
        },
    }

    out_rows = []
    for source in ['A']:
        for diag_name, cfg in DIAGNOSTICS.items():
            num_fluxes = [flux[(line, source)] for line in cfg['num']]
            denom_fluxes = [flux[(line, source)] for line in cfg['denom']]
            ratio, ratio_uncert, limit_type = ratio_with_uncert(num_fluxes, denom_fluxes)

            for pressure in PRESSURES:
                for y in METALLICITY_TRACKS:
                    result = apply_calibration(ratio, ratio_uncert, limit_type,
                                                pressure, diag_name, y)
                    out_rows.append({
                        'source': source,
                        'diagnostic': diag_name,
                        'pressure_logPk': pressure,
                        'metallicity_track': y,
                        'R': ratio,
                        'R_uncert': ratio_uncert,
                        'R_limit_type': limit_type,
                        'logU': result['logU'],
                        'logU_uncert': result['logU_uncert'],
                        'logU_in_valid_range': result['logU_in_range'],
                        'logq': result['logq'],
                        'logq_uncert': result['logq_uncert'],
                    })
                    tag = '' if limit_type == 'measurement' else f'  [ratio is {limit_type} limit]'
                    range_tag = '' if result['logU_in_range'] else '  [OUT OF K19 VALID RANGE]'
                    print(f"{source}  {diag_name:6s}  logP/k={pressure}  y={y}  "
                          f"R={ratio:.4g}{tag}  logU={result['logU']:.3f}+/-{result['logU_uncert']:.3f}  "
                          f"logq={result['logq']:.3f}+/-{result['logq_uncert']:.3f}{range_tag}")

    df = pd.DataFrame(out_rows)
    df.to_csv(f'{IPDIR}/uv_ionization_parameter.csv', index=False)
    with open(f'{IPDIR}/uv_ionization_parameter.pkl', 'wb') as f:
        dill.dump(out_rows, f)

    print(f"\nSaved ionization parameter table to {IPDIR}/uv_ionization_parameter.csv")
