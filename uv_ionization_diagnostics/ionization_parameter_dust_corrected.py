import os

import dill
import numpy as np
import pandas as pd

from ionization_parameter import (TABLE1, PRESSURES,
                                   ratio_with_uncert, apply_calibration)

UV_DIAGDIR = os.path.dirname(os.path.abspath(__file__))
DUST_FLUXDIR = f'{UV_DIAGDIR}/output/fluxes_dust_corrected'
DUST_IPDIR = f'{UV_DIAGDIR}/output/ionization_parameter_dust_corrected'
os.makedirs(DUST_IPDIR, exist_ok=True)

DIAGNOSTICS = {
    'SiIII': {'num': ['[SiIII]1883', '[SiIII]1892'], 'denom': ['[SiII]1808']},
    'CIII': {'num': ['[CIII]1907', '[CIII]1909'], 'denom': ['[CII]2325']},
}
# scoped to this dust-corrected table only (not the shared
# ionization_parameter.py METALLICITY_TRACKS, which still feeds the main
# CIII+SiIII table): 8.53/8.93 swapped for 8.6/8.9 (KD02's own 0.5/1.0
# Zsolar values under the Anders & Grevesse 1989 scale their models were
# built on); 8.5400703/8.833552953198032 (source A's own KD02 metallicity.py
# estimates, lines 115/121) are kept as before.
METALLICITY_TRACKS = [8.6, 8.9, 8.5400703, 8.833552953198032]

if __name__ == "__main__":
    with open(f'{DUST_FLUXDIR}/uv_line_fluxes_dust_corrected.pkl', 'rb') as f:
        rows = dill.load(f)
    flux = {r['line']: r for r in rows}

    # ratio_with_uncert() reads 'flux'/'flux_uncert'/'is_upper_limit' -- the
    # dust-corrected table instead has 'flux_after'/'flux_after_uncert', so
    # remap into the shape it expects rather than duplicating that logic.
    def as_ratio_input(line):
        r = flux[line]
        return {'flux': r['flux_after'], 'flux_uncert': r['flux_after_uncert'],
                'is_upper_limit': r['is_upper_limit']}

    out_rows = []
    for diag_name, cfg in DIAGNOSTICS.items():
        num_fluxes = [as_ratio_input(line) for line in cfg['num']]
        denom_fluxes = [as_ratio_input(line) for line in cfg['denom']]
        ratio, ratio_uncert, limit_type = ratio_with_uncert(num_fluxes, denom_fluxes)

        EBV = flux[cfg['denom'][0]]['E(B-V)']
        EBV_err = flux[cfg['denom'][0]]['E(B-V)_uncert']

        for pressure in PRESSURES:
            for y in METALLICITY_TRACKS:
                result = apply_calibration(ratio, ratio_uncert, limit_type,
                                            pressure, diag_name, y)
                out_rows.append({
                    'source': 'A',
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
                    'dust_corrected': True,
                    'E(B-V)': EBV,
                    'E(B-V)_uncert': EBV_err,
                })
                tag = '' if limit_type == 'measurement' else f'  [ratio is {limit_type} limit]'
                range_tag = '' if result['logU_in_range'] else '  [OUT OF K19 VALID RANGE]'
                print(f"A  {diag_name:6s}  logP/k={pressure}  y={y}  "
                      f"R={ratio:.4g}{tag}  logU={result['logU']:.3f}+/-{result['logU_uncert']:.3f}  "
                      f"logq={result['logq']:.3f}+/-{result['logq_uncert']:.3f}{range_tag}  [DUST CORRECTED]")

    # Average the two independent diagnostics' logU/logq per (pressure,
    # metallicity_track) -- same treatment as ionization_parameter.py; see
    # that file's comment for why R/R_limit_type aren't simply averaged.
    by_key = {(r['diagnostic'], r['pressure_logPk'], r['metallicity_track']): r
              for r in out_rows}
    avg_rows = []
    for pressure in PRESSURES:
        for y in METALLICITY_TRACKS:
            ciii_row = by_key[('CIII', pressure, y)]
            siiii_row = by_key[('SiIII', pressure, y)]
            logU_avg = np.mean([ciii_row['logU'], siiii_row['logU']])
            logU_uncert_avg = 0.5 * np.sqrt(ciii_row['logU_uncert']**2 +
                                            siiii_row['logU_uncert']**2)
            logq_avg = np.mean([ciii_row['logq'], siiii_row['logq']])
            logq_uncert_avg = 0.5 * np.sqrt(ciii_row['logq_uncert']**2 +
                                            siiii_row['logq_uncert']**2)
            avg_rows.append({
                'source': 'A',
                'diagnostic': 'CIII+SiIII_avg',
                'pressure_logPk': pressure,
                'metallicity_track': y,
                'R': np.nan,
                'R_uncert': np.nan,
                'R_limit_type': f"CIII={ciii_row['R_limit_type']}, SiIII={siiii_row['R_limit_type']}",
                'logU': logU_avg,
                'logU_uncert': logU_uncert_avg,
                'logU_in_valid_range': ciii_row['logU_in_valid_range'] and siiii_row['logU_in_valid_range'],
                'logq': logq_avg,
                'logq_uncert': logq_uncert_avg,
                'dust_corrected': True,
                'E(B-V)': ciii_row['E(B-V)'],
                'E(B-V)_uncert': ciii_row['E(B-V)_uncert'],
            })
            print(f"A  CIII+SiIII_avg  logP/k={pressure}  y={y}  "
                  f"logU={logU_avg:.3f}+/-{logU_uncert_avg:.3f}  "
                  f"logq={logq_avg:.3f}+/-{logq_uncert_avg:.3f}  [DUST CORRECTED]")
    out_rows.extend(avg_rows)

    df = pd.DataFrame(out_rows)
    df.to_csv(f'{DUST_IPDIR}/uv_ionization_parameter_dust_corrected.csv', index=False)
    with open(f'{DUST_IPDIR}/uv_ionization_parameter_dust_corrected.pkl', 'wb') as f:
        dill.dump(out_rows, f)

    print(f"\nSaved dust-corrected ionization parameter table (SiIII + CIII) to "
          f"{DUST_IPDIR}/uv_ionization_parameter_dust_corrected.csv")
