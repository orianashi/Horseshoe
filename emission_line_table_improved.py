import dill
import numpy as np
import pandas as pd

# ==================
# load pkl files
# improved (multi-gaussian, full-covariance) fluxes where available;
# legacy single-gaussian fluxes for lines with no improved refit yet
# (fluxes/ratios are pulled as already computed elsewhere, nothing is re-fit here)
# ==================
def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

Halpha_new = load_pkl('./output/improved_gaussians/Halpha/Halpha_fluxes.pkl')
Hbeta_new = load_pkl('./output/improved_gaussians/Hbeta/Hbeta_fluxes.pkl')
Hgamma_new = load_pkl('./output/improved_gaussians/Hgamma/Hgamma_fluxes.pkl')
OIII4959_new = load_pkl('./output/improved_gaussians/OIII4959/OIII4959_fluxes.pkl')
OIII5007_new = load_pkl('./output/improved_gaussians/OIII5007/OIII5007_fluxes.pkl')
OII_new = load_pkl('./output/improved_gaussians/OII/OII_fluxes.pkl')

NIIalphas = load_pkl('./output/NII/NII_Halpha_ratios.pkl')

A_ratios = load_pkl('./output/tabling/A_ratios_improved.pkl')
B_ratios = load_pkl('./output/tabling/B_ratios_improved.pkl')


# ==================
# helpers
# ==================
def flux_row(line, flux, flux_unc, notes=''):
    return dict(line=line, flux_ergs_s_cm2=flux, flux_uncert=flux_unc, notes=notes)


def build_flux_table(source):
    rows = []

    rows.append(flux_row('[OII]3726', OII_new['component_fluxes'][source][0],
                          OII_new['component_flux_uncerts'][source][0],
                          'improved multi-gaussian fit; directly fit'))
    rows.append(flux_row('[OII]3729', OII_new['component_fluxes'][source][1],
                          OII_new['component_flux_uncerts'][source][1],
                          'improved multi-gaussian fit; directly fit'))

    rows.append(flux_row('Hgamma', Hgamma_new['fluxes'][source], Hgamma_new['flux_uncerts'][source],
                          'improved multi-gaussian fit'))
    rows.append(flux_row('Hbeta', Hbeta_new['fluxes'][source], Hbeta_new['flux_uncerts'][source],
                          'improved multi-gaussian fit'))
    rows.append(flux_row('[OIII]4959', OIII4959_new['fluxes'][source], OIII4959_new['flux_uncerts'][source],
                          'improved multi-gaussian fit'))
    rows.append(flux_row('[OIII]5007', OIII5007_new['fluxes'][source], OIII5007_new['flux_uncerts'][source],
                          'improved multi-gaussian fit'))
    rows.append(flux_row('[NII]6548', NIIalphas['fluxes'][source][2], NIIalphas['flux_uncerts'][source][2],
                          'legacy single-gaussian fit (no improved refit)'))
    rows.append(flux_row('Halpha', Halpha_new['fluxes'][source], Halpha_new['flux_uncerts'][source],
                          'improved multi-gaussian fit'))
    rows.append(flux_row('[NII]6583', NIIalphas['fluxes'][source][1], NIIalphas['flux_uncerts'][source][1],
                          'legacy single-gaussian fit (no improved refit)'))

    return pd.DataFrame(rows).round(3)


# the dimensionless flux ratios currently in the csv (each gets a log10
# column); E(B-V) is a derived quantity (already itself proportional to a
# log, and can be negative), not a ratio, so it's listed separately with no
# log column.
RATIO_KEYS = [
    'N2', 'O3N2', 'Halpha/Hbeta', 'Hgamma/Hbeta', 'R23', 'kk04_R23',
    '[OIII]/Hbeta'
]
DERIVED_KEYS = ['E(B-V)']


def log_uncert(ratio, uncert):
    log_ratio = np.log10(ratio)
    log_ratio_uncert = 0.434 * uncert / ratio
    return log_ratio, log_ratio_uncert


def build_ratio_table(ratios_dict):
    rows = []
    for key in RATIO_KEYS:
        val = ratios_dict[key]
        unc = ratios_dict[f'{key}_err']
        log_val, log_unc = log_uncert(val, unc)
        rows.append(
            dict(ratio=key,
                 value=val,
                 uncert=unc,
                 log10_value=log_val,
                 log10_uncert=log_unc))
    for key in DERIVED_KEYS:
        rows.append(
            dict(ratio=key,
                 value=ratios_dict[key],
                 uncert=ratios_dict[f'{key}_err'],
                 log10_value=np.nan,
                 log10_uncert=np.nan))
    return pd.DataFrame(rows).round(3)


# ==================
# build tables
# ==================
df_A_fluxes = build_flux_table('A')
df_B_fluxes = build_flux_table('B')
df_A_ratios = build_ratio_table(A_ratios)
df_B_ratios = build_ratio_table(B_ratios)

# ==================
# save
# ==================
df_A_fluxes.to_csv('./output/tabling/emission_lines_A_fluxes_improved.csv', index=False)
df_B_fluxes.to_csv('./output/tabling/emission_lines_B_fluxes_improved.csv', index=False)
df_A_ratios.to_csv('./output/tabling/emission_lines_A_ratios_improved.csv', index=False)
df_B_ratios.to_csv('./output/tabling/emission_lines_B_ratios_improved.csv', index=False)

print("Source A fluxes:")
print(df_A_fluxes.to_string(index=False))
print()
print("Source B fluxes:")
print(df_B_fluxes.to_string(index=False))
print()
print("Source A ratios:")
print(df_A_ratios.to_string(index=False))
print()
print("Source B ratios:")
print(df_B_ratios.to_string(index=False))
