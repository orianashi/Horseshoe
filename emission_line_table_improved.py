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

Halpha_new = load_pkl('./output/joint_fit/all_detections/fluxes/Halpha_fluxes.pkl')
Hbeta_new = load_pkl('./output/joint_fit/all_detections/fluxes/Hbeta_fluxes.pkl')
Hgamma_new = load_pkl('./output/joint_fit/all_detections/fluxes/Hgamma_fluxes.pkl')
OIII4959_new = load_pkl('./output/joint_fit/all_detections/fluxes/OIII4959_fluxes.pkl')
OIII5007_new = load_pkl('./output/joint_fit/all_detections/fluxes/OIII5007_fluxes.pkl')
OII3726_new = load_pkl('./output/joint_fit/all_detections/fluxes/OII3726_fluxes.pkl')
OII3729_new = load_pkl('./output/joint_fit/all_detections/fluxes/OII3729_fluxes.pkl')

NIIalphas = load_pkl('./output/NII/NII_Halpha_ratios.pkl')

A_ratios = load_pkl('./output/diagnostics/A_ratios_improved.pkl')
B_ratios = load_pkl('./output/diagnostics/B_ratios_improved.pkl')

# wing-specific Balmer decrements (wing_balmer_flux_ratios.py): source A has
# central/red_wing components, source B has red_wing/central/blue_wing
wing_balmer = load_pkl('./output/diagnostics/wing_balmer_fluxes.pkl')


# ==================
# helpers
# ==================
def flux_row(line, flux, flux_unc, notes=''):
    return dict(line=line, flux_ergs_s_cm2=flux, flux_uncert=flux_unc, notes=notes)


def build_flux_table(source):
    rows = []

    rows.append(flux_row('[OII]3726', OII3726_new['fluxes'][source],
                          OII3726_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))
    rows.append(flux_row('[OII]3729', OII3729_new['fluxes'][source],
                          OII3729_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))

    rows.append(flux_row('Hgamma', Hgamma_new['fluxes'][source], Hgamma_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))
    rows.append(flux_row('Hbeta', Hbeta_new['fluxes'][source], Hbeta_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))
    rows.append(flux_row('[OIII]4959', OIII4959_new['fluxes'][source], OIII4959_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))
    rows.append(flux_row('[OIII]5007', OIII5007_new['fluxes'][source], OIII5007_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))
    rows.append(flux_row('[NII]6548', NIIalphas['fluxes'][source][2], NIIalphas['flux_uncerts'][source][2],
                          'legacy single-gaussian fit (no joint refit)'))
    rows.append(flux_row('Halpha', Halpha_new['fluxes'][source], Halpha_new['flux_uncerts'][source],
                          'joint fit (joint_fit/all_detections)'))
    rows.append(flux_row('[NII]6583', NIIalphas['fluxes'][source][1], NIIalphas['flux_uncerts'][source][1],
                          'legacy single-gaussian fit (no joint refit)'))

    return pd.DataFrame(rows).round(3)


# the dimensionless flux ratios currently in the csv (each gets a log10
# column). E(B-V) is computed solely in dust_extinction.py, not here.
RATIO_KEYS = [
    'N2', 'O3N2', 'Halpha/Hbeta', 'Hgamma/Hbeta', 'R23', 'kk04_R23',
    '[OIII]/Hbeta'
]


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
    return pd.DataFrame(rows).round(3)


def build_wing_balmer_table(source):
    """One row per velocity component (wing) the source has -- source A:
    central/red_wing; source B: red_wing/central/blue_wing (see
    wing_balmer_flux_ratios.py) -- with each wing's Halpha/Hbeta/Hgamma
    fluxes and the two Balmer-decrement-relevant ratios already computed
    there."""
    rows = []
    for wing, wing_data in wing_balmer[source].items():
        rows.append(
            dict(wing=wing,
                 Halpha_flux=wing_data['Halpha']['flux'],
                 Halpha_flux_uncert=wing_data['Halpha']['flux_uncert'],
                 Hbeta_flux=wing_data['Hbeta']['flux'],
                 Hbeta_flux_uncert=wing_data['Hbeta']['flux_uncert'],
                 Hgamma_flux=wing_data['Hgamma']['flux'],
                 Hgamma_flux_uncert=wing_data['Hgamma']['flux_uncert'],
                 **{
                     'Halpha/Hbeta': wing_data['Halpha_Hbeta']['ratio'],
                     'Halpha/Hbeta_uncert':
                     wing_data['Halpha_Hbeta']['ratio_uncert'],
                     'Hgamma/Hbeta': wing_data['Hgamma_Hbeta']['ratio'],
                     'Hgamma/Hbeta_uncert':
                     wing_data['Hgamma_Hbeta']['ratio_uncert'],
                 }))
    return pd.DataFrame(rows).round(3)


# ==================
# build tables
# ==================
df_A_fluxes = build_flux_table('A')
df_B_fluxes = build_flux_table('B')
df_A_ratios = build_ratio_table(A_ratios)
df_B_ratios = build_ratio_table(B_ratios)
df_A_wing_balmer = build_wing_balmer_table('A')
df_B_wing_balmer = build_wing_balmer_table('B')

# ==================
# save
# ==================
df_A_fluxes.to_csv('./output/diagnostics/emission_lines_A_fluxes_improved.csv', index=False)
df_B_fluxes.to_csv('./output/diagnostics/emission_lines_B_fluxes_improved.csv', index=False)
df_A_ratios.to_csv('./output/diagnostics/emission_lines_A_ratios_improved.csv', index=False)
df_B_ratios.to_csv('./output/diagnostics/emission_lines_B_ratios_improved.csv', index=False)
df_A_wing_balmer.to_csv('./output/diagnostics/wing_balmer_decrements_A.csv', index=False)
df_B_wing_balmer.to_csv('./output/diagnostics/wing_balmer_decrements_B.csv', index=False)

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
print()
print("Source A wing-specific Balmer decrements:")
print(df_A_wing_balmer.to_string(index=False))
print()
print("Source B wing-specific Balmer decrements:")
print(df_B_wing_balmer.to_string(index=False))
