import dill
import numpy as np
import pandas as pd
import diagnostics as diag

# ==================
# helpers
# ==================
def integrate(amp, stddev, amp_uncert, stddev_uncert):
    area = amp * stddev * np.sqrt(2 * np.pi)
    area_uncert = np.sqrt((amp_uncert / amp)**2 + (stddev_uncert / stddev)**2) * area
    return area, area_uncert

def get_std(model, param_name):
    """Return the uncertainty for a parameter by name; nan if tied/fixed."""
    try:
        return model.stds[param_name]
    except KeyError:
        return np.nan

def row(line, flux, flux_unc, stddev, stddev_unc, notes=''):
    return dict(line=line,
                flux_ergs_s_cm2=flux,
                flux_uncert=flux_unc,
                stddev_AA=stddev,
                stddev_uncert_AA=stddev_unc,
                notes=notes)

# ==================
# load models
# ==================
with open('./output/OIII/OIII_Hbeta_bestfit_gaussians.pkl', 'rb') as f:
    m_oiii = dill.load(f)
with open('./output/NII/Halpha_NII_bestfit_gaussians.pkl', 'rb') as f:
    m_nii = dill.load(f)
with open('./output/balmer/ori_balmer_bestfit_gaussians.pkl', 'rb') as f:
    m_balmer = dill.load(f)
with open('./output/OII/OII_wingfit_gaussians.pkl', 'rb') as f:
    m_oii = dill.load(f)

# ==================
# source A
# ==================
rows_A = []

# [OII]3729_A — directly fit (index 0 in OII model)
amp  = m_oii.amplitude_0.value;  std  = m_oii.stddev_0.value
aunc = get_std(m_oii, 'amplitude_0');  sunc = get_std(m_oii, 'stddev_0')
flux_3729A, flux_3729A_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('[OII]3726', flux_3729A / 1.5, flux_3729A_unc / 1.5,
                  np.nan, np.nan, 'derived from [OII]3729A / 1.5 (low-density ratio)'))
rows_A.append(row('[OII]3729', flux_3729A, flux_3729A_unc, std, sunc, 'directly fit'))

# Hgamma — from balmer fit, index 2 source A
amp  = m_balmer.amplitude_2.value;  std  = m_balmer.stddev_2.value
aunc = get_std(m_balmer, 'amplitude_2');  sunc = get_std(m_balmer, 'stddev_2')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('Hgamma', flux, flux_unc, std, sunc))

# Hbeta — from OIII fit, index 0 source A (reference)
amp  = m_oiii.amplitude_0.value;  std  = m_oiii.stddev_0.value
aunc = get_std(m_oiii, 'amplitude_0');  sunc = get_std(m_oiii, 'stddev_0')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('Hbeta', flux, flux_unc, std, sunc))

# [OIII]4959 — OIII fit, index 1 source A
amp  = m_oiii.amplitude_1.value;  std  = m_oiii.stddev_1.value
aunc = get_std(m_oiii, 'amplitude_1');  sunc = get_std(m_oiii, 'stddev_1')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('[OIII]4959', flux, flux_unc, std, sunc))

# [OIII]5007 — OIII fit, index 2 source A
amp  = m_oiii.amplitude_2.value;  std  = m_oiii.stddev_2.value
aunc = get_std(m_oiii, 'amplitude_2');  sunc = get_std(m_oiii, 'stddev_2')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('[OIII]5007', flux, flux_unc, std, sunc))

# [NII]6548 — NII fit, index 2 source A
amp  = m_nii.amplitude_2.value;  std  = m_nii.stddev_2.value
aunc = get_std(m_nii, 'amplitude_2');  sunc = get_std(m_nii, 'stddev_2')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('[NII]6548', flux, flux_unc, std, sunc))

# Halpha — NII fit, index 0 source A (reference)
amp  = m_nii.amplitude_0.value;  std  = m_nii.stddev_0.value
aunc = get_std(m_nii, 'amplitude_0');  sunc = get_std(m_nii, 'stddev_0')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('Halpha', flux, flux_unc, std, sunc))

# [NII]6583 — NII fit, index 1 source A
amp  = m_nii.amplitude_1.value;  std  = m_nii.stddev_1.value
aunc = get_std(m_nii, 'amplitude_1');  sunc = get_std(m_nii, 'stddev_1')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_A.append(row('[NII]6583', flux, flux_unc, std, sunc))

# diagnostic ratios (from diagnostics.py)
rows_A.append(row('R23', diag.R23_A, diag.R23_A_err, np.nan, np.nan,
                  'ratio, not a flux: ([OII]3726+[OII]3729+[OIII]4959+[OIII]5007)/Hbeta'))
rows_A.append(row('R23_KK04', diag.kR23_A, diag.kR23_A_err, np.nan, np.nan,
                  'ratio, not a flux: ([OII]3726+[OIII]4959+[OIII]5007)/Hbeta (Kewley & Dopita 2002)'))
rows_A.append(row('N2', diag.NIIalphas['A']['NIIalpha'], diag.NIIalphas['A']['NIIalpha_uncert'], np.nan, np.nan,
                  'ratio, not a flux: [NII]6583/Halpha'))
rows_A.append(row('O3N2', diag.o3n2A, diag.o3n2A_err, np.nan, np.nan,
                  'ratio, not a flux: ([OIII]5007/Hbeta) / ([NII]6583/Halpha)'))
rows_A.append(row('[OIII]/Hbeta', diag.OIIIbetas['A']['OIIIbeta'], diag.OIIIbetas['A']['OIIIbeta_uncert'], np.nan, np.nan,
                  'ratio, not a flux: [OIII]5007/Hbeta'))

# ==================
# source B
# ==================
rows_B = []

# [OII]3726_B — directly fit (index 1 in OII model)
amp  = m_oii.amplitude_1.value;  std  = m_oii.stddev_1.value
aunc = get_std(m_oii, 'amplitude_1');  sunc = get_std(m_oii, 'stddev_1')
flux_3726B, flux_3726B_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('[OII]3726', flux_3726B, flux_3726B_unc, std, sunc, 'directly fit'))
rows_B.append(row('[OII]3729', flux_3726B * 1.5, flux_3726B_unc * 1.5,
                  np.nan, np.nan, 'derived from [OII]3726B * 1.5 (low-density ratio)'))

# Hgamma — balmer fit, index 5 source B
amp  = m_balmer.amplitude_5.value;  std  = m_balmer.stddev_5.value
aunc = get_std(m_balmer, 'amplitude_5');  sunc = get_std(m_balmer, 'stddev_5')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('Hgamma', flux, flux_unc, std, sunc))

# Hbeta — OIII fit, index 3 source B (reference)
amp  = m_oiii.amplitude_3.value;  std  = m_oiii.stddev_3.value
aunc = get_std(m_oiii, 'amplitude_3');  sunc = get_std(m_oiii, 'stddev_3')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('Hbeta', flux, flux_unc, std, sunc))

# [OIII]4959 — OIII fit, index 4 source B
amp  = m_oiii.amplitude_4.value;  std  = m_oiii.stddev_4.value
aunc = get_std(m_oiii, 'amplitude_4');  sunc = get_std(m_oiii, 'stddev_4')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('[OIII]4959', flux, flux_unc, std, sunc))

# [OIII]5007 — OIII fit, index 5 source B
amp  = m_oiii.amplitude_5.value;  std  = m_oiii.stddev_5.value
aunc = get_std(m_oiii, 'amplitude_5');  sunc = get_std(m_oiii, 'stddev_5')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('[OIII]5007', flux, flux_unc, std, sunc))

# [NII]6548 — NII fit, index 5 source B
amp  = m_nii.amplitude_5.value;  std  = m_nii.stddev_5.value
aunc = get_std(m_nii, 'amplitude_5');  sunc = get_std(m_nii, 'stddev_5')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('[NII]6548', flux, flux_unc, std, sunc))

# Halpha — NII fit, index 3 source B (reference)
amp  = m_nii.amplitude_3.value;  std  = m_nii.stddev_3.value
aunc = get_std(m_nii, 'amplitude_3');  sunc = get_std(m_nii, 'stddev_3')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('Halpha', flux, flux_unc, std, sunc))

# [NII]6583 — NII fit, index 4 source B
amp  = m_nii.amplitude_4.value;  std  = m_nii.stddev_4.value
aunc = get_std(m_nii, 'amplitude_4');  sunc = get_std(m_nii, 'stddev_4')
flux, flux_unc = integrate(amp, std, aunc, sunc)
rows_B.append(row('[NII]6583', flux, flux_unc, std, sunc))

# diagnostic ratios (from diagnostics.py)
rows_B.append(row('R23', diag.R23_B, diag.R23_B_err, np.nan, np.nan,
                  'ratio, not a flux: ([OII]3726+[OII]3729+[OIII]4959+[OIII]5007)/Hbeta'))
rows_B.append(row('R23_KK04', diag.kR23_B, diag.kR23_B_err, np.nan, np.nan,
                  'ratio, not a flux: ([OII]3726+[OIII]4959+[OIII]5007)/Hbeta (Kewley & Dopita 2002)'))
rows_B.append(row('N2', diag.NIIalphas['B']['NIIalpha'], diag.NIIalphas['B']['NIIalpha_uncert'], np.nan, np.nan,
                  'ratio, not a flux: [NII]6583/Halpha'))
rows_B.append(row('O3N2', diag.o3n2B, diag.o3n2B_err, np.nan, np.nan,
                  'ratio, not a flux: ([OIII]5007/Hbeta) / ([NII]6583/Halpha)'))
rows_B.append(row('[OIII]/Hbeta', diag.OIIIbetas['B']['OIIIbeta'], diag.OIIIbetas['B']['OIIIbeta_uncert'], np.nan, np.nan,
                  'ratio, not a flux: [OIII]5007/Hbeta'))

# ==================
# save
# ==================
df_A = pd.DataFrame(rows_A).round(3)
df_B = pd.DataFrame(rows_B).round(3)
df_A.to_csv('./output/emission_lines_A.csv', index=False)
df_B.to_csv('./output/emission_lines_B.csv', index=False)

print("Source A:")
print(df_A.to_string(index=False))
print()
print("Source B:")
print(df_B.to_string(index=False))
