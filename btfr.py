import os

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

c =  299792 #km/s
z_A = 1.679
z_B = 1.677

# ====================
# Shared physical_values_A.csv / physical_values_B.csv are written to by
# several independent scripts (this one, metallicity.py,
# diagnostics_dust_corrected.py), each owning a different subset of
# quantities -- upsert-by-quantity-name so re-running any one script updates
# only its own rows and leaves the others' rows (written by other scripts)
# intact, regardless of run order.
# ====================
def write_physical_values(source, rows):
    path = f'./output/physical_values_{source}.csv'
    new_df = pd.DataFrame(rows)
    try:
        # skipinitialspace + strip: some editor/linter re-aligns this file
        # with padding spaces around each field (incl. before quoted fields,
        # which otherwise breaks the default C parser's quote detection)
        existing = pd.read_csv(path, skipinitialspace=True)
        existing.columns = existing.columns.str.strip()
        existing['quantity'] = existing['quantity'].astype(str).str.strip()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing = pd.DataFrame(columns=['quantity', 'value', 'uncertainty', 'notes'])
    existing = existing[~existing['quantity'].isin(new_df['quantity'])]
    pd.concat([existing, new_df], ignore_index=True).to_csv(path, index=False)

# ====================
# Load in Gauss parameters for dust-corrected central components of A and B
# ====================
# mean/stddev are kinematic fit outputs (dust correction only rescales flux,
# see dust_extinction.py), so both sources' central components come straight
# from the joint fit that feeds the whole dust-corrected pipeline (see
# multiple_gaussian_integration.py's LINES['Halpha']['pkl']) -- index 0 is
# Halpha_A_central, index 3 is Halpha_B_central (jointfit_all.py's index
# layout comment). Uncertainties are TRFLSQFitter's per-parameter stds.
with open('./output/joint_fit/jointfit_all/Halpha_joint_tied_fit.pkl', 'rb') as f:
    halpha_model = dill.load(f)['model']

mean_A, mean_A_unc = halpha_model.mean_0.value, halpha_model.stds['mean_0']
stddev_A, stddev_A_unc = halpha_model.stddev_0.value, halpha_model.stds['stddev_0']
mean_B, mean_B_unc = halpha_model.mean_3.value, halpha_model.stds['mean_3']
stddev_B, stddev_B_unc = halpha_model.stddev_3.value, halpha_model.stds['stddev_3']


# ====================
# Get V_c = W_20
# ====================
# get rid of instrumental broadening?
def unbroaden(stddev, stddev_unc, z):
    R = 5600
    delta_lam = (6562.819 * (1+z)) / R
    siginst = delta_lam / 2.355
    stddev_gal = np.sqrt(stddev**2 - siginst**2)
    # siginst treated as exactly known (R is a fixed instrument spec) --
    # only stddev's own uncertainty propagates through the sqrt-of-difference
    stddev_gal_unc = stddev * stddev_unc / stddev_gal
    return stddev_gal, stddev_gal_unc

# get W_20 analytically
def w20(stddev_gal, stddev_gal_unc):
    factor = 2*np.sqrt(-2*np.log(0.2))
    w20 = factor*stddev_gal
    w20_unc = factor*stddev_gal_unc
    return w20, w20_unc

# get W_20 in km/s
def w20_kms(w20, w20_unc, mean, mean_unc): # do this in observed
    w20_kms = c * w20 / mean
    w20_kms_unc = w20_kms * np.sqrt((w20_unc/w20)**2 + (mean_unc/mean)**2)
    return w20_kms, w20_kms_unc

# divide by 2 and deproject for V_c
def V_c(w20_kms, w20_kms_unc, i, i_unc):
    v_c = w20_kms / (2*np.sin(i))
    v_c_unc = np.sqrt((w20_kms_unc/(2*np.sin(i)))**2 +
                      (w20_kms*np.cos(i)/(2*np.sin(i)**2) * i_unc)**2)
    return v_c, v_c_unc

# ====================
# btfrs 
# ====================

# log_M = a*log10(v_c/v_ref) + b -- v_c_unc propagates through the log term,
# combined in quadrature with the relation's own zero-point uncertainty
# (b_unc) and intrinsic scatter (both are properties of the published
# relation itself, not of this measurement, so they add independently)
def _btfr_log_M_unc(v_c, v_c_unc, a, b_unc, scatter):
    return np.sqrt((a * v_c_unc / (v_c * np.log(10)))**2 + b_unc**2 + scatter**2)


# use Ubler 2017
def btfr_6_26(v_c, v_c_unc):
    a = 3.75
    b = 10.75
    b_unc = 0.03
    scatter = 0.23
    v_ref = 242 #km/s
    log_M = a * np.log10(v_c / v_ref) + b
    log_M_unc = _btfr_log_M_unc(v_c, v_c_unc, a, b_unc, scatter)
    return log_M, log_M_unc

def btfr_9(v_c, v_c_unc):
    a = 3.75
    b = 10.68
    b_unc = 0.04
    scatter = 0.22
    v_ref = 242 #km/s
    log_M = a * np.log10(v_c / v_ref) + b
    log_M_unc = _btfr_log_M_unc(v_c, v_c_unc, a, b_unc, scatter)
    return log_M, log_M_unc

def btfr_23(v_c, v_c_unc):
    a = 3.75
    b = 10.85
    b_unc = 0.05
    scatter = 0.26
    v_ref = 242 #km/s
    log_M = a * np.log10(v_c / v_ref) + b
    log_M_unc = _btfr_log_M_unc(v_c, v_c_unc, a, b_unc, scatter)
    return log_M, log_M_unc


# APPLY FOR EDGE ON 
# ====================
# Apply -- Source A, edge-on (i = pi/2, i_unc = 0 -- d(V_c)/di ~ cos(i) = 0
# here anyway, so this is exact, not an approximation), btfr_6_26
# ====================
i, i_unc = np.pi/2, 0.0

# with instrumental broadening correction
stddev_A_gal, stddev_A_gal_unc = unbroaden(stddev_A, stddev_A_unc, z=z_A)
w20_A_corr, w20_A_corr_unc = w20(stddev_A_gal, stddev_A_gal_unc)
w20_kms_A_corr, w20_kms_A_corr_unc = w20_kms(w20_A_corr, w20_A_corr_unc, mean_A, mean_A_unc)
v_c_A_corr, v_c_A_corr_unc = V_c(w20_kms_A_corr, w20_kms_A_corr_unc, i, i_unc)
log_M_A_corr, log_M_A_corr_unc = btfr_6_26(v_c_A_corr, v_c_A_corr_unc)

print("Source A -- BTFR (Ubler 2017 eq. 6.26) lower limit mass estimate:")
print(f"  with instrumental broadening correction:    "
      f"log(M) = {log_M_A_corr:.3f} +/- {log_M_A_corr_unc:.3f}  "
      f"(V_c = {v_c_A_corr:.2f} +/- {v_c_A_corr_unc:.2f} km/s)")

# ====================
# Apply -- Source B, edge-on, btfr_6_26
# ====================
# with instrumental broadening correction
stddev_B_gal, stddev_B_gal_unc = unbroaden(stddev_B, stddev_B_unc, z=z_B)
w20_B_corr, w20_B_corr_unc = w20(stddev_B_gal, stddev_B_gal_unc)
w20_kms_B_corr, w20_kms_B_corr_unc = w20_kms(w20_B_corr, w20_B_corr_unc, mean_B, mean_B_unc)
v_c_B_corr, v_c_B_corr_unc = V_c(w20_kms_B_corr, w20_kms_B_corr_unc, i, i_unc)
log_M_B_corr, log_M_B_corr_unc = btfr_6_26(v_c_B_corr, v_c_B_corr_unc)


print("Source B -- BTFR (Ubler 2017 eq. 6.26) lower limit mass estimate:")
print(f"  with instrumental broadening correction:    "
      f"log(M) = {log_M_B_corr:.3f} +/- {log_M_B_corr_unc:.3f}  "
      f"(V_c = {v_c_B_corr:.2f} +/- {v_c_B_corr_unc:.2f} km/s)")



# APPLY FOR 45 DEG
# with instrumental broadening correction
i_45 = np.pi/4
v_c_A_corr_45deg, v_c_A_corr_45deg_unc = V_c(w20_kms_A_corr, w20_kms_A_corr_unc, i_45, i_unc)
log_M_A_corr_45deg, log_M_A_corr_unc_45deg = btfr_6_26(v_c_A_corr_45deg, v_c_A_corr_45deg_unc)

v_c_B_corr_45deg, v_c_B_corr_45deg_unc = V_c(w20_kms_B_corr, w20_kms_B_corr_unc, i_45, i_unc)
log_M_B_corr_45deg, log_M_B_corr_unc_45deg = btfr_6_26(v_c_B_corr_45deg, v_c_B_corr_45deg_unc)

print("Source A -- BTFR (Ubler 2017 eq. 6.26) lower limit mass estimate, i=45deg:")
print(f"  with instrumental broadening correction:    "
      f"log(M) = {log_M_A_corr_45deg:.3f} +/- {log_M_A_corr_unc_45deg:.3f}  "
      f"(V_c = {v_c_A_corr_45deg:.2f} +/- {v_c_A_corr_45deg_unc:.2f} km/s)")

print("Source B -- BTFR (Ubler 2017 eq. 6.26) lower limit mass estimate, i=45deg:")
print(f"  with instrumental broadening correction:    "
      f"log(M) = {log_M_B_corr_45deg:.3f} +/- {log_M_B_corr_unc_45deg:.3f}  "
      f"(V_c = {v_c_B_corr_45deg:.2f} +/- {v_c_B_corr_45deg_unc:.2f} km/s)")

write_physical_values('A', [
    dict(quantity='log_M_btfr6_26_corrected_edgeon', value=log_M_A_corr, uncertainty=log_M_A_corr_unc,
         notes='Ubler 2017 eq. 6.26, instrumental-broadening corrected, edge-on (i=90deg)'),
    dict(quantity='log_M_btfr6_26_corrected_45deg', value=log_M_A_corr_45deg, uncertainty=log_M_A_corr_unc_45deg,
         notes='Ubler 2017 eq. 6.26, instrumental-broadening corrected, i=45deg'),
])
write_physical_values('B', [
    dict(quantity='log_M_btfr6_26_corrected_edgeon', value=log_M_B_corr, uncertainty=log_M_B_corr_unc,
         notes='Ubler 2017 eq. 6.26, instrumental-broadening corrected, edge-on (i=90deg)'),
    dict(quantity='log_M_btfr6_26_corrected_45deg', value=log_M_B_corr_45deg, uncertainty=log_M_B_corr_unc_45deg,
         notes='Ubler 2017 eq. 6.26, instrumental-broadening corrected, i=45deg'),
])

# ====================
# Mass-Metallicity Relation
# ====================
def k18_MZ(mass, mass_unc): # uses calibration from KD02
    a=28.0974
    b=-7.23631
    c=0.850344
    d=-0.0318315
    scatter = 0.1
    z = a + b*mass + c*mass**2 + d*mass**3
    # a/b/c/d are a fixed calibration (no quoted uncertainty) -- only mass
    # propagates, via the cubic's derivative, combined in quadrature with
    # the relation's own intrinsic scatter
    dz_dmass = b + 2*c*mass + 3*d*mass**2
    z_unc = np.sqrt((dz_dmass*mass_unc)**2 + scatter**2)
    return z, z_unc

# Zahid et al. 2014 eq. 6, COSMOS z~1.55 fit (Table 2, row 2):
# 12+log(O/H) = Zo - log10[1 + (M*/Mo)^-gamma]. log_mass is log10(M*/Msun);
# Mo is a characteristic turnover mass in the SAME (linear solar-mass) units
# as M*, so the ratio M*/Mo = 10**(log_mass - log_Mo) -- feeding log_mass
# straight in as if it were the ratio itself (the original bug) skips both
# the exponentiation and the Mo normalization entirely.
def cosmos_MZ(log_mass, log_mass_unc):
    Zo = 8.740
    Zo_unc = 0.042
    log_Mo = 9.93
    log_Mo_unc = 0.09
    gamma = 0.88
    gamma_unc = 0.18

    D = log_mass - log_Mo
    x = 10**(-gamma * D)  # (M*/Mo)^-gamma
    z = Zo - np.log10(1 + x)

    # d(z)/d(log_mass) = -d(z)/d(log_Mo) = gamma*x/(1+x) (the ln10 factors
    # from d(x)/dD and d(log10)/dx cancel exactly); d(z)/d(gamma) = D*x/(1+x).
    # log_mass_unc and log_Mo_unc are independent uncertainties, so even
    # though their derivatives have equal magnitude they add in quadrature
    # separately, not as a difference.
    common = gamma * x / (1 + x)
    dz_dgamma = D * x / (1 + x)
    z_unc = np.sqrt(Zo_unc**2 + (common*log_mass_unc)**2 + (common*log_Mo_unc)**2 +
                    (dz_dgamma*gamma_unc)**2)
    return z, z_unc


z_A_k18_edgeon, z_A_k18_edgeon_unc = k18_MZ(log_M_A_corr, log_M_A_corr_unc)
z_A_cosmos_edgeon, z_A_cosmos_edgeon_unc = cosmos_MZ(log_M_A_corr, log_M_A_corr_unc)

z_A_k18_45deg, z_A_k18_45deg_unc = k18_MZ(log_M_A_corr_45deg, log_M_A_corr_unc_45deg)
z_A_cosmos_45deg, z_A_cosmos_45deg_unc = cosmos_MZ(log_M_A_corr_45deg, log_M_A_corr_unc_45deg)

z_B_k18_edgeon, z_B_k18_edgeon_unc = k18_MZ(log_M_B_corr, log_M_B_corr_unc)
z_B_cosmos_edgeon, z_B_cosmos_edgeon_unc = cosmos_MZ(log_M_B_corr, log_M_B_corr_unc)

z_B_k18_45deg, z_B_k18_45deg_unc = k18_MZ(log_M_B_corr_45deg, log_M_B_corr_unc_45deg)
z_B_cosmos_45deg, z_B_cosmos_45deg_unc = cosmos_MZ(log_M_B_corr_45deg, log_M_B_corr_unc_45deg)

print()
print("Mass-metallicity relation (12+log(O/H)), from instrumental-broadening-corrected BTFR mass:")
print(f"Source A, edge-on:  k18 = {z_A_k18_edgeon:.3f} +/- {z_A_k18_edgeon_unc:.3f}   "
      f"cosmos = {z_A_cosmos_edgeon:.3f} +/- {z_A_cosmos_edgeon_unc:.3f}")
print(f"Source A, 45 deg:   k18 = {z_A_k18_45deg:.3f} +/- {z_A_k18_45deg_unc:.3f}   "
      f"cosmos = {z_A_cosmos_45deg:.3f} +/- {z_A_cosmos_45deg_unc:.3f}")
print(f"Source B, edge-on:  k18 = {z_B_k18_edgeon:.3f} +/- {z_B_k18_edgeon_unc:.3f}   "
      f"cosmos = {z_B_cosmos_edgeon:.3f} +/- {z_B_cosmos_edgeon_unc:.3f}")
print(f"Source B, 45 deg:   k18 = {z_B_k18_45deg:.3f} +/- {z_B_k18_45deg_unc:.3f}   "
      f"cosmos = {z_B_cosmos_45deg:.3f} +/- {z_B_cosmos_45deg_unc:.3f}")