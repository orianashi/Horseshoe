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
def unbroaden(stddev, stddev_unc, R, z):
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
stddev_A_gal, stddev_A_gal_unc = unbroaden(stddev_A, stddev_A_unc, R=5600, z=z_A)
w20_A_corr, w20_A_corr_unc = w20(stddev_A_gal, stddev_A_gal_unc)
w20_kms_A_corr, w20_kms_A_corr_unc = w20_kms(w20_A_corr, w20_A_corr_unc, mean_A, mean_A_unc)
v_c_A_corr, v_c_A_corr_unc = V_c(w20_kms_A_corr, w20_kms_A_corr_unc, i, i_unc)
log_M_A_corr, log_M_A_corr_unc = btfr_6_26(v_c_A_corr, v_c_A_corr_unc)

# without instrumental broadening correction (raw fitted stddev)
w20_A_raw, w20_A_raw_unc = w20(stddev_A, stddev_A_unc)
w20_kms_A_raw, w20_kms_A_raw_unc = w20_kms(w20_A_raw, w20_A_raw_unc, mean_A, mean_A_unc)
v_c_A_raw, v_c_A_raw_unc = V_c(w20_kms_A_raw, w20_kms_A_raw_unc, i, i_unc)
log_M_A_raw, log_M_A_raw_unc = btfr_6_26(v_c_A_raw, v_c_A_raw_unc)

print("Source A -- BTFR (Ubler 2017 eq. 6.26) mass estimate:")
print(f"  with instrumental broadening correction:    "
      f"log(M) = {log_M_A_corr:.3f} +/- {log_M_A_corr_unc:.3f}  "
      f"(V_c = {v_c_A_corr:.2f} +/- {v_c_A_corr_unc:.2f} km/s)")
print(f"  without instrumental broadening correction: "
      f"log(M) = {log_M_A_raw:.3f} +/- {log_M_A_raw_unc:.3f}  "
      f"(V_c = {v_c_A_raw:.2f} +/- {v_c_A_raw_unc:.2f} km/s)")

# ====================
# Apply -- Source B, edge-on, btfr_6_26
# ====================
# with instrumental broadening correction
stddev_B_gal, stddev_B_gal_unc = unbroaden(stddev_B, stddev_B_unc, R=5600, z=z_B)
w20_B_corr, w20_B_corr_unc = w20(stddev_B_gal, stddev_B_gal_unc)
w20_kms_B_corr, w20_kms_B_corr_unc = w20_kms(w20_B_corr, w20_B_corr_unc, mean_B, mean_B_unc)
v_c_B_corr, v_c_B_corr_unc = V_c(w20_kms_B_corr, w20_kms_B_corr_unc, i, i_unc)
log_M_B_corr, log_M_B_corr_unc = btfr_6_26(v_c_B_corr, v_c_B_corr_unc)

# without instrumental broadening correction (raw fitted stddev)
w20_B_raw, w20_B_raw_unc = w20(stddev_B, stddev_B_unc)
w20_kms_B_raw, w20_kms_B_raw_unc = w20_kms(w20_B_raw, w20_B_raw_unc, mean_B, mean_B_unc)
v_c_B_raw, v_c_B_raw_unc = V_c(w20_kms_B_raw, w20_kms_B_raw_unc, i, i_unc)
log_M_B_raw, log_M_B_raw_unc = btfr_6_26(v_c_B_raw, v_c_B_raw_unc)

print("Source B -- BTFR (Ubler 2017 eq. 6.26) mass estimate:")
print(f"  with instrumental broadening correction:    "
      f"log(M) = {log_M_B_corr:.3f} +/- {log_M_B_corr_unc:.3f}  "
      f"(V_c = {v_c_B_corr:.2f} +/- {v_c_B_corr_unc:.2f} km/s)")
print(f"  without instrumental broadening correction: "
      f"log(M) = {log_M_B_raw:.3f} +/- {log_M_B_raw_unc:.3f}  "
      f"(V_c = {v_c_B_raw:.2f} +/- {v_c_B_raw_unc:.2f} km/s)")

btfr_9(v_c_A_corr, v_c_A_corr_unc)
btfr_23(v_c_A_corr, v_c_A_corr_unc)

btfr_9(v_c_B_corr, v_c_B_corr_unc)
btfr_23(v_c_B_corr, v_c_B_corr_unc)


# APPLY FOR 45 DEG
# without instrumental broadening correction (raw fitted stddev)
i_45 = np.pi/4
v_c_A_raw_45deg, v_c_A_raw_45deg_unc = V_c(w20_kms_A_raw, w20_kms_A_raw_unc, i_45, i_unc)
log_M_A_raw_45deg, log_M_A_raw_unc_45deg = btfr_6_26(v_c_A_raw_45deg, v_c_A_raw_45deg_unc)

v_c_B_raw_45deg, v_c_B_raw_45deg_unc = V_c(w20_kms_B_raw, w20_kms_B_raw_unc, i_45, i_unc)
log_M_B_raw_45deg, log_M_B_raw_unc_45deg = btfr_6_26(v_c_B_raw_45deg, v_c_B_raw_45deg_unc)

