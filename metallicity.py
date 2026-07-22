import os

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# solve k_n*x^n + ... + k1*x + (k0 - target) = 0 and return the real roots,
# sorted ascending, as a plain array (drops any complex roots)
def _real_roots(coeffs_high_to_low, k0, target):
    poly = list(coeffs_high_to_low)
    poly[-1] = poly[-1] - target
    roots = np.roots(poly)
    real = np.sort(roots[np.abs(roots.imag) < 1e-6].real)
    return real

# physical_values_A.csv / physical_values_B.csv are shared with btfr.py and
# diagnostics_dust_corrected.py, each owning a different subset of
# quantities -- upsert-by-quantity-name so re-running any one script updates
# only its own rows and leaves the others' rows intact, regardless of order.
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

# load in everything
DIAGDIR = 'output/dust_corrected/dust_corrected_diagnostics'

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)

# 'component' is keyed by wing name (see diagnostics_dust_corrected.py's
# WING_NAMES: A has ['central', 'red_wing'], B has ['red_wing', 'central',
# 'blue_wing']) -- 'central' exists for both, giving the same ratio fields
# as the cumulative dict (R23, KD02_O32, log_N2, etc.) but for just the
# central velocity component instead of the source's full (all-components)
# flux.
A_pkl = load_pkl(f'{DIAGDIR}/A_ratios_dust_corrected_cumulativeEBV.pkl')
B_pkl = load_pkl(f'{DIAGDIR}/B_ratios_dust_corrected_cumulativeEBV.pkl')
A = A_pkl['cumulative']
B = B_pkl['cumulative']
A_central = A_pkl['component']['central']
B_central = B_pkl['component']['central']

# ==================
# KK04 IONIZATION PARAMETER FOR SOURCE A  
# ==================
def kk04_q(z, z_err, kk04_o32, kk04_o32_err):
    y = kk04_o32
    p1 = 32.81 - 1.153*y**2 + z*(-3.396 - 0.025*y + 0.1444*y**2)
    p2 = 4.603 - 0.3119*y - 0.163*y**2 + z*(-0.48 + 0.0271*y + 0.02037*y**2)
    logq = p1 * (p2**(-1))

    # logq = p1(y,z)/p2(y,z) -- quotient rule for each partial, then
    # combine in quadrature (y and z treated as independent, same
    # assumption as z94_m91's two-variable propagation above)
    dp1_dy = -2*1.153*y + z*(-0.025 + 2*0.1444*y)
    dp1_dz = -3.396 - 0.025*y + 0.1444*y**2
    dp2_dy = -0.3119 - 2*0.163*y + z*(0.0271 + 2*0.02037*y)
    dp2_dz = -0.48 + 0.0271*y + 0.02037*y**2

    dlogq_dy = (dp1_dy*p2 - p1*dp2_dy) / p2**2
    dlogq_dz = (dp1_dz*p2 - p1*dp2_dz) / p2**2

    logq_err = np.sqrt((dlogq_dy * kk04_o32_err)**2 + (dlogq_dz * z_err)**2)
    return logq, logq_err

# ==================
# KD02 METALLICITY NONCOMBINED  
# ==================
# KD02 eq. 4 / Table 3, log(R23) diagnostic at q=3.0e8 cm/s:
# log(R23) = k0 + k1*x + k2*x^2 + k3*x^3 + k4*x^4, x = log(O/H)+12
def z_r23_q3e8(kk04_r23, kk04_r23_unc):
    k0 = -1550.53
    k1 = 784.26
    k2 = -149.245
    k3 = 12.6618
    k4 = -0.403774
    roots = _real_roots([k4, k3, k2, k1, k0], k0, kk04_r23)
    if len(roots) == 0:
        # no real root -- this R23 value exceeds what this q track can
        # produce at any metallicity (same "no valid solution" case as
        # r23_q3_e8_combined's negative discriminant above), so return NaN
        # rather than crash
        return np.nan, np.nan
    z = roots[0]  # lower (low-metallicity) branch of the double-valued R23 track
    # z is a root of k4*x^4+k3*x^3+k2*x^2+k1*x+k0 = kk04_r23, not an explicit
    # function of kk04_r23 -- propagate via implicit differentiation (same
    # delta-method trick as high_metallicity_q_halfsolar/_solar above):
    # d(kk04_r23)/dz = 4*k4*z^3 + 3*k3*z^2 + 2*k2*z + k1, so dz/d(kk04_r23) =
    # 1/that, evaluated at the root itself.
    dFdz = 4*k4*z**3 + 3*k3*z**2 + 2*k2*z + k1
    z_unc = np.abs(kk04_r23_unc / dFdz)
    return z, z_unc 

# use [NII]/Halpha to identify low vs high metallicity regime
def n2ha_q1_e8(n2ha, n2ha_err):
    k0 = -2983.69
    k1 = 1454.45
    k2 = -266.015
    k3 = 21.6024
    k4 = -0.6566
    roots = _real_roots([k4, k3, k2, k1, k0], k0, n2ha)
    if len(roots) == 0:
        return np.nan, np.nan
    abundance = roots[0]  # lower (low-metallicity) branch, same convention as z_r23_q3e8
    # abundance is a root of k4*x^4+k3*x^3+k2*x^2+k1*x+k0 = n2ha, not an
    # explicit function of n2ha -- propagate via implicit differentiation
    # (same delta-method trick as z_r23_q3e8/high_metallicity_q_* above):
    # d(n2ha)/d(abundance) = 4*k4*x^3 + 3*k3*x^2 + 2*k2*x + k1, so
    # d(abundance)/d(n2ha) = 1/that, evaluated at the root itself.
    dFdx = 4*k4*abundance**3 + 3*k3*abundance**2 + 2*k2*abundance + k1
    abundance_err = np.abs(n2ha_err / dFdx)
    return abundance, abundance_err

def n2ha_q3_e8(n2ha, n2ha_err):
    k0 = -3100.57
    k1 = 1501.77
    k2 = -272.883
    k3 = 22.0132
    k4 = -0.6646
    roots = _real_roots([k4, k3, k2, k1, k0], k0, n2ha)
    if len(roots) == 0:
        return np.nan, np.nan
    abundance = roots[0]  # lower (low-metallicity) branch, same convention as z_r23_q3e8
    # same implicit-differentiation delta method as n2ha_q1_e8 above
    dFdx = 4*k4*abundance**3 + 3*k3*abundance**2 + 2*k2*abundance + k1
    abundance_err = np.abs(n2ha_err / dFdx)
    return abundance, abundance_err

# ==================
# Kewley et al. 2019 (K19) Table 2 bicubic surface fit: [NII]/Halpha
# ==================
# z = A + Bx + Cy + Dxy + Ex^2 + Fy^2 + Gxy^2 + Hyx^2 + Ix^3 + Jy^3, where
# x = log([NII]6584/Halpha), y = log(U) = log(q) - log(c), z = log(O/H)+12.
# Valid over 7.63 <= z <= 8.53 and -3.98 <= log(U) <= -1.98 (K19 Table 2
# footnote). Unlike KD02's n2ha_q1_e8/n2ha_q3_e8 (a discrete q-track,
# requiring root-finding), this is a direct, continuous surface in both the
# line ratio AND the ionization parameter -- no branch selection needed.
# K19 quotes a 0.67% RMS scatter for this diagnostic; added in quadrature to
# the propagated measurement uncertainty (as a fraction of |z|), same
# convention as KD02's own quadrature-added scatter terms (0.04/0.07)
# elsewhere in this file, just converted from a percentage to a dex value.
def k19_n2ha_q(n2ha, n2ha_err, log_q, log_q_err):
    Ac, Bc, Cc, Dc, Ec, Fc, Gc, Hc, Ic, Jc = (
        10.526, 1.9958, -0.6741, 0.2892, 0.5712, -0.6597, 0.0101, 0.0800, 0.0782, -0.0982)
    RMS_PCT = 0.67
    C_LIGHT_CGS = 2.99792458e10  # cm/s

    x = n2ha
    y = log_q - np.log10(C_LIGHT_CGS)  # log(U) = log(q) - log(c)

    z = (Ac + Bc*x + Cc*y + Dc*x*y + Ec*x**2 + Fc*y**2
         + Gc*x*y**2 + Hc*y*x**2 + Ic*x**3 + Jc*y**3)

    # partial derivatives of the bicubic surface, for error propagation
    dzdx = Bc + Dc*y + 2*Ec*x + Gc*y**2 + 2*Hc*y*x + 3*Ic*x**2
    dzdy = Cc + Dc*x + 2*Fc*y + 2*Gc*x*y + Hc*x**2 + 3*Jc*y**2
    z_err_prop = np.sqrt((dzdx*n2ha_err)**2 + (dzdy*log_q_err)**2)
    z_scatter = np.abs(z) * RMS_PCT / 100
    z_err = np.sqrt(z_err_prop**2 + z_scatter**2)
    return z, z_err

# ==================
# KD02 PROCESS FOR METALLICITIES below 8.5
# ==================


# use the metallicity + O32 to constrain the ionization parameter using KD02's "improved version"
# 0.1*Z_solar <=> 12 + log(O/H) =~ 7.9
def z_tenthsolar_combined(KD02_o32, KD02_o32_err):
    k0 = 7.46218
    k1 = 0.685835
    k2 = 0.0866086
    q = 10**(k0+k1*KD02_o32+k2*KD02_o32**2)
    # dq/d(o32) = q * ln(10) * (k1 + 2*k2*o32)
    qerr = q * np.log(10) * np.abs(k1 + 2*k2*KD02_o32) * KD02_o32_err
    return q, qerr

# 0.2*Z_solar <=> 12 + log(O/H) =~ 8.2
def z_fifthsolar_combined(KD02_o32, KD02_o32_err):
    k0 = 7.57817
    k1 = 0.739315
    k2 = 0.0843640
    q = 10**(k0+k1*KD02_o32+k2*KD02_o32**2)
    qerr = q * np.log(10) * np.abs(k1 + 2*k2*KD02_o32) * KD02_o32_err
    return q, qerr

# 0.5*Z_solar <=> 12 + log(O/H) =~ 8.6
def z_halfsolar_combined(KD02_o32, KD02_o32_err):
    k0 = 7.73013
    k1 = 0.843125
    k2 = 0.118166
    q = 10**(k0+k1*KD02_o32+k2*KD02_o32**2)
    # dq/d(o32) = q * ln(10) * (k1 + 2*k2*o32)
    qerr = q * np.log(10) * np.abs(k1 + 2*k2*KD02_o32) * KD02_o32_err
    return q, qerr

# use ionization parameter and correct R23 track to get the metallicity again
# here we are using the low-metallicity track
# q = 8e7 cm/s
def r23_q8_e7_combined(KK04_r23, KK04_r23_err):
    k0 = -44.7026
    k1 = 10.8052
    k2 = -0.640113
    disc = k1**2 - 4*k2*(k0-KK04_r23)
    abundance = (-k1 + np.sqrt(disc)) / (2*k2)
    abundance_err = np.abs(KK04_r23_err / np.sqrt(disc))
    return abundance, abundance_err

def r23_q1_e8_combined(KK04_r23, KK04_r23_err):
    k0 = -46.1589
    k1 = 11.2557
    k2 = -0.672731
    disc = k1**2 - 4*k2*(k0-KK04_r23)
    abundance = (-k1 + np.sqrt(disc)) / (2*k2)
    # d(abundance)/d(r23) = 1/sqrt(disc), from d(disc)/d(r23) = 4*k2
    abundance_err = np.abs(KK04_r23_err / np.sqrt(disc))
    return abundance, abundance_err

def r23_q3_e8_combined(KK04_r23, KK04_r23_err):
    k0 = -45.6075
    k1 = 11.2074
    k2 = -0.674460
    disc = k1**2 - 4*k2*(k0-KK04_r23)
    abundance = (-k1 + np.sqrt(disc)) / (2*k2)
    abundance_err = np.abs(KK04_r23_err / np.sqrt(disc))
    return abundance, abundance_err

# if the metallicity has changed significantly, go to metallicity + O32 to get the ionization parameter again 
# ideally we would be averaging the R23 diagnostic and the C01 diagnostic, but that uses [SII]

# ==================
# KD02 PROCESS FOR METALLICITIES ABOVE 8.6 
# ==================
# step 1 of KD02 optimized abundance determination
def n2o2_combined(n2o2,n2o2_err): #this only sensitive for above >8.6 
    u = 1.54020+1.26602*n2o2+0.167977*n2o2**2
    abundance = np.log10(u) + 8.93
    # d(abundance)/d(n2o2) = 0.434/u * du/d(n2o2)
    abundance_err = 0.434/u * np.abs(1.26602 + 2*0.167977*n2o2) * n2o2_err
    return abundance, abundance_err

# step 2a - if step 1 has abundance >8.6, use kd02_O32 ionization diagnostic
def high_metallicity_q_halfsolar(KD02_o32, KD02_o32_err):
    k0 = -81.1880
    k1=27.5082
    k2 = -3.19126
    k3 = 0.128252
    q = _real_roots([k3, k2, k1, k0], k0, KD02_o32)
    # q is a root of k3*x^3+k2*x^2+k1*x+k0 = KD02_o32, not an explicit
    # function of KD02_o32 -- propagate via implicit differentiation
    # (same delta-method trick as ionization_parameter.py's bicubic_dzdx):
    # d(o32)/dx = 3*k3*x^2 + 2*k2*x + k1, so dx/d(o32) = 1/that, evaluated
    # at the root itself.
    dFdx = 3*k3*q**2 + 2*k2*q + k1
    q_err = np.abs(KD02_o32_err / dFdx)
    return q, q_err

def high_metallicity_q_solar(KD02_o32, KD02_o32_err):
    k0 = -52.6367
    k1 = 16.0880
    k2 = -1.67443
    k3 = 0.0608004
    q = _real_roots([k3, k2, k1, k0], k0, KD02_o32)
    # same implicit-differentiation delta method as high_metallicity_q_halfsolar:
    # d(o32)/dx = 3*k3*x^2 + 2*k2*x + k1, so dx/d(o32) = 1/that, evaluated
    # at the root itself.
    dFdx = 3*k3*q**2 + 2*k2*q + k1
    q_err = np.abs(KD02_o32_err / dFdx)
    return q, q_err

# step 2B - if step 1 has abundance <8.6, do the average of Z94 and M91(upper) using kk04_o32 
def z94_m91(KK04_r23, KK04_r23_err, kk04_o32, kk04_o32_err):
    abundance_z94 = 9.265 - 0.33*KK04_r23 - 0.202*KK04_r23**2 - 0.207*KK04_r23**3 - 0.333*KK04_r23**4
    abundance_m91u = 12 - 4.944 + 0.767*KK04_r23 + 0.602*KK04_r23**2 - kk04_o32 * (0.29 + 0.332*KK04_r23-0.331*KK04_r23**2)
    avg = (abundance_m91u + abundance_z94)/2

    d_z94_dr23 = -0.33 - 2*0.202*KK04_r23 - 3*0.207*KK04_r23**2 - 4*0.333*KK04_r23**3
    z94_err = np.abs(d_z94_dr23) * KK04_r23_err

    d_m91u_dr23 = 0.767 + 2*0.602*KK04_r23 - kk04_o32 * (0.332 - 2*0.331*KK04_r23)
    d_m91u_do32 = -(0.29 + 0.332*KK04_r23 - 0.331*KK04_r23**2)
    m91u_err = np.sqrt((d_m91u_dr23 * KK04_r23_err)**2 + (d_m91u_do32 * kk04_o32_err)**2)

    avg_err = np.sqrt(z94_err**2 + m91u_err**2) / 2
    return avg, avg_err

# step 3b - if step 2 produces estimate velow 8.5, use R23 method
# since the combined R23 method has a sqrt that produces NaN in our range, use the original

# ========================
# Calculate cumulative metallicity 
# ========================
# SOURCE B
# ITERATION 1
abundance_B_initialguess = n2ha_q1_e8(B['log_N2'], B['log_N2_err'])[0] #7.99
qb_tenthsolar, qb_tenthsolar_unc = z_tenthsolar_combined(B['log_KD02_O32'], B['log_KD02_O32_err']) #q=7.1e7
abundance_B_q8e7, abundance_B_q8e7_unc = r23_q8_e7_combined(B['log_kk04_R23'], B['log_kk04_R23_err']) #8.14+-0.52
#is now much closer to 0.2Z_solar (8.2) than 0.1Z_solar (7.9), therefore re-evaluate 

#ITERATION 2 
qb_fifthsolar, qb_fifthsolar_unc = z_fifthsolar_combined(B['log_KD02_O32'], B['log_KD02_O32_err']) #q=9.9e7 
abundance_B_qe8, abundance_B_qe8_unc = r23_q1_e8_combined(B['log_kk04_R23'], B['log_kk04_R23_err']) #8.011 +- 0.42
#8.01 is now closer to 0.1Z_solar again... so the real value of 12_log(O/H) is probably something in between 8.14 and 8.01 

#FINAL
#take the average of the 12+log(O/H) calculated from q=8e7 and q=1e8 and 0.2Z_solar from R23
abundance_B = (abundance_B_q8e7 + abundance_B_qe8)/2
abundance_B_unc_input = 0.5*np.sqrt(abundance_B_qe8_unc**2 + abundance_B_q8e7_unc**2)
abundance_B_unc = np.sqrt(0.07**2 + abundance_B_unc_input**2) # account for scatter on the method itself per KD02
#take the average of the log(q) calculated from 0.1Z_solar and 0.2Z_solar and O32
q_B = (qb_fifthsolar + qb_tenthsolar)/2
q_B_unc = 0.5*np.sqrt(qb_fifthsolar_unc**2 + qb_tenthsolar_unc**2)

print(f"Source B 12+log(O/H) = {abundance_B:.3f} +- {abundance_B_unc:.3f}")
# q_B/q_B_unc are linear (cm/s) -- np.log10(q_B_unc) is NOT valid error
# propagation into log-space (that's the log of the error bar itself, not
# the propagated log-space uncertainty). Correct delta method, same as
# log_uncert() elsewhere in this codebase: sigma_log(q) = sigma_q/(q*ln10).
log_q_B = np.log10(q_B)
log_q_B_unc = q_B_unc / (q_B * np.log(10))
print(f"Source B log(q) = {log_q_B:.3f} +- {log_q_B_unc:.3f}")

# SOURCE A

#using the high-metallicity process
#step 1: [NII]/[OII]
n2o2_combined(A['log_NII_OII'], A['log_NII_OII_err']) # 8.34
#step 2: since step 1 is below 8.6
z94_m91(A['log_kk04_R23'], A['log_kk04_R23_err'],  A['log_KK04_O32'], A['log_KK04_O32_err']) #8.14
#since step 2 is below 8.5, switch to the low-metallicity process

# using the low-metallicity process
n2ha_q3_e8(A['log_N2'], A['log_N2_err'])  #8.42
q_A, q_A_unc = z_halfsolar_combined(A['log_KD02_O32'], A['log_KD02_O32_err']) #2.4e8
r23_q3_e8_combined(A['log_kk04_R23'], A['log_kk04_R23_err'])[0] # HERE IS WHERE THERE IS A NAN, but the desmos graph look like around 8.3-8.6
# now use the normal R23 method
z_r23_q3e8(A['log_kk04_R23'], A['log_kk04_R23_err'])[0] # ALSO NaN -- no real root at this q, confirms R23 can't constrain this source
# abundance in 8.5 - 8.9 region can't be reliably determined with R23
#finally, default back to the N2 method :(
# NOTE: KD02 explicitly excludes [NII]/Halpha from their quantitative scatter
# comparison (no Table 4 entry, unlike [NII]/[OII]'s 0.04 or R23's 0.07) and
# state plainly it is "relatively insensitive to abundance" -- n2ha_q3_e8's
# own uncertainty (above, unused now) is therefore just the propagated
# measurement error, not a KD02-vetted combined uncertainty. Kewley et al.
# (2019) Table 2 gives a continuous bicubic surface for this SAME [NII]/Halpha
# ratio, using log(U) (from log_q_A below) instead of a fixed q-track, and
# quotes an actual 0.67% RMS scatter -- use that instead (see k19_n2ha_q).

# q_A/q_A_unc are linear (cm/s) -- same delta-method conversion to log-space
# as Source B's log_q_B above: sigma_log(q) = sigma_q/(q*ln10).
log_q_A = np.log10(q_A)
log_q_A_unc = q_A_unc / (q_A * np.log(10))
print(f"Source A log(q) = {log_q_A:.3f} +- {log_q_A_unc:.3f}")

abundance_A, abundance_A_unc = k19_n2ha_q(A['log_N2'], A['log_N2_err'], log_q_A, log_q_A_unc)


# FINAL
print(f"Source A 12+log(O/H): {abundance_A:.5f} +- {abundance_A_unc:.5f}")

write_physical_values('A', [
    dict(quantity='abundance_cumulative_12log(O/H)', value=abundance_A, uncertainty=abundance_A_unc,
         notes='Kewley+2019 Table 2 [NII]/Halpha bicubic surface (k19_n2ha_q), using log(U) '
               'derived from log_q (KD02 z_halfsolar_combined) -- KD02\'s own [NII]/[OII] and '
               'R23 branches did not converge/apply for this source; uncertainty includes K19\'s '
               'quoted 0.67% RMS scatter added in quadrature to the propagated measurement error'),
    dict(quantity='log_q', value=log_q_A, uncertainty=log_q_A_unc,
         notes='KD02 ionization parameter (z_halfsolar_combined, KD02_O32)'),
])

# alternative Source B abundance from the same K19 [NII]/Halpha surface used
# for Source A -- added alongside (not replacing) the primary KD02
# R23-branch-averaged abundance_cumulative_12log(O/H) row above. [NII] is a
# 3-sigma upper limit for Source B, so log_N2_err is NaN and this inherits a
# NaN uncertainty (value is a ceiling), same convention as every other
# [NII]-derived quantity for Source B in this pipeline.
abundance_B_k19, abundance_B_k19_unc = k19_n2ha_q(B['log_N2'], B['log_N2_err'], log_q_B, log_q_B_unc)
print(f"Source B 12+log(O/H) [K19 N2Ha alt] = {abundance_B_k19:.3f} +- {abundance_B_k19_unc:.3f}")

write_physical_values('B', [
    dict(quantity='abundance_cumulative_12log(O/H)', value=abundance_B, uncertainty=abundance_B_unc,
         notes='KD02 R23 branch-averaged method'),
    dict(quantity='abundance_cumulative_12log(O/H)_K19_N2Ha', value=abundance_B_k19, uncertainty=abundance_B_k19_unc,
         notes='ALTERNATIVE estimate, not the primary abundance above -- Kewley+2019 Table 2 '
               '[NII]/Halpha bicubic surface (k19_n2ha_q), using log(U) derived from log_q '
               '(KD02/KK04); [NII] is a 3-sigma upper limit for Source B so uncertainty is NaN '
               '(value is a ceiling)'),
    dict(quantity='log_q', value=log_q_B, uncertainty=log_q_B_unc, notes='KD02/KK04 ionization parameter'),
])

# ========================
# Calculate galaxy component metallicity
# ========================
# SOURCE B
# ITERATION 1
abundance_Bgal_initialguess = n2ha_q1_e8(B_central['log_N2'], B_central['log_N2_err'])[0] #8.18
qbgal_fifthsolar, qbgal_fifthsolar_unc = z_fifthsolar_combined(B_central['log_KD02_O32'], B_central['log_KD02_O32_err']) #1e8.02
abundance_Bgal_qe8, abundance_Bgal_qe8_unc = r23_q1_e8_combined(B_central['log_kk04_R23'], B_central['log_kk04_R23_err']) #8.016 
#which is close to 0.1Z solar 

#ITERATION 2 
qbgal_tenthsolar, qbgal_tenthsolar_unc = z_tenthsolar_combined(B_central['log_KD02_O32'], B_central['log_KD02_O32_err']) #7e7 
abundance_Bgal_q8e7, abundance_Bgal_q8e7_unc = r23_q8_e7_combined(B_central['log_kk04_R23'], B_central['log_kk04_R23_err']) #1e8.14
#now has returned to being close to 0.2Z_solar again, and ionization parameter oscillated similarly. so follow same process as cumulative 
abundance_Bgal = (abundance_Bgal_qe8 + abundance_Bgal_q8e7)/2
abundance_Bgal_unc_input = 0.5*np.sqrt(abundance_Bgal_qe8_unc**2 + abundance_Bgal_q8e7_unc**2 )
abundance_Bgal_unc = np.sqrt(0.07**2 +abundance_Bgal_unc_input**2)
q_Bgal = (qbgal_fifthsolar + qbgal_tenthsolar)/2
q_Bgal_unc = 0.5*np.sqrt(qbgal_fifthsolar_unc**2 + qbgal_tenthsolar_unc**2)

print(f"Source B gal 12+log(O/H) = {abundance_Bgal:.3f} +- {abundance_Bgal_unc:.3f}")
# same fix as Source B cumulative above -- q_Bgal/q_Bgal_unc are linear
log_q_Bgal = np.log10(q_Bgal)
log_q_Bgal_unc = q_Bgal_unc / (q_Bgal * np.log(10))
print(f"Source B gal log(q) = {log_q_Bgal:.3f} +- {log_q_Bgal_unc:.3f}")

# SOURCE A is the same abundance as the cumulative, because source A had only one component of OII
high_metallicity_q_solar(A_central['log_KD02_O32'], A_central['log_KD02_O32_err'])
kk04_q_Agal, kk04_q_Agal_unc = kk04_q(abundance_A, abundance_A_unc, A_central['log_KK04_O32'], A_central['log_KK04_O32_err'])