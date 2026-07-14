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

# load in everything 


# ==================
# KD02 PROCESS FOR METALLICITIES below 8.5  
# ==================

# use [NII]/Halpha to identify low vs high metallicity regime 
def n2ha_q1_e8(n2ha, n2ha_err):
    k0 = -2983.69
    k1 = 1454.45
    k2 = -266.015
    k3 = 21.6024
    k4 = -0.6566
    abundance = _real_roots([k4, k3, k2, k1, k0], k0, n2ha)
    abundance_err = "placeholder"
    return abundance, abundance_err

def n2ha_q3_e8(n2ha, n2ha_err):
    k0 = -3100.57
    k1 = 1501.77
    k2 = -272.883
    k3 = 22.0132
    k4 = -0.6646
    abundance = _real_roots([k4, k3, k2, k1, k0], k0, n2ha)
    abundance_err = "placeholder"
    return abundance, abundance_err

# use the metallicity + O32 to constrain the ionization parameter using KD02's "improved version"
# 0.5*Z_solar <=> 12 + log(O/H) =~ 8.6
def z_halfsolar_combined(o32, o32_err):
    k0 = 7.73013
    k1 = 0.843125
    k2 = 0.118166
    q = 10**(k0+k1*o32+k2*o32**2)
    qerr = 'placeholder'
    return q, qerr

# use ionization parameter and correct R23 track to get the metallicity again 
# here we are using the low-metallicity track 
def r23_q1_e8_combined(r23, r23_err):
    k0 = -46.1589
    k1 = 11.2557
    k2 = -0.672731
    abundance = (-k1 + np.sqrt(k1**2 - 4*k2*(k0-r23))) / (2*k2)
    abundance_err = 'placeholder'
    return abundance, abundance_err

def r23_q3_e8_combined(r23, r23_err):
    k0 = -45.6075
    k1 = 11.2074
    k2 = -0.674460
    sqrt = k1**2 - 4*k2*(k0-r23)
    abundance = (-k1 + np.sqrt(sqrt)) / (2*k2)
    abundance_err = 'placeholder'
    return abundance, abundance_err

# if the metallicity has changed significantly, go to metallicity + O32 to get the ionization parameter again 
# ideally we would be averaging the R23 diagnostic and the C01 diagnostic, but that uses [SII]

# ==================
# KD02 PROCESS FOR METALLICITIES ABOVE 8.6 
# ==================
# step 1 of KD02 optimized abundance determination
def n2o2_combined(n2o2,n2o2_err):
    abundance = np.log10(1.54020+1.26602*n2o2+0.167977*n2o2**2) + 8.93
    abundance_err = 'placeholder'
    return abundance, abundance_err

# step 2a - if step 1 has abundance >8.6, use kd02_O32 ionization diagnostic 
def high_metallicity_q_halfsolar(o32, o32_err):
    k0 = -81.1880
    k1=27.5082
    k2 = -3.19126
    k3 = 0.128252
    q = _real_roots([k3, k2, k1, k0], k0, o32)
    q_err = "placeholder"
    return q, q_err


# step 2B - if step 1 has abundance <8.6, do the average of Z94 and M91(upper) using kk04_o32 
def z94_m91(r23, r23_err, kk04_o32, kk04_o32_err):
    abundance_z94 = 9.265 - 0.33*r23 - 0.202*r23**2 - 0.207*r23**3 - 0.333*r23**4
    abundance_m91u = 12 - 4.944 + 0.767*r23 + 0.602*r23**2 - kk04_o32 * (0.29 + 0.332*r23-0.331*r23**2)
    avg = (abundance_m91u + abundance_z94)/2 
    avg_err = 'placeholder'
    return avg, avg_err

# step 3b - if step 2 produces estimate velow 8.5, use R23 method 

# SOURCE B 
n2ha_q1_e8(-2.066, 'placeholder')[0] #7.99
z_halfsolar_combined(0.53, 'placeholder')[0] #1.6e8 
r23_q1_e8_combined(0.838, 'placeholder')[0] #8.011.

# SOURCE A
# using the low-metallicity process
n2ha_q3_e8(-1.657, 'placeholder')[0]  #8.54 
z_halfsolar_combined(0.953, 'placeholder')[0] #4.3e8
r23_q3_e8_combined(0.961, 'placeholder')[0] # HERE IS WHERE THERE IS A NAN 
# also, abundanced in 8.5 - 8.9 region can't be reliably determined with R23 

#using the high-metallicity process 
n2o2_combined(-0.638, 'placeholder')[0] # 8.83 
high_metallicity_q_halfsolar(0.953, 'placeholder')[0] #log(q) = 8.62 =~ 4e8
# the [NII]/[OII] diagnostic is only valid for ionization parameters between 5e6 and 3e8 
# HeII1640 emission present for Source A, indicates hard ionizing radiation field ?
