import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from functools import reduce
import operator

plt.ion()  

# ======================================
# initialize information
# ======================================
# balmer emission lines in air 

lines = {
    'H_alpha' : 6562.819,
    'H_beta' : 4861.333,
    'H_gamma' : 4340.471
}

# import the gauss parameters from pkl 
with open('./output/ori_balmer_bestfit_gaussians.pkl','rb') as f:
    bestfit_model = dill.load(f)

# separate out the two sources (and continuum) and extract the parameters 
means_A = []
means_B = []
stddevs_A = []
stddevs_B = []
amps_A = []
amps_B = []
for i in range(len(lines)):
    means_A.append(getattr(bestfit_model, f'mean_{i}').value)
    means_B.append(getattr(bestfit_model, f'mean_{len(lines)+i}').value)
    stddevs_A.append(getattr(bestfit_model, f'stddev_{i}').value)
    stddevs_B.append(getattr(bestfit_model, f'stddev_{len(lines)+i}').value)
    amps_A.append(getattr(bestfit_model, f'amplitude_{i}').value)
    amps_B.append(getattr(bestfit_model, f'amplitude_{len(lines)+i}').value)
gauss_params = {
    'A': { 'means': np.array(means_A),
          'amplitudes': np.array(amps_A),
          'stddevs': np.array(stddevs_A)
    },
    'B': { 'means': np.array(means_B),
          'amplitudes': np.array(amps_B),
          'stddevs': np.array(stddevs_B)
    }
}
continuum = getattr(bestfit_model, f'amplitude_{len(lines)*2}').value

# define integration function for one gaussian 
def integrate(amp, std):
    return amp * std * np.sqrt(2 * np.pi) # gaussian's technically don't "end", but the indefinite integral int_-inf^inf(gaussian(lam) dlam) = A * stddev * root(2pi)

# run on all gaussians for each source 
fluxes_A = np.zeros(len(lines))
fluxes_B = np.zeros(len(lines))
for i in range(len(lines)):
    fluxes_A[i] = integrate(gauss_params['A']['amplitudes'][i],gauss_params['A']['stddevs'][i])
    fluxes_B[i] = integrate(gauss_params['B']['amplitudes'][i],gauss_params['B']['stddevs'][i])

# calculate ratios 
ab_A = fluxes_A[0]/fluxes_A[1]
ab_B = fluxes_B[0]/fluxes_B[1]
gb_A = fluxes_A[2]/fluxes_A[1]
gb_B = fluxes_B[2]/fluxes_B[1]

# Case B 
ab = 2.86
gb = 0.47

# print 
print(f'H_alpha/H_beta for source A: {ab_A}')
print(f'H_alpha/H_beta for source B: {ab_B}')
print(f'Values for H_alpha/H_beta GREATER than the Case B value of {ab} indicate that dust is present')
print(f'H_gamma/H_beta for source A: {gb_A}')
print(f'H_gamma/H_beta for source B: {gb_B}')
print(f'Values for H_gamma/H_beta LESSER than the Case B value of {gb} indicate that dust is present')

# save 
balmer_ratios = {
    'A': {
        'Halpha_Hbeta': ab_A,
        'Hgamma_Hbeta': gb_A,
    },
    'B': {
        'Halpha_Hbeta': ab_B,
        'Hgamma_Hbeta': gb_B,
    },
    'case_B': {
        'Halpha_Hbeta': 2.86,
        'Hgamma_Hbeta': 0.47,
    },
    # keep fluxes too
    'luminosities': {
        'A': fluxes_A,
        'B': fluxes_B,
    },
}

with open('./output/balmer_decrement_ratios.pkl', 'wb') as f:
    dill.dump(balmer_ratios, f)