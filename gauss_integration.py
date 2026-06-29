import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

plt.ion()  

# ======================================
# initialize information
# ======================================
# NOTE : flux units are in ergs/s/cm^2 /Angstrom

# balmer emission lines in air for reference (not used in calculations )
# NOTE: if you change these, make sure to update the filename ratios to calcualte what you want 
#lines = ["H_alpha", "H_beta", "H_gamma"]
lines = ["H_alpha", "[NII]6583", "[NII]6548"] 

# import the gauss parameters from pkl 
with open('./output/NII//Halpha_NII_bestfit_gaussians.pkl','rb') as f:
    bestfit_model = dill.load(f)

# separate out the two sources (and continuum) and extract the parameters 
means_A = []
means_B = []
stddevs_A = []
stddevs_B = []
amps_A = []
amps_B = []
b_uncert_base = 3 + (len(lines)-1) * 2  #7 for 3 lines
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
          'stddevs': np.array(stddevs_A),
          'mean_uncert': bestfit_model.stds[1],
          'amplitudes_uncert': np.array([bestfit_model.stds[0], bestfit_model.stds[3], bestfit_model.stds[5]]),
          'stddevs_uncert': np.array([bestfit_model.stds[2],bestfit_model.stds[4],bestfit_model.stds[6]])
    },
    'B': { 'means': np.array(means_B),
          'amplitudes': np.array(amps_B),
          'stddevs': np.array(stddevs_B),
          'mean_uncert': bestfit_model.stds[b_uncert_base+1],
          'amplitudes_uncert': np.array([bestfit_model.stds[b_uncert_base],bestfit_model.stds[b_uncert_base+ 3], bestfit_model.stds[b_uncert_base+5]]),
          'stddevs_uncert': np.array([bestfit_model.stds[b_uncert_base+2],bestfit_model.stds[b_uncert_base + 4], bestfit_model.stds[b_uncert_base+6]])
    }
}
continuum = getattr(bestfit_model, f'amplitude_{len(lines)*2}').value


# define integration function for one gaussian 
def integrate(amp, stddev, amp_uncert, stddev_uncert):
    area = amp * stddev * np.sqrt(2 * np.pi) # gaussian's technically don't "end", but the indefinite integral int_-inf^inf(gaussian(lam) dlam) = A * stddev * root(2pi)
    area_uncert = np.sqrt((amp_uncert/amp)**2 + (stddev_uncert/stddev)**2 ) * area 
    return area, area_uncert

# define calculate ratios function
def ratios(num, denom, num_uncert, denom_uncert):
    ratio = num/denom
    ratio_uncert = np.sqrt((num_uncert/num)**2 + (denom_uncert/denom)**2) * ratio
    return ratio, ratio_uncert 

# run on all gaussians for each source 
fluxes_A = np.zeros(len(lines))
fluxes_B = np.zeros(len(lines))
flux_uncerts_A = np.zeros(len(lines))
flux_uncerts_B = np.zeros(len(lines))
for i in range(len(lines)):
    fluxes_A[i], flux_uncerts_A[i] = integrate(gauss_params['A']['amplitudes'][i],gauss_params['A']['stddevs'][i], gauss_params['A']['amplitudes_uncert'][i],gauss_params['A']['stddevs_uncert'][i])
    fluxes_B[i], flux_uncerts_B[i] = integrate(gauss_params['B']['amplitudes'][i],gauss_params['B']['stddevs'][i], gauss_params['A']['amplitudes_uncert'][i],gauss_params['A']['stddevs_uncert'][i])

for i in range(len(fluxes_A)):
    print(f"Flux for {lines[i]} for source A: {fluxes_A[i]:.3f} +/- {np.abs(flux_uncerts_A[i]):.3f} ergs/s/cm2")
    print(f"Flux for {lines[i]} for source B: {fluxes_B[i]:.3f} +/- {np.abs(flux_uncerts_B[i]):.3f} ergs/s/cm2")
    
# NII6584/Halpha.
NIIalpha_A, NIIalpha_A_uncert = ratios(fluxes_A[1], fluxes_A[0], flux_uncerts_A[1], flux_uncerts_A[0])
NIIalpha_B, NIIalpha_B_uncert = ratios(fluxes_B[1], fluxes_B[0], flux_uncerts_B[1], flux_uncerts_B[0])
print(f"[NII]/Halpha for source A: {NIIalpha_A} +- {NIIalpha_A_uncert}")
print(f"[NII]/Halpha for source B: {NIIalpha_B} +- {NIIalpha_B_uncert}")

#save 
NIIalphas = {
    "A": {
        'NIIalpha': NIIalpha_A,
        'NIIalpha_uncert': NIIalpha_A_uncert
    },
     "B": {
        'NIIalpha': NIIalpha_B,
        'NIIalpha_uncert': NIIalpha_B_uncert
    }
}
with open('./output/NII/NII_Halpha_ratios.pkl', 'wb') as f:
    dill.dump(NIIalphas, f)


"""
#BALMER DECREMENTS

ab_A, ab_A_uncert = ratios(fluxes_A[0], fluxes_A[1], flux_uncerts_A[0], flux_uncerts_A[1])
ab_B, ab_B_uncert = ratios(fluxes_B[0],fluxes_B[1], flux_uncerts_B[0], flux_uncerts_B[1])
gb_A, gb_A_uncert = ratios(fluxes_A[2], fluxes_A[1], flux_uncerts_A[2], flux_uncerts_A[1])
gb_B, gb_B_uncert = ratios(fluxes_B[2], fluxes_B[1], flux_uncerts_B[2], flux_uncerts_B[1])

# Case B 
ab = 2.86
gb = 0.47

# print 
print(f'H_alpha/H_beta for source A: {ab_A} +- {ab_A_uncert}')
print(f'H_alpha/H_beta for source B: {ab_B} +- {ab_B_uncert}')
print(f'Values for H_alpha/H_beta GREATER than the Case B value of {ab} indicate that dust is present')
print(f'H_gamma/H_beta for source A: {gb_A} +- {gb_A_uncert}')
print(f'H_gamma/H_beta for source B: {gb_B} +- {gb_B_uncert}')
print(f'Values for H_gamma/H_beta LESSER than the Case B value of {gb} indicate that dust is present')

# save 
balmer_ratios = {
    'A': {
        'Halpha_Hbeta': ab_A,
        'Hgamma_Hbeta': gb_A,
        'Halpha_Hbeta_1sig': ab_A_uncert,
        'Hgamma_Hbeta_1sig': gb_A_uncert
    },
    'B': {
        'Halpha_Hbeta': ab_B,
        'Hgamma_Hbeta': gb_B,
        'Halpha_Hbeta_1sig': ab_B_uncert,
        'Hgamma_Hbeta_1sig': gb_B_uncert
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
    'units' : 'ergs / s / cm2'
}

with open('./output/balmer/balmer_decrement_ratios.pkl', 'wb') as f:
    dill.dump(balmer_ratios, f)

print("="*20)
print("percentage diff calc e.g. ab_A / ab")
print(ab_A / ab * 100)
print(ab_B / ab * 100)
print(gb_A / gb * 100)
print(gb_B / gb * 100)
"""
# bolometric flux units:  ergs/s/cm^2 