import dill
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

plt.ion()

# ======================================
# initialize information
# ======================================
# NOTE : flux units are in ergs/s/cm^2 /Angstrom



# import the gauss parameters from pkl
with open('./output/OII/OII_wingfit_gaussians.pkl', 'rb') as f:
    bestfit_model = dill.load(f)

# separate out the two sources (and continuum) and extract the parameters

gauss_params = {
    'A': {
        'amplitude_3729':
        getattr(bestfit_model, f"amplitude_0").value,
        'mean_3729':
        getattr(bestfit_model, f"mean_0").value,
        'stddev_3729':
        getattr(bestfit_model, f"stddev_0").value,
        'amplitude_3729_uncert':
        bestfit_model.stds[0],
        'mean_3729_uncert':
        bestfit_model.stds[1],
        'stddev_3729_uncert':
        bestfit_model.stds[2]
    },
    'B': {
        'amplitude_3726':
        getattr(bestfit_model, f"amplitude_1").value,
        'mean_3726':
        getattr(bestfit_model, f"mean_1").value,
        'stddev_3726':
        getattr(bestfit_model, f"stddev_1").value,
        'amplitude_3726_uncert':
        bestfit_model.stds[3],
        'mean_3726_uncert':
        bestfit_model.stds[4],
        'stddev_3726_uncert':
        bestfit_model.stds[5]
    },
}
continuum = getattr(bestfit_model, f'amplitude_2').value


# define integration function for one gaussian
def integrate(amp, stddev, amp_uncert, stddev_uncert):
    area = amp * stddev * np.sqrt(
        2 * np.pi
    )  # gaussian's technically don't "end", but the indefinite integral int_-inf^inf(gaussian(lam) dlam) = A * stddev * root(2pi)
    area_uncert = np.sqrt((amp_uncert / amp)**2 +
                          (stddev_uncert / stddev)**2) * area
    return area, area_uncert


# define calculate ratios function
def ratios(num, denom, num_uncert, denom_uncert):
    ratio = num / denom
    ratio_uncert = np.sqrt((num_uncert / num)**2 +
                           (denom_uncert / denom)**2) * ratio
    return ratio, ratio_uncert


# run on the gaussian for each source
fluxes_A = np.zeros(2)
fluxes_B = np.zeros(2)
flux_uncerts_A = np.zeros(2)
flux_uncerts_B = np.zeros(2)

fluxes_A[1], flux_uncerts_A[1] = integrate(gauss_params['A']['amplitude_3729'], gauss_params['A']['stddev_3729'], gauss_params['A']['amplitude_3729_uncert'], gauss_params['A']['stddev_3729_uncert'])
fluxes_B[0], flux_uncerts_B[0] = integrate(gauss_params['B']['amplitude_3726'], gauss_params['B']['stddev_3726'], gauss_params['B']['amplitude_3726_uncert'], gauss_params['B']['stddev_3726_uncert'])


# introduce the intensity ratios: as log(Ne) -> 0, we have R1 = I(3729)/I(3726) -> 1.5 per Pradhan et al. 2006
fluxes_A[0] = fluxes_A[1] / 1.5
flux_uncerts_A[0]=flux_uncerts_A[1]/1.5
fluxes_B[1] = fluxes_B[0] * 1.5
flux_uncerts_B[1]=flux_uncerts_B[0]*1.5

print(f"Flux for [OII]3726A: {fluxes_A[0]} +- {flux_uncerts_A[0]} ergs/s/cm2")
print(f"Flux for [OII]3729A: {fluxes_A[1]} +- {flux_uncerts_A[1]} ergs/s/cm2")
print(f"Flux for [OII]3726B: {fluxes_B[0]} +- {flux_uncerts_B[0]} ergs/s/cm2")
print(f"Flux for [OII]3729B: {fluxes_B[1]} +- {flux_uncerts_B[1]} ergs/s/cm2")

OIIfluxes = {
    'fluxes': {
        'A': fluxes_A,
        'B': fluxes_B,
    },
    'flux_uncerts': {
        'A':flux_uncerts_A,
        'B': flux_uncerts_B 
    },
    'units' : 'ergs / s / cm2'
}
with open('./output/OII/OII_fluxes.pkl', 'wb') as f:
    dill.dump(OIIfluxes, f)
# bolometric flux units:  ergs/s/cm^2
