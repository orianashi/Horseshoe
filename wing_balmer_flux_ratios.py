import dill
import numpy as np

# ======================================
# initialize information
# ======================================
# NOTE : flux units are in ergs/s/cm^2 /Angstrom

# component_fluxes['B'] / component_flux_uncerts['B'] in each *_fluxes.pkl
# (written by multiple_gaussian_integration.py) is ordered [red_wing, central,
# blue_wing] -- confirmed by comparing the fitted means of each line's
# B_indices components (highest mean = reddest, listed first).
WING_NAMES = ['red_wing', 'central', 'blue_wing']

LINE_PKLS = {
    'Halpha': './output/improved_gaussians/Halpha/Halpha_fluxes.pkl',
    'Hbeta': './output/improved_gaussians/Hbeta/Hbeta_fluxes.pkl',
    'Hgamma': './output/improved_gaussians/Hgamma/Hgamma_fluxes.pkl',
}


def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)


data = {name: load_pkl(path) for name, path in LINE_PKLS.items()}


# define calculate ratios function
def ratios(num, denom, num_uncert, denom_uncert):
    ratio = num / denom
    ratio_uncert = np.sqrt((num_uncert / num)**2 +
                           (denom_uncert / denom)**2) * ratio
    return ratio, ratio_uncert


# run for each wing, for Source B
results = {}
for j, wing in enumerate(WING_NAMES):
    wing_data = {}
    for line in ('Halpha', 'Hbeta', 'Hgamma'):
        flux = data[line]['component_fluxes']['B'][j]
        uncert = data[line]['component_flux_uncerts']['B'][j]
        wing_data[line] = {'flux': flux, 'flux_uncert': uncert}

    ab, ab_uncert = ratios(wing_data['Halpha']['flux'],
                           wing_data['Hbeta']['flux'],
                           wing_data['Halpha']['flux_uncert'],
                           wing_data['Hbeta']['flux_uncert'])
    gb, gb_uncert = ratios(wing_data['Hgamma']['flux'],
                           wing_data['Hbeta']['flux'],
                           wing_data['Hgamma']['flux_uncert'],
                           wing_data['Hbeta']['flux_uncert'])
    wing_data['Halpha_Hbeta'] = {'ratio': ab, 'ratio_uncert': ab_uncert}
    wing_data['Hgamma_Hbeta'] = {'ratio': gb, 'ratio_uncert': gb_uncert}

    print(f"--- {wing} (Source B) ---")
    print(
        f"Halpha: {wing_data['Halpha']['flux']:.3f} +/- {wing_data['Halpha']['flux_uncert']:.3f} ergs/s/cm2"
    )
    print(
        f"Hbeta:  {wing_data['Hbeta']['flux']:.3f} +/- {wing_data['Hbeta']['flux_uncert']:.3f} ergs/s/cm2"
    )
    print(
        f"Hgamma: {wing_data['Hgamma']['flux']:.3f} +/- {wing_data['Hgamma']['flux_uncert']:.3f} ergs/s/cm2"
    )
    print(f"Halpha/Hbeta: {ab:.3f} +/- {ab_uncert:.3f}")
    print(f"Hgamma/Hbeta: {gb:.3f} +/- {gb_uncert:.3f}")

    results[wing] = wing_data

results['units'] = 'ergs / s / cm2'

with open('./output/diagnostics/sourceB_wing_balmer_fluxes.pkl', 'wb') as f:
    dill.dump(results, f)
