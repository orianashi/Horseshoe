import dill
import numpy as np

from multiple_gaussian_integration import LINES, load_model, load_tie_map, flux_and_uncert

# ======================================
# initialize information
# ======================================
# NOTE : flux units are in ergs/s/cm^2 /Angstrom

# Source B's components in each Balmer line's joint-tied-fit model are
# ordered [red_wing, central, blue_wing] in LINES[line]['B_indices'] (see
# balmer_joint_gaussian_fitting.py's index layout) -- flux for each wing is
# computed directly from the joint fit model + tie_map, not from a
# separately-run multiple_gaussian_integration.py output, so this always
# reflects the current joint-fit model.
WING_NAMES = ['red_wing', 'central', 'blue_wing']
BALMER_LINES = ('Halpha', 'Hbeta', 'Hgamma')

models = {}
tie_maps = {}
for line in BALMER_LINES:
    cfg = LINES[line]
    models[line] = load_model(cfg['pkl'])
    tie_maps[line] = load_tie_map(cfg['pkl'])


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
    for line in BALMER_LINES:
        idx = LINES[line]['B_indices'][j]
        flux, uncert = flux_and_uncert(models[line], [idx], tie_maps[line])
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
