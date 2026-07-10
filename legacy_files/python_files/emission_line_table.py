import dill
import numpy as np
import pandas as pd

# ==================
# load pkl files
# (fluxes/ratios are pulled as already computed elsewhere;
#  gaussian pkls are only used to read off widths, never re-fit or re-integrated)
# ==================
with open('./output/OIII/OIII_Hbeta_ratios.pkl', 'rb') as f:
    OIIIbetas = dill.load(f)
with open('./output/NII/NII_Halpha_ratios.pkl', 'rb') as f:
    NIIalphas = dill.load(f)
with open('./output/OII/OII_fluxes.pkl', 'rb') as f:
    OIIfluxes = dill.load(f)
with open('./output/balmer/balmer_decrement_ratios.pkl', 'rb') as f:
    balmer = dill.load(f)
with open('./output/tabling/A_ratios.pkl', 'rb') as f:
    A_ratios = dill.load(f)
with open('./output/tabling/B_ratios.pkl', 'rb') as f:
    B_ratios = dill.load(f)

with open('./output/OIII/OIII_Hbeta_bestfit_gaussians.pkl', 'rb') as f:
    m_oiii = dill.load(f)
with open('./output/NII/Halpha_NII_bestfit_gaussians.pkl', 'rb') as f:
    m_nii = dill.load(f)
with open('./output/balmer/ori_balmer_bestfit_gaussians.pkl', 'rb') as f:
    m_balmer = dill.load(f)
with open('./output/OII/OII_wingfit_gaussians.pkl', 'rb') as f:
    m_oii = dill.load(f)

# ==================
# helpers (lookups only, no fitting/integration)
# ==================
def width(model, idx):
    """Gaussian sigma (Angstrom) and its uncertainty for component `idx` of a fitted model."""
    stddev = getattr(model, f'stddev_{idx}').value
    try:
        stddev_unc = model.stds[f'stddev_{idx}']
    except (KeyError, ValueError, TypeError):
        stddev_unc = np.nan
    return stddev, stddev_unc


def flux_row(line, flux, flux_unc, w, w_unc, notes=''):
    return dict(line=line, flux_ergs_s_cm2=flux, flux_uncert=flux_unc,
                width_AA=w, width_uncert_AA=w_unc, notes=notes)


def build_flux_table(source, oiii_idx, nii_idx, balmer_idx, oii_idx, oii_directly_fit):
    """
    oiii_idx / nii_idx / balmer_idx: (line1_idx, line2_idx, line3_idx) into the
        respective bestfit_gaussians CompoundModel for this source.
        oiii_idx order = (Hbeta, [OIII]4959, [OIII]5007) -- only 4959/5007 used here,
            since Hbeta's width+flux come from the joint Balmer fit.
        nii_idx order   = ([NII]6583, [NII]6548) -- only the [NII] lines used here,
            since Halpha's width+flux come from the joint Balmer fit.
        balmer_idx order = (Halpha, Hbeta, Hgamma).
    oii_idx: index into OII_wingfit_gaussians for the directly-fit OII component.
    oii_directly_fit: '3726' or '3729' -- which line is directly fit for this source
        (the other is derived via the low-density 1.5 ratio, already baked into OII_fluxes.pkl).
    """
    rows = []

    w_oii, w_oii_unc = width(m_oii, oii_idx)
    if oii_directly_fit == '3729':
        rows.append(flux_row('[OII]3726', OIIfluxes['fluxes'][source][0], OIIfluxes['flux_uncerts'][source][0],
                              np.nan, np.nan, 'derived from [OII]3729 / 1.5 (low-density limit)'))
        rows.append(flux_row('[OII]3729', OIIfluxes['fluxes'][source][1], OIIfluxes['flux_uncerts'][source][1],
                              w_oii, w_oii_unc, 'directly fit'))
    else:
        rows.append(flux_row('[OII]3726', OIIfluxes['fluxes'][source][0], OIIfluxes['flux_uncerts'][source][0],
                              w_oii, w_oii_unc, 'directly fit'))
        rows.append(flux_row('[OII]3729', OIIfluxes['fluxes'][source][1], OIIfluxes['flux_uncerts'][source][1],
                              np.nan, np.nan, 'derived from [OII]3726 * 1.5 (low-density limit)'))

    w, w_unc = width(m_balmer, balmer_idx[2])
    rows.append(flux_row('Hgamma', balmer['fluxes'][source][2], balmer['flux_uncerts'][source][2], w, w_unc,
                          'from joint Balmer fit'))

    w, w_unc = width(m_balmer, balmer_idx[1])
    rows.append(flux_row('Hbeta', balmer['fluxes'][source][1], balmer['flux_uncerts'][source][1], w, w_unc,
                          'from joint Balmer fit'))

    w, w_unc = width(m_oiii, oiii_idx[1])
    rows.append(flux_row('[OIII]4959', OIIIbetas['fluxes'][source][1], OIIIbetas['flux_uncerts'][source][1], w, w_unc))

    w, w_unc = width(m_oiii, oiii_idx[2])
    rows.append(flux_row('[OIII]5007', OIIIbetas['fluxes'][source][2], OIIIbetas['flux_uncerts'][source][2], w, w_unc))

    w, w_unc = width(m_nii, nii_idx[2])
    rows.append(flux_row('[NII]6548', NIIalphas['fluxes'][source][2], NIIalphas['flux_uncerts'][source][2], w, w_unc))

    w, w_unc = width(m_balmer, balmer_idx[0])
    rows.append(flux_row('Halpha', balmer['fluxes'][source][0], balmer['flux_uncerts'][source][0], w, w_unc,
                          'from joint Balmer fit'))

    w, w_unc = width(m_nii, nii_idx[1])
    rows.append(flux_row('[NII]6583', NIIalphas['fluxes'][source][1], NIIalphas['flux_uncerts'][source][1], w, w_unc))

    return pd.DataFrame(rows).round(3)


def build_ratio_table(ratios_dict):
    rows = []
    for key, val in ratios_dict.items():
        if key.endswith('_err'):
            continue
        rows.append(dict(ratio=key, value=val, uncert=ratios_dict.get(f'{key}_err', np.nan)))
    return pd.DataFrame(rows).round(3)


# ==================
# build tables
# ==================
df_A_fluxes = build_flux_table('A', oiii_idx=(0, 1, 2), nii_idx=(0, 1, 2), balmer_idx=(0, 1, 2),
                                oii_idx=0, oii_directly_fit='3729')
df_B_fluxes = build_flux_table('B', oiii_idx=(3, 4, 5), nii_idx=(3, 4, 5), balmer_idx=(3, 4, 5),
                                oii_idx=1, oii_directly_fit='3726')
df_A_ratios = build_ratio_table(A_ratios)
df_B_ratios = build_ratio_table(B_ratios)

# ==================
# save
# ==================
df_A_fluxes.to_csv('./output/tabling/emission_lines_A_fluxes.csv', index=False)
df_B_fluxes.to_csv('./output/tabling/emission_lines_B_fluxes.csv', index=False)
df_A_ratios.to_csv('./output/tabling/emission_lines_A_ratios.csv', index=False)
df_B_ratios.to_csv('./output/tabling/emission_lines_B_ratios.csv', index=False)

print("Source A fluxes:")
print(df_A_fluxes.to_string(index=False))
print()
print("Source B fluxes:")
print(df_B_fluxes.to_string(index=False))
print()
print("Source A ratios:")
print(df_A_ratios.to_string(index=False))
print()
print("Source B ratios:")
print(df_B_ratios.to_string(index=False))
