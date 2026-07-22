import os

import dill
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


plt.ion()

# My Datapoints
def load_source(source, abundance_quantity='abundance_cumulative_12log(O/H)'):
    # skipinitialspace + strip: some editor/linter re-aligns this file with
    # padding spaces around each field, including the header itself (same
    # convention as diagnostics_dust_corrected.py's write_physical_values) --
    # so columns/quantity must be stripped BEFORE setting the index, since
    # index_col='quantity' would otherwise look for an exact match against
    # the padded header name and fail.
    df = pd.read_csv(f'./output/physical_values_{source}.csv', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df['quantity'] = df['quantity'].astype(str).str.strip()
    df = df.set_index('quantity')
    mass = df.loc['log_M_btfr6_26_corrected_45deg', 'value']
    mass_unc = df.loc['log_M_btfr6_26_corrected_45deg', 'uncertainty']
    z = df.loc[abundance_quantity, 'value']
    z_unc = df.loc[abundance_quantity, 'uncertainty']
    return mass, mass_unc, z, z_unc

mass_A, mass_A_unc, z_A, z_A_unc = load_source('A')
mass_B, mass_B_unc, z_B, z_B_unc = load_source('B')
# alternative Source B abundance from the Kewley+2019 Table 2 [NII]/Halpha
# bicubic surface (see metallicity.py's k19_n2ha_q) -- an alternative to the
# primary KD02 R23-branch-averaged abundance loaded into z_B above.
_, _, z_B_k19, z_B_k19_unc = load_source(
    'B', abundance_quantity='abundance_cumulative_12log(O/H)_K19_N2Ha')


# ==================
# EXTENDED M-Z PLOTS
# ==================
def make_extended_mz_plot(z_B_value, z_B_unc_value, output_path,
                          source_b_label=r"Source B $z$=1.677", source_b_fmt="o"):
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    plt.rcParams['font.family'] = 'serif'
    ax1.set_xlim(8.2, 11.7)
    ax1.set_ylim(7.7, 9.2)
    ax1.set_xlabel(r"Log $M_*$[$M_\odot$]")
    ax1.set_ylabel("12 + log(O/H) [N2; KD02]")
    ax1.set_title("Extended Mass-Metallicity Relationships")
    ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))

    # tremonti -- fitted range (8.5-11.5) matches the calibration-limited plot;
    # below that is extrapolation (faded, no legend entry), same treatment as
    # the Mass-SFR plot's Kashino+19 fit/extrapolation split
    t04x = np.linspace(8.5, 11.5, 500)
    t04y = -1.492 + 1.847*t04x - 0.08026*t04x**2  # eq 3
    ax1.plot(t04x, t04y, ls="-.", color="black", lw=1, label=r"$z$~0.1 (Tremonti+04)")

    t04x_lo = np.linspace(8.2, 8.5, 500)
    t04y_lo = -1.492 + 1.847*t04x_lo - 0.08026*t04x_lo**2
    ax1.plot(t04x_lo, t04y_lo, ls=":", color="black", alpha=0.6, lw=1)

    # 1.4 Yabe 14 -- fitted range (9.5-11) matches the calibration-limited plot
    y14x_ext = np.linspace(9.5, 11, 500)
    y14y_ext = -0.1082*y14x_ext**2 + 2.497*y14x_ext - 5.71
    ax1.plot(y14x_ext, y14y_ext, ls="--", color="red", lw=1, label=r"$z$~1.4 (Yabe+14)")

    y14x_ext_lo = np.linspace(8.2, 9.5, 500)
    y14y_ext_lo = -0.1082*y14x_ext_lo**2 + 2.497*y14x_ext_lo - 5.71
    ax1.plot(y14x_ext_lo, y14y_ext_lo, ls=":", color="red", alpha=0.6, lw=1)

    # ~1.6 FMOS -- fitted range (9.65-11.5) matches the calibration-limited plot
    fmos22x_ext = np.linspace(9.65, 11.5, 500)
    fmos22y_ext = 9.07 + np.log10(1 - np.exp((-10**(0.71*(fmos22x_ext-10.5)))))
    ax1.plot(fmos22x_ext, fmos22y_ext, ls="-", color="green", lw=1.6, label=r"$z$~1.6 (Kashino+18)")

    fmos22x_ext_lo = np.linspace(8.2, 9.65, 500)
    fmos22y_ext_lo = 9.07 + np.log10(1 - np.exp((-10**(0.71*(fmos22x_ext_lo-10.5)))))
    ax1.plot(fmos22x_ext_lo, fmos22y_ext_lo, ls=":", color='green', alpha=0.6, lw=1)

    # Erb 06 -- fitted range (9.4-11) matches the calibration-limited plot
    e06x_ext = np.linspace(9.4, 11, 500)
    e06y_ext = -1.492 + 1.847*e06x_ext - 0.08026*e06x_ext**2 - 0.3  # eq 3
    ax1.plot(e06x_ext, e06y_ext, ls="--", color='blue', lw=1, label=r"$z$~2 (Erb+06)")

    e06x_ext_lo = np.linspace(8.2, 9.4, 500)
    e06y_ext_lo = -1.492 + 1.847*e06x_ext_lo - 0.08026*e06x_ext_lo**2 - 0.3
    ax1.plot(e06x_ext_lo, e06y_ext_lo, ls=":", color='blue', alpha=0.6, lw=1)

    # My sources
    ax1.errorbar(mass_A, z_A, xerr=mass_A_unc, yerr=z_A_unc, fmt="o", color="orange",
                markersize=6, lw=0.45, label=r"Source A $z$=1.679")
    ax1.errorbar(mass_B, z_B_value, xerr=mass_B_unc, yerr=z_B_unc_value, fmt=source_b_fmt, color="purple",
                markersize=6, lw=0.45, label=source_b_label)

    plt.legend(loc="lower right")
    plt.show()
    fig1.savefig(output_path)
    return fig1


# once using the current (primary, KD02 R23-branch-averaged) Source B abundance
make_extended_mz_plot(z_B, z_B_unc, "./output/extended_M_Z")

# once using the alternative Kewley+2019 [NII]/Halpha Source B abundance --
# [NII] is a 3-sigma upper limit for B, and dz/dx > 0 at this source's
# operating point on the K19 surface, so this abundance is itself an upper
# bound on the true metallicity; a downward-pointing triangle marks that.
make_extended_mz_plot(z_B_k19, z_B_k19_unc, "./output/extended_M_Z_B_K19_N2Ha",
                      source_b_label=r"Source B $z$=1.677 (K19 N2H$\alpha$)",
                      source_b_fmt="v")


"""
# ==================
# MASS - SFR
# ==================

fig2, ax2 = plt.subplots(1,1, figsize=(10,6))
plt.rcParams['font.family'] = 'serif'
ax2.set_xlim(8.2, 10.6)
ax2.set_ylim(-0.75, 1.8)
ax2.set_xlabel(r"Log $M_*$[$M_\odot$]")
ax2.set_ylabel(r"log SFR [$M_\odot\,yr^{-1}$]")
ax2.set_title(r"Stellar Mass - Star Formation Rate Relationship at $z$~1.6")
ax2.xaxis.set_minor_locator(MultipleLocator(0.1))
ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

# FMOS kashino 19 fig 31 eq 13
# but they missed a minus sign from Lee 2013
# sfr = S0 - log10(1 + u), u = 10**(-gamma*(mass-m0))
# d(sfr)/d(mass) = gamma * u / (1 + u)  -- propagates mass_unc into sfr_unc
def k19_sfr(mass, mass_unc):
    gamma = 1.17
    m0 = 10.205
    S0 = 1.74
    u = 10**(-gamma * (mass - m0))
    sfr = S0 - np.log10(1 + u)
    sfr_unc = gamma * u / (1 + u) * mass_unc
    return sfr, sfr_unc
k19_m = np.linspace(9.4, 10.5, 500)
k19_m_ext = np.linspace(8.2, 9.4, 500)
k19_sfr_curve, _ = k19_sfr(k19_m, 0)
k19_sfr_curve_ext, _ = k19_sfr(k19_m_ext, 0)
SFR_A_fmos, SFR_A_fmos_unc = k19_sfr(mass_A, mass_A_unc)
SFR_B_fmos, SFR_B_fmos_unc = k19_sfr(mass_B, mass_B_unc)

ax2.plot(k19_m, k19_sfr_curve, ls = '-', color='green', lw = 1.6, label=r"$z$~1.6 (Kashino+19)")
ax2.plot(k19_m_ext, k19_sfr_curve_ext, ls = ':', color='green', alpha = 0.6, lw = 1)

# Whitaker 2014 eq 2 -- sfr = a + b*mass + c*mass**2, with both the fit
# parameters' own uncertainties and mass_unc propagated through:
# d(sfr)/da = 1, d(sfr)/db = mass, d(sfr)/dc = mass**2, d(sfr)/dmass = b + 2*c*mass
def w14_sfr(mass, mass_unc):
    a, a_unc = -24.04, 2.08
    b, b_unc = 4.17, 0.4
    c, c_unc = -0.16, 0.02
    sfr = a + b * mass + c * mass**2
    sfr_unc = np.sqrt(a_unc**2 + (mass * b_unc)**2 + (mass**2 * c_unc)**2 +
                      ((b + 2 * c * mass) * mass_unc)**2)
    return sfr, sfr_unc
w14_m = np.linspace(9.1, 11.1, 500)
w14_m_ext = np.linspace(8.2,9.1,300)
w14_sfr_curve, _ = w14_sfr(w14_m, 0)
w14_sfr_curve_ext, _ = w14_sfr(w14_m_ext, 0)
SFR_A_w14, SFR_A_w14_unc = w14_sfr(mass_A, mass_A_unc)
SFR_B_w14, SFR_B_w14_unc = w14_sfr(mass_B, mass_B_unc)

ax2.plot(w14_m, w14_sfr_curve, ls = '-', color='blue', lw = 1.6, label=r"1.5<$z$<2 (Whitaker+14)")
ax2.plot(w14_m_ext, w14_sfr_curve_ext, ls = ':', color='blue', alpha = 0.6, lw = 1)

# each source's SFR as predicted by each curve at that source's mass --
# no error bars, just the two curve-predicted points per source
mass_A_points = [mass_A, mass_A]
SFR_A_points = [SFR_A_fmos, SFR_A_w14]
mass_B_points = [mass_B, mass_B]
SFR_B_points = [SFR_B_fmos, SFR_B_w14]

ax2.plot(mass_A_points, SFR_A_points, 'o', color="orange", markersize=5, label=r"Source A $z$=1.679")
ax2.plot(mass_B_points, SFR_B_points, 'o', color="purple", markersize=5, label=r"Source B $z$=1.677")

plt.legend(loc="upper left")
plt.show()
print(f"SFR A: {SFR_A_fmos} +- {SFR_A_fmos_unc}")
print(f"SFR B: {SFR_B_fmos} +- {SFR_B_fmos_unc}")
"""
