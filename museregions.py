import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import pyregion
from matplotlib.ticker import MultipleLocator

plt.ion()

#REGION_FILE = './Data/MUSE/regions/7to9region.reg'
REGION_FILE = './Data/MUSE/regions/3to5region.reg'
LINE_FILE = './absorption_wls/horseshoe_atoms.dat'

z_B = 1.677  # redshift for source B (repo-wide convention)
z_A = 1.679

# --- load the cube ---
hdul = fits.open('./Data/MUSE/muse.fits', memmap=True)
data_hdu = hdul['DATA']
stat_hdu = hdul['STAT']
cube = data_hdu.data          # (n_wave, ny, nx)
var_cube = stat_hdu.data      # variance, same shape
header = data_hdu.header

# --- build a *2D* spatial header for the region->mask step ---
# This is the #1 gotcha: pyregion needs a 2D WCS, not the 3D cube header.
wcs2d = WCS(header).celestial
header2d = wcs2d.to_header()

# --- region mask (unions every shape in the region file) ---
reg = pyregion.open(REGION_FILE)
mask = reg.get_mask(header=header2d, shape=(header['NAXIS2'], header['NAXIS1']))
print(f'region mask: {mask.sum()} pixels')

# --- integrate flux (and propagate variance) over the masked spaxels ---
flux_obs = np.nansum(cube[:, mask], axis=1)
var_obs = np.nansum(var_cube[:, mask], axis=1)
err_obs = np.sqrt(var_obs)

hdul.close()

# --- observed-frame wavelength axis, then de-redshift to rest frame ---
n_wave = header['NAXIS3']
lam_obs = header['CRVAL3'] + (np.arange(n_wave) + 1 - header['CRPIX3']) * header['CD3_3']
lam_rest = lam_obs / (1 + z_B)

# --- load the horseshoe absorption line list (rest-frame wavelengths) ---
horseshoe_ref = np.genfromtxt(LINE_FILE, dtype=str)
line_names = horseshoe_ref[:, 0]
line_wl = horseshoe_ref[:, 1].astype(float)

# --- plot ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(lam_rest, flux_obs, ds='steps-mid', color='black', lw=0.8)
ax.fill_between(lam_rest, flux_obs - err_obs, flux_obs + err_obs,
                 alpha=0.2, color='lightgrey', label='1 sigma')

in_range = (line_wl > lam_rest.min()) & (line_wl < lam_rest.max())
for name, wl in zip(line_names[in_range], line_wl[in_range]):
    ax.axvline(wl, color='red', ls='--', alpha=0.5, lw=0.8)
    ax.text(wl, 1.01, name, transform=ax.get_xaxis_transform(),
            fontsize=5.5, ha='center', color='red')

ax.xaxis.set_minor_locator(MultipleLocator(10))
ax.set_xlabel(f'Rest-frame Wavelength [Angstrom] (source B, z={z_B})')
ax.set_ylabel(f"Flux [{header.get('BUNIT', '')}]")
ax.set_title(f'MUSE integrated spectrum: z = 1.677 ({os.path.basename(REGION_FILE)})')
ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()

plt.show()
