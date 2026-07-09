import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from astropy.io import fits
import astropy.units as u
# ===================
# constants 
# ===================
R_v = 3.1  #Cardelli 
ha_hb = 2.86 # T = 10^4 K and Ne = 10^2 

lines_AA = {
    'Halpha': 6562.819,
    '[OIII]5007': 5006.843,
    '[OIII]4959': 4958.911,
    'Hbeta': 4861.333,
    'Hgamma': 4340.47,
    '[OII]3726': 3726.03,
    '[OII]3729': 3728.815
}

# ====================
# load in data 
# ====================
#load in fluxes

# load in balmer decrements for source A central, red and source B central, red, blue 
bd = 3.5 

# ====================
# define functions 
# ====================
def wave_num(wl_AA):
    wl_AA = wl_AA * u.AA
    wl_um = wl_AA.to(u.um)
    return wl_um.value 

def k(wl_AA):
    wl_AA = wl_AA * u.AA
    wl_um = wl_AA.to(u.um)
    x = 1 / wl_um.value
    y = x - 1.82
    a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
    b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7
    k = R_v * a + b
    k_err = 'PLACEHOLDER '
    return k, k_err

def EB_V(bd, bd_err):
    denom  = k(lines_AA['Hbeta'])[0] - k(lines_AA['Halpha'])[0]
    EB_V = 2.5 / denom * np.log10(bd/2.86)
    EB_V_err = 'PLACEHOLDER'
    return EB_V, EB_V_err 

def flux_correct(flux, flux_err, line, EB_V):
    exp = 0.4*k(lines_AA['line'])[0]*EB_V
    flux_correct = flux * 10**exp 
    flux_correct_err = 'PLACEHOLDER'
    return flux_correct, flux_correct_err 

# ====================
# run while not distinguishing different BD for outflows:   
# ====================
# calculate k(Halpha) and k(Hbeta), which is the same for all 

# calculate E(B-V) for source A and B

# dust-correct cumulative source A flux for each line 

# dust correct cumulative source B flux for each line 

# ====================
# run while distinguishing outflows  
# ====================
#calculate k(Halpha) and k(Hbeta), which is the same for all 

#calculate 5 different E(B-V)s 

# dust-correct source A centrals for each line 

# dust-correct source B centrals for each line 

# dust-correct source B red wings for each line 

# dust-correct source A red wings for all except [OII]

# dust-correct source B blue wings except [OII]

# ====================
# run while distinguishing outflows  
# ====================
# save the fluxes in the same storage format so that diagnostics_improved and emission_line_table_improved can access 