import numpy as np
import galsim


def BBSED(T):
    """(unnormalized) Blackbody SED for temperature T in Kelvin.
    """
    waves_nm = np.arange(305.0, 1120.0, 10.0)
    def planck(t, w):
        # t in K
        # w in m
        c = 2.99792458e8  # speed of light in m/s
        kB = 1.3806488e-23  # Boltzmann's constant J per Kelvin
        h = 6.62607015e-34  # Planck's constant in J s
        return w**(-5) / (np.exp(h*c/(w*kB*t))-1)
    flambda = planck(T, waves_nm*1e-9)
    return galsim.SED(
        galsim.LookupTable(waves_nm, flambda),
        wave_type='nm',
        flux_type='flambda'
    )
