import numpy as np
import tqdm

import galsim
import batoid

import wfsim


# Making something completely up for now.
observation = {
    'zenith': 30*galsim.degrees,
    'raw_seeing': 0.7*galsim.arcsec,
    'wavelength': 725.0,  # nm
    'exptime': 20.0,
    'temperature': 293.15,  # K
    'pressure': 69.328,  # kPa
    'H2O_pressure': 1.067,  # kPa
}

atm_kwargs = {
    'kcrit': 0.2,
    'screen_size': 409.6,
    # 'screen_size': 51.2,
    'screen_scale': 0.1,
    'nproc': 6,
}

rng = np.random.default_rng(57721)
bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
pixel_scale = 10e-6
telescope = batoid.Optic.fromYaml("AuxTel.yaml")
telescope = telescope.withGloballyShiftedOptic("M2", [0, 0, 0.0008])
# telescope = telescope.withGloballyShiftedOptic("M2", [0.001, 0, 0])
telescope = telescope.withLocallyRotatedOptic(
    "M2",
    batoid.RotX(np.deg2rad(3/60.))  # 1 arcmin tilt
)

star_simulator = wfsim.SimpleSimulator(
    observation, atm_kwargs, telescope,
    bandpass=bandpass,
    pixel_scale=pixel_scale,
    rng=rng,
    debug=False
)

nstar = 30
fluxes = [int(n) for n in 10**rng.uniform(5, 6, size=nstar)]

with tqdm.tqdm(total=sum(fluxes)) as pbar:
    for nphoton in fluxes:
        rho = np.sqrt(rng.uniform(0, np.deg2rad(4.7/60)**2))
        th = rng.uniform(0, 2*np.pi)
        thx, thy = rho*np.cos(th), rho*np.sin(th)
        T = rng.uniform(4000, 10000)
        sed = wfsim.BBSED(T)
        star_simulator.add_star(thx, thy, sed, nphoton, rng)
        pbar.update(nphoton)

# star_simulator.add_background(rng, 300)

import matplotlib.pyplot as plt
plt.imshow(star_simulator.image.array)
plt.show()
