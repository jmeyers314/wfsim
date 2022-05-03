import matplotlib.pyplot as plt

import numpy as np
import galsim
import batoid
from tqdm import tqdm

import wfsim

rng = np.random.default_rng(5772156649015328606065120900824024310421)

bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
fiducial_telescope = batoid.Optic.fromYaml("LSST_r.yaml")
factory = wfsim.SSTFactory(fiducial_telescope)
pixel_scale = 10e-6

observation = {
    'zenith': 30 * galsim.degrees,
    'raw_seeing': 0.7 * galsim.arcsec,  # zenith 500nm seeing
    'wavelength': bandpass.effective_wavelength,
    'exptime': 15.0,  # seconds
    'temperature': 293.,  # Kelvin
    'pressure': 69.,  #kPa
    'H2O_pressure': 1.0  #kPa
}

atm_kwargs = {
    'screen_size': 819.2,
    'screen_scale': 0.1,
    'nproc': 6  # create screens in parallel using this many CPUs
}

dof = rng.normal(scale=0.05, size=50)
# but zero-out the hexafoil modes that aren't currently fit well.
dof[[28, 45, 46]] = 0
telescope = factory.get_telescope(dof=dof)

# We'll monkey patch in the correct telescope on-the-fly below.
intra_simulator = wfsim.SimpleSimulator(
    observation,
    atm_kwargs,
    telescope,
    bandpass,
    shape=(256, 256),
    rng=rng
)
extra_simulator = wfsim.SimpleSimulator(
    observation,
    atm_kwargs,
    telescope,
    bandpass,
    shape=(256, 256),
    rng=rng
)

star_T = rng.uniform(4000, 10000)
sed = wfsim.BBSED(star_T)
# flux = int(rng.uniform(1_000_000, 2_000_000))
flux = 1_000_000

for i, angle in enumerate(tqdm(np.linspace(0, 360, 40, endpoint=False))):
    rotated = telescope.withLocallyRotatedOptic("LSSTCamera", batoid.RotZ(np.deg2rad(angle)))
    intra = rotated.withGloballyShiftedOptic("Detector", [0, 0, -0.0015])
    extra = rotated.withGloballyShiftedOptic("Detector", [0, 0, +0.0015])
    intra_simulator.telescope = intra
    extra_simulator.telescope = extra

    intra_simulator.image.setZero()
    extra_simulator.image.setZero()

    intra_simulator.add_star(0.0, 0.0, sed, flux, rng)
    extra_simulator.add_star(0.0, 0.0, sed, flux, rng)

    intra_simulator.add_background(500.0, rng)
    extra_simulator.add_background(500.0, rng)

    intra_simulator.image.write(f"output/intra_{i:02d}.fits")
    extra_simulator.image.write(f"output/extra_{i:02d}.fits")

    # fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5))
    # axes[0].imshow(intra_simulator.image.array)
    # axes[1].imshow(extra_simulator.image.array)
    # plt.tight_layout()
    # plt.show()
