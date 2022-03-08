import batoid
import numpy as np
from wfsim import SSTFactory
from tqdm import tqdm
import astropy.io.fits as fits


fieldXY = np.array([
    [0.0, 0.0],
    [0.379, 0.0],
    [0.18950000000000006, 0.3282236280343022],
    [-0.18949999999999992, 0.3282236280343023],
    [-0.379, 4.641411368768469e-17],
    [-0.18950000000000017, -0.32822362803430216],
    [0.18950000000000006, -0.3282236280343022],
    [0.841, 0.0],
    [0.4205000000000001, 0.7283273645827129],
    [-0.4204999999999998, 0.728327364582713],
    [-0.841, 1.029927958082924e-16],
    [-0.4205000000000004, -0.7283273645827126],
    [0.4205000000000001, -0.7283273645827129],
    [1.237, 0.0],
    [0.6185000000000002, 1.0712734244813507],
    [-0.6184999999999998, 1.0712734244813509],
    [-1.237, 1.5148880905452761e-16],
    [-0.6185000000000006, -1.0712734244813504],
    [0.6185000000000002, -1.0712734244813507],
    [1.535, 0.0],
    [0.7675000000000002, 1.3293489948091133],
    [-0.7674999999999996, 1.3293489948091133],
    [-1.535, 1.879832836691187e-16],
    [-0.7675000000000006, -1.3293489948091128],
    [0.7675000000000002, -1.3293489948091133],
    [1.708, 0.0],
    [0.8540000000000002, 1.479171389663821],
    [-0.8539999999999996, 1.4791713896638212],
    [-1.708, 2.0916967329436793e-16],
    [-0.8540000000000008, -1.4791713896638208],
    [0.8540000000000002, -1.479171389663821],
    [ 1.176,  1.176],
    [-1.176,  1.176],
    [-1.176, -1.176],
    [ 1.176, -1.176],
])

factory = SSTFactory(batoid.Optic.fromYaml("LSST_g_500.yaml"))

# Do fiducial telescope first.
for ifield, (fieldX, fieldY) in enumerate(tqdm(fieldXY)):
    opd = batoid.wavefront(
        factory.fiducial,
        np.deg2rad(fieldX), np.deg2rad(fieldY),
        wavelength=500e-9, nx=255
    ).array*0.5
    opddata = opd.data.astype(np.float32)
    opddata[opd.mask] = 0.0
    fits.writeto(
        f"opd/opd_nominal_field_{ifield}.fits.gz",
        opddata,
        overwrite=True
    )

# Now do modes
with tqdm(total=50*35) as pbar:
    for imode in range(50):
        dof = np.zeros(50)
        dof[imode] = 1.0
        perturbed = factory.get_telescope(dof=dof)
        for ifield, (fieldX, fieldY) in enumerate(fieldXY):
            opd = batoid.wavefront(
                perturbed,
                np.deg2rad(fieldX), np.deg2rad(fieldY),
                wavelength=500e-9, nx=255
            ).array*0.5
            opddata = opd.data.astype(np.float32)
            opddata[opd.mask] = 0.0
            fits.writeto(
                f"opd/opd_mode_{imode}_field_{ifield}.fits.gz",
                opddata,
                overwrite=True
            )
            pbar.update()
