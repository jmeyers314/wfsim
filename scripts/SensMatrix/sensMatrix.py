import numpy as np
from wfsim import SSTFactory
import batoid
from tqdm import tqdm
import astropy.io.fits as fits


factory = SSTFactory(batoid.Optic.fromYaml("LSST_g_500.yaml"))

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

import yaml
with open("/Users/josh/src/ZEMAX_FEMAP/2senM/3senM/senM_35_19_50.yaml") as f:
    sensM = np.array(yaml.safe_load(f))
sensM2 = fits.getdata("sensM.fits")
import ipdb; ipdb.set_trace()

sensM_wfsim = np.empty_like(sensM)

def zkfn(telescope, fieldx, fieldy):
    return batoid.zernike(
        telescope,
        np.deg2rad(fieldx), np.deg2rad(fieldy),
        wavelength=500e-9,
        eps=0.61,
        nx=255
    )[4:]*0.5
    # return batoid.zernikeGQ(
    #     telescope,
    #     np.deg2rad(fieldx), np.deg2rad(fieldy),
    #     wavelength=500e-9,
    #     eps=0.61,
    # )[4:]*0.5

print("Computing fiducial zernikes")
zk0 = np.zeros((35, 19))
for ifield, (fieldx, fieldy) in enumerate(tqdm(fieldXY)):
    zk0[ifield] = zkfn(factory.fiducial, fieldx, fieldy)

ratio = np.zeros((35, 50))

print("Computing perturbed zernikes")
with tqdm(total=35*50) as pbar:
    for imode in range(50):
        dof = np.zeros(50)
        dof[imode] = 1.0
        telescope = factory.get_telescope(dof=dof)
        for ifield, (fieldx, fieldy) in enumerate(fieldXY):
            zk = zkfn(telescope, fieldx, fieldy) - zk0[ifield]
            sensM_wfsim[ifield, :, imode] = zk
            zktsph = sensM[ifield, :, imode]

            # print(f"{ifield = }    {imode = }")
            # for j in range(4, 23):
            #     print(f"{j:2d}  {zk[j-4]:10.6f}  {sensM[ifield, j-4, imode]:10.6f}")
            # print()

            rms = np.sqrt(np.sum(np.square(zktsph)))
            diff = zk - zktsph
            frac_diff = np.sqrt(np.sum(np.square(diff)))
            # print(f"{ifield = }    {imode = }  {frac_diff:.3f}")
            ratio[ifield, imode] = frac_diff
            pbar.update()

print(np.quantile(ratio.ravel(), [0.5, 0.95, 0.99, 1.0]))

import matplotlib.pyplot as plt
plt.imshow(ratio)
plt.colorbar()
plt.show()

plt.hist(ratio.ravel(), bins=np.linspace(0, 0.1, 100))
plt.show()
