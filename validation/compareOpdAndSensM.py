# Compare official sensitivity matrix to OPDs

import os
import numpy as np
import astropy.io.fits as fits
import galsim
from tqdm import tqdm

# import yaml
# with open("/Users/josh/src/ZEMAX_FEMAP/2senM/3senM/senM_35_19_50.yaml") as f:
#     sensM = np.array(yaml.safe_load(f))
sensM = fits.getdata("sensM.fits")

sensMOpd = np.empty_like(sensM)

print("Reading fiducial zernikes")
zk0 = np.zeros((35, 19))
ratio = np.zeros((35, 50))

xs = np.linspace(-1, 1, 255)
xs, ys = np.meshgrid(xs, xs)
xs -= np.mean(xs)
ys -= np.mean(ys)

# opd_dir = "ts_phosim_opds/opd/"
# opd_dir = "wfsim_opds/opd/"
opd_dir = "wfsim_simple_opds/opd/"

for ifield in range(35):
    fn = os.path.join(
        opd_dir,
        f"opd_nominal_field_{ifield}.fits.gz"
    )
    opd = fits.getdata(fn)
    w = (opd != 0)
    basis = galsim.zernike.zernikeBasis(22, xs[w], ys[w], R_inner=0.61)
    coefs, *_ = np.linalg.lstsq(basis.T, opd[w], rcond=None)
    zk0[ifield] = coefs[4:]

print("Reading perturbed zernikes")
with tqdm(total=35*50) as pbar:
    for imode in range(50):
    # for imode in range(10, 50):
        for ifield in range(35):
        # for ifield in range(0, 1):
            fn = os.path.join(
                opd_dir,
                f"opd_mode_{imode}_field_{ifield}.fits.gz"
            )
            opd = fits.getdata(fn)
            w = (opd != 0)
            nnan = np.sum(~np.isfinite(opd[w]))
            if nnan != 0:
                print("nnan = ", nnan)
                import matplotlib.pyplot as plt
                plt.imshow(opd)
                plt.show()
                opd[~np.isfinite(opd)] = 0.0
                w = (opd != 0)
            basis = galsim.zernike.zernikeBasis(22, xs[w], ys[w], R_inner=0.61)
            zk, *_ = np.linalg.lstsq(basis.T, opd[w], rcond=None)
            zk = zk[4:]
            dzk = zk - zk0[ifield]

            rssr = np.sqrt(np.sum(np.square(dzk-sensM[ifield,:,imode])))
            rss = np.sqrt(np.sum(np.square(sensM[ifield,:,imode])))
            ratio[ifield, imode] = rssr/rss
            pbar.update()

            # import matplotlib.pyplot as plt
            # plt.imshow(opd)
            # plt.colorbar()
            # plt.show()

            # print(f"{ifield = }    {imode = }")
            # for j in range(4, 23):
            #     print(f"{j:2d}  {dzk[j-4]:10.6f}  {sensM[ifield, j-4, imode]:10.6f}")
            # print(f"{ifield = }    {imode = }  {rssr/rss:.3f}")
            # print()

print(np.quantile(ratio.ravel(), [0.5, 0.95, 0.99, 1.0]))

import matplotlib.pyplot as plt
plt.imshow(ratio, vmin=0.0, vmax=0.08)
plt.colorbar()
plt.show()

plt.hist(ratio.ravel(), bins=np.linspace(0, 0.1, 100))
plt.show()
