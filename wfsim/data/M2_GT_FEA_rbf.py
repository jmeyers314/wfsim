import numpy as np
import astropy.io.fits as fits
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt

# Following reproduces what's in ts_phosim, but it's a little
# weird how much extrapolation is being applied.  The input
# domain only covers about 95% of the radius of the output
# domain.

data = fits.getdata("M2_1um_grid.fits.gz")  # (15984, 75)
bx = -data[:, 1]  # meters
by = data[:, 2]

data = fits.getdata("M2_GT_FEA.fits.gz")

zdz = Rbf(data[:, 0], data[:, 1], data[:, 2])(bx/1.71, by/1.71)
hdz = Rbf(data[:, 0], data[:, 1], data[:, 3])(bx/1.71, by/1.71)
tzdz = Rbf(data[:, 0], data[:, 1], data[:, 4])(bx/1.71, by/1.71)
trdz = Rbf(data[:, 0], data[:, 1], data[:, 5])(bx/1.71, by/1.71)

# Now dump these into a new fits.gz file
fits.writeto(
    "M2_GT_grid.fits.gz",
    np.vstack([
        zdz, hdz, tzdz, trdz
    ])
)

# # Weird that input data doesn't cover output domain.
# plt.figure()
# plt.scatter(bx, by, c=zdz, s=3)
# plt.scatter(data[:, 0]*1.71, data[:, 1]*1.71, c='red', s=1)
# plt.show()
