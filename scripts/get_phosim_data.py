import numpy as np
import os
from tqdm import tqdm
import astropy.io.fits as fits

def readzgrid(fn, scale):
    with open(fn) as f:
        splt = f.readline().split()
        nx = int(splt[0])
        ny = int(splt[1])
        dx = float(splt[2])*1e-3
        dy = float(splt[3])*1e-3
    x = np.arange(nx)*dx
    y = np.arange(ny)*dy
    x -= np.mean(x)
    y -= np.mean(y)

    z, dzdx, dzdy, d2zdxdy = np.loadtxt(fn, skiprows=1, unpack=True)
    z = np.flipud(z.reshape(204, 204)) * 1e-3 * scale
    dzdx = np.flipud(dzdx.reshape(204, 204)) * scale
    dzdy = np.flipud(dzdy.reshape(204, 204)) * scale
    d2zdxdy = np.flipud(d2zdxdy.reshape(204, 204)) * 1e3 * scale
    return x, y, z, dzdx, dzdy, d2zdxdy

phosim_data_dir = "/Users/josh/src/phosim/data/lsst/"

M1x = None
M1y = None
M1z = np.empty((20, 204, 204))
M1dzdx = np.empty((20, 204, 204))
M1dzdy = np.empty((20, 204, 204))
M1d2zdxdy = np.empty((20, 204, 204))

M2x = None
M2y = None
M2z = np.empty((20, 204, 204))
M2dzdx = np.empty((20, 204, 204))
M2dzdy = np.empty((20, 204, 204))
M2d2zdxdy = np.empty((20, 204, 204))

M3x = None
M3y = None
M3z = np.empty((20, 204, 204))
M3dzdx = np.empty((20, 204, 204))
M3dzdy = np.empty((20, 204, 204))
M3d2zdxdy = np.empty((20, 204, 204))

M1M3zk = np.zeros((20, 29))
M2zk = np.zeros((20, 29))


for i in tqdm(range(20)):
    m1fn = os.path.join(
        phosim_data_dir,
        f"M1_b{i+1}_0.50_grid.DAT"
    )
    M1x, M1y, M1z[i], M1dzdx[i], M1dzdy[i], M1d2zdxdy[i] = readzgrid(m1fn, -1 / 0.5)

    m2fn = os.path.join(
        phosim_data_dir,
        f"M2_b{i+1}_0.25_grid.DAT"
    )
    M2x, M2y, M2z[i], M2dzdx[i], M2dzdy[i], M2d2zdxdy[i] = readzgrid(m2fn, -1 / 0.25)

    m3fn = os.path.join(
        phosim_data_dir,
        f"M3_b{i+1}_0.50_grid.DAT"
    )
    M3x, M3y, M3z[i], M3dzdx[i], M3dzdy[i], M3d2zdxdy[i] = readzgrid(m3fn, -1 / 0.5)

    M1M3zk[i, 1:] = np.loadtxt(
        os.path.join(
            phosim_data_dir,
            f"M13_b{i+1}_0.50_gridz.txt"
        )
    ) * -1 * 1e-3 / 0.5

    M2zk[i, 1:] = np.loadtxt(
        os.path.join(
            phosim_data_dir,
            f"M2_b{i+1}_0.25_gridz.txt"
        )
    ) * -1 * 1e-3 / 0.25

fits.writeto("M1_bend_coords.fits", np.stack([M1x, M1y]))
fits.writeto("M2_bend_coords.fits", np.stack([M2x, M2y]))
fits.writeto("M3_bend_coords.fits", np.stack([M3x, M3y]))
fits.writeto("M1_bend_grid.fits", np.stack([M1z, M1dzdx, M1dzdy, M1d2zdxdy]))
fits.writeto("M2_bend_grid.fits", np.stack([M2z, M2dzdx, M2dzdy, M2d2zdxdy]))
fits.writeto("M3_bend_grid.fits", np.stack([M3z, M3dzdx, M3dzdy, M3d2zdxdy]))
fits.writeto("M13_bend_zk.fits", M1M3zk)
fits.writeto("M2_bend_zk.fits", M2zk)
