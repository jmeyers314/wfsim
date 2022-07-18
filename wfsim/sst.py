from copy import copy
import galsim
import os
import numpy as np
import astropy.io.fits as fits
import batoid
import functools
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial import Delaunay


class rbf:
    def __init__(self, xy, z):
        self._interp = RBFInterpolator(xy, z)
    def __call__(self, x, y):
        shape = x.shape
        out = self._interp(np.vstack([x.ravel(), y.ravel()]).T)
        out.shape = shape
        return out


@functools.lru_cache
def _fits_cache(fn):
    """Cache loading fits file data table

    Parameters
    ----------
    fn : string
        File name from datadir to load and cache

    Returns
    -------
    out : ndarray
        Loaded data.
    """
    from . import datadir
    return fits.getdata(
        os.path.join(
            datadir,
            fn
        )
    )

def _node_to_grid(nodex, nodey, nodez, grid_coords):
    """Convert FEA nodes positions into grid of z displacements,
    first derivatives, and mixed 2nd derivative.

    Parameters
    ----------
    nodex, nodey, nodez : ndarray (M, )
        Positions of nodes
    grid_coords : ndarray (2, N)
        Output grid positions in x and y

    Returns
    -------
    grid : ndarray (4, N, N)
        1st slice is interpolated z-position.
        2nd slice is interpolated dz/dx
        3rd slice is interpolated dz/dy
        4th slice is interpolated d2z/dxdy
    """
    # Fast
    interp = CloughTocher2DInterpolator(
        np.array([nodex, nodey]).T,
        nodez,
        fill_value=0.0
    )

    # # Slow
    # interp = rbf(
    #     np.array([nodex, nodey]).T,
    #     nodez,
    # )

    x, y = grid_coords
    nx = len(x)
    ny = len(y)
    out = np.zeros([4, ny, nx])
    # Approximate derivatives with finite differences.  Make the finite
    # difference spacing equal to 1/10th the grid spacing.
    dx = np.mean(np.diff(x))*1e-1
    dy = np.mean(np.diff(y))*1e-1
    x, y = np.meshgrid(x, y)
    out[0] = interp(x, y)
    out[1] = (interp(x+dx, y) - interp(x-dx, y))/(2*dx)
    out[2] = (interp(x, y+dy) - interp(x, y-dy))/(2*dy)
    out[3] = (
        interp(x+dx, y+dy) -
        interp(x-dx, y+dy) -
        interp(x+dx, y-dy) +
        interp(x-dx, y-dy)
    )/(4*dx*dy)

    # Zero out the central hole
    r = np.hypot(x, y)
    rmin = np.min(np.hypot(nodex, nodey))
    w = r < rmin
    out[:, w] = 0.0

    return out


class SSTFactory:
    def __init__(self, fiducial):
        """Create a Simony Survey Telescope factory.

        Parameters
        ----------
        fiducial : batoid.Optic
            Optic before finite-element analysis (FEA) or active optics system
            (AOS) perturbations are applied.
        """
        self.fiducial = fiducial

    @functools.cached_property
    def m1m3_fea_coords(self):
        """Load FEA grid for M1M3.

        Items are:
        [0] : x-coordinates in meters
        [1] : y-coordinates in meters
        [2] : Boolean indicating points are part of M1
        [3] : Boolean indicating points are part of M3 (complement of [2])
        """
        data = _fits_cache("M1M3_1um_156_grid.fits.gz")
        idx = data[:, 0]
        bx = data[:, 1]  # (5256,)
        by = data[:, 2]
        idx1 = (idx == 1)
        idx3 = (idx == 3)
        return bx, by, idx1, idx3

    @functools.cached_property
    def m2_fea_coords(self):
        """Load FEA grid for M2.

        Items are:
        [0] : x-coordinates in meters
        [1] : y-coordinates in meters
        """
        data = _fits_cache("M2_1um_grid.fits.gz")  # (15984, 75)
        bx = -data[:, 1]  # meters
        by = data[:, 2]
        return bx, by

    @functools.cached_property
    def m1_grid_coords(self):
        """Coordinates to use for M1 grid perturbations.

        Items are:
        [0] : x-coordinates in meters
        [1] : y-coordinates in meters
        """
        data = _fits_cache("M1_bend_coords.fits.gz")
        return data

    @functools.cached_property
    def m2_grid_coords(self):
        """Coordinates to use for M2 grid perturbations.

        Items are:
        [0] : x-coordinates in meters
        [1] : y-coordinates in meters
        """
        data = _fits_cache("M2_bend_coords.fits.gz")
        return data

    @functools.cached_property
    def m3_grid_coords(self):
        """Coordinates to use for M3 grid perturbations.

        Items are:
        [0] : x-coordinates in meters
        [1] : y-coordinates in meters
        """
        data = _fits_cache("M3_bend_coords.fits.gz")
        return data

    def _m1m3_gravity(self, zenith_angle):
        zdata = _fits_cache("M1M3_dxdydz_zenith.fits.gz")
        hdata = _fits_cache("M1M3_dxdydz_horizon.fits.gz")
        dxyz = (
            zdata * np.cos(zenith_angle) +
            hdata * np.sin(zenith_angle)
        )
        dz = dxyz[:,2]

        # Interpolate these node displacements into z-displacements at
        # original node x/y positions.
        bx, by, idx1, idx3 = self.m1m3_fea_coords

        # M1
        zRef = self.fiducial['M1'].surface.sag(bx[idx1], by[idx1])
        zpRef = self.fiducial['M1'].surface.sag(
            (bx+dxyz[:, 0])[idx1],
            (by+dxyz[:, 1])[idx1]
        )
        dz[idx1] += zRef - zpRef

        # M3
        zRef = self.fiducial['M3'].surface.sag(bx[idx3], by[idx3])
        zpRef = self.fiducial['M3'].surface.sag(
            (bx+dxyz[:, 0])[idx3],
            (by+dxyz[:, 1])[idx3]
        )
        dz[idx3] += zRef - zpRef

        # Subtract PTT
        # This kinda makes sense for M1, but why for combined M1M3?
        zBasis = galsim.zernike.zernikeBasis(
            3, bx, by, R_outer=4.18, R_inner=2.558
        )
        coefs, _, _, _ = np.linalg.lstsq(zBasis.T, dxyz[:, 2], rcond=None)
        zern = galsim.zernike.Zernike(coefs, R_outer=4.18, R_inner=2.558)
        dz -= zern(bx, by)

        return dz

    def _m1m3_temperature(
        self, m1m3TBulk, m1m3TxGrad, m1m3TyGrad, m1m3TzGrad, m1m3TrGrad,
    ):
        if m1m3TxGrad is None:
            m1m3TxGrad = 0.0
        bx, by, idx1, idx3 = self.m1m3_fea_coords
        normX = bx / 4.18
        normY = by / 4.18

        data = _fits_cache("M1M3_thermal_FEA.fits.gz")
        delaunay = Delaunay(data[:, 0:2])
        tbdz = CloughTocher2DInterpolator(delaunay, data[:, 2])(normX, normY)
        txdz = CloughTocher2DInterpolator(delaunay, data[:, 3])(normX, normY)
        tydz = CloughTocher2DInterpolator(delaunay, data[:, 4])(normX, normY)
        tzdz = CloughTocher2DInterpolator(delaunay, data[:, 5])(normX, normY)
        trdz = CloughTocher2DInterpolator(delaunay, data[:, 6])(normX, normY)

        out = m1m3TBulk * tbdz
        out += m1m3TxGrad * txdz
        out += m1m3TyGrad * tydz
        out += m1m3TzGrad * tzdz
        out += m1m3TrGrad * trdz
        out *= 1e-6
        return out

    def _m2_gravity(self, zenith_angle):
        # This reproduces ts_phosim with preCompElevInRadian=0, but what is
        # that?  Also, I have questions regarding the input domain of the Rbf
        # interpolation...
        bx, by = self.m2_fea_coords

        # data = _fits_cache("M2_GT_FEA.fits.gz")
        # from scipy.interpolate import Rbf
        # zdz = Rbf(data[:, 0], data[:, 1], data[:, 2])(bx/1.71, by/1.71)
        # hdz = Rbf(data[:, 0], data[:, 1], data[:, 3])(bx/1.71, by/1.71)

        # Faster to precompute the above only once
        data = _fits_cache("M2_GT_grid.fits.gz")
        zdz = data[0]
        hdz = data[1]

        out = zdz * (np.cos(zenith_angle) - 1)
        out += hdz * np.sin(zenith_angle)
        out *= 1e-6
        return out

    def _m2_temperature(self, m2TzGrad, m2TrGrad):
        # Same domain problem here as m2_gravity...
        bx, by = self.m2_fea_coords
        data = _fits_cache("M2_GT_FEA.fits.gz")

        # from scipy.interpolate import Rbf
        # tzdz = Rbf(data[:, 0], data[:, 1], data[:, 4])(bx/1.71, by/1.71)
        # trdz = Rbf(data[:, 0], data[:, 1], data[:, 5])(bx/1.71, by/1.71)

        # Faster to precompute the above only once
        data = _fits_cache("M2_GT_grid.fits.gz")
        tzdz = data[2]
        trdz = data[3]

        out = m2TzGrad * tzdz
        out += m2TrGrad * trdz
        out *= 1e-6
        return out

    # # This is Josh's preferred interpolator, but fails b/c domain issues.
    # def _m2_gravity(self, zenith_angle):
    #     bx, by = self.m2_fea_coords
    #     data = _fits_cache("M2_GT_FEA.fits.gz")
    #     # Hack to get interpolation points inside Convex Hull of input
    #     delaunay = Delaunay(data[:, 0:2]/0.95069)
    #     zdz = CloughTocher2DInterpolator(delaunay, data[:, 2])(bx/1.71, by/1.71)
    #     hdz = CloughTocher2DInterpolator(delaunay, data[:, 3])(bx/1.71, by/1.71)
    #     out = zdz * (np.cos(zenith_angle) - 1)
    #     out += hdz * np.sin(zenith_angle)
    #     out *= 1e-6  # micron -> meters
    #     return out

    # def _m2_temperature(self, m2TzGrad, m2TrGrad):
    #     # Same domain problem here as m2_gravity...
    #     bx, by = self.m2_fea_coords
    #     normX = bx / 1.71
    #     normY = by / 1.71
    #     data = _fits_cache("M2_GT_FEA.fits.gz")

    #     # Hack to get interpolation points inside Convex Hull of input
    #     delaunay = Delaunay(data[:, 0:2]/0.95069)
    #     tzdz = CloughTocher2DInterpolator(delaunay, data[:, 4])(normX, normY)
    #     trdz = CloughTocher2DInterpolator(delaunay, data[:, 5])(normX, normY)

    #     out = m2TzGrad * tzdz
    #     out += m2TrGrad * trdz
    #     out *= 1e-6
    #     return out

    def get_telescope(
        self,
        zenith_angle=None,    # radians
        rotation_angle=None,  # radians
        m1m3TBulk=0.0,        # 2-sigma spans +/- 0.8C
        m1m3TxGrad=0.0,       # 2-sigma spans +/- 0.4C
        m1m3TyGrad=0.0,       # 2-sigma spans +/- 0.4C
        m1m3TzGrad=0.0,       # 2-sigma spans +/- 0.1C
        m1m3TrGrad=0.0,       # 2-sigma spans +/- 0.1C
        m2TzGrad=0.0,
        m2TrGrad=0.0,
        camTB=None,
        dof=None,
        doM1M3Pert=False,
        doM2Pert=False,
        doCamPert=False,
        _omit_dof_grid=False,
        _omit_dof_zk=False,
    ):
        optic = self.fiducial

        if dof is None:
            dof = np.zeros(50)

        # order is z, dzdx, dzdy, d2zdxdy
        # These can get set either through grav/temp perturbations or through
        # dof
        m1_grid = np.zeros((4, 204, 204))
        m3_grid = np.zeros((4, 204, 204))
        m1m3_zk = np.zeros(29)

        if doM1M3Pert:
            # hard code for now
            # indices are over FEA nodes
            m1m3_fea_dz = np.zeros(5256)
            if zenith_angle is not None:
                m1m3_fea_dz = self._m1m3_gravity(zenith_angle)

            if any([m1m3TBulk, m1m3TxGrad, m1m3TyGrad, m1m3TzGrad, m1m3TrGrad]):
                m1m3_fea_dz += self._m1m3_temperature(
                    m1m3TBulk, m1m3TxGrad, m1m3TyGrad, m1m3TzGrad, m1m3TrGrad
                )

            if np.any(m1m3_fea_dz):
                bx, by, idx1, idx3 = self.m1m3_fea_coords
                zBasis = galsim.zernike.zernikeBasis(
                    28, -bx, by, R_outer=4.18
                )
                m1m3_zk, *_ = np.linalg.lstsq(zBasis.T, m1m3_fea_dz, rcond=None)
                zern = galsim.zernike.Zernike(m1m3_zk, R_outer=4.18)
                m1m3_fea_dz -= zern(-bx, by)

                m1_grid = _node_to_grid(
                    bx[idx1], by[idx1], m1m3_fea_dz[idx1], self.m1_grid_coords
                )

                m3_grid = _node_to_grid(
                    bx[idx3], by[idx3], m1m3_fea_dz[idx3], self.m3_grid_coords
                )
                m1_grid *= -1
                m3_grid *= -1
                m1m3_zk *= -1

        # M1M3 bending modes
        if np.any(dof[10:30] != 0):
            if not _omit_dof_grid:
                m1_bend = _fits_cache("M1_bend_grid.fits.gz")
                m3_bend = _fits_cache("M3_bend_grid.fits.gz")
                m1_grid += np.tensordot(m1_bend, dof[10:30], axes=[[1], [0]])
                m3_grid += np.tensordot(m3_bend, dof[10:30], axes=[[1], [0]])

            if not _omit_dof_zk:
                m1m3_zk += np.dot(dof[10:30], _fits_cache("M13_bend_zk.fits.gz"))

        if np.any([m1m3_zk]) or np.any(m1_grid):
            optic = optic.withSurface(
                'M1',
                batoid.Sum([
                    optic['M1'].surface,
                    batoid.Zernike(m1m3_zk, R_outer=4.18),
                    batoid.Bicubic(*self.m1_grid_coords, *m1_grid)
                ])
            )
        if np.any([m1m3_zk]) or np.any(m3_grid):
            optic = optic.withSurface(
                'M3',
                batoid.Sum([
                    optic['M3'].surface,
                    batoid.Zernike(m1m3_zk, R_outer=4.18),
                    batoid.Bicubic(*self.m3_grid_coords, *m3_grid)
                ])
            )

        m2_grid = np.zeros((4, 204, 204))
        m2_zk = np.zeros(29)

        if doM2Pert:
            # hard code for now
            # indices are over FEA nodes
            m2_fea_dz = np.zeros(15984)
            if zenith_angle is not None:
                m2_fea_dz = self._m2_gravity(zenith_angle)

            if any([m2TzGrad, m2TrGrad]):
                m2_fea_dz += self._m2_temperature(
                    m2TzGrad, m2TrGrad
                )

            if np.any(m2_fea_dz):
                bx, by = self.m2_fea_coords
                zBasis = galsim.zernike.zernikeBasis(
                    28, -bx, by, R_outer=1.71
                )
                m2_zk, *_ = np.linalg.lstsq(zBasis.T, m2_fea_dz, rcond=None)
                zern = galsim.zernike.Zernike(m2_zk, R_outer=1.71)
                m2_fea_dz -= zern(-bx, by)

                m2_grid = _node_to_grid(
                    bx, by, m2_fea_dz, self.m2_grid_coords
                )

                m2_grid *= -1
                m2_zk *= -1

        if np.any(dof[30:50] != 0):
            if not _omit_dof_grid:
                m2_bend = _fits_cache("M2_bend_grid.fits.gz")
                m2_grid += np.tensordot(m2_bend, dof[30:50], axes=[[1], [0]])

            if not _omit_dof_zk:
                m2_zk += np.dot(dof[30:50], _fits_cache("M2_bend_zk.fits.gz"))

        if np.any([m2_zk]) or np.any(m2_grid):
            optic = optic.withSurface(
                'M2',
                batoid.Sum([
                    optic['M2'].surface,
                    batoid.Zernike(m2_zk, R_outer=1.71),
                    batoid.Bicubic(*self.m2_grid_coords, *m2_grid)
                ])
            )

        if np.any(dof[0:3] != 0):
            optic = optic.withGloballyShiftedOptic(
                "M2",
                np.array([dof[1], dof[2], -dof[0]])*1e-6
            )
        if np.any(dof[3:5] != 0):
            rx = batoid.RotX(np.deg2rad(-dof[3]/3600))
            ry = batoid.RotY(np.deg2rad(-dof[4]/3600))
            optic = optic.withLocallyRotatedOptic(
                "M2",
                rx @ ry
            )
        if np.any(dof[5:8] != 0):
            optic = optic.withGloballyShiftedOptic(
                "LSSTCamera",
                np.array([dof[6], dof[7], -dof[5]])*1e-6
            )
        if np.any(dof[8:10] != 0):
            rx = batoid.RotX(np.deg2rad(-dof[8]/3600))
            ry = batoid.RotY(np.deg2rad(-dof[9]/3600))
            optic = optic.withLocallyRotatedOptic(
                "LSSTCamera",
                rx @ ry
            )

        if doCamPert:
            cam_data = [
                ('L1S1', 'L1_entrance', 0.775),
                ('L1S2', 'L1_exit', 0.775),
                ('L2S1', 'L2_entrance', 0.551),
                ('L2S2', 'L2_exit', 0.551),
                ('L3S1', 'L3_entrance', 0.361),
                ('L3S2', 'L3_exit', 0.361),
            ]
            for tname, bname, radius in cam_data:
                data = _fits_cache(tname+"zer.fits.gz")
                grav_zk = data[0, 3:] * (np.cos(zenith_angle) - 1)
                grav_zk += (
                    data[1, 3:] * np.cos(rotation_angle) +
                    data[2, 3:] * np.sin(rotation_angle)
                ) * np.sin(zenith_angle)
                # subtract pre-compensated grav...
                TB = np.clip(camTB, data[3, 2], data[10, 2])
                fidx = np.interp(camTB, data[3:, 2], np.arange(len(data[3:, 2])))+3
                idx = int(np.floor(fidx))
                whi = fidx - idx
                wlo = 1 - whi
                temp_zk = wlo * data[idx, 3:] + whi * data[idx+1, 3:]

                # subtract reference temperature zk (0 deg C is idx=5)
                temp_zk -= data[5, 3:]

                surf_zk = grav_zk + temp_zk

                # remap Andy -> Noll Zernike indices
                zIdxMapping = [
                    1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
                    17, 21, 16, 25, 24, 26, 23, 27, 22, 28
                ]
                surf_zk = surf_zk[[x - 1 for x in zIdxMapping]]
                surf_zk *= -1e-3  # mm -> m
                # tsph -> batoid 0-index offset
                surf_zk = np.concatenate([[0], surf_zk])

                optic = optic.withSurface(
                    bname,
                    batoid.Sum([
                        optic[bname].surface,
                        batoid.Zernike(-surf_zk, R_outer=radius)
                    ])
                )

        return optic

        # TODO:
        #  - M1M3 force error...
        #  - actuator forces


# Sentinel to mark when intermediate array needs to be updated
# Can't use None, since that means ignore this input instead.
class _Invalidated:
    pass

class SSTBuilder:
    # Use some class variables to store things that can be shared amongst
    # instances.
    _m1m3_fea_x = None
    _m1m3_fea_y = None
    _m1m3_fea_idx1 = None
    _m1m3_fea_idx3 = None
    _m1_grid_x = None
    _m1_grid_y = None
    _m3_grid_x = None
    _m3_grid_y = None

    _m2_fea_x = None
    _m2_fea_y = None
    _m2_grid_x = None
    _m2_grid_y = None

    # Some class methods to load them
    def _load_m1m3(cls):
        # FEA nodes
        data = _fits_cache("M1M3_1um_156_grid.fits.gz")
        idx = data[:, 0]
        bx = data[:, 1]  # (5256,)
        by = data[:, 2]
        idx1 = (idx == 1)
        idx3 = (idx == 3)
        cls._m1m3_fea_x = bx
        cls._m1m3_fea_y = by
        cls._m1m3_fea_idx1 = idx1
        cls._m1m3_fea_idx3 = idx3
        # Grid coords
        data = _fits_cache("M1_bend_coords.fits.gz")
        cls._m1_grid_x = data[0]
        cls._m1_grid_y = data[1]
        data = _fits_cache("M3_bend_coords.fits.gz")
        cls._m3_grid_x = data[0]
        cls._m3_grid_y = data[1]

    @property
    def m1m3_fea_x(self):
        if self._m1m3_fea_x is None:
            self._load_m1m3()
        return self._m1m3_fea_x

    @property
    def m1m3_fea_y(self):
        if self._m1m3_fea_y is None:
            self._load_m1m3()
        return self._m1m3_fea_y

    @property
    def m1m3_fea_idx1(self):
        if self._m1m3_fea_idx1 is None:
            self._load_m1m3()
        return self._m1m3_fea_idx1

    @property
    def m1m3_fea_idx3(self):
        if self._m1m3_fea_idx3 is None:
            self._load_m1m3()
        return self._m1m3_fea_idx3

    @property
    def m1_grid_x(self):
        if self._m1_grid_x is None:
            self._load_m1m3()
        return self._m1_grid_x

    @property
    def m1_grid_y(self):
        if self._m1_grid_y is None:
            self._load_m1m3()
        return self._m1_grid_y

    @property
    def m3_grid_x(self):
        if self._m3_grid_x is None:
            self._load_m1m3()
        return self._m3_grid_x

    @property
    def m3_grid_y(self):
        if self._m3_grid_y is None:
            self._load_m1m3()
        return self._m3_grid_y

    def _load_m2(cls):
        # FEA nodes
        data = _fits_cache("M2_1um_grid.fits.gz")  # (15984, 75)
        bx = -data[:, 1]  # meters
        by = data[:, 2]
        cls._m2_fea_x = bx
        cls._m2_fea_y = by

        # Grid coords
        data = _fits_cache("M2_bend_coords.fits.gz")
        cls._m2_grid_x = data[0]
        cls._m2_grid_y = data[1]

    @property
    def m2_fea_x(self):
        if self._m2_fea_x is None:
            self._load_m2()
        return self._m2_fea_x

    @property
    def m2_fea_y(self):
        if self._m2_fea_y is None:
            self._load_m2()
        return self._m2_fea_y

    @property
    def m2_grid_x(self):
        if self._m2_grid_x is None:
            self._load_m2()
        return self._m2_grid_x

    @property
    def m2_grid_y(self):
        if self._m2_grid_y is None:
            self._load_m2()
        return self._m2_grid_y

    def __init__(self, fiducial):
        """Create a Simony Survey Telescope factory.

        Parameters
        ----------
        fiducial : batoid.Optic
            Optic before finite-element analysis (FEA) or active optics system
            (AOS) perturbations are applied.
        """
        self.fiducial = fiducial

        # "Input" variables.

        # Awkward, but it's possible to set different zenith angles for
        # different subsystems.  Useful for testing one subsystem at a time
        # though.
        self.m1m3_zenith_angle = None
        self.m1m3_TBulk = 0.0
        self.m1m3_TxGrad = 0.0
        self.m1m3_TyGrad = 0.0
        self.m1m3_TzGrad = 0.0
        self.m1m3_TrGrad = 0.0

        self.m2_zenith_angle = None
        self.m2_TzGrad = None
        self.m2_TrGrad = None

        self.camera_zenith_angle = None
        self.camera_rotation_angle = None
        self.camera_TBulk = None

        self.dof = np.zeros(50)

        # Intermediate results caches.  Be careful to invalidate (by setting to
        # None) as required by any of the dependent inputs being changed.

        # Direct dependents of inputs
        # self._m1m3_fea_random = None
        self._m1m3_fea_gravity = None
        self._m1m3_fea_temperature = None
        self._m1_bend_grid = None
        self._m3_bend_grid = None
        self._m1m3_bend_zk = None

        # FEA intermediate dependencies
        self._m1_fea_grid = None
        self._m3_fea_grid = None
        self._m1m3_fea_zk = None

        # Final results we're trying to populate
        self.m1_grid = None
        self.m3_grid = None
        self.m1m3_zk = None

        # Similar for M2
        self._m2_fea_gravity = None
        self._m2_fea_temperature = None
        self._m2_bend_grid = None
        self._m2_bend_zk = None

        self._m2_fea_grid = None
        self._m2_fea_zk = None

        self.m2_grid = None
        self.m2_zk = None

        # Similar for Camera
        self._camera_gravity_zk = None
        self._camera_temperature_zk = None

        self.camera_zk = None

    def __copy__(self):
        ret = self.__class__.__new__(self.__class__)
        ret.__dict__ = self.__dict__.copy()
        return ret

    def with_m1m3_gravity(self, zenith_angle):
        """Return new SSTBuilder that includes gravitational flexure of M1M3.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 gravitation flexure applied.
        """
        ret = copy(self)
        ret.m1m3_zenith_angle = zenith_angle
        # Invalidate dependents
        ret._m1m3_fea_gravity = _Invalidated
        ret._m1_fea_grid = _Invalidated
        ret._m3_fea_grid = _Invalidated
        ret._m1m3_fea_zk = _Invalidated
        ret.m1_grid = _Invalidated
        ret.m3_grid = _Invalidated
        ret.m1m3_zk = _Invalidated
        return ret

    def with_m1m3_temperature(
        self,
        m1m3_TBulk,
        m1m3_TxGrad=0.0,
        m1m3_TyGrad=0.0,
        m1m3_TzGrad=0.0,
        m1m3_TrGrad=0.0,
    ):
        """Return new SSTBuilder that includes temperature flexure of M1M3.

        Parameters
        ----------
        m1m3_TBulk : float
            Bulk temperature in C.
        m1m3_TxGrad : float, optional
            Temperature gradient in x direction in C / m (?)
        m1m3_TyGrad : float, optional
            Temperature gradient in y direction in C / m (?)
        m1m3_TzGrad : float, optional
            Temperature gradient in z direction in C / m (?)
        m1m3_TrGrad : float, optional
            Temperature gradient in r direction in C / m (?)

        Returns
        -------
        ret : SSTBuilder
            New builder with M1M3 temperature flexure applied.
        """
        ret = copy(self)
        ret.m1m3_TBulk = m1m3_TBulk
        ret.m1m3_TxGrad = m1m3_TxGrad
        ret.m1m3_TyGrad = m1m3_TyGrad
        ret.m1m3_TzGrad = m1m3_TzGrad
        ret.m1m3_TrGrad = m1m3_TrGrad
        # Invalidate dependents
        ret._m1m3_fea_temperature = _Invalidated
        ret._m1_fea_grid = _Invalidated
        ret._m3_fea_grid = _Invalidated
        ret._m1m3_fea_zk = _Invalidated
        ret.m1_grid = _Invalidated
        ret.m3_grid = _Invalidated
        ret.m1m3_zk = _Invalidated
        return ret

    def with_m2_gravity(self, zenith_angle):
        """Return new SSTBuilder that includes gravitational flexure of M2.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with M2 gravitation flexure applied.
        """
        ret = copy(self)
        ret.m2_zenith_angle = zenith_angle
        # Invalidate dependents
        ret._m2_fea_gravity = _Invalidated
        ret._m2_fea_grid = _Invalidated
        ret._m2_fea_zk = _Invalidated
        ret.m2_grid = _Invalidated
        ret.m2_zk = _Invalidated
        return ret

    def with_m2_temperature(
        self,
        m2_TzGrad=0.0,
        m2_TrGrad=0.0,
    ):
        """Return new SSTBuilder that includes temperature flexure of M2.

        Parameters
        ----------
        m2_TzGrad : float, optional
            Temperature gradient in z direction in C / m (?)
        m2_TrGrad : float, optional
            Temperature gradient in r direction in C / m (?)

        Returns
        -------
        ret : SSTBuilder
            New builder with M2 temperature flexure applied.
        """
        ret = copy(self)
        ret.m2_TzGrad = m2_TzGrad
        ret.m2_TrGrad = m2_TrGrad
        # Invalidate dependents
        ret._m2_fea_temperature = _Invalidated
        ret._m2_fea_grid = _Invalidated
        ret._m2_fea_zk = _Invalidated
        ret.m2_grid = _Invalidated
        ret.m2_zk = _Invalidated
        return ret

    def with_camera_gravity(self, zenith_angle, rotation_angle):
        """Return new SSTBuilder that includes gravitational flexure of camera.

        Parameters
        ----------
        zenith_angle : float
            Zenith angle in radians
        rotation_angle : float
            Rotation angle in radians

        Returns
        -------
        ret : SSTBuilder
            New builder with camera gravitation flexure applied.
        """
        ret = copy(self)
        ret.camera_zenith_angle = zenith_angle
        ret.camera_rotation_angle = rotation_angle
        ret._camera_gravity_zk = _Invalidated
        ret.camera_zk = _Invalidated
        return ret

    def with_camera_temperature(self, camera_TBulk):
        """Return new SSTBuilder that includes temperature flexure of camera.

        Parameters
        ----------
        camera_TBulk : float
            Camera temperature in C

        Returns
        -------
        ret : SSTBuilder
            New builder with camera temperature flexure applied.
        """
        ret = copy(self)
        ret.camera_TBulk = camera_TBulk
        ret._camera_temperature_zk = _Invalidated
        ret.camera_zk = _Invalidated
        return ret

    def with_aos_dof(self, dof):
        """Return new SSTBuilder that includes specified AOS degrees of freedom

        Parameters
        ----------
        dof : ndarray (50,)
            AOS degrees of freedom.
            0,1,2 are M2 z,x,y in micron
            3,4 are M2 rot around x, y in arcsec
            5,6,7 are camera z,x,y in micron
            8,9 are camera rot around x, y in arcsec
            10-29 are M1M3 bending modes in micron
            30-49 are M2 bending modes in micron

        Returns
        -------
        ret : SSTBuilder
            New builder with specified AOS DOF.
        """
        ret = copy(self)
        ret.dof = dof
        # Invalidate dependents
        if np.any(dof[10:30]):
            ret._m1_bend_grid = _Invalidated
            ret._m3_bend_grid = _Invalidated
            ret._m1m3_bend_zk = _Invalidated
            ret.m1_grid = _Invalidated
            ret.m3_grid = _Invalidated
            ret.m1m3_zk = _Invalidated
        if np.any(dof[30:50]):
            ret._m2_bend_grid = _Invalidated
            ret._m2_bend_zk = _Invalidated
            ret.m2_grid = _Invalidated
            ret.m2_zk = _Invalidated
        return ret

    def _compute_m1m3_gravity(self):
        if self._m1m3_fea_gravity is not _Invalidated:
            return
        if self.m1m3_zenith_angle is None:
            self._m1m3_fea_gravity = None
            return

        zdata = _fits_cache("M1M3_dxdydz_zenith.fits.gz")
        hdata = _fits_cache("M1M3_dxdydz_horizon.fits.gz")
        dxyz = (
            zdata * np.cos(self.m1m3_zenith_angle) +
            hdata * np.sin(self.m1m3_zenith_angle)
        )
        dz = dxyz[:,2]

        # Interpolate these node displacements into z-displacements at
        # original node x/y positions.
        bx = self.m1m3_fea_x
        by = self.m1m3_fea_y
        idx1 = self.m1m3_fea_idx1
        idx3 = self.m1m3_fea_idx3

        # M1
        zRef = self.fiducial['M1'].surface.sag(bx[idx1], by[idx1])
        zpRef = self.fiducial['M1'].surface.sag(
            (bx+dxyz[:, 0])[idx1],
            (by+dxyz[:, 1])[idx1]
        )
        dz[idx1] += zRef - zpRef

        # M3
        zRef = self.fiducial['M3'].surface.sag(bx[idx3], by[idx3])
        zpRef = self.fiducial['M3'].surface.sag(
            (bx+dxyz[:, 0])[idx3],
            (by+dxyz[:, 1])[idx3]
        )
        dz[idx3] += zRef - zpRef

        # Subtract PTT
        # This kinda makes sense for M1, but why for combined M1M3?
        zBasis = galsim.zernike.zernikeBasis(
            3, bx, by, R_outer=4.18, R_inner=2.558
        )
        coefs, _, _, _ = np.linalg.lstsq(zBasis.T, dxyz[:, 2], rcond=None)
        zern = galsim.zernike.Zernike(coefs, R_outer=4.18, R_inner=2.558)
        dz -= zern(bx, by)

        self._m1m3_fea_gravity = dz

    def _compute_m1m3_temperature(self):
        if self._m1m3_fea_temperature is not _Invalidated:
            return
        if not np.any([
            self.m1m3_TBulk,
            self.m1m3_TxGrad,
            self.m1m3_TyGrad,
            self.m1m3_TzGrad,
            self.m1m3_TrGrad,
        ]):
            self._m1m3_fea_gravity = None
            return

        bx = self.m1m3_fea_x
        by = self.m1m3_fea_y
        normX = bx / 4.18
        normY = by / 4.18

        data = _fits_cache("M1M3_thermal_FEA.fits.gz")
        delaunay = Delaunay(data[:, 0:2])
        tbdz = CloughTocher2DInterpolator(delaunay, data[:, 2])(normX, normY)
        txdz = CloughTocher2DInterpolator(delaunay, data[:, 3])(normX, normY)
        tydz = CloughTocher2DInterpolator(delaunay, data[:, 4])(normX, normY)
        tzdz = CloughTocher2DInterpolator(delaunay, data[:, 5])(normX, normY)
        trdz = CloughTocher2DInterpolator(delaunay, data[:, 6])(normX, normY)

        out = self.m1m3_TBulk * tbdz
        out += self.m1m3_TxGrad * txdz
        out += self.m1m3_TyGrad * tydz
        out += self.m1m3_TzGrad * tzdz
        out += self.m1m3_TrGrad * trdz
        out *= 1e-6
        self._m1m3_fea_temperature = out

    def _consolidate_m1m3_fea(self):
        # Take
        #     _m1m3_fea_gravity,  _m1m3_fea_temperature
        # and set
        #     _m1_fea_grid, _m3_fea_grid, _m1m3_fea_zk
        if self._m1_fea_grid is not _Invalidated:
            return
        if (
            self._m1m3_fea_gravity is None
            and self._m1m3_fea_temperature is None
        ):
            self._m1_fea_grid = None
            self._m3_fea_grid = None
            self._m1m3_fea_zk = None
            return
        m1m3_fea = np.zeros(5256)
        if self._m1m3_fea_gravity is not None:
            m1m3_fea = self._m1m3_fea_gravity
        if self._m1m3_fea_temperature is not None:
            m1m3_fea += self._m1m3_fea_temperature

        if np.any(m1m3_fea):
            bx = self.m1m3_fea_x
            by = self.m1m3_fea_y
            idx1 = self.m1m3_fea_idx1
            idx3 = self.m1m3_fea_idx3
            zBasis = galsim.zernike.zernikeBasis(
                28, -bx, by, R_outer=4.18
            )
            m1m3_zk, *_ = np.linalg.lstsq(zBasis.T, m1m3_fea, rcond=None)
            zern = galsim.zernike.Zernike(m1m3_zk, R_outer=4.18)
            m1m3_fea -= zern(-bx, by)

            m1_grid_coords = np.vstack([self.m1_grid_x, self.m1_grid_y])
            m3_grid_coords = np.vstack([self.m3_grid_x, self.m3_grid_y])
            m1_grid = _node_to_grid(
                bx[idx1], by[idx1], m1m3_fea[idx1], m1_grid_coords
            )

            m3_grid = _node_to_grid(
                bx[idx3], by[idx3], m1m3_fea[idx3], m3_grid_coords
            )
            m1_grid *= -1
            m3_grid *= -1
            m1m3_zk *= -1
            self._m1_fea_grid = m1_grid
            self._m3_fea_grid = m3_grid
            self._m1m3_fea_zk = m1m3_zk
        else:
            self._m1_fea_grid = None
            self._m3_fea_grid = None
            self._m1m3_fea_zk = None

    def _compute_m1m3_bend(self):
        if self._m1m3_bend_zk is not _Invalidated:
            return
        dof = self.dof[10:30]
        if np.any(dof):
            m1_bend = _fits_cache("M1_bend_grid.fits.gz")
            m3_bend = _fits_cache("M3_bend_grid.fits.gz")
            self._m1_bend_grid = np.tensordot(m1_bend, dof, axes=[[1], [0]])
            self._m3_bend_grid = np.tensordot(m3_bend, dof, axes=[[1], [0]])

            self._m1m3_bend_zk = np.dot(dof, _fits_cache("M13_bend_zk.fits.gz"))
        else:
            self._m1_bend_grid = None
            self._m3_bend_grid = None
            self._m1m3_bend_zk = None

    def _consolidate_m1_grid(self):
        # Take m1_fea_grid, m1_bend_grid and make m1_grid.
        if self.m1_grid is not _Invalidated:
            return
        if (
            self._m1_bend_grid is None
            and self._m1_fea_grid is None
        ):
            self.m1_grid = None
            return

        if self._m1_bend_grid is not None:
            m1_grid = self._m1_bend_grid
        else:
            m1_grid = np.zeros((4, 204, 204))
        if self._m1_fea_grid is not None:
            m1_grid += self._m1_fea_grid
        self.m1_grid = m1_grid

    def _consolidate_m3_grid(self):
        # Take m3_fea_grid, m3_bend_grid and make m3_grid.
        if self.m3_grid is not _Invalidated:
            return
        if (
            self._m3_bend_grid is None
            and self._m3_fea_grid is None
        ):
            self.m3_grid = None
            return

        if self._m3_bend_grid is not None:
            m3_grid = self._m3_bend_grid
        else:
            m3_grid = np.zeros((4, 204, 204))
        if self._m3_fea_grid is not None:
            m3_grid += self._m3_fea_grid
        self.m3_grid = m3_grid

    def _consolidate_m1m3_zk(self):
        if self.m1m3_zk is not _Invalidated:
            return
        if (
            self._m1m3_bend_zk is None
            and self._m1m3_fea_zk is None
        ):
            self.m1m3_zk = None
            return
        m1m3_zk = np.zeros(29)
        if self._m1m3_bend_zk is not None:
            m1m3_zk += self._m1m3_bend_zk
        if self._m1m3_fea_zk is not None:
            m1m3_zk += self._m1m3_fea_zk
        self.m1m3_zk = m1m3_zk

    def _compute_m2_gravity(self):
        if self._m2_fea_gravity is not _Invalidated:
            return
        if self.m2_zenith_angle is None:
            self._m2_fea_gravity = None
            return

        data = _fits_cache("M2_GT_grid.fits.gz")
        zdz, hdz = data[0:2]

        out = zdz * (np.cos(self.m2_zenith_angle) - 1)
        out += hdz * np.sin(self.m2_zenith_angle)
        out *= 1e-6  # micron -> meters

        self._m2_fea_gravity = out

    def _compute_m2_temperature(self):
        if self._m2_fea_temperature is not _Invalidated:
            return
        if not np.any([self.m2_TrGrad, self.m2_TzGrad]):
            self._m2_fea_temperature = None
            return

        data = _fits_cache("M2_GT_grid.fits.gz")
        tzdz, trdz = data[2:4]

        out = self.m2_TzGrad * tzdz
        out += self.m2_TrGrad * trdz
        out *= 1e-6

        self._m2_fea_temperature = out

    def _consolidate_m2_fea(self):
        if self._m2_fea_grid is not _Invalidated:
            return
        if (
            self._m2_fea_gravity is None
            and self._m2_fea_temperature is None
        ):
            self._m2_fea_grid = None
            self._m2_fea_zk = None
            return
        m2_fea = np.zeros(15984)
        if self._m2_fea_gravity is not None:
            m2_fea = self._m2_fea_gravity
        if self._m2_fea_temperature is not None:
            m2_fea += self._m2_fea_temperature

        if np.any(m2_fea):
            bx = self.m2_fea_x
            by = self.m2_fea_y
            zBasis = galsim.zernike.zernikeBasis(
                28, -bx, by, R_outer=1.71
            )
            m2_zk, *_ = np.linalg.lstsq(zBasis.T, m2_fea, rcond=None)
            zern = galsim.zernike.Zernike(m2_zk, R_outer=1.71)
            m2_fea -= zern(-bx, by)
            m2_grid_coords = np.vstack([self.m2_grid_x, self.m2_grid_y])
            m2_grid = _node_to_grid(
                bx, by, m2_fea, m2_grid_coords
            )
            m2_grid *= -1
            m2_zk *= -1
            self._m2_fea_grid = m2_grid
            self._m2_fea_zk = m2_zk
        else:
            self._m2_fea_grid = None
            self._m2_fea_zk = None

    def _compute_m2_bend(self):
        if self._m2_bend_zk is not _Invalidated:
            return
        dof = self.dof[30:50]
        if np.any(dof):
            m2_bend = _fits_cache("M2_bend_grid.fits.gz")
            self._m2_bend_grid = np.tensordot(m2_bend, dof, axes=[[1], [0]])
            self._m2_bend_zk = np.dot(dof, _fits_cache("M2_bend_zk.fits.gz"))
        else:
            self._m2_bend_grid = None
            self._m2_bend_zk = None

    def _consolidate_m2_grid(self):
        # Take m2_fea_grid, m2_bend_grid and make m2_grid.
        if self.m2_grid is not _Invalidated:
            return
        if (
            self._m2_bend_grid is None
            and self._m2_fea_grid is None
        ):
            self.m2_grid = None
            return

        if self._m2_bend_grid is not None:
            m2_grid = self._m2_bend_grid
        else:
            m2_grid = np.zeros((4, 204, 204))
        if self._m2_fea_grid is not None:
            m2_grid += self._m2_fea_grid
        self.m2_grid = m2_grid

    def _consolidate_m2_zk(self):
        if self.m2_zk is not _Invalidated:
            return
        if (
            self._m2_bend_zk is None
            and self._m2_fea_zk is None
        ):
            self.m2_zk = None
            return
        m2_zk = np.zeros(29)
        if self._m2_bend_zk is not None:
            m2_zk += self._m2_bend_zk
        if self._m2_fea_zk is not None:
            m2_zk += self._m2_fea_zk
        self.m2_zk = m2_zk

    def _compute_camera_gravity(self):
        if self._camera_gravity_zk is not _Invalidated:
            return
        if self.camera_zenith_angle is None:
            self._camera_gravity_zk = None
            return

        rotation = self.camera_rotation_angle
        zenith = self.camera_zenith_angle
        self._camera_gravity_zk = {}
        cam_data = [
            ('L1S1', 'L1_entrance'),
            ('L1S2', 'L1_exit'),
            ('L2S1', 'L2_entrance'),
            ('L2S2', 'L2_exit'),
            ('L3S1', 'L3_entrance'),
            ('L3S2', 'L3_exit')
        ]
        for tname, bname in cam_data:
            data = _fits_cache(tname+"zer.fits.gz")
            grav_zk = data[0, 3:] * (np.cos(zenith) - 1)
            grav_zk += (
                data[1, 3:] * np.cos(rotation) +
                data[2, 3:] * np.sin(rotation)
            ) * np.sin(zenith)

            # remap Andy -> Noll Zernike indices
            zIdxMapping = [
                1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
                17, 21, 16, 25, 24, 26, 23, 27, 22, 28
            ]
            grav_zk = grav_zk[[x - 1 for x in zIdxMapping]]
            grav_zk *= 1e-3  # mm -> m
            # tsph -> batoid 0-index offset
            grav_zk = np.concatenate([[0], grav_zk])
            self._camera_gravity_zk[bname] = grav_zk

    def _compute_camera_temperature(self):
        if self._camera_temperature_zk is not _Invalidated:
            return
        if self.camera_TBulk is None:
            self._camera_temperature_zk = None
            return
        TBulk = self.camera_TBulk
        self._camera_temperature_zk = {}
        cam_data = [
            ('L1S1', 'L1_entrance'),
            ('L1S2', 'L1_exit'),
            ('L2S1', 'L2_entrance'),
            ('L2S2', 'L2_exit'),
            ('L3S1', 'L3_entrance'),
            ('L3S2', 'L3_exit')
        ]
        for tname, bname in cam_data:
            data = _fits_cache(tname+"zer.fits.gz")
            # subtract pre-compensated grav...
            fidx = np.interp(TBulk, data[3:, 2], np.arange(len(data[3:, 2])))+3
            idx = int(np.floor(fidx))
            whi = fidx - idx
            wlo = 1 - whi
            temp_zk = wlo * data[idx, 3:] + whi * data[idx+1, 3:]

            # subtract reference temperature zk (0 deg C is idx=5)
            temp_zk -= data[5, 3:]

            # remap Andy -> Noll Zernike indices
            zIdxMapping = [
                1, 3, 2, 5, 4, 6, 8, 9, 7, 10, 13, 14, 12, 15, 11, 19, 18, 20,
                17, 21, 16, 25, 24, 26, 23, 27, 22, 28
            ]
            temp_zk = temp_zk[[x - 1 for x in zIdxMapping]]
            temp_zk *= 1e-3  # mm -> m
            # tsph -> batoid 0-index offset
            temp_zk = np.concatenate([[0], temp_zk])
            self._camera_temperature_zk[bname] = temp_zk

    def _consolidate_camera(self):
        if self.camera_zk is not _Invalidated:
            return
        if (
            self._camera_gravity_zk is None
            and self._camera_temperature_zk is None
        ):
            self.camera_zk = None
            return
        zk = {}
        for bname, radius in [
            ('L1_entrance', 0.775),
            ('L1_exit', 0.775),
            ('L2_entrance', 0.551),
            ('L2_exit', 0.551),
            ('L3_entrance', 0.361),
            ('L3_exit', 0.361),
        ]:
            zk[bname] = (np.zeros(29), radius)
            if self._camera_gravity_zk is not None:
                zk[bname][0][:] += self._camera_gravity_zk[bname]
            if self._camera_temperature_zk is not None:
                zk[bname][0][:] += self._camera_temperature_zk[bname]
        self.camera_zk = zk

    def _apply_rigid_body_perturbations(self, optic):
        dof = self.dof
        if np.any(dof[0:3]):
            optic = optic.withGloballyShiftedOptic(
                "M2",
                np.array([dof[1], dof[2], -dof[0]])*1e-6
            )

        if np.any(dof[3:5] != 0):
            rx = batoid.RotX(np.deg2rad(-dof[3]/3600))
            ry = batoid.RotY(np.deg2rad(-dof[4]/3600))
            optic = optic.withLocallyRotatedOptic(
                "M2",
                rx @ ry
            )

        if np.any(dof[5:8] != 0):
            optic = optic.withGloballyShiftedOptic(
                "LSSTCamera",
                np.array([dof[6], dof[7], -dof[5]])*1e-6
            )

        if np.any(dof[8:10] != 0):
            rx = batoid.RotX(np.deg2rad(-dof[8]/3600))
            ry = batoid.RotY(np.deg2rad(-dof[9]/3600))
            optic = optic.withLocallyRotatedOptic(
                "LSSTCamera",
                rx @ ry
            )

        return optic

    def _apply_surface_perturbations(self, optic):
        # M1
        components = [optic['M1'].surface]
        if np.any(self.m1_grid):
            components.append(
                batoid.Bicubic(self.m1_grid_x, self.m1_grid_y, *self.m1_grid)
            )
        if np.any(self.m1m3_zk):
            components.append(
                batoid.Zernike(self.m1m3_zk, R_outer=4.18)
            )
        if len(components) > 1:
            optic = optic.withSurface('M1', batoid.Sum(components))

        # M3
        components = [optic['M3'].surface]
        if np.any(self.m3_grid):
            components.append(
                batoid.Bicubic(self.m3_grid_x, self.m3_grid_y, *self.m3_grid)
            )
        if np.any(self.m1m3_zk):
            components.append(
                # Note, using M1 R_outer here intentionally.
                batoid.Zernike(self.m1m3_zk, R_outer=4.18)
            )
        if len(components) > 1:
            optic = optic.withSurface('M3', batoid.Sum(components))

        # M2
        components = [optic['M2'].surface]
        if np.any(self.m2_grid):
            components.append(
                batoid.Bicubic(self.m2_grid_x, self.m2_grid_y, *self.m2_grid)
            )
        if np.any(self.m2_zk):
            components.append(
                batoid.Zernike(self.m2_zk, R_outer=1.71)
            )
        if len(components) > 1:
            optic = optic.withSurface('M2', batoid.Sum(components))

        # Camera
        if self.camera_zk is not None:
            for k, (zk, radius) in self.camera_zk.items():
                optic = optic.withSurface(
                    k,
                    batoid.Sum([
                        optic[k].surface,
                        batoid.Zernike(zk, R_outer=radius)
                    ])
                )

        return optic

    def build(self):
        # Fill arrays (possibly with None if all dependencies are None)
        # We're manually traversing the DAG effectively
        self._compute_m1m3_gravity()
        self._compute_m1m3_temperature()
        self._consolidate_m1m3_fea()
        self._compute_m1m3_bend()
        self._consolidate_m1_grid()
        self._consolidate_m3_grid()
        self._consolidate_m1m3_zk()

        self._compute_m2_gravity()
        self._compute_m2_temperature()
        self._consolidate_m2_fea()
        self._compute_m2_bend()
        self._consolidate_m2_grid()
        self._consolidate_m2_zk()

        self._compute_camera_gravity()
        self._compute_camera_temperature()
        self._consolidate_camera()

        optic = self.fiducial
        optic = self._apply_rigid_body_perturbations(optic)
        optic = self._apply_surface_perturbations(optic)
        return optic
