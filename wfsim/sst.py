from ast import Del
import galsim
import os
import numpy as np
import astropy.io.fits as fits
import batoid
import functools
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay


@functools.lru_cache
def _fitsCache(fn):
    from . import datadir
    return fits.getdata(
        os.path.join(
            datadir,
            fn
        )
    )

def _node_to_grid(nodex, nodey, nodez, grid_coords):
    interp = CloughTocher2DInterpolator(
        np.array([nodex, nodey]).T,
        nodez,
        fill_value=0.0
    )
    x, y = grid_coords
    nx = len(x)
    ny = len(y)
    out = np.zeros([4, ny, nx])
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
        self.fiducial = fiducial

    @functools.cached_property
    def m1m3_fea_coords(self):
        data = _fitsCache("M1M3_1um_156_grid.fits.gz")
        idx = data[:, 0]
        bx = data[:, 1]  # (5256,)
        by = data[:, 2]
        idx1 = (idx == 1)
        idx3 = (idx == 3)
        return bx, by, idx1, idx3

    @functools.cached_property
    def m2_fea_coords(self):
        data = _fitsCache("M2_1um_grid.fits.gz")  # (15984, 75)
        bx = -data[:, 1]  # meters
        by = data[:, 2]
        return bx, by

    @functools.cached_property
    def m1_grid_coords(self):
        data = _fitsCache("M1_bend_coords.fits.gz")
        return data

    @functools.cached_property
    def m2_grid_coords(self):
        data = _fitsCache("M2_bend_coords.fits.gz")
        return data

    @functools.cached_property
    def m3_grid_coords(self):
        data = _fitsCache("M3_bend_coords.fits.gz")
        return data

    def _m1m3_gravity(self, zenith_angle):
        zdata = _fitsCache("M1M3_dxdydz_zenith.fits.gz")
        hdata = _fitsCache("M1M3_dxdydz_horizon.fits.gz")
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

        data = _fitsCache("M1M3_thermal_FEA.fits.gz")
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

    # def _m2_gravity(self, zenith_angle):
    #     # This reproduces ts_phosim with preCompElevInRadian=0, but what is
    #     # that?  Also, I have questions regarding the input domain of the Rbf
    #     # interpolation...
    #     bx, by = self.m2_fea_coords
    #     data = _fitsCache("M2_GT_FEA.fits.gz")

    #     from scipy.interpolate import Rbf
    #     zdz = Rbf(data[:, 0], data[:, 1], data[:, 2])(bx/1.71, by/1.71)
    #     hdz = Rbf(data[:, 0], data[:, 1], data[:, 3])(bx/1.71, by/1.71)

    #     out = zdz * (np.cos(zenith_angle) - 1)
    #     out += hdz * np.sin(zenith_angle)
    #     out *= 1e-6  # micron -> meters
    #     return out

    # def _m2_temperature(self, m2TzGrad, m2TrGrad):
    #     # Same domain problem here as m2_gravity...
    #     bx, by = self.m2_fea_coords
    #     data = _fitsCache("M2_GT_FEA.fits.gz")

    #     from scipy.interpolate import Rbf
    #     tzdz = Rbf(data[:, 0], data[:, 1], data[:, 4])(bx/1.71, by/1.71)
    #     trdz = Rbf(data[:, 0], data[:, 1], data[:, 5])(bx/1.71, by/1.71)

    #     out = m2TzGrad * tzdz
    #     out += m2TrGrad * trdz
    #     out *= 1e-6  # micron -> meters
    #     return out

    # This is Josh's preferred interpolator, but fails b/c domain issues.
    def _m2_gravity(self, zenith_angle):
        bx, by = self.m2_fea_coords
        data = _fitsCache("M2_GT_FEA.fits.gz")
        # Hack to get interpolation points inside Convex Hull of input
        delaunay = Delaunay(data[:, 0:2]/0.95069)
        zdz = CloughTocher2DInterpolator(delaunay, data[:, 2])(bx/1.71, by/1.71)
        hdz = CloughTocher2DInterpolator(delaunay, data[:, 3])(bx/1.71, by/1.71)
        out = zdz * (np.cos(zenith_angle) - 1)
        out += hdz * np.sin(zenith_angle)
        out *= 1e-6  # micron -> meters
        return out

    def _m2_temperature(self, m2TzGrad, m2TrGrad):
        # Same domain problem here as m2_gravity...
        bx, by = self.m2_fea_coords
        normX = bx / 1.71
        normY = by / 1.71
        data = _fitsCache("M2_GT_FEA.fits.gz")

        # Hack to get interpolation points inside Convex Hull of input
        delaunay = Delaunay(data[:, 0:2]/0.95069)
        tzdz = CloughTocher2DInterpolator(delaunay, data[:, 4])(normX, normY)
        trdz = CloughTocher2DInterpolator(delaunay, data[:, 5])(normX, normY)

        out = m2TzGrad * tzdz
        out += m2TrGrad * trdz
        out *= 1e-6
        return out

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
                m1_bend = _fitsCache("M1_bend_grid.fits.gz")
                m3_bend = _fitsCache("M3_bend_grid.fits.gz")
                m1_grid += np.tensordot(m1_bend, dof[10:30], axes=[[1], [0]])
                m3_grid += np.tensordot(m3_bend, dof[10:30], axes=[[1], [0]])

            if not _omit_dof_zk:
                m1m3_zk += np.dot(dof[10:30], _fitsCache("M13_bend_zk.fits.gz"))

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

                m3_grid = _node_to_grid(
                    bx, by, m2_fea_dz, self.m2_grid_coords
                )

                m2_grid *= -1
                m2_zk *= -1

        if np.any(dof[30:50] != 0):
            if not _omit_dof_grid:
                m2_bend = _fitsCache("M2_bend_grid.fits.gz")
                m2_grid += np.tensordot(m2_bend, dof[30:50], axes=[[1], [0]])

            if not _omit_dof_zk:
                m2_zk += np.dot(dof[30:50], _fitsCache("M2_bend_zk.fits.gz"))

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
                data = _fitsCache(tname+"zer.fits.gz")
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
