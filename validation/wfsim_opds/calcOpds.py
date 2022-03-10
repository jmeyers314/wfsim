import os
import batoid
import numpy as np
from wfsim import SSTFactory
from tqdm import tqdm
import astropy.io.fits as fits


def phosim_style_opd(
    fiducial, perturbed, fieldX, fieldY, opd_size=255, opd_sampling=4,
    wavelength=500e-9
):
    # First, get direction cosines in PhoSim convention where
    # fieldX is RA and fieldY is DEC
    fieldX = np.deg2rad(fieldX)
    fieldY = np.deg2rad(fieldY)

    # Get direction cosines for this field angle
    vx = np.sin(fieldX)*np.cos(fieldY)
    vy = np.sin(fieldY)
    vz = -np.cos(fieldX)*np.cos(fieldY)
    dirCos = np.array([vx, vy, vz])

    # 1) Get reference sphere radius
    xPP = batoid.exitPupilPos(
        perturbed, wavelength=wavelength, smallAngle=1e-5,
        stopSurface=fiducial['M1']
    )
    ref_sph_rad = np.sqrt(np.sum(np.square(
        xPP - fiducial['Detector'].coordSys.origin
    )))

    # 2) Get Chief ray intersection with focal plane
    #    Launch wrt fiducial telescope, but trace wrt perturbed telescope
    chief_ray = batoid.RayVector.fromStop(
        0.0, 0.0,
        optic=fiducial,
        stopSurface=fiducial['M1'],
        wavelength=wavelength,
        dirCos=dirCos
    )
    fiducial.trace(chief_ray)

    # 3) Generate raygrid
    x = np.linspace(-4.18, 4.18, opd_size*opd_sampling)
    x, y = np.meshgrid(x, x)
    x = x.ravel()
    y = y.ravel()
    z = fiducial['M1'].surface.sag(x, y)
    t = 20.0 + vz*z + vx*x + vy*y  # I _think_
    rays = batoid.RayVector(x, y, z, vx, vy, vz, t, wavelength)

    # Get EP coordinates
    ep_rays = fiducial.stopSurface.surface.intersect(rays.copy())

    # Trace to detector
    perturbed.trace(rays)

    # Now need to trace to the reference sphere.
    # Place vertex of reference sphere one radius length away from the
    # intersection point.  So transform our rays into that coordinate system.
    targetCoordSys = rays.coordSys.shiftLocal(
        chief_ray.r[0] + np.array([0, 0, ref_sph_rad])
    )
    rays.toCoordSys(targetCoordSys)

    sphere = batoid.Sphere(-ref_sph_rad)
    sphere.intersect(rays)

    # 4) Bin based on EP coordinates.  Track both OPL sum and number of rays per
    # bin; then divide.
    edges = np.arange(opd_size+1)*8.36/(opd_size-1)
    edges -= np.mean(edges)
    H1, _, _ = np.histogram2d(ep_rays.x, ep_rays.y, weights=rays.t, bins=edges)
    bincount1, _, _ = np.histogram2d(ep_rays.x, ep_rays.y, bins=edges)
    with np.errstate(all='ignore'):
        t0 = (H1/bincount1)[opd_size//2, opd_size//2]

    good = ~rays.vignetted
    H, _, _ = np.histogram2d(
        ep_rays.x[good], ep_rays.y[good],
        weights=rays.t[good], bins=edges
    )
    bincount, _, _ = np.histogram2d(
        ep_rays.x[good], ep_rays.y[good],
        bins=edges
    )
    with np.errstate(all='ignore'):
        H /= bincount
    H -= t0
    H = np.ma.masked_array(H, mask=bincount==0)*1e6  # m -> micron

    return -H.T


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

os.makedirs('opd', exist_ok=True)

# Do fiducial telescope first.
for ifield, (fieldX, fieldY) in enumerate(tqdm(fieldXY)):
    opd = phosim_style_opd(
        factory.fiducial, factory.fiducial,
        fieldX, fieldY
    )
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
            opd = phosim_style_opd(
                factory.fiducial, perturbed,
                fieldX, fieldY
            )
            opddata = opd.data.astype(np.float32)
            opddata[opd.mask] = 0.0
            fits.writeto(
                f"opd/opd_mode_{imode}_field_{ifield}.fits.gz",
                opddata,
                overwrite=True
            )
            pbar.update()
