import time
import batoid
import galsim
import numpy as np
from scipy.optimize import minimize_scalar


telescope = batoid.Optic.fromYaml("LSST_i.yaml")
focal_length = np.sqrt(np.linalg.det(batoid.drdth(
    telescope, 0.0, 0.0, 720e-9
)))


def fwhm(telescope, thx, thy):
    zs = batoid.analysis.zernikeTA(
        telescope,
        thx, thy,
        720e-9,
        reference='chief',
        eps=0.61,
        focal_length=focal_length
    ) * 720e-9
    zern = galsim.zernike.Zernike(zs, R_outer=4.18, R_inner=0.61*4.18)
    rmsgrad = np.sqrt(np.sum(zern.gradX.coef[4:]**2 + zern.gradY.coef[4:]**2))
    # convert to arcsec
    rmsgrad = np.rad2deg(rmsgrad) * 3600
    # convert sigma to FWHM
    rmsgrad *= np.sqrt(np.log(256))
    return rmsgrad


def gq(func, outer, inner=0, rings=3, jmax=23):
    spokes = 2*rings+1
    Li, w = np.polynomial.legendre.leggauss(rings)
    eps = inner/outer
    area = np.pi*(1-eps**2)
    rings = np.sqrt(eps**2 + (1+Li)*(1-eps**2)/2)*outer
    weights = w*area/(2*spokes)
    spokes = np.linspace(0, 2*np.pi, spokes, endpoint=False)
    rings, spokes = np.meshgrid(rings, spokes)
    weights = np.broadcast_to(weights, rings.shape)
    rings = rings.ravel()
    spokes = spokes.ravel()
    weights = weights.ravel()
    x = rings * np.cos(spokes)
    y = rings * np.sin(spokes)
    vals = np.empty_like(x)
    for i, (x_, y_) in enumerate(zip(x, y)):
        vals[i] = func(x_, y_)

    return np.sum(vals*weights)/area


def fwhm_fov(telescope):
    return gq(lambda thx, thy: fwhm(telescope, thx, thy), outer=np.deg2rad(1.75), rings=5)


def merit_fwhm(dz):
    telescope = batoid.Optic.fromYaml("LSST_i.yaml")
    perturbed = telescope.withGloballyShiftedOptic("LSSTCamera", [0,0,dz])
    fwhm_ = fwhm_fov(perturbed)
    print(dz, fwhm_)
    return fwhm_


t0 = time.time()
# print(merit_fwhm(0.0))
result = minimize_scalar(merit_fwhm, bracket=[-0.0005, 0.0, 0.0005], tol=1e-4)
t1 = time.time()

# print(result)
print(t1-t0)
