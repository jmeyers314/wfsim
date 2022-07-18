import os
import batoid
import numpy as np
from wfsim import SSTBuilder
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

# Loop over survey parameters
with tqdm(total=4*4*35) as pbar:
    for zenith_angle, rotation_angle in [
        (0, 0),
        (45, 0),
        (45, 45),
        (30, -30)
    ]:
        for m1m3, m2, camera, feaname in [
            (True, False, False, "M1M3"),
            (False, True, False, "M2"),
            (False, False, True, "Cam"),
            (True, True, True, "All")
        ]:
            name = f"z{zenith_angle}_r{rotation_angle}_{feaname}"
            odir = os.path.join("fea", name)
            os.makedirs(odir, exist_ok=True)
            builder = SSTBuilder(batoid.Optic.fromYaml("LSST_g_500.yaml"))
            if m1m3:
                builder = (
                    builder
                    .with_m1m3_gravity(np.deg2rad(zenith_angle))
                    .with_m1m3_temperature(
                        m1m3_TBulk=0.0902,
                        m1m3_TxGrad=-0.0894,
                        m1m3_TyGrad=-0.1973,
                        m1m3_TzGrad=-0.0316,
                        m1m3_TrGrad=0.0187
                    )
                )
            if m2:
                builder = (
                    builder
                    .with_m2_gravity(np.deg2rad(zenith_angle))
                    .with_m2_temperature(
                        m2_TrGrad=-0.1416,
                        m2_TzGrad=-0.0675
                    )
                )
            if camera:
                builder = (
                    builder
                    .with_camera_gravity(
                        zenith_angle=np.deg2rad(zenith_angle),
                        rotation_angle=np.deg2rad(rotation_angle)
                    )
                    .with_camera_temperature(
                        camera_TBulk=6.5650
                    )
                )
            optic = builder.build()
            for ifield, (fieldX, fieldY) in enumerate(fieldXY):
                opd = batoid.wavefront(
                    optic,
                    np.deg2rad(fieldX), np.deg2rad(fieldY),
                    wavelength=500e-9, nx=255
                ).array * 0.5
                opddata = opd.data.astype(np.float32)
                opddata[opd.mask] = 0.0
                ofn = os.path.join("fea", name, f"opd_{name}_{ifield}.fits.gz")
                fits.writeto(
                    ofn,
                    opddata,
                    overwrite=True
                )
                pbar.update()
