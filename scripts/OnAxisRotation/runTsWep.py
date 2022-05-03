import os
import numpy as np
import astropy.io.fits as fits
from tqdm import tqdm

from lsst.ts.wep.cwfs.Algorithm import Algorithm
from lsst.ts.wep.cwfs.CompensableImage import CompensableImage
from lsst.ts.wep.cwfs.Instrument import Instrument
from lsst.ts.wep.Utility import (
    CamType,
    DefocalType,
    getConfigDir,
    getModulePath
)



# CWFS
cwfsConfigDir = os.path.join(getConfigDir(), "cwfs")
instDir = os.path.join(cwfsConfigDir, "instData")
inst = Instrument(instDir)
algoDir = os.path.join(cwfsConfigDir, "algo")


zernikes = np.zeros((40, 19))
angles = np.linspace(0, 360, 40, endpoint=False)
for i, angle in enumerate(tqdm(angles)):
    intra_image = fits.getdata(f"output/intra_{i:02d}.fits")
    extra_image = fits.getdata(f"output/extra_{i:02d}.fits")

    fieldXY = np.zeros(2)
    I1 = CompensableImage()
    I2 = CompensableImage()
    I1.setImg(fieldXY, DefocalType.Intra, image=intra_image.copy())
    I2.setImg(fieldXY, DefocalType.Extra, image=extra_image.copy())
    inst.config(CamType.LsstFamCam, I1.getImgSizeInPix(), announcedDefocalDisInMm=1.5)

    fftAlgo = Algorithm(algoDir)
    fftAlgo.config("fft", inst)
    fftAlgo.runIt(I1, I2, "offAxis", tol=1e-3)

    # # There's probably a reset method somewhere, but it's fast enough to just
    # # reconstruct these...
    # I1 = CompensableImage()
    # I2 = CompensableImage()
    # I1.setImg(fieldXY, DefocalType.Intra, image=intra_image.copy())
    # I2.setImg(fieldXY, DefocalType.Extra, image=extra_image.copy())
    # inst.config(CamType.LsstFamCam, I1.getImgSizeInPix(), announcedDefocalDisInMm=1.5)

    # expAlgo = Algorithm(algoDir)
    # expAlgo.config("exp", inst)
    # expAlgo.runIt(I1, I2, "offAxis", tol=1e-3)

    zernikes[i] = fftAlgo.getZer4UpInNm()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(ncols=3, nrows=7, figsize=(8, 12))
for i in range(19):
    ax = axes.ravel()[i]
    ax.plot(angles, zernikes[:, i])
    ax.set_title(f"Z{i+4:d}")
plt.tight_layout()
plt.show()