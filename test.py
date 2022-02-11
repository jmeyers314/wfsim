import batoid
import numpy as np
import wfsim
from tqdm import tqdm

fiducial = batoid.Optic.fromYaml("LSST_g_500.yaml")
factory = wfsim.SSTFactory(fiducial)

dof = None
optic = factory.get_telescope(
    zenith_angle=0.4728306383162878,
    rotation_angle=np.deg2rad(-70.60558909397135),
    m1m3TBulk=0.0902,
    m1m3TxGrad=-0.0894,
    m1m3TyGrad=-0.1973,
    m1m3TzGrad=-0.0316,
    m1m3TrGrad=0.0187,
    m2TzGrad=-0.0675,
    m2TrGrad=-0.1416,
    camTB=6.565,
    # dof=dof
)

zk = batoid.zernike(
    optic,
    0.0, 0.0,
    500e-9, eps=0.61, nx=255
) * 0.5

for i in range(1, 23):
    print(f"{i:2d} {zk[i]:8.4f}")
