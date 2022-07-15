import batoid
import numpy as np
import wfsim
import matplotlib.pyplot as plt

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

fiducial = batoid.Optic.fromYaml("LSST_r.yaml")

dof = np.zeros(50)
dof[44] = 1

m1m3_TBulk = 0.01
m1m3_TxGrad = 0.02
m1m3_TyGrad = 0.03
m1m3_TzGrad = 0.04
m1m3_TrGrad = 0.03
camera_TBulk = 0.4


builder = wfsim.SSTBuilder(fiducial)
builder = builder.with_m1m3_gravity(zenith_angle=0.1)
builder = builder.with_m1m3_temperature(
    m1m3_TBulk=m1m3_TBulk,
    m1m3_TrGrad=m1m3_TrGrad,
    m1m3_TxGrad=m1m3_TxGrad,
    m1m3_TyGrad=m1m3_TyGrad,
    m1m3_TzGrad=m1m3_TzGrad
)
builder = builder.with_camera_temperature(camera_TBulk=camera_TBulk)
builder = builder.with_camera_gravity(zenith_angle=0.1, rotation_angle=0.1)
builder = builder.with_m2_gravity(zenith_angle=0.1)

builder = builder.with_aos_dof(dof)
optic = builder.build()

factory = wfsim.SSTFactory(fiducial)
optic2 = factory.get_telescope(
    zenith_angle=0.1,
    rotation_angle=0.1,
    m1m3TBulk=m1m3_TBulk,
    m1m3TrGrad=m1m3_TrGrad,
    m1m3TxGrad=m1m3_TxGrad,
    m1m3TyGrad=m1m3_TyGrad,
    m1m3TzGrad=m1m3_TzGrad,
    dof=dof,
    camTB=camera_TBulk,
    doM1M3Pert=True,
    doM2Pert=True,
    doCamPert=True,
)

wf = batoid.wavefront(optic, 0.03, 0, 620e-9, nx=255)
wf2 = batoid.wavefront(optic2, 0.03, 0, 620e-9, nx=255)

fig, axes = plt.subplots(ncols=3, figsize=(12, 5))
colorbar(axes[0].imshow(wf.array*620))
colorbar(axes[1].imshow(wf2.array*620))
colorbar(axes[2].imshow((wf.array-wf2.array)*620))
plt.tight_layout()
plt.show()

zk = batoid.zernike(
    optic,
    0.0, 0.03,
    620e-9, eps=0.61, nx=255
) * 620

zk2 = batoid.zernike(
    optic2,
    0.0, 0.03,
    620e-9, eps=0.61, nx=255
) * 620

for i in range(1, 23):
    print(f"{i:2d} {zk[i]:12.4f} {zk2[i]:12.4f}")


# Make sure build order after chaining doesn't matter
builder = wfsim.SSTBuilder(fiducial)
builder2 = builder.with_m1m3_gravity(zenith_angle=0.1)

optic1 = builder.build()
optic2 = builder2.build()

builder = wfsim.SSTBuilder(fiducial)
builder2 = builder.with_m1m3_gravity(zenith_angle=0.1)

optic2a = builder2.build()
optic1a = builder.build()


zk1 = batoid.zernike(
    optic1,
    0.0, 0.03,
    620e-9, eps=0.61, nx=255
)
zk1a = batoid.zernike(
    optic1a,
    0.0, 0.03,
    620e-9, eps=0.61, nx=255
)


zk2 = batoid.zernike(
    optic2,
    0.0, 0.03,
    620e-9, eps=0.61, nx=255
)
zk2a = batoid.zernike(
    optic2a,
    0.0, 0.03,
    620e-9, eps=0.61, nx=255
)

print(zk1 - zk1a)
print(zk2 - zk2a)
