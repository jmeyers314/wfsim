import numpy as np
import yaml
import tqdm

import galsim
import batoid

import wfsim

"""
Simulate 10 AuxTel intra/extra pairs with random rigid-body M2 displacements
applied.
"""

bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
pixel_scale = 10e-6
fiducial_telescope = batoid.Optic.fromYaml("AuxTel.yaml")

DEVELOP = False
PLOT = False
if DEVELOP:
    NSIM = 3
    NSTAR = 5
else:
    NSIM = 10
    NSTAR = 30

# for isim in tqdm.trange(NSIM):
for isim in tqdm.trange(4, NSIM):
    seed = 57721+isim
    rng = np.random.default_rng(seed)
    zenith = rng.uniform(5, 50)*galsim.degrees
    raw_seeing = rng.uniform(0.7, 1.5)*galsim.arcsec
    # Making something completely up for now.
    observation = {
        'zenith': zenith,
        'raw_seeing': raw_seeing,
        'wavelength': 620.0,  # nm
        'exptime': 20.0,
        'temperature': 293.15,  # K
        'pressure': 69.328,  # kPa
        'H2O_pressure': 1.067,  # kPa
    }

    atm_kwargs = {
        'kcrit': 0.2,
        'screen_size': 25.6 if DEVELOP else 409.6,
        'screen_scale': 0.1,
        'nproc': 6,
    }

    # Apply perturbation to telescope
    dx = rng.uniform(-0.00075, 0.00075)  # 1.5 mm range
    dy = rng.uniform(-0.00075, 0.00075)  #
    dz = rng.uniform(-0.00005, 0.00005)  # 0.1 mm range
    dthx = np.deg2rad(rng.uniform(-1.0, 1.0)/60)  # 2 arcmin range
    dthy = np.deg2rad(rng.uniform(-1.0, 1.0)/60)
    perturbed = fiducial_telescope.withLocallyRotatedOptic(
        "M2", batoid.RotX(dthx) @ batoid.RotY(dthy)
    ).withGloballyShiftedOptic(
        "M2", [dx, dy, dz]
    )
    intra = perturbed.withGloballyShiftedOptic("M2", [0, 0, -0.0008])
    extra = perturbed.withGloballyShiftedOptic("M2", [0, 0, +0.0008])

    zs = batoid.zernike(perturbed, 0, 0, 620e-9, eps=0.423)
    if DEVELOP:
        for i in range(4, 12):
            print(f"Z{i:2}  {zs[i]*620:.6f}")
        print()
        print()
        print()

    if DEVELOP:
        fluxes = [int(n) for n in 10**rng.uniform(5, 5.5, size=NSTAR)]
    else:
        fluxes = [int(n) for n in 10**rng.uniform(6.5, 8, size=NSTAR)]
    background = rng.uniform(300, 500)

    intra_simulator = wfsim.SimpleSimulator(
        observation, atm_kwargs, intra,
        bandpass=bandpass,
        pixel_scale=pixel_scale,
        rng=rng,
        debug=False
    )
    extra_simulator = wfsim.SimpleSimulator(
        observation, atm_kwargs, extra,
        bandpass=bandpass,
        pixel_scale=pixel_scale,
        rng=rng,
        debug=False
    )

    thxs = []
    thys = []
    Ts = []

    with tqdm.tqdm(total=2*sum(fluxes), leave=False, unit_scale=True) as pbar:
        for flux in fluxes:
            rho = np.sqrt(rng.uniform(0, np.deg2rad(5/60)**2))
            th = rng.uniform(0, 2*np.pi)
            thx, thy = rho*np.cos(th), rho*np.sin(th)
            T = rng.uniform(4000, 10000)
            sed = wfsim.BBSED(T)
            Ts.append(float(T))
            thxs.append(float(thx))
            thys.append(float(thy))

            intra_simulator.add_star(thx, thy, sed, flux, rng)
            pbar.update(flux)
            extra_simulator.add_star(thx, thy, sed, flux, rng)
            pbar.update(flux)

    if not DEVELOP:
        intra_simulator.add_background(background, rng)
        extra_simulator.add_background(background, rng)

    if PLOT:
        import matplotlib.pyplot as plt
        plt.imshow(intra_simulator.image.array)
        plt.show()
        plt.imshow(extra_simulator.image.array)
        plt.show()

    intra_simulator.image.write(f"intra_{isim:03d}.fits")
    extra_simulator.image.write(f"extra_{isim:03d}.fits")
    out = dict(
        seed=seed,
        zenith=float(zenith.rad),
        raw_seeing=float(raw_seeing.rad),
        background=float(background),
        dx=float(dx),
        dy=float(dy),
        dz=float(dz),
        dthx=float(dthx),
        dthy=float(dthy),
        zern=zs.tolist(),
        thxs=thxs,
        thys=thys,
        Ts=Ts,
        fluxes=fluxes,
    )

    with open(f"sim_{isim:03d}.yaml", 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
