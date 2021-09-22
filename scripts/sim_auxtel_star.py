import numpy as np
import tqdm

import galsim
import batoid

import wfsim


class StarSimulator:
    def __init__(
        self,
        observation,
        atm_kwargs,
        bandpass,
        pixel_scale,
        rng,
        _seeing_factor=1.0,
        debug=False
    ):
        self.observation = observation
        self.atm, self.target_FWHM, self.r0_500, self.L0 = wfsim.make_atmosphere(
            observation['airmass'],
            observation['raw_seeing'],
            observation['wavelength'],
            rng,
            **atm_kwargs
        )
        psf = self.atm.makePSF(observation['wavelength'], diam=1.2)
        gsrng = galsim.BaseDeviate(rng.bit_generator.random_raw() % 2**63)
        _ = psf.drawImage(nx=1, ny=1, n_photons=1, rng=gsrng, method='phot')
        self.second_kick = psf.second_kick

        self.bandpass = bandpass
        self.pixel_scale = pixel_scale
        self._seeing_factor = _seeing_factor
        self.debug = debug

        self.telescope = (
            batoid.Optic.fromYaml("AuxTel.yaml")
            .withGloballyShiftedOptic("M2", [0.0, 0.0, 0.0008])
        )

        # Develop gnomonic projection from ra/dec to field angle using
        # GalSim TanWCS class.
        q = observation['rotTelPos'] - observation['rotSkyPos']
        cq, sq = np.cos(q), np.sin(q)
        affine = galsim.AffineTransform(cq, -sq, sq, cq)
        self.radec_to_field = galsim.TanWCS(
            affine,
            self.observation['boresight'],
            units=galsim.radians
        )

        # self.sensor = galsim.SiliconSensor(nrecalc=10_000)
        self.sensor = galsim.Sensor()
        self.image = galsim.Image(4096, 4096)
        self.image.setCenter(0, 0)
        self.silicon = batoid.TableMedium.fromTxt("silicon_dispersion.txt")

    def simulate_star(self, coord, sed, nphoton, rng):
        field_angle = self.radec_to_field.toImage(coord)

        # populate pupil u,v coordinates uniformly in annulus
        r_outer = 0.6
        r_inner = 0.2538
        gsrng = galsim.BaseDeviate(rng.bit_generator.random_raw() % 2**63)
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2, nphoton))
        th = rng.uniform(0, 2*np.pi, nphoton)
        u = r*np.cos(th)
        v = r*np.sin(th)
        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(u, v)
            plt.title("u, v")
            plt.show()

        # uniformly distribute photon times throughout exposure
        t = rng.uniform(0, self.observation['exptime'], nphoton)

        # evaluate phase gradients at appropriate location/time
        dku, dkv = self.atm.wavefront_gradient(
            u, v, t,
            (field_angle.x*galsim.radians, field_angle.y*galsim.radians)
        )  # output is in nm per m.  convert to radians
        dku *= 1.e-9
        dkv *= 1.e-9
        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(
                np.rad2deg(dku)*3600, np.rad2deg(dkv)*3600,
                extent=[-2, 2, -2, 2]
            )
            plt.title("first kick (arcsec)")
            plt.show()

        # add in second kick
        pa = galsim.PhotonArray(nphoton)
        self.second_kick._shoot(pa, gsrng)
        dku += pa.x*(galsim.arcsec/galsim.radians)
        dkv += pa.y*(galsim.arcsec/galsim.radians)
        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(
                np.rad2deg(dku)*3600, np.rad2deg(dkv)*3600,
                extent=[-2, 2, -2, 2]
            )
            plt.title("both kicks (arcsec)")
            plt.show()

        # assign wavelengths.
        wavelengths = sed.sampleWavelength(nphoton, self.bandpass, gsrng)

        # Chromatic seeing.  Scale deflections by (lam/500)**(-0.3)
        dku *= (wavelengths/500)**(-0.3)
        dkv *= (wavelengths/500)**(-0.3)

        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(
                np.rad2deg(dku)*3600, np.rad2deg(dkv)*3600,
                extent=[-2, 2, -2, 2]
            )
            plt.title("after chromatic seeing (arcsec)")
            plt.show()
            # What is the actual FWHM here?  2nd moments are tail sensitive.
            # Use scaled median absolute deviations.
            from scipy.stats import median_abs_deviation
            T = (
                median_abs_deviation(np.rad2deg(dku)*3600, scale='normal')
                + median_abs_deviation(np.rad2deg(dkv)*3600, scale='normal')
            )
            print(f"T = ({np.sqrt(T):.3f} arcsec)^2")
            print(f"FWHM = {np.sqrt(T/2)/0.6507373182979048:.3f} arcsec")

            # For comparison
            kolm = galsim.Kolmogorov(fwhm=self.target_FWHM)
            pa2 = galsim.PhotonArray(nphoton)
            kolm._shoot(pa2, gsrng)
            T2 = (
                median_abs_deviation(pa2.x, scale='normal')
                + median_abs_deviation(pa2.y, scale='normal')
            )
            print(f"T2 = ({np.sqrt(T2):.3f} arcsec)^2")
            print(f"FWHM2 = {np.sqrt(T2/2)/0.6507373182979048:.3f} arcsec")

            vk = galsim.VonKarman(lam=self.observation['wavelength'], r0_500=self.r0_500, L0=self.L0)
            pa3 = galsim.PhotonArray(nphoton)
            vk._shoot(pa3, gsrng)
            T3 = (
                median_abs_deviation(pa3.x, scale='normal')
                + median_abs_deviation(pa3.y, scale='normal')
            )
            print(f"T3 = ({np.sqrt(T3):.3f} arcsec)^2")
            print(f"FWHM3 = {np.sqrt(T3/2)/0.6507373182979048:.3f} arcsec")

        # DCR.  dkv is aligned along meridian, so only need to shift in this
        # direction (I think)
        base_refraction = galsim.dcr.get_refraction(
            self.observation['wavelength'],
            self.observation['zenith'],
            temperature=self.observation['temperature'],
            pressure=self.observation['pressure'],
            H2O_pressure=self.observation['H2O_pressure'],
        )
        refraction = galsim.dcr.get_refraction(
            wavelengths,
            self.observation['zenith'],
            temperature=self.observation['temperature'],
            pressure=self.observation['pressure'],
            H2O_pressure=self.observation['H2O_pressure'],
        )
        refraction -= base_refraction
        dkv += refraction

        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(
                np.rad2deg(dku)*3600, np.rad2deg(dkv)*3600,
                extent=[-2, 2, -2, 2]
            )
            plt.title("after refraction (arcsec)")
            plt.show()

        dku *= self._seeing_factor
        dkv *= self._seeing_factor

        # We're through the atmosphere!  Make a structure that batoid can use
        # now.  Note we're going to just do the sum in the tangent plane
        # coordinates.  This isn't perfect, but almost certainly good enough to
        # still be interesting.
        dku += field_angle.x
        dkv += field_angle.y
        vx, vy, vz = batoid.utils.fieldToDirCos(dku, dkv, projection='gnomonic')

        # Place rays on entrance pupil - the planar cap coincident with the rim
        # of M1.  Eventually may want to back rays up further so that they can
        # be obstructed by struts, e.g..
        x = u
        y = v
        z = self.telescope.stopSurface.surface.sag(x, y)
        ct = batoid.CoordTransform(
            self.telescope.stopSurface.coordSys,
            self.telescope.coordSys
        )
        x, y, z = ct.applyForwardArray(x, y, z)
        # Rescale velocities so that they're consistent with the current
        # refractive index.
        n = self.telescope.inMedium.getN(wavelengths)
        vx /= n
        vy /= n
        vz /= n
        rays = batoid.RayVector(
            x, y, z,
            vx, vy, vz,
            t=0.0,
            wavelength=wavelengths*1e-9,
            flux=1.0
        )

        self.telescope.trace(rays)
        if self.debug:
            import matplotlib.pyplot as plt
            x = rays.x - np.median(rays.x)
            y = rays.y - np.median(rays.y)
            plt.hexbin(
                x/10e-6, y/10e-6,
                extent=[-200, 200, -200, 200]
            )
            plt.title("through telescope (pixels)")
            plt.show()

            plt.scatter(
                x[:10000]/10e-6, y[:10000]/10e-6, c=r[:10000], s=1
            )
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)
            plt.title("through telescope w/ pupil radius")
            plt.show()

        # Now we need to refract the beam into the Silicon.
        self.telescope['Detector'].surface.refract(
            rays,
            self.telescope['Detector'].inMedium,
            self.silicon, coordSys=self.telescope['Detector'].coordSys
        )

        # Need to convert to pixels for galsim sensor object
        # Put batoid results back into photons
        # Use the same array.
        pa.x = rays.x/self.pixel_scale
        pa.y = rays.y/self.pixel_scale
        pa.dxdz = rays.vx/rays.vz
        pa.dydz = rays.vy/rays.vz
        pa.wavelength = wavelengths
        pa.flux = ~rays.vignetted

        self.sensor.accumulate(pa, self.image)

    def add_background(self, rng):
        bd = galsim.BaseDeviate(rng.bit_generator.random_raw() % 2**63)
        gd = galsim.GaussianDeviate(bd, sigma=np.sqrt(1000))
        gd.add_generate(self.image.array)


if __name__ == "__main__":
    # Making something completely up for now.
    observation = {
        'boresight': galsim.CelestialCoord(
            30*galsim.degrees, 10*galsim.degrees
        ),
        'zenith': 30*galsim.degrees,
        'airmass': 1.1547,
        'rotTelPos': 0.0*galsim.degrees,  # zenith measured CCW from up
        'rotSkyPos': 0.0*galsim.degrees,  # N measured CCW from up
        'raw_seeing': 0.7*galsim.arcsec,
        'band': 'i',
        'wavelength': 725.0,  # nm
        'exptime': 20.0,
        # 'exptime': 0.1,
        'temperature': 293.15,  # K
        'pressure': 69.328,  # kPa
        'H2O_pressure': 1.067,  # kPa
    }

    atm_kwargs = {
        'kcrit': 0.2,
        'screen_size': 409.6,
        # 'screen_size': 51.2,
        'screen_scale': 0.1,
        'nproc': 6,
    }

    rng = np.random.default_rng(57721)
    bandpass = galsim.Bandpass("LSST_r.dat", wave_type='nm')
    pixel_scale = 10e-6

    star_simulator = StarSimulator(
        observation, atm_kwargs, bandpass=bandpass,
        pixel_scale=pixel_scale,
        rng=rng,
        debug=False
    )

    # Try to fit ~ 10 stars
    for _ in tqdm.trange(10):
        rho = np.sqrt(rng.uniform(0, np.deg2rad(3.3/60)**2))
        # rho = np.sqrt(rng.uniform(0, np.deg2rad(0.3/60)**2))
        th = rng.uniform(0, 2*np.pi)
        u = rho * np.cos(th)
        v = rho * np.sin(th)
        coord = star_simulator.radec_to_field.toWorld(galsim.PositionD(u, v))
        T = rng.uniform(4000, 10000)
        sed = wfsim.BBSED(T)
        # nphoton = 100_000_000
        nphoton = int(10**rng.uniform(6, 7))
        star_simulator.simulate_star(coord, sed, nphoton, rng)

    star_simulator.add_background(rng)

    import matplotlib.pyplot as plt
    plt.imshow(star_simulator.image.array)
    plt.show()
