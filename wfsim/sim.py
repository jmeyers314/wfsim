import numpy as np
import galsim
import batoid

from .atm import make_atmosphere


class SimpleSimulator:
    """
    Parameters
    ----------
    observation : dict
        zenith : GalSim.Angle
        raw_seeing : GalSim.Angle
        exptime : float
            Seconds
        temperature : float
            Kelvin
        pressure : float
            kPa
        H2O_pressure : float
            kPa
    atm_kwargs : dict
        kcrit : float
        screen_size : float
        sreen_scale : float
        nproc : int
    telescope : batoid.Optic
    bandpass : galsim.Bandpass
    pixel_scale : float
        Meters
    shape : (int, int)
        Sensor shape
    rng : np.random.Generator
    debug : bool
    """
    def __init__(
        self,
        observation,
        atm_kwargs,
        telescope,
        bandpass,
        pixel_scale=10e-6,
        shape=(4000, 4072),
        rng=None,
        debug=False
    ):
        self.observation = observation
        airmass = 1/np.cos(observation['zenith'])
        self.atm, self.target_FWHM, self.r0_500, self.L0 = make_atmosphere(
            airmass,
            observation['raw_seeing'],
            observation['wavelength'],
            rng,
            **atm_kwargs
        )
        self.telescope = telescope
        self.bandpass = bandpass
        self.pixel_scale = pixel_scale
        self.rng = rng
        self.gsrng = galsim.BaseDeviate(
            self.rng.bit_generator.random_raw() % 2**63
        )
        self.debug = debug

        # Pre-cache a second kick
        psf = self.atm.makePSF(
            observation['wavelength'],
            diam=telescope.pupilSize
        )
        _ = psf.drawImage(
            nx=1, ny=1, n_photons=1, rng=self.gsrng, method='phot'
        )
        self.second_kick = psf.second_kick

        self.base_refraction = galsim.dcr.get_refraction(
            self.observation['wavelength'],
            self.observation['zenith'],
            temperature=self.observation['temperature'],
            pressure=self.observation['pressure'],
            H2O_pressure=self.observation['H2O_pressure'],
        )

        self.sensor = galsim.Sensor()  # don't worry about BF here.
        self.image = galsim.Image(*shape)
        self.image.setCenter(0, 0)

    def populate_pupil(self, nphoton, rng):
        """
        Parameters
        ----------
        nphoton : int
        rng : np.random.Generator

        Returns
        -------
        u, v : array of float
            Pupil coord in meters
        t : array of float
            Arrival time in seconds
        """
        r_outer = 0.5*self.telescope.pupilSize
        r_inner = r_outer * self.telescope.pupilObscuration
        # fudge inner to account for off-axis vignetting
        r_inner *= 0.95
        r = np.sqrt(rng.uniform(r_inner**2, r_outer**2, nphoton))
        th = rng.uniform(0, 2*np.pi, nphoton)
        u = r*np.cos(th)
        v = r*np.sin(th)
        t = rng.uniform(0, self.observation['exptime'], nphoton)
        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(u, v)
            plt.title("u, v")
            plt.show()
        return u, v, t

    def kick_1(self, u, v, t, thx, thy):
        """
        Parameters
        ----------
        u, v : array of float
            Pupil coord in meters
        t : array of float
            Arrival time in seconds
        thx, thy : float
            Field angle in radians
        Returns
        -------
        dku, dkv : array of float
            Field angle of rays
        """
        dku, dkv = self.atm.wavefront_gradient(
            u, v, t, (thx*galsim.radians, thy*galsim.radians)
        )
        # output is in nm per m.  Convert to radians
        dku *= 1e-9
        dkv *= 1e-9
        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(
                np.rad2deg(dku)*3600, np.rad2deg(dkv)*3600,
                extent=[-2, 2, -2, 2]
            )
            plt.title("first kick (arcsec)")
            plt.show()
        return dku, dkv

    def apply_kick_2(self, dku, dkv, gsrng):
        """
        Parameters
        ----------
        dku, dkv : array of float
            Field angle of rays (modified in place)
        gsrng : galsim.BaseDeviate
        """
        pa = galsim.PhotonArray(len(dku))
        self.second_kick._shoot(pa, gsrng)
        factor = galsim.arcsec/galsim.radians
        dku += pa.x*factor
        dkv += pa.y*factor
        if self.debug:
            import matplotlib.pyplot as plt
            plt.hexbin(
                np.rad2deg(dku)*3600, np.rad2deg(dkv)*3600,
                extent=[-2, 2, -2, 2]
            )
            plt.title("both kicks (arcsec)")
            plt.show()

    def apply_chromatic_seeing(self, dku, dkv, wavelengths):
        """
        Parameters
        ----------
        dku, dkv : array of float
            Field angle of rays (modified in place)
        wavelengths : array of float
            Nanometers
        """
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

            # # For comparison
            # kolm = galsim.Kolmogorov(fwhm=self.target_FWHM)
            # pa2 = galsim.PhotonArray(len(dku))
            # kolm._shoot(pa2, gsrng)
            # T2 = (
            #     median_abs_deviation(pa2.x, scale='normal')
            #     + median_abs_deviation(pa2.y, scale='normal')
            # )
            # print(f"T2 = ({np.sqrt(T2):.3f} arcsec)^2")
            # print(f"FWHM2 = {np.sqrt(T2/2)/0.6507373182979048:.3f} arcsec")

            # vk = galsim.VonKarman(lam=self.observation['wavelength'], r0_500=self.r0_500, L0=self.L0)
            # pa3 = galsim.PhotonArray(nphoton)
            # vk._shoot(pa3, gsrng)
            # T3 = (
            #     median_abs_deviation(pa3.x, scale='normal')
            #     + median_abs_deviation(pa3.y, scale='normal')
            # )
            # print(f"T3 = ({np.sqrt(T3):.3f} arcsec)^2")
            # print(f"FWHM3 = {np.sqrt(T3/2)/0.6507373182979048:.3f} arcsec")

    def apply_dcr(self, dkv, wavelengths):
        """
        Parameters
        ----------
        dkv : array of float
            Field angle of rays (modified in place)
        wavelengths : array of float
            Nanometers
        """
        refraction = galsim.dcr.get_refraction(
            wavelengths,
            self.observation['zenith'],
            temperature=self.observation['temperature'],
            pressure=self.observation['pressure'],
            H2O_pressure=self.observation['H2O_pressure'],
        )
        refraction -= self.base_refraction
        dkv += refraction

    def create_rayvector(self, u, v, dku, dkv, wavelengths):
        """
        Parameters
        ----------
        u, v : array of float
            Pupil coord in meters
        dku, dkv : array of float
            Field angle of rays (modified in place)
        wavelengths : array of float
            Nanometers

        Returns
        -------
        rays : batoid.RayVector
        """
        vx, vy, vz = batoid.utils.gnomonicToDirCos(dku, dkv)
        x, y = u, v
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
        return rays

    def rays_to_pa(self, rays):
        """
        Parameters
        ----------
        rays : batoid.RayVector

        Returns
        -------
        pa : galsim.PhotonArray
        """
        pa = galsim.PhotonArray(len(rays))
        pa.x = rays.x / self.pixel_scale
        pa.y = rays.y / self.pixel_scale
        pa.dxdz = rays.vx / rays.vz
        pa.dydz = rays.vy / rays.vz
        pa.wavelength = rays.wavelength * 1e9
        pa.flux = ~rays.vignetted
        return pa

    def add_star(self, thx, thy, sed, nphoton, rng):
        """ Add star to image

        Parameters
        ----------
        thx, thy : float
            Field angle in radians
        sed : galsim.SED
        nphoton : int
        rng : np.random.Generator
        """
        gsrng = galsim.BaseDeviate(rng.bit_generator.random_raw() % 2**63)
        u, v, t = self.populate_pupil(nphoton, rng)
        dku, dkv = self.kick_1(u, v, t, thx, thy)
        self.apply_kick_2(dku, dkv, gsrng)  # in place
        wavelengths = sed.sampleWavelength(nphoton, self.bandpass, rng=gsrng)
        self.apply_chromatic_seeing(dku, dkv, wavelengths)
        self.apply_dcr(dkv, wavelengths)
        # Through the atm.  Apply field offset before tracing telescope
        dku += thx
        dkv += thy
        rays = self.create_rayvector(u, v, dku, dkv, wavelengths)
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
                x[:10000]/10e-6, y[:10000]/10e-6, c=np.hypot(u, v)[:10000], s=1
            )
            plt.xlim(-200, 200)
            plt.ylim(-200, 200)
            plt.title("through telescope w/ pupil radius")
            plt.show()

        pa = self.rays_to_pa(rays)
        self.sensor.accumulate(pa, self.image)

    def add_background(self, rng, level):
        """ Add background flux to image

        Parameters
        ----------
        rng : np.random.Generator
        level : float
            Mean sky level
        """
        bd = galsim.BaseDeviate(rng.bit_generator.random_raw() % 2**63)
        gd = galsim.GaussianDeviate(bd, sigma=np.sqrt(level))
        gd.add_generate(self.image.array)


